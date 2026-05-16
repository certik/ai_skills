// build_tokenizer.c — Convert HF tokenizer.json into the compact binary blob
// `src-cpu/tokenizer.bin` consumed by src-cpu/tokenizer.c.
//
// Standalone C99: depends only on libc and src-cpu/utils/{json,bytebuf}.{h,c}.
//
// Works for **byte-level BPE** tokenizers (Qwen, Llama 3, GPT-2 / o200k
// family, gpt-oss, ...). For sentencepiece / unigram tokenizers (older
// Llama, Mistral, T5) the vocab-key decoding step is different; see SKILL.md.
//
// Output format (little-endian):
//   magic  : "LLMBPETK\1\0\0\0"        (12 bytes)
//   u32    n_vocab                     (BPE base entries; rank == id)
//   u32    n_special                   (named special tokens, sorted by id)
//   u32    bytes_blob_size
//   u32    specials_blob_size
//   u32[n_vocab+1]   vocab_offsets     (into bytes_blob; sentinel last)
//   u8 [bytes_blob_size]   bytes_blob  (raw bytes of every BPE token,
//                                       concatenated, in rank order)
//   records   n_special × { u32 id; u32 name_off; u32 name_len }
//   u8 [specials_blob_size] specials_blob   (UTF-8 names of special tokens)

#include "utils/json.h"
#include "utils/bytebuf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "LLMBPETK\x01\x00\x00\x00"
#define MAGIC_LEN 12

// GPT-2 byte-level encoding: 256 bytes -> 256 printable codepoints (max U+0142).
// Reverse map: codepoint -> byte, or -1 if not in the image.
static int g_cp2byte[512];

static void build_cp2byte(void) {
    for (int i = 0; i < 512; i++) g_cp2byte[i] = -1;
    int bs[256], cs[256], n = 0;
    for (int c = '!';  c <= '~';  c++) { bs[n] = c; cs[n] = c; n++; }
    for (int c = 0xa1; c <= 0xac; c++) { bs[n] = c; cs[n] = c; n++; }
    for (int c = 0xae; c <= 0xff; c++) { bs[n] = c; cs[n] = c; n++; }
    int in_bs[256] = {0};
    for (int i = 0; i < n; i++) in_bs[bs[i]] = 1;
    int extra = 0;
    for (int b = 0; b < 256; b++)
        if (!in_bs[b]) { bs[n] = b; cs[n] = 256 + extra++; n++; }
    for (int i = 0; i < n; i++) g_cp2byte[cs[i]] = bs[i];
}

// Decode one BPE vocab key (JSON string val) into raw bytes via g_cp2byte,
// appending to `out`. Returns 0 on success, -1 on unmapped codepoint or
// JSON parse error.
static int decode_vocab_key(json_val_t key_v, bytebuf_t* out) {
    enum { MAX_CP = 4096 };
    int cps[MAX_CP];
    long ncp = json_str_codepoints(key_v, cps, MAX_CP);
    if (ncp < 0) return -1;
    for (long i = 0; i < ncp; i++) {
        int cp = cps[i];
        if (cp < 0 || cp >= 512 || g_cp2byte[cp] < 0) {
            fprintf(stderr, "unmapped codepoint U+%04X\n", cp);
            return -1;
        }
        bb_append_u8(out, (uint8_t)g_cp2byte[cp]);
    }
    return 0;
}

// argparse-lite: return value following `flag`, else `def`.
static const char* arg(int argc, char** argv, const char* flag, const char* def) {
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}

typedef struct { uint32_t id, off, len; } spec_t;
static int spec_cmp(const void* a, const void* b) {
    uint32_t ai = ((const spec_t*)a)->id, bi = ((const spec_t*)b)->id;
    return (ai > bi) - (ai < bi);
}

int main(int argc, char** argv) {
    // [MODEL] Adjust the default --tokenizer-json path to point at your
    // model's HF tokenizer.json (or just always pass --tokenizer-json on
    // the command line and leave the default as "tokenizer.json").
    const char* in_path  = arg(argc, argv, "--tokenizer-json", "tokenizer.json");
    const char* out_path = arg(argc, argv, "--out", "src-cpu/tokenizer.bin");

    build_cp2byte();

    fprintf(stderr, "loading %s\n", in_path);
    char* err = NULL;
    json_doc_t* doc = json_open_file(in_path, &err);
    if (!doc) { fprintf(stderr, "json: %s\n", err); return 1; }
    json_val_t root  = json_root(doc);
    json_val_t vocab = json_path(root, "model.vocab");
    json_val_t added = json_get (root, "added_tokens");

    size_t n_base = json_obj_len(vocab);
    fprintf(stderr, "  base vocab entries: %zu\n", n_base);

    // Vocab pass: decode all keys into one scratch bytebuf; remember per-rank
    // (offset, length) so we can emit in rank order.
    bytebuf_t  scratch = BYTEBUF_INIT;
    uint32_t*  off  = calloc(n_base, sizeof(uint32_t));
    uint32_t*  len  = calloc(n_base, sizeof(uint32_t));
    char*      seen = calloc(n_base, 1);

    json_iter_t it;
    json_val_t  key_v, val_v;
    json_obj_iter(vocab, &it);
    while (json_obj_next(&it, &key_v, &val_v)) {
        size_t start = scratch.len;
        long rank;
        if (decode_vocab_key(key_v, &scratch) != 0 ||
            json_as_int(val_v, &rank) != 0 ||
            rank < 0 || rank >= (long)n_base || seen[rank]) {
            fprintf(stderr, "vocab: bad/duplicate rank or undecodable key\n");
            return 1;
        }
        seen[rank] = 1;
        off[rank] = (uint32_t)start;
        len[rank] = (uint32_t)(scratch.len - start);
    }
    for (size_t r = 0; r < n_base; r++)
        if (!seen[r]) { fprintf(stderr, "missing rank %zu\n", r); return 1; }

    // Rank-ordered blob + offset table.
    bytebuf_t blob = BYTEBUF_INIT, offsets = BYTEBUF_INIT;
    for (size_t r = 0; r < n_base; r++) {
        bb_append_u32(&offsets, (uint32_t)blob.len);
        bb_append(&blob, scratch.data + off[r], len[r]);
    }
    bb_append_u32(&offsets, (uint32_t)blob.len);

    // added_tokens pass: collect (id, off, len) records and accumulate names
    // into one scratch buffer. Sort records by id ascending.
    size_t n_added = json_arr_len(added);
    spec_t* specs = calloc(n_added ? n_added : 1, sizeof(spec_t));
    bytebuf_t names = BYTEBUF_INIT;

    json_arr_iter(added, &it);
    json_val_t elem;
    for (size_t k = 0; k < n_added && json_arr_next(&it, &elem); k++) {
        long id;
        char tmp[1024];
        long nlen = json_str_utf8(json_get(elem, "content"), tmp, sizeof(tmp));
        if (json_as_int(json_get(elem, "id"), &id) != 0 || id < 0 || nlen < 0) {
            fprintf(stderr, "added_tokens[%zu]: bad id/content\n", k);
            return 1;
        }
        specs[k] = (spec_t){ (uint32_t)id, (uint32_t)names.len, (uint32_t)nlen };
        bb_append(&names, tmp, (size_t)nlen);
    }
    qsort(specs, n_added, sizeof(spec_t), spec_cmp);

    // Emit sp_recs + sp_blob in id-sorted order, rewriting offsets to refer
    // to positions in sp_blob.
    bytebuf_t sp_recs = BYTEBUF_INIT, sp_blob = BYTEBUF_INIT;
    for (size_t i = 0; i < n_added; i++) {
        bb_append_u32(&sp_recs, specs[i].id);
        bb_append_u32(&sp_recs, (uint32_t)sp_blob.len);
        bb_append_u32(&sp_recs, specs[i].len);
        bb_append(&sp_blob, names.data + specs[i].off, specs[i].len);
    }

    // Assemble the output file in memory and write atomically.
    bytebuf_t out = BYTEBUF_INIT;
    bb_append    (&out, MAGIC, MAGIC_LEN);
    bb_append_u32(&out, (uint32_t)n_base);
    bb_append_u32(&out, (uint32_t)n_added);
    bb_append_u32(&out, (uint32_t)blob.len);
    bb_append_u32(&out, (uint32_t)sp_blob.len);
    bb_append    (&out, offsets.data, offsets.len);
    bb_append    (&out, blob.data,    blob.len);
    bb_append    (&out, sp_recs.data, sp_recs.len);
    bb_append    (&out, sp_blob.data, sp_blob.len);
    if (bb_write_file(&out, out_path) != 0) { perror(out_path); return 1; }

    fprintf(stderr, "wrote %s (%.2f MB)\n", out_path, (double)out.len / 1e6);
    fprintf(stderr, "  n_vocab = %zu, n_special = %zu\n", n_base, n_added);
    fprintf(stderr, "  bytes_blob = %zu, specials_blob = %zu\n",
            blob.len, sp_blob.len);
    return 0;   // exit reclaims everything
}
