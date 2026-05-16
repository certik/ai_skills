// Byte-level BPE tokenizer (HF-style: GPT-2 byte-to-unicode + ranked merges).
//
// Storage format (`tokenizer.bin`) matches starter/build-c-reference:
//   magic   : "LLMBPETK\1\0\0\0"               (12 bytes)
//   u32     : n_vocab
//   u32     : n_special
//   u32     : bytes_blob_size
//   u32     : specials_blob_size
//   u32[n_vocab+1]  vocab_offsets               (into bytes_blob)
//   u8 [bytes_blob_size]   bytes_blob
//   {u32 id; u32 name_off; u32 name_len}[n_special]
//   u8 [specials_blob_size]  specials_blob
//
// [MODEL] Pre-tokenizer regex below is Qwen 3.5 (also matches Llama 3 fairly
// well). For other families (gpt-oss/o200k_harmony, Mistral, ...) read the
// `pre_tokenizer.pattern` field from tokenizer.json and rewrite the
// `match_*` helpers + `pretokenize` alternation order. The BPE merge loop
// itself is model-agnostic.
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   |\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
//
// We implement an ASCII-only pre-tokenizer sufficient for short validation
// prompts and chat-template scaffolding (`<|im_start|>`, `user`,
// `assistant`, `\n`, etc.). Non-ASCII inputs go through a fallback path
// that classifies any byte >= 0x80 as "letter" (the largest natural bucket).

#include "tokenizer.h"
#include "utils/iofile.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------- file format helpers ----------

typedef struct {
    int id;
    uint32_t off;
    uint32_t len;
} sp_rec;

struct tokenizer_t {
    iofile_t file;
    uint32_t n_vocab;
    uint32_t n_special;
    const uint32_t* vocab_off;
    const uint8_t*  bytes_blob;
    const sp_rec*   sp_recs;
    const uint8_t*  sp_blob;

    uint32_t  ht_mask;
    int32_t*  ht;
};

static inline uint32_t hash_bytes(const uint8_t* p, size_t n) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x01000193u; }
    return h;
}

static inline int bytes_eq(const uint8_t* a, size_t na, const uint8_t* b, size_t nb) {
    return na == nb && (na == 0 || memcmp(a, b, na) == 0);
}

static inline int tk_lookup(const tokenizer_t* tk, const uint8_t* p, size_t n) {
    if (n == 0) return -1;
    uint32_t mask = tk->ht_mask;
    uint32_t i = hash_bytes(p, n) & mask;
    for (;;) {
        int32_t id = tk->ht[i];
        if (id < 0) return -1;
        size_t off = tk->vocab_off[id];
        size_t len = tk->vocab_off[id + 1] - off;
        if (bytes_eq(p, n, tk->bytes_blob + off, len)) return id;
        i = (i + 1) & mask;
    }
}

tokenizer_t* tk_load(const char* path, char** err) {
    iofile_t f;
    if (iofile_mmap_ro(path, &f, err) != 0) return NULL;

    const uint8_t* p = (const uint8_t*)f.data;
    if (f.size < 12 + 16 || memcmp(p, "LLMBPETK\x01\x00\x00", 12) != 0) {
        iofile_close(&f);
        if (err) *err = strdup("bad magic");
        return NULL;
    }
    p += 12;
    uint32_t n_vocab, n_special, bytes_size, sp_size;
    memcpy(&n_vocab,    p,  4); p += 4;
    memcpy(&n_special,  p,  4); p += 4;
    memcpy(&bytes_size, p,  4); p += 4;
    memcpy(&sp_size,    p,  4); p += 4;

    const uint32_t* vocab_off = (const uint32_t*)p; p += 4 * (n_vocab + 1);
    const uint8_t*  bytes_blob = p;                  p += bytes_size;
    const sp_rec*   sp_recs    = (const sp_rec*)p;   p += sizeof(sp_rec) * n_special;
    const uint8_t*  sp_blob    = p;                  p += sp_size;
    if ((size_t)(p - (const uint8_t*)f.data) > f.size) {
        iofile_close(&f);
        if (err) *err = strdup("truncated");
        return NULL;
    }

    uint32_t cap = 1;
    while (cap < 2u * n_vocab) cap <<= 1;
    int32_t* ht = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    if (!ht) {
        iofile_close(&f);
        if (err) *err = strdup("oom");
        return NULL;
    }
    for (uint32_t i = 0; i < cap; i++) ht[i] = -1;
    uint32_t mask = cap - 1;
    for (uint32_t id = 0; id < n_vocab; id++) {
        uint32_t off = vocab_off[id];
        uint32_t len = vocab_off[id + 1] - off;
        if (len == 0) continue;
        uint32_t i = hash_bytes(bytes_blob + off, len) & mask;
        while (ht[i] >= 0) i = (i + 1) & mask;
        ht[i] = (int32_t)id;
    }

    tokenizer_t* tk = (tokenizer_t*)calloc(1, sizeof(*tk));
    tk->file       = f;
    tk->n_vocab    = n_vocab;
    tk->n_special  = n_special;
    tk->vocab_off  = vocab_off;
    tk->bytes_blob = bytes_blob;
    tk->sp_recs    = sp_recs;
    tk->sp_blob    = sp_blob;
    tk->ht_mask    = mask;
    tk->ht         = ht;
    return tk;
}

void tk_free(tokenizer_t* tk) {
    if (!tk) return;
    free(tk->ht);
    iofile_close(&tk->file);
    free(tk);
}

int tk_special(const tokenizer_t* tk, const char* name) {
    size_t nl = strlen(name);
    for (uint32_t i = 0; i < tk->n_special; i++) {
        const sp_rec* r = &tk->sp_recs[i];
        if (r->len == nl && memcmp(tk->sp_blob + r->off, name, nl) == 0) return r->id;
    }
    return -1;
}

int tk_n_vocab(const tokenizer_t* tk) { return (int)tk->n_vocab; }

const char* tk_token_bytes(const tokenizer_t* tk, int id, int* out_len) {
    if (id < 0) return NULL;
    if ((uint32_t)id < tk->n_vocab) {
        uint32_t off = tk->vocab_off[id];
        uint32_t end = tk->vocab_off[id + 1];
        if (out_len) *out_len = (int)(end - off);
        return (const char*)(tk->bytes_blob + off);
    }
    for (uint32_t i = 0; i < tk->n_special; i++) {
        const sp_rec* r = &tk->sp_recs[i];
        if (r->id == id) {
            if (out_len) *out_len = (int)r->len;
            return (const char*)(tk->sp_blob + r->off);
        }
    }
    return NULL;
}

int tk_decode(const tokenizer_t* tk, const int* ids, int n_ids,
              char* out, int max_bytes)
{
    int w = 0;
    for (int i = 0; i < n_ids; i++) {
        int len = 0;
        const char* b = tk_token_bytes(tk, ids[i], &len);
        if (!b) continue;
        if (w + len > max_bytes) return -1;
        memcpy(out + w, b, (size_t)len);
        w += len;
    }
    return w;
}

// ---------- BPE byte-pair encode (rank = id; lowest rank wins) ----------

static inline uint32_t lookup_rank(const tokenizer_t* tk,
                                   const uint8_t* piece, size_t s, size_t e)
{
    int id = tk_lookup(tk, piece + s, e - s);
    return (id < 0) ? UINT32_MAX : (uint32_t)id;
}

static int bpe_byte_pair_encode(const tokenizer_t* tk,
                                const uint8_t* piece, size_t n,
                                int* out)
{
    if (n == 0) return 0;
    if (n == 1) {
        out[0] = tk_lookup(tk, piece, 1);
        return 1;
    }

    typedef struct { uint32_t off; uint32_t rank; } part_t;
    part_t* parts = (part_t*)malloc(sizeof(part_t) * (n + 1));

    uint32_t min_rank = UINT32_MAX, min_idx = UINT32_MAX;
    for (size_t i = 0; i + 1 < n; i++) {
        parts[i].off = (uint32_t)i;
        parts[i].rank = lookup_rank(tk, piece, i, i + 2);
        if (parts[i].rank < min_rank) { min_rank = parts[i].rank; min_idx = (uint32_t)i; }
    }
    parts[n - 1].off = (uint32_t)(n - 1); parts[n - 1].rank = UINT32_MAX;
    parts[n].off     = (uint32_t)n;       parts[n].rank     = UINT32_MAX;
    size_t np = n + 1;

    while (min_rank != UINT32_MAX) {
        uint32_t i = min_idx;
        if (i > 0) {
            size_t s = parts[i - 1].off;
            size_t e = (i + 2 < np) ? parts[i + 2].off : (uint32_t)n;
            parts[i - 1].rank = (i + 2 < np) ? lookup_rank(tk, piece, s, e) : UINT32_MAX;
        }
        {
            size_t s = parts[i].off;
            size_t e = (i + 3 < np) ? parts[i + 3].off : (uint32_t)n;
            parts[i].rank = (i + 3 < np) ? lookup_rank(tk, piece, s, e) : UINT32_MAX;
        }
        memmove(&parts[i + 1], &parts[i + 2], sizeof(part_t) * (np - (i + 2)));
        np--;

        min_rank = UINT32_MAX; min_idx = UINT32_MAX;
        for (size_t j = 0; j + 1 < np; j++) {
            if (parts[j].rank < min_rank) {
                min_rank = parts[j].rank;
                min_idx  = (uint32_t)j;
            }
        }
    }

    int nout = 0;
    for (size_t k = 0; k + 1 < np; k++) {
        size_t s = parts[k].off, e = parts[k + 1].off;
        out[nout++] = tk_lookup(tk, piece + s, e - s);
    }
    free(parts);
    return nout;
}

// ---------- Qwen pre-tokenizer (ASCII subset) ----------

static inline int is_lf(int c)     { return c == '\n' || c == '\r'; }
static inline int is_letter_ascii(int c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static inline int is_digit_ascii(int c) { return c >= '0' && c <= '9'; }
static inline int is_space_ascii(int c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Treat non-ASCII bytes (>= 0x80) as "letter" so they collect into the
// letter-run rule (Qwen's `[\p{L}\p{M}]+`).  This is a coarse approximation
// but matches behaviour for the validation prompt.
static inline int is_letter(int c) {
    return c >= 0x80 || is_letter_ascii(c);
}
static inline int is_letnum(int c) {
    return is_letter(c) || is_digit_ascii(c);
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d)
static int match_contraction(const uint8_t* p, size_t n) {
    if (n < 2 || p[0] != '\'') return 0;
    int c = tolower(p[1]);
    if (c == 's' || c == 't' || c == 'm' || c == 'd') return 2;
    if (n >= 3) {
        int c2 = tolower(p[2]);
        if (c == 'r' && c2 == 'e') return 3;
        if (c == 'v' && c2 == 'e') return 3;
        if (c == 'l' && c2 == 'l') return 3;
    }
    return 0;
}

// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   - optional leading non-letter/digit/CR/LF byte (a single separator)
//   - then >= 1 letter byte
static int match_letter_run(const uint8_t* p, size_t n) {
    if (n == 0) return 0;
    size_t i = 0;
    int c0 = p[0];
    if (!is_lf(c0) && !is_letnum(c0)) {
        if (i + 1 < n && is_letter(p[i + 1])) i++;
    }
    if (i >= n || !is_letter(p[i])) return 0;
    while (i < n && is_letter(p[i])) i++;
    return (int)i;
}

// \p{N}  (single digit per match)
static int match_digit(const uint8_t* p, size_t n) {
    return (n > 0 && is_digit_ascii(p[0])) ? 1 : 0;
}

//  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
static int match_other(const uint8_t* p, size_t n) {
    size_t i = 0;
    if (i < n && p[i] == ' ') i++;
    size_t start = i;
    while (i < n) {
        int c = p[i];
        if (is_space_ascii(c) || is_letter(c) || is_digit_ascii(c)) break;
        i++;
    }
    if (i == start) return 0;
    while (i < n && is_lf(p[i])) i++;
    return (int)i;
}

// \s*[\r\n]+
static int match_ws_then_lf(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n && is_space_ascii(p[i]) && !is_lf(p[i])) i++;
    if (i == n || !is_lf(p[i])) return 0;
    while (i < n && is_lf(p[i])) i++;
    return (int)i;
}

// \s+(?!\S)  — greedy whitespace ending right before another whitespace or EOF
static int match_ws_negahead(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n && is_space_ascii(p[i])) i++;
    if (i == 0) return 0;
    if (i < n) i--;
    return (int)i;
}

// \s+
static int match_ws(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n && is_space_ascii(p[i])) i++;
    return (int)i;
}

// Pre-tokenize: alternation order matches the Qwen pre-tokenizer regex.
static int pretokenize(const uint8_t* text, size_t n,
                       int (*cb)(const uint8_t*, size_t, void*), void* ud)
{
    size_t i = 0;
    while (i < n) {
        const uint8_t* p = text + i;
        size_t r = n - i;
        int m = match_contraction(p, r);
        if (!m) m = match_letter_run(p, r);
        if (!m) m = match_digit(p, r);
        if (!m) m = match_other(p, r);
        if (!m) m = match_ws_then_lf(p, r);
        if (!m) m = match_ws_negahead(p, r);
        if (!m) m = match_ws(p, r);
        if (!m) m = 1;
        if (cb(p, (size_t)m, ud) != 0) return -1;
        i += (size_t)m;
    }
    return 0;
}

// ---------- byte-level (GPT-2) re-encode ----------
//
// HF's ByteLevel pre-tokenizer (`add_prefix_space=false`) remaps each raw
// byte to a unicode codepoint via the standard 256→unicode bijection:
//   - printable ASCII '!'..'~' (33..126), '¡'..'¬' (161..172), '®'..'ÿ' (174..255)
//     map to themselves;
//   - other bytes (0..32, 127..160, 173) map to 256..256+n_other-1.
// We store vocab tokens in raw-byte form (via build_tokenizer.py), so the
// BPE merger works directly on raw bytes — no GPT-2 remap needed at
// runtime, since the vocab keys were already decoded to raw bytes.

// ---------- public encode ----------

typedef struct {
    const tokenizer_t* tk;
    int* out;
    int  cap;
    int  n;
} enc_state;

static int enc_emit(const uint8_t* p, size_t n, void* ud) {
    enc_state* s = (enc_state*)ud;
    int id = tk_lookup(s->tk, p, n);
    if (id >= 0) {
        if (s->n + 1 > s->cap) return -1;
        s->out[s->n++] = id;
        return 0;
    }
    if (s->n + (int)n > s->cap) return -1;
    int k = bpe_byte_pair_encode(s->tk, p, n, s->out + s->n);
    s->n += k;
    return 0;
}

int tk_encode_ordinary(const tokenizer_t* tk,
                       const char* text, size_t n_bytes,
                       int* out_ids, int max_ids)
{
    enc_state st = { tk, out_ids, max_ids, 0 };
    if (pretokenize((const uint8_t*)text, n_bytes, enc_emit, &st) != 0) return -1;
    return st.n;
}
