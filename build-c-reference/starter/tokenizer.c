// Tiktoken-style BPE tokenizer for o200k_harmony.
// Algorithm ported from openai/harmony src/tiktoken.rs.

#include "tokenizer.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------- file format ----------
//   magic  : "GPTOSSTOK\1\0\0\0"           (12 bytes)
//   u32    n_vocab
//   u32    n_special
//   u32    bytes_blob_size
//   u32    specials_blob_size
//   u32[n_vocab+1]   vocab_offsets
//   u8 [bytes_blob_size]  bytes_blob
//   {u32 id; u32 name_off; u32 name_len}[n_special]
//   u8 [specials_blob_size]  specials_blob

typedef struct {
    int id;
    uint32_t off;
    uint32_t len;
} sp_rec;

struct gptoss_tokenizer {
    void*    map;
    size_t   map_size;
    uint32_t n_vocab;
    uint32_t n_special;
    const uint32_t* vocab_off;     // length n_vocab+1
    const uint8_t*  bytes_blob;
    const sp_rec*   sp_recs;
    const uint8_t*  sp_blob;

    // Open-addressing hash table: bytes -> id.
    uint32_t  ht_mask;        // ht_size = ht_mask + 1, power of 2
    int32_t*  ht;             // -1 = empty slot, else token id
};

static void set_err(char** err, const char* msg) {
    if (!err) return;
    *err = strdup(msg);
}

// FNV-1a 32-bit, sufficient given small key set + good distribution.
static inline uint32_t hash_bytes(const uint8_t* p, size_t n) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x01000193u; }
    return h;
}

static inline int bytes_eq(const uint8_t* a, size_t na, const uint8_t* b, size_t nb) {
    return na == nb && (na == 0 || memcmp(a, b, na) == 0);
}

static inline int tk_lookup(const gptoss_tokenizer* tk, const uint8_t* p, size_t n) {
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

gptoss_tokenizer* tk_load(const char* path, char** err) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { set_err(err, strerror(errno)); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); set_err(err, "fstat"); return NULL; }
    void* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { set_err(err, "mmap"); return NULL; }

    const uint8_t* p = (const uint8_t*)m;
    if (st.st_size < 12 + 16 || memcmp(p, "GPTOSSTOK\x01\x00\x00", 12) != 0) {
        munmap(m, (size_t)st.st_size); set_err(err, "bad magic"); return NULL;
    }
    p += 12;
    uint32_t n_vocab, n_special, bytes_size, sp_size;
    memcpy(&n_vocab,    p,  4); p += 4;
    memcpy(&n_special,  p,  4); p += 4;
    memcpy(&bytes_size, p,  4); p += 4;
    memcpy(&sp_size,    p,  4); p += 4;

    const uint32_t* vocab_off = (const uint32_t*)p; p += 4 * (n_vocab + 1);
    const uint8_t*  bytes_blob = p;                  p += bytes_size;
    const sp_rec*   sp_recs   = (const sp_rec*)p;   p += sizeof(sp_rec) * n_special;
    const uint8_t*  sp_blob   = p;                   p += sp_size;
    if ((size_t)(p - (const uint8_t*)m) > (size_t)st.st_size) {
        munmap(m, (size_t)st.st_size); set_err(err, "truncated"); return NULL;
    }

    // Build hash table: smallest power of 2 >= 2 * n_vocab.
    uint32_t cap = 1;
    while (cap < 2u * n_vocab) cap <<= 1;
    int32_t* ht = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    if (!ht) { munmap(m, (size_t)st.st_size); set_err(err, "oom"); return NULL; }
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

    gptoss_tokenizer* tk = (gptoss_tokenizer*)calloc(1, sizeof(*tk));
    tk->map        = m;
    tk->map_size   = (size_t)st.st_size;
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

void tk_free(gptoss_tokenizer* tk) {
    if (!tk) return;
    free(tk->ht);
    munmap(tk->map, tk->map_size);
    free(tk);
}

int tk_special(const gptoss_tokenizer* tk, const char* name) {
    size_t nl = strlen(name);
    for (uint32_t i = 0; i < tk->n_special; i++) {
        const sp_rec* r = &tk->sp_recs[i];
        if (r->len == nl && memcmp(tk->sp_blob + r->off, name, nl) == 0) return r->id;
    }
    return -1;
}

int tk_n_vocab(const gptoss_tokenizer* tk) { return (int)tk->n_vocab; }

const char* tk_token_bytes(const gptoss_tokenizer* tk, int id, int* out_len) {
    if (id < 0) return NULL;
    if ((uint32_t)id < tk->n_vocab) {
        uint32_t off = tk->vocab_off[id];
        uint32_t end = tk->vocab_off[id + 1];
        if (out_len) *out_len = (int)(end - off);
        return (const char*)(tk->bytes_blob + off);
    }
    // For special tokens we return their printable name (so decoding shows
    // "<|channel|>" etc.).
    for (uint32_t i = 0; i < tk->n_special; i++) {
        const sp_rec* r = &tk->sp_recs[i];
        if (r->id == id) {
            if (out_len) *out_len = (int)r->len;
            return (const char*)(tk->sp_blob + r->off);
        }
    }
    return NULL;
}

int tk_decode(const gptoss_tokenizer* tk, const int* ids, int n_ids,
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

// ---------- BPE merge (port of harmony tiktoken.rs::_byte_pair_merge) ----------

// Returns rank of the bytes [start, end) in the encoder, or UINT32_MAX if
// not present.
static inline uint32_t lookup_rank(const gptoss_tokenizer* tk,
                                   const uint8_t* piece, size_t s, size_t e)
{
    int id = tk_lookup(tk, piece + s, e - s);
    return (id < 0) ? UINT32_MAX : (uint32_t)id;
}

// Apply BPE to a single pre-token chunk. Writes ids to out, returns count.
// Capacity of out must be >= n.
static int bpe_byte_pair_encode(const gptoss_tokenizer* tk,
                                const uint8_t* piece, size_t n,
                                int* out)
{
    if (n == 0) return 0;
    if (n == 1) {
        int id = tk_lookup(tk, piece, 1);
        out[0] = id;  // every single byte is in vocab
        return 1;
    }

    // parts: array of (start_offset, rank). Total length = n + 1; the last
    // entry is a sentinel (offset=n, rank=MAX).
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
    size_t np = n + 1;     // active size of `parts`

    while (min_rank != UINT32_MAX) {
        uint32_t i = min_idx;
        // Recompute parts[i-1].rank (pair starting at i-1, spanning to parts[i+2]).
        if (i > 0) {
            size_t s = parts[i - 1].off;
            size_t e = (i + 2 < np) ? parts[i + 2].off : (uint32_t)n;
            parts[i - 1].rank = (i + 2 < np) ? lookup_rank(tk, piece, s, e) : UINT32_MAX;
        }
        // Recompute parts[i].rank (pair starting at i after deleting i+1).
        {
            size_t s = parts[i].off;
            size_t e = (i + 3 < np) ? parts[i + 3].off : (uint32_t)n;
            parts[i].rank = (i + 3 < np) ? lookup_rank(tk, piece, s, e) : UINT32_MAX;
        }
        // Remove parts[i+1].
        memmove(&parts[i + 1], &parts[i + 2], sizeof(part_t) * (np - (i + 2)));
        np--;

        // Find new minimum rank among parts[0..np-1].
        min_rank = UINT32_MAX; min_idx = UINT32_MAX;
        for (size_t j = 0; j + 1 < np; j++) {
            if (parts[j].rank < min_rank) {
                min_rank = parts[j].rank;
                min_idx  = (uint32_t)j;
            }
        }
    }

    // Emit token ids: spans [parts[k].off, parts[k+1].off).
    int nout = 0;
    for (size_t k = 0; k + 1 < np; k++) {
        size_t s = parts[k].off, e = parts[k + 1].off;
        out[nout++] = tk_lookup(tk, piece + s, e - s);
    }
    free(parts);
    return nout;
}

// ---------- pre-tokenizer (ASCII-aware o200k regex) ----------

static inline int is_lf(int c) { return c == '\n' || c == '\r'; }
static inline int is_letter(int c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static inline int is_upper(int c) { return c >= 'A' && c <= 'Z'; }
static inline int is_lower(int c) { return c >= 'a' && c <= 'z'; }
static inline int is_digit(int c) { return c >= '0' && c <= '9'; }
static inline int is_letnum(int c) { return is_letter(c) || is_digit(c); }
static inline int is_space(int c) {
    // \s in PCRE2/Rust regex includes:  \t \n \v \f \r and 0x20.  For ASCII
    // we don't worry about unicode whitespace.
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Try contraction suffix at p: matches case-insensitive 's|'t|'re|'ve|'m|'ll|'d.
// Returns matched length (0 if no match).
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

// Try alternatives 1 and 2 starting at p[0..n]:
//   [^\r\n\p{L}\p{N}]?[\p{Lu}...]*[\p{Ll}...]+(?i:'s|...)?  (alt 1)
//   [^\r\n\p{L}\p{N}]?[\p{Lu}...]+[\p{Ll}...]*(?i:'s|...)?  (alt 2)
// Returns matched length, or 0.
static int match_letter_run(const uint8_t* p, size_t n) {
    if (n == 0) return 0;
    size_t i = 0;
    // Optional separator: not CR/LF/letter/digit. (For ASCII bytes only.)
    int c0 = p[0];
    if (!is_lf(c0) && !is_letnum(c0) && c0 < 0x80) {
        // Only consume the separator if there is at least one letter following.
        if (i + 1 < n && is_letter(p[i + 1])) i++;
    }

    size_t upper_start = i;
    while (i < n && is_upper(p[i])) i++;
    size_t lower_start = i;
    while (i < n && is_lower(p[i])) i++;
    size_t lower_end = i;

    int n_upper = (int)(lower_start - upper_start);
    int n_lower = (int)(lower_end - lower_start);

    int matched = 0;
    if (n_lower >= 1) {
        // Alt 1 ok
        matched = (int)i;
    } else if (n_upper >= 1 && n_lower == 0) {
        // Alt 2 ok
        matched = (int)i;
    } else {
        return 0;
    }

    // Optional contraction suffix.
    matched += match_contraction(p + matched, n - matched);
    return matched;
}

// \p{N}{1,3}
static int match_digits(const uint8_t* p, size_t n) {
    int k = 0;
    while (k < 3 && (size_t)k < n && is_digit(p[k])) k++;
    return k;
}

//  ?[^\s\p{L}\p{N}]+[\r\n/]*
// Matches optional leading space, then one-or-more "other" chars, then any
// trailing CR/LF/'/'.
static int match_other(const uint8_t* p, size_t n) {
    size_t i = 0;
    if (i < n && p[i] == ' ') i++;
    size_t other_start = i;
    while (i < n) {
        int c = p[i];
        if (is_space(c) || is_letter(c) || is_digit(c)) break;
        // Non-ASCII bytes (>= 0x80) are also "other".
        i++;
    }
    if (i == other_start) return 0;
    while (i < n && (p[i] == '\r' || p[i] == '\n' || p[i] == '/')) i++;
    return (int)i;
}

// \s*[\r\n]+
static int match_ws_then_lf(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n && is_space(p[i]) && !is_lf(p[i])) i++;
    if (i == n || !is_lf(p[i])) return 0;
    while (i < n && is_lf(p[i])) i++;
    return (int)i;
}

// \s+(?!\S)  — runs of whitespace that are NOT followed by a non-ws char.
// I.e. trailing whitespace at end-of-string, or whitespace before another
// whitespace block.  In practice this means: consume whitespace as long as
// there is whitespace strictly after the run, OR we're at end-of-string.
// Equivalent rule used by tiktoken: greedy whitespace followed by another ws
// or EOF.  We implement: take a maximal whitespace run; if the char after
// the run (if any) is non-whitespace, shorten by 1.
static int match_ws_negahead(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n && is_space(p[i])) i++;
    if (i == 0) return 0;
    if (i < n) i--;             // (?!\S) — leave one ws for the alt-7 \s+ rule
    return (int)i;
}

// \s+
static int match_ws(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n && is_space(p[i])) i++;
    return (int)i;
}

// Pre-tokenize: split text into chunks per the o200k_harmony regex.
// Calls cb(piece, len, ud) for each chunk.
static int pretokenize(const uint8_t* text, size_t n,
                       int (*cb)(const uint8_t*, size_t, void*), void* ud)
{
    size_t i = 0;
    while (i < n) {
        const uint8_t* p = text + i;
        size_t r = n - i;
        int m = match_letter_run(p, r);
        if (!m) m = match_digits(p, r);
        if (!m) m = match_other(p, r);
        if (!m) m = match_ws_then_lf(p, r);
        if (!m) m = match_ws_negahead(p, r);
        if (!m) m = match_ws(p, r);
        if (!m) {
            // Last resort: emit a single byte.  Should never happen for ASCII.
            m = 1;
        }
        if (cb(p, (size_t)m, ud) != 0) return -1;
        i += (size_t)m;
    }
    return 0;
}

// ---------- public encode ----------

typedef struct {
    const gptoss_tokenizer* tk;
    int* out;
    int  cap;
    int  n;
} enc_state;

static int enc_emit(const uint8_t* p, size_t n, void* ud) {
    enc_state* s = (enc_state*)ud;
    // ignore_merges: if the whole piece is already a token, emit as-is.
    int id = tk_lookup(s->tk, p, n);
    if (id >= 0) {
        if (s->n + 1 > s->cap) return -1;
        s->out[s->n++] = id;
        return 0;
    }
    // BPE on the piece.
    if (s->n + (int)n > s->cap) return -1;
    int k = bpe_byte_pair_encode(s->tk, p, n, s->out + s->n);
    s->n += k;
    return 0;
}

int tk_encode_ordinary(const gptoss_tokenizer* tk,
                       const char* text, size_t n_bytes,
                       int* out_ids, int max_ids)
{
    enc_state st = { tk, out_ids, max_ids, 0 };
    if (pretokenize((const uint8_t*)text, n_bytes, enc_emit, &st) != 0) return -1;
    return st.n;
}
