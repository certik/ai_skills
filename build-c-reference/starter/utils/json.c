// json.c — tiny Python-like JSON reader on top of jsmn.

#define _POSIX_C_SOURCE 200809L
#define JSMN_STATIC
#define JSMN_STRICT
#define JSMN_PARENT_LINKS   // O(1) back-walk on ',' and '}' / ']'
#include "jsmn.h"

#include "json.h"
#include "iofile.h"
#include "utf8.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct json_doc {
    const char* buf;
    size_t      len;
    int         owns;        // 0=external, 1=mmap'd via iofile
    iofile_t    file;        // valid iff owns == 1
    jsmntok_t*  toks;
    int         n_tok;
};

// ---- jsmn helpers --------------------------------------------------------

// Advance past tokens[i] and all of its descendants; return the next index.
// Recursion depth equals JSON nesting depth (tiny in practice).
static int tok_skip(const jsmntok_t* t, int i) {
    if (t[i].type == JSMN_OBJECT) {
        int n = t[i].size, j = i + 1;
        for (int k = 0; k < n; k++) {
            j = tok_skip(t, j);  // key
            j = tok_skip(t, j);  // value
        }
        return j;
    }
    if (t[i].type == JSMN_ARRAY) {
        int n = t[i].size, j = i + 1;
        for (int k = 0; k < n; k++) j = tok_skip(t, j);
        return j;
    }
    return i + 1;
}

static int hexval(unsigned char h) {
    if (h >= '0' && h <= '9') return h - '0';
    if (h >= 'a' && h <= 'f') return h - 'a' + 10;
    if (h >= 'A' && h <= 'F') return h - 'A' + 10;
    return -1;
}

// Decode one JSON `\xxx` escape at p[*i]. Advances *i; returns codepoint or -1.
static int decode_escape(const char* p, size_t len, size_t* i) {
    if (*i + 1 >= len) return -1;
    unsigned char e = (unsigned char)p[*i + 1];
    switch (e) {
        case '"':  *i += 2; return '"';
        case '\\': *i += 2; return '\\';
        case '/':  *i += 2; return '/';
        case 'b':  *i += 2; return 0x08;
        case 'f':  *i += 2; return 0x0c;
        case 'n':  *i += 2; return 0x0a;
        case 'r':  *i += 2; return 0x0d;
        case 't':  *i += 2; return 0x09;
        case 'u': {
            if (*i + 6 > len) return -1;
            int cp = 0;
            for (int k = 0; k < 4; k++) {
                int v = hexval((unsigned char)p[*i + 2 + k]);
                if (v < 0) return -1;
                cp = (cp << 4) | v;
            }
            *i += 6;
            return cp;
        }
        default: return -1;
    }
}

// ---- doc lifetime --------------------------------------------------------

static json_doc_t* parse(const char* buf, size_t len, char** err) {
    jsmn_parser p;
    jsmn_init(&p);
    int n = jsmn_parse(&p, buf, len, NULL, 0);
    if (n < 0) {
        if (err) {
            char b[64];
            snprintf(b, sizeof(b), "jsmn parse error %d", n);
            *err = strdup(b);
        }
        return NULL;
    }
    if (n == 0) {
        if (err) *err = strdup("empty JSON");
        return NULL;
    }
    json_doc_t* d = (json_doc_t*)calloc(1, sizeof(*d));
    if (!d) { if (err) *err = strdup("oom"); return NULL; }
    d->buf  = buf;
    d->len  = len;
    d->toks = (jsmntok_t*)malloc((size_t)n * sizeof(jsmntok_t));
    if (!d->toks) {
        free(d);
        if (err) *err = strdup("oom");
        return NULL;
    }
    jsmn_init(&p);
    int n2 = jsmn_parse(&p, buf, len, d->toks, (unsigned int)n);
    if (n2 != n) {
        free(d->toks);
        free(d);
        if (err) {
            char b[64];
            snprintf(b, sizeof(b), "jsmn parse mismatch %d/%d", n2, n);
            *err = strdup(b);
        }
        return NULL;
    }
    d->n_tok = n;
    return d;
}

json_doc_t* json_open_mem(const void* data, size_t len, char** err) {
    json_doc_t* d = parse((const char*)data, len, err);
    if (!d) return NULL;
    d->owns = 0;
    return d;
}

json_doc_t* json_open_file(const char* path, char** err) {
    iofile_t f;
    if (iofile_mmap_ro(path, &f, err) != 0) return NULL;
    json_doc_t* d = parse((const char*)f.data, f.size, err);
    if (!d) {
        iofile_close(&f);
        return NULL;
    }
    d->owns = 1;
    d->file = f;
    return d;
}

void json_close(json_doc_t* d) {
    if (!d) return;
    free(d->toks);
    if (d->owns == 1) iofile_close(&d->file);
    free(d);
}

json_val_t json_root(const json_doc_t* d) {
    if (!d || d->n_tok < 1) return (json_val_t){d, -1};
    return (json_val_t){d, 0};
}

// ---- type / null ---------------------------------------------------------

static int is_valid(json_val_t v) {
    return v.doc && v.idx >= 0 && v.idx < v.doc->n_tok;
}

json_type_t json_type(json_val_t v) {
    if (!is_valid(v)) return JSON_NULL;
    const jsmntok_t* t = &v.doc->toks[v.idx];
    switch (t->type) {
        case JSMN_OBJECT:    return JSON_OBJECT;
        case JSMN_ARRAY:     return JSON_ARRAY;
        case JSMN_STRING:    return JSON_STRING;
        case JSMN_PRIMITIVE: {
            char c = v.doc->buf[t->start];
            if (c == 't' || c == 'f') return JSON_BOOL;
            if (c == 'n')             return JSON_NULL;
            return JSON_NUMBER;
        }
        default: return JSON_NULL;
    }
}

int json_is_null(json_val_t v) { return json_type(v) == JSON_NULL; }

// ---- string view / eq ----------------------------------------------------

int json_str_view(json_val_t v, const char** out_ptr, size_t* out_len) {
    if (!is_valid(v)) return -1;
    const jsmntok_t* t = &v.doc->toks[v.idx];
    if (t->type != JSMN_STRING) return -1;
    *out_ptr = v.doc->buf + t->start;
    *out_len = (size_t)(t->end - t->start);
    return 0;
}

int json_str_eq(json_val_t v, const char* s) {
    const char* p; size_t l;
    if (json_str_view(v, &p, &l) != 0) return 0;
    return strlen(s) == l && memcmp(p, s, l) == 0;
}

// ---- object access -------------------------------------------------------

json_val_t json_get(json_val_t obj, const char* key) {
    if (!is_valid(obj)) return (json_val_t){obj.doc, -1};
    const jsmntok_t* toks = obj.doc->toks;
    if (toks[obj.idx].type != JSMN_OBJECT) return (json_val_t){obj.doc, -1};
    int n = toks[obj.idx].size;
    int j = obj.idx + 1;
    size_t klen = strlen(key);
    for (int k = 0; k < n; k++) {
        const jsmntok_t* kt = &toks[j];
        if (kt->type == JSMN_STRING &&
            (size_t)(kt->end - kt->start) == klen &&
            memcmp(obj.doc->buf + kt->start, key, klen) == 0) {
            return (json_val_t){obj.doc, j + 1};
        }
        j = tok_skip(toks, j);
        j = tok_skip(toks, j);
    }
    return (json_val_t){obj.doc, -1};
}

json_val_t json_path(json_val_t v, const char* path) {
    while (*path) {
        const char* dot = strchr(path, '.');
        size_t seg = dot ? (size_t)(dot - path) : strlen(path);
        char key[128];
        if (seg >= sizeof(key)) return (json_val_t){v.doc, -1};
        memcpy(key, path, seg);
        key[seg] = 0;
        v = json_get(v, key);
        if (v.idx < 0) return v;
        path = dot ? dot + 1 : path + seg;
    }
    return v;
}

size_t json_obj_len(json_val_t obj) {
    if (!is_valid(obj)) return 0;
    if (obj.doc->toks[obj.idx].type != JSMN_OBJECT) return 0;
    return (size_t)obj.doc->toks[obj.idx].size;
}

// ---- array access --------------------------------------------------------

size_t json_arr_len(json_val_t arr) {
    if (!is_valid(arr)) return 0;
    if (arr.doc->toks[arr.idx].type != JSMN_ARRAY) return 0;
    return (size_t)arr.doc->toks[arr.idx].size;
}

json_val_t json_arr_at(json_val_t arr, size_t i) {
    if (!is_valid(arr)) return (json_val_t){arr.doc, -1};
    if (arr.doc->toks[arr.idx].type != JSMN_ARRAY)
        return (json_val_t){arr.doc, -1};
    if (i >= (size_t)arr.doc->toks[arr.idx].size)
        return (json_val_t){arr.doc, -1};
    int j = arr.idx + 1;
    for (size_t k = 0; k < i; k++) j = tok_skip(arr.doc->toks, j);
    return (json_val_t){arr.doc, j};
}

// ---- iterators -----------------------------------------------------------

void json_obj_iter(json_val_t obj, json_iter_t* it) {
    it->doc = obj.doc;
    if (!is_valid(obj) || obj.doc->toks[obj.idx].type != JSMN_OBJECT) {
        it->next = -1; it->remaining = 0; return;
    }
    it->next      = obj.idx + 1;
    it->remaining = obj.doc->toks[obj.idx].size;
}

int json_obj_next(json_iter_t* it, json_val_t* out_key, json_val_t* out_val) {
    if (it->remaining <= 0 || !it->doc) return 0;
    int k = it->next;
    int v = tok_skip(it->doc->toks, k);
    int after = tok_skip(it->doc->toks, v);
    if (out_key) *out_key = (json_val_t){it->doc, k};
    if (out_val) *out_val = (json_val_t){it->doc, v};
    it->next = after;
    it->remaining--;
    return 1;
}

void json_arr_iter(json_val_t arr, json_iter_t* it) {
    it->doc = arr.doc;
    if (!is_valid(arr) || arr.doc->toks[arr.idx].type != JSMN_ARRAY) {
        it->next = -1; it->remaining = 0; return;
    }
    it->next      = arr.idx + 1;
    it->remaining = arr.doc->toks[arr.idx].size;
}

int json_arr_next(json_iter_t* it, json_val_t* out_val) {
    if (it->remaining <= 0 || !it->doc) return 0;
    int v = it->next;
    if (out_val) *out_val = (json_val_t){it->doc, v};
    it->next = tok_skip(it->doc->toks, v);
    it->remaining--;
    return 1;
}

// ---- scalars -------------------------------------------------------------

int json_as_int(json_val_t v, long* out) {
    if (!is_valid(v)) return -1;
    const jsmntok_t* t = &v.doc->toks[v.idx];
    if (t->type != JSMN_PRIMITIVE) return -1;
    int len = t->end - t->start;
    if (len < 1 || len > 30) return -1;
    char buf[32];
    memcpy(buf, v.doc->buf + t->start, (size_t)len);
    buf[len] = 0;
    char* ep = NULL;
    long val = strtol(buf, &ep, 10);
    if (*ep != 0) return -1;
    *out = val;
    return 0;
}

int json_as_double(json_val_t v, double* out) {
    if (!is_valid(v)) return -1;
    const jsmntok_t* t = &v.doc->toks[v.idx];
    if (t->type != JSMN_PRIMITIVE) return -1;
    int len = t->end - t->start;
    if (len < 1 || len > 64) return -1;
    char buf[68];
    memcpy(buf, v.doc->buf + t->start, (size_t)len);
    buf[len] = 0;
    char* ep = NULL;
    double val = strtod(buf, &ep);
    if (*ep != 0) return -1;
    *out = val;
    return 0;
}

int json_as_bool(json_val_t v, int* out) {
    if (!is_valid(v)) return -1;
    const jsmntok_t* t = &v.doc->toks[v.idx];
    if (t->type != JSMN_PRIMITIVE) return -1;
    char c = v.doc->buf[t->start];
    if (c == 't') { *out = 1; return 0; }
    if (c == 'f') { *out = 0; return 0; }
    return -1;
}

long json_int_or(json_val_t v, long def) {
    long x;
    return json_as_int(v, &x) == 0 ? x : def;
}

double json_double_or(json_val_t v, double def) {
    double x;
    return json_as_double(v, &x) == 0 ? x : def;
}

// ---- string decoding -----------------------------------------------------

long json_str_codepoints(json_val_t v, int* out, size_t cap) {
    const char* p; size_t l;
    if (json_str_view(v, &p, &l) != 0) return -1;
    size_t n = 0, i = 0;
    while (i < l) {
        unsigned char c = (unsigned char)p[i];
        int cp;
        if (c == '\\') {
            cp = decode_escape(p, l, &i);
            if (cp < 0) return -1;
        } else {
            size_t w = utf8_decode(p + i, p + l, &cp);
            if (w == 0) return -1;
            i += w;
        }
        if (n >= cap) return -1;
        out[n++] = cp;
    }
    return (long)n;
}

long json_str_utf8(json_val_t v, char* out, size_t cap) {
    const char* p; size_t l;
    if (json_str_view(v, &p, &l) != 0) return -1;
    size_t n = 0, i = 0;
    while (i < l) {
        unsigned char c = (unsigned char)p[i];
        if (c != '\\') {
            // Pass UTF-8 bytes through verbatim — no decode/re-encode round trip.
            if (n >= cap) return -1;
            out[n++] = (char)c;
            i++;
            continue;
        }
        int cp = decode_escape(p, l, &i);
        if (cp < 0) return -1;
        size_t w = utf8_encode(cp, out + n, cap - n);
        if (w == 0) return -1;
        n += w;
    }
    return (long)n;
}
