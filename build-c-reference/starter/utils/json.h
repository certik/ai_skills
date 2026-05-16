// json.h — tiny Python-like JSON reader on top of jsmn.
//
// Lifetime: a `json_doc_t` owns the parsed token array (and, for files, the
// mmap mapping). `json_val_t` handles into it are invalidated by json_close.
//
// All lookups are read-only. Missing keys, out-of-range indices, and type
// mismatches yield a "null" json_val_t (idx == -1) rather than aborting,
// so chained accesses like
//     json_get(json_get(root, "a"), "b")
// short-circuit safely.

#ifndef JSON_H
#define JSON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct json_doc json_doc_t;

typedef struct {
    const json_doc_t* doc;
    int               idx;   // -1 = null / missing
} json_val_t;

typedef enum {
    JSON_NULL    = 0,
    JSON_OBJECT  = 1,
    JSON_ARRAY   = 2,
    JSON_STRING  = 3,
    JSON_NUMBER  = 4,
    JSON_BOOL    = 5,
} json_type_t;

// ---- Loading ------------------------------------------------------------

// Open a JSON file (mmap'd, owned by the doc).
//
// On failure returns NULL and, if `err` is non-NULL, writes a malloc'd error
// string to *err (caller frees).
json_doc_t* json_open_file(const char* path, char** err);

// Wrap an in-memory JSON span. The buffer is referenced, not copied — the
// caller must keep it alive until json_close.
json_doc_t* json_open_mem(const void* data, size_t len, char** err);

void        json_close(json_doc_t* d);

json_val_t  json_root (const json_doc_t* d);

// ---- Inspect ------------------------------------------------------------

json_type_t json_type   (json_val_t v);
int         json_is_null(json_val_t v);  // true for missing / JSON null

// ---- Object access ------------------------------------------------------

// Look up a key in an object. Returns null if obj isn't an object or the key
// is missing.
json_val_t  json_get (json_val_t obj, const char* key);

// Walk a dotted path: json_path(root, "text_config.rope_parameters.rope_theta").
// Path segments must not themselves contain '.'.
json_val_t  json_path(json_val_t v,   const char* dotted_path);

size_t      json_obj_len(json_val_t obj);

// ---- Array access -------------------------------------------------------

size_t      json_arr_len(json_val_t arr);
json_val_t  json_arr_at (json_val_t arr, size_t i);  // O(i); prefer iter for loops

// ---- Iteration (O(N) total over a container of N children) -------------

typedef struct {
    const json_doc_t* doc;
    int               next;
    int               remaining;
} json_iter_t;

void json_obj_iter(json_val_t obj, json_iter_t* it);
int  json_obj_next(json_iter_t* it, json_val_t* out_key, json_val_t* out_val);

void json_arr_iter(json_val_t arr, json_iter_t* it);
int  json_arr_next(json_iter_t* it, json_val_t* out_val);

// ---- Scalars ------------------------------------------------------------
//
// `json_as_*` return 0 on success and write *out; non-zero on type/parse
// error. `json_*_or` return the value or a default if missing/wrong type.

int    json_as_int   (json_val_t v, long*   out);
int    json_as_double(json_val_t v, double* out);
int    json_as_bool  (json_val_t v, int*    out);

long   json_int_or   (json_val_t v, long   def);
double json_double_or(json_val_t v, double def);

// ---- Strings ------------------------------------------------------------

// Raw byte view of the string body (between the JSON quotes), with escapes
// NOT resolved. Cheapest accessor; use when you know values are pure ASCII
// without escapes (e.g., enum tags like "BF16", tensor names like
// "model.embed_tokens.weight"). Returns 0 on success, -1 on type mismatch.
int   json_str_view(json_val_t v, const char** out_ptr, size_t* out_len);

// Compare a string value to a literal byte sequence (raw, no escape handling).
int   json_str_eq  (json_val_t v, const char* s);

// Decode JSON escapes into UTF-8 bytes (Python `str` equivalent).
// Writes at most `cap` bytes; returns # bytes written, or -1 on error/overflow.
long  json_str_utf8(json_val_t v, char* out, size_t cap);

// Decode JSON escapes + UTF-8 into a codepoint array.
// Returns # codepoints written, or -1 on error/overflow.
long  json_str_codepoints(json_val_t v, int* out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif
