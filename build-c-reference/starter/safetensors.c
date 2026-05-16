#define _DARWIN_C_SOURCE
#include "safetensors.h"
#include "utils/json.h"
#include "utils/iofile.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>      // madvise

// ----- archive structure ----------------------------------------------------
typedef struct {
    iofile_t io;          // file mapping
    size_t   data_off;    // byte offset where tensor data section starts
    char     path[1024];
} st_shard;

struct st_archive {
    st_shard*    shards;
    size_t       n_shards;
    st_tensor*   tensors;
    size_t       n_tensors;
    size_t       cap_tensors;
};

// ----- dtype helpers --------------------------------------------------------
struct dtype_entry { const char* s; st_dtype d; size_t bytes; };
static const struct dtype_entry DT[] = {
    {"BF16", ST_DTYPE_BF16, 2}, {"F16",  ST_DTYPE_F16,  2},
    {"F32",  ST_DTYPE_F32,  4}, {"U8",   ST_DTYPE_U8,   1},
    {"I8",   ST_DTYPE_I8,   1}, {"U16",  ST_DTYPE_U16,  2},
    {"I16",  ST_DTYPE_I16,  2}, {"U32",  ST_DTYPE_U32,  4},
    {"I32",  ST_DTYPE_I32,  4}, {"U64",  ST_DTYPE_U64,  8},
    {"I64",  ST_DTYPE_I64,  8},
};

const char* st_dtype_name(st_dtype d) {
    for (size_t i = 0; i < sizeof(DT)/sizeof(*DT); i++) if (DT[i].d == d) return DT[i].s;
    return "?";
}
size_t st_dtype_size(st_dtype d) {
    for (size_t i = 0; i < sizeof(DT)/sizeof(*DT); i++) if (DT[i].d == d) return DT[i].bytes;
    return 0;
}
static st_dtype dtype_from_str(const char* s, size_t n) {
    for (size_t i = 0; i < sizeof(DT)/sizeof(*DT); i++) {
        if (strlen(DT[i].s) == n && memcmp(DT[i].s, s, n) == 0) return DT[i].d;
    }
    return ST_DTYPE_UNKNOWN;
}

// ----- per-tensor entry from one parsed JSON object ------------------------
//
//   info = {"dtype":"BF16","shape":[..],"data_offsets":[a,b]}
//
// Tensor names in safetensors are ASCII (no JSON escapes), so we can read them
// via the raw byte view.
static int parse_tensor_info(json_val_t info, const char* name, size_t name_len,
                             st_archive* a, st_shard* sh, char** err) {
    if (json_type(info) != JSON_OBJECT) {
        if (err) *err = strdup("tensor info: not an object"); return -1;
    }

    const char* dt_s; size_t dt_l;
    if (json_str_view(json_get(info, "dtype"), &dt_s, &dt_l) != 0) {
        if (err) *err = strdup("tensor info: missing/invalid dtype"); return -1;
    }
    st_dtype dt = dtype_from_str(dt_s, dt_l);

    st_tensor t = {0};
    if (name_len >= sizeof(t.name)) name_len = sizeof(t.name) - 1;
    memcpy(t.name, name, name_len); t.name[name_len] = 0;
    t.shard_idx = (int)(sh - a->shards);
    t.dtype = dt;

    json_val_t shape = json_get(info, "shape");
    if (json_type(shape) != JSON_ARRAY) {
        if (err) *err = strdup("tensor info: missing shape"); return -1;
    }
    json_iter_t sit;
    json_arr_iter(shape, &sit);
    json_val_t dim_v;
    t.rank = 0;
    while (json_arr_next(&sit, &dim_v)) {
        if (t.rank >= ST_MAX_RANK) {
            if (err) *err = strdup("shape rank > ST_MAX_RANK"); return -1;
        }
        long d;
        if (json_as_int(dim_v, &d) != 0) {
            if (err) *err = strdup("shape: non-integer dim"); return -1;
        }
        t.shape[t.rank++] = d;
    }

    json_val_t off = json_get(info, "data_offsets");
    if (json_type(off) != JSON_ARRAY || json_arr_len(off) != 2) {
        if (err) *err = strdup("tensor info: bad data_offsets"); return -1;
    }
    long o0, o1;
    if (json_as_int(json_arr_at(off, 0), &o0) != 0 ||
        json_as_int(json_arr_at(off, 1), &o1) != 0) {
        if (err) *err = strdup("tensor info: non-integer data_offsets"); return -1;
    }
    t.nbytes = (size_t)(o1 - o0);
    t.data   = (char*)sh->io.data + sh->data_off + (size_t)o0;

    if (a->n_tensors == a->cap_tensors) {
        a->cap_tensors = a->cap_tensors ? a->cap_tensors * 2 : 256;
        a->tensors = (st_tensor*)realloc(a->tensors,
                                         a->cap_tensors * sizeof(*a->tensors));
    }
    a->tensors[a->n_tensors++] = t;
    return 0;
}

// Parse the header object of one shard.
static bool parse_shard_header(st_archive* a, st_shard* sh, char** err) {
    if (sh->io.size < 8) { if (err) *err = strdup("shard too small"); return false; }
    uint64_t hdr_len = 0;
    memcpy(&hdr_len, sh->io.data, 8);
    if (hdr_len > sh->io.size - 8) {
        if (err) *err = strdup("bad header length"); return false;
    }
    sh->data_off = 8 + hdr_len;

    json_doc_t* doc = json_open_mem((const char*)sh->io.data + 8, hdr_len, err);
    if (!doc) return false;

    json_val_t root = json_root(doc);
    if (json_type(root) != JSON_OBJECT) {
        if (err) *err = strdup("header: not a JSON object");
        json_close(doc);
        return false;
    }

    json_iter_t it;
    json_obj_iter(root, &it);
    json_val_t name_v, info_v;
    while (json_obj_next(&it, &name_v, &info_v)) {
        if (json_str_eq(name_v, "__metadata__")) continue;
        const char* nm; size_t nl;
        if (json_str_view(name_v, &nm, &nl) != 0) {
            if (err) *err = strdup("header: bad tensor name");
            json_close(doc);
            return false;
        }
        if (parse_tensor_info(info_v, nm, nl, a, sh, err) != 0) {
            json_close(doc);
            return false;
        }
    }
    json_close(doc);
    return true;
}

// ----- shard discovery ------------------------------------------------------
static int cmpstr(const void* a, const void* b) { return strcmp(*(const char**)a, *(const char**)b); }

st_archive* st_open(const char* dir, char** err) {
    DIR* d = opendir(dir);
    if (!d) { if (err) { char b[256]; snprintf(b,256,"opendir %s: %s",dir,strerror(errno)); *err=strdup(b);} return NULL; }
    char** files = NULL; size_t n_files = 0, cap = 0;
    struct dirent* de;
    while ((de = readdir(d))) {
        const char* nm = de->d_name;
        size_t L = strlen(nm);
        if (L < 12) continue;
        if (strcmp(nm + L - 12, ".safetensors") != 0) continue;
        if (n_files == cap) { cap = cap ? cap*2 : 8; files = (char**)realloc(files, cap*sizeof(char*)); }
        char path[1024]; snprintf(path,1024,"%s/%s",dir,nm);
        files[n_files++] = strdup(path);
    }
    closedir(d);
    if (n_files == 0) { if (err) *err = strdup("no .safetensors shards found"); free(files); return NULL; }
    qsort(files, n_files, sizeof(char*), cmpstr);

    st_archive* a = (st_archive*)calloc(1, sizeof(*a));
    a->shards = (st_shard*)calloc(n_files, sizeof(st_shard));
    a->n_shards = n_files;

    for (size_t i = 0; i < n_files; i++) {
        st_shard* sh = &a->shards[i];
        snprintf(sh->path, sizeof(sh->path), "%s", files[i]);
        if (iofile_mmap_ro(files[i], &sh->io, err) != 0) goto fail;
        // Hint kernel we'll touch all of it sequentially during load.
        madvise(sh->io.data, sh->io.size, MADV_WILLNEED);

        if (!parse_shard_header(a, sh, err)) goto fail;
    }

    for (size_t i = 0; i < n_files; i++) free(files[i]);
    free(files);
    return a;

fail:
    for (size_t i = 0; i < n_files; i++) free(files[i]);
    free(files);
    st_close(a);
    return NULL;
}

void st_close(st_archive* a) {
    if (!a) return;
    for (size_t i = 0; i < a->n_shards; i++) {
        iofile_close(&a->shards[i].io);
    }
    free(a->shards);
    free(a->tensors);
    free(a);
}

size_t           st_count(const st_archive* a)   { return a ? a->n_tensors : 0; }
const st_tensor* st_at(const st_archive* a, size_t i) { return (a && i < a->n_tensors) ? &a->tensors[i] : NULL; }

const st_tensor* st_find(const st_archive* a, const char* name) {
    if (!a) return NULL;
    for (size_t i = 0; i < a->n_tensors; i++)
        if (strcmp(a->tensors[i].name, name) == 0) return &a->tensors[i];
    return NULL;
}

size_t st_n_shards(const st_archive* a) { return a ? a->n_shards : 0; }

void st_shard_info(const st_archive* a, size_t idx, void** ptr, size_t* size) {
    if (!a || idx >= a->n_shards) { if (ptr) *ptr = NULL; if (size) *size = 0; return; }
    if (ptr)  *ptr  = a->shards[idx].io.data;
    if (size) *size = a->shards[idx].io.size;
}

size_t st_tensor_offset(const st_archive* a, const st_tensor* t) {
    if (!a || !t || t->shard_idx < 0 || (size_t)t->shard_idx >= a->n_shards) return 0;
    return (size_t)((char*)t->data - (char*)a->shards[t->shard_idx].io.data);
}
