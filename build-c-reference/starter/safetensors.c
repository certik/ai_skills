#define _DARWIN_C_SOURCE
#include "safetensors.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ----- archive structure ----------------------------------------------------
typedef struct {
    int     fd;
    void*   map;
    size_t  size;
    size_t  data_off;     // byte offset where tensor data section starts
    char    path[1024];
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

// ----- minimal JSON walker --------------------------------------------------
// Cursor over a buffer; parser does not allocate. Only enough JSON to walk
// the safetensors header.
typedef struct {
    const char* p;
    const char* end;
    char        err[256];
    bool        failed;
} jc;

static void j_fail(jc* c, const char* msg) {
    if (!c->failed) { c->failed = true; snprintf(c->err, sizeof(c->err), "%s", msg); }
}
static void j_ws(jc* c) { while (c->p < c->end && (*c->p==' '||*c->p=='\n'||*c->p=='\t'||*c->p=='\r')) c->p++; }
static bool j_eat(jc* c, char ch) { j_ws(c); if (c->p<c->end && *c->p==ch) { c->p++; return true; } return false; }
static void j_expect(jc* c, char ch) { if (!j_eat(c, ch)) { char m[64]; snprintf(m,64,"expected '%c'",ch); j_fail(c,m);} }

// Parse a JSON string into (out_ptr, out_len) view into the buffer.
// Does NOT support escapes — safetensors keys/dtype names are plain ASCII.
static bool j_string(jc* c, const char** out, size_t* out_len) {
    j_ws(c);
    if (c->p>=c->end || *c->p!='"') { j_fail(c,"expected string"); return false; }
    c->p++;
    const char* start = c->p;
    while (c->p<c->end && *c->p!='"') {
        if (*c->p == '\\') c->p += 2; else c->p++;
    }
    if (c->p>=c->end) { j_fail(c,"unterminated string"); return false; }
    *out = start; *out_len = (size_t)(c->p - start);
    c->p++;
    return true;
}

static int64_t j_int(jc* c) {
    j_ws(c);
    int64_t sign = 1;
    if (c->p<c->end && *c->p=='-') { sign = -1; c->p++; }
    int64_t v = 0; bool any = false;
    while (c->p<c->end && isdigit((unsigned char)*c->p)) { v = v*10 + (*c->p - '0'); c->p++; any=true; }
    if (!any) j_fail(c, "expected integer");
    return v * sign;
}

// Skip a JSON value of any type (object/array/string/number/true/false/null).
static void j_skip(jc* c) {
    j_ws(c);
    if (c->failed || c->p>=c->end) return;
    char ch = *c->p;
    if (ch == '{') {
        c->p++;
        j_ws(c);
        if (j_eat(c,'}')) return;
        while (!c->failed) {
            const char* ks; size_t kl;
            if (!j_string(c,&ks,&kl)) return;
            if (!j_eat(c,':')) { j_fail(c,"expected ':'"); return; }
            j_skip(c);
            if (j_eat(c,',')) continue;
            j_expect(c,'}'); return;
        }
    } else if (ch == '[') {
        c->p++;
        j_ws(c);
        if (j_eat(c,']')) return;
        while (!c->failed) { j_skip(c); if (j_eat(c,',')) continue; j_expect(c,']'); return; }
    } else if (ch == '"') {
        const char* s; size_t n; j_string(c,&s,&n);
    } else if (ch == 't' || ch == 'f' || ch == 'n') {
        // true/false/null
        while (c->p<c->end && (isalpha((unsigned char)*c->p))) c->p++;
    } else {
        // number
        while (c->p<c->end && (isdigit((unsigned char)*c->p)||*c->p=='-'||*c->p=='+'||*c->p=='.'||*c->p=='e'||*c->p=='E')) c->p++;
    }
}

// Parse one tensor info object: {"dtype":"BF16","shape":[..],"data_offsets":[a,b]}
static void parse_tensor_info(jc* c, const char* name, size_t name_len,
                              st_archive* a, st_shard* sh)
{
    if (!j_eat(c,'{')) { j_fail(c,"tensor info: expected '{'"); return; }
    st_tensor t = {0};
    if (name_len >= sizeof(t.name)) name_len = sizeof(t.name)-1;
    memcpy(t.name, name, name_len); t.name[name_len] = 0;
    t.shard_idx = (int)(sh - a->shards);
    int64_t off_start = -1, off_end = -1;
    bool got_dtype = false, got_shape = false, got_off = false;

    j_ws(c);
    if (j_eat(c,'}')) goto store;
    while (!c->failed) {
        const char* ks; size_t kl;
        if (!j_string(c,&ks,&kl)) return;
        if (!j_eat(c,':')) { j_fail(c,"expected ':'"); return; }
        if (kl==5 && memcmp(ks,"dtype",5)==0) {
            const char* vs; size_t vl;
            if (!j_string(c,&vs,&vl)) return;
            t.dtype = dtype_from_str(vs, vl);
            got_dtype = true;
        } else if (kl==5 && memcmp(ks,"shape",5)==0) {
            if (!j_eat(c,'[')) { j_fail(c,"shape: expected '['"); return; }
            t.rank = 0;
            j_ws(c);
            if (!j_eat(c,']')) {
                for (;;) {
                    if (t.rank >= ST_MAX_RANK) { j_fail(c,"shape rank > ST_MAX_RANK"); return; }
                    t.shape[t.rank++] = j_int(c);
                    if (j_eat(c,',')) continue;
                    j_expect(c,']'); break;
                }
            }
            got_shape = true;
        } else if (kl==12 && memcmp(ks,"data_offsets",12)==0) {
            if (!j_eat(c,'[')) { j_fail(c,"data_offsets: expected '['"); return; }
            off_start = j_int(c);
            if (!j_eat(c,',')) { j_fail(c,"expected ','"); return; }
            off_end = j_int(c);
            j_expect(c,']');
            got_off = true;
        } else {
            j_skip(c);
        }
        if (j_eat(c,',')) continue;
        j_expect(c,'}'); break;
    }
store:
    if (c->failed) return;
    if (!got_dtype || !got_shape || !got_off) { j_fail(c,"tensor info missing fields"); return; }
    t.nbytes = (size_t)(off_end - off_start);
    t.data   = (char*)sh->map + sh->data_off + (size_t)off_start;

    if (a->n_tensors == a->cap_tensors) {
        a->cap_tensors = a->cap_tensors ? a->cap_tensors * 2 : 256;
        a->tensors = (st_tensor*)realloc(a->tensors, a->cap_tensors * sizeof(*a->tensors));
    }
    a->tensors[a->n_tensors++] = t;
}

// Parse the header object of one shard.
static bool parse_shard_header(st_archive* a, st_shard* sh, char** err) {
    if (sh->size < 8) { if (err) *err = strdup("shard too small"); return false; }
    uint64_t hdr_len = 0;
    memcpy(&hdr_len, sh->map, 8);
    if (hdr_len > sh->size - 8) { if (err) *err = strdup("bad header length"); return false; }
    sh->data_off = 8 + hdr_len;

    jc c = { .p = (const char*)sh->map + 8, .end = (const char*)sh->map + 8 + hdr_len };
    j_ws(&c);
    if (!j_eat(&c,'{')) { if (err) *err = strdup("header: expected '{'"); return false; }
    j_ws(&c);
    if (j_eat(&c,'}')) return true;
    for (;;) {
        const char* ks; size_t kl;
        if (!j_string(&c, &ks, &kl)) goto fail;
        if (!j_eat(&c,':')) { j_fail(&c,"expected ':'"); goto fail; }
        if (kl == 11 && memcmp(ks, "__metadata__", 11) == 0) {
            j_skip(&c);
        } else {
            parse_tensor_info(&c, ks, kl, a, sh);
            if (c.failed) goto fail;
        }
        if (j_eat(&c, ',')) continue;
        if (!j_eat(&c, '}')) { j_fail(&c,"expected ','/'}'"); goto fail; }
        break;
    }
    return true;
fail:
    if (err) {
        char buf[320];
        snprintf(buf, sizeof(buf), "json parse: %s (at offset %ld)",
                 c.err, (long)(c.p - (const char*)sh->map));
        *err = strdup(buf);
    }
    return false;
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
        sh->fd = open(files[i], O_RDONLY);
        if (sh->fd < 0) { if (err) { char b[512]; snprintf(b,512,"open %s: %s",files[i],strerror(errno)); *err=strdup(b);} goto fail; }
        struct stat st;
        if (fstat(sh->fd, &st) != 0) { if (err) *err = strdup("fstat failed"); goto fail; }
        sh->size = (size_t)st.st_size;
        sh->map = mmap(NULL, sh->size, PROT_READ, MAP_PRIVATE, sh->fd, 0);
        if (sh->map == MAP_FAILED) {
            if (err) { char b[256]; snprintf(b,256,"mmap %s: %s",files[i],strerror(errno)); *err=strdup(b);}
            sh->map = NULL; goto fail;
        }
        // Hint kernel we'll touch all of it sequentially during load.
        madvise(sh->map, sh->size, MADV_WILLNEED);

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
        st_shard* sh = &a->shards[i];
        if (sh->map) munmap(sh->map, sh->size);
        if (sh->fd > 0) close(sh->fd);
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
    if (ptr)  *ptr  = a->shards[idx].map;
    if (size) *size = a->shards[idx].size;
}

size_t st_tensor_offset(const st_archive* a, const st_tensor* t) {
    if (!a || !t || t->shard_idx < 0 || (size_t)t->shard_idx >= a->n_shards) return 0;
    return (size_t)((char*)t->data - (char*)a->shards[t->shard_idx].map);
}
