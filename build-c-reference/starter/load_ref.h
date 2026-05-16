// tests/load_ref.h — tiny loader for the .bin format produced by
// tools/dump_ref.py.  Drop into tests/.

#ifndef LOAD_REF_H
#define LOAD_REF_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define MAGIC_LEN 8
static const char REF_MAGIC[MAGIC_LEN+1] = "LLMTNSR1";

typedef struct {
    uint32_t dtype;        // 0=f32, 1=i32
    uint32_t ndim;
    int64_t  shape[6];
    size_t   n_elem;
    void*    data;
} ref_t;

static ref_t* ref_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[MAGIC_LEN];
    if (fread(magic, 1, MAGIC_LEN, f) != MAGIC_LEN ||
        memcmp(magic, REF_MAGIC, MAGIC_LEN) != 0) {
        fprintf(stderr, "%s: bad magic\n", path); exit(1);
    }
    ref_t* r = calloc(1, sizeof(*r));
    fread(&r->dtype, 4, 1, f);
    fread(&r->ndim,  4, 1, f);
    size_t n = 1;
    for (uint32_t i = 0; i < r->ndim; i++) { fread(&r->shape[i], 8, 1, f); n *= (size_t)r->shape[i]; }
    r->n_elem = n;
    size_t elem = (r->dtype == 0) ? 4 : 4;
    r->data = malloc(n * elem);
    if (fread(r->data, elem, n, f) != n) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    fclose(f);
    return r;
}

static void ref_free(ref_t* r) { if (r) { free(r->data); free(r); } }

// Compare an f32 reference against a bf16 buffer; print max-abs-error.
// Returns 0 on success (<= tol), nonzero otherwise.
#include "../src-cpu/bf16.h"
static int ref_check_bf16_vs_f32(const bf16* got, const float* want, size_t n,
                                 float tol, const char* label) {
    float max_d = 0.0f;
    size_t bad_at = (size_t)-1;
    float bad_got = 0, bad_want = 0;
    for (size_t i = 0; i < n; i++) {
        float g = bf16_to_f32(got[i]);
        float w = want[i];
        float d = g > w ? g - w : w - g;
        if (d > max_d) { max_d = d; bad_at = i; bad_got = g; bad_want = w; }
    }
    printf("  %-30s max|d|=%g  (n=%zu)\n", label, max_d, n);
    if (max_d > tol) {
        printf("    first bad at %zu: got=%g want=%g\n", bad_at, bad_got, bad_want);
        return 1;
    }
    return 0;
}

#endif
