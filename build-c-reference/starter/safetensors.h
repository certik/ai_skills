// Minimal safetensors reader. Each shard file is mmap'd and its JSON header
// parsed into a flat list of array (tensor) descriptors that point into the
// mapped region. We support exactly the subset of JSON the safetensors
// format produces (objects, arrays, strings, integers, "null" values, no
// escapes inside array-name keys beyond plain ASCII).

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ST_DTYPE_UNKNOWN = 0,
    ST_DTYPE_BF16,
    ST_DTYPE_F16,
    ST_DTYPE_F32,
    ST_DTYPE_U8,
    ST_DTYPE_I8,
    ST_DTYPE_U16,
    ST_DTYPE_I16,
    ST_DTYPE_U32,
    ST_DTYPE_I32,
    ST_DTYPE_U64,
    ST_DTYPE_I64,
} st_dtype;

#define ST_MAX_RANK 6

typedef struct {
    char        name[160];
    st_dtype    dtype;
    int         rank;
    int64_t     shape[ST_MAX_RANK];
    size_t      nbytes;
    void*       data;        // pointer into mmap'd region
    int         shard_idx;
} st_tensor;

typedef struct st_archive st_archive;

// Open a directory containing one or more `model-XXXXX-of-YYYYY.safetensors`
// shards (or a single `model.safetensors`). All shards are mmap'd and their
// arrays merged into one table.
st_archive* st_open(const char* dir, char** err);
void        st_close(st_archive* a);

size_t            st_count(const st_archive* a);
const st_tensor*  st_at(const st_archive* a, size_t i);
const st_tensor*  st_find(const st_archive* a, const char* name);

const char*       st_dtype_name(st_dtype d);
size_t            st_dtype_size(st_dtype d);

// Shard accessors for zero-copy MTLBuffer wrapping.
size_t            st_n_shards(const st_archive* a);
void              st_shard_info(const st_archive* a, size_t idx, void** ptr, size_t* size);
size_t            st_tensor_offset(const st_archive* a, const st_tensor* t);

#ifdef __cplusplus
}
#endif
#endif
