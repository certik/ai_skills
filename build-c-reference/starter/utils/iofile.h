// iofile.h — mmap a file read-only (C answer to Python's `mmap.mmap`).
//
// Usage:
//   iofile_t f;
//   char* err = NULL;
//   if (iofile_mmap_ro("path", &f, &err) != 0) { fprintf(stderr, "%s\n", err); free(err); ... }
//   // ... use f.data / f.size ...
//   iofile_close(&f);
//
// iofile_close is idempotent and safe to call on a zero-initialized iofile_t.

#ifndef IOFILE_H
#define IOFILE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*  data;    // mapped region (NULL when closed)
    size_t size;    // size in bytes
} iofile_t;

// On success: returns 0, writes mapping to *out.
// On failure: returns -1, writes a malloc'd error message to *err (if non-NULL).
int  iofile_mmap_ro(const char* path, iofile_t* out, char** err);

// Unmap and zero out *f. No-op on a zero/closed iofile_t.
void iofile_close(iofile_t* f);

#ifdef __cplusplus
}
#endif
#endif
