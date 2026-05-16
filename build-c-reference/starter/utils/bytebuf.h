// bytebuf.h — growable byte buffer (C answer to Python's `bytearray`).
//
// Aborts the program on allocation failure (same policy as Python's bytearray
// raising MemoryError). All operations are O(amortized 1) per byte appended.
//
// Usage:
//   bytebuf_t b = BYTEBUF_INIT;
//   bb_append(&b, "hello", 5);
//   bb_append_u32(&b, 0x12345678);          // little-endian
//   bb_write_file(&b, "out.bin");
//   bb_free(&b);
//
// To hand the underlying allocation to another owner, use bb_release().

#ifndef BYTEBUF_H
#define BYTEBUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned char* data;
    size_t         len;
    size_t         cap;
} bytebuf_t;

#define BYTEBUF_INIT { 0, 0, 0 }

void   bb_init       (bytebuf_t* b);
void   bb_free       (bytebuf_t* b);
void   bb_reserve    (bytebuf_t* b, size_t need_cap);
void   bb_append     (bytebuf_t* b, const void* data, size_t n);
void   bb_append_u8  (bytebuf_t* b, uint8_t  v);
void   bb_append_u32 (bytebuf_t* b, uint32_t v);  // little-endian

// Hand over the allocation to the caller (who must free()).
// Writes the length to *out_len (if non-NULL) and resets the bytebuf.
void*  bb_release    (bytebuf_t* b, size_t* out_len);

// Write the buffer to a file. Returns 0 on success, -1 on error (errno set
// by the failing fopen/fwrite/fclose).
int    bb_write_file (const bytebuf_t* b, const char* path);

#ifdef __cplusplus
}
#endif
#endif
