// bytebuf.c — growable byte buffer.

#include "bytebuf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void oom(void) {
    fprintf(stderr, "bytebuf: out of memory\n");
    abort();
}

void bb_init(bytebuf_t* b) {
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

void bb_free(bytebuf_t* b) {
    free(b->data);
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

void bb_reserve(bytebuf_t* b, size_t need) {
    if (b->cap >= need) return;
    size_t new_cap = b->cap ? b->cap : 64;
    while (new_cap < need) new_cap *= 2;
    unsigned char* p = (unsigned char*)realloc(b->data, new_cap);
    if (!p) oom();
    b->data = p;
    b->cap  = new_cap;
}

void bb_append(bytebuf_t* b, const void* data, size_t n) {
    if (n == 0) return;
    bb_reserve(b, b->len + n);
    memcpy(b->data + b->len, data, n);
    b->len += n;
}

void bb_append_u8(bytebuf_t* b, uint8_t v) {
    bb_reserve(b, b->len + 1);
    b->data[b->len++] = v;
}

void bb_append_u32(bytebuf_t* b, uint32_t v) {
    bb_reserve(b, b->len + 4);
    b->data[b->len++] = (unsigned char)( v        & 0xff);
    b->data[b->len++] = (unsigned char)((v >>  8) & 0xff);
    b->data[b->len++] = (unsigned char)((v >> 16) & 0xff);
    b->data[b->len++] = (unsigned char)((v >> 24) & 0xff);
}

void* bb_release(bytebuf_t* b, size_t* out_len) {
    void* d = b->data;
    if (out_len) *out_len = b->len;
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
    return d;
}

int bb_write_file(const bytebuf_t* b, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    if (b->len > 0 && fwrite(b->data, 1, b->len, f) != b->len) {
        fclose(f);
        return -1;
    }
    if (fclose(f) != 0) return -1;
    return 0;
}
