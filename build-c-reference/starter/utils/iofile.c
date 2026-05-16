// iofile.c — minimal read-only file mmap helper.

#define _POSIX_C_SOURCE 200809L
#include "iofile.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static char* errf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return strdup(buf);
}

int iofile_mmap_ro(const char* path, iofile_t* out, char** err) {
    out->data = NULL;
    out->size = 0;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (err) *err = errf("open %s: %s", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        if (err) *err = errf("fstat %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    size_t sz = (size_t)st.st_size;
    if (sz == 0) {
        if (err) *err = errf("empty file: %s", path);
        close(fd);
        return -1;
    }
    void* m = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        if (err) *err = errf("mmap %s: %s", path, strerror(errno));
        return -1;
    }
    out->data = m;
    out->size = sz;
    return 0;
}

void iofile_close(iofile_t* f) {
    if (!f) return;
    if (f->data) munmap(f->data, f->size);
    f->data = NULL;
    f->size = 0;
}
