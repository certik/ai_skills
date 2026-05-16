// utf8.c — minimal UTF-8 codec.

#include "utf8.h"

size_t utf8_encode(int cp, char* out, size_t cap) {
    if (cp < 0) return 0;
    if (cp < 0x80) {
        if (cap < 1) return 0;
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        if (cap < 2) return 0;
        out[0] = (char)(0xc0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        if (cap < 3) return 0;
        out[0] = (char)(0xe0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    if (cp <= 0x10ffff) {
        if (cap < 4) return 0;
        out[0] = (char)(0xf0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[3] = (char)(0x80 | (cp & 0x3f));
        return 4;
    }
    return 0;
}

size_t utf8_decode(const char* p, const char* end, int* cp) {
    if (p >= end) return 0;
    unsigned char c = (unsigned char)p[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xe0) == 0xc0) {
        if (p + 2 > end) return 0;
        *cp = ((c & 0x1f) << 6) | ((unsigned char)p[1] & 0x3f);
        return 2;
    }
    if ((c & 0xf0) == 0xe0) {
        if (p + 3 > end) return 0;
        *cp = ((c & 0x0f) << 12)
            | (((unsigned char)p[1] & 0x3f) << 6)
            |  ((unsigned char)p[2] & 0x3f);
        return 3;
    }
    if ((c & 0xf8) == 0xf0) {
        if (p + 4 > end) return 0;
        *cp = ((c & 0x07) << 18)
            | (((unsigned char)p[1] & 0x3f) << 12)
            | (((unsigned char)p[2] & 0x3f) << 6)
            |  ((unsigned char)p[3] & 0x3f);
        return 4;
    }
    return 0;
}
