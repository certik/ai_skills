// utf8.h — minimal UTF-8 codec (C answer to Python's `str` <-> codepoint).
//
// Both functions are pure: no allocations, no globals, no errno.

#ifndef UTF8_H
#define UTF8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Encode one Unicode codepoint as 1..4 UTF-8 bytes into `out`.
// Returns the number of bytes written, or 0 on error (invalid codepoint
// or insufficient capacity).
size_t utf8_encode(int cp, char* out, size_t cap);

// Decode one UTF-8 sequence at [p, end). Writes the codepoint to *cp and
// returns the number of bytes consumed (1..4), or 0 on error (truncated
// sequence or invalid lead byte). Does not validate continuation bytes
// beyond the lead-byte length signal.
size_t utf8_decode(const char* p, const char* end, int* cp);

#ifdef __cplusplus
}
#endif
#endif
