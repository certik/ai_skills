// bf16.h — brain-float-16 <-> float32 conversion helpers.
// Round-to-nearest-even on the f32 -> bf16 path; NaN preserved.
//
// bf16 is the upper 16 bits of an IEEE-754 binary32 (1 sign + 8 exponent
// + 7 mantissa).  Conversion both ways is a single uint16 <-> uint32 op.
//
// Drop-in. Include in kernels.c and tests.

#ifndef BF16_H
#define BF16_H

#include <stdint.h>

typedef uint16_t bf16;

static inline float bf16_to_f32(bf16 h) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)h) << 16;
    return v.f;
}

static inline bf16 f32_to_bf16(float f) {
    union { uint32_t u; float f; } v; v.f = f;
    uint32_t x = v.u;
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        // NaN: keep top mantissa bit set so it stays NaN after truncation.
        return (bf16)((x >> 16) | 0x40u);
    }
    // Round-to-nearest-even.
    uint32_t lsb  = (x >> 16) & 1u;
    uint32_t bias = 0x7fffu + lsb;
    return (bf16)((x + bias) >> 16);
}

#endif
