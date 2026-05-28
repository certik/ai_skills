// patterns/linear_dot4_ktile.c — THE big GEMV win on bf16 / CPUs without bf16-FMA.
//
// WHAT: Compute 4 outputs of a (K)x4 GEMV: o0..o3 = dot(xf, w0..w3).
//       xf is pre-converted fp32; w0..w3 are bf16. K is large (≥hidden dim).
//
// WHEN: As soon as the canonical `linear_bf16` is your hot kernel (KPROF
//       confirms it dominates). Decode tok/s rarely benefits (BW-bound) but
//       prefill speedup is typically 2-3x.
//
// WHY:  The naive fused inner loop
//
//           for (k=0;k<K;k++) { float x = bf16_to_f32(xrow[k]);
//                               s0 += x * bf16_to_f32(w0[k]); ... }
//
//       does NOT auto-vectorize on gcc 13. Inspect with:
//           gcc -S -O3 -march=native kernels.c -o /tmp/k.s
//           grep -cE 'vfmadd231ps|vfmadd132ss' /tmp/k.s
//       You'll see lots of scalar vfmadd132ss. Splitting into two phases
//       per TILE_K block (Phase A: convert TILE bf16 -> TILE fp32 scratch;
//       Phase B: pure-fp32 FMA reduction from scratch) lets gcc emit
//       vpmovzxwd + vpslld for A and vfmadd231ps for B.
//
// EXPECTED SPEEDUP: Prefill 3.1x (Gemma 4 E4B AMD EPYC 7763 baseline 8.9 -> 27.6),
//                   decode unchanged (already memory-BW-bound past phase 1).
//
// Tradeoff: -ffast-math + bf16 (8 mantissa bits) lets borderline values
// flip after the FMA reduction is reordered by the vectorizer; greedy
// argmax may diverge from the scalar reference after ~50 tokens. See the
// "Validation discipline" section of the parent SKILL.md.
//
// Original commit: see `src-cpu: K-tile linear_dot4 (split bf16->fp32 from FMA)`
// in the Gemma 4 E4B `cpu` branch.

#include <stdint.h>
#include <string.h>

typedef uint16_t bf16;

#ifndef LINEAR_TILE_K
#define LINEAR_TILE_K 128u   // 4*128 fp32 = 2 KB stack tile; fits L1 trivially
#endif

static inline void linear_dot4_ktile(const float* __restrict__ xf,
                                     const bf16*  __restrict__ w0,
                                     const bf16*  __restrict__ w1,
                                     const bf16*  __restrict__ w2,
                                     const bf16*  __restrict__ w3,
                                     uint32_t K,
                                     float* __restrict__ o0, float* __restrict__ o1,
                                     float* __restrict__ o2, float* __restrict__ o3) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    // Stack scratch (must be aligned for gcc to pick AVX2 vmovaps).
    float b0[LINEAR_TILE_K] __attribute__((aligned(64)));
    float b1[LINEAR_TILE_K] __attribute__((aligned(64)));
    float b2[LINEAR_TILE_K] __attribute__((aligned(64)));
    float b3[LINEAR_TILE_K] __attribute__((aligned(64)));

    uint32_t k = 0;
    for (; k + LINEAR_TILE_K <= K; k += LINEAR_TILE_K) {
        const bf16* p0 = w0 + k;
        const bf16* p1 = w1 + k;
        const bf16* p2 = w2 + k;
        const bf16* p3 = w3 + k;

        // Phase A: convert TILE bf16 -> TILE fp32 (vectorizes to
        //          vpmovzxwd + vpslld + vmovups -> AVX2 8-wide).
        #pragma omp simd
        for (uint32_t i = 0; i < LINEAR_TILE_K; i++) {
            uint32_t u = ((uint32_t)p0[i]) << 16; memcpy(&b0[i], &u, 4);
        }
        #pragma omp simd
        for (uint32_t i = 0; i < LINEAR_TILE_K; i++) {
            uint32_t u = ((uint32_t)p1[i]) << 16; memcpy(&b1[i], &u, 4);
        }
        #pragma omp simd
        for (uint32_t i = 0; i < LINEAR_TILE_K; i++) {
            uint32_t u = ((uint32_t)p2[i]) << 16; memcpy(&b2[i], &u, 4);
        }
        #pragma omp simd
        for (uint32_t i = 0; i < LINEAR_TILE_K; i++) {
            uint32_t u = ((uint32_t)p3[i]) << 16; memcpy(&b3[i], &u, 4);
        }

        // Phase B: pure fp32 FMA (vectorizes to vfmadd231ps -> AVX2 8-wide).
        const float* xfi = xf + k;
        #pragma omp simd reduction(+:a0,a1,a2,a3)
        for (uint32_t i = 0; i < LINEAR_TILE_K; i++) {
            float xv = xfi[i];
            a0 += xv * b0[i];
            a1 += xv * b1[i];
            a2 += xv * b2[i];
            a3 += xv * b3[i];
        }
    }
    // Tail (k < TILE_K leftovers).
    for (; k < K; k++) {
        float xv = xf[k];
        uint32_t u0 = ((uint32_t)w0[k]) << 16; float f0; memcpy(&f0, &u0, 4);
        uint32_t u1 = ((uint32_t)w1[k]) << 16; float f1; memcpy(&f1, &u1, 4);
        uint32_t u2 = ((uint32_t)w2[k]) << 16; float f2; memcpy(&f2, &u2, 4);
        uint32_t u3 = ((uint32_t)w3[k]) << 16; float f3; memcpy(&f3, &u3, 4);
        a0 += xv * f0; a1 += xv * f1; a2 += xv * f2; a3 += xv * f3;
    }
    *o0 = a0; *o1 = a1; *o2 = a2; *o3 = a3;
}

// Usage pattern in the caller (M=1, decode):
//
//   #pragma omp parallel for schedule(static)
//   for (uint32_t n0 = 0; n0 < (N & ~3u); n0 += 4) {
//       float s0,s1,s2,s3;
//       linear_dot4_ktile(xf,
//                         W + (size_t)(n0  )*K, W + (size_t)(n0+1)*K,
//                         W + (size_t)(n0+2)*K, W + (size_t)(n0+3)*K,
//                         K, &s0,&s1,&s2,&s3);
//       /* + bias, + bf16 store */
//   }
//
// Where xf was pre-converted from X once at the top of `linear_bf16`:
//
//   float* xf = aligned_alloc(64, (size_t)M * K * sizeof(float));
//   #pragma omp parallel for schedule(static) if(M >= 2)
//   for (uint32_t m = 0; m < M; m++)
//       bf16_row_to_f32(X + (size_t)m*K, xf + (size_t)m*K, K);
