// patterns/linear_dot4xM_gemm_tile.c — proper GEMM tile for M>1 prefill.
//
// WHAT: For M rows of X (the prefill case where M = prompt-length tokens
//       fed in parallel), produce M*N output. Loop order is N (parallel)
//       -> K -> M so the W tile is loaded ONCE per (n0..n0+3) and reused
//       across all M X rows. Equivalent to a small 4xM-output register
//       tile fed by a TILE_K column block of W.
//
// WHEN: After linear_dot4_ktile is in place. Decode (M=1) routes to the
//       GEMV path; this kernel only runs for prefill / batched decode.
//
// WHY:  A decode-shaped GEMV looped over M re-reads W from main memory
//       M times. With M=21 (a 21-token prompt), prefill is doing 21x the
//       weight traffic of a single decode step — for-free 21x speedup is
//       sitting on the floor as long as we restructure the loops to amortize
//       W. Pre-converting all of X once (M*K floats) before the parallel
//       region also pays off because each W tile sees all M rows.
//
// EXPECTED SPEEDUP: Prefill 2.1x (Gemma 4 E4B baseline after K-tile: 27.6 -> 58.9
//                   tok/s); decode unchanged (still the M=1 path).
//
// Original commit: see `src-cpu: turn linear M>1 path into real GEMM` in
// the Gemma 4 E4B `cpu` branch.

#include <stdint.h>
#include <string.h>

typedef uint16_t bf16;

#ifndef LINEAR_TILE_K
#define LINEAR_TILE_K 128u
#endif

// Accumulator layout: acc[4*m + j] is for output column (n0+j), input row m.
// Caller is responsible for allocating M*4 floats and clearing them — this
// kernel *adds into* `acc` (so the caller can call it once per N tile and
// reuse the accumulator buffer for the bias add + bf16 store).
//
// Actually no — for simplicity here we clear inside; caller zeros once per
// N-tile call. See linear_bf16 wiring at the bottom.
static inline void linear_dot4xM_tile(const float* __restrict__ xf, uint32_t M,
                                      uint32_t K_stride,
                                      const bf16* __restrict__ w0,
                                      const bf16* __restrict__ w1,
                                      const bf16* __restrict__ w2,
                                      const bf16* __restrict__ w3,
                                      uint32_t K,
                                      float* __restrict__ acc /* M*4 */) {
    for (uint32_t i = 0; i < 4 * M; i++) acc[i] = 0.0f;

    float b0[LINEAR_TILE_K] __attribute__((aligned(64)));
    float b1[LINEAR_TILE_K] __attribute__((aligned(64)));
    float b2[LINEAR_TILE_K] __attribute__((aligned(64)));
    float b3[LINEAR_TILE_K] __attribute__((aligned(64)));

    uint32_t k = 0;
    for (; k + LINEAR_TILE_K <= K; k += LINEAR_TILE_K) {
        const bf16* p0=w0+k, *p1=w1+k, *p2=w2+k, *p3=w3+k;

        // Phase A: convert 4 W rows of TILE_K bf16 -> fp32 (loaded ONCE per
        // tile, reused across all M X rows below).
        #pragma omp simd
        for (uint32_t i=0;i<LINEAR_TILE_K;i++){uint32_t u=((uint32_t)p0[i])<<16;memcpy(&b0[i],&u,4);}
        #pragma omp simd
        for (uint32_t i=0;i<LINEAR_TILE_K;i++){uint32_t u=((uint32_t)p1[i])<<16;memcpy(&b1[i],&u,4);}
        #pragma omp simd
        for (uint32_t i=0;i<LINEAR_TILE_K;i++){uint32_t u=((uint32_t)p2[i])<<16;memcpy(&b2[i],&u,4);}
        #pragma omp simd
        for (uint32_t i=0;i<LINEAR_TILE_K;i++){uint32_t u=((uint32_t)p3[i])<<16;memcpy(&b3[i],&u,4);}

        // Phase B: accumulate against M X rows. For each m, hoist acc[..]
        // into registers, run the vectorized fp32 FMA, write back. This
        // re-uses b0..b3 (= ~2 KB fp32 in L1) for each of the M rows.
        for (uint32_t m = 0; m < M; m++) {
            const float* xfi = xf + (size_t)m * K_stride + k;
            float a0 = acc[4*m+0], a1 = acc[4*m+1], a2 = acc[4*m+2], a3 = acc[4*m+3];
            #pragma omp simd reduction(+:a0,a1,a2,a3)
            for (uint32_t i = 0; i < LINEAR_TILE_K; i++) {
                float xv = xfi[i];
                a0 += xv * b0[i]; a1 += xv * b1[i];
                a2 += xv * b2[i]; a3 += xv * b3[i];
            }
            acc[4*m+0]=a0; acc[4*m+1]=a1; acc[4*m+2]=a2; acc[4*m+3]=a3;
        }
    }
    // Tail (k < TILE_K leftover).
    for (; k < K; k++) {
        uint32_t u; float f0,f1,f2,f3;
        u=((uint32_t)w0[k])<<16; memcpy(&f0,&u,4);
        u=((uint32_t)w1[k])<<16; memcpy(&f1,&u,4);
        u=((uint32_t)w2[k])<<16; memcpy(&f2,&u,4);
        u=((uint32_t)w3[k])<<16; memcpy(&f3,&u,4);
        for (uint32_t m = 0; m < M; m++) {
            float xv = xf[(size_t)m * K_stride + k];
            acc[4*m+0] += xv*f0; acc[4*m+1] += xv*f1;
            acc[4*m+2] += xv*f2; acc[4*m+3] += xv*f3;
        }
    }
}

// Wiring inside linear_bf16 for the M>1 path. Note the per-thread
// persistent accumulator (allocated ONCE per parallel region, not once per
// linear call). For M=1024 that's 16 KB per thread, trivial.
//
//     const uint32_t N4 = N & ~3u;
//     #pragma omp parallel
//     {
//         float* acc = aligned_alloc(64, (size_t)M * 4 * sizeof(float));
//         #pragma omp for schedule(static)
//         for (uint32_t n0 = 0; n0 < N4; n0 += 4) {
//             linear_dot4xM_tile(xf, M, K,
//                                W + (size_t)(n0  )*K, W + (size_t)(n0+1)*K,
//                                W + (size_t)(n0+2)*K, W + (size_t)(n0+3)*K,
//                                K, acc);
//             /* bias add into acc[], bf16 store M*4 outputs */
//         }
//         free(acc);
//     }
