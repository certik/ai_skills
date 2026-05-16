// patterns/gemv_qmv4_register_tile.metal
//
// WHAT: Each SIMD group computes K outputs (typically K_OUT = 4 or 8) of
//       a GEMV.  The X vector is loaded ONCE per SG into registers and
//       reused across all K_OUT outputs.  Cuts X bandwidth by K_OUT.
//       This is MLX's qmv_fast pattern.
// WHEN: After gemv_simdgroup_per_output.  Most GEMVs are X-bandwidth bound.
// SPEEDUP: 1.2–1.4× per GEMV; 1.10–1.20× overall decode.
// SHAPES: K must be a multiple of SIMD_WIDTH; N must be a multiple of K_OUT.
// COMMITS:
//   9fce171 — gemv_bf16_4 (4 outputs/SG, 64 threads/TG; mlx qmv_fast pattern)
//   b7d7cf2 — 8-X register tile in bf16 GEMV
//   b69d31b — register-cached X qmv4 for mxfp4

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint K_OUT      = 8;     // outputs per SG
constant constexpr uint K_VEC      = 4;     // bfloat4 vector load width

// Y[n + 0..K_OUT] = bias[n + 0..K_OUT] + W[n + 0..K_OUT, :] @ X
// Grid:        (ceil(N/K_OUT) * SIMD_WIDTH, 1, 1)
// Threadgroup: (SIMD_WIDTH, 1, 1)
kernel void gemv_qmv4_bf16(device const bfloat*  X        [[buffer(0)]],
                           device const bfloat*  W        [[buffer(1)]],   // [N, K]
                           device const bfloat*  B        [[buffer(2)]],
                           device bfloat*        Y        [[buffer(3)]],
                           constant uint2&       dims     [[buffer(4)]],   // (K, N)
                           uint                  tid      [[thread_position_in_grid]],
                           uint                  lane     [[thread_index_in_simdgroup]])
{
    uint K = dims.x;
    uint N = dims.y;
    uint n_base = (tid / SIMD_WIDTH) * K_OUT;
    if (n_base >= N) return;

    const device bfloat4* xrow4 = (const device bfloat4*)X;

    // Accumulators in registers: one per output.
    float acc[K_OUT];
    for (uint o = 0; o < K_OUT; ++o) acc[o] = 0.0f;

    // Inner loop: each lane reads X[k:k+K_VEC] ONCE and uses it for ALL K_OUT outputs.
    for (uint k4 = lane; k4 < K / K_VEC; k4 += SIMD_WIDTH) {
        float4 xv = float4(xrow4[k4]);                       // shared across outputs

        // Hand-unrolled across the K_OUT outputs.
        for (uint o = 0; o < K_OUT; ++o) {
            uint n = n_base + o;
            if (n >= N) continue;
            const device bfloat4* wrow4 = (const device bfloat4*)(W + n * K);
            float4 wv = float4(wrow4[k4]);
            acc[o] += dot(xv, wv);
        }
    }

    // Reduce + write.
    for (uint o = 0; o < K_OUT; ++o) {
        float r = simd_sum(acc[o]);
        if (lane == 0) {
            uint n = n_base + o;
            if (n < N) Y[n] = bfloat(float(B[n]) + r);
        }
    }
}

// EXTENSION: fuse residual epilogue.  Add a 5th buffer `residual` and
// at the write step do `Y[n] = bfloat(float(B[n]) + r + float(residual[n]))`.
// See patterns/gemv_with_residual_epilogue.metal.
