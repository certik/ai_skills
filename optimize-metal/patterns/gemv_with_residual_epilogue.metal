// patterns/gemv_with_residual_epilogue.metal
//
// WHAT: The qmv4 GEMV pattern (gemv_qmv4_register_tile.metal) with a
//       fused residual-add in the write step.  Replaces the separate
//       residual_add kernel for o_proj and expert_mix in decode.
// WHEN: After the qmv4 register-tile pattern is working.  Apply on
//       o_proj (after attention) and expert_mix (after MoE).
// SPEEDUP: 1.03–1.05× decode (saves 24 + 24 dispatches per token + a
//          full pass over the residual buffer).
// COMMITS:
//   130075a — decode: fuse residual add into o_proj and mix epilogues
//   4e3bb14 — csrc: fuse res_attn into gemm_bf16 (template DO_ADD); drop residual_add kernel
//   ba83808 — csrc: fuse res_mlp into combine_scatter; drop unused metal_shim API

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint K_OUT      = 8;
constant constexpr uint K_VEC      = 4;     // bfloat4 vector load width

// Y[n] = residual[n] + bias[n] + W[n, :] @ X
//
// Drops the subsequent residual_add(X, Y) kernel — caller passes the
// residual buffer (typically the pre-attention X) as an extra input, and
// the kernel writes the sum directly to X.
kernel void gemv_bf16_add(device const bfloat*  X        [[buffer(0)]],
                          device const bfloat*  W        [[buffer(1)]],
                          device const bfloat*  B        [[buffer(2)]],
                          device bfloat*        residual [[buffer(3)]],  // INPUT + OUTPUT
                          constant uint2&       dims     [[buffer(4)]],
                          uint                  tid      [[thread_position_in_grid]],
                          uint                  lane     [[thread_index_in_simdgroup]])
{
    uint K = dims.x;
    uint N = dims.y;
    uint n_base = (tid / SIMD_WIDTH) * K_OUT;
    if (n_base >= N) return;

    const device bfloat4* xrow4 = (const device bfloat4*)X;

    float acc[K_OUT];
    for (uint o = 0; o < K_OUT; ++o) acc[o] = 0.0f;

    for (uint k4 = lane; k4 < K / K_VEC; k4 += SIMD_WIDTH) {
        float4 xv = float4(xrow4[k4]);
        for (uint o = 0; o < K_OUT; ++o) {
            uint n = n_base + o;
            if (n >= N) continue;
            const device bfloat4* wrow4 = (const device bfloat4*)(W + n * K);
            float4 wv = float4(wrow4[k4]);
            acc[o] += dot(xv, wv);
        }
    }

    for (uint o = 0; o < K_OUT; ++o) {
        float r = simd_sum(acc[o]);
        if (lane == 0) {
            uint n = n_base + o;
            if (n < N) {
                float fused = float(residual[n]) + float(B[n]) + r;
                residual[n] = bfloat(fused);
            }
        }
    }
}

// NOTE: bf16 round point shifts vs the un-fused version (residual is now
// added inside the same f32 accumulator instead of as a separate bf16
// round-trip).  Validate first N tokens still match; tolerate the small
// numerical drift beyond.
