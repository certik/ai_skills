// patterns/gemv_simdgroup_per_output.metal
//
// WHAT: One SIMD group (32 threads on Apple GPUs) computes one output
//       element of a GEMV.  Each thread accumulates K/32 of the inner
//       product; final reduction via simd_sum.
// WHEN: For ANY bf16 GEMV.  This is the baseline above 1-thread-per-output.
// SPEEDUP: 1.3–1.5× vs single-thread-per-output (5b7b16c).
// SHAPES: K must be a multiple of SIMD_WIDTH (32).
// COMMIT: 5b7b16c — "gemv_bf16: simdgroup-per-output + bfloat4 vector loads"

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_WIDTH = 32;

// Compute Y[n] = bias[n] + sum_k X[k] * W[n, k]   (one row of W per output).
// Grid:        (N * SIMD_WIDTH, 1, 1)    one SG per output
// Threadgroup: (SIMD_WIDTH, 1, 1)        32 threads per SG
kernel void gemv_bf16(device const bfloat*  X        [[buffer(0)]],
                      device const bfloat*  W        [[buffer(1)]],   // [N, K], row-major
                      device const bfloat*  B        [[buffer(2)]],   // [N]
                      device bfloat*        Y        [[buffer(3)]],   // [N]
                      constant uint2&       dims     [[buffer(4)]],   // (K, N)
                      uint                  tid      [[thread_position_in_grid]],
                      uint                  lane     [[thread_index_in_simdgroup]],
                      uint                  sgid     [[simdgroup_index_in_threadgroup]])
{
    uint K = dims.x;
    uint N = dims.y;
    uint n = tid / SIMD_WIDTH;      // one SG per output element
    if (n >= N) return;

    const device bfloat* xrow = X;
    const device bfloat* wrow = W + n * K;

    // Each lane sums K/SIMD_WIDTH terms.
    float acc = 0.0f;
    for (uint k = lane; k < K; k += SIMD_WIDTH) {
        acc += float(xrow[k]) * float(wrow[k]);
    }
    // SIMD-group reduction in one instruction.
    acc = simd_sum(acc);

    if (lane == 0) {
        Y[n] = bfloat(float(B[n]) + acc);
    }
}

// VARIANT — with bfloat4 vector loads (K multiple of 4 * SIMD_WIDTH = 128):
//
//   const device bfloat4* xrow4 = (const device bfloat4*)X;
//   const device bfloat4* wrow4 = (const device bfloat4*)(W + n * K);
//   float acc = 0.0f;
//   for (uint k4 = lane; k4 < K/4; k4 += SIMD_WIDTH) {
//       float4 x = float4(xrow4[k4]);
//       float4 w = float4(wrow4[k4]);
//       acc += dot(x, w);
//   }
//   acc = simd_sum(acc);
//   if (lane == 0) Y[n] = bfloat(float(B[n]) + acc);
