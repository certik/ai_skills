// patterns/mxfp4_qmv4_decode.metal
//
// WHAT: MXFP4 quantized GEMV for decode-time MoE.  Combines:
//   * gather expert weights by indices (no materialization)
//   * inline e2m1 + e8m0 dequant (per-32-element block)
//   * qmv4 register-tile pattern (K_OUT outputs per SG, X reused)
//   * fully unrolled inner 32-element block loop
// WHEN: First MoE optimization for decode.  Combined with E2 (fuse
//       gate+up+swiglu) this gets you 2–4× decode speedup.
// SHAPES: K_BLOCKS = K / 32.  K must be a multiple of 32.
// COMMITS:
//   86e542e — mxfp4 + gemv share x via threadgroup memory
//   eee8282 — mxfp4: fully unrolled inner loop with vector loads
//   f536ba1 — mxfp4 expert GEMV: simdgroup-per-output
//   b69d31b — decode: register-cached X qmv4 kernels (gate_up_swiglu + down)
//   b15e957 — Fuse mxfp4 gate+up+swiglu into one kernel
//
// MXFP4 format:
//   * weight rows stored as bytes; each byte packs 2 fp4_e2m1 values
//     -> 32 weights per 16-byte "block"
//   * one e8m0 scale byte per 32-weight block (per row)
//   * dequant(byte&0xF) = fp4_lut[byte&0xF] * ldexpf(1.0, scale-127)

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint K_OUT      = 4;      // outputs per SG (qmv4)

constant float fp4_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float e8m0_scale(uchar b) {
    return (b == 0xFF) ? 0.0f : exp2(float(int(b) - 127));
}

// y[l, kt, n] = bias[E_kt, n] + sum_b dequant(W[E_kt, n, b]) . X[l, b*32:(b+1)*32]
// Grid:        (N * SIMD_WIDTH, K_top, L)
// Threadgroup: (SIMD_WIDTH, 1, 1)
kernel void mxfp4_qmv4_bf16(device const bfloat*   X        [[buffer(0)]],  // [L, K]
                            device const uchar*    blocks   [[buffer(1)]],  // [E, N, K_blocks, 16]
                            device const uchar*    scales   [[buffer(2)]],  // [E, N, K_blocks]
                            device const bfloat*   bias     [[buffer(3)]],  // [E, N]
                            device const int*      indices  [[buffer(4)]],  // [L, K_top]
                            device bfloat*         Y        [[buffer(5)]],  // [L, K_top, N]
                            constant uint4&        dims     [[buffer(6)]],  // (L, K_top, N, K_blocks)
                            uint3                  gid      [[thread_position_in_grid]],
                            uint                   lane     [[thread_index_in_simdgroup]])
{
    uint L = dims.x, K_top = dims.y, N = dims.z, K_blocks = dims.w;
    uint n_base = (gid.x / SIMD_WIDTH) * K_OUT;
    uint kt = gid.y;
    uint l  = gid.z;
    if (n_base >= N) return;

    int e = indices[l * K_top + kt];

    // Per-lane register: 32 X values across one K-block.
    const device bfloat* xrow = X + l * (K_blocks * 32);

    float acc[K_OUT];
    for (uint o = 0; o < K_OUT; ++o) acc[o] = 0.0f;

    // K-block loop.  Each iteration: load 32 X values (one each per lane),
    // load+dequant 32 W values per output, mac.
    for (uint b = 0; b < K_blocks; ++b) {
        // One x value per lane (32 lanes × 1 = 32 elements per block).
        float xv = float(xrow[b * 32 + lane]);

        // For each output: dequant the 32-elem block then dot.
        // Layout: blocks[e, n, b, 0..15] is 16 bytes, 32 nibbles.
        for (uint o = 0; o < K_OUT; ++o) {
            uint n = n_base + o;
            if (n >= N) continue;
            size_t blk_off = (((size_t)e * N + n) * K_blocks + b) * 16;
            float sc = e8m0_scale(scales[((size_t)e * N + n) * K_blocks + b]);
            // Each lane reads its own byte (lane in [0,32) -> byte in [0,16) + nibble half).
            uint byte_i = lane >> 1;
            uchar byte  = blocks[blk_off + byte_i];
            uint  nib   = (lane & 1) ? (byte >> 4) : (byte & 0xF);
            float w     = fp4_lut[nib] * sc;
            acc[o] += xv * w;
        }
    }

    // Reduce + write.
    for (uint o = 0; o < K_OUT; ++o) {
        float r = simd_sum(acc[o]);
        if (lane == 0) {
            uint n = n_base + o;
            if (n < N) {
                Y[(l * K_top + kt) * N + n] = bfloat(float(bias[e * N + n]) + r);
            }
        }
    }
}

// FUSED GATE+UP+SWIGLU variant: same loop structure but iterates the
// gate row (even n) and up row (odd n) together, writes only mid[l,kt,n]
// after applying SwiGLU: (clamp(gate, -inf, limit) * sigmoid(alpha*gate))
//                       * (clamp(up, -limit, limit) + 1)
