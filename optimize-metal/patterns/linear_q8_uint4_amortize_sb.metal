// patterns/linear_q8_uint4_amortize_sb.metal
//
// WHAT: q8 GEMV decode kernel that loads packed W as uint4 (16 bytes
//       = 16 q8 weights = exactly one 64-element quant group), so one
//       (scale, bias) load suffices per 16 weights. The naive form
//       loads W as uint32 (4 weights) and re-reads (s, b) every 4
//       weights even though all 4 belong to the same group — pure
//       redundant memory traffic.
//
//       This is catalog B6 — "amortize per-group state across the
//       group". Same principle applies to q4 (one (s, b) per 8
//       weights) and mxfp4 (one exponent per 32-byte group), with
//       different load widths and unpacking arithmetic.
//
// WHEN: After B1 (SG-per-output) + B2 (bfloat4 X loads) + B3 (K_OUT
//       register tile) are in place. The signal is "quantized GEMV
//       family dominates KPROF (70%+) and runs at ~25–35% of
//       theoretical W-bandwidth peak". B6 closes most of the gap
//       to the W-BW floor (~50% peak).
//
// SPEEDUP: 1.15–1.20× decode on whole pipeline when quantized GEMV
//          dominates. Qwen3.6-35B-A3B: 82.8 → 98.4 tok/s (+18.7%).
//
// SHAPES:
//   - K must be a multiple of 16 (or 4 × pack_unit for q4).
//     Trivial for all transformer K (256, 512, 2048, 4096).
//   - group_size of the quant format must be ≥ 4 × pack_unit
//     (q8 group_size=64 ≥ 16; q4 group_size=32 ≥ 8). Both hold
//     for standard MLX-style affine formats.
//   - K_OUT = 4. Do NOT try K_OUT=8 — the heavier inner loop
//     (4 X-regs + 8 W-regs + 8 accs) spills registers and
//     regresses badly (gotcha #8).
//
// COMMIT: cba7169 — "linear_q8 uint4 W loads (1 (s,b) load per 16
//                    q8) — decode +18.7% (82.8→98.3 tok/s)"

#include <metal_stdlib>
using namespace metal;

constant constexpr uint LQ8_SIMD_WIDTH = 32u;
constant constexpr uint LQ8_K_OUT      = 4u;     // outputs per SG. Stay at 4.

struct linear_dims { uint M; uint K; uint N; };

// Helper — dequantize one uint32 of 4 packed q8 bytes via FMA.
// One scale/bias supplied externally (constant across the uint4).
static inline float4 lq8_deq_word(uint w, float s, float b) {
    float4 v;
    v.x = float((w      ) & 0xffu) * s + b;
    v.y = float((w >>  8) & 0xffu) * s + b;
    v.z = float((w >> 16) & 0xffu) * s + b;
    v.w = float((w >> 24) & 0xffu) * s + b;
    return v;
}

// Y[m, n_base..n_base+K_OUT] = bias[n_base..] +
//     dequant(W[n_base.., :]; S, Bq) @ X[m, :]
//
// W is stored as uint32 (4 packed q8 bytes per word, K/4 words/row).
// S, Bq are bf16 (or fp16/fp32) per quant group, K/64 groups/row.
//
// Grid:        (LQ8_SIMD_WIDTH * ceil(N/K_OUT), M, 1)
// Threadgroup: (LQ8_SIMD_WIDTH, 1, 1)
kernel void linear_q8_bf16(device const bfloat*  X     [[buffer(0)]],
                           device const uint*    W     [[buffer(1)]],   // packed q8
                           device const bfloat*  S     [[buffer(2)]],   // [N, K/64]
                           device const bfloat*  Bq    [[buffer(3)]],   // [N, K/64]
                           device const bfloat*  B     [[buffer(4)]],   // [N]
                           device bfloat*        Y     [[buffer(5)]],   // [M, N]
                           constant linear_dims& dims  [[buffer(6)]],
                           uint3 tg   [[threadgroup_position_in_grid]],
                           uint  lane [[thread_index_in_simdgroup]])
{
    uint n_base = tg.x * LQ8_K_OUT;
    uint m      = tg.y;
    if (m >= dims.M || n_base >= dims.N) return;

    uint K       = dims.K;
    uint Kpack   = K >> 2;        // # uint32 per row (= K/4)
    uint K_u4    = Kpack >> 2;    // # uint4  per row (= K/16)
    uint Kgroups = K >> 6;        // # quant groups per row (group_size = 64)

    // X row as bfloat4 for fast vector loads.
    const device bfloat4* xrow4 = (const device bfloat4*)(X + m * K);

    // Per-output row pointers: reinterpret W as uint4 (one uint4 = one quant group).
    const device uint4*   w_rows4[LQ8_K_OUT];
    const device bfloat*  s_rows[LQ8_K_OUT];
    const device bfloat*  b_rows[LQ8_K_OUT];
    bool valid[LQ8_K_OUT];
    for (uint o = 0; o < LQ8_K_OUT; ++o) {
        uint n = n_base + o;
        valid[o]   = (n < dims.N);
        w_rows4[o] = (const device uint4*)(W + (uint)n * Kpack);
        s_rows[o]  = S  + (uint)n * Kgroups;
        b_rows[o]  = Bq + (uint)n * Kgroups;
    }

    float acc[LQ8_K_OUT];
    for (uint o = 0; o < LQ8_K_OUT; ++o) acc[o] = 0.0f;

    // KEY LOOP: lane processes K/16 uint4 packs per row. Each uint4
    // covers exactly one quant group → ONE (s, b) device read per
    // uint4 per output, regardless of K_OUT or unroll factor.
    for (uint u4 = lane; u4 < K_u4; u4 += LQ8_SIMD_WIDTH) {
        uint k4_base = u4 << 2;
        // 4 bfloat4 X loads, reused across all K_OUT outputs (B3).
        float4 x0 = float4(xrow4[k4_base + 0]);
        float4 x1 = float4(xrow4[k4_base + 1]);
        float4 x2 = float4(xrow4[k4_base + 2]);
        float4 x3 = float4(xrow4[k4_base + 3]);
        // Quant group: all 16 weights in this uint4 fall in group u4/4
        // (since k4_base*4 .. k4_base*4+15 = 16 consecutive K positions
        // and group_size = 64 = 4 × 16).
        uint g = u4 >> 2;

        for (uint o = 0; o < LQ8_K_OUT; ++o) {
            if (!valid[o]) continue;
            uint4  wv = w_rows4[o][u4];          // ONE uint4 (16 q8 weights)
            float  s  = float(s_rows[o][g]);     // ONE scale  load per uint4
            float  b  = float(b_rows[o][g]);     // ONE bias   load per uint4
            // 4× dot(x, dequant(word, s, b)) — fully unrolled.
            acc[o] += dot(x0, lq8_deq_word(wv.x, s, b));
            acc[o] += dot(x1, lq8_deq_word(wv.y, s, b));
            acc[o] += dot(x2, lq8_deq_word(wv.z, s, b));
            acc[o] += dot(x3, lq8_deq_word(wv.w, s, b));
        }
    }

    // Reduce + write.
    for (uint o = 0; o < LQ8_K_OUT; ++o) {
        float r = simd_sum(acc[o]);
        if (lane == 0u) {
            uint n = n_base + o;
            if (n < dims.N) Y[m * dims.N + n] = bfloat(float(B[n]) + r);
        }
    }
}

// ---------------------------------------------------------------------------
// EXTENSIONS (also touched in the same commit):
//
// 1. `linear_q8_bf16_add`: read residual at the write step.
//      Y[m*N + n] = bfloat(float(B[n]) + r + float(R[m*N + n]));
//    Catalog B5; same B6 inner loop.
//
// 2. `linear_q8_gather_bf16`: per-expert/per-token gather. Add a
//    routing index (expert_id, src_token) and offset W/S/B per
//    expert. Same B6 inner loop, just with W_e = W + expert * stride.
//
// ---------------------------------------------------------------------------
// GENERALIZATION TO OTHER QUANT FORMATS:
//
//   q4 (4-bit, group_size 32):
//     - pack_unit = 8 weights per uint32 (4-bit × 8 = 32 bits)
//     - load W as uint  -> 8 weights at once, one (s, b) per uint
//       Or load W as uint4 -> 32 weights, one (s, b) per uint4
//       (group_size 32 → 32 weights = one group, same trick).
//
//   mxfp4 (4-bit + shared 8-bit exponent, group_size 32):
//     - pack_unit = 16 weights per uint32 (2 × 4-bit packed nybbles
//       per byte... convention varies — check your dequant header).
//     - One uint4 (128 bits) = 32 weights = one group → one shared
//       exponent load per uint4, same amortization win.
//
// The principle is invariant: load packed W in chunks of EXACTLY one
// quant group's worth, then the per-group state (scale / bias /
// exponent) is one device read per chunk, regardless of pack format.
