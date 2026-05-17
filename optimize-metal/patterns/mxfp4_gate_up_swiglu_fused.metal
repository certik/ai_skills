// patterns/mxfp4_gate_up_swiglu_fused.metal
//
// WHAT: ONE kernel does BOTH the gate and up projections from MXFP4
//       expert weights AND applies the SwiGLU activation, writing only
//       the fused activation `mid`. The unfused path materialises two
//       separate buffers (`gate`, `up`) and runs a third SwiGLU
//       dispatch over them.
//
// WHEN: Decode path for MoE models with quantized expert weights
//       (gpt-oss MXFP4, similar quantization schemes). Apply after
//       the basic mxfp4 qmv4 decode kernel (E1) so you're amortising
//       the same expert-gather, not adding a new code path.
//
// SPEEDUP: 1.05–1.10× decode. Saves 2 dispatches × N_LAYERS per
//          token AND a full pass over `gate` and `up` buffers (which
//          for K_top * N_intermediate elements are non-trivial).
//
// COMMIT: b15e957 — "Fuse mxfp4 gate+up+swiglu into one kernel"
//         (gpt-oss reference impl).
//
// PATTERN: per SG: cache X in registers ONCE, dot-product against the
//          gate and the up weight rows in parallel, then apply the
//          SwiGLU at the end. Bias is baked into the dot product;
//          scaling is per-32-element block.
//
//          out = (g * sigmoid(alpha * g)) * (u + 1)   with g,u clamped to ±limit
//
// SHAPES: One expert per (kt, l). Output `mid[l, kt, n]`. For gpt-oss:
//         K_blocks=90, N=2880, K_top=4. Full row count is 2*N (gate +
//         up interleaved). Dispatch: grid = (N*8, K_top, L), TG = (64, 1, 1).
//
// CONTRAST WITH E1: the basic E1 kernel (`mxfp4_qmv4_decode.metal`)
// runs ONCE for gate, again for up, then a third dispatch for SwiGLU.
// This fused kernel keeps the X register-tile (qmv4) hot for BOTH gate
// and up rows of the same expert, since the dequant cost dominates.

#include <metal_stdlib>
using namespace metal;

constant float k_fp4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float e8m0_scale(uchar b) {
    return (b == 0xFF) ? 0.0f : exp2(float(int(b) - 127));
}

// 64 threads/TG (2 SGs × 32 lanes), 4 outputs/SG = 8 outputs per TG.
// Each lane caches its own 8 X values in REGISTERS (not threadgroup
// memory) and dot-products them against 4 gate-rows AND 4 up-rows per
// SG, amortizing the X reads across 8 effective weight rows.
//
// Layout: weight rows interleave gate and up along the row axis
//   row 2*nj    = gate row nj
//   row 2*nj+1  = up   row nj
//
// Dispatch: grid = (N*8, K_top, L), TG = (64, 1, 1). N must be %8==0.

kernel void mxfp4_gus_qmv4_bf16(
    device const bfloat*  x        [[buffer(0)]],   // [L, K]
    device const uchar*   blocks   [[buffer(1)]],   // [E, 2*N, K_blocks] of fp4-byte pairs
    device const uchar*   scales   [[buffer(2)]],   // [E, 2*N, K_blocks] of e8m0 scales
    device const bfloat*  bias     [[buffer(3)]],   // [E, 2*N]
    device const int*     indices  [[buffer(4)]],   // [L, K_top]
    device bfloat*        mid      [[buffer(5)]],   // [L, K_top, N]
    constant uint4&       dims     [[buffer(6)]],   // (-, K_top, N, K_blocks)
    constant float2&      ab       [[buffer(7)]],   // (alpha, clamp_limit)
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sgid                     [[simdgroup_index_in_threadgroup]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]])
{
    const uint K_top    = dims.y;
    const uint N        = dims.z;
    const uint K_blocks = dims.w;
    const uint kt  = tg_pos.y;
    const uint l   = tg_pos.z;
    const uint K   = K_blocks * 32;
    const uint N_total_rows = 2 * N;

    const uint n_base = tg_pos.x * 8 + sgid * 4;

    int e = indices[l * K_top + kt];

    // Base offsets for the 4 gate rows and the 4 up rows handled by THIS SG.
    uint64_t base_g[4], base_u[4];
    for (int j = 0; j < 4; j++) {
        uint nj = n_base + j;
        base_g[j] = ((uint64_t)e * N_total_rows + 2u*nj    ) * K_blocks;
        base_u[j] = ((uint64_t)e * N_total_rows + 2u*nj + 1) * K_blocks;
    }

    float acc_g[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_u[4] = {0.f, 0.f, 0.f, 0.f};

    const uint sub  = lane & 3u;        // 0..3 (byte offset within a 4-byte fp4 chunk)
    const uint subg = lane >> 2;        // 0..7 (group within an outer step)
    const uint full_steps  = K_blocks / 8;
    const uint tail_groups = K_blocks - full_steps * 8;

    device const bfloat* xrow = x + (uint64_t)l * K;

    for (uint s = 0; s < full_steps; s++) {
        // Each lane reads 8 contiguous X values from registers ONCE per outer step.
        device const bfloat4* xb = (device const bfloat4*)(xrow + s * 256u + lane * 8u);
        float4 X0 = float4(xb[0]);
        float4 X1 = float4(xb[1]);
        uint g_blk = s * 8u + subg;

        for (int j = 0; j < 4; j++) {
            // 1 uint32 of W bytes covers 8 fp4 weight values.
            uint Wg = *(device const uint*)(blocks + (base_g[j] + g_blk) * 16u + sub * 4u);
            uint Wu = *(device const uint*)(blocks + (base_u[j] + g_blk) * 16u + sub * 4u);
            float sg = e8m0_scale(scales[base_g[j] + g_blk]);
            float su = e8m0_scale(scales[base_u[j] + g_blk]);

            uchar4 wg = as_type<uchar4>(Wg);
            uchar4 wu = as_type<uchar4>(Wu);
            #define ACC(byte, scale, accv, lo_x, hi_x) {                                 \
                accv = fma(k_fp4_lut[(byte)&0xF]*(scale),    lo_x, accv);               \
                accv = fma(k_fp4_lut[((byte)>>4)&0xF]*(scale), hi_x, accv); }
            ACC(wg.x, sg, acc_g[j], X0.x, X0.y);
            ACC(wg.y, sg, acc_g[j], X0.z, X0.w);
            ACC(wg.z, sg, acc_g[j], X1.x, X1.y);
            ACC(wg.w, sg, acc_g[j], X1.z, X1.w);
            ACC(wu.x, su, acc_u[j], X0.x, X0.y);
            ACC(wu.y, su, acc_u[j], X0.z, X0.w);
            ACC(wu.z, su, acc_u[j], X1.x, X1.y);
            ACC(wu.w, su, acc_u[j], X1.z, X1.w);
            #undef ACC
        }
    }
    // Tail (partial outer step) — only lanes with subg < tail_groups are active.
    if (subg < tail_groups) {
        uint s = full_steps;
        device const bfloat4* xb = (device const bfloat4*)(xrow + s * 256u + lane * 8u);
        float4 X0 = float4(xb[0]);
        float4 X1 = float4(xb[1]);
        uint g_blk = s * 8u + subg;
        for (int j = 0; j < 4; j++) {
            uint Wg = *(device const uint*)(blocks + (base_g[j] + g_blk) * 16u + sub * 4u);
            uint Wu = *(device const uint*)(blocks + (base_u[j] + g_blk) * 16u + sub * 4u);
            float sg = e8m0_scale(scales[base_g[j] + g_blk]);
            float su = e8m0_scale(scales[base_u[j] + g_blk]);
            uchar4 wg = as_type<uchar4>(Wg);
            uchar4 wu = as_type<uchar4>(Wu);
            #define ACC(byte, scale, accv, lo_x, hi_x) {                                 \
                accv = fma(k_fp4_lut[(byte)&0xF]*(scale),    lo_x, accv);               \
                accv = fma(k_fp4_lut[((byte)>>4)&0xF]*(scale), hi_x, accv); }
            ACC(wg.x, sg, acc_g[j], X0.x, X0.y);
            ACC(wg.y, sg, acc_g[j], X0.z, X0.w);
            ACC(wg.z, sg, acc_g[j], X1.x, X1.y);
            ACC(wg.w, sg, acc_g[j], X1.z, X1.w);
            ACC(wu.x, su, acc_u[j], X0.x, X0.y);
            ACC(wu.y, su, acc_u[j], X0.z, X0.w);
            ACC(wu.z, su, acc_u[j], X1.x, X1.y);
            ACC(wu.w, su, acc_u[j], X1.z, X1.w);
            #undef ACC
        }
    }

    // SwiGLU epilogue (gpt-oss form: gate is clamped on the positive side,
    // up is clamped on both sides; activation = (g * sigmoid(alpha*g)) * (u+1)).
    float alpha = ab.x;
    float limit = ab.y;
    for (int j = 0; j < 4; j++) {
        float g = simd_sum(acc_g[j]);
        float u = simd_sum(acc_u[j]);
        if (lane == 0) {
            uint nj = n_base + j;
            g += (float)bias[(uint64_t)e * N_total_rows + 2u*nj    ];
            u += (float)bias[(uint64_t)e * N_total_rows + 2u*nj + 1];
            if (g > limit) g = limit;
            if (u > limit) u = limit; else if (u < -limit) u = -limit;
            float sig = 1.0f / (1.0f + exp(-alpha * g));
            float out = (g * sig) * (u + 1.0f);
            mid[((uint64_t)l * K_top + kt) * N + nj] = (bfloat)out;
        }
    }
}
