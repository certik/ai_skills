// patterns/moe_sorted_gather_glue.metal
//
// WHAT: The "sorted-gather" MoE fast path for prefill (Lq > 1).  Instead
//       of each token gathering its K_top experts independently, we bucket
//       all (token, kt) pairs by expert, then run ONE dense quantized
//       GEMM per expert covering its bucket.
//
// PIPELINE (4–5 small kernels + 2 quantized GEMMs):
//   1. zero_uints              clear per-expert counters
//   2. expert_bucketize        scatter (l, kt) into per-expert lists
//   3. moe_flatten_buckets     prefix-sum + sorted flat lists + reverse map
//   4. moe_gather_x_sorted     gather X rows into expert-sorted layout
//   5. qmm_t_gather_rhs_*      ONE quantized GEMM per layer (all experts)
//      → gate_up
//   6. moe_swiglu_epilogue     clamp + sigmoid + (u+1) SwiGLU
//   7. qmm_t_gather_rhs_*      → down
//   8. moe_combine_scatter     weighted sum + residual + scatter back
//
// WHEN: Prefill MoE with Lq * K_top > N_experts (almost always when
//       Lq > 4 in a 32-expert / k_top=4 model).  This single restructuring
//       was the biggest prefill MoE win in the gpt-oss reference impl (+24–32%).
// SPEEDUP: 1.30–1.60× prefill.
// COMMITS:
//   1c4ddfd — MoE: grouped expert GEMM for prefill (264→348 tok/s, +32%)
//   37ad895 — MoE grouped: M_GROUP=4→8 (348→363 tok/s prefill)
//   51ced6b — sorted-gather MMA MoE for prefill (+24%)
//   3fce46d — qmm_t_gather_rhs MMA kernel + multi-expert correctness
//   7691d2f — gather-sort MoE for prefill (387 -> 608 tok/s in mlx-cpp)

#include <metal_stdlib>
using namespace metal;

// --- Kernel 1: zero per-expert counters -----------------------------------
kernel void zero_uints(device uint* counts [[buffer(0)]],
                       constant uint& N   [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < N) counts[i] = 0;
}

// --- Kernel 2: scatter (l, kt) into per-expert bucket lists ---------------
// indices[l, kt] = expert id e for row (l, kt).  We want to produce a list
// per expert: which rows belong to it.
kernel void expert_bucketize(device const int*  indices    [[buffer(0)]], // [L, K_top]
                             device uint*       counts     [[buffer(1)]], // [E]
                             device uint*       lists      [[buffer(2)]], // [E, MAX_PER_E]
                             constant uint3&    dims       [[buffer(3)]], // (L, K_top, MAX_PER_E)
                             uint tid [[thread_position_in_grid]])
{
    uint L = dims.x, K_top = dims.y, MAX_PER_E = dims.z;
    uint total = L * K_top;
    if (tid >= total) return;
    uint l  = tid / K_top;
    uint kt = tid - l * K_top;
    uint e  = uint(indices[tid]);
    uint pos = atomic_fetch_add_explicit((device atomic_uint*)&counts[e], 1u, memory_order_relaxed);
    // Pack (l, kt) into a single u32 (l: high bits, kt: low 8 bits).
    lists[e * MAX_PER_E + pos] = (l << 8) | kt;
}

// --- Kernel 3: prefix-sum + flat sorted list + reverse map ----------------
//   prefix[e] = sum_{i<e} counts[i]
//   flat[prefix[e] + k] = lists[e, k]   for k in 0..counts[e]-1
//   rev[ (l, kt) ] = position in flat   (used by combine_scatter)
// Implemented as a single small kernel for simplicity.  Use threadgroup
// memory for the prefix-sum if you have many experts; for E ≤ 32 a single
// SG suffices.
kernel void moe_flatten_buckets(device const uint* counts [[buffer(0)]],
                                device const uint* lists  [[buffer(1)]],
                                device uint*       flat   [[buffer(2)]],
                                device uint*       prefix [[buffer(3)]],
                                device uint*       rev    [[buffer(4)]],
                                constant uint2&    dims   [[buffer(5)]], // (E, MAX_PER_E)
                                uint lane [[thread_index_in_simdgroup]])
{
    // Tiny example for E <= 32: single SG does the whole prefix sum.
    uint E = dims.x, MAX_PER_E = dims.y;
    uint c = (lane < E) ? counts[lane] : 0;
    uint p = simd_prefix_exclusive_sum(c);
    if (lane < E) prefix[lane] = p;

    // Each expert writes its rows into flat[p .. p + c).
    // (In practice this is parallelized over a 2D grid (E, MAX_PER_E).)
    if (lane < E) {
        uint cnt = counts[lane];
        uint base = p;
        for (uint k = 0; k < cnt; ++k) {
            uint packed = lists[lane * MAX_PER_E + k];
            flat[base + k] = packed;
            // Reverse map: from (l, kt) back to position in flat.
            uint l  = packed >> 8;
            uint kt = packed & 0xFF;
            rev[l * /*K_top*/4 + kt] = base + k;
        }
    }
}

// --- Kernel 4: gather X rows in expert-sorted layout ----------------------
// Xs[r, :] = X[ flat[r] >> 8 , : ]    for r in 0..total
kernel void moe_gather_x_sorted(device const bfloat* X    [[buffer(0)]],
                                device const uint*   flat [[buffer(1)]],
                                device bfloat*       Xs   [[buffer(2)]],
                                constant uint2&      dims [[buffer(3)]], // (total, D)
                                uint2 gid [[thread_position_in_grid]])
{
    uint total = dims.x, D = dims.y;
    uint r = gid.y;
    uint d = gid.x;
    if (r >= total || d >= D) return;
    uint l = flat[r] >> 8;
    Xs[r * D + d] = X[l * D + d];
}

// --- Kernel 7: SwiGLU epilogue on gate_up output --------------------------
// gate_up: [total, 2*INTER], even cols = gate, odd cols = up (or split).
// mid[r, n] = (clamp(gate, _, limit) * sigmoid(alpha*gate)) * (clamp(up, -limit, limit) + 1)
kernel void moe_swiglu_epilogue(device const bfloat* gate_up [[buffer(0)]], // [total, 2*INTER]
                                device bfloat*       mid     [[buffer(1)]], // [total, INTER]
                                constant uint2&      dims    [[buffer(2)]], // (total, INTER)
                                constant float2&     ab      [[buffer(3)]], // (alpha, limit)
                                uint2 gid [[thread_position_in_grid]])
{
    uint total = dims.x, N = dims.y;
    uint r = gid.y, n = gid.x;
    if (r >= total || n >= N) return;
    float g = float(gate_up[r * 2 * N + 2*n]);
    float u = float(gate_up[r * 2 * N + 2*n + 1]);
    float alpha = ab.x, limit = ab.y;
    if (g > limit) g = limit;
    u = clamp(u, -limit, limit);
    float sig = 1.0f / (1.0f + exp(-alpha * g));
    mid[r * N + n] = bfloat((g * sig) * (u + 1.0f));
}

// --- Kernel 8: combine + scatter back + fused residual --------------------
// out[l, h] = residual[l, h] + sum_kt topk_w[l, kt] * down_out[rev[l, kt], h]
kernel void moe_combine_scatter(device const bfloat* down_out [[buffer(0)]], // [total, H]
                                device const float*  topk_w   [[buffer(1)]], // [L, K_top]
                                device const uint*   rev      [[buffer(2)]], // [L, K_top]
                                device bfloat*       residual [[buffer(3)]], // INPUT + OUTPUT
                                constant uint3&      dims     [[buffer(4)]], // (L, K_top, H)
                                uint2 gid [[thread_position_in_grid]])
{
    uint L = dims.x, K_top = dims.y, H = dims.z;
    uint l = gid.y, h = gid.x;
    if (l >= L || h >= H) return;
    float acc = float(residual[l * H + h]);
    for (uint kt = 0; kt < K_top; ++kt) {
        uint r = rev[l * K_top + kt];
        float w = topk_w[l * K_top + kt];
        acc += w * float(down_out[r * H + h]);
    }
    residual[l * H + h] = bfloat(acc);
}

// --- The quantized GEMM step (Kernel 5/7) ---------------------------------
// `qmm_t_gather_rhs_bf16_g32_b4` — one dense MXFP4 GEMM per layer, with
// per-row expert indirection (each row r selects expert e = flat[r]>>somewhere
// or use prefix[]/counts[] to know which contiguous range belongs to each
// expert).  See src-metal/kernels/phase6_mma.metal in the gpt-oss repo for the
// real implementation (uses MLX `steel` MMA tile primitives).
