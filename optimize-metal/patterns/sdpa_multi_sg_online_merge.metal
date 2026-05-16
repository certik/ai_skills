// patterns/sdpa_multi_sg_online_merge.metal
//
// WHAT: SDPA where each (lq, hq) is processed by N_SG SIMD groups in
//       parallel, each handling a contiguous Lk stripe with single-pass
//       ONLINE-SOFTMAX accumulation, then merged across SGs using the
//       online-softmax merge formula (no two-pass softmax needed).
// WHEN: After D1 (SG-per-(lq,hq)) is in. The single-SG variant caps at
//       D2 (online softmax). To push further when Lk grows long,
//       parallelize the Lk dimension across multiple SGs in the same TG.
// SPEEDUP: 1.2-1.5x decode SDPA on long contexts. On Qwen3.6-35B-A3B
//          at n=210, this lifted decode 52 -> 65 tok/s (+25%).
//
// Dispatch: grid = (32 * N_SG * Nq * Lq, 1, 1), tg = (32 * N_SG, 1, 1).
// CRITICAL: dispatchThreads: takes TOTAL THREAD COUNT, not TG count!
// Forgetting the 32*N_SG multiplier silently launches fewer SGs that
// each compute multiple (lq,hq) pairs, producing wrong output.
//
// Online-softmax merge formula:
//   m_g    = max(m_i)
//   alpha_i = exp(m_i - m_g)
//   s_g    = sum_i(alpha_i * s_i)
//   o[d]   = sum_i(alpha_i * o_i[d]) / s_g
//
// Per-SG state: m_run (rolling max), s_run (rolling normalizer),
// o_acc[D/SIMD_WIDTH] (per-lane partial output stripe).
//
// Pitfalls:
//   * per_lane_q (D/4/SIMD for bfloat4 Q/K dot) and per_lane_o
//     (D/SIMD for scalar V/O) are DIFFERENT counts. Conflating them
//     drops 75% of output dims silently.
//   * TG-mem sg_o[N_SG][D] can be sized to D=1024 max; use uint16_t
//     or split if D > 1024.
//   * sg_id 0 does the merge; other SGs sit idle. Acceptable since
//     merge is O(D) work and SGs are otherwise unused after the stripe
//     loop.
//   * If you forget the dispatchThreads grid-size factor, the kernel
//     "works" but mis-assigns work — first 16 tokens may match by
//     luck, then diverge. Always verify with a longer run (>= 64
//     tokens).

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SDPA_SIMD_WIDTH = 32u;
constant constexpr uint SDPA_N_SG       = 4u;
constant constexpr uint SDPA_MAX_DVEC   = 8u;   // D / 4 / SIMD <= 8 for D<=1024

struct sdpa_dims {
    uint Lq;
    uint Lk;
    uint Nq;
    uint Nkv;
    uint D;
    uint q_offset;
    float scale;
    uint causal;
};

kernel void sdpa_gqa_bf16(device const bfloat*     Q     [[buffer(0)]],
                          device const bfloat*     K     [[buffer(1)]],
                          device const bfloat*     V     [[buffer(2)]],
                          device bfloat*           OUT   [[buffer(3)]],
                          constant sdpa_dims&      dims  [[buffer(4)]],
                          uint  tg_id  [[threadgroup_position_in_grid]],
                          uint  tid    [[thread_position_in_threadgroup]],
                          uint  lane   [[thread_index_in_simdgroup]],
                          uint  sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint hq = tg_id % dims.Nq;
    uint lq = tg_id / dims.Nq;
    if (lq >= dims.Lq || hq >= dims.Nq) return;

    uint D     = dims.D;
    uint group = dims.Nq / dims.Nkv;
    uint hkv   = hq / group;
    uint q_abs = dims.q_offset + lq;

    const device bfloat4* q4   = (const device bfloat4*)(Q + (lq * dims.Nq + hq) * D);
    device bfloat*        o_row = OUT + (lq * dims.Nq + hq) * D;

    uint Dvec = D >> 2;
    uint per_lane_q = (Dvec + SDPA_SIMD_WIDTH - 1u) / SDPA_SIMD_WIDTH;
    uint per_lane_o = (D    + SDPA_SIMD_WIDTH - 1u) / SDPA_SIMD_WIDTH;
    float4 qv[SDPA_MAX_DVEC];
    for (uint i = 0; i < SDPA_MAX_DVEC; ++i) {
        uint idx = lane + i * SDPA_SIMD_WIDTH;
        qv[i] = (i < per_lane_q && idx < Dvec) ? float4(q4[idx]) : float4(0.0f);
    }

    // Per-lane output: each lane owns D/SIMD output dims (= 8 for D=256).
    float o_acc[32];
    for (uint i = 0; i < 32; ++i) o_acc[i] = 0.0f;
    float m_run = -INFINITY;
    float s_run = 0.0f;

    // SG's Lk stripe: contiguous blocks for cache locality.
    uint stripe = (dims.Lk + SDPA_N_SG - 1u) / SDPA_N_SG;
    uint lk_beg = sg_id * stripe;
    uint lk_end = min(lk_beg + stripe, dims.Lk);

    for (uint lk = lk_beg; lk < lk_end; ++lk) {
        float score;
        if (dims.causal != 0u && lk > q_abs) {
            score = -INFINITY;
        } else {
            const device bfloat4* k4 = (const device bfloat4*)(K + (lk * dims.Nkv + hkv) * D);
            float partial = 0.0f;
            for (uint i = 0; i < per_lane_q; ++i) {
                uint idx = lane + i * SDPA_SIMD_WIDTH;
                if (idx < Dvec) {
                    partial += dot(qv[i], float4(k4[idx]));
                }
            }
            score = simd_sum(partial) * dims.scale;
        }

        if (score == -INFINITY) continue;

        float new_m = max(m_run, score);
        float alpha = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_m);
        float w     = exp(score - new_m);

        for (uint i = 0; i < per_lane_o; ++i) {
            uint d_idx = lane + i * SDPA_SIMD_WIDTH;
            float vd = (d_idx < D) ? float(V[(lk * dims.Nkv + hkv) * D + d_idx]) : 0.0f;
            o_acc[i] = o_acc[i] * alpha + w * vd;
        }
        s_run = s_run * alpha + w;
        m_run = new_m;
    }

    // Merge across SGs.
    threadgroup float sg_m[SDPA_N_SG];
    threadgroup float sg_s[SDPA_N_SG];
    threadgroup float sg_o[SDPA_N_SG][1024];  // D <= 1024

    if (lane == 0u) { sg_m[sg_id] = m_run; sg_s[sg_id] = s_run; }
    for (uint i = 0; i < per_lane_o; ++i) {
        uint d_idx = lane + i * SDPA_SIMD_WIDTH;
        if (d_idx < D) sg_o[sg_id][d_idx] = o_acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float m_g = -INFINITY;
        for (uint k = 0; k < SDPA_N_SG; ++k) m_g = max(m_g, sg_m[k]);
        float alpha[SDPA_N_SG];
        float s_g = 0.0f;
        for (uint k = 0; k < SDPA_N_SG; ++k) {
            alpha[k] = (sg_m[k] == -INFINITY) ? 0.0f : exp(sg_m[k] - m_g);
            s_g     += alpha[k] * sg_s[k];
        }
        float inv = 1.0f / s_g;
        for (uint d = lane; d < D; d += SDPA_SIMD_WIDTH) {
            float v = 0.0f;
            for (uint k = 0; k < SDPA_N_SG; ++k) v += alpha[k] * sg_o[k][d];
            o_row[d] = bfloat(v * inv);
        }
    }
}
