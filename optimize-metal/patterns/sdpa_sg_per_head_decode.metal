// patterns/sdpa_sg_per_head_decode.metal
//
// WHAT: First-step SDPA parallelization — one SIMD group per
//       (query_token, head). Replaces the naive 1-thread-per-(lq,hq)
//       port. This is the CORRECTNESS-PRESERVING precursor to the
//       multi-SG-per-head split (D3). Apply this BEFORE D3 — without
//       it you can't measure D3's contribution meaningfully.
// WHEN: Always, as the first SDPA optimization. In Qwen3.6 this took
//       decode from 15 -> 18 tok/s.
// SPEEDUP: 1.10-1.20x decode (more if SDPA was a hot kernel).
// SHAPES: D must be a multiple of 4 (uses bfloat4). Lk bounded by
//         per-SG TG-mem scores[MAX_CTX] allocation.
//
// Dispatch: grid = (32 * Nq, Lq, 1), tg = (32, 1, 1).
// Per SG:
//   1. Load Q chunks into per-lane registers (D/4/32 bfloat4 per lane).
//   2. For each lk, dot via float4 dot + simd_sum -> score; lane 0
//      stores into TG-mem scores[lk].
//   3. Softmax: per-lane stride scan + simd_max for max, then exp+sum
//      reduced via simd_sum. Probs are stashed BACK into scores[].
//   4. OV: each lane handles D/32 output dims, walking lk via TG-mem
//      scores broadcast.
//
// Pitfalls (each cost an iteration in practice):
//  * Allocate scores[MAX_CTX] for the MAXIMUM Lk ever seen, NOT the
//    typical Lk. main.c's KV cache max_ctx and this MAX_CTX must
//    agree. Overruns are silent and only show up at long prompts.
//  * GQA index: hkv = hq / (Nq/Nkv). Recompute, don't reuse.
//  * Insert threadgroup_barrier between step 2 (writing scores) and
//    step 3 (reading them). Also between step 3 (writing probs back)
//    and step 4 (OV reading probs).

#include <metal_stdlib>
using namespace metal;

#define MAX_CTX 1024u

constant constexpr uint SDPA_SIMD_WIDTH = 32u;

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
                          uint3 tg   [[threadgroup_position_in_grid]],
                          uint  lane [[thread_index_in_simdgroup]])
{
    uint hq = tg.x;
    uint lq = tg.y;
    if (lq >= dims.Lq || hq >= dims.Nq) return;

    uint D     = dims.D;
    uint group = dims.Nq / dims.Nkv;
    uint hkv   = hq / group;
    uint q_abs = dims.q_offset + lq;

    const device bfloat4* q4 = (const device bfloat4*)(Q + (lq * dims.Nq + hq) * D);
    device bfloat*        o_row = OUT + (lq * dims.Nq + hq) * D;

    uint Dvec     = D >> 2;
    uint per_lane = (Dvec + SDPA_SIMD_WIDTH - 1u) / SDPA_SIMD_WIDTH;
    float4 qv[8];
    for (uint i = 0; i < per_lane; ++i) {
        uint idx = lane + i * SDPA_SIMD_WIDTH;
        qv[i] = (idx < Dvec) ? float4(q4[idx]) : float4(0.0f);
    }

    threadgroup float scores[MAX_CTX];

    for (uint lk = 0; lk < dims.Lk; ++lk) {
        if (dims.causal != 0u && lk > q_abs) {
            if (lane == 0u) scores[lk] = -INFINITY;
            continue;
        }
        const device bfloat4* k4 = (const device bfloat4*)(K + (lk * dims.Nkv + hkv) * D);
        float partial = 0.0f;
        for (uint i = 0; i < per_lane; ++i) {
            uint idx = lane + i * SDPA_SIMD_WIDTH;
            if (idx < Dvec) partial += dot(qv[i], float4(k4[idx]));
        }
        float dot_val = simd_sum(partial) * dims.scale;
        if (lane == 0u) scores[lk] = dot_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // easily forgotten

    float local_max = -INFINITY;
    for (uint i = lane; i < dims.Lk; i += SDPA_SIMD_WIDTH) {
        float v = scores[i];
        if (!isinf(v) || v > 0.0f) if (v > local_max) local_max = v;
    }
    float max_s = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint i = lane; i < dims.Lk; i += SDPA_SIMD_WIDTH) {
        float v = scores[i];
        float e = (isinf(v) && v < 0.0f) ? 0.0f : exp(v - max_s);
        scores[i] = e;
        local_sum += e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // easily forgotten
    float inv_sum = 1.0f / simd_sum(local_sum);

    for (uint d = lane; d < D; d += SDPA_SIMD_WIDTH) {
        float acc = 0.0f;
        for (uint lk = 0; lk < dims.Lk; ++lk) {
            float w = scores[lk];
            if (w == 0.0f) continue;
            acc += w * float(V[(lk * dims.Nkv + hkv) * D + d]);
        }
        o_row[d] = bfloat(acc * inv_sum);
    }
}
