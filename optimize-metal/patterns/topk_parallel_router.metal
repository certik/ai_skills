// patterns/topk_parallel_router.metal
//
// WHAT: SIMD-group-per-row router top-K. For E=N_EXPERTS experts and
//       K=K_TOP top-K, ONE SIMD group does the whole job using only
//       registers + simd intrinsics (no TG memory, no barriers).
// WHEN: As soon as decode is in the 30-50 tok/s ballpark and softmax_topk
//       shows up as a fat slice in KPROF. The naive 1-thread-per-row
//       softmax_topk over 32-256 experts is a silent killer of decode
//       (think of it as the router equivalent of the parallel argmax win).
// SPEEDUP: 1.10-1.25x decode on Qwen3.6-35B-A3B (41.5 -> 52.0 tok/s),
//          larger if E > 32 and K > 4.
//
// Algorithm:
//   Phase 1 — softmax: each lane loads E/32 logits, simd_max for max,
//   exp + simd_sum for normalizer.
//   Phase 2 — top-K: K iterations of (simd_max, tie-break with
//   simd_min, broadcast winner). The winning lane sets its entry to
//   -INFINITY so it loses the next round.
//
// Tie-break trick: when two lanes hold the same max, simd_max alone
// can't say which. Use simd_min(cand) where cand = lane if my_max ==
// gmax else WIDTH; the lowest-indexed winning lane is selected.
//
// Pitfalls:
//   * simd_ballot returns simd_vote (a 64-bit wrapper), NOT uint.
//     If you try to extract winner via simd_ballot, you need to cast:
//       uint v = uint((simd_vote::vote_t)simd_ballot(cond));
//     The simd_min trick above avoids ballot entirely. Prefer it.
//   * If renorm is requested, renorm in float — bf16 sum can lose
//     precision when K=8 and values are small.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint TOPK_SIMD_WIDTH = 32u;
constant constexpr uint TOPK_PER_LANE   = 8u;   // E / SIMD_WIDTH (E=256)
constant constexpr uint TOPK_MAX_K      = 8u;

struct topk_dims { uint L; uint E; uint K; uint renorm; };

kernel void softmax_topk_bf16(device const bfloat*  logits  [[buffer(0)]],
                              device int*           out_idx [[buffer(1)]],
                              device float*         out_w   [[buffer(2)]],
                              constant topk_dims&   dims    [[buffer(3)]],
                              uint  tg   [[threadgroup_position_in_grid]],
                              uint  lane [[thread_index_in_simdgroup]])
{
    uint l = tg;
    if (l >= dims.L) return;
    const uint E = dims.E;
    const uint K = dims.K;

    const device bfloat* row = logits + l * E;

    float vals[TOPK_PER_LANE];
    uint  ids[TOPK_PER_LANE];

    // Phase 1: load + simd_max for the softmax max.
    float local_max = -INFINITY;
    uint per_lane = (E + TOPK_SIMD_WIDTH - 1u) / TOPK_SIMD_WIDTH;
    for (uint i = 0; i < TOPK_PER_LANE; ++i) {
        uint e = lane + i * TOPK_SIMD_WIDTH;
        if (i < per_lane && e < E) {
            vals[i] = float(row[e]);
            ids[i] = e;
        } else {
            vals[i] = -INFINITY;
            ids[i] = 0u;
        }
        if (vals[i] > local_max) local_max = vals[i];
    }
    float max_x = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint i = 0; i < TOPK_PER_LANE; ++i) {
        vals[i] = (vals[i] == -INFINITY) ? 0.0f : exp(vals[i] - max_x);
        local_sum += vals[i];
    }
    float sum_x = simd_sum(local_sum);
    float inv   = 1.0f / sum_x;
    for (uint i = 0; i < TOPK_PER_LANE; ++i) vals[i] *= inv;

    // Phase 2: parallel top-K. K iterations; each picks the global max
    // and the winning lane removes its entry. K * O(1) simd reductions.
    float out_vals[TOPK_MAX_K];
    int   out_ids[TOPK_MAX_K];

    for (uint k = 0; k < K; ++k) {
        float my_max = -INFINITY;
        uint  my_i   = 0;
        for (uint i = 0; i < TOPK_PER_LANE; ++i) {
            if (vals[i] > my_max) { my_max = vals[i]; my_i = i; }
        }
        float gmax = simd_max(my_max);
        // Tie-break: lowest lane index that ties wins.
        uint cand = (my_max == gmax) ? lane : TOPK_SIMD_WIDTH;
        uint winner = simd_min(cand);
        int e_idx = (lane == winner) ? (int)ids[my_i] : 0;
        e_idx = simd_broadcast(e_idx, winner);

        out_vals[k] = gmax;
        out_ids[k]  = e_idx;
        if (lane == winner) vals[my_i] = -INFINITY;
    }

    if (lane == 0u) {
        float r = 1.0f;
        if (dims.renorm != 0u) {
            float ssum = 0.0f;
            for (uint k = 0; k < K; ++k) ssum += out_vals[k];
            r = 1.0f / ssum;
        }
        device int*   idx = out_idx + l * K;
        device float* w   = out_w   + l * K;
        for (uint k = 0; k < K; ++k) {
            idx[k] = out_ids[k];
            w[k]   = out_vals[k] * r;
        }
    }
}
