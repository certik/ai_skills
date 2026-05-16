// patterns/argmax_parallel.metal
//
// WHAT: Parallel argmax over a large vocabulary (e.g., VOCAB=200k).
//       Naive single-threaded argmax is the bottleneck of decode at
//       40+ tok/s.  This pattern uses:
//         (1) per-thread strided pass to find local (max, idx)
//         (2) simdgroup reduction via simd_max + indirection
//         (3) threadgroup-level reduction across SGs via threadgroup mem
// WHEN: Apply EARLY in optimization.  This single change took csrc from
//       40 → 84 tok/s decode (+110%).
// SPEEDUP: 1.5–2.0× decode.
// COMMIT: e34bf80 — "Parallel argmax (single-thread bottleneck): 40→84 tok/s"

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint TG_SIZE    = 256;       // threads per threadgroup
constant constexpr uint N_SG       = TG_SIZE / SIMD_WIDTH;

// Grid:        (TG_SIZE, 1, 1)        ONE threadgroup processes the whole vocab
// Threadgroup: (TG_SIZE, 1, 1)
kernel void argmax_bf16(device const bfloat* X       [[buffer(0)]],
                        device int*          out_idx [[buffer(1)]],
                        constant uint&       N       [[buffer(2)]],
                        uint                 tid     [[thread_index_in_threadgroup]],
                        uint                 lane    [[thread_index_in_simdgroup]],
                        uint                 sgid    [[simdgroup_index_in_threadgroup]])
{
    // Per-thread strided scan.
    float best_v = -INFINITY;
    int   best_i = -1;
    for (uint i = tid; i < N; i += TG_SIZE) {
        float v = float(X[i]);
        if (v > best_v) { best_v = v; best_i = int(i); }
    }

    // SIMD-group reduction: lane with the max value broadcasts its idx.
    float sg_max = simd_max(best_v);
    // The lane whose value equals sg_max is the candidate winner.  If
    // multiple lanes have the same max, simd_ballot picks the lowest.
    uint vote = simd_ballot(best_v == sg_max);
    uint winner_lane = ctz(vote);
    int  sg_best_i   = simd_shuffle(best_i, winner_lane);

    // Stage per-SG (max, idx) into threadgroup memory.
    threadgroup float tg_max[N_SG];
    threadgroup int   tg_idx[N_SG];
    if (lane == 0) {
        tg_max[sgid] = sg_max;
        tg_idx[sgid] = sg_best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SGs (done by SG 0).
    if (sgid == 0) {
        float v = (lane < N_SG) ? tg_max[lane] : -INFINITY;
        int   i = (lane < N_SG) ? tg_idx[lane] : -1;
        float fmax = simd_max(v);
        uint w = simd_ballot(v == fmax);
        uint wl = ctz(w);
        int  fi = simd_shuffle(i, wl);
        if (lane == 0) out_idx[0] = fi;
    }
}
