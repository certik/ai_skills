// patterns/argmax_parallel.metal
//
// WHAT: Parallel argmax over a large vocabulary (e.g., VOCAB=200k).
//       Naive single-threaded argmax is the bottleneck of decode at
//       40+ tok/s.  This pattern uses:
//         (1) per-thread strided pass to find local (max, idx)
//         (2) simdgroup reduction via simd_max + indirection
//         (3) threadgroup-level reduction across SGs via threadgroup mem
//       Two variants in this file:
//         A. SINGLE-TG  — 1 TG of 256 threads.  Good for V <= ~64k.
//         B. 2-STAGE    — N_TG TGs of 256 each + 1 reducer TG of 32.
//                         Required when V >> per-TG saturation (e.g.
//                         V=248k Qwen3.6); single-TG uses only 1 GPU
//                         core (~253 µs/call), 2-stage saturates and
//                         drops to ~10 µs/call.
// WHEN:
//   - Single-TG: apply EARLY in optimization.  This single change
//                took the gpt-oss reference impl from 40 → 84 tok/s
//                decode (+110%).
//   - 2-stage:   apply LATE, once decode is mid-pipeline and KPROF
//                still shows argmax >100 µs/call with single-TG at
//                V > ~100k.  Qwen3.6 second-pass: 98.4 → 101.5 tok/s
//                (+3.2%).  Commit 6c49e38.
// COMMIT (single-TG): e34bf80 — "Parallel argmax (single-thread bottleneck): 40→84 tok/s"
// COMMIT (2-stage):  6c49e38 — "parallel argmax (2-stage) — decode +3.2% (98.4→101.5 tok/s)"

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

// ---------------------------------------------------------------------------
// VARIANT B — 2-stage parallel argmax for HUGE V (>~100k).
//
// Single-TG argmax over V=248320 on M4 Max measured 253 µs/call — the
// kernel uses 1 GPU core because there's only one TG.  Splitting into
// N_TG TGs (each scanning V/N_TG elements) saturates all cores; a
// second 1-TG reducer kernel consumes the N_TG partials.
//
// Pick N_TG ≈ number of GPU cores (e.g. 64 on M4 Max).  Stage-1
// per-TG work is ~V/N_TG elements; final stage is N_TG entries
// (trivial for one SG of 32 lanes when N_TG <= 32 * inner-loop-iters,
// which holds for all reasonable N_TG).
//
// Pipeline hazard: stage 2 reads stage 1's partial buffers.  Either
// keep both dispatches in the SAME cmdbuf (with the encoder's implicit
// barrier — they have RAW deps via the partial buffer) or put them in
// successive cmdbufs (Metal queue order guarantees RAW for free, see
// gotchas #13).  Do NOT issue them concurrently.
//
// Partial buffers are tiny (N_TG × 8 bytes); allocate once at startup
// and reuse.

constant constexpr uint ARGMAX_P_TG_SIZE = 256u;
constant constexpr uint ARGMAX_P_N_SG    = ARGMAX_P_TG_SIZE / 32u;

struct argmax_p_dims { uint N; uint N_TG; };

// Stage 1: N_TG threadgroups, each writes one (max_val, max_idx).
// Launch:  grid = (ARGMAX_P_TG_SIZE * N_TG, 1, 1), tg = (256, 1, 1).
kernel void argmax_stage1_bf16(device const bfloat*    logits   [[buffer(0)]],
                               device float*           part_max [[buffer(1)]],
                               device int*             part_idx [[buffer(2)]],
                               constant argmax_p_dims& dims     [[buffer(3)]],
                               uint  tg_id [[threadgroup_position_in_grid]],
                               uint  tid   [[thread_index_in_threadgroup]],
                               uint  lane  [[thread_index_in_simdgroup]],
                               uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    uint tile = (dims.N + dims.N_TG - 1u) / dims.N_TG;
    uint beg  = tg_id * tile;
    uint end  = min(beg + tile, dims.N);

    float best_v = -INFINITY;
    int   best_i = -1;
    for (uint i = beg + tid; i < end; i += ARGMAX_P_TG_SIZE) {
        float v = float(logits[i]);
        if (v > best_v) { best_v = v; best_i = int(i); }
    }

    float sg_max = simd_max(best_v);
    uint  vote   = simd_ballot(best_v == sg_max);
    uint  wlane  = ctz(vote);
    int   sg_idx = simd_shuffle(best_i, wlane);

    threadgroup float tg_max[ARGMAX_P_N_SG];
    threadgroup int   tg_idx[ARGMAX_P_N_SG];
    if (lane == 0u) { tg_max[sgid] = sg_max; tg_idx[sgid] = sg_idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float v = (lane < ARGMAX_P_N_SG) ? tg_max[lane] : -INFINITY;
        int   i = (lane < ARGMAX_P_N_SG) ? tg_idx[lane] : -1;
        float fmax = simd_max(v);
        uint  w    = simd_ballot(v == fmax);
        uint  wl   = ctz(w);
        int   fi   = simd_shuffle(i, wl);
        if (lane == 0u) {
            part_max[tg_id] = fmax;
            part_idx[tg_id] = fi;
        }
    }
}

// Stage 2: ONE TG of one SG (32 lanes) reduces N_TG partials.
// Launch:  grid = (32, 1, 1), tg = (32, 1, 1).
kernel void argmax_stage2_bf16(device const float*     part_max [[buffer(0)]],
                               device const int*       part_idx [[buffer(1)]],
                               device int*             out_idx  [[buffer(2)]],
                               constant argmax_p_dims& dims     [[buffer(3)]],
                               uint  lane [[thread_index_in_simdgroup]])
{
    float best_v = -INFINITY;
    int   best_i = -1;
    for (uint i = lane; i < dims.N_TG; i += 32u) {
        float v = part_max[i];
        if (v > best_v) { best_v = v; best_i = part_idx[i]; }
    }
    float fmax = simd_max(best_v);
    uint  w    = simd_ballot(best_v == fmax);
    uint  wl   = ctz(w);
    int   fi   = simd_shuffle(best_i, wl);
    if (lane == 0u) out_idx[0] = fi;
}
