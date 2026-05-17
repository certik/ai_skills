// patterns/softmax_argmax_per_row.metal
//
// WHAT: Per-row (argmax_idx, softmax_confidence) over a wide vocab.
//       For each of M logits rows of shape [M, V] (bf16), produce one
//       (int idx, float conf) where conf = softmax(logits[m])[idx] =
//       1 / sum_v exp(logits[m, v] - max_logit). Output is just M × 8 bytes.
//
//       One threadgroup per row, 256 threads (8 SG) cooperate via
//       SG-wide reductions + one cross-SG TG-mem fixup. Two passes:
//         (1) Find global_max + argmax across V (simd_max + simd_min
//             tiebreak on lane index to match CPU's "first occurrence").
//         (2) Sum exp(logit - global_max) across V (simd_sum + cross-SG).
//
// WHEN: Diffusion-LLM samplers (Dream / LLaDA / fastdllm) where each
//       refine step needs the argmax AND confidence at every of BL
//       positions to decide which mask tokens to commit. The naive
//       host-side version (`gpu_buf_contents` + per-position softmax)
//       shows up as a large `host_post` invisible to gpu_busy.
//
//       NOT useful for autoregressive decode (one row at a time —
//       argmax_parallel.metal already covers that path with no need
//       for the softmax sum pass).
//
//       Track `host_post` (see references/profiling.md). If it's
//       more than ~5% of wall, this kernel is your win.
//
// SPEEDUP: 1.05–1.15× wall on diffusion samplers. In the Dream-7B
//          port: host_post 0.25 → 0.00 s, wall 2.90 → 2.63 s (closed
//          half the remaining gap to MLX in one commit).
//
// COMMIT:  df3d9d1 — "GPU per-row softmax+argmax+confidence
//          (host_post 0.25→0.00s, wall 2.90→2.63s)"
//
// HOST USAGE:
//   Define `struct argmax_out { int idx; float conf; }` on the host with
//   matching layout (`int32_t idx; float conf;` is 8 bytes).
//   Allocate one gpu_buf of M × 8 bytes. Dispatch with
//   grid=(TG_SIZE × M, 1, 1) and threadgroup=(TG_SIZE, 1, 1). After
//   commit_wait, read M × 8 bytes from gpu_buf_contents — the only
//   per-step host read.

#include <metal_stdlib>
using namespace metal;

struct argmax_out { int idx; float conf; };

constant constexpr uint TG_SIZE = 256u;
constant constexpr uint N_SG    = TG_SIZE / 32u;     // 8

kernel void softmax_argmax_rows(
    device const bfloat*  logits  [[buffer(0)]],
    device argmax_out*    out     [[buffer(1)]],
    constant uint&        V       [[buffer(2)]],
    uint                  row     [[threadgroup_position_in_grid]],
    uint                  lid     [[thread_position_in_threadgroup]],
    uint                  sgid    [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]])
{
    const device bfloat* row_ptr = logits + (size_t)row * (size_t)V;

    // Pass 1: thread-local max + argmax over its V/TG_SIZE strip.
    float local_max = -INFINITY;
    uint  local_idx = 0u;
    for (uint v = lid; v < V; v += TG_SIZE) {
        float lg = float(row_ptr[v]);
        if (lg > local_max) { local_max = lg; local_idx = v; }
    }

    // SG-wide max. Tiebreak by smallest lane index whose local_max == sg_max
    // (matches CPU keeping the first occurrence).
    float sg_max = simd_max(local_max);
    uint  sg_idx_candidate = (local_max == sg_max) ? local_idx : 0xffffffffu;
    sg_idx_candidate = simd_min(sg_idx_candidate);

    // Cross-SG reduction via TG mem.
    threadgroup float tg_max_per_sg[N_SG];
    threadgroup uint  tg_idx_per_sg[N_SG];
    if (lane == 0u) {
        tg_max_per_sg[sgid] = sg_max;
        tg_idx_per_sg[sgid] = sg_idx_candidate;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = tg_max_per_sg[0];
    uint  global_idx = tg_idx_per_sg[0];
    for (uint i = 1u; i < N_SG; ++i) {
        float m = tg_max_per_sg[i];
        if (m > global_max ||
            (m == global_max && tg_idx_per_sg[i] < global_idx)) {
            global_max = m;
            global_idx = tg_idx_per_sg[i];
        }
    }

    // Pass 2: thread-local sum of exp(logit - global_max).
    float local_sum = 0.0f;
    for (uint v = lid; v < V; v += TG_SIZE) {
        local_sum += exp(float(row_ptr[v]) - global_max);
    }
    float sg_sum = simd_sum(local_sum);

    threadgroup float tg_sum_per_sg[N_SG];
    if (lane == 0u) tg_sum_per_sg[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        float gsum = 0.0f;
        for (uint i = 0u; i < N_SG; ++i) gsum += tg_sum_per_sg[i];
        out[row].idx  = (int)global_idx;
        out[row].conf = 1.0f / gsum;
    }
}
