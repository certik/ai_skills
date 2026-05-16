// patterns/sdpa_online_softmax.metal
//
// WHAT: Scaled-dot-product attention with online (one-pass) softmax.
//       Instead of two passes (find max → exp/sum → normalize), we
//       maintain a rolling (max, sum, weighted_sum_of_V) and rescale
//       the accumulator whenever a new score exceeds the current max.
//       Combined with SIMD-reductions (simd_max / simd_sum) for the
//       per-tile max and sum.  GQA-aware (q-head h reads kv-head h/(Nq/Nkv)).
//       Optional attention sink (a constant logit added to the softmax
//       denominator, no value contribution).
// WHEN: SDPA is in top-3 hottest kernels.  Apply on top of multi-SG
//       and per-lane-score parallelism.
// SPEEDUP: 1.10–1.20× decode combined (D1+D2+D3 layered).
// COMMITS:
//   ff5ad86 — per-lane score computation + simd_max/sum softmax (+12% decode)
//   2d59eed — rolling online softmax, 32 simdgroups/head + transpose-merge
//   11b7562 — simdgroup-per-(head,token) parallelism (1.96x decode)
//   cd2ec6b — multi-simdgroup SDPA (4 SGs per head, +5% decode)
//
// This is a single-SG-per-(head,query_token) sketch.  Multi-SG variant
// (D3) splits the K-dimension across multiple SGs and merges via the
// online-softmax-merge formula:
//   new_max = max(maxA, maxB)
//   new_sum = sumA * exp(maxA - new_max) + sumB * exp(maxB - new_max)
//   new_O   = OA * exp(maxA - new_max) + OB * exp(maxB - new_max)

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_WIDTH = 32;

struct sdpa_params {
    uint Lq, Lk;
    uint Nq, Nkv, D;
    uint window;        // 0 = full attention
    uint q_offset;
    float scale;
};

// Grid:        (Lq, Nq, 1)            one SG per (query_token, head)
// Threadgroup: (SIMD_WIDTH, 1, 1)     32 threads
kernel void sdpa_with_sinks_bf16(device const bfloat*       Q     [[buffer(0)]],
                                 device const bfloat*       K     [[buffer(1)]],
                                 device const bfloat*       V     [[buffer(2)]],
                                 device const bfloat*       sinks [[buffer(3)]],  // [Nq]
                                 device bfloat*             OUT   [[buffer(4)]],
                                 constant sdpa_params&      P     [[buffer(5)]],
                                 uint2                      gid   [[thread_position_in_grid]],
                                 uint                       lane  [[thread_index_in_simdgroup]])
{
    uint lq = gid.x;
    uint hq = gid.y;
    if (lq >= P.Lq || hq >= P.Nq) return;

    uint group = P.Nq / P.Nkv;
    uint hkv   = hq / group;

    int lq_abs = int(lq + P.q_offset);
    int win_lo = (P.window > 0) ? max(0, lq_abs - int(P.window) + 1) : 0;
    int lk_hi  = lq_abs;     // causal

    // Rolling online-softmax state.
    float m = sinks ? float(sinks[hq]) : -INFINITY;
    float l = sinks ? 1.0f : 0.0f;     // sink contributes a constant 1 to denominator
    float o[256];                       // per-thread partial O; D ≤ 256
    for (uint d = 0; d < P.D; ++d) o[d] = 0.0f;

    const device bfloat* qrow = Q + ((size_t)lq * P.Nq + hq) * P.D;

    // Iterate over key positions.  Each lane handles a 1-lane subset of D
    // for the score dot-product (here: each thread does the FULL D dot;
    // SIMD-parallelism is over keys via lane).  For better performance,
    // split D across lanes (see real csrc kernel).
    for (int lk = win_lo; lk <= lk_hi; ++lk) {
        const device bfloat* krow = K + ((size_t)lk * P.Nkv + hkv) * P.D;
        const device bfloat* vrow = V + ((size_t)lk * P.Nkv + hkv) * P.D;

        // Per-lane partial score over D / SIMD_WIDTH stripes.
        float s = 0.0f;
        for (uint d = lane; d < P.D; d += SIMD_WIDTH) {
            s += float(qrow[d]) * float(krow[d]);
        }
        s = simd_sum(s) * P.scale;       // full dot in lane 0; broadcast via simd_sum

        // Online softmax update.
        if (s > m) {
            float rescale = exp(m - s);
            l = l * rescale + 1.0f;
            for (uint d = 0; d < P.D; ++d) o[d] *= rescale;
            m = s;
        } else {
            l += exp(s - m);
        }
        float w = exp(s - m);
        for (uint d = lane; d < P.D; d += SIMD_WIDTH) {
            o[d] += w * float(vrow[d]);
        }
    }

    // Normalize and write.
    float inv_l = 1.0f / l;
    device bfloat* orow = OUT + ((size_t)lq * P.Nq + hq) * P.D;
    for (uint d = lane; d < P.D; d += SIMD_WIDTH) {
        orow[d] = bfloat(o[d] * inv_l);
    }
}

// MULTI-SG VARIANT: split Lk into K_SG stripes, each SG handles one stripe,
// then merge across SGs via the online-softmax-merge formula above.  Use
// threadgroup memory to communicate (m, l, O) between SGs.  Yields 2–4×
// on top of the single-SG version when Lk is long enough.
