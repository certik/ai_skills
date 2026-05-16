// patterns/recurrent_state_sg_per_row.metal
//
// WHAT: Parallelize a single-token recurrent state update where the
//       state is laid out as [Hv, Dv, Dk] floats by giving ONE SIMD
//       group to each (hv, dv) pair. Each lane in the SG handles
//       Dk/SIMD_WIDTH state elements (caching them in registers across
//       the read-multiply-add-write phases); reductions are simd_sum.
// WHEN: Naive port has 1 thread per hv that loops over Dv*Dk serially.
//       For Hv=32, Dv=Dk=128 in Qwen3.6 that's 32 threads * 32K serial
//       ops = grossly under-uses the GPU (only 1 SG of 40 cores busy).
// WHY:  Recurrent updates of the form
//           state *= g
//           kv_mem[dv] = sum_dk state[dv,dk] * k[dk]
//           delta[dv] = (v[dv] - kv_mem) * beta
//           state[dv,dk] += k[dk] * delta[dv]
//           y[dv] = sum_dk state[dv,dk] * q[dk]
//       have independent (hv, dv) work units. dv is the natural outer
//       parallelism axis; Dk is the inner reduction axis.
// SPEEDUP: 2–3× decode on models that use GatedDeltaNet / Mamba-style
//          linear-attention layers (in the Qwen3.6 port this took
//          decode from 5 → ~15 tok/s).
//
// Dispatch: grid = (32 * Dv, Hv, 1), tg = (32, 1, 1).
// Per token (Lq=1, decode), this launches Hv*Dv = 32*128 = 4096 SGs.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint GDS_SIMD_WIDTH = 32u;
constant constexpr uint GDS_DK_PER_LANE = 4u;     // Dk(128) / 32

struct gd_dims { uint Hk; uint Hv; uint Dk; uint Dv; };

kernel void gated_delta_step_bf16(device const bfloat*  q     [[buffer(0)]],
                                  device const bfloat*  k     [[buffer(1)]],
                                  device const bfloat*  v     [[buffer(2)]],
                                  device const bfloat*  gbuf  [[buffer(3)]],
                                  device const bfloat*  beta  [[buffer(4)]],
                                  device float*         state [[buffer(5)]],
                                  device bfloat*        y     [[buffer(6)]],
                                  constant gd_dims&     dims  [[buffer(7)]],
                                  uint3 tg   [[threadgroup_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]])
{
    uint dv = tg.x;
    uint hv = tg.y;
    if (hv >= dims.Hv || dv >= dims.Dv) return;

    uint Dk  = dims.Dk;
    uint rep = dims.Hv / dims.Hk;
    uint hk  = hv / rep;

    const device bfloat* q_row = q + hk * Dk;
    const device bfloat* k_row = k + hk * Dk;
    float gv = float(gbuf[hv]);
    float bv = float(beta[hv]);
    float vv = float(v[hv * dims.Dv + dv]);

    device float* st = state + (hv * dims.Dv + dv) * Dk;

    // Each lane owns Dk/SIMD_WIDTH contiguous state elements (cached in regs).
    float s[GDS_DK_PER_LANE];
    float kv = 0.0f;
    for (uint i = 0; i < GDS_DK_PER_LANE; ++i) {
        uint dk = lane * GDS_DK_PER_LANE + i;
        float v0 = st[dk] * gv;
        s[i] = v0;
        kv += v0 * float(k_row[dk]);
    }
    kv = simd_sum(kv);
    float delta = (vv - kv) * bv;

    float out = 0.0f;
    for (uint i = 0; i < GDS_DK_PER_LANE; ++i) {
        uint dk = lane * GDS_DK_PER_LANE + i;
        float v1 = s[i] + float(k_row[dk]) * delta;
        st[dk] = v1;
        out += v1 * float(q_row[dk]);
    }
    out = simd_sum(out);

    if (lane == 0u) y[hv * dims.Dv + dv] = bfloat(out);
}

// VARIATION — Mamba SSM step: same shape (state[Hv, Dv, Dk]), same
// parallelism. The only differences are the update equations
// (state = state * decay + dB ⊙ x, y = sum_dk state * C). Same SG-per-
// (hv, dv) tiling applies.
