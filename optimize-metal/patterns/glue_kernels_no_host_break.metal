// patterns/glue_kernels_no_host_break.metal
//
// WHAT: Replace EVERY host-side scratch op inside forward() (memcpy,
//       scalar broadcast, ring-buffer shift, split-by-channel, KV-cache
//       append, etc.) with a tiny GPU "glue" kernel. The goal isn't
//       speed of those ops themselves — they're trivial. The goal is to
//       collapse forward() into ONE command buffer with ONE commit_wait
//       at the end.
// WHEN: Right after correctness; this is often the *single largest
//       optimization in the entire pipeline*. In one Qwen3.6 port this
//       single change took decode from 19 → 34 tok/s (1.79x).
// WHY:  Naive ports tend to break out to the host for any "weird" op
//       (state roll, broadcast multiply, conv-state update, KV cache
//       append). Each break = `commit_wait` + new cmdbuf. On a model
//       with N_LAYERS=40 and ~3 such breaks per layer, decode pays
//       ~120 commit_wait round-trips per token at 0.3–1.0 ms each =
//       a fixed 40–120 ms / token overhead, regardless of GPU speed.
// SPEEDUP: 1.5–2.0× decode on models with many small host-side ops.
//
// The kernels below are intentionally trivial — one thread per output
// element. They run in microseconds each. Their value is purely
// removing the commit_wait between them and the rest of the cmdbuf.
//
// PATTERN: any time `forward()` does
//   commit_wait(cb)
//   memcpy / for-loop on host
//   cb = new_cmdbuf()
// audit whether the host work can be a 5-line Metal kernel.
//
// Six concrete glue kernels used in the Qwen3.6 port:
//   shared_expert_combine — r = moe + sigmoid(sg) * sd  (per-token scalar broadcast)
//   conv_state_update     — depthwise conv1d state ring shift
//   qkv_split             — split conv_out [L,2*Dk+Dv] into qlin|klin|vlin
//   q_proj_split          — split q_proj [L,Nq,2*D] into Q + gate
//   kv_cache_write        — append K/V rows to per-layer KV cache
//   copy_bf16             — plain bf16 -> bf16 (e.g., beta = b before sigmoid)

#include <metal_stdlib>
using namespace metal;

// 1) Shared expert gate broadcast + residual fold.
//    r[l, d] = moe[l, d] + sigmoid(sg[l]) * sd[l, d]
struct shared_combine_dims { uint L; uint H; };
kernel void shared_expert_combine_bf16(device const bfloat*  moe  [[buffer(0)]],
                                       device const bfloat*  sg   [[buffer(1)]],
                                       device const bfloat*  sd   [[buffer(2)]],
                                       device bfloat*        r    [[buffer(3)]],
                                       constant shared_combine_dims& dims [[buffer(4)]],
                                       uint2 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint l = gid.y;
    if (l >= dims.L || d >= dims.H) return;
    float gs = 1.0f / (1.0f + exp(-float(sg[l])));
    size_t off = (size_t)l * dims.H + d;
    r[off] = bfloat(float(moe[off]) + gs * float(sd[off]));
}

// 2) Depthwise conv1d state ring-buffer update.
//    if Lq >= P:  state[:, :] = X[Lq-P:Lq, :]
//    else:        state[:P-Lq, :] = state[Lq:P, :]; state[P-Lq:, :] = X[:Lq, :]
struct conv_state_dims { uint Lq; uint C; uint P; };
kernel void conv_state_update_bf16(device const bfloat*    X     [[buffer(0)]],
                                   device bfloat*          state [[buffer(1)]],
                                   constant conv_state_dims& dims [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    uint c   = gid.x;
    uint row = gid.y;
    if (row >= dims.P || c >= dims.C) return;
    uint Lq = dims.Lq;
    uint P  = dims.P;
    if (Lq >= P)             state[row * dims.C + c] = X[(Lq - P + row) * dims.C + c];
    else if (row < P - Lq)   state[row * dims.C + c] = state[(row + Lq) * dims.C + c];
    else                     state[row * dims.C + c] = X[(row - (P - Lq)) * dims.C + c];
}

// 3) Append K_new, V_new to per-layer KV cache at offset q_off.
struct kv_write_dims { uint L; uint Nkv; uint D; uint q_off; };
kernel void kv_cache_write_bf16(device const bfloat*    K_new  [[buffer(0)]],
                                device const bfloat*    V_new  [[buffer(1)]],
                                device bfloat*          K_cache[[buffer(2)]],
                                device bfloat*          V_cache[[buffer(3)]],
                                constant kv_write_dims& dims   [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint l = gid.y;
    uint NKV_D = dims.Nkv * dims.D;
    if (l >= dims.L || i >= NKV_D) return;
    size_t src_off = (size_t)l * NKV_D + i;
    size_t dst_off = (size_t)(dims.q_off + l) * NKV_D + i;    K_cache[dst_off] = K_new[src_off];
    V_cache[dst_off] = V_new[src_off];
}

// 4) Split q_proj output [L, Nq, 2*D] into Q [L, Nq, D] and gate [L, Nq, D].
struct q_split_dims { uint L; uint Nq; uint D; };
kernel void q_proj_split_bf16(device const bfloat*   src   [[buffer(0)]],
                              device bfloat*         q_out [[buffer(1)]],
                              device bfloat*         g_out [[buffer(2)]],
                              constant q_split_dims& dims  [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]])
{
    uint d  = gid.x;
    uint hq = gid.y;
    uint l  = gid.z;
    if (l >= dims.L || hq >= dims.Nq || d >= dims.D) return;
    size_t base = ((size_t)l * dims.Nq + hq) * 2u * dims.D;
    size_t out  = ((size_t)l * dims.Nq + hq) * dims.D + d;
    q_out[out] = src[base + d];
    g_out[out] = src[base + dims.D + d];
}

// 5) Split conv1d output [L, Dk*2 + Dv] into qlin/klin/vlin.
struct qkv_split_dims { uint L; uint Dk; uint Dv; uint C_total; };
kernel void qkv_split_bf16(device const bfloat*    src  [[buffer(0)]],
                           device bfloat*          qlin [[buffer(1)]],
                           device bfloat*          klin [[buffer(2)]],
                           device bfloat*          vlin [[buffer(3)]],
                           constant qkv_split_dims& dims [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]])
{
    uint c = gid.x;
    uint l = gid.y;
    if (l >= dims.L || c >= dims.C_total) return;
    bfloat v = src[l * dims.C_total + c];
    if (c < dims.Dk)            qlin[l * dims.Dk + c]                  = v;
    else if (c < 2u * dims.Dk)  klin[l * dims.Dk + (c - dims.Dk)]      = v;
    else                        vlin[l * dims.Dv + (c - 2u * dims.Dk)] = v;
}

// 6) Plain bf16 -> bf16 copy (replaces host memcpy when the copy feeds
//    a downstream GPU kernel and the source is also a GPU buffer).
struct copy_dims { uint N; };
kernel void copy_bf16(device const bfloat*  src [[buffer(0)]],
                      device bfloat*        dst [[buffer(1)]],
                      constant copy_dims&   p   [[buffer(2)]],
                      uint i [[thread_position_in_grid]])
{
    if (i >= p.N) return;
    dst[i] = src[i];
}
