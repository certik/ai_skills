// Pattern: per-layer embed_gather, TG-per-row + 128-thread cooperative bfloat4 copy.
//
// When to use:
//   Models with per-layer additional embeddings — i.e. an extra
//   [V, L*D_pli] embedding table where each layer gathers one row per
//   token and adds it to the hidden state. Examples: Gemma 3, Gemma 4,
//   some recent multimodal models. KPROF will show this as a kernel
//   that's surprisingly hot — typically 4% of GPU at ~600 µs/call
//   even though the work is "just" gathering one row.
//
// What it replaces:
//   The naive port-c-to-metal kernel, which is structurally:
//     kernel void embed_gather_bf16_naive(
//         device const bfloat* table,
//         device const uint*   ids,
//         device bfloat*       out, ...)
//     {
//         uint m = tid;
//         uint row = ids[m];
//         for (uint d = 0; d < D; d++) out[m*D + d] = table[row*D + d];
//     }
//   dispatched as `(M, 1, 1)` / `(1, 1, 1)` — i.e. 1 thread per row,
//   serially reading D bfloat values. On Gemma 4 E4B that's 10752
//   serial bf16 loads = ~597 µs/call at memory-latency floor (~36
//   GB/s, vs M4 Max's 410 GB/s peak).
//
// Expected speedup (Gemma 4 E4B, M4 Max, bf16):
//   - Per-call: 596.7 µs → 9.0 µs  (66x kernel speedup)
//   - Wall:     short 48.7 → 50.9 tok/s (+4.5%)
//               long  43.7 → 45.3 tok/s (+3.7%)
//   - This single change crossed MLX parity → MLX-beating on Gemma 4.
//
// Why TG-per-row + 128 threads (not just N_SG=1)?
//   At decode (M=1) we want all 128 threads (4 SGs) cooperating on
//   ONE row of 10752 bf16 = 2688 bfloat4 loads. 128 threads × 21
//   loads/thread = full row in 21 sequential bfloat4 reads. Single SG
//   (32 threads × 84 loads) would also work but 4 SGs amortize
//   per-TG overhead better. At prefill (M=L), each TG still owns one
//   row — natural scaling.
//
// Alignment requirement:
//   bfloat4 loads must be at 8-byte-aligned addresses. All Gemma /
//   Llama / Qwen embed dims are divisible by 4, and the table is
//   stored contiguously in safetensors order, so this lands cleanly.
//   If your D is NOT divisible by 4 (rare), fall back to bfloat2 or
//   plain bfloat with a tail loop.
//
// Dispatch (host C):
//
//   gpu_cmdbuf_dispatch(cb, pso_embed_gather, args, 5,
//                       (size_t)M * 128, 1, 1,   // total threads
//                       128, 1, 1);              // 128 threads/TG
//
// (gotcha #57 in references/gotchas.md has the full story.)

#include <metal_stdlib>
using namespace metal;

kernel void embed_gather_bf16(
    device const bfloat4* table          [[buffer(0)]],   // [V, D/4]
    device const uint*    ids            [[buffer(1)]],   // [M]
    device bfloat4*       out            [[buffer(2)]],   // [M, D/4]
    constant uint&        D              [[buffer(3)]],   // hidden dim
    constant uint&        M              [[buffer(4)]],   // # rows
    uint  tid_in_tg      [[thread_position_in_threadgroup]],
    uint  tg_id          [[threadgroup_position_in_grid]],
    uint  threads_per_tg [[threads_per_threadgroup]])
{
    if (tg_id >= M) return;

    uint  D4  = D >> 2u;            // D / 4 — # bfloat4 elements per row
    uint  row = ids[tg_id];

    device const bfloat4* src = table + (size_t)row * D4;
    device bfloat4*       dst = out   + (size_t)tg_id * D4;

    // Cooperative copy: each thread strides through the row at
    // `threads_per_tg` lanes apart. For D=10752, D4=2688, and 128
    // threads/TG, each thread does 21 sequential bfloat4 loads.
    for (uint i = tid_in_tg; i < D4; i += threads_per_tg) {
        dst[i] = src[i];
    }
}

// ---------------------------------------------------------------
// VARIANT: transposed table (table is [V, L, D_pli] interleaved by
// layer, and you gather layer-l's slice per call). Same idea —
// only the `src` pointer offsets differ. Pass `layer_id` and
// `D_per_layer` as constants:
//
//   kernel void embed_gather_bf16_per_layer(
//       device const bfloat4* table  [[buffer(0)]],  // [V, L*D_pli/4]
//       device const uint*    ids    [[buffer(1)]],  // [M]
//       device bfloat4*       out    [[buffer(2)]],  // [M, D_pli/4]
//       constant uint&        D_pli      [[buffer(3)]],
//       constant uint&        M          [[buffer(4)]],
//       constant uint&        layer_id   [[buffer(5)]],
//       constant uint&        L          [[buffer(6)]],
//       uint  tid_in_tg      [[thread_position_in_threadgroup]],
//       uint  tg_id          [[threadgroup_position_in_grid]],
//       uint  threads_per_tg [[threads_per_threadgroup]])
//   {
//       if (tg_id >= M) return;
//       uint D_pli_4 = D_pli >> 2u;
//       uint row = ids[tg_id];
//       device const bfloat4* src =
//           table + (size_t)row * (L * D_pli_4) + layer_id * D_pli_4;
//       device bfloat4* dst = out + (size_t)tg_id * D_pli_4;
//       for (uint i = tid_in_tg; i < D_pli_4; i += threads_per_tg)
//           dst[i] = src[i];
//   }
//
// Use whichever matches your model's per-layer embed layout.
