// patterns/gemm_x_offset_row_prune.c
//
// WHAT: Compute the final linear (e.g., lm_head, V = 152064) for only the
//       row(s) the host actually consumes, by passing a byte offset into
//       the input buffer instead of slicing/copying it.
//
//       The GEMM kernel reads `X + x_off_bytes` as if it were the start
//       of the input, computes `M_used` output rows, and writes to row 0
//       of the output buffer. The host then reads from row 0 — no per-row
//       shift logic, no temporary buffer, no extra dispatch.
//
// WHEN: Final linear is huge in N (vocab × hidden) AND the host only
//       consumes a small subset of M_full output rows. Concretely:
//
//       - Diffusion-LLM prefetch: the full forward at M = L (e.g., 89)
//         exists only to populate the KV cache; only ONE lm_head row
//         (the position-to-decode) is needed.
//       - Speculative draft-sample heads where you score only k draft
//         positions out of L.
//
//       Most autoregressive decoders already do M=1, so this is moot
//       there. It's a diffusion-LLM / spec-decode-specific win.
//
// SPEEDUP: Saves (M_full / M_used) - 1 weight-streaming passes of W.
//          For lm_head with N=152k, K=3584 at BM=32, dropping M=89 → 1
//          saves ~2 GB of W streaming.
//          Wall: 20–50 ms per prefetch call on M4 Max.
//          (Dream-7B prefetch: 0.65 → 0.62 s in our port.)
//
// CAVEAT: The semantics of "which row is needed" depend on sampler
//         shift conventions. Most commonly: "logits row for position p
//         comes from hidden state at position p-1" (causal LM with a
//         left-shift). Trace the host code carefully — the off-by-one
//         here trips up most refactors. If `lm_head_only_row` is wrong
//         by 1, the model still generates plausible text — just the
//         WRONG plausible text.
//
// COMMIT:  304484b — "Single-row lm_head prefetch via X offset
//          (prefetch 0.65→0.62s)"
//
// HOW IT WORKS:
//   The Metal shim's `gpu_arg_buf` accepts `{ buf, byte_offset }`. The
//   driver hands the kernel `device const T* X = (const T*)(buf + off)`,
//   so a row offset becomes a pointer-add — zero-copy, zero-dispatch.

#include "metal_shim.h"

// Existing helper: dispatch the GEMM at offset 0 (the common case).
static void d_linear_full(gpu_cmdbuf* cb, gpu_buf* X, gpu_buf* W, gpu_buf* B,
                          gpu_buf* Y, gpu_buf* R,
                          uint32_t M, uint32_t K, uint32_t N,
                          int has_bias, int has_residual);

// New helper: same dispatch but with a byte offset into X. The kernel
// itself is unchanged — only the device pointer changes.
static void d_linear_full_off(gpu_cmdbuf* cb, gpu_buf* X, size_t x_off_bytes,
                              gpu_buf* W, gpu_buf* B, gpu_buf* Y, gpu_buf* R,
                              uint32_t M, uint32_t K, uint32_t N,
                              int has_bias, int has_residual)
{
    const uint32_t GBN = 64;
    const uint32_t GBM = (M > 16) ? 32u : 16u;          // dual-tile by M
    const uint32_t THREADS_PER_TG = 64;
    gpu_pipeline* pso = (M > 16) ? pso_gemm_bm32 : pso_gemm;
    uint32_t flags = (has_bias ? 1u : 0u) | (has_residual ? 2u : 0u);
    struct { uint32_t M, K, N, flags; } pp = { M, K, N, flags };
    gpu_buf* bias_buf = has_bias     ? B : g_zero_bias_buf;
    gpu_buf* res_buf  = has_residual ? R : g_zero_bias_buf;

    // The critical line: pass x_off_bytes alongside X. The kernel will
    // see &X[x_off_bytes / sizeof(elem)] as its row-0 pointer.
    gpu_arg_buf args[6] = {
        { X, x_off_bytes },           // <-- offset into the same buffer
        ARG(W), ARG(bias_buf), ARG(Y),
        push_params(&pp, sizeof pp),
        ARG(res_buf)
    };

    uint32_t n_tiles_x = (N + GBN - 1) / GBN;
    uint32_t n_tiles_y = (M + GBM - 1) / GBM;
    gpu_cmdbuf_dispatch(cb, pso, args, 6,
                        n_tiles_x * THREADS_PER_TG, n_tiles_y, 1,
                        THREADS_PER_TG, 1, 1);
}

// Caller — at the end of forward_cached(), choose:
//
//   if (lm_head_only_row >= 0) {
//       size_t x_off = (size_t)lm_head_only_row * (size_t)H * sizeof(bf16);
//       d_linear_full_off(cb, h_buf, x_off, g_lm_head_buf, NULL,
//                         logits_buf, NULL, /*M=*/1, H, V,
//                         /*has_bias*/0, /*has_residual*/0);
//       logits_rows = 1;
//   } else {
//       d_linear(cb, h_buf, g_lm_head_buf, NULL, logits_buf, LqU, H, V, 0);
//       logits_rows = LqU;
//   }
//
// `lm_head_only_row` is the *single* row index of h_buf that the host
// will care about (typically `mask_position_to_decode - 1` for causal
// LMs). Set to -1 to fall back to the full-M path (refine, training,
// or when multiple rows are scored).
