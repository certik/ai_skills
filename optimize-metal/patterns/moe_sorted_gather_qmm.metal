// patterns/moe_sorted_gather_qmm.metal
//
// WHAT: The MMA quantized GEMM that the sorted-gather MoE prefill path
//       (E3) calls. After bucketize → flatten → gather_x_sorted (see
//       moe_sorted_gather_glue.metal), THIS kernel does the dense
//       quantized matmul:
//
//           y[m, n] = sum_k x[m, k] * W[indices[m], n, k]
//
//       where `indices[m]` selects the expert for row m. The cleverness
//       is that AFTER the gather, rows are already sorted by expert, so
//       each (BM × BN) MMA tile sees ONE expert per inner-K traversal —
//       no per-row gather inside the K loop.
//
// WHEN: Prefill MoE, once you've already wired E3's bucketize +
//       gather_x_sorted glue (moe_sorted_gather_glue.metal). Before that
//       glue is in place this kernel can't be called.
//
// SPEEDUP: 1.30–1.60× prefill over decode-style per-token gather GEMV.
//
// COMMITS:
//   3fce46d — qmm_t_gather_rhs MMA kernel + multi-expert correctness
//   51ced6b — wire phase11 sorted-gather MMA MoE for prefill (+24%)
//   6316747 — BM=16 BN=32 WM=1 WN=2 matches MLX non-NAX tile sizes
//
// PATTERN: This file is intentionally a THIN SHIM over an MLX `steel`
// helper (`fp_gather_qmm_rhs_impl`). MLX's steel headers do the heavy
// lifting: TG-mem staging of X and W, simdgroup_matrix MMA primitives,
// inline e8m0/fp4 dequant. We just pick the tile sizes (BM, BN, BK,
// WM, WN) and forward.
//
// TILE-SIZE NOTE: BM=16 BN=32 WM=1 WN=2 matches MLX's non-NAX path for
// the gpt-oss MoE shapes. Sweep `(BM, BN, WM, WN) ∈ {(8,16,1,1),
// (16,16,1,2), (16,32,1,2), (32,32,2,2)}` on YOUR hardware and shape.
// See GOTCHA: tile sizes that work for MLX on M2 may not be optimal on
// M4 or M1.
//
// Where to get `mlx_steel`: vendor MLX's `mlx/backend/metal/kernels/steel/`
// headers. See commit dd9aee3 in the gpt-oss reference impl.

#include "mlx_steel/fp_quantized_lite.h"

// Sorted-gather quantized matmul.
//   y[m, n] = sum_k x[m,k] * W[indices[m], n, k]
//
// x       — dense [M, K], rows already sorted by expert (gather output)
// w       — quantized [E, N, K_blocks] of fp4 (2 vals/byte)
// scales  — e8m0 [E, N, K_blocks] (one scale per 32-element block)
// indices — dense [M], expert id per row
// y       — output [M, N] (bias applied in a separate epilogue kernel)
//
// Dispatch: TG = (32 * WM*WN, 1, 1) — one TG per BM × BN output tile.
// Grid: (ceil(M/BM), ceil(N/BN), 1).
[[kernel]] void qmm_t_gather_rhs_bf16_g32_b4(
    const device bfloat*    x        [[buffer(0)]],
    const device uint8_t*   w        [[buffer(1)]],
    const device uint8_t*   scales   [[buffer(2)]],
    const device uint32_t*  indices  [[buffer(3)]],
    device bfloat*          y        [[buffer(4)]],
    constant uint&          M        [[buffer(5)]],
    constant uint&          N        [[buffer(6)]],
    constant uint&          K        [[buffer(7)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BM = 16, BN = 32, BK = 32;
    constexpr int WM = 1,  WN = 2;
    constexpr int BK_padded = BK + 16 / sizeof(bfloat);   // avoid TG-mem bank conflicts

    threadgroup bfloat Xs[BM * BK_padded];
    threadgroup bfloat Ws[BN * BK_padded];

    // group=32, bits=4 (mxfp4); bias=nullptr (epilogue kernel applies bias).
    mlx_lite::fp_gather_qmm_rhs_impl<bfloat, /*group*/32, /*bits*/4,
                                     BM, BK, BN, WM, WN>(
        x, w, scales, /*bias*/(const device bfloat*)0, indices, y,
        Xs, Ws,
        (int)M, (int)N, (int)K,
        tid, simd_gid, simd_lid);
}

// EXTENSION: a second variant `qmm_t_gather_rhs_bf16_g32_b4_add` adds a
// residual via DO_ADD. Pattern is identical — just thread a residual
// buffer through and OR-in `Y[m, n] += residual[m, n]` in the epilogue.
// See the gemv_with_residual_epilogue.metal pattern for the analogous
// fusion idea on the decode side.
