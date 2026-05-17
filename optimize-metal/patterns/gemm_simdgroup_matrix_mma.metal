// patterns/gemm_simdgroup_matrix_mma.metal
//
// WHAT: Use Apple's hardware matrix-multiply unit via the MSL
//       `simdgroup_matrix<T, 8, 8>` primitive.  Each SIMD group computes
//       an 8x8 tile of output per `simdgroup_multiply_accumulate` call.
//       Tile across threadgroups (BM × BN) and within threadgroup across
//       warps (WM × WN SIMD-groups).
// WHEN: Prefill GEMM is hot.  Naive 1-thread-per-output prefill GEMM is
//       ~50–100× slower than this on M2+.
// SPEEDUP: 3–10× on prefill GEMMs.
// SHAPES: M, N, K must be multiples of 8.  Recommended tile:
//         BM=16 BN=32 WM=1 WN=2 (matches MLX non-NAX, see commit 6316747).
// COMMITS:
//   dd9aee3 — vendor MLX steel + fp_quantized headers
//   28afb2e — MMA POC kernel passes correctness test (max|d|=0.028 vs MLX-Py)
//   3fce46d — qmm_t_gather_rhs MMA + multi-expert correctness
//   6316747 — BM=16 BN=32 WM=1 WN=2 — match MLX non-NAX tile sizes (+26%)
//
// This is a SKETCH — the actual gpt-oss reference impl uses MLX's `steel` matmul
// headers (vendored from MLX 0.31.2).  See src-metal/kernels/phase6_mma.metal
// and src-metal/kernels/mlx_steel/ in the gpt-oss repo for a working version.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 16;     // output tile rows per threadgroup
constant constexpr uint BN = 32;     // output tile cols per threadgroup
constant constexpr uint WM = 1;      // SG tiling along M
constant constexpr uint WN = 2;      // SG tiling along N
constant constexpr uint TM = BM/WM/8; // 8x8 MMA tiles per SG along M
constant constexpr uint TN = BN/WN/8; // 8x8 MMA tiles per SG along N
constant constexpr uint THREADS_PER_TG = 32 * WM * WN;   // SIMD_WIDTH * WM * WN

// Y[m,n] = sum_k X[m,k] * W[n,k]    (W row-major [N,K], so reading
// row n of W is contiguous in K — good for the K dimension)
// Grid:        (BN_grid * 32, BM_grid, 1)   where BN_grid = ceil(N/BN),
//                                                 BM_grid = ceil(M/BM)
// Threadgroup: (THREADS_PER_TG, 1, 1)
kernel void gemm_bf16_mma(device const bfloat*   X       [[buffer(0)]],   // [M, K]
                          device const bfloat*   W       [[buffer(1)]],   // [N, K]
                          device const bfloat*   B       [[buffer(2)]],   // [N]
                          device bfloat*         Y       [[buffer(3)]],   // [M, N]
                          constant uint3&        dims    [[buffer(4)]],   // (M, K, N)
                          uint2                  tgid    [[threadgroup_position_in_grid]],
                          uint                   sgid    [[simdgroup_index_in_threadgroup]])
{
    uint M = dims.x, K = dims.y, N = dims.z;
    uint m0 = tgid.y * BM;
    uint n0 = tgid.x * BN;

    // SG tile coords within the threadgroup.
    uint sg_m = sgid / WN;
    uint sg_n = sgid % WN;

    // Output accumulator: TM × TN tiles of 8x8 floats per SG.
    simdgroup_matrix<float, 8, 8> C[TM * TN];
    for (uint i = 0; i < TM * TN; ++i) C[i] = simdgroup_matrix<float, 8, 8>(0);

    // K-loop: load 8-K-wide slabs of X and W, multiply-accumulate.
    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<bfloat, 8, 8> A[TM];
        simdgroup_matrix<bfloat, 8, 8> Bt[TN];

        // Load A[i] = X[m0 + sg_m*TM*8 + i*8 : ..., k : k+8]
        for (uint i = 0; i < TM; ++i) {
            uint mm = m0 + sg_m * TM * 8 + i * 8;
            simdgroup_load(A[i], X + mm * K + k, K);
        }
        // Load Bt[j] = W[n0 + sg_n*TN*8 + j*8 : ..., k : k+8]^T
        // (W is [N, K], so W[n, k] is what we want — but simdgroup_matrix
        // expects 8x8 column-major; use the transpose flag.)
        for (uint j = 0; j < TN; ++j) {
            uint nn = n0 + sg_n * TN * 8 + j * 8;
            simdgroup_load(Bt[j], W + nn * K + k, K, ulong2(0, 0), /*transpose=*/true);
        }

        // C[i,j] += A[i] @ Bt[j]
        for (uint i = 0; i < TM; ++i)
            for (uint j = 0; j < TN; ++j)
                simdgroup_multiply_accumulate(C[i*TN+j], A[i], Bt[j], C[i*TN+j]);
    }

    // Store: add bias + cast to bf16 + write to Y.
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            uint mm = m0 + sg_m * TM * 8 + i * 8;
            uint nn = n0 + sg_n * TN * 8 + j * 8;
            // (sketch only — real version uses simdgroup_store with bias add)
            simdgroup_store(C[i*TN+j], Y + mm * N + nn, N);
        }
    }
}

// FOR REAL USE: see MLX 0.31.2's `mlx/backend/metal/kernels/steel/gemm/*.h`
// or the gpt-oss repo's `src-metal/kernels/mlx_steel/` directory.  Those handle
// the boundary conditions (M, N, K not multiples of tile), masking,
// SoftMax fusion, transposed-LHS variants, and the qmm (reduced-precision) variant.
