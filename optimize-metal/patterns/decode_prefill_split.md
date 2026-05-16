# patterns/decode_prefill_split.md

## What

Some optimizations are great for **decode** (`Lq == 1`) but bad for
**prefill** (`Lq > 1`), or vice-versa. When a single kernel cannot hit
both targets, ship **two implementations** and dispatch the correct one
based on `Lq`:

```c
if (Lq > 1) {
    gpu_cmdbuf_dispatch(cb, pso_gemm,        ...);   // MMA tile path
} else {
    gpu_cmdbuf_dispatch(cb, pso_gemv_qmv4_8, ...);   // qmv4 register-tile path
}
```

## The natural splits

| Op            | Decode (`Lq=1`) kernel              | Prefill (`Lq>1`) kernel             |
|---------------|--------------------------------------|-------------------------------------|
| Linear (dense)| `gemv_bf16_4_v8` (qmv4 8 outputs/SG) | `gemm_bf16` (simdgroup_matrix MMA)  |
| MoE quant lin | `mxfp4_gus_qmv4_bf16` (per-token gather) | `qmm_t_gather_rhs_*` (sorted GEMM)|
| Expert mix    | `expert_mix_bf16_add` (K_top combine)| `moe_combine_scatter` (scatter)     |
| SDPA          | many SGs per (head,query_token)      | one SG per (head,query_token)       |

The two paths can share everything else (RMSNorm, RoPE, SwiGLU,
residual_add) — those are equally efficient at any `Lq`.

## Why

The fundamental difference is **arithmetic intensity**:

- Decode is `M=1` matmul: bandwidth-bound. The X vector is loaded once
  per output; the qmv4 register-tile + bf16 gather pattern is optimal.
- Prefill is `M=Lq` matmul: compute-bound. The MMA-tile (8x8 × 8x8 → 8x8)
  primitive gets you near peak FLOPS.

A single kernel that does both well does not exist on Apple GPUs.

## When to introduce the split

Don't split prematurely. Order of operations:

1. Get both phases working with ONE kernel set (the decode-friendly
   ones). Prefill will be slow but correct.
2. Profile. If prefill is the dominant cost (almost always when
   prompts are >128 tokens), introduce the prefill-optimized variant.
3. Switch to two-path dispatch.

## Pattern in main.c

```c
static gpu_pipeline* gemvm = (Lq > 1) ? pso_gemm : pso_gemv;
size_t MTILE = (Lq > 1) ? 2 : 1;
size_t Mblk  = (Lq + MTILE - 1) / MTILE;
gpu_arg_buf args[] = { ABUF(H), abQw, abQb, ABUF(Q), ABUF(dimsQp) };
must_dispatch(gemvm, args, 5, (size_t)(N_QHEADS*HEAD_DIM*32), Mblk, 1,
              256, 1, 1, "q_proj");
```

The grid/threadgroup sizes also differ — match what each kernel needs.

## Commits

The full sweep of decode/prefill split work is spread across many src-metal
commits:

- 1c4ddfd — MoE: grouped expert GEMM for prefill
- 9fce171 — gemv_bf16_4 — qmv4 pattern (decode)
- 28afb2e — MMA POC kernel
- 51ced6b — wire phase11 sorted-gather MMA MoE for prefill
- b69d31b — decode: register-cached X qmv4 kernels (gate_up_swiglu + down)
- b7d7cf2 — decode: 8-X register tile in bf16 GEMV
