# Optimization Catalog

The full menu of named optimization techniques, organized by bottleneck
category. SKILL.md gives the **empirical attack order** (what to apply
first); this file is the menu you consult when the profile tells you
which kernel category is hot.

Each entry has:

- **What** the technique does
- **When** to apply (signal in the profile)
- **Speedup** (typical, from the gpt-oss / Qwen3.6 reference impls)
- **Snippet** → file in `patterns/`
- **Commit** → SHA in the original work, in case you want to see the
  full diff

The lettered ordering (A–I) is by category, NOT by importance. Within
each section the entries are roughly in the order you'd apply them.

## Table of contents

- [A. Host-side scheduling](#a-host-side-scheduling) — cmdbuf batching, param ring, pipeline, concurrent encoder, parallel weight loader
- [B. GEMV (decode Linear)](#b-gemv--decode-time-linear-layers) — simdgroup-per-output, bfloat4, qmv4 register tile, TG-mem X share, residual epilogue
- [C. GEMM (prefill Linear)](#c-gemm--prefill-time-linear-layers) — simdgroup_matrix MMA, tile sweep, fused-op templates, aspect-ratio dispatch
- [D. SDPA](#d-sdpa--attention) — parallel softmax, online softmax, multi-SG, sliding KV
- [E. MoE](#e-moe--typically-the-biggest-bottleneck) — decode qmv4, fused gate_up_swiglu, prefill sorted-gather, fused combine_scatter
- [F. Reductions](#f-reductions--small-but-pesky-kernels) — parallel argmax, parallel topk softmax, per-row softmax+argmax (diffusion samplers)
- [G. Decode/Prefill split](#g-decodeprefill-split--only-when-techniques-diverge)
- [H. Host/GPU boundary cleanup](#h-hostgpu-boundary-cleanup--often-the-biggest-single-win) — glue kernels, embed+forward+argmax merge, prune unused output rows
- [I. Recurrent / state-update kernels](#i-recurrent--state-update-kernels) — SG-per-(row,dim) for RNN/SSM

---

## A. Host-side scheduling

Apply first; costs nothing in correctness.

### A1. Batch all dispatches per forward() pass

**What**: For each `forward()` call, open ONE command buffer, record all
~340 dispatches per token, then commit + wait once. Don't commit between
layers. Don't recreate pipelines / buffers per step.

**When**: Always, from the start. The naive port may already do this
(see `port-c-to-metal` Phase 3). If not, this alone is ~1.8× decode
speedup.

**Speedup**: 1.5–1.8× decode (per-dispatch overhead is ~30 µs at full
queue depth).

**Snippet**: `patterns/cmdbuf_batch_dispatches.c`

**Commit**: a13d0de — *Batch dispatches per forward pass (1.78x decode speedup)*

### A2. Persistent param buffers

**What**: The forward() function writes many small "dims" / "params"
buffers each call. Don't allocate fresh; allocate once, refill the same
buffer in place (safe because each forward ends with commit+wait, so the
GPU is idle).

```c
#define PARAM_BUF(name, sz) \
    static gpu_buf* name##_buf = NULL; \
    if (!name##_buf) name##_buf = gpu_buf_new(g_ctx, (sz)); \
    void* name##_dst = gpu_buf_contents(name##_buf)

uint32_t dimsRMS[2] = {(uint32_t)Lq, HIDDEN};
PARAM_BUF(dimsRMS, sizeof(dimsRMS));
memcpy(dimsRMS_dst, dimsRMS, sizeof(dimsRMS));
```

**When**: After A1. Easy win.

**Speedup**: 1.01–1.02× decode (88→89 tok/s in the gpt-oss reference impl).

**Snippet**: `patterns/param_buf_persistent.c`

**Commit**: 2b1d6ef.

### A3. CONST_PARAM_BUF — lift call-invariant constants

**What**: Some "params" are the same across every forward() call (e.g.,
the `dims` for an o_proj that has fixed shapes). Move them to a separate
const ring; only the per-call params (q_off, Lq) refill.

**When**: After A2, if you see param-buffer setup time non-trivial in
the encode-side profile.

**Speedup**: ~1.01×, but reduces CPU encode time materially when
pipelined.

**Snippet**: `patterns/param_buf_const.c`

**Commit**: 8001db6.

### A4. 2-deep command-buffer pipeline

**What**: While GPU runs decode step `k`, CPU encodes step `k+1` into a
fresh command buffer using a duplicated param ring (`slot ∈ {0,1}`).
At step `k+1`, swap roles. Two cmdbufs in flight at any time.

**When**: After A1–A3. The decode loop's GPU/CPU overlap is the limit.

**Speedup**: 1.10–1.20× decode (varies by kernel mix).

**Snippets**: `patterns/cmdbuf_pipeline_2deep.c`,
`patterns/cmdbuf_pipeline_2deep_id_swap.c` (variant with id-buffer swap
so the CPU never races with the in-flight argmax write).

**Commit**: 1f1afbd.

### A5. Concurrent encoder with explicit barriers

**What**: By default, Metal's compute encoder serializes every dispatch.
Switching to the *concurrent* encoder lets independent dispatches run in
parallel. Insert explicit barriers (`memoryBarrierWithScope:`) where
real data dependencies exist (e.g., between q_proj and rope_q).
Independent ops (q_proj/k_proj/v_proj or gate/up) may overlap.

**When**: After A1–A4, if profile shows GPU idle gaps between dispatches.

**Speedup**: 1.05–1.10× depending on overlap opportunity.

**Snippet**: `patterns/concurrent_encoder.c`

**Commit**: d08def0.

### A6. JIT-compile each kernel from source

**What**: Apple's `MTLLibrary newLibraryWithSource:` JIT-compiles MSL.
Per-kernel sources can be compiled lazily on first dispatch and cached.
For many kernels, dedicate one combined library at startup (see
`kernel_concat.c`) so JIT happens once.

**When**: Always. The naive port already does this through `kernel_concat`.

**Speedup**: indirect — avoids first-dispatch hiccups in timing.

**Commits**: a42da85, d460f8f, bd6dd1b, 2c40b09, 1ee3915, 039ae22, fc99872.

### A7. `wired_limit` hint for working-set residency

**What**: `[MTLDevice setShouldMaximizeConcurrentCompilation:YES]` and
`[buffer setPurgeableState:MTLPurgeableStateNonVolatile]` keep the
weights resident in unified memory. The `recommendedMaxWorkingSetSize`
API gives you the residency budget.

**When**: If you see paging hiccups with large models.

**Speedup**: variable; eliminates worst-case stalls.

**Commit**: 6361113.

### A8. Parallel independent computation chains (END-GAME WIN)

**What**: With concurrent encoder enabled (A5), look for SUBGRAPHS that
share only an input buffer and merge only at a single fan-in. Two
independent chains (e.g., MoE main vs shared-expert in Qwen MoE
models, or gate vs up projection trees) can run **fully concurrently**
when you drop ALL barriers between them and re-add barriers only at
true fan-in points.

This is qualitatively different from A5: A5 overlaps pairs of
neighbouring independent dispatches; A8 interleaves entire chains so
they execute as parallel pipelines.

**When**: Late-stage optimization, when you're at ~95% of MLX and
kernel-level techniques are exhausted. Required signal: `gpu_busy ≈
wall` (you ARE GPU-bound) AND your encoder graph shows two subgraphs
sharing only the input.

**Speedup**: 1.03–1.05× decode (the gap to MLX parity in Qwen3.6:
66.9 → 69.5 tok/s).

**Reference doc**: `references/parallel-chains.md`

**Commit**: d813cf5 (Qwen3.6-35B-A3B: parallel MoE + shared expert chains,
MLX parity in 1 commit).

### A9. Parallel pread weight loader — beats mmap+memcpy 2.5–3×

**What**: At startup, weights are read from disk into MTLBuffers.
The naive "mmap shard + memcpy per array (tensor)" is **page-fault
bound** on macOS (the VM lock serializes faults at ~1 GB/s, even with
parallel threads). The fix is to skip mmap for the bulk copy entirely
and use **parallel `pread()` per shard**:

1. Pre-allocate one MTLBuffer per array (serial — alloc isn't
   guaranteed thread-safe).
2. Bucket arrays by `shard_idx`, preserving order so each shard's
   reads are sequential on disk.
3. `dispatch_apply(n_shards)`: each iteration opens one fd and
   `pread`s its arrays directly into the matching MTLBuffer's host
   pointer (`gpu_buf_contents`).

Also: **delete any `madvise(MADV_WILLNEED)` calls** — on macOS that's
blocking, adding ~5s for 35 GB (see `references/gotchas.md` #28).

**Diagnostic — phase-breakdown line**: before applying A9 (and after,
to confirm), log a one-liner separating tokenizer / config / metal /
weights / total. If `weights` is 30%+ of `total`, A9 will pay off
significantly:

```
[startup] tokenizer=0.003s config=0.000s metal=0.035s weights=0.817s total=0.856s
```

**Why not zero-copy `newBufferWithBytesNoCopy:` per tensor?** Apple's
API requires the host pointer AND length to be `vm_page_size`-aligned
(16 KB on Apple Silicon). Inside a safetensors shard, individual
tensor offsets are 8-byte aligned (set by the JSON header layout), so
per-tensor wrap fails. The *whole shard* IS page-aligned (mmap
guarantees this for the base ptr; round length up to a page), so you
*can* wrap one buffer per shard and pass per-tensor byte offsets via
`gpu_arg_buf.off_bytes` to every dispatch — but plumbing offsets
through every weight reference is invasive and parallel pread (this
section) already lands startup below MLX, so the refactor isn't worth
it on shipping ports. See gotcha #40.

**When**: From the very first port — startup is the most user-visible
"wait" and it's trivial to fix.

**Speedup**:
- 28.5 s → 5 s startup (Qwen3.6-35B, 8 shards × ~4 GB, M4 Max).
- 1.71 s → 0.82 s weights (Dream-7B, 4 shards × ~3.5 GB, M4 Max),
  total wall 3.08 s → 2.17 s — beats MLX (2.67 s) on a short
  diffusion bench by 19%.

Both beat MLX's `ParallelFileReader` on the same machine. See
`references/gotchas.md` #28, #29, #40.

**Snippet**: `patterns/load_parallel_pread.c`

---

## B. GEMV — for decode-time Linear layers

These apply to every `linear_bf16` call where `M == Lq == 1` (decode):
q_proj, k_proj, v_proj, o_proj, router, lm_head.

### B1. SIMD-group-per-output

**What**: One SIMD group (32 threads on Apple GPUs) computes one output
element of the GEMV. Each thread handles `K/32` of the inner-product;
final reduction via `simd_sum`. Instead of one thread doing K
multiply-adds serially, you have 32 threads doing K/32 in parallel.

**When**: Always, for any bf16 GEMV. Single-thread-per-output is ~30×
slower than this.

**Speedup**: 1.3–1.5× decode (5b7b16c gemv_bf16 = +38%).

**Snippet**: `patterns/gemv_simdgroup_per_output.metal`

**Commit**: 5b7b16c.

### B2. `bfloat4` vector loads

**What**: Load 4 bf16 values at a time using `bfloat4*` reinterpret
casts (or the equivalent `device const bfloat4*`). Halves memory
transactions.

**When**: After B1. K must be a multiple of 4 (true for HEAD_DIM=64,
HIDDEN=2880, INTER=2880, etc.).

**Speedup**: 1.10–1.15× on bandwidth-bound GEMVs.

**Snippet**: `patterns/gemv_simdgroup_per_output.metal` — see the
*VARIANT — with bfloat4 vector loads* section at the bottom. The
bfloat4 form is a 6-line diff over the B1 kernel, not a separate file.

**Commit**: 5b7b16c (combined with B1 in the same commit).

### B3. Register-tile X across multiple outputs (qmv4 pattern)

**What**: Each SIMD group computes K outputs (e.g., 4 or 8) of the
GEMV. The X-vector is loaded **once** per SG and reused across the K
outputs (cached in registers). Cuts X bandwidth by Kx. Pattern matches
MLX's `qmv_fast`.

**When**: After B1, B2. Most GEMVs are X-bandwidth bound by then.

**Speedup**: 1.2–1.4× per GEMV; 1.10–1.20× overall decode.

**Snippet**: `patterns/gemv_qmv4_register_tile.metal`

**Commits**: 9fce171 (gemv_bf16_4 — 4 outputs/SG), b7d7cf2
(8 outputs/SG), b69d31b (register-cached X for mxfp4 qmv4).

**Pitfall**: K_OUT has a sweet spot. For q8 affine dequant (4 bytes
unpacked + 4 dot-products per inner iter), K_OUT=4 wins; K_OUT=8
*hurts* (decode 41 → 31 tok/s in one port) because the inner loop
exceeds the register budget and the compiler spills. Default to 4
for q8/affine; revisit only after MMA has displaced GEMV. See
`references/gotchas.md` #8.

### B4. Share X via threadgroup memory

**What**: If you have multiple SIMD groups in one threadgroup all
reading the same X, stage X into threadgroup memory once. Avoids each
SG re-reading X from device memory.

**When**: After B3, if you bumped to multi-SG threadgroups for B5 or to
better fill the GPU.

**Speedup**: 1.05–1.10× on bandwidth-bound GEMVs.

**Commit**: 86e542e.

### B5. Fuse residual into the GEMV epilogue (decode)

**What**: For `o_proj` and `expert_mix`, the next op is `residual_add`.
Read the residual in the same kernel that writes the output:
`Y[n] = bias[n] + dot(X, W[n]) + residual[n]`. Drops a dispatch + a
full pass over `residual`.

**When**: After B1–B3, on `o_proj` and `expert_mix`.

**Speedup**: 1.03–1.05× decode (saves 24 + 24 dispatches per token).
Smaller when concurrent encoder is already hiding the residual_add
dispatch (see `references/gotchas.md` #21).

**Snippet**: `patterns/gemv_with_residual_epilogue.metal`

**Commits**: 130075a, 4e3bb14, ba83808.

---

## C. GEMM — for prefill-time Linear layers

When `M = Lq > 1` (prefill), GEMV becomes GEMM. Different techniques
apply.

### C1. `simdgroup_matrix` MMA tile

**What**: Apple GPUs (M2+) have hardware matrix-multiply units exposed
via `simdgroup_matrix<T, 8, 8>` in MSL. Each `simdgroup_multiply_accumulate`
does an `8×8 × 8×8 → 8×8` matmul in 8 cycles. Tile the GEMM so each
SIMD group computes a `BM × BN` output tile via these primitives.

**When**: Prefill GEMM is hot. Naive 1-thread-per-output GEMM is
~50–100× slower.

**Speedup**: 3–10× on prefill GEMMs.

**Snippet**: `patterns/gemm_simdgroup_matrix_mma.metal`

**Commits**: dd9aee3 (vendor MLX steel headers), 28afb2e (POC),
3fce46d (qmm_t_gather_rhs MMA), 6316747 (BM=16 BN=32 WM=1 WN=2).

### C2. Tile-size sweep (BM, BN, BK, WM, WN)

**What**: With `simdgroup_matrix`, you tile across three axes — output
rows (BM), output cols (BN), and reduction depth (BK) — and within
each threadgroup further across SIMD-groups (WM × WN). The full knob
set is `(BM, BN, BK, WM, WN)`. Best tile depends on the matmul shape
AND the hardware.

- The `gpt-oss` reference impl found `BM=16 BN=32 BK=8 WM=1 WN=2`
  matches MLX's "non-NAX" tile for the gpt-oss MoE shapes.
- The Dream-7B reference port found `BM=16 BN=64 BK=16 WM=1 WN=2` for
  refine (M=16) and `BM=32 BN=64 BK=8 WM=1 WN=2` for prefetch (M=89)
  on M4 Max. BK=16 at BM=32 spills registers (TM=4 doubles the A
  footprint).

**When**: After C1. Sweep `(BM, BN, BK, WM, WN)` and pick the fastest
for each kernel shape.

**Speedup**: 1.10–1.30× per kernel.

**Knobs and gotchas**:

- **BM**: output rows per TG. Larger BM amortises W streaming (W is
  loaded once per row-band) but TM = BM/(WM*8) registers grow as `TM²`.
- **BN**: output cols per TG. TN = BN/(WN*8). Larger BN amortises X
  streaming but TN registers grow as `TN²`. TN=8 overflows on M4 Max
  for bf16 (port-c-to-metal target).
- **BK**: K-bands processed per loop iter. BK=8 is one 8×8 MMA per K
  step; BK=16 unrolls two 8×8 MMAs per step. Halves the K-loop count
  and improves ILP at the cost of doubling A/Bt mma temp registers.
  Worked at BM=16 in our reference port; at BM=32 the A footprint
  (`TM × (BK/8)` mma tiles) overflowed. The win was 5–8% on refine.
- **WM**: SIMD-groups stacked in M. Forces Bt to be redundant across
  WM SGs (each loads the same Bt tile). Usually pick `WM=1` so that A
  is shared (small) and Bt is split (large).
- **WN**: SIMD-groups stacked in N. Forces A to be redundant across
  WN SGs. Pick `WN=2` for 64 threads/TG.

**Dual-tile dispatch**: If your model has two distinct `M` values
(e.g., diffusion-LLM prefetch M=89 vs refine M=16), compile two PSOs
with different `BM`/`BK`. Pick at dispatch time:

```c
gpu_pipeline* pso = (M > 16) ? pso_gemm_bm32 : pso_gemm_bm16_bk16;
```

Buffer padding must round up to the larger BM (use `(L + 31) & ~31`
if any path is BM=32) so out-of-range row loads stay in-buffer.

**Commits**: 6316747 (gpt-oss), 697baaa + 80f26b2 (Dream-7B dual-tile +
BK=16).

### C3. Template kernels over fused-op flags

**What**: Same GEMM with optional bias / residual / dequant / different
output dtype. Use Metal function constants or C++ templates in MSL to
generate variants without duplicating code.

```metal
template <bool DO_ADD>
[[host_name("gemm_bf16")]]        // variant 1: no add
[[host_name("gemm_bf16_add")]]    // variant 2: + residual
kernel void gemm_bf16(...) { /* if (DO_ADD) y += residual; */ }
```

**When**: When you start fusing prefill epilogues.

**Commit**: 4e3bb14.

### C4. Aspect-ratio dispatch — route K > N matmuls to the SMALLER BM

**What**: Within a single M shape, route by (K, N) aspect ratio.
The same BM=32 BN=64 tile that's optimal for `N ≥ N_min` shapes
under-occupies the GPU when `N < N_min`. For Dream-7B's down_proj
(K=18944, N=3584) at BM=32 BN=64, you get only N/BN = 56 TGs — below
M4 Max's ~80 TG-slot capacity — and effective bandwidth collapses
to 125 GB/s vs 314 GB/s on gate_proj of the same matrix size.

```c
// d_linear dispatcher with aspect-ratio routing:
int use_bm32 = (M > 16) && (K <= N);    // K > N → BM=16 even at M>16
const uint32_t BM = use_bm32 ? 32u : 16u;
gpu_pipeline* pso = use_bm32 ? pso_gemm_bm32 : pso_gemm;
```

The trigger condition: `N < N_tiles_min × BN` where `N_tiles_min` is
what you need to fill GPU SG-slots. For M4 Max ~80 TG slots at
WM×WN = 2 SGs/TG, that's N ≲ 5120 at BN=64. Going BM=16 doubles
M-bands and so doubles TG count (56 → 112), restoring memory-level
parallelism. Yes, this also doubles W bytes read from DRAM (the
M-band W stream is repeated), but at the higher TG count effective
bandwidth recovers to 350+ GB/s and the net wall time DROPS.

**When**: After C1+C2. KPROF any matmul that takes >2× longer than
another of the same total matrix size — the slow one is almost
certainly K >> N and TG-starved.

**Speedup**: 12% wall on Dream-7B BL=32 refine (1.72s → 1.52s) from
a single-line dispatch change. The matmul itself dropped from
1082us to 741us (-31%). Generalizes to any LLaMA-style MLP
down_proj where the intermediate dimension I is large enough that
K = I makes (M > 16 → BM=32) under-occupy.

**Don't generalize "smaller BM always wins"**: the N >> K cases
(gate_proj, lm_head, qkv when num_heads × head_dim >> hidden) are
already well-occupied at BM=32 and would slow down at BM=16. The
rule is *use smaller BM only when N alone can't fill the GPU*.

**Commit**: see Dream-7B `src-metal/main.c` `d_linear_full_off`.

---

## D. SDPA — attention

### D1. Parallel softmax via SIMD reductions

**What**: Each thread in a SIMD group computes scores for a subset of
keys. Reductions (max for stability, sum for normalization) via
`simd_max` and `simd_sum` intrinsics — one instruction, no threadgroup
memory.

**When**: SDPA is in the top-3 hottest kernels. Naive serial softmax is
the bottleneck.

**Speedup**: 1.10–1.15× decode (per-lane score + simd_max/sum).

**Snippet**: `patterns/sdpa_sg_per_head_decode.metal` — D1 is the
softmax phase of that kernel. (The D1/parallel-softmax technique is
the necessary precursor to D2/D3, so it's baked into the worked example
rather than shipped as its own file.)

**Commit**: ff5ad86.

### D2. Online (one-pass) softmax with rolling max

**What**: Instead of two passes (find max → exp/sum → normalize), do one
pass: maintain a rolling `(max, sum, weighted_sum_of_V)`. When a new
score is higher than current max, rescale the accumulator. Cuts memory
traffic on K by 2×.

**When**: After D1. Especially helpful when K-cache is long.

**Speedup**: 1.03–1.05× decode.

**Snippet**: `patterns/sdpa_online_softmax.metal`

**Commit**: 2d59eed.

### D3. Multi-simdgroup SDPA per head

**What**: Split each `(query_token, head)` work across multiple SIMD
groups (e.g., 4 or 32 SGs per head). Each SG handles a K-stripe; final
reduction merges the per-SG online-softmax states via the
"online-softmax merge" formula.

**When**: After D2, when sequence is long enough to keep multiple SGs
busy per head.

**Speedup**: 1.05–2.0× decode (cd2ec6b +5%, 11b7562 +96%, 2d59eed +3%).

**Snippet**: `patterns/sdpa_multi_sg_online_merge.metal`

**Commits**: 11b7562 (1.96× decode), cd2ec6b (4 SGs/head),
2d59eed (32 SGs + transpose-merge), b2c8736 (Qwen3.6 N_SG=4 + online
merge, 52 → 65 tok/s).

### D4. Sliding-window / rotating fixed-size KV cache

**What**: For sliding-window layers (window W ≪ MAX_CTX), the KV cache
only needs W slots. Use a ring buffer with circular indexing in the SDPA
kernel. Saves memory AND cache bandwidth.

**When**: After D1–D3, if the model has sliding-window attention.

**Speedup**: 1.05–1.20× decode for sliding layers (varies with W).

**Reference doc**: `references/sliding-kv-cache.md` — describes index
math.

**Commits**: b09f12b, fc59ccf.

---

## E. MoE — typically the biggest bottleneck

Two completely different paths for decode (`Lq=1`) and prefill (`Lq>1`).

### E1. Decode MoE: qmv4 (register-tile) on reduced-precision (quantized) weights

**What**: Same as B3 but applied to the MXFP4 (or whatever
precision-reduction format) MoE GEMV. Each SG computes K outputs from
the same X, dequantizing K-blocks inline as you go. The K_TOP active
experts are gathered per token; only those K_TOP × N rows of weights
are touched.

**When**: First MoE optimization for decode.

**Speedup**: 2–4× decode (mxfp4 inner-loop unrolling + qmv4 + reg cache
combined).

**Snippet**: `patterns/mxfp4_qmv4_decode.metal`

**Commits**: 86e542e (share X via threadgroup mem), eee8282
(fully unrolled inner loop with vector loads), f536ba1 (SG-per-output
for mxfp4 expert GEMV), b69d31b (register-cached X qmv4 for mxfp4).

### E2. Fused gate_up_swiglu

**What**: The reference does `gate, up = gate_up_proj(x); mid =
swiglu(gate, up)`. The fused kernel does both matmuls and the SwiGLU
in one launch, writes only `mid` (never materialises `gate` and `up`
as separate buffers).

**When**: After E1. Saves 2 dispatches × N_LAYERS per token.

**Speedup**: 1.05–1.10× decode.

**Snippet**: `patterns/mxfp4_gate_up_swiglu_fused.metal`

**Commit**: b15e957.

### E3. Prefill MoE: sorted-gather grouped GEMM

**What**: For `Lq > 1`, bucketing tokens by expert lets you run **one
dense reduced-precision GEMM per expert** (no per-row gather inside the
inner loop). The flow is:

1. `expert_bucketize` — scatter `(token_idx, kt)` into per-expert lists
2. `moe_flatten_buckets` — prefix-sum, sorted flat lists, reverse map
3. `moe_gather_x_sorted` — gather X rows into expert-sorted layout
4. `qmm_t_gather_rhs_*` (× 2: gate_up, down) — one MMA reduced-precision
   GEMM per layer covering all experts
5. `moe_swiglu_epilogue` — clamp + sigmoid + (u+1) SwiGLU
6. `moe_combine_scatter` — weighted sum + residual, scatter back

**When**: When prefill is dominated by per-token expert gather (almost
always in MoE models with `Lq > 4`).

**Speedup**: 1.30–1.60× prefill (264→608 tok/s in mlx-cpp; +32% in
src-metal at first wire-up, more with MMA).

**Snippets**: `patterns/moe_sorted_gather_glue.metal` (the 4–5 small
glue kernels) + `patterns/moe_sorted_gather_qmm.metal` (the MMA
reduced-precision GEMM).

**Commits**: 1c4ddfd (264→348 +32%), 37ad895 (348→363, M_GROUP),
51ced6b (sorted-gather MMA wire +24%), 7691d2f (mlx-cpp 387→608),
3fce46d (qmm_t_gather_rhs MMA correctness).

### E4. Fuse combine_scatter + residual

**What**: The expert-output scatter back to `[L, HIDDEN]` is followed by
a residual add. Fuse them: `combine_scatter` reads the residual and
writes `out + residual`. Drops a dispatch.

**When**: After E3.

**Speedup**: 1.02× prefill.

**Commit**: ba83808.

---

## F. Reductions — small but pesky kernels

### F1. Parallel argmax

**What**: Naive argmax over VOCAB=200k is single-threaded → it's THE
bottleneck of decode if you don't parallelize. Solution: each SG
computes argmax over a stripe, threadgroup-level reduction with simd
intrinsics + threadgroup memory.

**When**: As soon as decode tok/s is in the right ballpark but
argmax shows up as a fat slice in the profile.

**Speedup**: 1.5–2.0× decode (e34bf80: 40→84 tok/s, +110%).

**Snippet**: `patterns/argmax_parallel.metal`

**Commit**: e34bf80.

### F2. Parallel topk_softmax (router)

**What**: SIMD-group-per-row router softmax + top-K. Each lane holds
E/SIMD logits in registers; one simd_max for softmax; K iterations of
(simd_max, simd_min tie-break, simd_broadcast winner) to extract top-K
without any threadgroup memory or barriers.

**When**: As soon as decode is mid-pipeline and softmax_topk shows up
in KPROF. The naive 1-thread-per-row softmax_topk over even just E=32
experts costs >100 µs/token of dispatch+serial work.

**Speedup**: 1.10–1.25× decode on Qwen3.6-35B-A3B (41.5 → 52.0 tok/s).
Bigger for larger E or K.

**Snippet**: `patterns/topk_parallel_router.metal`

**Commit**: 2177f38 (Qwen3.6 parallel top-K, +25%).

### F3. Per-row softmax + argmax + confidence (diffusion samplers)

**What**: For each of `M` logits rows, compute `(argmax_idx,
softmax_confidence)` in a single kernel. Output is just `M × 8 bytes`.
One threadgroup per row, two-pass:

1. Find `max + argmax` over V (SG-wide `simd_max`, then cross-SG via
   TG-mem reduction with `simd_min` on lane indices for stable
   tiebreak).
2. Sum `exp(logit - max)` over V (`simd_sum` + cross-SG reduction);
   `conf = 1 / sum_exp`.

**When**: Your sampler is a diffusion-LLM block-refinement sampler
(Dream / LLaDA / fastdllm) where each refine step needs the confidence
of every position to decide which mask tokens to commit. The host
version of this loop ran `BL × V × 3` scalar ops per step on the CPU
and showed up as a large `host_post` time invisible to `gpu_busy`.

In one port the host loop was `host_post = 0.25 s` out of `wall =
2.90 s`; replacing it with this kernel dropped `host_post` to `0.00 s`
and `wall` to `2.63 s` — closing more than half the remaining gap to
MLX in one commit.

**Speedup**: For diffusion samplers, 1.05–1.15× depending on
`BL × V × steps`. Negligible for autoregressive decoders (they read
one row at a time; F1 already covers that case).

**Snippet**: `patterns/softmax_argmax_per_row.metal`

**Commit**: df3d9d1 (Dream-7B fastdllm reference port).

---

## G. Decode/Prefill split — only when techniques diverge

Some optimizations are great for decode but bad for prefill (or vice
versa). When the optimal kernel differs, **maintain two implementations**
and dispatch the correct one based on `Lq`.

**When**: When a single kernel can't hit both targets.

**Pattern**: `if (Lq > 1) dispatch(prefill_kernel); else dispatch(decode_kernel);`

**Reference doc**: `references/decode-prefill-split.md` — table of
natural splits and a worked main.c snippet.

**Commits**: most of the gpt-oss reference impl optimization series
builds on this.

---

## H. Host/GPU boundary cleanup — often the biggest single win

### H1. Eliminate host-side breaks via "glue" kernels

**What**: A naive port `commit_wait`s + opens a fresh cmdbuf every
time the host does a small scratch op inside `forward()` — conv state
shift, broadcast-multiply with a scalar gate, channel split, KV cache
append, etc. Replace each one with a 5–10 line Metal kernel so the
WHOLE `forward()` runs in ONE command buffer.

**When**: ANY time you see `commit_wait` followed by a `for`-loop +
`memcpy` on the host inside `forward()`. Audit aggressively.

**Speedup**: 1.5–2.0× decode. In the Qwen3.6 port this was the
SINGLE LARGEST optimization in the entire pipeline (decode
19 → 34 tok/s, 1.79×). Each commit_wait costs 0.3–1.0 ms even for
tiny cmdbufs; on a 40-layer model with 3 breaks per layer, that's
40–120 ms / token of fixed overhead REGARDLESS of GPU kernel speed.

**Signal that this is your bottleneck**: after batched dispatches
(A1), `gpu_busy << wall`. The gap is host overhead.

**Snippet**: `patterns/glue_kernels_no_host_break.metal`

### H2. Merge embed + forward + argmax into one cmdbuf per step

**What**: Make `forward()` take `cb` as a parameter (don't own /
commit it). Then the decode loop builds:

```
cb = new_cmdbuf()
dispatch_embed(cb, ids_buf, ...)
forward(cb, q_off, 1)
dispatch_argmax(cb, logits, next_id_buf, ...)
commit_wait(cb)
```

Was 3 commit_waits per token, now 1.

**When**: Always, after H1.

**Speedup**: 1.02–1.05× decode (a few ms saved per step).

### H3. Prune unused output rows in the final linear

**What**: When the host only consumes a small subset of the final
linear's output rows (e.g., a diffusion-LLM prefetch reads ONE
`lm_head` row out of `L=89`), dispatch that final linear with `M`
restricted to the rows the host actually needs. The trick is to pass a
byte-offset into the input buffer so you reuse the same kernel:

```c
// Compute lm_head for just one row of h_buf, output to logits_buf[0].
size_t x_off = (size_t)needed_row * H * sizeof(bfloat);
gpu_arg_buf args[] = {
    { h_buf, x_off },  // <-- offset into the same buffer
    ARG(W), ARG(B), ARG(logits_buf), push_params(&pp, 1, K, N), ...
};
gpu_cmdbuf_dispatch(cb, pso_gemm, args, n, ...);
```

The kernel reads from `h_buf + x_off` as if it were the start of the
input, computes `M=1` rows, and writes to `logits_buf[0]`. The host
reads from row 0, skipping the per-row shift logic.

**When**: The final linear is huge in `N` (e.g., `V=152064` for
`lm_head`) AND the host only consumes a few rows. Most autoregressive
decoders already only do `M=1`, so this is irrelevant there — it's a
diffusion-LLM specific win.

**Speedup**: Saves `(M_full / M_used) - 1` weight-streaming passes of
`W`. For `lm_head` with `N=152k K=3584` at `BM=32`, dropping
M=89 → M=1 saves ~3 row-bands of streaming ≈ 2 GB. Wall savings:
20–50 ms per prefetch call on M4 Max.

**Caveat**: The semantics of "what row is needed" may depend on
sampler-specific shift conventions. Trace the host code carefully —
the off-by-one between "logits row for position p comes from hidden
state at position p-1" trips up most refactors.

**Snippet**: `patterns/gemm_x_offset_row_prune.c` — shows the
`d_linear_full_off` host-side dispatch helper.

**Commit**: 304484b (Dream-7B fastdllm reference port).

---

## I. Recurrent / state-update kernels

### I1. SG-per-(state_row, state_dim) for RNN/SSM steps

**What**: For models with a per-token state recurrence (GatedDeltaNet,
Mamba SSM), the naive port has 1 thread per `hv` (or per head). For
Hv=32, that's 1 SIMD-group on the whole GPU — 2.5% utilization.

Parallelize over BOTH state-row (`hv`) and state-dim (`dv`):
one SG per `(hv, dv)`. Lanes inside the SG cover the inner `dk`
dimension (Dk/32 elements per lane in registers); reductions are
`simd_sum`. Saturates the GPU.

**When**: First optimization for any RNN/SSM step.

**Speedup**: 2–3× decode on models with such layers. In Qwen3.6:
decode 5 → ~15 tok/s.

**Snippet**: `patterns/recurrent_state_sg_per_row.metal`
