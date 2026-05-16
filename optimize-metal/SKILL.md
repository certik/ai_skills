---
name: optimize-metal
description: >
  Optimize a working but naive Apple-GPU Metal LLM implementation (from
  the port-c-to-metal skill, in ./src-metal/) to match MLX speed within
  ±5%. Catalog: SIMD-group-per-output GEMV, bfloat4 loads, qmv4 register
  tiling, simdgroup_matrix MMA for prefill, sorted-gather MoE, online-
  softmax SDPA, multi-SG SDPA, parallel argmax, 2-deep cmdbuf pipeline,
  persistent + const param buffers, concurrent encoder, fused residual
  epilogues, decode/prefill split, GPU glue kernels for host-side
  breaks (often the biggest single decode win), SG-per-(row,dim) for
  RNN/SSM recurrences. Validates tokens against the C reference after
  each change. Triggers: optimize metal, speed up metal, match mlx
  speed, gpu optimization, metal performance, kernel tuning.
---

# optimize-metal — make the Metal port fast (match MLX ±5%)

Take a working but slow Metal LLM implementation (`./src-metal/`, produced by
`port-c-to-metal`) and iteratively optimize it until decode and prefill
throughput are within ±5% of the MLX-equivalent reference on the same
machine.

The optimization is **not** a single rewrite. It is a disciplined
iteration: profile, identify the top-1 bottleneck, look up the next
applicable technique in the catalog below, apply it, validate against
the C reference, commit. Repeat.

## When to use

After `port-c-to-metal` has produced a correct-but-slow `./src-metal/`. Use
this skill to:

- Profile the Metal kernels and identify per-kernel bottlenecks.
- Apply named optimization techniques (catalog in this file) and verify
  correctness after each.
- Drive performance to within ±5% of the MLX reference.

**Do not** use this skill to fix correctness bugs. If a kernel produces
wrong output before optimization, fix that in `port-c-to-metal` first.

## Pipeline

```
build-c-reference  →  port-c-to-metal  →  optimize-metal
                                            (this skill)
```

## Prerequisites

- `./src-metal/` exists, builds, and produces tokens that match `./src-cpu/`.
- `./src-cpu/` (the C reference) is **kept untouched** — it remains the
  ground truth for correctness.
- A way to run the MLX reference (`infer_mlx.py` or `mlx_lm`) for tok/s
  comparison.
- The model + a fixed validation prompt (same one used in earlier skills).

## Inputs to gather

1. Path to `./src-metal/` (default: sibling of this skill's working dir).
2. The MLX reference command to compare against, e.g.
   `uv run python infer_mlx.py --prompt "<P>" --max-tokens 256`.
3. Target machine spec (M1/M2/M3/M4/Pro/Max/Ultra) — affects tile sizes
   and SIMD-group widths.

## Success criteria

- Decode tok/s within ±5% of MLX on the same prompt + machine.
- Prefill tok/s within ±5% of MLX on the same prompt.
- First N generated tokens (N ≥ 4, ideally 16) match the C reference
  exactly. Later tokens may diverge due to pure numerical drift; that
  is acceptable as long as the divergence is *purely numerical* (no
  bug) and you can demonstrate it via per-kernel tolerance checks
  against the C reference.
- All optimizations committed individually so any one can be reverted.

## Workflow

Single, repeating cycle:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Profile current build                                        │
│    - measure decode + prefill tok/s                              │
│    - per-kernel GPU time (KPROF=1 idiom — one cmdbuf      │
│      per dispatch + timestamps)                                  │
│    - identify hottest kernel (typically: GEMV for decode,        │
│      GEMM for prefill, MoE for both)                              │
│                                                                  │
│ 2. Pick next technique                                           │
│    - look up the bottleneck kernel category in the catalog      │
│    - choose the next applicable, not-yet-applied technique       │
│                                                                  │
│ 3. Implement the technique on a branch / WIP commit             │
│    - apply ONLY the one technique                                │
│                                                                  │
│ 4. Validate                                                      │
│    - per-kernel correctness against C reference (tolerance ~1e-2 │
│      abs in bf16)                                                │
│    - end-to-end tokens against C reference for fixed prompt      │
│    - first N tokens must match exactly                           │
│    - if regression: revert; try a different technique            │
│                                                                  │
│ 5. Measure                                                       │
│    - if speedup ≥ 1.02× on the dominant phase (decode or prefill)│
│      it's a win — commit                                         │
│    - if neutral or slowdown: revert                              │
│                                                                  │
│ 6. Check stop condition                                          │
│    - if within ±5% of MLX on both prefill and decode → done      │
│    - else: back to step 1                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Setup — profiling infrastructure

**Always track `g_gpu_time` per cmdbuf and print it alongside wall time:**

```c
// metal_shim already sums (GPUEndTime - GPUStartTime) into g_gpu_time.
// Reset before timing, then print BOTH wall and gpu_busy:
g_gpu_time = 0.0;
clock_gettime(...t0);
// ... run prefill or decode loop ...
clock_gettime(...t1);
double wall = (t1 - t0);
fprintf(stderr, "decode: %.2fs (%d toks; %.1f tok/s; gpu_busy=%.2fs)\n",
        wall, n_gen, n_gen / wall, g_gpu_time);
```

The `wall - gpu_busy` gap tells you whether to chase kernel speed or
host-side scheduling:

- `gpu_busy ≈ wall`  → you're GPU-bound; optimize kernels (B, C, D, E, F, I).
- `gpu_busy << wall` → you're CPU-bound; optimize scheduling (A, H).

The single biggest win in a typical naive port comes from eliminating
host-side breaks (H1 / glue kernels) — you can see this immediately as
`gpu_busy/wall` ratio jumping from ~30% to ~95%.

KPROF (per-kernel timing) is OPTIONAL — see GOTCHAS #11. For a model
whose architecture you already know (you wrote the C ref), you can
predict the top hot kernels from first principles: largest GEMV
(lm_head, expert weights) and the SSM step if present. Spend the
KPROF effort only if your profile genuinely surprises you.

For end-to-end timing, use `mlx_lm`-equivalent timing (`tps =
(n-1) / decode_time`, exclude the first decode step) so you can compare
apples to apples with MLX. AND use ≥64 tokens AND best-of-3 runs:
short runs on busy machines have ±20% variance. See GOTCHAS #6.

## The Optimization Catalog

Each entry describes:
- **What** the technique does
- **When** to apply it (signal in the profile)
- **Expected speedup** (typical, from the gpt-oss commits)
- **Snippet**: pointer to a worked example in `patterns/`

The techniques are ordered roughly by "biggest first wins". You'll
typically want to apply them in this order, but the profile is the
ground truth — go where the heat is.

---

### A. Host-side scheduling — apply first, costs nothing in correctness

#### A1. Batch all dispatches per forward() pass

**What**: For each `forward()` call, open ONE command buffer, record all
~340 dispatches per token, then commit + wait once. Don't commit between
layers. Don't recreate pipelines / buffers per step.

**When**: Always, from the start. The naive port may already do this
(see `port-c-to-metal` Phase 3). If not, this alone is ~1.8× decode
speedup.

**Speedup**: 1.5–1.8× decode (per-dispatch overhead is ~30 µs at full
queue depth).

**Snippet**: `patterns/cmdbuf_batch_dispatches.c`

**Original commit**: a13d0de — *Batch dispatches per forward pass (1.78x decode speedup)*

#### A2. Persistent param buffers

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

**Original commit**: 2b1d6ef.

#### A3. CONST_PARAM_BUF — lift call-invariant constants

**What**: Some "params" are the same across every forward() call (e.g.,
the `dims` for an o_proj that has fixed shapes). Move them to a separate
const ring; only the per-call params (q_off, Lq) refill.

**When**: After A2, if you see param-buffer setup time non-trivial in
the encode-side profile.

**Speedup**: ~1.01×, but reduces CPU encode time materially when
pipelined.

**Snippet**: `patterns/param_buf_const.c`

**Original commit**: 8001db6.

#### A4. 2-deep command-buffer pipeline

**What**: While GPU runs decode step `k`, CPU encodes step `k+1` into a
fresh command buffer using a duplicated param ring (`slot ∈ {0,1}`).
At step `k+1`, swap roles. Two cmdbufs in flight at any time.

**When**: After A1–A3. The decode loop's GPU/CPU overlap is the limit.

**Speedup**: 1.10–1.20× decode (varies by kernel mix).

**Snippet**: `patterns/cmdbuf_pipeline_2deep.c`

**Original commit**: 1f1afbd.

#### A5. Concurrent encoder with explicit barriers

**What**: By default, Metal's compute encoder serializes every dispatch.
Switching to the *concurrent* encoder lets independent dispatches run in
parallel. Insert explicit barriers (`memoryBarrierWithScope:`) where
real data dependencies exist (e.g., between q_proj and rope_q).
Independent ops (q_proj/k_proj/v_proj or gate/up) may overlap.

**When**: After A1–A4, if profile shows GPU idle gaps between dispatches.

**Speedup**: 1.05–1.10× depending on overlap opportunity.

**Snippet**: `patterns/concurrent_encoder.c`

**Original commit**: d08def0.

#### A8. Parallel independent computation chains (END-GAME WIN)

**What**: With concurrent encoder enabled (A5), look for SUBGRAPHS that
share only an input buffer and merge only at a single fan-in. Two
independent chains (e.g., MoE main vs shared-expert in Qwen MoE
models, or gate vs up projection trees) can run **fully concurrently**
when you drop ALL barriers between them and re-add barriers only at
true fan-in points.

This is qualitatively different from A5: A5 overlaps pairs of
neighboring independent dispatches; A8 interleaves entire chains so
they execute as parallel pipelines.

**When**: Late-stage optimization, when you're at ~95% of MLX and
kernel-level techniques are exhausted. Required signal: `gpu_busy ≈
wall` (you ARE GPU-bound) AND your encoder graph shows two subgraphs
sharing only the input.

**Speedup**: 1.03–1.05× decode (the gap to MLX parity in Qwen3.6:
66.9 → 69.5 tok/s).

**Snippet**: `patterns/parallel_independent_chains.md`

**Original commit**: d813cf5 (Qwen3.6-35B-A3B: parallel MoE + shared
expert chains, MLX parity in 1 commit).

#### A6. JIT-compile each kernel from source

**What**: Apple's `MTLLibrary newLibraryWithSource:` JIT-compiles MSL.
Per-kernel sources can be compiled lazily on first dispatch and cached.
For many kernels, dedicate one combined library at startup (see
`kernel_concat.c`) so JIT happens once.

**When**: Always. The naive port already does this through `kernel_concat`.

**Speedup**: indirect — avoids first-dispatch hiccups in timing.

**Original commits**: a42da85, d460f8f, bd6dd1b, 2c40b09, 1ee3915, 039ae22, fc99872.

#### A7. `wired_limit` hint for working-set residency

**What**: `[MTLDevice setShouldMaximizeConcurrentCompilation:YES]` and
`[buffer setPurgeableState:MTLPurgeableStateNonVolatile]` keep the
weights resident in unified memory. The `recommendedMaxWorkingSetSize`
API gives you the residency budget.

**When**: If you see paging hiccups with large models.

**Speedup**: variable; eliminates worst-case stalls.

**Original commit**: 6361113.

#### A9. Parallel pread weight loader — beats mmap+memcpy 2.5–3×

**What**: At startup, weights are read from disk into MTLBuffers.
The naive "mmap shard + memcpy per tensor" is **page-fault bound** on
macOS (the VM lock serializes faults at ~1 GB/s, even with parallel
threads). The fix is to skip mmap for the bulk copy entirely and use
**parallel `pread()` per shard**:

1. Pre-allocate one MTLBuffer per tensor (serial — alloc isn't
   guaranteed thread-safe).
2. Bucket tensors by `shard_idx`, preserving order so each shard's
   reads are sequential on disk.
3. `dispatch_apply(n_shards)`: each iteration opens one fd and
   `pread`s its tensors directly into the matching MTLBuffer's host
   pointer (`gpu_buf_contents`).

Also: **delete any `madvise(MADV_WILLNEED)` calls** — on macOS that's
blocking, adding ~5s for 35 GB (see GOTCHA #28).

**When**: From the very first port — startup is the most user-visible
"wait" and it's trivial to fix.

**Speedup**: 28.5s → 5s startup (Qwen3.6-35B, M4 Max, 8 shards). This
**beats MLX's** ParallelFileReader on the same machine. See GOTCHAS
#28, #29.

**Snippet**: `patterns/load_parallel_pread.c`

---

### B. GEMV optimization — for decode-time Linear layers

These apply to every `linear_bf16` call where `M == Lq == 1` (decode):
q_proj, k_proj, v_proj, o_proj, router, lm_head.

#### B1. SIMD-group-per-output

**What**: One SIMD group (32 threads on Apple GPUs) computes one output
element of the GEMV. Each thread handles `K/32` of the inner-product;
final reduction via `simd_sum`. Instead of one thread doing K
multiply-adds serially, you have 32 threads doing K/32 in parallel.

**When**: Always, for any bf16 GEMV. Single-thread-per-output is ~30×
slower than this.

**Speedup**: 1.3–1.5× decode (5b7b16c gemv_bf16 = +38%).

**Snippet**: `patterns/gemv_simdgroup_per_output.metal`

**Original commit**: 5b7b16c.

#### B2. `bfloat4` vector loads

**What**: Load 4 bf16 values at a time using `bfloat4*` reinterpret
casts (or the equivalent `device const bfloat4*`). Halves memory
transactions.

**When**: After B1. K must be a multiple of 4 (true for HEAD_DIM=64,
HIDDEN=2880, INTER=2880, etc.).

**Speedup**: 1.10–1.15× on bandwidth-bound GEMVs.

**Snippet**: `patterns/gemv_bfloat4_loads.metal`

**Original commit**: 5b7b16c.

#### B3. Register-tile X across multiple outputs (qmv4 pattern)

**What**: Each SIMD group computes K outputs (e.g., 4 or 8) of the
GEMV. The X-vector is loaded **once** per SG and reused across the K
outputs (cached in registers). Cuts X bandwidth by Kx. Pattern matches
MLX's `qmv_fast`.

**When**: After B1, B2. Most GEMVs are X-bandwidth bound by then.

**Speedup**: 1.2–1.4× per GEMV; 1.10–1.20× overall decode.

**Snippet**: `patterns/gemv_qmv4_register_tile.metal`

**Original commits**: 9fce171 (gemv_bf16_4 — 4 outputs/SG), b7d7cf2
(8 outputs/SG), b69d31b (register-cached X for mxfp4 qmv4).

**Pitfall**: K_OUT has a sweet spot. For q8 affine dequant (4 bytes
unpacked + 4 dot-products per inner iter), K_OUT=4 wins; K_OUT=8
*hurts* (decode 41 → 31 tok/s in one port) because the inner loop
exceeds the register budget and the compiler spills. Default to 4
for q8/affine; revisit only after MMA has displaced GEMV.

#### B4. Share X via threadgroup memory

**What**: If you have multiple SIMD groups in one threadgroup all
reading the same X, stage X into threadgroup memory once. Avoids each
SG re-reading X from device memory.

**When**: After B3, if you bumped to multi-SG threadgroups for B5 or to
better fill the GPU.

**Speedup**: 1.05–1.10× on bandwidth-bound GEMVs.

**Original commit**: 86e542e.

#### B5. Fuse residual into the GEMV epilogue (decode)

**What**: For `o_proj` and `expert_mix`, the next op is `residual_add`.
Read the residual in the same kernel that writes the output:
`Y[n] = bias[n] + dot(X, W[n]) + residual[n]`. Drops a dispatch + a
full pass over `residual`.

**When**: After B1–B3, on `o_proj` and `expert_mix`.

**Speedup**: 1.03–1.05× decode (saves 24 + 24 dispatches per token).

**Snippet**: `patterns/gemv_with_residual_epilogue.metal`

**Original commits**: 130075a, 4e3bb14, ba83808.

---

### C. GEMM optimization — for prefill-time Linear layers

When `M = Lq > 1` (prefill), GEMV becomes GEMM. Different techniques apply.

#### C1. `simdgroup_matrix` MMA tile

**What**: Apple GPUs (M2+) have hardware matrix-multiply units exposed
via `simdgroup_matrix<T, 8, 8>` in MSL. Each `simdgroup_multiply_accumulate`
does an `8×8 × 8×8 -> 8×8` matmul in 8 cycles. Tile the GEMM so each
SIMD group computes a `BM × BN` output tile via these primitives.

**When**: Prefill GEMM is hot. Naive 1-thread-per-output GEMM is
~50–100× slower.

**Speedup**: 3–10× on prefill GEMMs.

**Snippet**: `patterns/gemm_simdgroup_matrix_mma.metal`

**Original commits**: dd9aee3 (vendor MLX steel headers), 28afb2e (POC),
3fce46d (qmm_t_gather_rhs MMA), 6316747 (BM=16 BN=32 WM=1 WN=2).

#### C2. Tile-size sweep (BM, BN, WM, WN)

**What**: With `simdgroup_matrix`, you tile across two axes — across
threadgroups (BM × BN), and within threadgroup across warps (WM × WN
SIMD-groups). Best tile size depends on the matmul shape and the
hardware. the gpt-oss reference impl found `BM=16 BN=32 WM=1 WN=2` matches MLX's "non-NAX"
tile sizes for the gpt-oss MoE shapes.

**When**: After C1. Sweep `(BM, BN, WM, WN) ∈ {(8,16,1,1), (16,16,1,2),
(16,32,1,2), (32,32,2,2), ...}` and pick the fastest for each kernel
shape.

**Speedup**: 1.10–1.30× per kernel.

**Original commit**: 6316747.

#### C3. Template kernels over fused-op flags

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

**Original commit**: 4e3bb14.

---

### D. SDPA optimization — for attention

#### D1. Parallel softmax via SIMD reductions

**What**: Each thread in a SIMD group computes scores for a subset of
keys. Reductions (max for stability, sum for normalization) via
`simd_max` and `simd_sum` intrinsics — one instruction, no threadgroup
memory.

**When**: SDPA is in the top-3 hottest kernels. Naive serial softmax is
the bottleneck.

**Speedup**: 1.10–1.15× decode (per-lane score + simd_max/sum).

**Snippet**: `patterns/sdpa_simd_softmax.metal`

**Original commit**: ff5ad86.

#### D2. Online (one-pass) softmax with rolling max

**What**: Instead of two passes (find max → exp/sum → normalize), do one
pass: maintain a rolling `(max, sum, weighted_sum_of_V)`. When a new
score is higher than current max, rescale the accumulator. Cuts memory
traffic on K by 2×.

**When**: After D1. Especially helpful when K-cache is long.

**Speedup**: 1.03–1.05× decode.

**Snippet**: `patterns/sdpa_online_softmax.metal`

**Original commit**: 2d59eed.

#### D3. Multi-simdgroup SDPA per head

**What**: Split each `(query_token, head)` work across multiple SIMD
groups (e.g., 4 or 32 SGs per head). Each SG handles a K-stripe; final
reduction merges the per-SG online-softmax states via the
"online-softmax merge" formula.

**When**: After D2, when sequence is long enough to keep multiple SGs
busy per head.

**Speedup**: 1.05–2.0× decode (cd2ec6b +5%, 11b7562 +96%, 2d59eed +3%).

**Snippet**: `patterns/sdpa_multi_sg_online_merge.metal`

**Original commits**: 11b7562 (1.96× decode), cd2ec6b (4 SGs/head),
2d59eed (32 SGs + transpose-merge), b2c8736 (Qwen3.6 N_SG=4 + online
merge, 52 → 65 tok/s).

#### D4. Sliding-window / rotating fixed-size KV cache

**What**: For sliding-window layers (window W ≪ MAX_CTX), the KV cache
only needs W slots. Use a ring buffer with circular indexing in the SDPA
kernel. Saves memory AND cache bandwidth.

**When**: After D1–D3, if the model has sliding-window attention.

**Speedup**: 1.05–1.20× decode for sliding layers (varies with W).

**Snippet**: `patterns/sdpa_rotating_kv_cache.md` — describes index math.

**Original commits**: b09f12b, fc59ccf.

---

### E. MoE optimization — typically the biggest bottleneck

Two completely different paths for decode (`Lq=1`) and prefill (`Lq>1`).

#### E1. Decode MoE: qmv4 (register-tile) on quantized weights

**What**: Same as B3 but applied to the MXFP4 (or whatever quant) MoE
GEMV. Each SG computes K outputs from the same X, dequantizing K-blocks
inline as you go. The K_TOP active experts are gathered per token; only
those K_TOP × N rows of weights are touched.

**When**: First MoE optimization for decode.

**Speedup**: 2–4× decode (mxfp4 inner-loop unrolling + qmv4 + reg cache
combined).

**Snippet**: `patterns/mxfp4_qmv4_decode.metal`

**Original commits**: 86e542e (share X via threadgroup mem), eee8282
(fully unrolled inner loop with vector loads), f536ba1 (SG-per-output
for mxfp4 expert GEMV), b69d31b (register-cached X qmv4 for mxfp4).

#### E2. Fused gate_up_swiglu

**What**: The reference does `gate, up = gate_up_proj(x); mid = swiglu(gate, up)`.
The fused kernel does both matmuls and the SwiGLU in one launch, writes
only `mid` (never materializes `gate` and `up` as separate buffers).

**When**: After E1. Saves 2 dispatches × N_LAYERS per token.

**Speedup**: 1.05–1.10× decode.

**Snippet**: `patterns/mxfp4_gate_up_swiglu_fused.metal`

**Original commits**: b15e957.

#### E3. Prefill MoE: sorted-gather grouped GEMM

**What**: For `Lq > 1`, bucketing tokens by expert lets you run **one
dense quantized GEMM per expert** (no per-row gather inside the inner
loop). The flow is:

1. `expert_bucketize` — scatter `(token_idx, kt)` into per-expert lists
2. `moe_flatten_buckets` — prefix-sum, sorted flat lists, reverse map
3. `moe_gather_x_sorted` — gather X rows into expert-sorted layout
4. `qmm_t_gather_rhs_*` (× 2: gate_up, down) — one MMA quantized GEMM
   per layer covering all experts
5. `moe_swiglu_epilogue` — clamp + sigmoid + (u+1) SwiGLU
6. `moe_combine_scatter` — weighted sum + residual, scatter back

**When**: When prefill is dominated by per-token expert gather (almost
always in MoE models with `Lq > 4`).

**Speedup**: 1.30–1.60× prefill (264→608 tok/s in mlx-cpp; +32% in src-metal
at first wire-up, more with MMA).

**Snippet**: `patterns/moe_sorted_gather_glue.metal` +
`patterns/moe_sorted_gather_qmm.metal`

**Original commits**: 1c4ddfd (264→348 +32%), 37ad895 (348→363, M_GROUP),
51ced6b (sorted-gather MMA wire +24%), 7691d2f (mlx-cpp 387→608),
3fce46d (qmm_t_gather_rhs MMA correctness).

#### E4. Fuse combine_scatter + residual

**What**: The expert-output scatter back to `[L, HIDDEN]` is followed by
a residual add. Fuse them: `combine_scatter` reads the residual and
writes `out + residual`. Drops a dispatch.

**When**: After E3.

**Speedup**: 1.02× prefill.

**Original commit**: ba83808.

---

### F. Reductions — small but pesky kernels

#### F1. Parallel argmax

**What**: Naive argmax over VOCAB=200k is single-threaded → it's THE
bottleneck of decode if you don't parallelize. Solution: each SG
computes argmax over a stripe, threadgroup-level reduction with simd
intrinsics + threadgroup memory.

**When**: As soon as decode tok/s is in the right ballpark but
argmax shows up as a fat slice in the profile.

**Speedup**: 1.5–2.0× decode (e34bf80: 40→84 tok/s, +110%).

**Snippet**: `patterns/argmax_parallel.metal`

**Original commit**: e34bf80.

#### F2. Parallel topk_softmax (router)

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

**Original commit**: 2177f38 (Qwen3.6 parallel top-K, +25%).

---

### H. Host/GPU boundary cleanup — often the biggest single win

#### H1. Eliminate host-side breaks via "glue" kernels

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

#### H2. Merge embed + forward + argmax into one cmdbuf per step

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

### I. Recurrent / state-update kernels

#### I1. SG-per-(state_row, state_dim) for RNN/SSM steps

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

---

### G. Decode/Prefill split — only when techniques diverge

Some optimizations are great for decode but bad for prefill (or vice
versa). When the optimal kernel differs, **maintain two implementations**
and dispatch the correct one based on `Lq`:

- `gemv_bf16_4_v8` for decode; `gemm_bf16` (MMA) for prefill.
- `mxfp4_gus_qmv4_bf16` for decode; `qmm_t_gather_rhs_*` for prefill MoE.
- `expert_mix_bf16_add` for decode (per-token K_TOP combine);
  `moe_combine_scatter` for prefill (sorted-scatter).

**When**: When a single kernel can't hit both targets.

**Pattern**: `if (Lq > 1) dispatch(prefill_kernel); else dispatch(decode_kernel);`

**Original commits**: most of the gpt-oss reference impl optimization series builds on this.

---

## Patterns / snippets

Each technique above points to a file in `patterns/`. These are
**inspiration**, not drop-ins — adapt to your model's shapes.

```
patterns/
├── GOTCHAS.md                              <-- READ THIS FIRST
├── cmdbuf_batch_dispatches.c
├── cmdbuf_pipeline_2deep.c
├── cmdbuf_pipeline_2deep_id_swap.c         <-- pipeline + id swap
├── concurrent_encoder.c
├── param_buf_persistent.c
├── param_buf_const.c
├── gemv_simdgroup_per_output.metal
├── gemv_bfloat4_loads.metal
├── gemv_qmv4_register_tile.metal
├── gemv_with_residual_epilogue.metal
├── gemm_simdgroup_matrix_mma.metal
├── sdpa_simd_softmax.metal
├── sdpa_sg_per_head_decode.metal           <-- D1 worked example
├── sdpa_online_softmax.metal
├── sdpa_multi_sg.metal
├── sdpa_rotating_kv_cache.md
├── mxfp4_qmv4_decode.metal
├── mxfp4_gate_up_swiglu_fused.metal
├── moe_sorted_gather_glue.metal
├── moe_sorted_gather_qmm.metal
├── argmax_parallel.metal
├── glue_kernels_no_host_break.metal        <-- H1 (often biggest win)
├── recurrent_state_sg_per_row.metal        <-- I1 for SSM/RNN
├── topk_parallel_router.metal              <-- F2 worked example
├── sdpa_multi_sg_online_merge.metal        <-- D3 worked example
├── parallel_independent_chains.md          <-- A8 (END-GAME WIN)
├── load_parallel_pread.c                   <-- A9 (startup beats MLX)
└── decode_prefill_split.md
```

Each pattern file contains:
- A header comment: what / when / expected speedup / original commit.
- A working kernel snippet, lightly annotated.
- Pointers to where to look in the gpt-oss repo if you want the full
  picture (commit SHA + file path).

## Validation discipline

**After every change**, run the same validation suite:

1. Per-kernel correctness: the test for the touched kernel must pass
   against the C reference (`src-cpu`) within tolerance.
2. End-to-end token match against the previous Metal commit on the same
   prompt. First N tokens (N ≥ 16 ideally) MUST match. If they don't,
   investigate before committing.
3. Tok/s measurement (decode + prefill) on a fixed prompt + max-tokens.
   Record before/after numbers in the commit message.
4. If regression on either tok/s or correctness: revert. Try a
   different technique or different tile size.

When tokens diverge beyond the first N but match for the first N: this
is *probably* pure numerical drift. To confirm: dump per-layer
intermediates from both the old commit and the new commit on the SAME
input ids; verify they agree to bf16 tolerance. If yes, the divergence
is purely numerical and acceptable. If no, you have a bug — revert.

## Stop condition

When both prefill and decode are within ±5% of MLX on the validation
prompt, you are done. Document the final tok/s and machine in a
`src-metal/PERF.md` and write a summary commit:

```
src-metal: optimization complete (prefill X.X tok/s vs MLX Y.Y, decode A.A vs B.B)
```

## Empirical attack order — what to actually try first

For a quantized MoE decoder LLM on Apple Silicon, this is the empirical
order of biggest wins from a naive port (validated on Qwen3.6-35B-A3B
on M4 Max, with the same patterns from gpt-oss work):

1. **Parallel argmax** (F1). 1-thread argmax over 200k+ vocab is the
   most common "why is decode capped at <5 tok/s" answer.
2. **SIMD-group-per-output for ALL quantized GEMV** (B1). Affects
   q/k/v/o_proj, lm_head, router, every MoE linear.
3. **SG-per-row for the reduction kernels**: rmsnorm, softmax_topk,
   embed_dequant_gather. Cheap.
4. **SG-per-(hv, dv) for the recurrent state step** (I1) if the model
   has GatedDelta / Mamba / SSM layers. 32 serial threads → 4096 SGs.
5. **SDPA SG-per-(lq, hq) with TG-mem scores** (D1 worked example).
   Necessary precursor to D2/D3.
6. **ELIMINATE HOST-SIDE BREAKS** (H1, glue kernels). Usually the
   single biggest jump in the whole pipeline. Often 1.5–2×.
7. **Merge embed + forward + argmax into one cmdbuf** (H2). Small win.
8. **2-deep cmdbuf pipeline with id-swap** (A4 variant). Closes the
   first CPU-encode gap to MLX.

The above sequence took Qwen3.6-35B-A3B from 1.9 → ~43 tok/s.

For the LAST 5-10% to MLX parity, you need:

9. **Parallel top-K** (F2 generalized) — same trick as parallel argmax
   but applied to router top-K over N_EXPERTS=32-128. Use simd_max +
   tie-break simd_min for K iterations, all in registers.
10. **Multi-SG SDPA with online-softmax merge** (D3). Once D1+D2 are
    in, splitting each (lq, hq) across N_SG=4 SIMD groups + merging
    per-SG (m_i, s_i, o_i) via the online-softmax formula is the
    decode SDPA endgame.
11. **TG=(1,1,1) → (32,1,1) bump** for all elemwise/glue kernels (see
    GOTCHA #17). Free win.
12. **Drop barriers between independent ops** (subset of A5): e.g.,
    q_norm/k_norm, rope Q/K, rmsnorm_scale q/k, sigmoid+compute_g.
13. **Fuse residual into LAST linear epilogue** (B5) — o_proj,
    out_proj, shared_combine_add. Drops dispatches.
14. **PARALLEL INDEPENDENT CHAINS** (A8). Restructure your encoder so
    multi-kernel subgraphs (e.g., MoE chain vs shared-expert chain in
    Qwen MoE) run as parallel pipelines, not in series. Final 3-5%
    to MLX parity — usually the BIGGEST single late-stage commit.

Steps 9-14 took Qwen3.6-35B-A3B from 41.5 → 69.5 tok/s, hitting MLX
parity (70.1 tok/s) on M4 Max.

Optimizations that *should* help but landed in the noise band on this
model (don't skip but don't expect miracles):
- bfloat4 vector loads in isolation (baked into B1 already).
- qmv4 K_OUT=4 register tile (q8 dequant is bandwidth-bound, not
  register-bound; K_OUT=8 actually regressed).
- Concurrent encoder ALONE without barrier surgery (most barriers
  already needed, A8 is where the big win lives).

Optimizations to skip in this order for decode-only goals:
- C1/C2/C3 (GEMM MMA for prefill) — only relevant if your goal is to
  match prefill speed, not decode.

## Late-stage barrier-removal checklist

When you're at ~95% of MLX and decode is GPU-bound (`gpu_busy ≈ wall`),
the remaining gap is almost always serialized concurrency. Walk through
this checklist for each layer type in your model:

**Attention layer**:
- [ ] `q_proj || k_proj || v_proj` (read same H; disjoint outputs)
- [ ] `q_norm || k_norm` (disjoint inputs and outputs)
- [ ] `rope_q || rope_k` (disjoint inputs and outputs)

**Linear-attention / SSM layer** (if present):
- [ ] `in_proj_qkv || in_proj_z || in_proj_b || in_proj_a` (fan-out)
- [ ] `silu_inplace(conv_out) || conv_state_update` (different buffers)
- [ ] `rmsnorm_scale_q || rmsnorm_scale_k || sigmoid(beta) || compute_g`
      (4-way concurrent, all different inputs/outputs)

**MoE layer (BIGGEST WIN ZONE)**:
- [ ] `moe.gate_proj || moe.up_proj` (read same H; disjoint outputs)
- [ ] `shared.gate || shared.up || shared_expert_gate` (same H, disjoint)
- [ ] **MoE main chain || shared-expert chain** end-to-end (A8) — this
      is usually the last 3-5% to MLX parity.

**Across-layer fusions**:
- [ ] Residual fused into `o_proj` epilogue (B5)
- [ ] Residual fused into `out_proj` epilogue (linear-attn B5)
- [ ] Residual fused into `shared_expert_combine` (B5 variant)

**Elemwise fusions**:
- [ ] `copy + sigmoid_inplace` → `sigmoid_bf16` (out-of-place)
- [ ] Any "init buffer; then modify in-place" pair → single oop kernel

For each item, verify with a token-id match against the prior commit
(`--max-tokens 16`). For each commit, run 5 times and use median to
distinguish a real 1-2% win from noise.

## Common pitfalls

See `patterns/GOTCHAS.md` for the full set with concrete fixes.
Highlights:

- **Premature multi-SG SDPA**: if you split a head across SGs before
  fixing the dumb 1-thread-per-output SDPA, you'll measure a tiny gain
  and conclude SDPA isn't worth optimizing. Fix the per-thread serial
  softmax first.
- **Tile size cargo-culting**: tile sizes that work for MLX on M2 may
  not be optimal on M4 or M1. Always sweep on your target machine.
- **Forgetting GQA in SDPA tile**: when each SG handles a `(query_token,
  head)` pair, the K head index is `h / (Nq/Nkv)`. Many tile
  refactorings break this and produce garbage.
- **Fused kernels diverging from C reference**: when you fuse residual
  into o_proj, the bf16 round point shifts (the residual is now added
  inside the SAME f32 accumulator vs as a separate bf16 round trip).
  Token output may differ. Verify the difference is within tolerance
  *and* that first N tokens still match.
- **bfloat4 alignment**: load `bfloat4` only at 8-byte-aligned
  addresses, or you'll get garbage on some Apple GPU generations.
- **Persistent param buffers + 2-deep pipeline**: if you have ONE persistent
  param buffer and two cmdbufs in flight, you have a race. Duplicate
  the param ring (slot ∈ {0,1}).
- **Argmax over VOCAB=200k single-threaded**: this is by far the most
  common "why is decode capped at 40 tok/s" answer. Apply F1 immediately
  once decode is in the 40+ tok/s ballpark.
- **`dispatchThreads:` takes TOTAL THREADS, not threadgroups**: passing
  `grid_x = N` with `tg_x = 32` launches `ceil(N/32)` SGs, not N. For
  SG-per-output kernels, you want `grid_x = 32 * N`. See GOTCHAS #2.
- **KV cache `max_ctx` and SDPA `MAX_CTX` must agree AND be ≥ Lp +
  max_tokens**: silent buffer overruns produce wrong tokens that look
  like "numerical drift" past the first ~32 tokens. See GOTCHAS #3.
- **`gpu_busy << wall` means CPU encode / host roundtrips are the
  bottleneck**, not GPU work. Look for `commit_wait` inside `forward()`
  and replace each host-side scratch op with a glue kernel (H1). This
  is usually the single biggest win in the whole pipeline.
- **The first generated token IS the prefill argmax**: when wiring up
  a 2-deep pipeline that primes itself, it's easy to skip emitting
  this token, shifting all output by 1. The "first N tokens match"
  check then fails for purely off-by-one reasons. See GOTCHAS #10.
- **K_OUT in qmv4 sweet spot is small** (4 for q8 affine dequant on
  M4). Bigger values increase register pressure and may *slow* you
  down — measure before scaling up. See GOTCHAS #8.
- **Per-file `constant constexpr` collisions** when kernels are
  concatenated into one library — prefix per-file (e.g.,
  `LQ8_SIMD_WIDTH`, `ARGMAX_TG_SIZE`). See GOTCHAS #4.
- **First cmdbuf wall ≠ first cmdbuf GPU time**: Metal warms up
  pipeline state objects, residency, etc. lazily on first dispatch
  (0.5–3s wall for nothing). Always report both wall and `gpu_busy`
  so you can tell the difference. See GOTCHAS #5.

## Debugging recipes

### "Tokens diverged after change X — how do I localize?"

1. Make sure the C reference (`./src-cpu/`) is still committed and
   produces the same tokens it always did.
2. Add a `--dump <dir>` flag to BOTH `./src-cpu/<BIN_CPU>` and
   `./src-metal/<BIN>` that writes every per-kernel input + output to
   `<dir>/<kernel>_L<layer>.bin` (using the same .bin format as
   `tools/dump_ref.py`).
3. Run both with the same prompt:
   ```
   ./src-cpu/<BIN_CPU>  --prompt "..." --dump src-cpu/refs
   ./src-metal/<BIN>    --prompt "..." --dump src-metal/refs
   ```
4. Binary-diff per kernel:
   ```
   for f in src-cpu/refs/*.bin; do
       name=$(basename "$f")
       cmp -l "$f" "src-metal/refs/$name" | wc -l
   done | sort -nr | head
   ```
   The first non-zero kernel in forward order is the culprit.

### "Decode is fast, prefill is slow"

You probably have not yet implemented the **prefill GEMM path** (MMA
tile) or the **sorted-gather MoE** for prefill (E3). Apply C1+C2 and E3.

### "Prefill is fast, decode is slow"

You probably have not yet applied **qmv4 register tile** (B3), **fused
gate+up+swiglu** (E2), or **parallel argmax** (F1).

### "Tok/s drops every few tokens"

Memory pressure / paging. Check `wired_limit` (A7) and verify weights are
zero-copy mmap'd (`gpu_buf_wrap_nocopy`).

### "Performance is bursty"

CPU encode is becoming the bottleneck. Apply 2-deep cmdbuf pipeline (A4)
+ persistent / const param buffers (A2 + A3).

### "Validated kernel-by-kernel but end-to-end tokens still diverge"

Check `barrier()` / cmdbuf ordering. With concurrent encoder (A5), a
missing barrier produces non-deterministic divergence that often passes
per-kernel tests (which serialize one kernel at a time).

## Commit strategy

One commit per technique. Each commit message should include:

```
src-metal: <technique short name> (decode +X.X%, prefill +Y.Y%)

- describe the change in 1-3 lines
- mention the bottleneck-before kernel
- mention the kernel-after improvement
- include before/after tok/s if material
```

Examples (from this repo):

```
src-metal: gemv_bf16_4 — 4 outputs/simdgroup, 64 threads/TG (mlx qmv_fast pattern)
SDPA: simdgroup-per-(head,token) parallelism (1.96x decode)
src-metal: BM=16 BN=32 WM=1 WN=2 — match MLX non-NAX tile sizes (+26%)
src-metal: pipeline decode (depth=2) — overlap CPU encode with GPU compute
decode: 8-X register tile in bf16 GEMV (q/k/v/o/router/lm_head)
src-metal: fuse res_attn into gemm_bf16 (template DO_ADD); drop residual_add kernel
```

These commit messages tell you exactly what was changed AND give the
expected speedup, so when you bisect a regression later you can find it
fast.

## Hand-off / wrap-up

When the skill is done:

1. `./src-metal/PERF.md` documents final tok/s vs MLX on the target machine.
2. The optimized `./src-metal/` still passes per-kernel correctness against
   `./src-cpu/` within tolerance.
3. End-to-end tokens match `./src-cpu/` for the validation prompt for
   at least the first 16 tokens.
4. Each optimization is its own commit, individually revertable.

Optional follow-ups (out of scope for this skill):

- Port the optimized src-metal/ to CUDA / ROCm / Vulkan / TPU (each its own
  future skill, starting from the C reference, not from this Metal
  implementation).
- Add batch>1 support.
- Add longer-context / paged-KV cache.
- Add sampling beyond greedy.
