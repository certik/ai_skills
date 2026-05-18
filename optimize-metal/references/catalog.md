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
- [B. GEMV (decode Linear)](#b-gemv--decode-time-linear-layers) — simdgroup-per-output, bfloat4, qmv4 register tile, TG-mem X share, residual epilogue, uint4 W loads w/ amortized (s,b) for q-affine dequant (B6)
- [C. GEMM (prefill Linear)](#c-gemm--prefill-time-linear-layers) — simdgroup_matrix MMA, tile sweep (incl. WN as register-pressure relief), fused-op templates, aspect-ratio dispatch (K>N AND small-N axes)
- [D. SDPA](#d-sdpa--attention) — parallel softmax, online softmax, multi-SG, K-per-iter ILP unrolling, sliding KV
- [E. MoE](#e-moe--typically-the-biggest-bottleneck) — decode qmv4, fused gate_up_swiglu, prefill sorted-gather, fused combine_scatter
- [F. Reductions](#f-reductions--small-but-pesky-kernels) — parallel argmax, parallel topk softmax, per-row softmax+argmax (diffusion samplers), multi-SG-per-row rmsnorm
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

### A7b. Async `MTLResidencySet` to hide first-cmdbuf residency tax (macOS 15+)

**What**: Pre-pay the first-cmdbuf residency-wiring cost on a
background queue while parallel pread runs. Replaces gotcha #30's
"this is a fixed cost" with an actual mitigation.

**Why**: The first cmdbuf to touch a multi-GB working set of
MTLBuffers blocks for ~80–100 ms inside its commit→GPUStart gap
(invisible to encoder timing — it shows up only as a `gpu_busy/wall`
ratio drop). On macOS 15+, `MTLResidencySet` + `requestResidency`
removes this. The blocking ~85 ms `requestResidency` call must run
on a `dispatch_async(QOS_CLASS_USER_INITIATED, ...)` queue so it
overlaps with the (typically longer) weight-pread phase. After
pread finishes, you `dispatch_group_wait` the residency block and
call `[queue addResidencySet:]` to bind it.

**When**: Always, on macOS 15+, for any LLM that loads weights from
disk into MTLBuffers and runs a forward pass shortly after. The
larger the model the bigger the win.

**Speedup**: gen wall -5%, GPU utilization 94% → 99%, total wall
-90 ms on Dream-7B (M4 Max). Cost is fully hidden inside pread.

**Catch**: `addAllocations:count:` takes a C array of
`id<MTLAllocation>`; under ARC, declare it `__strong id<MTLAllocation>*`
(see gotcha #43). `id<MTLBuffer>` conforms to `id<MTLAllocation>` on
macOS 15+. Skeleton in gotcha #30.

**Commit**: 87e9e02.

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

1. Pre-allocate one MTLBuffer **per shard** (not per tensor — see the
   "shard-sized MTLBuffer + per-tensor views" variant below, gotcha #50).
2. Bucket arrays by `shard_idx`, preserving order so each shard's
   reads are sequential on disk.
3. `dispatch_apply(n_shards × SHARD_SPLIT)`: each iteration opens one
   fd, takes a *contiguous slice* of that shard's tensor list, and
   `pread`s sequentially. Multiple workers per fd is OK as long as
   each worker's reads are sequential within its slice (gotcha #29
   warns about scattered offsets inside one fd disabling readahead).

`SHARD_SPLIT` must be **co-tuned** with bg compile + residency_async
threads: target `n_shards × SHARD_SPLIT ≈ P_cores`. On M4 Max
(12 P-cores, 4 shards) `SHARD_SPLIT=3` is optimal; `SHARD_SPLIT=4`
regresses because bg compile starves. See gotcha #45 for the table.

**Companion wins (apply together for full startup savings)**:

- **Drop `madvise(MADV_WILLNEED)`** — it's blocking on macOS, costs
  ~340 ms for 14 GB. Gotcha #28.
- **Start compile async** before `cache_weights()` via
  `gpu_init_async()` + `gpu_init_finish()`. The 74 ms first-ever
  compile hides inside the pread tail; warm-run `lib_join` is 0.
  Use `QOS_CLASS_USER_INITIATED` (not `USER_INTERACTIVE` — gotcha
  #46). -20 ms on cold compile, free on warm.
- **Async `MTLResidencySet requestResidency`** on a bg dispatch
  queue right after MTLBuffer allocation, overlapping with pread.
  Hides the ~85 ms residency-wiring tax. Gotcha #30.
- **Shard-sized MTLBuffer + per-tensor views** (one MTLBuffer per
  safetensors shard, each tensor is a `(parent, offset, size)`
  wrapper). Residency set shrinks N_tensors → N_shards (e.g.
  339 → 4 on Dream-7B). Gotcha #50.

Also: **delete any `madvise(MADV_WILLNEED)` calls** — on macOS that's
blocking, adding ~5s for 35 GB (see `references/gotchas.md` #28).

**Diagnostic — phase-breakdown line**: before applying A9 (and after,
to confirm), log a one-liner separating tokenizer / config / metal /
weights / total. If `weights` is 30%+ of `total`, A9 will pay off
significantly:

```
[startup] tokenizer=0.003s config=0.000s metal=0.017s lib_join=0.000s weights=0.340s total=0.398s
```

**Why not zero-copy `newBufferWithBytesNoCopy:` per tensor?** Apple's
API requires the host pointer AND length to be `vm_page_size`-aligned
(16 KB on Apple Silicon). Inside a safetensors shard, individual
tensor offsets are 8-byte aligned (set by the JSON header layout), so
per-tensor wrap fails (gotcha #40).

**Why not zero-copy `newBufferWithBytesNoCopy:` per shard?** The shard
*is* page-aligned (mmap guarantees it), so the API call succeeds —
but on warm cache, the page-fault path that materializes the buffer
on first GPU touch is ~20 GB/s, slower than pread's page-cache memcpy
fast path at ~50 GB/s. Counter-intuitively, "zero copy" is slower
here. Gotcha #47.

**When**: From the very first port — startup is the most user-visible
"wait" and it's trivial to fix.

**Speedup**:
- 28.5 s → 5 s startup (Qwen3.6-35B, 8 shards × ~4 GB, M4 Max).
- 1.71 s → 0.82 s weights (Dream-7B, 4 shards × ~3.5 GB, M4 Max),
  total wall 3.08 s → 2.17 s — beats MLX (2.67 s) on a short
  diffusion bench by 19%.
- With full companion stack (sub-shard split, async compile,
  async residency, shard-buffer-views, no madvise): startup
  ~0.40 s on Dream-7B (14 GB) and total wall 1.43 s, leaving
  gen time as the only remaining knob (M4 Max, fastdllm BL=32).

Both beat MLX's `ParallelFileReader` on the same machine. See
`references/gotchas.md` #28, #29, #40, #45, #46, #47, #50.

**Snippet**: `patterns/load_parallel_pread.c` — includes both the
single-thread-per-shard form and the sub-shard splitting variant.

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

**Speedup**: 1.05–1.10× on bandwidth-bound GEMVs **when X is not
already cache-resident across SGs**. On Apple-Silicon GPUs with shared
per-core L1/L2 and many concurrent SGs reading the same X, the cache
already serves X efficiently — explicit TG-mem staging then ADDS a
cooperative load + `threadgroup_barrier` whose cost can exceed the
saved DRAM BW.

**Negative result — when B4 hurts**: Tried on Qwen3.6-35B-A3B
(M4 Max, K=2048, post-B6 uint4 W loads) with WG=2 (TG=64) and
WG=4 (TG=128) across all `linear_q8`/`linear_q8_add`/`linear_q8_gather`
sites. Both **regressed ~1–2%** (96.7 → 95.0 at WG=4, → 93.5 at WG=2).
Root cause: with B6 already amortizing `(s, b)` reads down to ~14% of
traffic, W dominates the inner loop. X reads were ~29% of traffic but
in practice nearly all X reads HIT in cache because adjacent SGs on
the same core access the same `X[m, :]` close in time. The
cooperative-load barrier overhead exceeded the marginal X-BW savings.
B4 stays in the catalog because the win IS real on some chip/model
combos (was originally validated on smaller models w/ different cache
behavior), but **always measure before keeping it**. Revert if it
regresses; the kernel changes alone (dynamic WG via
`threads_per_threadgroup`) are reusable for other techniques.

**Implementation gotchas** (see `references/gotchas.md` #51, #52):
- All thread-position attributes in the kernel signature must agree
  in dimensionality — if you add `uint3 tg [[threadgroup_position]]`,
  the existing `uint tid [[thread_position_in_threadgroup]]` and
  `uint thr_per_tg [[threads_per_threadgroup]]` must become `uint3`
  too. INDEX attributes (`sg_id`, `lane`) stay `uint` regardless.
- `dispatchThreads:` with `grid_x % tg_x != 0` produces a non-uniform
  last threadgroup. A cooperative TG-mem load with stride
  `thr_per_tg` then only fills `actual_tg_x / 1` elements, leaving
  most of `X_tg` uninitialized. Keep grid_x divisible by tg_x, OR
  use dynamic WG via `threads_per_threadgroup` so the small-N sites
  (e.g. `shared_expert_gate` with N=1, grid_x=32) fall back to WG=1.

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

### B6. Amortize scale/bias across the quant group (uint4 W loads)

**What**: In an affine-quantized GEMV (q8 with `group_size = 64`,
or q4 with `group_size = 32`), the dequant cost in MLX-style ports
is dominated by the **scale/bias loads**, not the W byte loads.
The naive inner loop reads one uint32 of packed W per iter (4
weights), then reloads `(scale, bias)` from device memory every
iter even though all 4 weights share the same `(s, b)`. Two
properties combine to make this wasteful:

1. The quant group is wider than a single packed word (16
   q8-weights fit inside one 64-element group; 8 q4-weights fit
   inside one 32-element group).
2. `(s, b)` is per-group — so a single `(s, b)` load can dequantize
   a whole `uint4` of packed W (= 16 q8 weights or 32 q4 weights).

Fix: load packed W as `uint4` (16 bytes) per outer iter instead of
`uint32`. Reuse one `(s, b)` across all 16 weights. The outer loop
iterates `K / 16` instead of `K / 4`, and you do 4× fewer device
reads of `S[]` and `Bq[]` per output. Inside the iter, unpack the 4
words with a small helper:

```metal
static inline float4 lq8_deq_word(uint w, float s, float b) {
    float4 v;
    v.x = float((w      ) & 0xffu) * s + b;
    v.y = float((w >>  8) & 0xffu) * s + b;
    v.z = float((w >> 16) & 0xffu) * s + b;
    v.w = float((w >> 24) & 0xffu) * s + b;
    return v;
}

for (uint u4 = lane; u4 < K_u4; u4 += SIMD_WIDTH) {
    uint k4_base = u4 << 2;
    float4 x0 = float4(xrow4[k4_base + 0]);  // 4 × bfloat4 X reads
    float4 x1 = float4(xrow4[k4_base + 1]);  // reused across K_OUT
    float4 x2 = float4(xrow4[k4_base + 2]);
    float4 x3 = float4(xrow4[k4_base + 3]);
    uint g = u4 >> 2;                        // same group for all 16 weights
    for (uint o = 0; o < K_OUT; ++o) {
        uint4 wv = w_rows4[o][u4];           // one uint4 = 16 q8 weights
        float s = float(s_rows[o][g]);       // ONE (s,b) load per uint4
        float b = float(b_rows[o][g]);
        acc[o] += dot(x0, lq8_deq_word(wv.x, s, b));
        acc[o] += dot(x1, lq8_deq_word(wv.y, s, b));
        acc[o] += dot(x2, lq8_deq_word(wv.z, s, b));
        acc[o] += dot(x3, lq8_deq_word(wv.w, s, b));
    }
}
```

**Why it wins**: Apple GPUs at decode are W-bandwidth bound, but the
W-bandwidth budget includes `(s, b)` traffic. A per-uint32 loop reads
`8 bytes` of `(s, b)` per `4 bytes` of W per output — `(s, b)` are
half the traffic. Amortizing over 16 q8 weights drops `(s, b)` to
1/8 of the W-byte traffic, putting the budget back on the W bytes
where it should be.

**When**: After B1–B3 are in. The signal is "linear_q8 (or your
quantized GEMV family) is the dominant kernel by 70%+ in KPROF and
it's at ~25-35% of theoretical W-BW peak". If you're already at
~50%+ of theoretical W-BW peak, you're at the bandwidth floor and
this technique has nothing left to give.

**Speedup**: 1.15–1.20× decode on the whole pipeline when q8 GEMV
is the dominant kernel family. Qwen3.6-35B-A3B (linear_q8 ≈ 75%
of GPU): decode 82.8 → 98.4 tok/s (+18.7%). Bigger speedup on
smaller models where linear_q8 is an even larger fraction.

**Constraints**:
- `K % 16 == 0` (or `K % (4 × pack_unit)` for q4; trivial for all
  transformer K values).
- W must be stored as `device const uint4*`, which is the natural
  layout of `device const uint*` — no on-disk format change needed.
  Just `reinterpret_cast` the pointer.
- The helper compiles down to 4 byte extracts + 4 FMAs — no SIMD
  shuffle, no extra ALU pressure.

**Compose with**: B3 (K_OUT register tile) — they're independent and
multiplicative. Apply both. Note that **K_OUT=8 still regresses even
with uint4 W loads** — `4 X registers per uint4 × K_OUT=8 outputs ×
4 acc lanes` exceeds the per-thread register budget; the compiler
spills. Stick to K_OUT=4 with uint4 (gotcha #8).

**Apply to**: every `linear_q8_*` kernel — bare, with-residual, and
gather variants all benefit. Touched 3 kernels in Qwen3.6 in one
commit.

**Snippet**: `patterns/linear_q8_uint4_amortize_sb.metal`

**Commit**: cba7169 (Qwen3.6 — `linear_q8 uint4 W loads (1 (s,b) load
per 16 q8) — decode +18.7% (82.8→98.3 tok/s)`).

**Generalization**: This is "amortize per-group state across the
group". The same principle applies to q4 (one `(s, b)` per `uint`-of-
packed-q4 = 8 weights), mxfp4 (one shared exponent per 32-byte
group → load all 32 weights with one `uint8`-vector load), and any
future block-quant format where the group is wider than the natural
packed-word width.

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
  The early gpt-oss reference port stopped at BK=16 because BK=32
  spilled at WN=2 — but **BK=16 is NOT a fundamental ceiling**, it's
  set by per-SG register pressure at the chosen `(BM, BN, WN)`. On
  Dream-7B refine, going `WN=2 → WN=4` (see below) halves per-SG
  C+Bt regs and unlocks BK=32 (BN=64 path, 24 regs/lane) and BK=64
  (BN=32 path, 28 regs/lane; BN=64 path, 40 regs/lane), each landing
  -2% to -10% wall on top of dual-tile + C4. Always sweep BK ∈ {16,
  32, 64} with the WN that brings per-SG regs ≤ ~50. See gotcha #41
  for the budget formula.
- **WM**: SIMD-groups stacked in M. Forces Bt to be redundant across
  WM SGs (each loads the same Bt tile). Usually pick `WM=1` so that A
  is shared (small) and Bt is split (large).
- **WN**: SIMD-groups stacked in N. Treat WN as TWO knobs in one: it
  picks how many SIMD groups split the BN dimension (more parallelism
  per TG), AND it sets per-SG register pressure (more SGs → smaller
  per-SG share of C accumulator and Bt staging). `WN=2` is a good
  starting point (64 threads/TG); **`WN=4` is the register-pressure
  relief lever** that unlocks larger BK without changing total
  threads/TG. On Dream-7B: WN=2 BK=64 in BN=64 catastrophically spills
  (64 regs/lane → 4× slowdown); WN=4 BK=64 (40 regs/lane) fits and
  beats WN=2 BK=32 by -6% wall. Cost: A is now redundant across WN
  SGs, but at BN=64 with BK=64 the A footprint is small enough that
  this is a clear win.

**Register-budget formula** (Apple GPU, validated on M4 Max). Counted
in fp32 32-bit registers per lane (one `simdgroup_matrix<float, 8, 8>`
tile = 2 fp32 regs/lane; one `simdgroup_matrix<bfloat, 8, 8>` tile =
1 fp32-reg-equivalent/lane):

```
TM = BM / WM / 8;  TN = BN / WN / 8;  TKB = BK / 8;

regs/lane ≈ TM·TN·2         # C accumulator (fp32, 2 regs/lane/tile)
          + TM·TKB           # A staging   (bf16, 1 reg/lane/tile)
          + TN·TKB           # Bt staging  (bf16, 1 reg/lane/tile)
```

(The kernel files in Dream-7B's `src-metal/kernels/gemm_bf16*.metal`
have comments quoting these same C/A/Bt counts — this formula matches
the live source.)

M4 Max ceilings (empirical, from Dream-7B refine sweep):

- ~28–32 regs/lane → easy, high occupancy
- ~40 regs/lane → fits but occupancy drops; pays its way only if the
  larger BK halves the K-loop count
- ~50 regs/lane → border; some shapes regress silently
- ~60+ regs/lane → catastrophic spill (4× slowdown observed)

Examples (BM=16, threads/TG = 32·WM·WN):

| Tile (BM, BN, BK, WM, WN) | TM, TN, TKB | C  | A  | Bt | Total | Status |
|---------------------------|-------------|----|----|----|-------|--------|
| 16, 64, 16, 1, 2          | 2, 4, 2     | 16 |  4 |  8 |  28   | OK (original cargo-cult choice) |
| 16, 64, 32, 1, 4          | 2, 2, 4     |  8 |  8 |  8 |  24   | OK, -1.6% wall over BK=16 |
| 16, 32, 64, 1, 4          | 2, 1, 8     |  4 | 16 |  8 |  28   | OK, **-10% wall** (biggest mid-stage win) |
| 16, 64, 64, 1, 4          | 2, 2, 8     |  8 | 16 | 16 |  40   | OK (border), -6% wall vs BK=32 |
| 16, 64, 64, 1, 2          | 2, 4, 8     | 16 | 16 | 32 |  64   | catastrophic 4× slowdown (spill) |

The "WN=4 unlocks BK=64" pattern generalizes — if BK=64 puts the
total over ~50 at your current WN, doubling WN halves both C and Bt
contributions (A is unchanged: it's shared across the N-direction
SGs). See gotcha #41 for the full sweep procedure.

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

### C5. Aspect-ratio dispatch — N-axis variant for small-N refine

**What**: The N-axis analog of C4. When refine matmuls (`M ≤ 16`,
already in the BM=16 path) have `N` too small to fill the GPU at the
default BN, halve BN. For Dream-7B refine, N ∈ {3584 (q/o/down_proj),
512 (k/v_proj GQA), 18944 (gate/up_proj), 152064 (lm_head)}. At BM=16
BN=64, the N=3584 matmuls produce N/BN = 56 N-tiles — well below M4
Max's "2 TGs/core" target (40 cores × 2 = 80 TGs) and the kernels
become memory-latency-bound (effective BW ~150 GB/s vs ~400 GB/s for
the wide N=18944 matmuls on the same tile). Halving BN to 32 doubles
N-tiles to 112, fitting the target.

```c
// Refine dispatcher (called when M <= 16):
int use_bn32 = (M <= 16) && (N < 5120);   // small-N → BN=32
gpu_pipeline* pso = use_bn32 ? pso_gemm_bn32 : pso_gemm;   // BN=64 default
```

The trigger condition is symmetric to C4: at the chosen BM and WN,
you need enough N-tiles to fill the GPU. On M4 Max the empirical
threshold is `N < ~80 × BN_default`. The 12% wall win on Dream-7B
BL=16 refine came from down_proj going 1082us → 741us (-31%) **plus**
qkv/o_proj also picking up smaller wins at their N=3584 shape.

**Don't confuse with C4**: C4 routes PREFETCH/prefill `K > N` shapes
to smaller BM. C5 routes REFINE/decode small-N shapes to smaller BN.
They're orthogonal — different M shape, different starvation axis —
and they compose (you want both PSOs compiled and the dispatcher
checks both conditions).

**3-way tile routing** (the Dream-7B `d_linear_full_off` pattern):

```c
if (M > 16 && K <= N) {
    pso = pso_gemm_bm32;             // prefetch wide-N (gate/up, lm_head)
} else if (M <= 16 && N < 5120) {
    pso = pso_gemm_bn32;             // refine small-N (qkv, o, down)
} else {
    pso = pso_gemm;                  // refine wide-N + prefetch K>N
}
```

**When**: After C2 and C4. KPROF every refine matmul; the ones with
effective BW well below another refine matmul of comparable arithmetic
intensity are starved on N-tiles.

**Speedup**: -5.4% wall on Dream-7B BL=16 refine bench. Stacks with
C2 BK=64 + WN=4 for cumulative refine wins of ~25% from these tile-routing changes alone.

**Commit**: Dream-7B `src-metal/main.c` `d_linear_full_off`
(`use_bn32 = (M <= 16) && (N < 5120)`).

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

**N_SG sweep on long-context decode**: The optimal `N_SG` for SDPA is
**NOT** the theoretical perfect-occupancy divisor of (cores × SGs/core).
M4 Max nominally fits 16 cores × 24 SGs/core, so `N_SG ∈ {16, 24}`
would "perfectly" tile the GPU at 1 TG/(query_token, head) — but that's
not what wins. The TG-mem allocation grows with `N_SG`
(`sg_o[N_SG][D]`, `sg_m[N_SG]`, `sg_s[N_SG]`), which competes with W
caching, and the per-SG K-stripe shrinks at high N_SG to where the
fixed per-SG overhead (Q load, sm reduce, alpha update) dominates.

Empirical sweep on Qwen3.6-35B-A3B M4 Max, 1500-tok decode
(best-of-5 with 30 s thermal cool-down):

| N_SG | tok/s | Δ vs 16 |
|------|-------|---------|
| 12   | 91.2  | -5.7%   |
| 16   | 96.7  | baseline|
| **20** | **97.5** | **+0.8%** (best) |
| 24   | 95.6  | -1.1%   |
| 28   | 94.2  | -2.6%   |

The sweep is **flat near the optimum** (16 and 20 within 1% of each
other) but **steep on either side**. Always sweep
`N_SG ∈ {12, 16, 20, 24, 28}` on YOUR long-decode bench, not a short
one — at short Lk, SDPA's share is small and the optimum is fuzzy;
at long Lk SDPA grows linearly and the choice matters.

**Speedup**: 1.05–2.0× decode (cd2ec6b +5%, 11b7562 +96%, 2d59eed +3%).
Late-stage N_SG retuning typically nets +1–2% on long decode.

**CRITICAL — kernel `N_SG` constexpr and host dispatch
`threads_per_threadgroup` MUST match**: see `references/gotchas.md`
#53. The kernel sizes its TG-mem arrays as `sg_m[N_SG]`,
`sg_o[N_SG][D]` from the constexpr. If the host dispatch threads/TG
is `32 * 16` while the kernel constexpr is `12`, SGs 12..15 will
overflow these arrays and write garbage (often visible as `-1` or
NaN tokens). When sweeping N_SG, change BOTH the kernel constant
AND every `gpu_cmdbuf_dispatch*` call to `32 * N_SG` in lock-step.

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

### D5. K-per-iter ILP unrolling in the SDPA inner loop

**What**: The SG-per-(query_token, head) tile's inner loop does one
K-fetch + one `simd_sum` reduction per `lk` iteration. The K-band
loads and the dot-accumulator chain form a serial dependency: the
next K's vector load can't start until the previous `simd_sum`
retires. Apple GPU instruction scheduling has to stall there.

Unroll by N (typically 2, 4, or 8): process N K-vectors per outer
iter with N independent dot accumulators. Same total instructions —
N load groups, N FMA chains, N simd_sums — but the compiler now has
N×D-per-lane worth of independent FMAs to interleave with the load
group, hiding latency.

```metal
// 8 independent K-vectors per outer iter (M4 Max sweet spot for Lk≤~128):
uint lk = 0;
for (; lk + 8u <= Lk; lk += 8u) {
    const device bfloat* k0 = K + ( lk       * Nkv + kvh) * D;
    const device bfloat* k1 = K + ((lk + 1u) * Nkv + kvh) * D;
    /* k2 .. k7 */
    float dot0 = 0, dot1 = 0, /* dot2..dot7 */ dot7 = 0;
    for (uint i = 0; i < D_PER_LANE; ++i) {
        uint d = i * 32u + lane;
        float qi = q_reg[i];
        dot0 += qi * float(k0[d]);
        dot1 += qi * float(k1[d]);
        /* dot2 += ... ; dot7 += qi * float(k7[d]); */
    }
    dot0 = simd_sum(dot0); /* ... */ dot7 = simd_sum(dot7);
    if (lane == 0u) {
        scores[lk]      = dot0 * scale;
        scores[lk + 1u] = dot1 * scale;
        /* ... scores[lk + 7] = dot7 * scale; */
    }
}
for (; lk < Lk; ++lk) { /* scalar tail, runs at most N-1 times */ }
```

Apply the SAME unrolling to the AV phase (8 V-vectors per iter with
8 lane-local accumulators).

**When**: After D1 (parallel softmax). Independent of D2 and D3 — the
three compose orthogonally (D2 = pass count, D3 = SGs per row, D5 =
ILP within each SG's K-stripe). For diffusion-LLM refine where
`Lk ≈ L ≤ 100`, D5 alone gets most of the SDPA win; D3 helps less
because there aren't many K-tiles per (q, h) to split.

**Speedup**: -81% cumulative SDPA per-call on Dream-7B refine
(224 us → 42 us across N=2 → N=4 → N=8). That's ~7–9% total wall
depending on SDPA's share of the budget.

**Tuning N**: 2 is risk-free and gets ~22%; 4 gets ~50% cumulative; 8
gets ~81% on M4 Max. Going to 16 starts hurting register pressure
(N×D_PER_LANE worth of K loads held in registers per iter). For Lk
known to be < N at runtime, the unrolled main loop never runs and you
pay the slow tail — gate the unrolled path on `Lk > N_UNROLL`.

**Snippet**: extracted from Dream-7B `src-metal/kernels/sdpa_sg.metal`
(QK + AV phases each have an 8-K-per-iter main loop and a scalar tail).

**Commits**: Dream-7B `src-metal` bed13e6 (N=2, -22% per call),
eebedc9 (N=4, -33% on top), a226fe4 (N=8, -57% on top). Cumulative
-81%.

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

**Two flavors — pick by vocab size**:

| Vocab `V`         | Flavor                          | Why                                                |
|-------------------|---------------------------------|----------------------------------------------------|
| ≤ ~64k            | **Single-TG, 256 threads**      | One TG fully saturates the lanes; no merge needed. |
| ≥ ~100k (modern)  | **2-stage, `N_TG` TGs**         | One TG uses one GPU core; need `N_TG ≈ 64` to fill 40+ core M4 Max. |

The 2-stage variant for huge V is a separate kernel pair, not a
parameter tweak. **You will not notice the need for it in early
optimization** — at 40 tok/s decode you'll celebrate F1 single-TG
and move on. Re-profile once decode is mid-pipeline. If KPROF
still shows argmax at 200+ µs/call on a single TG with `V > 100k`,
that's a 1-core-bound kernel hiding inside an otherwise saturated
pipeline.

**Stage 1**: `N_TG` threadgroups (each 256 threads) each scan a
contiguous tile of `V / N_TG` elements and write one
`(max_val, max_idx)` partial to `part_max[tg_id]`, `part_idx[tg_id]`.

**Stage 2**: 1 TG of 32 threads (one SG) reduces `N_TG` partials to
a single index via `simd_max` + `simd_ballot` + `simd_shuffle`. No
TG-mem needed at this scale.

Buffer sizes are tiny (`N_TG × 8 bytes`); allocate once at startup.

**Hazard**: stage 2 must read what stage 1 wrote, so they live in
the SAME cmdbuf with a barrier (or in different cmdbufs — Metal
queue order gives free RAW between cmdbufs, gotcha #13). Do NOT
issue them concurrently.

**Speedup of 2-stage vs single-TG**: Qwen3.6 (V=248320, 1 TG = 253
µs/call → 2-stage ≈ 10 µs/call, -150 µs/token): decode 98.4 → 101.5
tok/s (+3.2%). The win scales with `V` — bigger vocab = bigger win,
because the 1-TG single-core ceiling stays at ~1 µs / 1000 elements
regardless of GPU size.

**Pattern updates**: see the `argmax_stage1_bf16` / `argmax_stage2_bf16`
section at the bottom of `patterns/argmax_parallel.metal`.

**Commit**: 6c49e38 (Qwen3.6 — `parallel argmax (2-stage) — decode
+3.2% (98.4→101.5 tok/s)`).

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

### F4. Multi-SG-per-row reduction for large D (rmsnorm / per-row softmax)

**What**: When `D` is large (typically `≥ 1024`), 1-SG-per-row RMSNorm
launches only `M` SGs total. At `M = BL = 16` (Dream-7B refine), that's
16 SGs — well below M4 Max's ~160 SG-slot capacity. Each SG chews
through all `D` elements serially, so the kernel is memory-latency-
bound rather than bandwidth-bound (a single rmsnorm call at D=3584 ran
~62 µs when peak BW says it should be < 1 µs).

Fix: assign `N_SGs` (4, 8, 16) SGs to each row. Each SG handles `D /
N_SGs` of the row. After SG-local `simd_sum` of squares, the `N_SGs`
partials merge through TG-mem into the row-wide `rrms`, then each SG
normalizes its own slice.

```metal
// 16 SGs / row, 32 lanes / SG → 512 threads/TG covering D up to 16384:
constant constexpr uint RN_SGS = 16u;

kernel void rmsnorm_sg_bf16(/* X, W, Y, params */
                            uint tgid [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid;
    float s = 0.0f;
    for (uint d = sgid * 32u + lane; d < D; d += RN_SGS * 32u) {
        float x = float(X[row * D + d]);
        s += x * x;
    }
    s = simd_sum(s);                           // SG-local

    threadgroup float partials[RN_SGS];
    if (lane == 0) partials[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {                           // SG 0 reduces RN_SGS partials
        float p = (lane < RN_SGS) ? partials[lane] : 0.0f;
        p = simd_sum(p);
        if (lane == 0) partials[0] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rrms = rsqrt(partials[0] / float(D) + eps);

    for (uint d = sgid * 32u + lane; d < D; d += RN_SGS * 32u) {
        float xv = float(X[row*D + d]);
        Y[row*D + d] = bfloat(bf16_round(xv * rrms) * float(W[d]));
    }
}
```

Launch with `dispatchThreadgroups: (M, 1, 1)`, threads/TG `(RN_SGS *
32, 1, 1)`, no per-TG iteration over rows.

**When**: After F1 (1-SG-per-row already in). KPROF rmsnorm/per-row
softmax: if the kernel runs `≫ 1 µs` per call at moderate D and there
are 60+ calls/forward, you're SG-undersubscribed and F4 is a free win.

**Tuning N_SGs**: 4 SGs covers `D ≤ 1024` per lane×8; 16 SGs covers
`D ≤ 16384` per lane×32. The aim is `M × N_SGs × 4 ≳ 160-320` SG
slots scheduled at once on M4 Max. For Dream-7B refine
(`M = 16, D = 3584`): 4 SGs (-4% wall), then 16 SGs (-3% wall on top,
per-call 62 µs → 9 µs, -85%). Going beyond 16 SGs (32 SGs / 1024
threads/TG) regressed on M4 Max — TG-mem partial-array overhead
plus final SG-0 reduction work exceed the marginal SG win.

**Don't confuse with D3**: D3 is multi-SG SDPA per (q, h) tile. F4 is
multi-SG rmsnorm/softmax per row. Same principle (spawn more SGs when
the natural launch shape under-fills the GPU) but different kernel
category, different reduction semantics.

**Commits**: Dream-7B `src-metal` f6a3e44 (rmsnorm 4 SGs/row, -4%
wall), fbd01fe (rmsnorm 16 SGs/row, -3% more on top, per-call -85%).

**Autoregressive datapoint (M=1)**: F4 is not just for diffusion
LLMs. In autoregressive decode, every row of the per-step rmsnorm
is `M=1` and you launch a single 1-SG-per-row dispatch — that's
**ONE SG total** for the whole rmsnorm call. On a 40-core M4 Max
that's 2.5% utilization, and at `D=2048` it's still 23+ µs/call.
With 60+ rmsnorm calls per forward (one before each of 24 attention
+ MLP blocks + a final), this is a meaningful slice — Qwen3.6-35B
KPROF showed rmsnorm at 15% before F4, 2.8% after.

For Qwen3.6 at `D=2048, M=1`: `N_SG=4` → +16.1% decode (71 → 83
tok/s); bumping to `N_SG=8` → another +1% on top (101.5 → 102.5).
`N_SG=16` was a tie — by then per-lane work is 8 elements, TG-mem
merge overhead approaches the per-lane work. Sweep `N_SG ∈ {4, 8,
16}` per D; pick the smallest that ties the largest (less TG-mem
pressure, lower TG-size).

**Where to dispatch the multi-SG variant**: ONLY at the call sites
where `M` is small (i.e. decode `M=1` or diffusion `M=BL=16` per
row). Keep the single-SG kernel for prefill `M ≥ Lq=16` and for
in-attention rmsnorms with large effective `M` (e.g. `q_norm`,
`k_norm` over all heads). Branch in `main.c` by `Lq` — this is a
mini decode/prefill split (catalog G) on the rmsnorm-only axis.

```c
// Dispatch by phase: multi-SG only when launch shape would
// undersubscribe the GPU.
if (Lq < 8) {
    // decode-style M=1: spawn N_SG SGs per row.
    gpu_cmdbuf_dispatch(cb, pso_rmsnorm_msg, ..., (RMSNORM_MSG_NSG * 32, M, 1), ...);
} else {
    // prefill M >= 8: 1 SG per row already gives enough SGs total.
    gpu_cmdbuf_dispatch(cb, pso_rmsnorm,     ..., (32, M, 1), ...);
}
```

**Commit**: 46c7949 (Qwen3.6 — N_SG=4, decode +16.1%); dbd51e7
(N_SG 4→8 follow-up, +1.0%).

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
