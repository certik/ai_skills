# Diffusion-LLM optimization notes

The catalog and the empirical attack order were largely written from
**autoregressive decoder** ports (Qwen3.6, gpt-oss). Diffusion-LLM
samplers (Dream, LLaDA, "fastdllm" style block-diffusion) have a
different forward structure, and that changes which optimizations
matter and in what order.

Read this file when your `./src-metal/` is a diffusion LLM and you've
just finished the kernel-level wins. The end-game looks different from
an autoregressive port.

## Table of contents

- [What's different](#whats-different) — autoregressive vs block-diffusion forward structure
- [Profiling diffusion runs](#profiling-diffusion-runs) — `gpu_busy` vs `host_post` vs wall, per-step breakdown
- [What does NOT help](#what-does-not-help-relative-to-autoregressive-ports) — A4, A8, output pruning that doesn't fire
- [What DOES help (in order)](#what-does-help-in-order) — the diffusion attack order, A9 → F3 → H3 → C2 dual-tile → C2 BK/WN → B5 → C4 → C5 → D5 → F4
- [End-to-end validation](#end-to-end-validation) — fastdllm-style block-by-block diffs
- [Worked example: Dream-7B on M4 Max](#worked-example-dream-7b-on-m4-max-this-skills-reference-port) — naive ~10 s wall → 2.17 s total, beats MLX by 19% wall / 39% gen

## What's different

A diffusion LLM doesn't generate one token at a time. Instead:

1. Initialize a sequence `x = [prompt; mask, mask, ..., mask]` of fixed
   length `L`.
2. Process the gen region in blocks of length `BL` (e.g., 16). For each
   block (4 blocks for a 64-token gen):
   - **Prefetch pass**: forward the entire `L`-long context once, fresh
     KV cache. Read ONE logits row (the one for `block_start`) to fill
     the first token of the block.
   - **Refine loop**: re-run forward with `Lq = BL` (block-local),
     using the cached KV plus replacing positions in `[block_start,
     block_start+BL)` of the cache. After each refine step, score the
     softmax confidence of every mask position in the block and unmask
     the high-confidence ones. Repeat until all `BL` positions are
     unmasked (or `steps_per_block` is exhausted).

So in numbers, for the validation prompt that drove this skill (Dream
7B, `L=64+25, BL=16, steps=64`):

| Phase     | Calls            | M (Lq) | What runs                                        |
|-----------|------------------|--------|--------------------------------------------------|
| Prefetch  | `n_blocks` = 4   | L = 89 | Full forward over the whole sequence             |
| Refine    | ~35 (early-exit) | BL = 16 | Block-local forward with KV-replace              |
| host_post | every step       | n/a    | Per-position softmax + argmax + confidence over V |

Refine is the long pole (~80% of GPU time). Prefetch is ~20%.

## Profiling diffusion runs

Tracking only `wall` and `gpu_busy` is insufficient — you also need
**`host_post`** as a separate accumulator, because the host loop that
does `softmax + argmax + confidence` for every mask position is
roughly:

```
for each refine step:
    for i in [0, BL):
        for v in [0, V):              # V = 152064 for Dream
            probs[v] = exp(logits[i, v] - max_logit)
        # then argmax + threshold check
```

That's `BL × V × 3` scalar ops per refine, all on the CPU. In one
session we saw `host_post = 0.25 s` out of `wall = 2.90 s` — **8.6%
of total wall, completely invisible in `gpu_busy`**. Optimizing
kernels can't reach this. Always log it separately:

```c
double t = now_sec();
double g = gpu_time_get();
forward_cached(...);
prefetch_wall += now_sec() - t;
prefetch_gpu  += gpu_time_get() - g;

double th = now_sec();
... host softmax / argmax / threshold ...
host_post_wall += now_sec() - th;
```

Then your stats line is `wall = gpu_busy + host_post + (encode_gap)`,
where `encode_gap = wall - gpu_busy - host_post` is the residual
host-encode overhead — usually small once H1/H2 are in.

## What does NOT help (relative to autoregressive ports)

### A4 (2-deep cmdbuf pipeline) is mostly inert

For autoregressive decode, A4 wins because the host can encode `cb_{t+1}`
while the GPU executes `cb_t`. For diffusion refine, this **fails**:

- Each refine step's input ids depend on `select_confident_updates(...)`
  on the previous step's logits.
- The host therefore can't start encoding the next refine until the
  current refine has returned its argmax/confidence to host memory.

A4 can still save the prefetch wall-gpu gap (~0.08 s in one port) but
that's not where the bottleneck is. Don't reach for A4 on a diffusion
LLM unless prefetch dominates.

### Decode/prefill split (G) is replaced by prefetch/refine split

A diffusion LLM never does true "decode" (single-token-at-a-time). The
useful split is by *Lq value*: prefetch (`Lq=L=89` for Dream-7B
validation prompt) vs refine (`Lq=BL=16`). Make sure your GEMM tile is
right for BOTH. See "dual-tile dispatch" below.

## What DOES help (in order)

### Step 0: Parallel pread weight loader (A9)

**Do this first** — even more important on diffusion LLMs than on
autoregressive ports. Diffusion gen is short (Dream-7B BL=32 short
bench: ~1.2 s) so weight load can be a comparable fraction of total
wall. A 1.7 s naive `mmap+memcpy` weight load against a 1.2 s gen
makes "we beat MLX on tok/s" still lose on `time ./binary`.

Replace `gpu_buf_new_from(g_ctx, t->data, t->nbytes)` (memcpy from
the mmap'd shard) with one `pread()` thread per shard via
`dispatch_apply`. On Dream-7B (14 GB, 4 shards, M4 Max): weight load
drops 1.71 s → 0.82 s, total wall 3.08 s → 2.17 s, MLX target 2.67 s
— we beat MLX on total wall by 19% (and that's *before* the gen-side
wins below).

Pattern: `patterns/load_parallel_pread.c`. Catalog: A9.
Gotcha: #29 (why mmap+memcpy is slow), #40 (why the obvious zero-copy
shortcut doesn't work).

### Step 1: GPU per-row softmax + argmax + confidence (F3)

The single biggest *gen-time* diffusion-LLM win (the single biggest
overall win is Step 0 above when load > gen). Replace the host loop
above with one Metal kernel:

```
input:  logits   bf16 [BL, V]
output: out      {int idx; float conf}[BL]
```

One TG of 256 threads per row, two-pass:
1. `max + argmax over V` (SG-wide `simd_max`, then cross-SG via TG-mem).
2. `sum_exp(logit - max)` over V. `conf = 1 / sum_exp`.

The output is `BL * 8 bytes = 128 bytes`, tiny. The host then only
applies the threshold logic and writes back the chosen ids.

In one port this dropped `host_post` from 0.25 s → 0.00 s and the
wall from 2.90 s → 2.63 s on the same prompt. See catalog F3, pattern
`patterns/softmax_argmax_per_row.metal`.

### Step 2: Prune unused output rows in prefetch lm_head (H3)

The prefetch forward does `lm_head(h_buf[0..L-1])` even though the host
only reads ONE row of the output (the row at `block_start - 1`, used
to fill the first token of the next block). For `L=89`, `V=152064`,
that's 88 rows of compute and 88 rows of `W` streaming that get
discarded.

Pass an `lm_head_only_row` hint to `forward_cached`; on that path,
dispatch lm_head with `M=1` and an `x_off_bytes` offset into `h_buf`.
The kernel writes one row to `logits_buf[0]`, the host reads row 0.

Save: ~30 ms on prefetch (small but free). See catalog H3.

### Step 3: Dual-tile GEMM dispatch by M

Prefetch (M=89) is compute-bound and benefits from `BM=32 TM=4`
(halves the number of M row-bands, halves W-streaming). Refine (M=16)
stays at `BM=16 TM=2` (BM=32 wastes 50% of the threads and Cs TG-mem).

The simplest layout: two PSOs (`pso_gemm` and `pso_gemm_bm32`), one
kernel file each, `d_linear` picks based on `M > 16`. Each kernel file
defines its tile constants and is otherwise identical algorithm.

Buffer padding: with BM=32 in play, ALL workspace buffers (`x_buf`,
`h_buf`, `attn_buf`, etc.) must be padded to `(L + 31) & ~31` rows so
the BM=32 row-band loads are in-buffer. If you forget this and a load
runs off the end, you get nondeterministic garbage past the first ~30
tokens.

### Step 4: Fused residual into o_proj / down_proj (B5)

Same as autoregressive. Drops 56 `residual_add` dispatches per
forward. Token-identical on Dream-7B in our test (bf16 round point
happened to coincide).

### Step 5: Concurrent encoder + barrier surgery (A5)

Same as autoregressive. q/k/v are independent, gate/up are independent,
rope_Q/rope_K are independent, cache_write_K/cache_write_V are
independent. Wrap each independent group in a `gpu_cmdbuf_barrier`
fence.

### Step 6 (BL ≥ 32 only): Aspect-ratio dispatch K>N for down_proj (C4)

At small BL (e.g., 16) the refine M is ≤16 and everything routes to
the BM=16 path, so this step is a no-op. At BL ≥ 32 refine starts
hitting `M > 16` and the dual-tile dispatch (Step 3) sends EVERYTHING
to BM=32. That's right for gate_proj, up_proj, qkv, o_proj, lm_head —
all of which have `N ≥ K`. It's WRONG for down_proj.

down_proj has `K = I = 18944, N = H = 3584` — the only LLaMA-style MLP
matmul with K > N. At BM=32 BN=64 you get N/BN = 56 TGs, which on
M4 Max (~80 TG slots) leaves the memory subsystem starved (measured
125 GB/s effective vs gate_proj's 314 GB/s on the SAME matrix size).

Fix: extend the dual-tile dispatch with an aspect-ratio check.

```c
int use_bm32 = (M > 16) && (K <= N);   // K > N → BM=16 even at M>16
gpu_pipeline* pso = use_bm32 ? pso_gemm_bm32 : pso_gemm;
```

On Dream-7B BL=32 this drops down_proj from 1082us → 741us per call
(-31%), refine from 73ms → 63ms per step, and wall from **1.72 s →
1.52 s (12% faster, beats MLX 1.56 s)**. See gotcha #39 and catalog
C4 for the full analysis.

### Step 7: Aspect-ratio dispatch — small-N refine BN=32 (C5)

The N-axis sibling of Step 6. Even within the BM=16 refine path, the
matmuls with N=3584 (qkv, o_proj, down_proj) at BN=64 produce only
56 N-tiles — under M4 Max's ~80 TG-slot target. Halving BN to 32
doubles N-tiles to 112 and restores memory-level parallelism.

Extend Step 3's dispatcher to a 3-way:

```c
if (M > 16 && K <= N) {
    pso = pso_gemm_bm32;            // prefetch wide-N (Step 6/C4)
} else if (M <= 16 && N < 5120) {
    pso = pso_gemm_bn32;            // refine small-N (this step, C5)
} else {
    pso = pso_gemm;                 // refine wide-N + prefetch K>N
}
```

On Dream-7B BL=16 refine: -5.4% wall. See catalog C5.

### Step 8: BK > 16 with WN=4 in the BN=32 / BN=64 paths (C2 + gotcha #41)

The "BK=16 is the maximum on M4 Max" rule is a `WN=2` artifact, not
a hardware ceiling. The hard limit is per-SG register pressure (see
gotcha #41's register-budget formula). Going `WN=2 → WN=4` halves
per-SG regs and unlocks BK=32 and BK=64.

On Dream-7B:
- BK=32 BN=32 WN=4: -2.5% wall over BK=16
- BK=32 BN=64 WN=4: -1.6% wall
- **BK=64 BN=32 WN=4** (BM=16, 48 regs/lane): -10% wall — biggest
  single mid-stage win.
- BK=64 BN=64 WN=4: -6% wall.

At WN=2 BK=64 catastrophically spills (104 regs/lane → 4× slowdown).
**Sweep BK ∈ {16, 32, 64} × WN ∈ {2, 4}** before declaring a tile
unfit.

### Step 9: SDPA K-per-iter ILP unrolling (D5)

Even after D1 (parallel softmax), the inner `lk` loop has a serial
dependency between the K-fetch and the `simd_sum` reduction.
Unroll the QK phase (and AV phase) by N K-vectors per outer iter
with N independent dot/V accumulators. The compiler then interleaves
N K-band loads with N FMA chains, hiding load latency.

On Dream-7B refine: -22% per call at N=2, -33% on top at N=4, -57%
on top at N=8 (cumulative -81%, 224 µs → 42 µs per call, ~7% of
wall). Tail loop handles `Lk mod N` iters scalar.

```metal
for (; lk + 8u <= Lk; lk += 8u) {
    // load 8 K-vectors, compute 8 independent dots,
    // simd_sum each into 8 scores; write all 8.
}
for (; lk < Lk; ++lk) { /* scalar tail */ }
```

See catalog D5 for the full snippet.

### Step 10: Multi-SG-per-row RMSNorm for large D (F4)

At M = BL = 16, the 1-SG-per-row rmsnorm launches only 16 SGs total
— well below M4 Max's ~160 SG-slot capacity. The kernel becomes
memory-latency-bound. Fix: spawn 4 (or 16) SGs per row, each
processing `D / N_SGs` of the row, merge partials through TG-mem.

On Dream-7B (D=3584):
- 4 SGs/row: -4% wall (per-call 62 µs → ~16 µs)
- 16 SGs/row: -3% wall on top (per-call → 9 µs, -85% total)

32 SGs/row regressed — 1024 threads/TG exceeds the practical
occupancy sweet spot, and the final SG-0 reduction overhead grows.
16 SGs is the peak on M4 Max for D=3584. See catalog F4.

## End-to-end validation

Diffusion LLMs are particularly easy to validate because the entire
output sequence is bounded by `L`. Generate the full sequence and
compare token-by-token against `src-cpu/` and MLX. The "first N
tokens" rule is essentially "all generated tokens" here.

## Worked example: Dream-7B on M4 Max (this skill's reference port)

| Stage                           | Wall (s) | gpu_busy | host_post |
|---------------------------------|----------|----------|-----------|
| Naive port (1-thread argmax+softmax host) | ~10  | ~3       | ~5        |
| After B1, SG-per-row reductions | ~5       | ~3       | ~1        |
| After H1 (glue kernels)         | ~4       | ~2.7     | ~0.5      |
| After C1+C2 (MMA GEMM)          | 3.40     | 2.70     | 0.25      |
| After A5 (concurrent encoder)   | 3.04     | 2.71     | 0.25      |
| After C2 dual-tile (BM=32/16)   | 3.04     | 2.71     | 0.25      |
| After B5 (fused residual)       | 3.04     | 2.72     | 0.25      |
| After H3 (single-row lm_head)   | 3.04     | 2.69     | 0.25      |
| After BK=16 GEMM inner loop     | 2.90     | 2.54     | 0.25      |
| **After F3 (GPU softmax+argmax)** | **2.63** | **2.53** | **0.00**  |
| MLX target                      | 2.74     | —        | —         |

Final: 4% faster than MLX (`2.63 s` vs `2.74 s`). F3 alone closed half
the remaining gap.

### Follow-up: BL=32 with max_tokens=32, where down_proj dominates

A second, shorter benchmark (`max-new-tokens=32 --steps=24 --block-length=32`)
exposed a different bottleneck — **down_proj alone took 38.5% of refine
GPU time** because at BM=32 BN=64 it produced only 56 TGs (vs gate_proj's
296), under-occupying the GPU's memory subsystem.

| Stage                                            | Wall (s) | Notes                            |
|--------------------------------------------------|----------|----------------------------------|
| All optimizations above (≤ Step 5)               | 1.72     | refine 73 ms/step, down_proj 1082 us/call (125 GB/s) |
| **After C4 (down_proj K>N → BM=16 routing)**     | **1.52** | refine 63 ms/step, down_proj 741 us/call (-31%) |
| MLX target                                       | 1.56     | —                                |

Final on this bench: **3% faster than MLX** (1.52 s vs 1.56 s),
single-line dispatch change. This is the lesson that prompted gotcha
#39 / catalog C4 and made aspect-ratio routing a permanent step in
the recipe (Step 6 above).

**Diagnostic note**: this win would have stayed hidden without KPROF.
The wall gap to MLX was only ~10% (not 2×, despite a misleading `tok/s`
ratio in the surface output — see "Profiling diffusion runs" above),
and the bottleneck wasn't one of the usual suspects. Always KPROF
before optimizing past the empirical attack order; per-kernel attribution
is what makes "this matmul is twice as slow as the matmul of the same
size" visible.

### Follow-up #2: total wall including startup latency (Step 0 / A9)

After landing the gen wins above, gen on the BL=32 short bench was
~1.18 s — already faster than MLX's 1.57 s — but `time ./dream`
reported 3.08 s total vs MLX's 2.67 s total. We were losing on total
wall despite winning on gen. Phase-breakdown timing showed weight
loading at 1.71 s, paid before any GPU compute starts.

| Stage                                    | Weights (s) | Gen (s) | Total wall (s) | Notes               |
|------------------------------------------|-------------|---------|----------------|---------------------|
| Per-tensor `gpu_buf_new_from` (mmap+memcpy) | 1.71     | 1.18    | 3.08           | slower than MLX (2.67) |
| **After A9 (parallel pread per shard)**  | **0.82**    | 1.18    | **2.17**       | **19% faster than MLX** |
| MLX target                               | ~1.1        | 1.57    | 2.67           | —                   |

This is the lesson that put A9 as **Step 0** in this attack order
(and gotcha #40 for the trap of trying per-tensor zero-copy as a
"simpler" alternative — it doesn't work because safetensors offsets
aren't page-aligned). Always log `[startup] tokenizer=... weights=...
total=...` before declaring the gen wins sufficient.

### Follow-up #3: pushing refine wall from 2.60 → 1.68 s on the BL=16 long bench

The BL=32 short bench (Follow-up #1, #2) doesn't exercise everything
— it has only 24 refine steps. The BL=16 long bench (max_new_tokens=64,
steps=64, threshold=0.9 — Dream's reference benchmark) runs ~1064
refine forwards in this port, so per-call wins compound. After the
short-bench wins above, the long-bench refine wall was 2.60 s — fine
but with room left. The Steps 7–10 refinements drove it to 1.68 s
(39% faster than MLX's 2.74 s).

| Stage                                            | Wall (s) | Per-call wins                      |
|--------------------------------------------------|----------|------------------------------------|
| All optimizations above (Steps 0–6)              | 2.60     | refine 73 ms/step                  |
| **Step 7** (BN=32 small-N refine, C5)            | 2.46     | qkv/o/down at N=3584: -X%/-Y%      |
| **Step 8a** (BK=32 in BN=32 GEMM, WN=4)          | 2.42     | -1.6% wall                         |
| **Step 8b** (BK=32 in BN=64 GEMM, WN=4)          | 2.36     | -2.5% wall on top                  |
| **Step 9a** (SDPA 2-K-per-iter ILP, D5)          | 2.27     | sdpa per-call: 224 → 173 µs (-22%) |
| **Step 9b** (SDPA 4-K-per-iter ILP)              | 2.19     | sdpa per-call: 173 → 116 µs (-33%) |
| **Step 8c** (BK=64 in BN=32 GEMM, WN=4)          | **1.97** | -10% wall — biggest mid-stage win  |
| **Step 8d** (WN=4 BK=64 in BN=64 GEMM, C2)       | 1.85     | -6% wall on top                    |
| **Step 8e** (WN=4 BK=64 in BN=32 GEMM, C2 refresh) | 1.81  | -2% wall                           |
| **Step 9c** (SDPA 8-K-per-iter ILP)              | 1.77     | sdpa per-call: 116 → 42 µs (-64%)  |
| **Step 10a** (rmsnorm 4 SGs/row, F4)             | 1.74     | rmsnorm per-call: 62 → 16 µs       |
| **Step 10b** (rmsnorm 16 SGs/row, F4)            | **1.68** | rmsnorm per-call: 16 → 9 µs (-85% total) |
| MLX target                                       | 2.74     | —                                  |

Final on this bench: **39% faster than MLX** (1.68 s vs 2.74 s),
gpu_busy/wall = 94%.

Diagnostic observations from this run:

1. **WN=4 was the unlock for BK > 16.** Without the WN=2 → WN=4
   change, BK=32 spilled and BK=64 catastrophically spilled (4×
   slowdown). The "BK=16 is the maximum on M4 Max" rule of thumb
   from the gpt-oss / Qwen3.6 ports was a WN=2 artifact. Sweep BK ∈
   {16, 32, 64} × WN ∈ {2, 4} on every new model.

2. **SDPA ILP unrolling is orthogonal to D1/D2/D3.** Even with parallel
   softmax already in (D1) and short Lk (≤89, so D3 multi-SG split
   has limited room), unrolling the inner loop by 8 K-vectors with
   8 independent accumulators got -81% per-call. Worth applying
   even on autoregressive decoders if SDPA is hot.

3. **Multi-SG rmsnorm only matters when D and call count are both
   large.** At M=16 D=3584 with 60+ calls/forward, the kernel was
   memory-latency-bound at 62 µs/call. 16 SGs/row brought it under
   10 µs. On a model where rmsnorm is < 1% of wall, skip this step.

4. **Per-kernel attribution kept us honest.** Several "obvious" wins
   (WN=8, BK=128, pre-transposed W, TG-mem A staging, fused
   gate_up_silu) regressed when measured. KPROF every change. The
   list of failures-from-this-port is in the dream repo's
   `src-metal/PERF.md` "What was tried and didn't help".
