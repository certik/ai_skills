---
name: optimize-metal
description: >
  Optimize a working but naive Apple-GPU Metal LLM implementation (from
  the port-c-to-metal skill, in ./src-metal/) to match MLX speed within
  ±5% on BOTH per-token tok/s AND total wall time (including weight
  load). Catalog: SIMD-group-per-output GEMV, bfloat4 loads, qmv4
  register tiling, simdgroup_matrix MMA for prefill, sorted-gather MoE,
  online-softmax SDPA, multi-SG SDPA, parallel argmax, 2-deep cmdbuf
  pipeline, persistent + const param buffers, concurrent encoder, fused
  residual epilogues, decode/prefill split, GPU glue kernels for
  host-side breaks (often the biggest single decode win),
  SG-per-(row,dim) for RNN/SSM recurrences, parallel pread weight
  loader (one fd per shard via dispatch_apply — beats mmap+memcpy 2–3×
  and brings startup below MLX). Also handles diffusion-LLM samplers
  (Dream / LLaDA / fastdllm): GPU per-row softmax+argmax+confidence
  (replaces the host_post loop), single-row output pruning via X
  offset, dual-tile (BM, BK) GEMM dispatch by M. Validates tokens
  against the C reference after each change. Triggers: optimize metal,
  speed up metal, match mlx speed, gpu optimization, metal performance,
  kernel tuning, startup latency, weight loading, total wall time,
  diffusion LLM (Dream / LLaDA / fastdllm) on metal.
---

# optimize-metal — make the Metal port fast (match MLX ±5%)

Take a working but slow Metal LLM implementation (`./src-metal/`,
produced by `port-c-to-metal`) and iteratively optimize it until decode
and prefill throughput are within ±5% of the MLX-equivalent reference
on the same machine.

The optimization is **not** a single rewrite. It is a disciplined
iteration: profile, identify the top-1 bottleneck, look up the next
applicable technique in the catalog, apply it, validate against the C
reference, commit. Repeat.

## When to use

After `port-c-to-metal` has produced a correct-but-slow `./src-metal/`.
Use this skill to:

- Profile the Metal kernels and identify per-kernel bottlenecks.
- Apply named optimization techniques (catalog) and verify correctness
  after each.
- Drive performance to within ±5% of the MLX reference.

**Do not** use this skill to fix correctness bugs. If a kernel produces
wrong output before optimization, fix that in `port-c-to-metal` first.

## Pipeline

```
build-c-reference  →  port-c-to-metal  →  optimize-metal
                                            (this skill)
```

## Prerequisites

- `./src-metal/` exists, builds, and produces tokens that match
  `./src-cpu/`.
- `./src-cpu/` (the C reference) is **kept untouched** — it remains the
  ground truth for correctness.
- A way to run the MLX reference (`infer_mlx.py` or `mlx_lm`) for
  tok/s comparison.
- The model + a fixed validation prompt (same one used in earlier
  skills).

## Inputs to gather

1. Path to `./src-metal/` (default: sibling of this skill's working dir).
2. The MLX reference command to compare against, e.g.
   `uv run python infer_mlx.py --prompt "<P>" --max-tokens 256`.
3. Target machine spec (M1/M2/M3/M4/Pro/Max/Ultra) — affects tile sizes
   and SIMD-group widths.

## Success criteria

- Decode tok/s within ±5% of MLX on the same prompt + machine.
- Prefill tok/s within ±5% of MLX on the same prompt.
- **Total wall time within ±5% of MLX** when startup latency is a
  meaningful fraction of total (e.g., short benchmarks, diffusion LLMs
  where gen is ~1 s — load can be 30–50% of total). This is what the
  user sees from `time ./your_binary`. Hitting per-token parity but
  losing on total because weight load is 2× slower than MLX is a real
  failure mode that the tok/s metric hides.
- First N generated tokens (N ≥ 4, ideally 16) match the C reference
  exactly. Later tokens may diverge due to pure numerical drift; that
  is acceptable as long as the divergence is *purely numerical* (no
  bug) and you can demonstrate it via per-kernel tolerance checks
  against the C reference.
- All optimizations committed individually so any one can be reverted.

## The workflow loop

Single, repeating cycle:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Profile current build                                        │
│    - measure decode + prefill tok/s                              │
│    - print BOTH wall and gpu_busy (the key diagnostic)           │
│    - identify hottest kernel (GEMV for decode, GEMM for          │
│      prefill, MoE for both, SSM step if present)                 │
│                                                                  │
│ 2. Pick next technique                                           │
│    - look up the bottleneck kernel category in the catalog       │
│    - choose the next applicable, not-yet-applied technique       │
│                                                                  │
│ 3. Implement the technique on a branch / WIP commit              │
│    - apply ONLY the one technique                                │
│                                                                  │
│ 4. Validate                                                      │
│    - per-kernel correctness against C reference (~1e-2 abs bf16) │
│    - end-to-end tokens against C reference for fixed prompt      │
│    - first N tokens must match exactly                           │
│    - if regression: revert; try a different technique            │
│                                                                  │
│ 5. Measure                                                       │
│    - if speedup ≥ 1.02× on the dominant phase (decode or         │
│      prefill) it's a win — commit                                │
│    - if neutral or slowdown: revert                              │
│                                                                  │
│ 6. Check stop condition                                          │
│    - if within ±5% of MLX on both prefill and decode → done      │
│    - else: back to step 1                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Profiling — the key diagnostic

**Always print `wall` AND `gpu_busy` for every timed phase.** The Metal
shim already sums `(GPUEndTime - GPUStartTime)` into `g_gpu_time`:

```c
g_gpu_time = 0.0;
clock_gettime(...t0);
// ... run prefill or decode loop ...
clock_gettime(...t1);
double wall = (t1 - t0);
fprintf(stderr, "decode: %.2fs (%d toks; %.1f tok/s; gpu_busy=%.2fs)\n",
        wall, n_gen, n_gen / wall, g_gpu_time);
```

The `wall - gpu_busy` gap tells you which family of optimization to
chase:

- `gpu_busy ≈ wall`  → you're GPU-bound; optimize kernels (B, C, D, E,
  F, I).
- `gpu_busy << wall` → you're CPU-bound; optimize scheduling (A, H).

**The single biggest win in a typical naive port comes from eliminating
host-side breaks (H1 / glue kernels).** You can see this happen
immediately as `gpu_busy / wall` ratio jumping from ~30% to ~95%.

**Diffusion-LLM samplers need a third number.** If your sampler does
non-trivial per-step CPU work (per-position softmax + argmax +
confidence over V at every refine step — typical of Dream / LLaDA /
fastdllm), also accumulate `host_post`. A large `wall - gpu_busy` gap
on these models often looks like a host-scheduling problem (A/H) but
is actually a sampler problem (F3): there's nothing to overlap with,
the CPU is genuinely the gap. See `references/profiling.md` and
`references/diffusion-llm.md`.

**Print a phase-breakdown line at startup.** Before any tok/s
optimization, log a one-liner:

```c
fprintf(stderr,
    "[startup] tokenizer=%.3fs config=%.3fs metal=%.3fs weights=%.3fs total=%.3fs\n",
    t_tok - t_start, t_cfg - t_tok, t_metal - t_cfg,
    t_weights - t_metal, t_weights - t_start);
```

If `weights` dominates startup (almost always does — it's GB of I/O),
apply **A9 parallel pread weight loader** immediately. On Dream-7B
(14 GB, 4 shards, M4 Max) this dropped `weights` from 1.71 s → 0.82 s
and total wall from 3.08 s → 2.17 s, beating MLX's 2.67 s by 19% on a
short fastdllm bench where gen is only 1.18 s. See catalog A9 and
gotcha #29.

Use `mlx_lm`-equivalent timing (`tps = (n_gen - 1) / decode_time`,
exclude the first decode step) so you can compare apples to apples with
MLX, AND use ≥64 tokens AND best-of-3 runs: short runs on busy machines
have ±20% variance.

Full profiling guide (KPROF, variance discipline, first-cmdbuf
hiccup): see `references/profiling.md`.

## The empirical attack order

For a reduced-precision (quantized) MoE decoder LLM on Apple Silicon,
this is the empirical order of biggest wins from a naive port
(validated on Qwen3.6-35B-A3B on M4 Max). **Go in this order unless
your profile clearly says otherwise** — the profile is always ground
truth, but this list saves you most of the exploration cost.

### First-pass — naive → mid-pipeline (~1.9 → ~43 tok/s for Qwen3.6)

0. **Parallel pread weight loader** (A9). Apply from the very first
   port — startup is the most user-visible "wait" and the win is huge
   (2–3× on weight load, often turning a 10 s startup into 3.5 s on a
   35 GB model, or 1.7 s into 0.82 s on a 7 B model). The naive
   `mmap + memcpy per tensor` is page-fault-bound at ~1 GB/s on macOS
   regardless of how many threads you throw at it. See catalog A9 and
   gotcha #29.
1. **Parallel argmax** (F1). 1-thread argmax over 200k+ vocab is the
   most common "why is decode capped at <5 tok/s" answer.
2. **SIMD-group-per-output for ALL reduced-precision GEMV** (B1).
   Affects q/k/v/o_proj, lm_head, router, every MoE linear.
3. **SG-per-row for the reduction kernels**: rmsnorm, softmax_topk,
   embed_dequant_gather. Cheap.
4. **SG-per-(hv, dv) for the recurrent state step** (I1) if the model
   has GatedDelta / Mamba / SSM layers. 32 serial threads → 4096 SGs.
5. **SDPA SG-per-(lq, hq) with TG-mem scores** (D1 worked example).
   Necessary precursor to D2/D3.
6. **ELIMINATE HOST-SIDE BREAKS** (H1, glue kernels). Usually the
   single biggest jump in the whole pipeline — often 1.5–2×.
7. **Merge embed + forward + argmax into one cmdbuf** (H2). Small win.
8. **2-deep cmdbuf pipeline with id-swap** (A4 variant). Closes the
   first CPU-encode gap to MLX.

### Last 5–10% — closing the gap to MLX (~43 → 69.5 tok/s for Qwen3.6)

9. **Parallel top-K** (F2) — same trick as parallel argmax but applied
   to router top-K over N_EXPERTS=32–128.
10. **Multi-SG SDPA with online-softmax merge** (D3). Once D1+D2 are
    in, splitting each (lq, hq) across N_SG=4 SIMD groups + merging
    per-SG `(m_i, s_i, o_i)` via the online-softmax formula is the
    decode SDPA endgame.
11. **TG=(1,1,1) → (32,1,1) bump** for all elemwise/glue kernels (see
    `references/gotchas.md` #17). Free win.
12. **Drop barriers between independent ops** (subset of A5): e.g.,
    `q_norm/k_norm`, `rope Q/K`, `rmsnorm_scale q/k`, `sigmoid+compute_g`.
13. **Fuse residual into LAST linear epilogue** (B5) — `o_proj`,
    `out_proj`, `shared_combine_add`. Drops dispatches.
14. **PARALLEL INDEPENDENT CHAINS** (A8). Restructure your encoder so
    multi-kernel subgraphs (e.g., MoE chain vs shared-expert chain in
    Qwen MoE) run as parallel pipelines, not in series. Final 3–5% to
    MLX parity — usually the BIGGEST single late-stage commit. See
    `references/parallel-chains.md`.

Steps 9–14 took the Qwen3.6-35B-A3B reference port from 41.5 → 69.5
tok/s, hitting MLX parity (MLX runs at ~70.1 tok/s on the same M4 Max).

### Optimizations that often land in the noise band

(Validated on Qwen3.6-35B-A3B; may differ on your model. Still worth
applying for code hygiene but don't expect headline speedups.)

- bfloat4 vector loads in isolation (baked into B1 already).
- qmv4 K_OUT=4 register tile (q8 dequant is bandwidth-bound, not
  register-bound; K_OUT=8 actually regressed).
- Concurrent encoder ALONE without barrier surgery — most barriers are
  needed; A8 is where the big win lives.

### Skip for decode-only goals

- C1/C2/C3 (GEMM MMA for prefill) — only relevant if your goal is to
  match prefill speed.

### Diffusion LLMs (Dream / LLaDA / fastdllm)

These are NOT autoregressive decoders. The forward pass runs twice
per generated block: once at `M = L` (prefetch, populates the KV
cache) and many times at `M = BL` (refine, iteratively unmasks
positions). The naive host post-processing — per-position softmax +
argmax + confidence over V — dominates the gap to MLX in a way the
A/H techniques can't reach.

The empirical order is different. Most A/H techniques (A4 2-deep
pipeline, B5 fused residual, A8 parallel chains) do less work or
don't help at all. **Total wall (load + gen) matters as much as
gen tok/s on these models** — gen is only ~1 s so a slow weight
loader can easily make the whole process slower than MLX even with
faster gen. The wins, in order:

0. **A9 parallel pread weight loader** — even more important than for
   autoregressive ports, because diffusion gen is short (~1 s on
   short benches), so weight load is a comparable fraction of total
   wall. On Dream-7B (4 shards, 14 GB, M4 Max): naive 1.71 s →
   parallel pread 0.82 s, total wall 3.08 s → 2.17 s beating MLX's
   2.67 s by 19% even before any gen-side optimization on the BL=32
   bench.
1. **F3** — per-row softmax+argmax+confidence on GPU. Replaces the
   host loop that shows up as a large `host_post` invisible to
   `gpu_busy`. In one port: 0.25 s → 0.00 s, wall 2.90 → 2.63 s.
2. **H3** — single-row output pruning for the prefetch path. Saves
   most of one full lm_head streaming pass.
3. **C2 dual-tile-by-M** — two PSOs with different (BM, BK), picked by
   `M > 16` at dispatch. Best (BM, BK) for refine ≠ best for prefetch.
4. **C2 BK=16 (and beyond) with WN tuning** — unroll multiple K-bands
   per loop iter in the BM=16 GEMM. BK=16 is the easy first step. On
   Dream-7B refine, BK=32 and BK=64 are also viable once you double
   WN from 2 → 4 (which halves per-SG register pressure — see gotcha
   #41 and the register-budget formula in catalog C2). Cumulative
   gains: BK=32 at WN=4 was -1.6% / -2.5% wall in the two tile paths;
   BK=64 at WN=4 in the BN=32 path was **-10% wall** (the single
   biggest mid-stage win on this port). At WN=2 BK=64 catastrophically
   spills — don't lower BK before trying WN=4 first.
5. **B5 fused residual** — same as for autoregressive, modest win.
6. **C4 aspect-ratio dispatch (K > N, BL ≥ 32 only)** — extend C2
   with `use_bm32 = (M > 16) && (K <= N)`. The MLP down_proj (K=I,
   N=H, I >> H) is the only K > N matmul; at BM=32 BN=64 it has too
   few N-tiles to fill the GPU, collapsing effective bandwidth.
   Routing it to BM=16 doubles TG count. In our Dream-7B BL=32 port:
   down_proj 1082us → 741us, wall 1.72 s → 1.52 s (12% win, beats MLX
   1.56 s).
7. **C5 aspect-ratio dispatch (small-N refine)** — N-axis sibling of
   C4. Even within the BM=16 path, refine matmuls with N=3584 at
   BN=64 produce only 56 N-tiles — under the M4 Max ~80 TG-slot
   target. Extend the dispatcher with `use_bn32 = (M <= 16) && (N <
   5120)`. Affects qkv, o_proj, down_proj in Dream-7B refine. -5.4%
   wall on the BL=16 bench. (Compose with C4: 3-way routing —
   prefetch-wide → BN=64 BM=32; refine small-N → BN=32 BM=16; else →
   BN=64 BM=16.)
8. **D5 SDPA K-per-iter ILP unrolling** — even after D1 (parallel
   softmax), the inner `lk` loop has a serial dependency chain
   between K-fetch and `simd_sum`. Unroll the QK and AV phases by
   N ∈ {2, 4, 8} K-vectors per outer iter with N independent
   accumulators. On Dream-7B refine: -22% per call at N=2, -33% on
   top at N=4, -57% on top at N=8 (cumulative -81%, 224 µs → 42 µs
   per call, ~7% of wall). Three commits, each safely revertable.
9. **F4 Multi-SG-per-row RMSNorm** — at M=BL=16 the 1-SG-per-row
   rmsnorm launches only 16 SGs total, well below the GPU's
   capacity. Spawn N_SGs ∈ {4, 16} SGs per row, merge partials via
   TG-mem. On Dream-7B (D=3584): 4 SGs/row -4% wall, then 16
   SGs/row -3% wall on top. Per-call: 62 µs → 9 µs (-85%). Same
   principle as D3 (multi-SG SDPA), different kernel category.

Skip A4 (no usable overlap — refine step N+1's ids depend on step N's
argmax). Skip A8 (the forward chain is mostly serial — Q/K/V→SDPA→o→
gate/up→down — with very little independent work to interleave).

See `references/diffusion-llm.md` for the full worked Dream-7B example
(naive ~10 s wall → 1.68 s gen / 2.17 s total wall on the BL=16
long bench, 39% faster than MLX gen / 19% faster than MLX total).

## The optimization catalog

The catalog is organised by bottleneck category, lettered A–I. Browse
the table below to find the section that matches the kernel your
profile flagged, then read the relevant entry in
**`references/catalog.md`** for the When/What/Speedup/Snippet/Commit
details.

| Section | Category                    | Key techniques                                                       |
|---------|-----------------------------|----------------------------------------------------------------------|
| A       | Host-side scheduling        | cmdbuf batching · param ring · 2-deep pipeline · concurrent encoder · parallel pread loader · parallel independent chains (A8) |
| B       | GEMV (decode Linear)        | SG-per-output · bfloat4 · qmv4 register tile · TG-mem X share · residual epilogue |
| C       | GEMM (prefill Linear)       | simdgroup_matrix MMA · tile-size sweep (incl. WN as register-pressure relief) · fused-op templates · aspect-ratio dispatch K>N (C4) AND small-N (C5) |
| D       | SDPA (attention)            | parallel softmax · online softmax · multi-SG · K-per-iter ILP unrolling · sliding KV cache       |
| E       | MoE                         | decode qmv4 · fused gate_up_swiglu · prefill sorted-gather GEMM · fused combine_scatter |
| F       | Reductions                  | parallel argmax · parallel topk softmax · per-row softmax+argmax (F3, diffusion samplers) · multi-SG-per-row rmsnorm (F4, large D) |
| G       | Decode/Prefill split        | two implementations, dispatch by Lq                                   |
| H       | Host/GPU boundary cleanup   | glue kernels (H1, often biggest single win) · embed+forward+argmax merge · output-row pruning via X offset (H3) |
| I       | Recurrent / SSM             | SG-per-(state_row, state_dim)                                         |

Each catalog entry points to a code snippet under `patterns/` — those
are **inspiration**, not drop-ins. Adapt to your model's shapes.

## Validation discipline

**After every change**, run the same validation suite:

1. Per-kernel correctness: the test for the touched kernel must pass
   against the C reference (`src-cpu`) within tolerance.
2. End-to-end token match against the previous Metal commit on the
   same prompt. First N tokens (N ≥ 16 ideally) MUST match. If they
   don't, investigate before committing.
3. Tok/s measurement (decode + prefill) on a fixed prompt + max-tokens.
   Record before/after numbers in the commit message.
4. If regression on either tok/s or correctness: revert. Try a
   different technique or different tile size.

When tokens diverge beyond the first N but match for the first N: this
is *probably* pure numerical drift. To confirm: dump per-layer
intermediates from both the old commit and the new commit on the SAME
input ids; verify they agree to bf16 tolerance. If yes, the divergence
is purely numerical and acceptable. If no, you have a bug — revert.

For symptom-driven debugging recipes (token divergence localization,
"decode fast / prefill slow", etc.), see **`references/debugging.md`**.

## Top pitfalls (read these before you start)

Each is a war story that cost at least one debug iteration. Full list
in `references/gotchas.md`; here are the ones most likely to bite during
optimization:

- **`dispatchThreads:` takes TOTAL THREADS, not threadgroups**. For
  SG-per-output kernels, you want `grid_x = 32 * N`. Passing `N` directly
  silently launches `ceil(N/32)` SGs. (gotcha #2)
- **KV cache `max_ctx` and SDPA TG-mem `MAX_CTX` must agree AND be ≥
  prompt_len + max_tokens**. Silent buffer overruns produce wrong
  tokens that look like numerical drift past the first ~32 tokens.
  (gotchas #3, #15)
- **`gpu_busy << wall` after batched dispatches** → host-side breaks
  (look for `commit_wait` inside `forward()`); apply H1 glue kernels.
  Usually the single biggest win. (gotcha #9)
- **Argmax over VOCAB=200k single-threaded** is by far the most common
  "why is decode capped at <5 tok/s" answer (also caps mid-pipeline
  decode at ~40 tok/s once everything else is in). Apply F1 immediately.
- **K_OUT in qmv4 has a sweet spot.** 4 wins for q8 affine dequant; 8
  hurts (41 → 31 tok/s in one port) due to register spills. Measure
  before scaling up. (gotcha #8)
- **The first generated token IS the prefill argmax.** When wiring up a
  2-deep pipeline that primes itself, it's easy to skip emitting this
  token, shifting all output by 1. (gotcha #10)
- **Concurrent encoder + missing barrier** = non-deterministic
  divergence that often passes per-kernel tests (which serialize one
  kernel at a time) but fails end-to-end. (gotchas #18, #26)
- **bfloat4 alignment.** Load `bfloat4` only at 8-byte-aligned
  addresses, or you'll get garbage on some Apple GPU generations.
- **Premature multi-SG SDPA hides the real SDPA win.** Apply D1 (parallel
  softmax) BEFORE D3 (multi-SG split), or you'll measure +5–10% and
  conclude SDPA isn't worth optimizing. (gotcha #31)
- **Tile-size cargo-culting.** MLX's tile sizes for M2 are not optimal
  for M4 or M1. Always sweep `(BM, BN, WM, WN)` on your target
  machine. (gotcha #32)
- **Forgetting GQA in the SDPA tile.** When each SG handles a
  `(query_token, head_q)` pair, the K/V head index is
  `head_kv = head_q / (Nq/Nkv)` — not `head_q`. Easy to break during
  tile refactorings. (gotcha #33)
- **Fused-residual epilogue shifts the bf16 round point.** Fusing
  `y = W·x + residual` rounds once (in f32) instead of twice. Token
  output may legitimately differ; regenerate the oracle dump with the
  same fusion OR loosen tolerance and verify against MLX greedy.
  (gotcha #34)
- **Persistent param buffer + 2-deep pipeline = race.** Duplicate the
  param ring (`gpu_param_buf[slot]`, `slot ∈ {0,1}`) the same way you
  duplicated the id ring. (gotcha #35)
- **Diffusion-LLM samplers do CPU work invisible to `gpu_busy`.** If
  your per-step CPU loop computes softmax / argmax / confidence over V
  for many positions, that time shows up only in wall, not gpu_busy.
  Add a third accumulator `host_post`; if > ~5% of wall, apply F3.
  (gotcha #36)
- **Two M shapes need two PSOs.** When prefetch (M=L) and refine
  (M=BL) differ by 4–5×, compile two GEMM PSOs with different
  `(BM, BK)` and switch at dispatch by `M > 16`. Pad workspace buffers
  to the larger `BM`. (gotcha #38)
- **K > N matmuls are TG-starved at BM=32.** A LLaMA-style MLP
  down_proj (K=I, N=H with I >> H) at BM=32 BN=64 produces only N/BN
  TGs — often well below the GPU's TG-slot capacity, collapsing
  effective bandwidth to ~30% of peak. Extend the dual-tile dispatch
  with `use_bm32 = (M > 16) && (K <= N)` so K > N shapes route to the
  smaller BM and double their TG count. 12% wall win on Dream-7B BL=32
  refine from a single-line change. (gotcha #39, catalog C4)
- **Small-N REFINE matmuls are also TG-starved — but on the N-axis.**
  The N-axis sibling of the above: refine (M ≤ 16, already in BM=16)
  with small N (3584 etc.) at BN=64 produces too few N-tiles to fill
  the GPU. Extend the dispatcher with `use_bn32 = (M <= 16) && (N <
  5120)` to halve BN and double N-tiles. -5.4% wall on Dream-7B BL=16
  refine bench. (catalog C5)
- **"BK=16 is the maximum" is a WN=2 artifact, not a fundamental
  ceiling.** Per-SG register pressure (which sets the spill ceiling)
  drops by ~half when you go `WN=2` → `WN=4`, often unlocking BK=32 or
  BK=64. -6% to -10% wall on Dream-7B refine from this realization,
  per tile. Always sweep BK ∈ {16, 32, 64} with the matching WN. (gotcha
  #41, catalog C2 + register-budget formula)
- **SDPA K-per-iter ILP unrolling is a separate axis from D1/D2/D3.**
  Even after parallel/online/multi-SG SDPA, the inner `lk` loop has a
  serial dependency chain. Unrolling 8 K-vectors per iter with 8
  independent dot accumulators lets the compiler interleave loads and
  arithmetic. -81% cumulative SDPA per-call on Dream-7B refine across
  three commits (N=2→4→8). (catalog D5)
- **1-SG-per-row rmsnorm undersubscribes the GPU when M is small.**
  At `M = BL = 16`, 1 SG/row launches only 16 SGs total — vs M4 Max's
  ~160 SG-slot capacity. The kernel becomes memory-latency-bound
  (62 µs/call when peak BW says < 1 µs). Spawn 4 or 16 SGs/row, merge
  the per-SG partials through TG-mem. -7% wall on Dream-7B refine
  cumulative (4 SGs/row + later 16 SGs/row). Same principle as D3 for
  SDPA, different kernel. (catalog F4)
- **Weight loading via `mmap + memcpy` is page-fault-bound at ~1 GB/s
  on macOS.** Even with `dispatch_apply` parallelism, you cap around
  4 GB/s aggregate because the macOS VM fault handler serializes on
  per-page locks. Use **parallel `pread()` per shard** (A9) — one fd
  per shard, sequential reads inside, `dispatch_apply` across shards.
  Brings startup below MLX on the same machine. Especially important
  for diffusion LLMs / short benches where gen is comparable in size
  to load. (gotcha #29, catalog A9)
- **`newBufferWithBytesNoCopy:` per-tensor isn't a shortcut.** It
  looks attractive (skip the bulk copy entirely!) but tensor data
  offsets inside a safetensors shard are typically 8-byte aligned,
  not the 16 KB page-aligned that `newBufferWithBytesNoCopy:` requires
  on Apple Silicon. The only viable zero-copy approach is wrap the
  *whole shard* as one Metal buffer + plumb per-tensor byte offsets
  through every dispatch helper, which is invasive. Parallel pread
  (A9) is simpler and already beats MLX. (gotcha #40)

## Commit strategy

One commit per technique. Each commit message should include:

```
src-metal: <technique short name> (decode +X.X%, prefill +Y.Y%)

- describe the change in 1–3 lines
- mention the bottleneck-before kernel
- mention the kernel-after improvement
- include before/after tok/s if material
```

Examples from real ports:

```
src-metal: gemv_bf16_4 — 4 outputs/simdgroup, 64 threads/TG (mlx qmv_fast pattern)
SDPA: simdgroup-per-(head,token) parallelism (1.96x decode)
src-metal: BM=16 BN=32 WM=1 WN=2 — match MLX non-NAX tile sizes (+26%)
src-metal: pipeline decode (depth=2) — overlap CPU encode with GPU compute
decode: 8-X register tile in bf16 GEMV (q/k/v/o/router/lm_head)
src-metal: fuse res_attn into gemm_bf16 (template DO_ADD); drop residual_add kernel
```

These commit messages tell you exactly what was changed AND give the
expected speedup, so when you bisect a regression later you can find
it fast.

## Hand-off / wrap-up

When the skill is done:

1. `./src-metal/PERF.md` documents final tok/s vs MLX on the target
   machine.
2. The optimized `./src-metal/` still passes per-kernel correctness
   against `./src-cpu/` within tolerance.
3. End-to-end tokens match `./src-cpu/` for the validation prompt for
   at least the first 16 tokens.
4. Each optimization is its own commit, individually revertable.

Optional follow-ups (out of scope for this skill):

- Port the optimized src-metal/ to CUDA / ROCm / Vulkan / TPU (each
  its own future skill, starting from the C reference, not from this
  Metal implementation).
- Add batch>1 support.
- Add longer-context / paged-KV cache.
- Add sampling beyond greedy.

## Stop condition

When both prefill and decode are within ±5% of MLX on the validation
prompt, you are done. Document the final tok/s and machine in a
`src-metal/PERF.md` and write a summary commit:

```
src-metal: optimization complete (prefill X.X tok/s vs MLX Y.Y, decode A.A vs B.B)
```

## Bundled material

### Code-shaped reference (`patterns/`)

Each catalog entry's "Snippet" points here. Files are working kernels /
shim snippets lightly annotated; each has a header block describing
what / when / expected speedup / original commit.

```
patterns/
├── cmdbuf_batch_dispatches.c            (A1)
├── cmdbuf_pipeline_2deep.c              (A4)
├── cmdbuf_pipeline_2deep_id_swap.c      (A4 with id-swap)
├── concurrent_encoder.c                 (A5)
├── load_parallel_pread.c                (A9 — startup beats MLX)
├── param_buf_persistent.c               (A2)
├── param_buf_const.c                    (A3)
├── gemv_simdgroup_per_output.metal      (B1, B2 variant inline)
├── gemv_qmv4_register_tile.metal        (B3)
├── gemv_with_residual_epilogue.metal    (B5)
├── gemm_simdgroup_matrix_mma.metal      (C1)
├── sdpa_sg_per_head_decode.metal        (D1 — first SDPA win)
├── sdpa_online_softmax.metal            (D2)
├── sdpa_multi_sg_online_merge.metal     (D3 — endgame SDPA)
├── mxfp4_qmv4_decode.metal              (E1)
├── mxfp4_gate_up_swiglu_fused.metal     (E2)
├── moe_sorted_gather_glue.metal         (E3 glue kernels)
├── moe_sorted_gather_qmm.metal          (E3 MMA reduced-precision GEMM)
├── argmax_parallel.metal                (F1)
├── topk_parallel_router.metal           (F2)
├── softmax_argmax_per_row.metal         (F3 — diffusion samplers)
├── gemm_x_offset_row_prune.c            (H3 — single-row output pruning)
├── glue_kernels_no_host_break.metal     (H1 — often biggest single win)
└── recurrent_state_sg_per_row.metal     (I1 for SSM/RNN)
```

### Long-form references (`references/`)

Loaded on demand — read the one your situation calls for.

```
references/
├── catalog.md                  Full A–I optimization catalog (the menu).
├── profiling.md                wall vs gpu_busy diagnostic, host_post for samplers, KPROF, variance discipline.
├── debugging.md                Symptom → recipe table + late-stage barrier checklist.
├── gotchas.md                  38 numbered war stories. Has its own TOC.
├── decode-prefill-split.md     G — when and how to maintain two kernel paths.
├── parallel-chains.md          A8 — the late-decode end-game win.
├── sliding-kv-cache.md         D4 — rotating fixed-size KV cache index math.
└── diffusion-llm.md            Diffusion-LLM (Dream / LLaDA / fastdllm) attack order, what works, what doesn't.
```

When you start the skill, **read `references/gotchas.md` first** — many
of the gotchas there will save you from rediscovering them the hard
way. The TOC at the top lets you scan in under a minute.
