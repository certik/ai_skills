---
name: optimize-metal
description: >
  Optimize a naive Apple-GPU Metal LLM implementation (./src-metal/
  from port-c-to-metal) to match MLX speed within ±5% on per-token
  tok/s AND total wall time. Catalog: SIMD-group-per-output GEMV,
  bfloat4 loads, qmv4 register tiling, simdgroup_matrix MMA for
  prefill, sorted-gather MoE, online-softmax + multi-SG SDPA, parallel
  argmax, 2-deep cmdbuf pipeline, persistent/const buffers, concurrent
  encoder, fused residual epilogues, decode/prefill split, GPU glue
  kernels for host-side breaks (often the biggest decode win),
  SG-per-(row,dim) for RNN/SSM recurrences, parallel pread weight
  loader (one fd per shard via dispatch_apply, beats mmap+memcpy 2–3×).
  Also handles diffusion-LLM samplers (Dream / LLaDA / fastdllm): GPU
  per-row softmax+argmax+confidence, single-row output pruning,
  dual-tile GEMM dispatch by M. Validates tokens against the C
  reference after each change. Triggers: optimize metal, speed up
  metal, match mlx speed, gpu optimization, kernel tuning,
  startup latency, diffusion LLM on metal.
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
  meaningful fraction of total (short benches, diffusion LLMs where
  gen is ~1 s — load can be 30–50% of total). Hitting per-token
  parity but losing total because weight load is 2× slower is a real
  failure mode the tok/s metric hides.
- First N generated tokens (N ≥ 4, ideally 16) match the C reference
  exactly. Later tokens may diverge due to pure numerical drift; that
  is acceptable as long as the divergence is *purely numerical* (no
  bug) and you can demonstrate it via per-kernel tolerance checks
  against the C reference.
- All optimizations committed individually so any one can be reverted.

## The workflow loop

Single, repeating cycle:

1. **Profile** current build — decode + prefill tok/s, print BOTH
   wall and gpu_busy (the key diagnostic); identify hottest kernel.
2. **Pick** next technique — look up the bottleneck category in the
   catalog; choose the next applicable, not-yet-applied entry.
3. **Implement** ONLY that one technique on a WIP commit.
4. **Validate** — per-kernel correctness against C reference; first
   N tokens (N ≥ 16 ideally) match exactly end-to-end. If regression,
   revert; try a different technique.
5. **Measure** — speedup ≥ 1.02× on the dominant phase → commit.
   Otherwise revert.
6. **Stop** when both prefill and decode are within ±5% of MLX
   (and total wall too, if startup is a meaningful fraction).
   Otherwise back to step 1.

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

**Diffusion-LLM samplers need a third number — `host_post`.** Per-step
softmax + argmax + confidence over V often looks like a scheduling
problem (A/H) but is actually a sampler problem (F3): the CPU is
genuinely the gap. See `references/profiling.md` and
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
and total wall from 3.08 s → 2.17 s, beating MLX's 2.67 s by 19%.
The full A9 stack (sub-shard split + async compile + async residency
+ shard-buffer-views + no madvise) takes startup to 0.40 s and total
wall to 1.43 s on the same prompt. See catalog A9 and gotchas #28–30,
#45–47, #50.

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

### Second pass — pushing PAST MLX parity (~69.5 → ~102.5 tok/s for Qwen3.6)

Once you're at MLX parity (or above) it's tempting to stop. But the
parity bar is set by what MLX has been optimized for — modern Apple-
silicon chips often have enough headroom for *another* 40–50%. Three
techniques applied on top of MLX-parity Qwen3.6 took decode from
71.3 → 102.5 tok/s (+44%, +46% over MLX). Re-profile with KPROF
first; the bottleneck shifts dramatically after step 14.

15. **F4 multi-SG rmsnorm AT M=1** (autoregressive). At decode the
    rmsnorm dispatch is `M=1` → ONE SG total → 2.5% GPU
    utilization. Bump to `N_SG=4`. Qwen3.6 `D=2048`: 71.3 → 82.8
    tok/s (+16.1%). Branch in main.c by `Lq < 8` so prefill still
    uses single-SG. F4 is in the catalog as a diffusion technique
    — at M=1 it's an even bigger win because the under-subscription
    is 4–16× worse.
16. **B6 uint4 W loads in quantized GEMV.** Load packed W as
    `uint4` instead of `uint32` → reuse one `(scale, bias)` load
    per 16 q8 weights (the quant group). 4× fewer `(s, b)` device
    reads = the dominant cost of MLX-style quant dequant.
    Qwen3.6: 82.8 → 98.4 tok/s (+18.7%). Single biggest second-
    pass commit. See catalog B6.
17. **F1 2-stage parallel argmax FOR LARGE V.** Single-TG F1 still
    uses ONE GPU core (~253 µs/call at V=248k). Split into 64 TGs
    + 1 reducer TG (~10 µs total). Qwen3.6: 98.4 → 101.5 tok/s
    (+3.2%). Only matters once V is huge and everything else is
    optimized — but then the 1-core ceiling pops up clear as day.
18. **F4 N_SG sweep — revisit AFTER B6 + F1.** Once the heavy
    kernels shrink, the relative cost of rmsnorm grows back. Try
    bumping `N_SG=4` → `N_SG=8` (`N_SG=16` ties at D=2048; sweep
    per-D). Qwen3.6: 101.5 → 102.5 tok/s (+1.0%). Cheap, free
    1-line change.
19. **D3 N_SG sweep on LONG decode.** SDPA's relative cost grows
    linearly with Lk, so short-bench picks (e.g. `N_SG=4` from 64-
    token tests) become wrong at long context (1500+ tokens). On
    Qwen3.6-35B-A3B M4 Max, 1500-tok decode picks `N_SG=20` (not 16
    or 24) — see gotcha #54 and catalog D3 for the full sweep
    (12=91.2 < 16=96.7 ≤ **20=97.5** > 24=95.6 > 28=94.2 tok/s).
    Always re-sweep `N_SG ∈ {12, 16, 20, 24, 28}` once long-decode
    becomes the goal. Update BOTH the kernel `constexpr` AND the
    host dispatch `threads_per_threadgroup` in lock-step
    (gotcha #53) — desync gives garbage tokens. Worth +1-2% on
    long-context decode.

After these, KPROF shows `linear_q8` family at ~77% of GPU at ~50%
of theoretical W-BW peak — you're at the W-bandwidth floor on
quantized weights, and further wins require *algorithmic* changes
(mxfp4, different quant format, smaller weights). The remaining
small kernels (sdpa 3.3%, gated_delta 3.4%, silu_mul 2.1%, etc.)
each cap at <1% wall.

### Optimizations that often land in the noise band

(Validated on Qwen3.6-35B-A3B; may differ on your model.)

- bfloat4 vector loads in isolation (baked into B1 already).
- qmv4 K_OUT=8 — regressed at every layout we tried, including
  bare q8 (41 → 31 tok/s) AND post-B6 uint4 layout (103 → 90.7
  tok/s). The sweet spot only moves DOWN with heavier inner
  loops. Stay at K_OUT=4. (gotcha #8 has the second datapoint.)
- **B4 (TG-mem X-share) — risks regression once B6 is in.** The
  catalog speedup (1.05-1.10×) was measured on early ports where
  W-BW dominated and X-share saved 25% of DRAM. After B6 amortizes
  S/B, Apple-Silicon's per-core L1/L2 effectively gives you X
  sharing for free between adjacent SGs reading the same row, so
  explicit TG-mem staging adds cooperative-load + barrier cost
  with no DRAM savings. On Qwen3.6 M4 Max (K=2048), both WG=2 and
  WG=4 regressed 1-2%. Try if profile says X-BW is dominant; revert
  if it doesn't measure faster. (gotcha #56, catalog B4)
- Concurrent encoder ALONE without barrier surgery — A8 is the win.
- **Fusion across kernels that the concurrent encoder is already
  overlapping** — see gotcha #42. The Qwen3.6 fused
  `gate+up+silu_mul` was a *correct* kernel that regressed by 0.3%
  because it serialized a `gate || up` overlap. Cheap to try, but
  check the dispatch graph FIRST.

### Skip for decode-only goals

- C1/C2/C3 (GEMM MMA for prefill) — only relevant if your goal is to
  match prefill speed.

### Dense bf16 LLMs (Gemma 3 / Gemma 4 / Llama bf16 / Mistral bf16)

Same default order from B1/D1/H1 up through SDPA endgame, but the
late-stage list is different (per-layer `embed_gather` is the single
biggest win on Gemma 3/4; A8/B4/B6/K_OUT=8 all skip). The DRAM-BW
ceiling check often retires the model before chasing the last 1-2%.

Full ordered list + skip rules + worked Gemma 4 E4B example (5.6 →
49.7 tok/s, +3.1% over MLX) in **`references/dense-bf16.md`**.

### Diffusion LLMs (Dream / LLaDA / fastdllm)

NOT autoregressive — `forward` runs at two M shapes per block
(prefetch `M=L`, refine `M=BL`), `host_post` (sampler softmax + argmax
+ confidence over V) is large and invisible to `gpu_busy`, and total
wall (load + gen) matters as much as gen tok/s. Different attack
order: A9 → F3 (host_post → GPU) → H3 (single-row prefetch lm_head) →
C2 dual-tile-by-M → C2 BK/WN sweep → B5 → C4/C5 aspect-ratio routing
→ D5 SDPA K-per-iter ILP → F4 multi-SG rmsnorm. Skip A4 and A8.

Full attack order, skip rules, profile recipe, and worked Dream-7B
example (~10 s naive → 2.17 s total, +19% wall / +39% gen over MLX)
in **`references/diffusion-llm.md`**.

## The optimization catalog

The catalog is organised by bottleneck category, lettered A–I. Browse
the table below to find the section that matches the kernel your
profile flagged, then read the relevant entry in
**`references/catalog.md`** for the When/What/Speedup/Snippet/Commit
details.

| Section | Category                    | Key techniques                                                       |
|---------|-----------------------------|----------------------------------------------------------------------|
| A       | Host-side scheduling        | cmdbuf batching · param ring · 2-deep pipeline · concurrent encoder · parallel pread loader · parallel independent chains (A8) |
| B       | GEMV (decode Linear)        | SG-per-output · bfloat4 · qmv4 register tile · TG-mem X share · residual epilogue · uint4 W loads w/ amortized (s,b) for q-affine (B6) |
| C       | GEMM (prefill Linear)       | simdgroup_matrix MMA · tile-size sweep (incl. WN as register-pressure relief) · fused-op templates · aspect-ratio dispatch K>N (C4) AND small-N (C5) |
| D       | SDPA (attention)            | parallel softmax · online softmax · multi-SG · K-per-iter ILP unrolling · sliding KV cache       |
| E       | MoE                         | decode qmv4 · fused gate_up_swiglu · prefill sorted-gather GEMM · fused combine_scatter |
| F       | Reductions                  | parallel argmax · parallel topk softmax · per-row softmax+argmax (F3, diffusion samplers) · multi-SG-per-row rmsnorm (F4, large D) |
| G       | Decode/Prefill split        | two implementations, dispatch by Lq                                   |
| H       | Host/GPU boundary cleanup   | glue kernels (H1, often biggest single win) · embed+forward+argmax merge · output-row pruning via X offset (H3) · per-layer embed_gather TG-per-row (H4, Gemma 3/4) |
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

The curated short-list of war stories most likely to bite during an
optimization session lives in **`references/pitfalls.md`** — organised
by what kind of debugging session you're about to have (encoding
traps, correctness traps, profile-reading, tuning sweeps, regressions,
weight loader, "unreproducible" benches). Read it once before you
start; re-read whenever a benchmark goes sideways.

The full numbered war-story archive (60 entries with full
reproductions and fixes) is in **`references/gotchas.md`** — that's
where each curated pitfall links to.

The absolute must-internalize-before-you-touch-anything subset:

- **`dispatchThreads:` takes TOTAL THREADS, not threadgroups**. For
  SG-per-output kernels you want `grid_x = 32 * N`. Passing `N`
  silently launches `ceil(N/32)` SGs. (gotcha #2)
- **KV cache `max_ctx` and SDPA TG-mem `MAX_CTX` must agree AND be ≥
  prompt_len + max_tokens**. Silent overruns produce wrong tokens
  that look like numerical drift past ~32 tokens. (gotchas #3, #15)
- **`gpu_busy << wall` after batched dispatches** → host-side breaks
  (look for `commit_wait` inside `forward()`); apply H1 glue kernels.
  Usually the single biggest win in the whole pipeline. (gotcha #9)
- **Argmax over VOCAB=200k single-threaded** is by far the most common
  "why is decode capped at <5 tok/s" answer. Apply F1 immediately.
- **K_OUT in qmv4 has a sweet spot. Stay at 4.** 8 hurts at every
  precision measured (q8/uint4/bf16) due to register spills. Don't
  even sweep it on the way up. (gotchas #8, #60)
- **Concurrent encoder + missing barrier** = non-deterministic
  divergence that PASSES per-kernel tests (they serialize one kernel
  at a time) but FAILS end-to-end. (gotchas #18, #26)
- **Stale binary / thermal trap.** Unreproducible regression? Almost
  always `make clean && make`, GPU thermals, or DYLD cache. Standard
  bench discipline: `sleep 60` before first run, best-of-5 with
  `sleep 30` between. (gotcha #55)

Everything else — bfloat4 alignment, fused-residual round-point
shifts, `dispatchThreads` non-uniform last-TG, SDPA `N_SG` desync,
parallel pread weight loader, per-layer `embed_gather` on Gemma 3/4,
B4 regression once B6 is in, A8 uselessness on dense bf16, DRAM-BW
ceiling check, etc. — is in `references/pitfalls.md`. Don't skip it
in a real port; you WILL hit several of those.

## Commit strategy

One commit per technique. Each commit message names the technique and
the measured speedup, so a future bisect can find regressions fast:

```
src-metal: <technique short name> (decode +X.X%, prefill +Y.Y%)

- 1–3 lines: what changed, before/after kernel, before/after tok/s
```

Examples:

```
src-metal: gemv_bf16_4 — 4 outputs/simdgroup (mlx qmv_fast pattern)
SDPA: simdgroup-per-(head,token) parallelism (1.96x decode)
src-metal: BM=16 BN=32 WM=1 WN=2 — match MLX non-NAX tile sizes (+26%)
src-metal: fuse res_attn into gemm_bf16 (template DO_ADD)
```

## Hand-off / stop condition

Done when EITHER of:

1. **Both prefill and decode within ±5% of MLX** (plus total wall,
   if startup is a meaningful fraction — see Success criteria), OR
2. **You are at the DRAM-BW ceiling.** Compute `(per-token weight
   bytes) / (peak GB/s)` = theoretical minimum ms/token. If your
   measured ms/token is within 5% (≥ 95% of theoretical), the
   forward pass is reading weights at memory-controller saturation
   and no kernel-level technique in this catalog can help further.
   This is a HARD ceiling for bf16; only weight quantization
   (mxfp4/q4/q8 — out of scope here) breaks it. See gotcha #59.

Document final tok/s, machine, AND ceiling-% in `./src-metal/PERF.md`
and write a summary commit:

```
src-metal: optimization complete (prefill X.X tok/s vs MLX Y.Y, decode A.A vs B.B)
```

The optimized `./src-metal/` still passes per-kernel correctness
against `./src-cpu/` and end-to-end tokens match for the validation
prompt for at least the first 16 tokens. Each optimization is its
own commit, individually revertable.

Optional follow-ups (out of scope): port to CUDA/ROCm/Vulkan/TPU
(future skills, starting from the C reference), batch>1, longer-
context / paged-KV cache, sampling beyond greedy.

## Bundled material

### Code-shaped reference (`patterns/`)

Each catalog entry's "Snippet" points here. Files are working kernel /
shim snippets with a header describing what / when / expected speedup /
original commit. Adapt to your model's shapes — these are inspiration,
not drop-ins. Full inventory in `references/catalog.md`. Highlights:

- **A** (host scheduling) — `cmdbuf_*.c`, `concurrent_encoder.c`,
  `load_parallel_pread.c` (incl. sub-shard + shard-buffer-view variants),
  `param_buf_*.c`.
- **B/C** (GEMV/GEMM) — `gemv_*.metal`, `gemm_simdgroup_matrix_mma.metal`,
  `linear_q8_uint4_amortize_sb.metal` (B6, q-affine `(s,b)` amortization).
- **D** (SDPA) — `sdpa_sg_per_head_decode.metal`,
  `sdpa_online_softmax.metal`, `sdpa_multi_sg_online_merge.metal`.
- **E** (MoE) — `mxfp4_*.metal`, `moe_sorted_gather_*.metal`.
- **F** (reductions) — `argmax_parallel.metal`,
  `topk_parallel_router.metal`, `softmax_argmax_per_row.metal`.
- **H/I** — `glue_kernels_no_host_break.metal`,
  `gemm_x_offset_row_prune.c`, `recurrent_state_sg_per_row.metal`,
  `embed_gather_per_layer.metal` (TG-per-row + 128-thread bfloat4
  for per-layer embed gather — Gemma 3/4-style models, gotcha #57).

### Long-form references (`references/`)

Loaded on demand — read the one your situation calls for.

- `catalog.md` — full A–I optimization catalog (the menu, with TOC).
- `profiling.md` — wall vs gpu_busy diagnostic, host_post for samplers,
  KPROF, variance discipline, startup phase-breakdown.
- `debugging.md` — symptom → recipe table + late-stage barrier checklist.
- `pitfalls.md` — **curated short-list of war stories most likely to
  bite during optimization**, organised by debugging session type
  (encoding traps, correctness, profile-reading, tuning, regressions,
  weight loader, "unreproducible" benches). Each entry links into
  `gotchas.md`.
- `gotchas.md` — full 60-entry numbered war-story archive (with TOC).
  The deepest reference; read the TOC top-to-bottom once, then dive in
  when a specific gotcha number is referenced.
- `dense-bf16.md` — dense-bf16 attack order (Gemma 3/4, Llama bf16,
  Mistral bf16). Same B1/D1/H1/SDPA base, different late-stage list,
  DRAM-BW ceiling stop condition.
- `diffusion-llm.md` — Dream / LLaDA / fastdllm attack order + worked
  Dream-7B example.
- `decode-prefill-split.md` — G category.
- `parallel-chains.md` — A8 late-decode endgame.
- `sliding-kv-cache.md` — D4 rotating KV cache.

When you start the skill, **read `references/pitfalls.md` first**
(curated, ~10 minutes) — it covers the war stories most likely to bite
during the session in front of you. Then keep `references/gotchas.md`
open for when a specific number is cross-referenced.
