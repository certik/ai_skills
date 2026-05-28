---
name: optimize-cpu
description: >
  Optimize a naive pure-C CPU LLM inference reference (./src-cpu/ from
  build-c-reference) to run as fast as possible on a multi-core x86_64
  (or aarch64) CPU. Catalog: -O3 -march=native -ffast-math -fopenmp -flto
  baseline; OpenMP-per-output GEMV; K-tile split bf16->fp32-from-FMA so
  gcc auto-vectorizes as vfmadd231ps + vpmovzxwd+vpslld (THE big GEMV
  win on bf16 / CPUs without bf16-FMA); real GEMM tile for M>1 prefill
  (W loaded once per N-tile, reused across M); KPROF per-kernel timer to
  confirm `linear` dominates; mmap MADV_HUGEPAGE + multi-threaded
  parallel page prefault for fast warm cache; OMP_NUM_THREADS = physical
  cores (NOT SMT-doubled); OMP_WAIT_POLICY=active + OMP_PROC_BIND=close
  OMP_PLACES=cores; DRAM-bandwidth ceiling stop condition.
  Pure C only -- no inline asm and no SIMD intrinsics (no _mm_*, no
  arm_neon), but ANY compiler flag and gcc -S asm inspection is
  encouraged to coax auto-vectorization.
  Validates tokens against a saved reference output after each change;
  accepts numerical drift past first ~50 tokens when -ffast-math
  reorders the fp32 reduction. Use this skill whenever the user wants
  to make a C/C++ LLM inference run faster on a CPU, mentions
  OpenMP/auto-vectorization/AVX2/NEON for an LLM forward pass, says
  "optimize src-cpu" or "speed up the C reference" or "make decode
  faster on the CPU", or asks for a CPU analog of optimize-metal -- even
  if they don't explicitly say "skill".
---

# optimize-cpu — make the pure-C CPU reference fast

Take a working but slow pure-C LLM inference reference (`./src-cpu/`,
produced by `build-c-reference`) and iteratively optimize it until decode
throughput is at the DRAM-bandwidth ceiling of the target CPU and prefill
is well past its initial naive auto-vectorization plateau.

The optimization is **not** a single rewrite. It is a disciplined
iteration: profile, identify the top-1 bottleneck, look up the next
applicable technique in the catalog, apply it, validate against a saved
token reference, commit. Repeat.

## When to use

After `build-c-reference` has produced a correct-but-slow `./src-cpu/`.
Use this skill to:

- Profile the C kernels and identify per-kernel bottlenecks.
- Apply named optimization techniques (catalog) and verify correctness
  after each.
- Drive performance to the DRAM-BW ceiling for decode and to a plateau
  for prefill (well past the autovec-baseline tok/s).

**Do not** use this skill to fix correctness bugs. If a kernel produces
wrong output before optimization, fix that in `build-c-reference` first.

**Hard rule: no inline assembly, no SIMD intrinsics** (no `_mm_*`, no
`__m256`, no `arm_neon.h`). The whole point of the C reference is to
stay portable. Compiler flags, `__attribute__((aligned))`, `__restrict__`,
`#pragma omp simd`, and reading `gcc -S` output to coax
auto-vectorization are all fair game and encouraged.

`__builtin_prefetch` and other gcc/clang builtins are a grey area —
they ARE compiler intrinsics that emit hardware-specific instructions.
Default to not using them. Ask the user if you think a specific one will
materially help.

## Pipeline

```
build-c-reference  →  optimize-cpu    (this skill)
                  ↘
                    port-c-to-metal  →  optimize-metal
```

Optimize-cpu and the Metal pipeline are independent branches off the
same C reference — you can apply this skill without ever touching the
Metal side.

## Prerequisites

- `./src-cpu/` exists, builds, and produces sensible tokens on the
  validation prompt.
- A multi-core CPU (this skill assumes ≥ 8 physical cores; everything
  still applies on fewer, but the SMT-vs-physical-cores tuning has less
  headroom).
- gcc 11+ (or clang 14+ with libomp installed). Both auto-vectorize
  bf16->fp32+FMA loops well when the loop is split (see B-K-tile below);
  neither vectorizes the fused loop.
- A canonical benchmark prompt + max-tokens that takes 5–30 s on the
  unoptimized build. The Gemma 4 E4B example uses
  `--prompt "Solve x^2 + x + 1 = 0" --max-tokens 64` (21 prompt + 64
  decoded tokens) — long enough that prefill and decode are both
  meaningfully timed, short enough to iterate.

## Inputs to gather

1. Path to `./src-cpu/` (default: sibling of this skill's working dir).
2. Path to the model weights (passed as `--model` to the binary).
3. The canonical benchmark prompt + max-tokens. Save a reference output
   to `tools/ref_tokens.txt` early — you will diff against it after
   every change.
4. CPU spec (physical cores, SMT factor, AVX2/AVX-512/NEON, L1/L2/L3
   sizes, DRAM channel count) — affects thread count, tile size, and
   ceiling estimation. `lscpu`, `cat /proc/cpuinfo`, `numactl -H` are
   your friends.

## Success criteria

- **Decode tok/s within 5% of the measured DRAM-BW ceiling** for the
  model's per-token weight read (see `references/profiling.md` —
  "Measuring the DRAM-BW ceiling"). For bf16 LLMs this is the hard
  upper bound on this hardware; only weight quantization (q8, q4,
  mxfp4 — out of scope here) can break it.
- **Prefill tok/s** stops improving with applied techniques (compute-
  bound plateau, ~30% off theoretical FMA peak is normal for an
  auto-vectorized GEMM).
- First N generated tokens (N ≥ 16 ideally) match the saved reference
  output. Later tokens may diverge due to pure numerical drift from
  `-ffast-math` reordering the fp32 reduction within each output's dot
  product (see Validation discipline below); that is acceptable as
  long as the divergence is *purely numerical* (no bug) AND the output
  is still semantically correct (still solves the math problem, still
  answers the prompt coherently).
- All optimizations committed individually so any one can be reverted.
- `./src-cpu/PERF.md` updated with the numbers table, what-worked,
  what-didn't-work, and theoretical ceiling.

## The workflow loop

Single, repeating cycle:

1. **Profile** current build with `KPROF=1 ./tools/bench.sh` — print
   per-kernel wall time, calls, us/call, and percent of total.
   Identify the hottest kernel (almost always `linear` for an LLM).
2. **Pick** next technique — look up the bottleneck category in the
   catalog (`references/catalog.md`); choose the next applicable,
   not-yet-applied entry.
3. **Implement** ONLY that one technique on a WIP commit.
4. **Validate**:
   - Build, run `./tools/bench.sh`, diff the generated tokens against
     `tools/ref_tokens.txt`.
   - If first N tokens (N ≥ 16) still match → numerical change is
     bounded enough to be safe; you may proceed even if later tokens
     differ (see Validation discipline).
   - If tokens diverge inside the first ~16, you have a bug. Revert.
5. **Measure** — speedup ≥ 1.02× on the dominant phase → commit.
   Otherwise revert.
6. **Stop** when decode is within 5% of the DRAM-BW ceiling
   AND prefill stops improving. Otherwise back to step 1.

## Profiling — the key diagnostic

Always integrate a per-kernel timer (KPROF) as the **second** thing you
do (right after turning on `-O3 -march=native -fopenmp -ffast-math`). It
takes ~5 minutes to add and answers the most important question: which
kernel actually matters?

For a typical dense-bf16 LLM the answer is overwhelmingly **`linear` at
95–98% of total time**. Everything else — rmsnorm, sdpa, rope, geglu,
residual_add — combined is under 5%. This pins all attention to the GEMV
inner loop and frees you from spending time on micro-optimisations of
already-cheap kernels.

KPROF integration pattern (60 lines total): see
**`patterns/kprof.{h,c}`** and the wrapping pattern at the top of
`patterns/kernels_kprof.c`. Build with `KPROF=1 ./gemma4-cpu ...` to get
the per-kernel table printed at exit.

Full profiling guide (KPROF, DRAM-BW ceiling measurement, thread-count
sweep, variance discipline): see **`references/profiling.md`**.

## The empirical attack order

Validated on Gemma 4 E4B (bf16, 42 layers, 9 GB weights/token) on an
AMD EPYC 7763 VM (16 physical cores, AVX2+FMA+F16C, no AVX-512, single
NUMA node, ~50–65 GB/s DRAM): naive 0.2 → final 7.7 decode / 0.2 → 73
prefill tok/s.

**Go in this order unless your profile clearly says otherwise** — the
profile is always ground truth, but this list saves you most of the
exploration cost.

### Phase 1 — get into the modern century (~0.2 → ~7 decode / ~0.2 → ~9 prefill)

0. **`-O3 -march=native -mtune=native -funroll-loops -ffast-math -fopenmp`**
   (catalog C0). Single biggest factor — 45× decode by itself. The
   `-march=native` is what pulls in AVX2 / AVX-512 / NEON for
   auto-vectorization; without it you're stuck on the SSE2 baseline.
1. **OpenMP `parallel for schedule(static)` across the slow (output)
   axis of every kernel** (catalog B1, F1, etc.). For decode (M=1) on a
   GEMV this means parallel over N output columns. Same on rmsnorm,
   argmax, embed_gather. Use `OMP_NUM_THREADS = physical_cores` (NOT
   SMT-doubled) — see thread-count discussion below.
2. **Set `OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores`**
   in your bench runner. The libgomp default `passive` makes threads
   `sleep()` between regions; for thousands of small parallel-fors this
   wakeup cost is ~30% of decode time. `close + cores` pins threads
   onto distinct physical cores in NUMA-local order.

### Phase 2 — fix the broken GEMV (~7 → ~7.7 decode / ~9 → ~28 prefill)

3. **Add KPROF.** Confirm `linear` is 95%+ before optimizing anything
   else. Cheap (<1% overhead when enabled).
4. **K_OUT=4 register tile in linear (GEMV).** Compute 4 outputs in one
   inner loop pass to amortize the X load. Catalog B-tile.
5. **K-tile split bf16->fp32 from FMA — THE big GEMV win.**
   The naive fused inner loop:
   ```c
   for (k=0; k<K; k++) {
     float x = bf16_to_f32(xrow[k]);
     s0 += x * bf16_to_f32(w0[k]);
     ...
   }
   ```
   does NOT auto-vectorize on gcc 13 — the asm contains scalar
   `vfmadd132ss`. Splitting into two phases per `TILE_K=128` block
   (Phase A: convert TILE bf16 → TILE fp32 into a stack scratch tile;
   Phase B: pure-fp32 FMA reduction from scratch) lets both halves
   vectorize cleanly. **Single biggest commit in the whole pipeline
   after C0.** Verify by inspecting `gcc -S kernels.c -o /tmp/k.s` and
   counting `vfmadd231ps` (want lots) vs `vfmadd132ss` (want zero in
   the inner loop). Catalog B-K-tile. Code: `patterns/linear_dot4_ktile.c`.

### Phase 3 — fix prefill GEMM (~28 → ~60 prefill, decode unchanged)

6. **Real GEMM tile for M>1 (prefill).** A decode-shaped GEMV looped
   over M re-reads W from main memory M times. Reorder to
   `parallel for n0 in N: for k_tile: convert 4 W rows once; for m in M:
   FMA into acc[m, 0..3]` so W is loaded ONCE per N-tile and reused
   across all M X rows. **Pre-convert X once (M*K floats) before the
   parallel region.** Catalog B-GEMM-tile. Code:
   `patterns/linear_dot4xM_gemm_tile.c`.

### Phase 4 — fast iteration polish (~60 → ~73 prefill, decode unchanged)

7. **`-flto`.** Inter-procedural inlining across translation units.
   Catalog C0.
8. **`madvise(MADV_HUGEPAGE)` on the weight mmap.** Encourages the
   kernel to back the bf16 weight mapping with 2 MB pages. Tangible
   TLB-pressure relief during decode (a 50 MB MLP weight needs ~12,800
   4 KB TLB entries vs 25 with 2 MB pages). Catalog A9-hugepage.
9. **Parallel multi-thread weight prefault.** After `mmap`, touch one
   byte per 4 KB page from all OMP threads. ~2× faster than
   `MAP_POPULATE`'s single-threaded kernel readahead. Catalog A9-prefault.
   Code: `patterns/parallel_prefault.c`.

### Stop condition

After Phase 4, re-profile. On the Gemma 4 E4B example, decode = 7.7
tok/s reading ~9 GB/token = ~70 GB/s effective, with the parallel-sum
microbenchmark showing 50–65 GB/s DRAM ceiling. We are above the simple
ceiling thanks to L3 reuse across consecutive layers; we are within 5%
of the wall. **Stop.**

For prefill, you may try further optimizations (M-tile blocking inside
the GEMM, gate+up_proj fusion, fewer fork-joins via batched parallel
regions), but the absolute time is now small (0.3 s for 21 prompt
tokens) and the user-visible impact is tiny vs decode.

### Optimizations that often land in the noise band

(Validated on Gemma 4 E4B bf16; may differ on your model.)

- **K_OUT=8 register tile.** Same asymptotic memory traffic as K_OUT=4,
  larger register pressure, larger stack tile. Tied on bandwidth-bound
  decode. Stay at K_OUT=4.
- **TILE_K sizes other than 128.** TILE_K=64 (regresses prefill 64
  vs 73), TILE_K=256 (regresses 70 vs 73), TILE_K=512 (68 vs 73).
  128 fp32 = 512 bytes/row × 4 rows = 2 KB tile, fits L1 trivially
  with plenty of slack. Stick with 128 unless your CPU has unusual L1
  geometry.
- **`MAP_POPULATE` alone.** Single-threaded kernel readahead; ~2×
  slower than the parallel OMP prefault above. Use the prefault
  instead.
- **`-Ofast` beyond `-ffast-math`.** No measurable change at this
  point in the pipeline.
- **`aligned_alloc` per `linear` call.** ~0.1% of decode time, not
  worth a per-thread persistent scratch refactor.
- **Pre-converting bf16 weights to fp32 in memory.** Doubles DRAM
  traffic, halves decode speed. Trust me, I checked. (Decode is
  weight-BW-bound; conversion in cache is essentially free.)

### Skip / often-irrelevant for dense bf16

- Quantization (q8/q4/mxfp4) — out of scope here, but is the ONLY
  thing that breaks the bf16 DRAM-BW ceiling. If a user is unhappy
  with the decode number and your profile shows DRAM saturation,
  quantization is the answer, not more CPU optimization.

## Thread-count tuning

**Decode on a multi-NUMA-die CPU is memory-bandwidth-bound, not
compute-bound.** Above the physical-core count, SMT siblings share L1/L2
and fight for cache, hurting decode while gaining nothing on memory BW.

Rule of thumb (Gemma 4 E4B, EPYC 7763 16C/32T VM, decode tok/s):

| Threads | Decode | Notes |
|---------|--------|-------|
|       8 |  4.1   | undersaturated DRAM channels |
|  **16** |**7.7** | **physical core count — best** |
|      24 |  6.0   | SMT collisions on 8 cores |
|      32 |  6.2   | SMT collisions everywhere |

**Always cap `OMP_NUM_THREADS` at the physical-core count.** A
microbenchmark may show memory BW scaling beyond physical cores (e.g.
50 → 59 GB/s on this VM); the compute penalty from SMT contention
outweighs the BW win.

Multi-NUMA: bind to one NUMA node per process and use `numactl`. Beyond
the scope of the worked example; see `references/profiling.md` if you
hit it.

## Validation discipline

Save a reference output of the *first run with -O3 -fopenmp* (Phase 1
output) into `tools/ref_tokens.txt`. **Re-run and diff against it after
every change.** The bench script (`tools/bench.sh`) should:

- Set `OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES / OMP_WAIT_POLICY`
  to the canonical values.
- Run on the canonical prompt + max-tokens.
- Print prefill + decode tok/s.

**Token-drift policy.** With `-ffast-math` the vector FMA is permitted
to reorder additions vs scalar code. After bf16 round-trips (only 8
mantissa bits), borderline values flip and greedy argmax can diverge
after ~50 tokens. Three outcomes:

1. **Tokens match exactly** → no concern, commit.
2. **Tokens match for first N (N ≥ 16) then diverge** → numerical
   drift. Inspect the diverged output: does it still solve the problem
   coherently? If yes, this is acceptable. Update
   `tools/ref_tokens.txt` to the new output and commit. Note the drift
   in the commit message.
3. **Tokens diverge inside the first 16** → almost certainly a bug
   (off-by-one, wrong reduction, wrong tile boundary). REVERT and
   inspect.

For step 2, the user should bless the new reference before you commit
it. Don't unilaterally update the reference and accept arbitrary drift.

If tokens diverge and you can't tell whether it's a bug or drift: dump
the per-layer logit-residual into a file from BOTH the old and new
build on the SAME prompt and confirm they agree to bf16 tolerance
(~1e-3 relative). If they do, it's drift.

## DRAM-BW ceiling check

Before declaring decode "as fast as possible", you must measure the
ceiling on the target machine. Procedure:

1. Build a tiny parallel-sum microbenchmark: allocate `S = 1 GB`,
   write zeros to fault it in (or use the same OMP-prefault pattern
   as for weights), then `#pragma omp parallel for reduction(+:s)`
   sum every byte interpreted as float. Time it.
2. `effective_bw = bytes_read / time` = your "ceiling".
3. Compute the per-token weight bytes: sum sizeof(W) over every layer
   linear's W you read per forward pass, + lm_head + per-layer-embed
   + per-layer-embed-projection if present. For Gemma 4 E4B this is
   ~9 GB.
4. `ceiling_tps = effective_bw / per_token_bytes`. If your decode is
   within 5% of `ceiling_tps`, **STOP** — no kernel-level technique
   below quantization can break this wall.

See `references/profiling.md` for the full procedure + worked numbers.

## Commit strategy

One commit per technique. Each commit message names the technique and
the measured speedup, so a future bisect can find regressions fast:

```
src-cpu: <technique short name> (decode +X.X%, prefill +Y.Y%)

- 1–3 lines: what changed, before/after kernel, before/after tok/s
```

Examples (from the Gemma 4 E4B port):

```
src-cpu: -O3 -march=native -ffast-math -fopenmp (prefill 0.2→1.5, decode 0.2→1.3 tok/s)
src-cpu: OpenMP + auto-vectorized AVX2 kernels (decode 1.3→7.5, prefill 1.5→9.6 tok/s)
src-cpu: K-tile linear_dot4 (split bf16->fp32 from FMA); prefill 8.9 -> 27.6 tok/s
src-cpu: turn linear M>1 path into real GEMM (prefill 27.6 -> 58.9 tok/s)
src-cpu: enable LTO (prefill 58.9 -> 66.4 tok/s)
src-cpu: MAP_POPULATE + MADV_HUGEPAGE for weights (warmer page cache)
src-cpu: parallel weight prefault (faster than MAP_POPULATE)
```

## Top pitfalls (read these before you start)

The curated list of war stories most likely to bite during a CPU
optimization session lives in **`references/pitfalls.md`**. Read it
once before you start.

The absolute must-internalize-before-you-touch-anything subset:

- **gcc can't auto-vectorize a fused bf16->fp32+FMA inner loop.** If
  you see scalar `vfmadd132ss` in `gcc -S` for your hot GEMV, split
  the loop into convert-then-FMA phases. THE biggest commit.
- **`OMP_NUM_THREADS` defaults to logical CPUs, not physical cores.**
  This actively hurts decode (BW-bound, SMT collides on L1/L2).
  Always set to physical core count.
- **`OMP_WAIT_POLICY` default is `passive`.** With thousands of small
  parallel-for regions this wakeup cost is ~30% of decode. Set to
  `active`.
- **`-march=native` is not implied by `-O3`.** Without it you're on
  the SSE2 baseline (8-byte vectors) instead of AVX2 (32-byte) — 4×
  slower.
- **`-ffast-math` allows reordering of reductions** → bit-for-bit
  reproducibility lost. Decide upfront with the user whether token
  drift is acceptable (it almost always is for inference; loss-of-
  reproducibility is the price of admission to GEMV vectorization).
- **Single-threaded `mmap + MAP_POPULATE` for weights** is page-fault-
  bound. Use parallel OMP touch on top of plain `mmap`.
- **`madvise(MADV_HUGEPAGE)` needs system-wide THP enabled.** Check
  `cat /sys/kernel/mm/transparent_hugepage/enabled` — `[always]` is
  what you want. If `[never]`, the hint is a no-op.

Full list (with reproductions and fixes) in `references/pitfalls.md`.

## Hand-off / stop condition

Done when ALL of:

1. **Decode is within 5% of the measured DRAM-BW ceiling** for the
   per-token weight bytes (see DRAM-BW ceiling check above).
2. **Prefill stops improving** after applying applicable catalog
   entries; you're within ~30% of theoretical FMA peak (full peak is
   unreachable for auto-vectorized GEMM on a CPU without
   hand-tuned MMA intrinsics — which this skill forbids).
3. **Tokens match the saved reference for the first N ≥ 16** under
   the canonical bench.
4. **`./src-cpu/PERF.md` is updated** with the numbers table,
   what-worked, what-didn't-work, ceiling measurement, and attack-
   order recipe.

Write a summary commit:

```
src-cpu: optimization complete (prefill X.X tok/s, decode Y.Y tok/s, Z% of BW ceiling)
```

Optional follow-ups (out of scope here, but worth flagging to the
user):

- **Weight quantization** (q8/q4/mxfp4) is the only thing that can
  break the bf16 DRAM-BW ceiling. Decode tok/s scales inversely with
  weight bytes / token. A q4 build of the same model would 4× decode.
  This is the natural next step but requires a separate skill / model
  conversion pipeline.

## Bundled material

### Code-shaped reference (`patterns/`)

Each catalog entry's "Snippet" points here. Files are working
kernel snippets with a header describing what / when / expected
speedup / original commit. Adapt to your model's shapes — these are
inspiration, not drop-ins.

- `patterns/linear_dot4_ktile.c` — K-tile split bf16->fp32 from FMA
  (B-K-tile, THE big GEMV win).
- `patterns/linear_dot4xM_gemm_tile.c` — proper GEMM tile for M>1
  prefill (B-GEMM-tile).
- `patterns/parallel_prefault.c` — multi-threaded weight prefault
  after `mmap` (A9-prefault).
- `patterns/kprof.h`, `patterns/kprof.c`, `patterns/kernels_kprof.c`
  — per-kernel timer integration (KPROF).
- `patterns/bench.sh` — canonical benchmark script (OMP env +
  bench run + tokens diff).
- `patterns/Makefile.fragment` — recommended `CC / CFLAGS / LDFLAGS`
  for an optimization build.

### Long-form references (`references/`)

Loaded on demand — read the one your situation calls for.

- `catalog.md` — full optimization catalog by bottleneck category
  (the menu).
- `profiling.md` — KPROF integration, DRAM-BW ceiling measurement,
  thread-count sweep, variance discipline, asm-inspection workflow.
- `pitfalls.md` — curated war stories likely to bite during a CPU
  optimization session.

When you start the skill, **read `references/pitfalls.md` first**
(~5 minutes) — it covers the war stories most likely to bite during
the session in front of you. Then keep `references/catalog.md` open
for technique lookup.
