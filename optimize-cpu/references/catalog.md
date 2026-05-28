# optimize-cpu — full optimization catalog

The menu of techniques, organized by bottleneck category. Use this when the
SKILL.md attack order points you here, or when KPROF shows a bottleneck
outside the usual `linear`-dominates pattern.

Each entry has:
- **Category code** (e.g. B1-cpu) for cross-referencing.
- **When**: profile signature that says "use this".
- **How**: terse description.
- **Snippet**: pattern file under `patterns/` to copy from.
- **Expected speedup** (Gemma 4 E4B numbers; calibrate to your model).
- **Pitfalls**: things that bite during this technique.

## TOC

- [C0 — build flags](#c0--build-flags)
- [B1-cpu — OpenMP across the slow axis](#b1-cpu--openmp-across-the-slow-axis)
- [B-loop-tile — K_OUT=4 register tile](#b-loop-tile--k_out4-register-tile)
- [B-K-tile — split bf16→fp32 from FMA](#b-k-tile--split-bf16fp32-from-fma)
- [B-GEMM-tile — proper GEMM for M>1 prefill](#b-gemm-tile--proper-gemm-for-m1-prefill)
- [B5-cpu — fuse residual+rmsnorm (rarely needed)](#b5-cpu--fuse-residualrmsnorm-rarely-needed)
- [D1-cpu — SDPA parallelize over (lq, qh)](#d1-cpu--sdpa-parallelize-over-lq-qh)
- [F1-cpu — parallel argmax](#f1-cpu--parallel-argmax)
- [F4-cpu — parallel rmsnorm](#f4-cpu--parallel-rmsnorm)
- [H4-cpu — embed_gather bulk-memcpy](#h4-cpu--embed_gather-bulk-memcpy)
- [A9-cpu — weight loader (HUGEPAGE + parallel prefault)](#a9-cpu--weight-loader-hugepage--parallel-prefault)
- [J1-cpu — thread-count tuning + OMP env](#j1-cpu--thread-count-tuning--omp-env)
- [J2-cpu — KPROF](#j2-cpu--kprof)

---

## C0 — build flags

**When**: Always, before anything else.

**How**:
```
CFLAGS  := -O3 -march=native -mtune=native -funroll-loops -ffast-math \
           -fopenmp -flto
LDFLAGS := -fopenmp -flto -lm
```

Use `CC := gcc` (`:=` not `?=` — Make's built-in `cc` will silently win and
might be a different compiler with different defaults).

**Speedup**: 45x decode alone (`-O3 -march=native` does most of it).

**Pitfalls**:
- `-march=native` is NOT implied by `-O3`. Without it: SSE2 baseline.
- `-flto` must appear in BOTH `CFLAGS` and `LDFLAGS`.
- `-ffast-math` allows the vectorizer to reorder fp32 reductions; bf16's
  8 mantissa bits means borderline values flip and greedy argmax may
  diverge from the scalar reference after ~50 tokens. Decide upfront with
  the user that this is OK (it almost always is for inference).

**Snippet**: `patterns/Makefile.fragment`.

---

## B1-cpu — OpenMP across the slow axis

**When**: Every kernel, immediately after C0. The profile signature is
"single core pinned" (one thread at 100%, 15 others idle).

**How**: For each kernel, identify the slow output axis (N for linear, V
for argmax/softmax, M for rmsnorm if M is large, D otherwise) and add:
```c
#pragma omp parallel for schedule(static)
for (uint32_t n0 = 0; n0 < N; n0 += K_OUT) { ... }
```

For `linear_bf16` decode (M=1) this means parallel over output columns
N. Each thread independently computes K_OUT=4 dot products.

**Speedup**: ~5–10x on most kernels (linear pulls 5x because of the
register-tile, rmsnorm/argmax saturate at memory BW quickly).

**Pitfalls**:
- Tiny kernels (M=1 rmsnorm over D=2560) don't benefit — fork-join
  overhead drowns the parallel win. Use `if(...)` to disable when N
  is small.
- `schedule(dynamic)` adds overhead. Static is right for uniform work.

**Snippet**: `patterns/linear_dot4_ktile.c` shows the GEMV wiring; same
pattern applies to every other kernel.

---

## B-loop-tile — K_OUT=4 register tile

**When**: Right after B1-cpu, BEFORE B-K-tile. Compute 4 outputs of GEMV
in one inner-loop pass to amortize the X load.

**How**: Replace
```c
for (n=0; n<N; n++) {
    acc = 0;
    for (k=0; k<K; k++) acc += xf[k] * W[n*K + k];
    Y[n] = acc;
}
```
with a 4-output tile (see `linear_dot4_ktile.c`). K_OUT=4 is the sweet spot:
- K_OUT=2 leaves perf on the table.
- K_OUT=4 keeps 4 fp32 accumulators in registers and 4 W rows hot in L1.
- K_OUT=8 has worse register pressure on AVX2 (16 ymm registers — 4 for
  accumulators leaves 12 for tiles + X, tight) and gains essentially
  nothing on bandwidth-bound decode.

**Speedup**: ~2x prefill over plain BG1.

**Snippet**: `patterns/linear_dot4_ktile.c`.

---

## B-K-tile — split bf16→fp32 from FMA

**THE biggest GEMV-specific win** on bf16 weights / CPUs without bf16-FMA.

**When**: AFTER B-loop-tile. `gcc -S kernels.c` shows scalar
`vfmadd132ss` in the GEMV inner loop, NOT `vfmadd231ps`.

**How**: Convert TILE_K bf16 values → fp32 into a stack scratch, then
run a pure-fp32 FMA loop over the scratch:
```c
float b0[TILE_K] __attribute__((aligned(64)));
// Phase A — converts to vpmovzxwd + vpslld (vectorizes).
#pragma omp simd
for (i=0; i<TILE_K; i++) {
    uint32_t u = ((uint32_t)w0[k+i]) << 16; memcpy(&b0[i], &u, 4);
}
// Phase B — pure fp32 (vectorizes to vfmadd231ps).
#pragma omp simd reduction(+:a0)
for (i=0; i<TILE_K; i++) a0 += xf[k+i] * b0[i];
```

Same trick for all 4 W rows (one stack scratch each). TILE_K=128
empirically wins on AVX2. Verify with:
```
gcc -S -O3 -march=native kernels.c -o /tmp/k.s
grep -cE 'vfmadd231ps|vfmadd132ss' /tmp/k.s   # want lots of 231ps, 0 132ss
```

**Speedup**: Prefill 3.1x. Decode unchanged (already BW-bound).

**Pitfalls**:
- Without `__restrict__` on the pointer args gcc generates an
  alias-checking branch that disables vectorization.
- Stack scratch MUST be aligned (`__attribute__((aligned(64)))`).
- TILE_K must be a compile-time constant for the inner loop bound — gcc
  won't unroll a runtime bound.
- Mixing the two phases (one fused loop) reverts to scalar. Don't.

**Snippet**: `patterns/linear_dot4_ktile.c`.

---

## B-GEMM-tile — proper GEMM for M>1 prefill

**When**: After B-K-tile makes the M=1 path fast. Now M>1 prefill is the
remaining hotspot.

**How**: For M>1 reorder the loops to N (parallel) → K-tile → M so the
W tile is loaded ONCE per (n0..n0+3) and reused across all M X rows.
Pre-convert all of X (M*K floats) once before entering the parallel
region.

```c
#pragma omp parallel
{
    float* acc = aligned_alloc(64, M*4*sizeof(float));  // per-thread
    #pragma omp for schedule(static)
    for (n0 = 0; n0 < N; n0 += 4) {
        linear_dot4xM_tile(xf, M, K, W+(n0)*K, ..., K, acc);
        // bias add, bf16 store M*4 outputs
    }
    free(acc);
}
```

**Speedup**: Prefill 2.1x. Decode unchanged.

**Snippet**: `patterns/linear_dot4xM_gemm_tile.c`.

**Pitfalls**:
- Don't `aligned_alloc` per linear call inside the parallel-for — use a
  per-thread persistent scratch (the parallel-region scope is correct;
  allocates 16x not 16xN_tiles).
- The bias add and bf16 store inside the n0 loop pay off in cache locality
  vs a separate pass.

---

## B5-cpu — fuse residual+rmsnorm (rarely needed)

**When**: KPROF shows rmsnorm+residual_add combined > 5% of total. This
is rare for dense bf16 — usually they're <3% combined.

**How**: One fused kernel that reads X+H once, writes the RMSnorm'd
result. Saves one read pass over (M, D) bf16 elements.

**Speedup**: Few %. Skip unless you've already done everything in
Phase 4 and prefill needs more.

---

## D1-cpu — SDPA parallelize over (lq, qh)

**When**: SDPA shows up in KPROF as a non-trivial line item (>2%). For
short context (Gemma 4 E4B 512-window) it usually doesn't, because the
quadratic attention term is small.

**How**: Parallelize over the outer loop:
```c
#pragma omp parallel for collapse(2) schedule(static)
for (uint32_t lq = 0; lq < Mq; lq++)
for (uint32_t qh = 0; qh < n_qheads; qh++) { /* compute one attn row */ }
```

Inside one row: scalar softmax then a small GEMV against V — neither
benefits much from per-row vectorization beyond what `-O3` already gets.

---

## F1-cpu — parallel argmax

**When**: Argmax over V = 262 144 (Gemma) takes ~3 ms single-threaded
and is called once per generated token. Often the second-biggest line
item in KPROF after `linear`.

**How**: Block-wise parallel reduction with thread-local (max_val, idx),
then a small serial merge of `n_threads` local maxima.

**Speedup**: ~10x on the argmax kernel itself. Saves ~1 ms/token at 16
threads.

---

## F4-cpu — parallel rmsnorm

**When**: rmsnorm shows up as >5% of decode in KPROF. Decode case (M=1)
parallelizes over D not M.

**How**: Two-pass over (M=1, D): pass 1 reduces sum-of-squares (use
`#pragma omp parallel for reduction(+:s) schedule(static)`); pass 2
scales and writes back.

For prefill (M>1) parallel over rows instead — each thread does a full
rmsnorm of one M row, no reduction barrier.

---

## H4-cpu — embed_gather bulk-memcpy

**When**: Always — cheap to do. The gather is fully bandwidth-bound and
the per-row memcpy is the right primitive.

**How**: For each output row, look up the embedding index and `memcpy`
the bf16 row. `#pragma omp parallel for` over M.

---

## A9-cpu — weight loader (HUGEPAGE + parallel prefault)

**When**: Always — startup latency / cold-cache test is faster with this
on. Warm-cache decode is unaffected, but TLB pressure during decode is
materially lower with huge pages.

**How**: After `mmap(..., MAP_PRIVATE, ...)`:
1. `madvise(MADV_WILLNEED)` — kernel will start reading ahead.
2. `madvise(MADV_HUGEPAGE)` — back with 2 MB pages if THP enabled.
3. `#pragma omp parallel for` touching one byte per 4 KB page.

Do NOT use `MAP_POPULATE` — it's single-threaded inside the kernel and
~2x slower than the parallel touch.

**Speedup**: Cold startup halves (Gemma 4 E4B 9 GB: 0.95s → 0.44s).

**Snippet**: `patterns/parallel_prefault.c`.

**Pitfalls**:
- THP must be enabled in the kernel (`cat /sys/kernel/mm/transparent_hugepage/enabled`).
  If `[never]`, the MADV_HUGEPAGE hint is a no-op.
- The prefault loop uses a `reduction(+:sink)` on a `volatile` to keep the
  read alive — without it the compiler may DCE the loop body.

---

## J1-cpu — thread-count tuning + OMP env

**When**: After B1-cpu. Critical — wrong thread count or wrong OMP env
can cost 30% on decode.

**How**: Set in your bench runner (NOT in the binary):
```sh
export OMP_NUM_THREADS=<PHYSICAL CORES, not logical>
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
```

Sweep `OMP_NUM_THREADS ∈ {phys/2, phys, phys*2}` once and confirm
`phys` is the sweet spot (almost always — SMT siblings collide on L1/L2
for memory-bound decode).

`OMP_WAIT_POLICY=active` makes worker threads busy-wait on a futex
between regions instead of `sleep()`ing (libgomp default is `passive`).
For thousands of small parallel-fors per decode token, the wakeup cost
of `passive` is ~30%.

**Snippet**: `patterns/bench.sh`.

---

## J2-cpu — KPROF

**When**: Before any kernel-specific optimization.

**How**: See `patterns/kprof.{h,c}` and the wiring patterns. Build /
run with `KPROF=1 ./bench.sh`. Inspect the printed table at exit.

The ONE diagnostic that prevents wasted effort.

**Snippet**: `patterns/kprof.h`, `patterns/kprof.c`.
