# optimize-cpu — pitfalls (war stories)

Curated short list of war stories most likely to bite during a CPU
optimization session. Read this once before you start. Each entry has a
one-line symptom, the cause, and the fix.

---

## 1. gcc can't autovectorize fused bf16→fp32 + FMA

**Symptom**: `linear_bf16` at >95% of KPROF, prefill barely above
9 tok/s on a 16-core EPYC, prefilling 21 tokens takes 2+ seconds.

**Diagnosis**:
```
gcc -S -O3 -march=native -ffast-math -fopenmp -funroll-loops \
    -Isrc-cpu src-cpu/kernels.c -o /tmp/k.s
grep -c vfmadd132ss /tmp/k.s   # >0 → scalar FMA in the inner loop
grep -c vfmadd231ps /tmp/k.s   # 0 or very few → not vectorizing
```

**Cause**: The fused inner loop
```c
for (k=0; k<K; k++) {
    float x = bf16_to_f32(xrow[k]);
    s0 += x * bf16_to_f32(w0[k]);  // <-- this whole expression
}
```
mixes 16-bit loads, a 16→32 shift+bitcast, and a 32-bit FMA. gcc's
vectorizer cost model bails and falls back to scalar `vfmadd132ss`.

**Fix**: Split into two phases per TILE_K block — Phase A converts
TILE_K bf16 to TILE_K fp32 into a stack scratch buffer, Phase B runs
pure-fp32 FMA from the scratch. See `patterns/linear_dot4_ktile.c`.

This is by far the biggest single optimization in the whole pipeline.
Prefill goes 8.9 → 27.6 tok/s in one commit.

---

## 2. OMP_NUM_THREADS defaults to logical CPUs, not physical cores

**Symptom**: Decode tok/s is fine at 16 threads but REGRESSES at 32.

**Cause**: Hyperthreading / SMT. The two SMT siblings of a physical core
share L1 + L2. Decode is memory-bandwidth bound and L1/L2-coherent — two
siblings sharing the same cache line context-switch on cache events and
pay a serialization cost. Net result: decode regresses 7.7 → 6.2 tok/s
when going from 16 (physical) to 32 (logical) threads on this EPYC.

**Fix**: Always set `OMP_NUM_THREADS` to the physical core count, never
trust the libgomp default. Encode in `bench.sh` and document in
`PERF.md`.

**Memory BW caveat**: The parallel-sum microbenchmark may show BW
INCREASING with SMT threads (50 → 59 GB/s on this VM). This is real —
SMT helps a pure memory-streaming workload because more outstanding
loads. But decode is not pure-streaming; it's intermixed with FMA, and
the compute penalty wins.

---

## 3. OMP_WAIT_POLICY defaults to "passive" → ~30% decode loss

**Symptom**: KPROF says `linear` is 97% but decode tok/s is way below
the DRAM-BW ceiling.

**Cause**: libgomp's default `OMP_WAIT_POLICY=passive` means worker
threads `sleep()` (futex wait) between parallel regions. For decode
with thousands of small parallel-fors per token (~6 parallel-fors per
layer × 42 layers ~= 250 per token), the wakeup latency from sleep is
~30% of total decode time.

**Fix**: `export OMP_WAIT_POLICY=active` in `bench.sh`. Workers busy-
wait between regions; safe because we have plenty of cores and no
contention with other work.

**Don't set in the binary** with `omp_set_wait_policy` — it's a hint
and gcc's libgomp ignores it. Use the env var.

---

## 4. -march=native is NOT implied by -O3

**Symptom**: `gcc -O3 -fopenmp` gives ~5x speedup over `-O2`, you expected
~30x for a vectorizable kernel.

**Cause**: Default ISA is x86_64 baseline = SSE2 = 4-wide fp32 vectors.
AVX2 (256-bit = 8-wide) requires `-mavx2` or `-march=native`.

**Fix**: Always include `-march=native -mtune=native`. Speedup from 4
to 8 lanes is roughly 2x by itself for vectorizable loops, plus FMA is
unlocked.

---

## 5. -ffast-math reorders fp reductions → ~50-token argmax drift

**Symptom**: After applying B-K-tile (vectorized linear), the model
still produces a coherent answer on the validation prompt, but the
exact tokens diverge from a saved reference after ~50 tokens.

**Cause**: `-ffast-math` (specifically `-funsafe-math-optimizations`
inside it) lets the vectorizer reorder fp32 additions inside the
reduction. With bf16 weights (8 mantissa bits) borderline pre-softmax
values can swap rank, and greedy argmax then picks a different token.
The model is still computing the same answer mathematically — there's no
bug — but bit-for-bit reproducibility is gone.

**Fix**: Save a NEW reference (the output of the first
ffast-math+vectorized build) and validate against it from now on. Get
user sign-off on the new reference before committing.

**When to worry**: Token mismatch INSIDE the first ~16 tokens. That's
almost always a real bug (off-by-one in a tile boundary, wrong reduction
order, wrong tail handling). Don't accept first-16-token drift.

---

## 6. MAP_POPULATE is single-threaded inside the kernel

**Symptom**: Adding `MAP_POPULATE` flag to `mmap()` makes cold-startup
faster than no flag at all (good), but slower than expected on a
multi-core machine.

**Cause**: `MAP_POPULATE` causes the kernel to prefault pages
sequentially inside the `mmap()` syscall, all on the calling thread. For
a 9 GB weight file this is a single-threaded DRAM read.

**Fix**: Use plain `MAP_PRIVATE` and add a parallel prefault in userspace:
```c
#pragma omp parallel for schedule(static) reduction(+:sink)
for (size_t off = 0; off < bytes; off += 4096) sink += p[off];
```

~2x faster on 16 threads (Gemma 4 E4B 9 GB: 0.95 s → 0.44 s).

See `patterns/parallel_prefault.c`.

---

## 7. madvise(MADV_HUGEPAGE) silently no-op when THP is disabled

**Symptom**: Added `madvise(MADV_HUGEPAGE)` after `mmap`, expected
prefill speedup, saw none.

**Cause**: System-wide Transparent Huge Pages must be enabled for the
hint to do anything. On Linux:
```
cat /sys/kernel/mm/transparent_hugepage/enabled
```
Should show one of `[always]` (recommended), `[madvise]` (your hint is
honored), or `[never]` (your hint is ignored).

**Fix**: If `[never]`, ask the user (or sysadmin) to switch to `[madvise]`
or `[always]`:
```
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

Cloud VMs sometimes set this to `[never]`. Test environments often
inherit the host setting.

---

## 8. CC ?= gcc silently uses /usr/bin/cc

**Symptom**: Your Makefile says `CC ?= gcc` but `make -n` shows `cc -O3
-march=native ...` — and `cc --version` is a different compiler with
different defaults than `gcc --version`.

**Cause**: `?=` in Make does not override Make's BUILT-IN default of
`CC = cc`. The built-in always wins unless you use `:=` or override
explicitly.

**Fix**: `CC := gcc` (immediate assignment). Or invoke as `make
CC=gcc`. Either works.

This silently cost a project ~5% on one machine where `cc` was an old
clang without the same vectorization heuristics. Worth a `make -n | head`
sanity check whenever you start.

---

## 9. K_OUT=8 doesn't help on bandwidth-bound decode

**Symptom**: After B-loop-tile with K_OUT=4, you try doubling to K_OUT=8
expecting more throughput. Decode is flat, prefill regresses slightly.

**Cause**: Decode is DRAM-bandwidth-bound, not compute-bound. K_OUT=8
doesn't reduce the W traffic (still K*N total bytes); it just changes
the ratio of W reads per X re-use. Bigger register tile = larger stack
scratch = harder L1 fit on the prefill GEMM tile = slight regression.

**Fix**: Stay at K_OUT=4. Don't even try K_OUT=8 unless your CPU has
16+ vector registers AND you're compute-bound (rare for bf16 decode).

---

## 10. TILE_K too small or too big

**Symptom**: After getting B-K-tile working at TILE_K=128, you sweep
{64, 256, 512} expecting more speedup. All regress.

**Cause**:
- TILE_K=64 → loop overhead per tile dominates (more tile-boundary
  fixups, less work per parallel iteration).
- TILE_K=256 → 4 × 256 × 4 B = 4 KB stack scratch per tile + 4 KB X
  per tile = pressures L1 (32 KB) once you add code + other working
  set.
- TILE_K=512 → spills L1 entirely on the read pass for B, falls back to
  L2.

**Fix**: TILE_K=128 is the sweet spot on AVX2 with 32 KB L1. On Apple
M-series (128 KB L1) try 256. On AVX-512 with 32 KB L1 try 256 (since
the per-iter throughput doubles).

---

## 11. Per-call aligned_alloc adds up to 0.1% — don't bother

**Symptom**: KPROF says `linear` is 97%; you notice `linear_bf16`
mallocs and frees a per-thread scratch per call.

**Investigation**: 22 360 calls × ~5 µs malloc/free overhead = 110 ms,
or 0.1% of an 8 s decode. Below noise.

**Fix**: Don't refactor. Spend the time on something that matters. ONLY
worth it if KPROF shows malloc/free dominating (it won't, on a sane
glibc with arena caching).

---

## 12. -Ofast offers nothing over -ffast-math here

**Symptom**: You try `-Ofast` to see if there's a free win past
`-O3 -ffast-math`. No change.

**Cause**: `-Ofast` = `-O3 -ffast-math -fno-protect-parens` plus a
couple of small things. None of them apply to bf16 inference kernels.

**Fix**: Stick with `-O3 -ffast-math`. Predictable; same correctness
profile.

---

## 13. clang skipped without libomp installed

**Symptom**: `clang -fopenmp ...` fails with `unable to find OpenMP
runtime`.

**Cause**: clang doesn't ship libomp by default on most distros.

**Fix**: Either `apt install libomp-dev` (Debian/Ubuntu) or just use
gcc, which ships libgomp built-in. The performance is essentially
identical for these workloads; both compilers vectorize the K-tile
pattern.

---

## 14. cold-cache decode 10x slower than warm — confusing the bench

**Symptom**: First run of the bench prints 0.8 tok/s decode. Second run
prints 7.7 tok/s. Did the optimization regress?

**Cause**: First run was cold — every page of weights had to be read
from disk into the page cache. Once warm, the OS keeps the file in RAM
and subsequent forward passes hit DRAM, not disk.

**Fix**:
- Always discard the first run.
- Report best-of-3 from runs 2..N.
- For "cold startup" measurements, drop the page cache between runs:
  `echo 3 | sudo tee /proc/sys/vm/drop_caches`. Otherwise you're just
  measuring page-cache warmth.

---

## 15. NUMA-cross page allocation halves decode

**Symptom (multi-NUMA host)**: Same binary, same env, decode is 60% on
some runs.

**Cause**: Linux first-touch allocates the weight pages on whichever
NUMA node first touched them during prefault. If your prefault threads
were on node 0 but later the kernels run on node 1, every DRAM read is
a cross-socket trip.

**Fix**: Pin the whole process to one NUMA node:
```
numactl --cpunodebind=0 --membind=0 ./bench.sh
```
Or, on a multi-NUMA setup, run one process per NUMA node and shard work
between them (outside this skill's scope).

For single-NUMA VMs (most cloud setups) this is a non-issue.

---

## Quick reference card

When in doubt, this is the order:

1. `CC := gcc` (not `?=`).
2. `CFLAGS += -O3 -march=native -funroll-loops -ffast-math -fopenmp -flto`
3. `LDFLAGS += -fopenmp -flto -lm`
4. `OMP_NUM_THREADS = <physical cores>` (NOT logical).
5. `OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active`
6. KPROF the build. Confirm `linear` dominates.
7. `gcc -S` the build. Confirm `vfmadd231ps` in the linear inner loop.
8. If not vectorized, split bf16→fp32 from FMA into two TILE_K phases.
9. For prefill (M>1), reorder loops to N → K-tile → M.
10. `mmap` + `MADV_HUGEPAGE` + parallel prefault, no MAP_POPULATE.
11. Measure DRAM BW with a parallel-sum microbench. Stop when decode is
    within 5% of `effective_bw / weight_bytes_per_token`.
