# optimize-cpu — profiling

You can only optimize what you can measure. This document covers:

1. KPROF — per-kernel timing
2. asm inspection — confirm auto-vectorization happened
3. DRAM-BW ceiling measurement — the stop condition for decode
4. Thread-count sweep
5. Variance discipline — best-of-N, warm vs cold cache

## 1. KPROF — per-kernel timing

**Always integrate KPROF as the second thing you do** (right after
`-O3 -fopenmp`). Without it you have no idea which kernel dominates.

### Wiring

Drop in `patterns/kprof.{h,c}`. Add to `main.c`:
```c
if (getenv("KPROF")) kp_enable(1);
// ... run forward passes ...
kp_report();
return 0;
```

At the top of each public kernel in `kernels.c`:
```c
void linear_bf16(...) {
    KPROF_ENTER();
    /* body */
    KPROF_LEAVE(KP_LINEAR);
}
```

The macros are `#define KPROF_ENTER() double _kp_t0 = kp_is_on() ? kp_now() : 0.0`
and `#define KPROF_LEAVE(K) do { if (kp_is_on()) kp_add((K), kp_now()-_kp_t0); } while (0)`.
Cost when off: 1 load + 1 branch per kernel entry (~ns). Cost when on:
~30 ns per kernel entry (one `clock_gettime`).

### Reading the report

```
KPROF=1 OMP_NUM_THREADS=16 ./bench.sh
...
[KPROF] per-kernel wall (sum over the whole run):
  kernel                 wall_s      calls    us/call
  linear                 8.3457      22360      373.2   (97.4%)
  rmsnorm                0.0826       2106       39.2   ( 1.0%)
  embed_gather           0.0524        128      409.4   ( 0.6%)
  ...
  TOTAL                  8.5697
```

**Interpretation**:
- `linear` at 97% → focus exclusively here. Optimizing rmsnorm /
  embed_gather / rope is a waste of effort until linear drops below ~70%.
- 22 360 linear calls over 8.6 s = 373 µs/call average. With 16 threads,
  per-call serial work is ~6 ms — so fork-join overhead (~10 µs) is
  noise.
- TOTAL ≈ wall time of forward passes (excluding startup, sampling,
  detokenize). Compare against your bench's prefill+decode time.

### When to re-profile

- After every "big" optimization (B-K-tile, B-GEMM-tile, A9). The hot
  kernel shouldn't change but the percentages will, telling you the
  next thing to chase.
- Whenever a change DOESN'T speed things up. KPROF often reveals that
  the change worked but only on a non-bottleneck kernel (i.e., wrong
  thing to optimize).

## 2. asm inspection — confirm auto-vectorization

The MOST important debug for the B-K-tile technique. The compiler
silently falling back to scalar code is what made the naive
`linear_bf16` slow.

### Workflow

```sh
gcc -S -O3 -march=native -ffast-math -fopenmp -funroll-loops \
    -Isrc-cpu src-cpu/kernels.c -o /tmp/kernels.s

# Count vectorized FMA (want LOTS):
grep -c vfmadd231ps /tmp/kernels.s

# Count scalar FMA (want ZERO in the inner loop):
grep -c vfmadd132ss /tmp/kernels.s

# Inspect linear_dot4 specifically:
awk '/^linear_dot4_ktile.*:/{p=1} p; /^[._]?L[0-9]+:.*ret/{p=0}' /tmp/kernels.s
```

On a healthy AVX2 build:
- Inner FMA loop emits `vfmadd231ps %ymm, %ymm, %ymm` (8-wide single).
- bf16→fp32 conversion emits `vpmovzxwd %xmm, %ymm` (zero-extend 16→32)
  followed by `vpslld $16, %ymm, %ymm` (shift left 16).
- Loads/stores are `vmovaps` (aligned) or `vmovups` (unaligned).

If you see scalar `vfmadd132ss` in the inner FMA loop, your code isn't
vectorizing. Likely causes:
- Missing `__restrict__` → gcc inserts alias checks → bails on vectorize.
- Mixed type in the loop (e.g., bf16 load fused with fp32 FMA) → loop
  has multiple types of stmts and SLP fails.
- Reduction order not advertised → add `#pragma omp simd reduction(+:a0,a1,...)`.
- `-ffast-math` missing → strict fp prevents reordering.

### AVX-512 vs AVX2

If your CPU has AVX-512 (`grep -c avx512f /proc/cpuinfo`), gcc with
`-march=native` will emit 16-wide `vfmadd231ps %zmm` instead of 8-wide
`%ymm`. Faster, but only on Sapphire Rapids / EPYC Genoa / Sierra
Forest. AMD EPYC 7763 (Zen 3) is AVX2 only.

### NEON / aarch64

On Apple Silicon / Graviton / Snapdragon X you'll see `fmla v0.4s, v1.4s,
v2.4s` (4-wide fp32 FMA). Same auto-vectorization conditions apply —
split the bf16→fp32 conversion from the FMA reduction.

## 3. DRAM-BW ceiling measurement — the stop condition for decode

### Why

Decode reads ~9 GB of bf16 weights per token (Gemma 4 E4B). At 50 GB/s
DRAM bandwidth that's a 5.5 tok/s upper bound regardless of how fast
your CPU is — nothing short of weight quantization can break it.

### The microbenchmark

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(void) {
    size_t N = (size_t)1 << 30;          // 1 GB
    float* x = (float*)aligned_alloc(64, N);
    memset(x, 1, N);                     // fault in
    // Warm.
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < N/sizeof(float); i++) s += x[i];

    double t0 = now();
    int iters = 4;
    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for reduction(+:s) schedule(static)
        for (size_t i = 0; i < N/sizeof(float); i++) s += x[i];
    }
    double t1 = now();
    double bw = (double)N * iters / (t1 - t0) / 1e9;
    printf("BW: %.2f GB/s (sink=%g)\n", bw, s);
}
```

Build: `gcc -O3 -march=native -fopenmp bw.c -o bw -lm`
Run: `OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bw`

Typical results:
- AMD EPYC 7763 16C (single-NUMA, single VM): ~50–65 GB/s
- AMD EPYC 9554P 64C dual-NUMA: ~250 GB/s (with `numactl --interleave=all`)
- Apple M3 Pro 12C (LPDDR5): ~150 GB/s
- Desktop Intel i9-13900K DDR5: ~75–90 GB/s

### Compute the ceiling

```
weight_bytes_per_token = sum_over_layers(sizeof(W_attn) + sizeof(W_mlp))
                       + sizeof(lm_head) + sizeof(embed)

For Gemma 4 E4B bf16:
  ~210 MB/layer * 42 = ~8.8 GB attention+MLP
  + 1.34 GB tied lm_head
  ≈ 9 GB weight reads per decode token (tied embed, so we count it once).

ceiling_tps = effective_bw / weight_bytes_per_token
            = 60 GB/s / 9 GB/tok = 6.7 tok/s
```

If your decode hits 6.4 tok/s of the 6.7 ceiling, **you are done**.
Anything above the simple ceiling (we hit 7.7 vs 6.7) means you're
also getting L3 reuse across consecutive layers — bonus, not a bug.

### Sanity on weight bytes

To get the exact weight count for your model:
```
python -c "import safetensors, json, glob, os
total = 0
for f in glob.glob('/path/to/model/*.safetensors'):
    with safetensors.safe_open(f, framework='pt') as h:
        for k in h.keys():
            t = h.get_tensor(k)
            total += t.nelement() * t.element_size()
print(total / 1e9, 'GB')"
```

Then subtract bookkeeping tensors that don't show up in the decode hot
path (e.g., `lm_head` if not tied, RoPE base tensors, layernorm scales —
these are tiny). For Gemma 4 E4B the answer is ~9 GB.

## 4. Thread-count sweep

Once at the end (after B-K-tile etc. are in), sweep:
```sh
for T in 4 8 12 16 20 24 32; do
    OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores \
    OMP_WAIT_POLICY=active ./bench.sh 2>&1 | grep tok/s
done
```

Expected shape (memory-bound decode on a CPU with SMT):
- T = phys/2: undersaturated. Adding more cores helps a lot.
- T = phys: PEAK decode.
- T > phys (SMT): decode REGRESSES because SMT siblings fight for L1/L2.
- Prefill is roughly flat from T=phys onward because it's compute-bound
  and SMT's 2nd thread is mostly waiting on L2 misses.

**Always pin to physical core count.** Doesn't matter how many logical
CPUs the OS sees.

## 5. Variance discipline

CPU benchmarks are noisier than GPU benchmarks because of:
- Page-cache state (cold vs warm — easily 10x decode difference on the
  first iteration).
- NUMA migration (mitigated by `OMP_PROC_BIND=close`).
- DVFS / boost clock variation (mitigated by running back-to-back).
- Other tenants on the VM.

Best practice:
- **Always best-of-3.** Run the bench 3+ times after a warm-up run,
  take the BEST decode tok/s. The minimum measures noise floor.
- **Always warm-cache.** First run prefaults weights into page cache;
  reported numbers should be from later runs.
- **Always same thread count, same OMP env.** Pin them in `bench.sh`.
- **Time prefill and decode separately.** Don't average them.
- **Re-run after >24 h.** AWS / Azure / GCP can shift you to a CPU on a
  different memory channel between sessions, ±10% BW.

A speedup below 2% is usually noise unless you have ≥5 samples and a
clean variance distribution. A speedup below 1% is always noise.
