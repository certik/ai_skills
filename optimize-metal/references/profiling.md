# Profiling — how to know what to optimize next

Optimization decisions are driven by the profile, not the catalog. This
file collects the diagnostics and measurement protocols that make
"profile, identify bottleneck, attack" actually work.

## The two-number rule: print `wall` AND `gpu_busy` always

The single most useful diagnostic in this whole skill is the gap
between wall-clock time and the GPU's busy time. The Metal shim already
sums `(GPUEndTime - GPUStartTime)` into a global `g_gpu_time`; reset it
before timing, then print both:

```c
g_gpu_time = 0.0;
clock_gettime(...t0);
// ... run prefill or decode loop ...
clock_gettime(...t1);
double wall = (t1 - t0);
fprintf(stderr, "decode: %.2fs (%d toks; %.1f tok/s; gpu_busy=%.2fs)\n",
        wall, n_gen, n_gen / wall, g_gpu_time);
```

The `wall - gpu_busy` gap tells you which family of optimization to chase:

| Condition                    | Meaning                                 | Look at         |
|------------------------------|-----------------------------------------|-----------------|
| `gpu_busy ≈ wall`            | GPU-bound — kernels are the bottleneck  | B, C, D, E, F, I |
| `gpu_busy ≪ wall`            | CPU-bound — host scheduling is the cost | A, H (esp. H1) |

The largest single jump in a typical naive port comes from eliminating
host-side breaks (H1 / glue kernels). You can see this happen
immediately as `gpu_busy / wall` ratio jumping from ~30% to ~95%.

## Diffusion samplers: add `host_post` as a third number

Some samplers — notably block-refinement diffusion LLMs (Dream, LLaDA,
fastdllm) — do meaningful per-step work on the CPU AFTER the forward
pass. For a refine step it might be: read logits, compute softmax over
V at each of BL positions, find argmax + confidence per position, pick
which positions to commit, build the next step's input ids.

That work happens between `commit_wait()` and the next dispatch, so
the GPU is idle but the CPU is not. It shows up as a gap in `wall -
gpu_busy` that LOOKS like a host-scheduling problem (A/H family) but
isn't — it's a sampler problem (F3).

Add a third accumulator:

```c
double host_post_wall = 0.0;
// ... inside the sampler step, after commit_wait but before the next dispatch:
clock_gettime(...t_pp0);
// CPU softmax + argmax + confidence loop here
clock_gettime(...t_pp1);
host_post_wall += (t_pp1 - t_pp0);
```

And print the three-number breakdown:

```
refine: 1.31s wall (64 steps; gpu_busy=1.03s; host_post=0.25s)
```

The mental model becomes:

```
wall ≈ gpu_busy + host_post + encode_gap
```

If `host_post` is more than ~5% of wall, attack F3 (move the per-row
softmax/argmax to the GPU as a single kernel). For a Dream-7B port,
F3 dropped `host_post` 0.25 → 0.00 s and `wall` 2.90 → 2.63 s.

## KPROF — per-kernel GPU time (optional)

`KPROF=1` wraps each dispatch in its own cmdbuf with commit+wait so you
can attribute GPU time to individual kernels. This is optional. For a
model whose architecture you already know (you wrote the C ref!), you
can predict the top hot kernels from first principles:

- Largest GEMV (lm_head, expert weights, expert gather).
- SDPA in long-context regimes.
- The SSM / recurrent state step if the model has one.

Spend the KPROF effort only if your profile genuinely surprises you.

### KPROF caveats

- **KPROF inflates total runtime ~30%.** Each commit/wait costs
  0.3–1 ms and at ~340 dispatches/token that adds up. Do NOT use KPROF
  tok/s as a real performance number. Use it for:
  - Correct RELATIVE ordering of which kernels are hot.
  - Correct ABSOLUTE per-kernel GPU time.

- **`KPROF total > wall time` means the concurrent encoder is actually
  working.** KPROF serialises every dispatch; real runs with the
  concurrent encoder overlap independent work. For a Qwen3.6 decode at
  n=210 we measured KPROF total 22.2 ms/tok vs real wall 14.4 ms/tok —
  a 35% overlap factor.

- **`KPROF total ≈ wall` means the concurrent encoder is NOT
  overlapping.** Apply A5 / A8 barrier surgery, not kernel-internal
  optimization.

See `references/gotchas.md` #20 and #23 for the full story.

### KPROF as a TG-starvation detector

After getting the per-kernel breakdown, compute **effective bandwidth**
for each large matmul:

```
eff_bw[k] = (W_bytes[k] + X_bytes[k] + Y_bytes[k]) / per_call_time[k]
```

For weight-bound shapes (typical at decode/refine), `W_bytes` dominates.
Compare each matmul's `eff_bw` against your device's peak (M4 Max:
~410 GB/s; M2 Max: ~400 GB/s; M1 Max: ~400 GB/s; M3 Max: ~300 GB/s).

- **eff_bw > 60% of peak**: matmul is bandwidth-saturated; no easy
  kernel-internal win. Move on.
- **eff_bw < 40% of peak**: matmul is TG-starved or has a register/
  occupancy issue. Check `TG_count = N/BN × ceil(M/BM)`:
  - If TG_count < ~80 (M4 Max SG-slot capacity / WM·WN SGs per TG),
    you don't have enough TGs in flight. Apply **catalog C4 (aspect-
    ratio dispatch)**: route to a smaller BM to double M-bands and
    therefore TGs. See gotcha #39.
  - If TG_count ≥ 80 but eff_bw is still low, the issue is per-TG —
    register spill, bad load pattern, or compute-bound (rare).

Concrete diagnostic from Dream-7B BL=32 refine (KPROF output):

```
gate_proj  K=3584  N=18944  per_call=430us  W=135MB → 314 GB/s (77% peak)  ← fine
down_proj  K=18944 N=3584   per_call=1082us W=135MB → 125 GB/s (30% peak)  ← TG-starved
```

Same matrix size, 2.5× different bandwidth. The slow one had only
N/BN = 56 TGs vs the fast one's 296. A one-line dispatch change (C4)
closed the gap.

## Measuring tok/s — match MLX exactly

Use `mlx_lm`-equivalent timing so you can compare apples to apples
with MLX:

```
tps = (n_gen - 1) / decode_time      // exclude the FIRST decoded token
```

The first decoded token is the prefill argmax; including it inflates
your tok/s number AND distorts decode-only measurements.

### Watch out: MLX's `tok/s` denominator is `max_new_tokens`, not `actual_emitted`

For diffusion-LLM benches in particular, MLX's example scripts compute
`tps = args.max_new_tokens / elapsed` — i.e., the DENOMINATOR is the
slot budget, not the number of tokens actually emitted before EOS. If
your port instead reports `tps = actual_gen / elapsed` (the natural
metric for an autoregressive decoder), you'll see a HUGE apparent
ratio when EOS hits early:

```
MLX  bench:   1.58s, ~20.3 tok/s on 32 slots  ← 32 / 1.58
Yours bench:  1.72s, 9.3 tok/s wall            ← 16 actual / 1.72
```

That's 2.2× on tok/s but only 9% on wall — a metric mismatch, not
a real perf gap. Before chasing what looks like a 2× regression,
recompute both with the same denominator:

```
MLX  (slot):   32 / 1.58s = 20.3 tok/s
Yours (slot):  32 / 1.72s = 18.6 tok/s   ← actual 8% gap, not 2×
```

For real comparisons, always **report wall-time directly** in addition
to whatever tok/s convention your code uses. Wall is unambiguous.

### Run length and best-of-N

Short runs on a busy machine routinely show 3–5× tok/s variance from
other apps, thermal state, and Metal queue contention. To get a stable
number:

- **At least 64 decoded tokens.** Below that, variance dominates.
- **Best-of-3 runs** for headline numbers. For commit messages, 5-run
  median if you need to distinguish a 1–2% win from noise.

Anything below 2% is in the noise band. Conversely, a 5-run-confirmed
+3% is definitely real.

## Reading the first-cmdbuf hiccup

Even after JIT-compiling all kernels at startup, the FIRST cmdbuf in a
process pays a large fixed cost: Metal lazily materialises pipeline
state objects, makes weight buffers resident, etc. Concretely
(Qwen3.6-35B-A3B on M4 Max):

```
prefill: 3.09s wall (16 tokens; 5.2 tok/s; gpu_busy=0.09s)   ← 1st cmdbuf
decode:  3.12s wall (128 tokens; 41.0 tok/s; gpu_busy=3.11s) ← steady state
```

The GPU did only 90 ms of prefill compute but the cmdbuf took 3.09s
wall. Quantitatively: ~1s of residency wiring per ~30 GB resident, paid
exactly once.

For benchmarking your decode loop, IGNORE the first cmdbuf. Either:

- Run a warmup `forward(Lq=1)` BEFORE timing the real workload.
- Report prefill and decode timings separately so the first-cmdbuf hit
  lands cleanly in prefill.
- Always print both wall and gpu_busy so you can tell the difference.

See `references/gotchas.md` #5 and #30 for the full breakdown.

## Common surprises to check

- **`gpu_busy ≪ wall` after batched dispatches** → host breaks (H1). Look
  for `commit_wait` inside `forward()`.
- **Tok/s drops every few tokens** → memory pressure / paging. Check
  `wired_limit` (A7) and verify weights are zero-copy mmap'd
  (`gpu_buf_wrap_nocopy`).
- **Performance is bursty** → CPU encode becoming the bottleneck. Apply
  2-deep cmdbuf pipeline (A4) + persistent / const param buffers
  (A2 + A3).
- **A small elemwise kernel (residual_add, sigmoid, qkv_split) reports
  ~25 µs/call in KPROF when it should be <2 µs** → TG size is (1,1,1).
  Bump to 32. See `references/gotchas.md` #17.
