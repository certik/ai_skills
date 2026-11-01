# Profiling — identify the next cause without distorting acceptance

For the full timer/cohort/first-submit protocol, read
[measurement-and-diagnosis.md](measurement-and-diagnosis.md).
Start from the observed reference shapes/graph, not a catalog priority list.

## Wall and GPU coverage

Print wall, valid GPU interval union and counts for prefill/decode separately.
For a synchronous nonoverlapping single-buffer path, summing completed
`GPUEndTime-GPUStartTime` gives coverage. For pipelined/concurrent buffers,
merge intervals and clip to the relevant CPU phase; owner totals can overlap
or straddle TTFT. Reject invalid clocks; unavailable native GPU time is null.

| Signal | Interpretation / next test |
|---|---|
| GPU coverage near wall | Profile actual hot kernels/shapes and critical path. |
| Low GPU coverage | Unattributed setup/host/queue/driver gap; measure encode, submit, GPU start, waits and residency separately. |
| Large CPU sampler `host_post` | Consider F3/GPU sampling/glue while preserving native normalization/rounding. |
| Large weight-load component | Compare A9/pinned loader with the current method under declared file-cache scope. |
| Long-context slowdown | Re-profile attention, KV/state and selector boundaries at the actual context; short profiles cannot choose long geometry. |

Do not infer CPU encoding solely from wall–GPU difference. A post-submit
stall can remain after an 8ms encoder. Do not infer overlap solely because
a serialized KPROF run's total exceeds an uninstrumented wall interval.

## Startup breakdown

Record config/tokenizer, libraries/PSOs, weight reads, workspace/cache
allocation, residency entry, generation start and restore/teardown.
Async stages overlap; their medians do not add into outer child wall.
Prewiring/warmup is work, even when moved outside conventional prefill.

Historical Dream-7B/M4 Max A9 observation:

```
naive: weights=1.709s startup=1.735s
pread: weights=0.817s startup=0.856s
```

Useful evidence for trying parallel pread when loading is hot, not proof
that tokenizer/JIT or residency can be ignored on the next model.
Do not assume mmap is always the reference: pinned MLX can pread directly
into Shared Metal storage.

## KPROF attribution (untimed acceptance-wise)

Per-dispatch command-buffer commit/wait changes concurrency, batching and
runtime, sometimes substantially. Use one bounded representative profile
to locate hot families, then remove instrumentation for speed claims.
Relative rankings and absolute dispatch durations are conditional on that
instrumented schedule, not exact costs of the uninstrumented engine.

**KPROF must be absent, not `0`, in throughput children after activation.**
The same exclusion applies to observer/capture/Python-injection hooks.
Separate profiles, correctness controls and final measurement cohorts.

For each dominant family collect actual M/N/K/context, route, call count,
TG count/threads/shared bytes and source/options identity. Small-N shapes
can be TG-starved even when larger equal-weight shapes are fast. Nominal
`(W+X+Y bytes)/time` helps form an occupancy/cache hypothesis; it is not
memory-controller traffic or a saturation threshold.

A historical Dream down projection had only56 N-axis TGs versus296 for
gate projection. Smaller-BM routing improved utilization. Check shape,
alignment, spills and selector before cargo-culting that remedy.

## Bounded tuning

1. Verify the literal native baseline/options/ABI on real shapes.
2. Choose a few evidence-backed candidate geometries for the top bottleneck.
3. Screen with short repeated operator batches long enough for the timer.
4. Cover tails/thresholds/causal/stride/guards and buffer reuse.
5. Run one short production control; escalate to relevant long controls.
6. Use one predeclared counterbalanced cohort once correctness/source is stable.

Record INCONCLUSIVE when variability obscures a small effect. Retain all
observations; best-of is supplementary, not final acceptance. Matching MLX
decode does not close a prefill/setup gap or prove a bandwidth floor.
