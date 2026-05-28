# Top pitfalls (curated short-list)

These are the war stories most likely to bite during an optimization
session. Each one cost at least one debug iteration in a real port and
points to the full numbered entry in `gotchas.md` for context. Read
this list once before you start optimizing; re-read whenever a
benchmark goes sideways.

The full numbered list (60 entries with full reproductions) lives in
`gotchas.md`. This file is the curated subset most directly relevant
to the empirical attack order in SKILL.md.

## During encoding / dispatch

- **`dispatchThreads:` takes TOTAL THREADS, not threadgroups**. For
  SG-per-output kernels, you want `grid_x = 32 * N`. Passing `N` directly
  silently launches `ceil(N/32)` SGs. (gotcha #2)
- **`dispatchThreads:` with `grid_x % tg_x != 0` makes the last TG
  non-uniform.** Cooperative TG-mem loads using `stride = tg_x` then
  leave most of TG-mem uninitialized. Either keep grid_x a multiple
  of tg_x, or use `threads_per_threadgroup` dynamically inside the
  kernel and handle the small case. (gotcha #52)
- **MSL position attributes must agree in dimensionality.** All
  `[[thread_position_in_*]]` / `[[threadgroup_position_in_grid]]` /
  `[[threads_per_threadgroup]]` parameters must be all-uint or
  all-uint3 — mixing gives "expecting input declarations with either
  all scalar types or all vector types". INDEX attrs (`sg_id`,
  `lane`) are exempt. (gotcha #51)

## Correctness traps that pass per-kernel but fail end-to-end

- **KV cache `max_ctx` and SDPA TG-mem `MAX_CTX` must agree AND be ≥
  prompt_len + max_tokens**. Silent buffer overruns produce wrong
  tokens that look like numerical drift past the first ~32 tokens.
  (gotchas #3, #15)
- **Concurrent encoder + missing barrier** = non-deterministic
  divergence that often passes per-kernel tests (which serialize one
  kernel at a time) but fails end-to-end. (gotchas #18, #26)
- **The first generated token IS the prefill argmax.** When wiring up a
  2-deep pipeline that primes itself, it's easy to skip emitting this
  token, shifting all output by 1. (gotcha #10)
- **bfloat4 alignment.** Load `bfloat4` only at 8-byte-aligned
  addresses, or you'll get garbage on some Apple GPU generations.
  (gotcha #40)
- **Forgetting GQA in the SDPA tile.** The K/V head index is
  `head_kv = head_q / (Nq/Nkv)`, not `head_q`. Easy to break during
  tile refactorings. (gotcha #33)
- **SDPA `N_SG` constexpr and host `threads_per_threadgroup` MUST
  match.** Desync (kernel `N_SG=12`, host dispatches `32 * 16`) →
  SGs 12..15 overflow `sg_m[]` / `sg_o[]` → garbage tokens. Update
  BOTH in lock-step when sweeping. (gotcha #53)
- **Persistent param buffer + 2-deep pipeline = race.** Duplicate the
  param ring (`gpu_param_buf[slot]`, `slot ∈ {0,1}`) the same way you
  duplicated the id ring. (gotcha #35)
- **Fused-residual epilogue shifts the bf16 round point.** `y = W·x +
  residual` rounds once (in f32) instead of twice; tokens may
  legitimately differ — regenerate the oracle or loosen tolerance.
  (gotcha #34)

## Profile-reading and "where's the gap?"

- **`gpu_busy << wall` after batched dispatches** → host-side breaks
  (look for `commit_wait` inside `forward()`); apply H1 glue kernels.
  Usually the single biggest win. (gotcha #9)
- **Argmax over VOCAB=200k single-threaded** is by far the most common
  "why is decode capped at <5 tok/s" answer (also caps mid-pipeline
  decode at ~40 tok/s once everything else is in). Apply F1 immediately.
- **Diffusion-LLM samplers do CPU work invisible to `gpu_busy`.**
  Per-step softmax/argmax/confidence over V shows up only in wall.
  Add `host_post`; if > ~5% of wall, apply F3. (gotcha #36)
- **Per-layer `embed_gather` is a HIDDEN TG=(1,1,1) variant
  worth 4-5% wall on Gemma 3/4-style models.** Naive ports keep
  the C-style `for (d=0; d<D; d++) out[d]=tab[r,d]` as 1 thread
  per row → 596 µs/call × 42 layers ≈ 25 ms/token of pure latency.
  KPROF surfaces it as a 4% kernel that "shouldn't be that slow".
  Fix is TG-per-row + 128-thread bfloat4 cooperative copy. Single
  biggest late-stage win on Gemma 4 E4B (+4.5% short, +3.7% long).
  (gotcha #57)
- **1-SG-per-row rmsnorm undersubscribes GPU at small M.** At M=BL=16
  you spawn 16 SGs vs M4 Max's ~160 capacity — kernel becomes
  memory-latency-bound. Use 4–16 SGs/row, merge via TG-mem. (catalog F4)
- **"Are we done?" DRAM-BW ceiling check.** Sum per-token weight
  bytes ÷ peak GB/s = your theoretical floor. At ≥ 95% of that
  floor, further bf16 optimization is impossible; only quantization
  can win more. Use this BEFORE chasing the last 1-2% — it usually
  isn't there. (gotcha #59)

## Tuning sweeps

- **K_OUT in qmv4 has a sweet spot.** 4 wins for q8 affine dequant; 8
  hurts (41 → 31 tok/s in one port) due to register spills. Measure
  before scaling up. The sweet spot only moves DOWN with heavier
  inner loops (e.g. after applying B6 uint4 W loads, K_OUT=8
  regressed AGAIN — 103 → 90.7 tok/s). Stay at 4. (gotcha #8, #60)
- **Quant GEMV: amortize `(scale, bias)` across the quant group.**
  Naive q8 inner loops re-read `(s, b)` every uint32 even though
  they're constant for the whole group. Load packed W as `uint4`
  (16 q8 weights = one 64-elem group) → one `(s, b)` per uint4.
  Single biggest second-pass win on Qwen3.6 — +18.7% decode.
  Same trick generalizes to q4 and mxfp4. See catalog B6.
- **Premature multi-SG SDPA hides the real SDPA win.** Apply D1 (parallel
  softmax) BEFORE D3 (multi-SG split), or you'll measure +5–10% and
  conclude SDPA isn't worth optimizing. (gotcha #31)
- **Tile-size cargo-culting.** MLX's tile sizes for M2 are not optimal
  for M4 or M1. Always sweep `(BM, BN, WM, WN)` on your target
  machine. (gotcha #32)
- **"BK=16 is the maximum" is a `WN=2` artifact.** At `WN=4` per-SG
  register pressure halves, unlocking BK=32/BK=64. Always sweep
  BK ∈ {16,32,64} × WN ∈ {2,4} with the register-budget formula.
  (gotcha #41, catalog C2)
- **SDPA inner-loop ILP unrolling is separate from D1/D2/D3.**
  Process 8 K-vectors/iter with 8 independent accumulators. -81%
  cumulative SDPA on Dream-7B. (catalog D5)
- **Optimal SDPA `N_SG` is NOT the perfect-occupancy divisor.**
  Always sweep `N_SG ∈ {12, 16, 20, 24, 28}` on LONG decode (not
  short — at low Lk SDPA share is too small to discriminate). On
  Qwen3.6 M4 Max 1500-tok decode, 20 wins; 16/20 within 1% of each
  other but 12/28 are 5%+ worse. (gotcha #54, catalog D3)
- **Two M shapes need two PSOs.** When prefetch (M=L) and refine
  (M=BL) differ by 4–5×, compile two GEMM PSOs with different
  `(BM, BK)` and switch at dispatch by `M > 16`. Pad workspace buffers
  to the larger `BM`. (gotcha #38)
- **GEMM aspect-ratio routing — both K>N and small-N need attention.**
  K>N matmuls at BM=32 BN=64 produce only N/BN TGs (`use_bm32 =
  (M > 16) && (K <= N)`); small-N refine matmuls at M≤16 BM=16 N<5120
  BN=64 are TG-starved on the N-axis (`use_bn32 = (M <= 16) &&
  (N < 5120)`). (gotcha #39, catalog C4 + C5)

## Optimizations that REGRESS in some regimes

- **B4 (TG-mem X-share) can REGRESS once B6 is in.** Apple-Silicon
  cores effectively share X across SGs via L1/L2, so explicit
  TG-mem staging adds barrier + load cost without saving DRAM BW.
  Cataloged 1.05-1.10× was on early-stage W-dominant ports; after
  B6 the ratio flips and B4 can lose 1-2%. Measure and revert if
  it regresses. (gotcha #56, catalog B4)
- **A8 concurrent encoder is useless on dense bf16 models.**
  Catalog A8 works on quantized/MoE because the dominant linear
  is at ~50% of W-BW peak (dequant overhead leaves bandwidth
  free). On dense bf16 every chain is at 100% of peak — running
  two BW-bound chains concurrently means each takes 2× longer.
  Skip A8 for Gemma 4, Llama-3 bf16, Mistral bf16, etc. (gotcha #58)
- **At decode, GEMM is W-bandwidth bound, not M.** Halving M
  (row-selective compute) when `M ≪ K` saves <1% — the W read
  dominates. Only worth it on compute-bound kernels. (gotcha #49)

## Weight loader

- **`mmap + memcpy` is page-fault-bound at ~1 GB/s on macOS.** Even
  with `dispatch_apply` parallelism you cap ~4 GB/s aggregate (VM
  fault handler serializes on per-page locks). Use parallel `pread`
  per shard — one fd per shard, sequential reads, dispatch_apply
  across shards. Critical for diffusion / short benches. (gotcha #29, A9)
- **`newBufferWithBytesNoCopy:` per-tensor doesn't work; per-shard
  works but is slower than pread on warm cache.** Per-tensor fails on
  8-byte alignment (#40); per-shard is API-valid but page-fault BW
  (~20 GB/s) loses to pread's page-cache memcpy fast path (~50 GB/s).
  Counter-intuitive trap. (gotchas #40, #47)
- **Co-tune `SHARD_SPLIT` against bg compile + residency_async.**
  pread workers + bg compile + bg residency all contend for P-cores.
  Target `n_shards × SHARD_SPLIT ≈ P_cores` (12 on M4 Max). Going
  higher regresses by starving bg compile. (gotcha #45)
- **Async kernel compile QoS matters.** Start compile + PSO creation
  on a bg dispatch queue *before* `cache_weights()`. Use
  `QOS_CLASS_USER_INITIATED`, not `USER_INTERACTIVE` (steals from
  pread). Metal compile is system-cached cross-process (~3 ms warm),
  so the overlap is essentially free. (gotcha #46)
- **Shard-sized MTLBuffer + per-tensor views.** One MTLBuffer per
  safetensors shard; each tensor is a `(parent, offset, size)` view.
  Residency set shrinks N_tensors → N_shards (339 → 4). Small but
  consistent wall win. (gotcha #50)

## "Unreproducible" regressions

- **Stale binary / thermal trap.** User reports a tok/s regression
  that you cannot reproduce. Common causes: thermally loaded GPU
  (background process), stale `.o` files in build, or DYLD caches
  from prior debug session. Fix: `make clean && make`, close GPU-
  using apps, `sleep 60` before first bench, best-of-5 with
  `sleep 30` between runs. (gotcha #55)
