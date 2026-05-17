# Gotchas — bug-and-fix war stories

Concrete bug-and-fix war stories from optimizing src-metal/ ports.
Each one cost at least one debug iteration; consult BEFORE chasing
the same symptoms again.

## Table of contents

| #  | Topic                                                                | Category    |
|----|----------------------------------------------------------------------|-------------|
| 1  | `simd_ballot` returns `simd_vote`, not uint                          | MSL syntax  |
| 2  | `dispatchThreads:` takes TOTAL THREADS, not threadgroups             | Dispatch    |
| 3  | TG-memory MAX_CTX must match KV-cache `max_ctx`                      | SDPA bug    |
| 4  | `constant constexpr uint NAME` collides across kernel files          | MSL build   |
| 5  | First cmdbuf has 0.5–3 s of one-time driver overhead                 | Timing      |
| 6  | Decode tok/s variance is ±20% with short runs                        | Measurement |
| 7  | Forgetting `threadgroup_barrier` between phases of a same-SG kernel  | Barriers    |
| 8  | K_OUT in qmv4 has a sweet spot — bigger isn't always better          | Register tile |
| 9  | Host-side breaks are the single biggest hidden cost                  | Host overhead |
| 10 | Verify the FIRST decoded token IS the prefill argmax                 | Off-by-one  |
| 11 | `forward()` should not own its cmdbuf                                | Architecture |
| 12 | Order of attack — bottlenecks worth checking in this order           | Strategy    |
| 13 | Metal queue ordering = free RAW correctness between cmdbufs          | Pipeline    |
| 14 | Use `commit()` (async) for pipelining, `commit_wait()` only at boundary | Pipeline |
| 15 | port-c-to-metal often leaves `max_ctx` at the smoke-test value       | SDPA bug    |
| 16 | Two id buffers, NOT one shared id buffer for the 2-deep pipeline     | Pipeline    |
| 17 | Tiny TG sizes — (1,1,1) wastes 31/32 of every SIMD group             | Dispatch    |
| 18 | Concurrent encoder + no_bar = YOU own hazard tracking                | Barriers    |
| 19 | Critical-path concurrency > kernel speedups in late optimization     | Strategy    |
| 20 | KPROF total > wall time means concurrent encoder is actually working | Profiling   |
| 21 | Fuse residual into the LAST linear, not into a separate kernel       | Fusion      |
| 22 | Common barrier-removable groups in transformer LLMs                  | Barriers    |
| 23 | KPROF inflates total runtime ~30% but is the right tool              | Profiling   |
| 24 | Fuse out-of-place elemwise pairs into one kernel                     | Fusion      |
| 25 | tok/s differences < 2% are noise — run 5 times                       | Measurement |
| 26 | Two different "barriers" — don't confuse them                        | Barriers    |
| 27 | `forward()` should ACCEPT a cmdbuf, not own one                      | Architecture |
| 28 | macOS `madvise(MADV_WILLNEED)` is BLOCKING — never use it on big files | Startup |
| 29 | mmap + memcpy is page-fault-bound at ~1 GB/s — use parallel pread    | Startup     |
| 30 | First Metal cmdbuf pays ~1 s residency wiring per ~30 GB of buffers  | Startup     |
| 31 | Premature multi-SG SDPA hides the real SDPA win                      | SDPA strategy |
| 32 | Tile-size cargo-culting — MLX tiles for M2 ≠ optimal for M4/M1       | GEMM tuning |
| 33 | Forgetting GQA when SG handles a (token, head) pair                  | SDPA bug    |
| 34 | Fused-residual epilogue shifts the bf16 round point                  | Fusion / numerics |
| 35 | Persistent param buffer + 2-deep pipeline = race; duplicate the ring | Pipeline    |
| 36 | Per-step CPU sampler work is invisible to `gpu_busy` — track host_post | Profiling |
| 37 | `gpu_buf_contents()` is zero-copy but per-step CPU reads still hit caches | Sampler perf |
| 38 | Two M shapes → two PSOs with different (BM, BK); pad workspaces to max BM | GEMM tuning |
| 39 | Aspect-ratio routing: K > N matmuls need MORE TGs (use the smaller BM)    | GEMM tuning |

---

# 1. simd_ballot returns simd_vote, not uint

The MSL type for simd_ballot() is simd_vote (a 64-bit wrapper) that
does NOT implicitly convert to uint. Naive:

    uint vote = simd_ballot(best_v == sg_max);    // ERROR

Fix:

    uint vote = uint((simd_vote::vote_t)simd_ballot(best_v == sg_max));

Same for simd_active_threads_mask().

# 2. dispatchThreads: takes TOTAL THREADS, not threadgroups

The shim's gpu_cmdbuf_dispatch(grid_x, ..., tg_x, ...) maps to Metal's
[encoder dispatchThreads:MTLSizeMake(grid_x,...)
 threadsPerThreadgroup:...]. The grid_* values are TOTAL threads, NOT
threadgroup counts. For one SIMD group per output element:

    // Want N threadgroups of 32 threads each:
    gpu_cmdbuf_dispatch(cb, pso, args, n,
                        32 * N, M, 1,   //  <-- grid is 32*N, NOT N
                        32, 1, 1);

Inside the kernel threadgroup_position_in_grid.x then ranges 0..N-1.
Forgetting the 32* factor silently launches ceil(N/32) SGs that each
compute up to 32 outputs the kernel was not designed for.

# 3. TG-memory MAX_CTX must match the KV-cache max_ctx

SDPA stages per-key scores in TG memory:
threadgroup float scores[MAX_CTX];. There is a SEPARATE max_ctx in
main.c that sizes the KV cache buffers. BOTH must agree, AND both
must be >= prompt_len + max_tokens.

If the KV cache max_ctx is too small, writes silently overrun adjacent
Metal buffers (no error, no crash; just wrong tokens far into the
output). The "first N tokens match" check still passes for small N,
masking the bug.

# 4. constant constexpr uint NAME collides across kernel files

Multiple .metal files JIT-compiled into one library share a namespace.
Two files each defining
    constant constexpr uint SIMD_WIDTH = 32u;
fail to compile with "previous definition is here".

Fix: prefix per-file:
    constant constexpr uint LQ8_SIMD_WIDTH = 32u;    // linear_q8.metal
    constant constexpr uint ARGMAX_TG_SIZE = 256u;   // argmax.metal

# 5. First cmdbuf has 0.5-3 s of one-time driver overhead

Even with kernel_concat + JIT-compiled metallib, the FIRST cmdbuf in a
process pays a large fixed cost — Metal lazily materializes pipeline
state objects, makes weight buffers resident, etc. This is NOT counted
in cmdbuf.GPUEndTime - GPUStartTime.

Concretely (Qwen3.6-35B-A3B on M4 Max):
    prefill: 3.09s wall (16 tokens; 5.2 tok/s; gpu_busy=0.09s)
    decode:  3.12s wall (128 tokens; 41.0 tok/s; gpu_busy=3.11s)

GPU did 90 ms of prefill compute but the cmdbuf took 3.09s wall.

Always print BOTH wall and gpu_busy in your decode line — the gap is
your remaining CPU-encode overhead, which is what the 2-deep pipeline
(A4) eliminates.

# 6. Decode tok/s variance is +/-20% with short runs

Single short runs (8-32 tokens) on a busy machine routinely show
3-5x tok/s variance from other apps / thermal state / Metal queue
contention. Stable measurements need AT LEAST 64 decoded tokens AND
multiple runs (best-of-3).

# 7. Forgetting threadgroup_barrier between phases of a same-SG kernel

Within one SIMD group, threads execute lockstep, so within-kernel
phases that all 32 lanes participate in don't strictly need a barrier.
BUT as soon as you stash data via TG memory and a different phase
READS it back, the barrier IS needed even within one SG.

Concrete: in sdpa_sg_per_head_decode.metal, writes to scores[lk] by
lane 0 (one per outer loop iter) are followed by reads from scores[i]
by lanes i = lane, lane+32, ... Other lanes need a barrier to see
lane 0's writes from earlier in the loop. Forgetting it produces
non-deterministic garbage that varies run-to-run.

# 8. K_OUT in qmv4 has a sweet spot — bigger isn't always better

K_OUT=4 in our Qwen3.6 q8 port was a win. K_OUT=8 made decode WORSE
(41 -> 31 tok/s) because the inner loop bloated register pressure:
8 weight unpacks + 8 float accumulators + 8 dot4s caused the
compiler to spill registers.

Rule: if K_OUT=X helps but X+1 (or 2X) hurts, X is your sweet spot.
Default to K_OUT=4 for q8/affine dequant. Revisit only for bf16 dense
weights or after MMA has displaced GEMV.

# 9. Host-side breaks are the single biggest hidden cost

A naive port has commit_wait + cb = new_cmdbuf() every time the host
does a small scratch op (state shift, broadcast multiply, KV cache
append). Each commit_wait costs 0.3-1.0 ms even for tiny cmdbufs. On
a 40-layer model with 3 breaks per layer, decode pays 40-120 ms/token
of fixed overhead REGARDLESS of GPU speed.

See glue_kernels_no_host_break.metal. In the Qwen3.6 port this was
the LARGEST single optimization: decode 19 -> 34 tok/s (1.79x).

When gpu_busy << wall after batched dispatches, suspect host breaks
FIRST, before chasing kernel optimizations.

# 10. Verify the FIRST decoded token is the prefill argmax

The prefill argmax IS the first generated token:
    prefill -> argmax -> next_id   <-- THIS IS GENERATED TOKEN 1
    loop:
        embed(next_id); forward; argmax -> next_id   <-- TOKEN 2, 3, ...

When you switch to a 2-deep pipeline that primes itself, it's easy
to skip emitting the prefill argmax. Result: every generated token is
shifted by one, "first N tokens match" check fails because everything
is offset by 1.

# 11. forward() should not own its cmdbuf

Naive forward() creates a cmdbuf, dispatches, and commit_waits at the
end. For pipelining and for fusing embed+forward+argmax into one
cmdbuf, forward() must take cb as a parameter and append into it
without committing. The caller owns commit().

# 12. Order of attack — bottlenecks worth checking in this order

For decode (Lq=1) on a reduced-precision (quantized) MoE LLM, this is
the empirical order of biggest wins from a naive port:

  1. Parallelize ANY single-thread-per-row reduction:
     argmax over vocab (the king of decode bottlenecks at >50k vocab),
     softmax_topk over N_experts, rmsnorm rows.
  2. SIMD-group-per-output for ALL reduced-precision GEMV (B1).
  3. Parallelize the recurrent state update if the model has one
     (gated_delta / Mamba SSM). See recurrent_state_sg_per_row.metal.
  4. SDPA SG-per-(lq,hq) with TG-mem scores + parallel softmax.
  5. ELIMINATE HOST-SIDE BREAKS (glue kernels). Usually the biggest
     single jump.
  6. Merge embed + forward + argmax into one cmdbuf per decode step.
  7. 2-deep cmdbuf pipeline with id-swap (closes the last gap to MLX).

Optimizations whose theoretical wins are real but practical wins
were noise-band in this model (and may differ on others):
  - K_OUT register tile (qmv4) — only +0% to +5% for q8 dequant on M4.
  - bfloat4 vector loads (already baked into the SIMD-group-per-output
    kernels — separating them out doesn't add much).

# 13. Metal queue ordering = free RAW correctness between cmdbufs

Cmdbufs committed to the same `MTLCommandQueue` execute in commit
order. This is what makes the 2-deep pipeline's id_io_buf swap correct
WITHOUT explicit fences/events: argmax in cb[k] writes id_io_buf[1-s];
embed in cb[k+1] reads it. Even though both are in-flight, Metal
guarantees cb[k] finishes (and its GPU stores are visible) before
cb[k+1] starts.

If you ever introduce a SECOND queue (e.g. for async loading), you
lose this guarantee and need `MTLSharedEvent` signalling.

# 14. Use commit() (async) for pipelining, commit_wait() (sync) only at boundary

The shim has two committers:
  * gpu_cmdbuf_commit(cb)         — kick off, return immediately
  * gpu_cmdbuf_commit_wait(cb, e) — kick off, BLOCK until done
                                    (== commit() then waitUntilCompleted)

The 2-deep pipeline MUST use the async `commit()` + later `wait()`,
not `commit_wait()`. Mixing them up reduces the pipeline back to
serial.

# 15. Port-c-to-metal often leaves max_ctx at the smoke-test value

The naive port from port-c-to-metal validates with --max-tokens 8 and
hardcodes c->max_ctx accordingly (e.g., 64). The first thing the
optimize-metal skill should do is bump max_ctx to a real benchmark
size (1024+) AND match it in the SDPA TG-mem MAX_CTX. Skipping this
produces silent buffer overruns that "first N tokens match" tests
won't catch.

# 16. Two id buffers, NOT one shared id buffer for the 2-deep pipeline

You might think: "the CPU only reads next_id after wait, and Metal
serializes cmdbufs, so one shared id buffer should work". It does
NOT, for a subtle reason: the moment cb[k] finishes (releasing wait),
cb[k+1] is already RUNNING (it was committed first), and its argmax
is going to OVERWRITE the same buffer before the CPU has time to
fread it. Use two buffers, with step at slot s writing to
id_io_buf[1-s], so the CPU read of id_io_buf[1-s] (the buffer cb[k]
wrote) is safe because cb[k+1] is writing to id_io_buf[s] instead.

# 17. Tiny TG sizes — (1,1,1) wastes 31/32 of every SIMD group

dispatchThreads:(N, 1, 1) threadsPerThreadgroup:(1, 1, 1) puts one
thread per TG. Metal allocates an entire SIMD group (32 lanes) per
TG regardless. Result: 31/32 lanes idle, ~32x dispatch overhead for
nothing.

Symptom in KPROF: a kernel that should take <2 us (e.g. residual_add,
silu_inplace, sigmoid, qkv_split) reports ~25 us/call.

Fix: bump tg_x to 32, ensure the kernel has a `if (i >= N) return;`
bounds check (most port-c-to-metal naive kernels already do).

```
gpu_cmdbuf_dispatch(cb, pso, args, n_args,
                    (size_t)N, 1, 1,         // total threads
                    32, 1, 1);               // 32-thread TG, not 1!
```

Real measured impact on Qwen3.6 (M4 Max, batch of ~15 small kernels):
+1 tok/s overall (65.9 → 66.0). Small but free.

# 18. concurrent encoder + no_bar = YOU own hazard tracking

`computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent` means
Metal does NOT auto-serialize dispatches. The shim's
`gpu_cmdbuf_dispatch` (no_bar) lets the next dispatch start
immediately. If there's a data hazard you didn't account for, you get
non-deterministic garbage that often "passes" on small N because the
race resolved lucky.

Rule: use `dispatch_bar` after any kernel whose output is read by
the next kernel. Use plain `dispatch` only when you've verified
no_hazard (different buffers, or same buffer but disjoint regions).

Test: after changing barriers, ALWAYS re-run `--max-tokens 16` and
verify the exact token-id sequence matches a known-good run. If it
diverges, you have a missing barrier.

# 19. Critical-path concurrency > kernel speedups in late optimization

Once individual kernels are ~bandwidth-saturated (linear_q8 at 25us
for K=N=2880 is reading ~11MB of weights at ~430 GB/s — within 7%
of M4 Max's peak), making the kernel "faster" buys nothing. The only
remaining wins are:

  (a) Reducing the critical path (fuse dispatches that produce X→consume X)
  (b) Removing barriers between independent chains so they run in
      parallel (see parallel_independent_chains.md)

For Qwen3.6-35B-A3B at >65 tok/s, ALL further wins came from category (b).
None from kernel-internal optimization.

# 20. KPROF total > wall time means concurrent encoder is actually working

`KPROF` mode wraps each dispatch in its own cmdbuf with commit+wait,
serializing everything. KPROF total time = serialized GPU work.

Real run wall ≪ KPROF total ⇒ concurrent encoder is overlapping work.
For decode at n=210 we measured:
  - KPROF total: 22.2 ms/tok
  - Real wall: 14.4 ms/tok
  - Overlap factor: 35%

If KPROF total ≈ wall, your concurrent encoder ISN'T overlapping;
suspect missing barrier-removal opportunities, not slow kernels.

# 21. Fuse residual into the LAST linear, not into a separate kernel

The classic B5 pattern (residual into o_proj/out_proj epilogue):
the catalog says +3-5%. In our q8 case it was +0.5%. Why? Because
residual_add at TG=(32,1,1) was already overlapping with the
subsequent rmsnorm via the concurrent encoder, so the dispatch
"cost" was mostly hidden.

The fusion is still WORTH doing because:
  1. It eliminates a launch-overhead dispatch
  2. It keeps the bf16 round point identical to what an "in-fused"
     reference produces (helps token agreement for longer N)
  3. It composes well with later concurrency restructuring

But don't expect huge numbers when concurrent encoder is on; the win
shows up cumulatively, not per-fusion.

# 22. Common barrier-removable groups in transformer-LLMs

Late-stage optimization is largely about finding pairs / groups of
dispatches that have NO data hazard, so they can run concurrently.
This is the checklist that pays off in practice (validated on
Qwen3.6-35B-A3B and similar models):

WITHIN one attention layer:
  * QKV: q_proj || k_proj || v_proj  (all read post-rmsnorm H)
  * post-projection norms: q_norm || k_norm  (different inputs)
  * RoPE: rope_q || rope_k  (different inputs)
  * For RMSNorm + dim-scale fused: rmsnorm_scale_q || rmsnorm_scale_k

WITHIN one linear-attention/SSM layer:
  * silu_inplace(conv_out) || conv_state_update  (different buffers)
  * rmsnorm_scale_q || rmsnorm_scale_k || sigmoid(beta) || compute_g
    (4-way, all different inputs/outputs)
  * QKV gates: in_proj_qkv || in_proj_z || in_proj_b || in_proj_a
    (4-way, all read same H, write disjoint outputs)

WITHIN one MoE layer (the BIG win):
  * MoE chain (gate_gather + up_gather + silu_mul + down_gather +
    expert_mix) || shared-expert chain (gate_proj + up_proj +
    silu_mul + down_proj + shared_expert_gate)
  * Within MoE: moe.gate_proj || moe.up_proj  (both read h_buf)
  * Within shared: shared.gate || shared.up || shared_expert_gate

PATTERN: any dispatch group with the property "shared input(s),
disjoint outputs" is barrier-removable. The fan-out points (post-
rmsnorm) and fan-in points (o_proj + residual, shared_combine_add)
are where you still need barriers.

When in doubt, list every dispatch's buffer reads and writes; group
the ones with no overlap.

# 23. KPROF inflates total runtime ~30% but is the right tool

KPROF mode wraps each dispatch in its own cmdbuf with commit+wait,
which adds ~30% to total runtime (each commit/wait costs 0.3-1ms,
and we're now doing ~340 commits/token).

DO NOT use KPROF tok/s as a real performance number. KPROF gives:
  * Correct RELATIVE ordering of which kernels are hot.
  * Correct ABSOLUTE per-kernel GPU time.
  * INFLATED total wall time.

The procedure:
  1. Run normally to get real tok/s.
  2. Run with KPROF=1 to get the breakdown.
  3. Attack the top-1 kernel by GPU time, OR look for opportunities
     to overlap the top-N kernels (per-kernel time + concurrency).

In late-stage optimization, look at KPROF total time vs real wall
time: if KPROF total ≈ wall, you're NOT overlapping (apply A5/A8).
If KPROF total >> wall (e.g., 1.4×), the concurrent encoder is
overlapping work successfully.

# 24. Fuse out-of-place elemwise pairs into one kernel

A common pattern from the naive port:
  1. copy_bf16(src, dst)         — N writes
  2. sigmoid_inplace_bf16(dst)   — N reads + N writes

Fuse into a single out-of-place kernel:
  sigmoid_bf16(src, dst)         — N reads + N writes (no inplace)

Eliminates one dispatch (~50us launch overhead) AND drops a
buffer-to-buffer copy. Generalizes to any pair where the first
kernel writes and the second reads/writes the same buffer.

Examples from Qwen3.6:
  copy + sigmoid_inplace → sigmoid_bf16 (out-of-place)
  copy + silu_inplace    → could similarly be silu_bf16
  copy + scale_by_alpha  → fused_scale_bf16

Even more general: any "init buffer; then modify it" pair is fusable.

# 25. tok/s differences < 2% are noise — run 5 times

Single-run decode tok/s on a quiet machine still has ~1-2% variance
from:
  * Metal command-queue scheduling jitter
  * Other processes on the GPU
  * Thermal state
  * Memory pressure

When a commit claims a 1-2% speedup, verify with 5 runs and take
median or best-of-3. Anything below 2% is in the noise band.

Conversely: if a 5-run median is +3% or better, the win is real
(in our session 66.0 → 66.9 was a 5-run-confirmed real +1.4% win,
just barely).

# 26. Two different "barriers" — don't confuse them

Metal has TWO barrier mechanisms; they are NOT interchangeable:

  (a) `threadgroup_barrier(mem_flags::mem_threadgroup)` — synchronize
      threads within ONE threadgroup. Used INSIDE a kernel to
      coordinate phases (e.g., write TG-mem then read).

  (b) `memoryBarrierWithScope:MTLBarrierScopeBuffers` — synchronize
      dispatches within ONE cmdbuf when concurrent encoder is in use.
      Used BETWEEN dispatches in main.c.

When porting from serial encoder (which auto-emits (b) after every
dispatch) to concurrent encoder, you only need (b). (a) is unaffected.

If you see "first 16 tokens match, then garbage", check (b) — a
missing inter-dispatch barrier.
If you see "every token is garbage", check (a) — a missing intra-
kernel barrier (often introduced when refactoring multi-phase kernels
like SDPA).

# 27. forward() should ACCEPT a cmdbuf, not own one

(See also GOTCHA #11.) Specifically: with 2-deep pipeline AND with
the embed+forward+argmax merge, forward() must take `gpu_cmdbuf*
cb` as a parameter and only append dispatches. The caller decides
when to commit. The naive port (one commit_wait at end of forward())
caps decode at the ~110-tok/s ceiling determined by per-cmdbuf
overhead — even if your kernels are infinitely fast.


# 28. macOS madvise(MADV_WILLNEED) is BLOCKING — never use it on big files

On Linux, `madvise(addr, len, MADV_WILLNEED)` is a hint: the kernel
prefetches pages asynchronously. On macOS it is **synchronous** — the
call does not return until the entire range has been read into the
page cache. For a 35 GB safetensors archive this is ~5s of pure
"loading" with no progress feedback.

**Symptom**: `st_open` (or equivalent header-scan function) takes
inexplicably long. Profile breakdown reveals it's the madvise call.

**Fix**: Just delete the call. The actual page-ins should happen during
your weight-copy step instead, where you can parallelize them.

# 29. mmap + memcpy is page-fault-bound at ~1 GB/s — use parallel pread instead

The naive port loads weights as: `mmap(shard)` → `memcpy(dst, mmap_ptr,
nbytes)` per array (tensor). This is fast on first inspection because
mmap is zero-cost. But the **memcpy triggers page faults**, and on
macOS the fault handler serializes on VM locks. Even with
`dispatch_apply` to parallelize across arrays, you get ~4 GB/s
aggregate.

**Fix**: skip mmap for the bulk copy entirely. Use parallel
`pread()`:

  1. Keep the shard JSON header parse via mmap (it's tiny).
  2. For the bulk data, open ONE fd per shard.
  3. `dispatch_apply` over `n_shards` (e.g., 8): each iteration
     sequentially `pread`s its shard's arrays directly into the
     preallocated MTLBuffers.

This matches MLX's `ParallelFileReader` strategy (4 worker threads,
32 MB chunks). On M4 Max with 8 shards it hits ~10 GB/s — a 2.5×
speedup, and brings startup time to **below MLX's**.

**Pattern**: `patterns/load_parallel_pread.c`

**Measurement** (35 GB Qwen3.6 model, M4 Max, warm cache):
  - mmap + single-threaded memcpy + madvise: 5s (madvise) + 5s (copy) = 10s
  - mmap + dispatch_apply(per-array) memcpy:  8s
  - parallel pread per shard:                   3.5s   ← winner

# 30. First Metal cmdbuf pays ~1 s residency wiring per ~30 GB of buffers

The very first cmdbuf submitted to Metal that references a large
working set of MTLBuffers blocks for ~1s before the GPU starts work.
`gpuStartTime - committedTime` is the wait; actual GPU execution time
(`gpuEndTime - gpuStartTime`) is much smaller. This is Metal's
residency / wiring of the buffers into the GPU address space, paid
exactly once.

**Symptom**: `prefill: 1.5s (16 tokens; 11 tok/s; gpu_busy=0.08s)` —
huge gap between wall time and gpu_busy.

**Quantification**: ~1s per ~30 GB resident, scales roughly linearly
with the number of distinct MTLBuffers (2090 in our case).

**Workaround attempts that did NOT work**:
- Trivial argmax-only warmup cmdbuf: ineffective because it only
  wires the 2 buffers it references; the rest still pay on the real
  cmdbuf.
- Full forward(Lq=1) warmup cmdbuf: works (shifts the 1.4s out of
  prefill into "warmup"), but TOTAL wall time is unchanged.
- Run warmup forward(Lq=1) CONCURRENTLY with parallel pread copy:
  fails because the CPU pread writes to wired buffers, and Metal
  appears to invalidate residency on concurrent CPU writes, so the
  subsequent real prefill re-pays the residency cost.

**Workable mitigation**: do the full forward(Lq=1) warmup AFTER copy
finishes. It shifts the 1.4s out of the user-visible "prefill" phase
(time-to-first-token) but does not reduce total wall time. Only use
this if TTFT matters more than total time.

**Conclusion**: this is a fixed cost. Beating MLX on total wall time
comes from saving I/O time, not from removing this 1s.

# 31. Premature multi-SG SDPA hides the real SDPA win

**Symptom**: you split a head across multiple SGs (D3) before fixing
the dumb 1-thread-per-(query, head) SDPA. You measure +5–10%, conclude
"SDPA isn't worth optimizing", and move on.

**Fix**: apply D1 first (parallel softmax via SIMD reductions). The
serial softmax inside the 1-thread baseline is what's hiding the win.
Once each (query, head) pair is handled by a full SG with parallel
softmax, multi-SG splitting (D3) gives the real 1.5–2× on long context.

The order matters: D1 → D2 (online softmax) → D3 (multi-SG). Doing
D3 first measures a fraction of the available speedup and is misleading.

# 32. Tile-size cargo-culting — MLX tiles for M2 ≠ optimal for M4/M1

`BM=16 BN=32 WM=1 WN=2` happens to match MLX's "non-NAX" GEMM path for
gpt-oss-style shapes on M4 Max. **It is NOT a universal sweet spot.**

Tile sizes are sensitive to (model shapes, Apple GPU generation,
whether MLX is using its NAX path). Always sweep
`(BM, BN, WM, WN) ∈ {(16,16,1,1), (16,32,1,2), (32,32,2,2), (32,64,2,4)}`
on your target machine and your model's actual K-dim.

The cost of sweeping is one afternoon. The cost of not sweeping can be
30%+ left on the floor in prefill.

# 33. Forgetting GQA when SG handles a (token, head) pair

When the SDPA tile assigns one SG per `(query_token, head_q)` pair, the
K/V head index is **not** `head_q`. With grouped-query attention,

    head_kv = head_q / (N_q / N_kv);

Many tile refactorings get this right the first time, then break it
during a follow-up restructure (e.g., switching from "one SG per query
token" to "one SG per (query_token, head_q)"). Symptom: tokens diverge
after layer 1, with output that *looks* numerical but is actually
reading the wrong K head.

**Fix**: keep `head_kv = head_q / GROUP_SIZE` recomputed inside the
kernel; never reuse `head_q` as the K index.

# 34. Fused-residual epilogue shifts the bf16 round point

When you fuse `y = (W·x + residual)` instead of doing
`tmp = W·x` (bf16 round) followed by `y = tmp + residual` (bf16 round),
the residual is now added inside the same f32 accumulator. The bf16
round happens once at the end, not twice. **Token output may differ.**

This is correct (one fewer rounding step), but:

1. The first-N-tokens-match check against the unfused C reference may
   fail at token ~10–50 even though the kernel is "right".
2. The diff vs an ORACLE dump (MLX or HF) typically *shrinks* (fewer
   rounds = closer to f32).
3. Per-element tolerance vs the unfused C ref may need to loosen from
   1e-2 to ~5e-2 absolute on residual-stream arrays.

**Fix**: ALSO regenerate the C-reference oracle dump with the
equivalent fused-rounding kernel, OR loosen the tolerance and re-verify
the first N tokens match the model's actual greedy output (e.g., MLX
greedy on the same prompt).

# 35. Persistent param buffer + 2-deep pipeline = race; duplicate the ring

You have:

- ONE persistent `gpu_param_buf` that the host writes layer params into
  before each cmdbuf.
- 2-deep pipelining (cmdbuf N+1 encoded while N runs).

Then while encoding cmdbuf N+1, you overwrite `gpu_param_buf` with new
params, but cmdbuf N is still reading the OLD params on the GPU. Race.

**Symptom**: tokens diverge at layer boundaries; reverting to
1-cmdbuf-deep "fixes" it. You blame the pipeline.

**Fix**: duplicate the param ring — `gpu_param_buf[slot]` with
`slot ∈ {0, 1}` swapping per cmdbuf. Same fix as the id-buffer ring
(gotcha #16), applied to all per-cmdbuf scratch the host writes into.

# 36. Per-step CPU sampler work is invisible to `gpu_busy` — track host_post

Diffusion-LLM samplers (Dream, LLaDA, fastdllm) and any sampler with
non-trivial per-step CPU logic do work between `commit_wait()` and the
next dispatch:

- per-position softmax over V,
- argmax + confidence per position,
- threshold-based selection of which mask positions to commit,
- building next step's input ids.

Wall time contains this CPU work but `gpu_busy` does NOT. So you see a
large `wall - gpu_busy` gap and reach for the host-scheduling toolbox
(A4 / H1) — but those won't help, because there are no GPU dispatches
to overlap with the CPU work; the host loop is genuinely the gap.

**Fix the diagnostic first**: add a third accumulator (see
`references/profiling.md` "host_post" section). Print
`wall / gpu_busy / host_post`. If `host_post` is more than ~5% of
wall, attack F3 (move the per-row softmax+argmax+confidence to the GPU
as a single kernel).

In a Dream-7B port `host_post` was 0.25 s of 2.90 s wall. F3 dropped
it to ~0 s and wall to 2.63 s — closing more than half the remaining
gap to MLX in one commit. Without tracking `host_post` separately we'd
have spent the same effort hunting non-existent host-scheduling bugs.

# 37. `gpu_buf_contents()` is zero-copy but per-step CPU reads still hit caches

On Apple Silicon unified memory, `gpu_buf_contents()` returns a CPU
pointer to the same physical page the GPU just wrote — there is no
copy. So in principle reading a few KB of logits per step is free.

But per refine step you might read `BL × V × sizeof(bfloat)` bytes —
e.g., for Dream-7B with BL=16 and V=152064 that's 4.7 MB per step.
Across 64 steps × N decode batches that's hundreds of MB of CPU cache
reads on top of the per-position softmax computation itself.

**Symptom**: `host_post` is bigger than you'd predict from raw
arithmetic. CPU `top` shows your process at 60–90% on one core during
"GPU phase".

**Fix**: same as #36 — move the per-row reduction to the GPU (F3) so
the only host-side reads are the `M × 8 bytes` of
`(argmax_idx, confidence)`. Even at BL=16 that's 128 bytes/step, which
truly is free.

# 38. Two M shapes → two PSOs with different (BM, BK); pad workspaces to max BM

A diffusion-LLM port has two very different shapes for the same matmul
kernel:

- **Prefetch**: one full forward at `M = L` (e.g., 89). The host
  consumes only one or two rows of the output.
- **Refine**: many forwards at `M = BL` (e.g., 16).

A single (BM, BK) tile is suboptimal for both. The best tile at M=16
typically has BM=16 + aggressive BK=16 unroll; at M=89 it's BM=32 +
modest BK=8 (BK=16 spills the A register footprint at TM=4).

**Strategy**: compile two PSOs — one per (BM, BK) combo — and pick at
dispatch time:

```c
gpu_pipeline* pso = (M > 16) ? pso_gemm_bm32 : pso_gemm_bm16_bk16;
```

**Buffer caveat**: when ANY path uses BM=32, ALL workspace buffers
(intermediate activations, the lm_head input, etc.) must be padded to
`(L + 31) & ~31` rows. The BM=32 path's tail TG reads rows [L, L+31)
unconditionally; if the buffer ends at row L you get a GPU page fault
or corrupted neighbouring data. The smaller tile (BM=16) needs
`(L + 15) & ~15` padding; use the LCM.

This won us ~5–8% on Dream-7B refine on top of the dual-tile dispatch.
The cost is one extra PSO compile at startup and one extra weight
buffer view (no copy — same backing store).

# 39. Aspect-ratio routing: K > N matmuls need MORE TGs (use the smaller BM)

Extends #38. Even within a single M shape, two matmuls of *identical
matrix size* can have wildly different bandwidth efficiency depending
on (K, N) aspect ratio. The same BM=32 BN=64 tile that's optimal for
N >> K shapes (gate_proj, lm_head, etc.) can leave 30%+ on the floor
for K >> N shapes (down_proj is the textbook case in any LLaMA-style
MLP: H=3584 hidden ↔ I=18944 intermediate).

In Dream-7B BL=32 refine, KPROF attributed 38.5% of refine GPU time
to a SINGLE kernel — down_proj. Same kernel as gate_proj (BM=32 BN=64
WM=1 WN=2), same total matrix size (3584×18944×2 = 135 MB W), but:

  gate_proj (K=3584, N=18944):  430us/call,  ~314 GB/s effective
  down_proj (K=18944, N=3584): 1082us/call,  ~125 GB/s effective

**Root cause**: TG count is N/BN tiles per M-band. With BM=32 and
BN=64, that's:
  gate_proj: N/BN = 18944/64 = **296 TGs** (well above GPU capacity →
             plenty of in-flight memory ops → bandwidth fills)
  down_proj: N/BN = 3584/64  =  **56 TGs** (below M4 Max's ~80 TG-slot
             capacity → outstanding loads starve → bandwidth collapses)

The W matrix doesn't fit in L2 either way, so the W traffic IS the
bottleneck. More TGs in flight means more memory requests in flight
means more bandwidth used.

**Fix**: at dispatch time, route K > N shapes to the SMALLER BM kernel.
This doubles M-bands from 1 to 2 (for M=32), doubling TG count from
56 to 112 — enough to fill the memory subsystem:

```c
// In d_linear_full_off (or wherever you pick a PSO by M):
//   normally:  pso = (M > 16) ? pso_gemm_bm32 : pso_gemm;
// becomes:
int use_bm32 = (M > 16) && (K <= N);  // K > N → use BM=16 even at M>16
const uint32_t BM = use_bm32 ? 32u : 16u;
gpu_pipeline* pso = use_bm32 ? pso_gemm_bm32 : pso_gemm;
```

The "double the W reads from DRAM" downside of going BM=16 is real
(2 M-bands × 135 MB = 270 MB streamed instead of 1 × 135 MB), BUT
the bandwidth gain dominates because at the smaller-BM occupancy
W reads HIT 350+ GB/s effective. Concretely on Dream-7B BL=32:
  before: down_proj 1082us, refine 73ms, wall 1.72s
  after:  down_proj  741us, refine 63ms, wall 1.52s

That's **12% wall improvement from a single-line dispatch change**.

**How to spot**: when KPROF shows one matmul taking >2× longer than
another of the same matrix size, check (K, N): the larger-K one is
almost certainly under-occupied at BM=32. The trigger condition is
simply `K > N` for the K > N matmul, OR equivalently `N < N_tiles_min × BN`
where `N_tiles_min` is what you need to saturate the GPU. For M4 Max
~80 TG slots at WM*WN=2 SGs per TG → `N_tiles_min ≈ 80`. So at BN=64
this is N < 5120; at BN=32 it's N < 2560.

**Don't generalize "smaller BM is always better"**: for the typical
N >> K case, BM=32 IS the right choice — its lower per-row W-streaming
cost outweighs the parallelism loss because N is already big enough
to fill the GPU. The rule is "use smaller BM only when N alone can't
fill the GPU".

Generalizes to any model: any K > N MLP down_proj (or attention
o_proj when the head_dim×num_heads is smaller than hidden) is a
candidate. Always KPROF the candidate matmuls and check effective
bandwidth = (matrix_bytes / per_call_time) against your device peak.
If you're below 60% of peak on a single matmul, you're TG-starved.
