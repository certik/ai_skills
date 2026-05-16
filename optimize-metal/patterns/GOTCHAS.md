Concrete bug-and-fix war stories from optimizing src-metal/ ports.
Each one cost at least one debug iteration; consult BEFORE chasing
the same symptoms again.

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

For decode (Lq=1) on a quantized MoE LLM, this is the empirical
order of biggest wins from a naive port:

  1. Parallelize ANY single-thread-per-row reduction:
     argmax over vocab (the king of decode bottlenecks at >50k vocab),
     softmax_topk over N_experts, rmsnorm rows.
  2. SIMD-group-per-output for ALL quantized GEMV (B1).
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

