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
