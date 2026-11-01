# Debugging recipes & late-stage checklists

Symptom-driven debugging table for the most common situations during
optimization, plus a barrier-removal checklist for the last 5% to MLX
parity.

## Symptom → recipe

### "Tokens diverged after change X — how do I localize?"

Verify frozen C, candidate and pinned native source/binary/options/input
identities. Use identical teacher-forced tokens/state, then native sampler
on candidate logits, native head on candidate normalized hidden, and first
upstream primitive replay. Retain sparse first-divergence slices/hashes or
streaming comparisons, not every large tensor indiscriminately.
Follow [the diagnosis ladder](measurement-and-diagnosis.md#1-first-divergence-ladder).
The first differing tensor localizes evidence; it is not automatically
the cause or a compiler defect.

### "Decode is fast, prefill is slow"

Check timing/setup exclusions first, then actual native prefill routes and
hot shapes. MMA/split reducers (C1/C2), attention or recurrence may dominate.
Use E3 only for an actual MoE path. D256 prefill can select generic native
QK/softmax/PV rather than flash; inspect the selector.

### "Prefill is fast, decode is slow"

You probably have not yet applied **qmv4 register tile** (B3), **fused
gate+up+swiglu** (E2), or **parallel argmax** (F1).

### "Tok/s drops every few tokens"

Check memory pressure, queue/copy/allocation and residency clocks before
attribution. Inspect the actual pinned loader; direct pread into Metal
buffers can be native, so zero-copy mmap is not a prerequisite.

### "Performance is bursty"

Separate encode from post-submit GPU delay and host sampling/setup. Apply
A4/A2/A3 only if measured encode/lifetime overhead is the bottleneck.

### "Validated kernel-by-kernel but end-to-end tokens still diverge"

Check `barrier()` / cmdbuf ordering. With concurrent encoder (A5), a
missing barrier produces non-deterministic divergence that often passes
per-kernel tests (which serialize one kernel at a time).

For the specific "first 16 tokens match, then garbage" vs "every token
is garbage" distinction, see `gotchas.md` #26 — the two kinds of
barriers (`threadgroup_barrier` inside a kernel vs
`memoryBarrierWithScope:` between dispatches) are not interchangeable.

### "First N tokens match, then diverge slowly"

Do not assume drift or waive the ID gate. Check native rounded sampler,
identical-input head/hidden and first upstream state, plus dispatch/capacity/
hazard boundaries. Primitive bounds on identical inputs do not automatically
cover an accumulated divergent trajectory. Record unlocalized causes honestly.

Common silent-bug source: KV cache `max_ctx` and SDPA TG-mem `MAX_CTX`
not agreeing, OR both not being ≥ `prompt_len + max_tokens`. The first
N tokens still match because the overrun hasn't happened yet. See
`gotchas.md` #3 and #15.

### "Numbers look great but I can't reproduce them"

Check actual binary/library/sidecar/options, timer boundaries, power/thermal
and guard receipts first. Use a predeclared counterbalance with all samples,
medians/spread/pairs; see `measurement-and-diagnosis.md`. A fixed percentage
does not by itself distinguish a real effect from noise.

## Late-stage barrier-removal checklist

If the actual graph/profile shows independent chains with critical-path
headroom, walk through this checklist. Near-MLX throughput and GPU coverage
alone do not prove concurrency is the remaining cause.

For each changed dependency, validate focused guards/reuse and short production
IDs; require all requested native workload IDs before final promotion. Avoid
a full timing series for every barrier. Preserve native round points in fusion.

### Attention layer

- [ ] `q_proj || k_proj || v_proj` (read same H; disjoint outputs)
- [ ] `q_norm || k_norm` (disjoint inputs and outputs)
- [ ] `rope_q || rope_k` (disjoint inputs and outputs)

### Linear-attention / SSM layer (if present)

- [ ] `in_proj_qkv || in_proj_z || in_proj_b || in_proj_a` (fan-out)
- [ ] `silu_inplace(conv_out) || conv_state_update` (different buffers)
- [ ] `rmsnorm_scale_q || rmsnorm_scale_k || sigmoid(beta) || compute_g`
      (4-way concurrent, all different inputs/outputs)

### MoE layer (BIGGEST WIN ZONE)

- [ ] `moe.gate_proj || moe.up_proj` (read same H; disjoint outputs)
- [ ] `shared.gate || shared.up || shared_expert_gate` (same H, disjoint)
- [ ] **MoE main chain || shared-expert chain** end-to-end (A8) — this
      is usually the last 3–5% to MLX parity. See
      `parallel-chains.md`.

### Across-layer fusions

- [ ] Residual fused into `o_proj` epilogue (B5)
- [ ] Residual fused into `out_proj` epilogue (linear-attn B5)
- [ ] Residual fused into `shared_expert_combine` (B5 variant)

### Elemwise fusions

- [ ] `copy + sigmoid_inplace` → `sigmoid_bf16` (out-of-place)
- [ ] Any "init buffer; then modify in-place" pair → single oop kernel
      (see `gotchas.md` #24)

### Threadgroup-size hygiene

- [ ] Every elemwise / glue kernel uses TG=(32,1,1) not (1,1,1) — see
      `gotchas.md` #17.

## Pattern-spotting for barriers

A dispatch group is **barrier-removable** when it has the property
"shared input(s), disjoint outputs". The fan-out points (post-rmsnorm)
and fan-in points (o_proj + residual, shared_combine_add) are where
you still need barriers.

When in doubt, list every dispatch's buffer reads and writes; group
the ones with no overlap. See `gotchas.md` #22 for the full pattern.

## Critical-path concurrency wins last

When measured critical-path evidence indicates heavy kernels have little
headroom, useful remaining candidates include:

1. Reducing the critical path (fuse dispatches that produce X →
   consume X).
2. Removing barriers between independent chains so they run in
   parallel (see `parallel-chains.md`).

For Qwen3.6-35B-A3B at >65 tok/s, ALL further wins came from category
(2) in that historical stage. This is not a universal ceiling derived from
nominal weight bytes/time. See `gotchas.md` #19 and #59.
