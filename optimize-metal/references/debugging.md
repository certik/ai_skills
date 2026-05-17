# Debugging recipes & late-stage checklists

Symptom-driven debugging table for the most common situations during
optimization, plus a barrier-removal checklist for the last 5% to MLX
parity.

## Symptom → recipe

### "Tokens diverged after change X — how do I localize?"

1. Make sure the C reference (`./src-cpu/`) is still committed and
   produces the same tokens it always did.
2. Add a `--dump <dir>` flag to BOTH `./src-cpu/<BIN_CPU>` and
   `./src-metal/<BIN>` that writes every per-kernel input + output to
   `<dir>/<kernel>_L<layer>.bin` (same .bin format as
   `tools/dump_ref.py`).
3. Run both with the same prompt:
   ```
   ./src-cpu/<BIN_CPU>  --prompt "..." --dump src-cpu/refs
   ./src-metal/<BIN>    --prompt "..." --dump src-metal/refs
   ```
4. Binary-diff per kernel:
   ```
   for f in src-cpu/refs/*.bin; do
       name=$(basename "$f")
       cmp -l "$f" "src-metal/refs/$name" | wc -l
   done | sort -nr | head
   ```
   The first non-zero kernel in forward order is the culprit.

### "Decode is fast, prefill is slow"

You probably have not yet implemented the **prefill GEMM path** (MMA
tile) or the **sorted-gather MoE** for prefill (E3). Apply C1+C2 and
E3.

### "Prefill is fast, decode is slow"

You probably have not yet applied **qmv4 register tile** (B3), **fused
gate+up+swiglu** (E2), or **parallel argmax** (F1).

### "Tok/s drops every few tokens"

Memory pressure / paging. Check `wired_limit` (A7) and verify weights
are zero-copy mmap'd (`gpu_buf_wrap_nocopy`).

### "Performance is bursty"

CPU encode is becoming the bottleneck. Apply 2-deep cmdbuf pipeline (A4)
+ persistent / const param buffers (A2 + A3).

### "Validated kernel-by-kernel but end-to-end tokens still diverge"

Check `barrier()` / cmdbuf ordering. With concurrent encoder (A5), a
missing barrier produces non-deterministic divergence that often passes
per-kernel tests (which serialize one kernel at a time).

For the specific "first 16 tokens match, then garbage" vs "every token
is garbage" distinction, see `gotchas.md` #26 — the two kinds of
barriers (`threadgroup_barrier` inside a kernel vs
`memoryBarrierWithScope:` between dispatches) are not interchangeable.

### "First N tokens match, then diverge slowly"

This is *probably* pure numerical drift. To confirm: dump per-layer
intermediates from both the old commit and the new commit on the SAME
input ids; verify they agree to bf16 tolerance. If yes, the divergence
is purely numerical and acceptable. If no, you have a bug — revert.

Common silent-bug source: KV cache `max_ctx` and SDPA TG-mem `MAX_CTX`
not agreeing, OR both not being ≥ `prompt_len + max_tokens`. The first
N tokens still match because the overrun hasn't happened yet. See
`gotchas.md` #3 and #15.

### "Numbers look great but I can't reproduce them"

Variance is the most likely culprit. See `profiling.md` "Run length and
best-of-N". Below 2% is noise; require best-of-3 or 5-run median
before declaring a win.

## Late-stage barrier-removal checklist

When you're at ~95% of MLX and decode is GPU-bound (`gpu_busy ≈ wall`),
the remaining gap is almost always serialized concurrency. Walk
through this checklist for each layer type in your model.

For each item, verify with a token-id match against the prior commit
(`--max-tokens 16`). For each commit, run 5 times and use median to
distinguish a real 1–2% win from noise.

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

Once individual kernels are ~bandwidth-saturated (e.g., linear_q8 at
25µs for K=N=2880 is reading ~11MB of weights at ~430 GB/s — within 7%
of M4 Max's peak), making the kernel "faster" buys nothing. The only
remaining wins are:

1. Reducing the critical path (fuse dispatches that produce X →
   consume X).
2. Removing barriers between independent chains so they run in
   parallel (see `parallel-chains.md`).

For Qwen3.6-35B-A3B at >65 tok/s, ALL further wins came from category
(2). None from kernel-internal optimization. See `gotchas.md` #19.
