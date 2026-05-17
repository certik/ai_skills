# Kernel patterns

Recurring shapes that come up during a 1:1 port but aren't part of
every kernel. Read the section that matches the kernel you're porting.

## Stateful kernels (conv1d / SSM)

Some kernels both read input AND update a carried `state` buffer
(e.g. `conv1d_depthwise_causal_bf16`). A naive parallel kernel hits a
write-after-read hazard across threads.

**Cleanest naive port: split into "compute on GPU" + "update state on host".**

1. Dispatch the read-only compute kernel.
2. `gpu_cmdbuf_commit_wait`.
3. `memcpy` / `memmove` on `gpu_buf_contents(state)` to advance the
   state in place.
4. Open a fresh cmdbuf and continue.

`optimize-metal` can later replace the host shift with a dedicated
kernel + barrier.

## Block-diffusion KV cache (Fast-dLLM-style)

Diffusion LMs with block-local KV caching (e.g. Fast-dLLM's
`DualKVCache` — append for prefetch, slice-replace inside the active
block) port very cleanly with the same compute-on-GPU /
update-state-on-host split as conv1d:

1. Allocate one Metal buffer per layer for K and per layer for V via
   `gpu_buf_new(ctx, n_layers * 2 * max_L * Nkv * D * sizeof(bfloat))`,
   or one buffer per (layer, K|V) — either works.
2. For each layer in `forward_cached()`:
   - **Phase A cmdbuf**: rmsnorm → q/k/v projections → RoPE on Q and
     fresh K. `commit_wait`.
   - Host-side `memcpy` of fresh K and V into the per-layer cache
     buffer at the right offset (append mode: `cache->offset`;
     slice-replace: `eff_pos`). Mirrors `kv_cache_write_layer` from
     `src-cpu/cache.c`.
   - **Phase B cmdbuf**: SDPA against the *full* `Lk` slice of the
     cache → o_proj → residual → rmsnorm → MLP → residual.
     `commit_wait`.
3. Bookkeeping (`cache->offset += Lq` for append; nothing for
   replace) happens in host C code in the outer loop, *not* via a
   kernel.

Cost: `2 * n_layers` commit_waits per block-local forward. Slow
(GPU under-utilised), but trivially correct. `optimize-metal` will
fuse this into one cmdbuf per forward with a GPU-side "write to
cache" kernel and an SDPA that reads K/V directly without the host
round-trip.

**`Lk` must be captured before the per-layer loop**, not re-read
each iteration — once you start writing into the cache, `offset`
advances, but every layer's SDPA needs the same `Lk` (computed once
as `replace ? offset : (eff_pos + Lq)`). Same gotcha as `src-cpu`'s
`forward_cached`; just mirror its `Lk` capture.

## Host-side scalar broadcasts

For one-off ops that don't decompose cleanly into a kernel — e.g.
"sigmoid of a per-row scalar gate × a per-row vector", or "add two
host-side intermediates" — just commit_wait, do them on the host with
`gpu_buf_contents(...)`, and open a fresh cmdbuf. Naive is fine; the
optimisation skill will reshape these as broadcast kernels.

## Diffusion / non-AR samplers stay on host

For diffusion LMs (no per-token loop; one forward fills many mask
positions at once), the **entire sampler** can stay on CPU in the
naive port — only the model forward runs on GPU:

- `forward()` ends with `commit_wait` and produces `logits_buf` in
  shared (unified) memory.
- The C-reference sampler logic — `softmax_fp32`, `argmax_bf16`,
  `neg_entropy_fp32`, topk-by-confidence, confidence-threshold
  fill, "shift logits left by 1" — all reads `gpu_buf_contents(
  logits_buf)` and runs in plain C99 on the host.
- Same code as `src-cpu/main.c`'s sampler; no need to port these to
  MSL.

Argmax over the vocab (152k for Qwen2.5, 200k for GPT-OSS) takes
about a millisecond on host — well under the model forward time. If
the sampler turns out to be a hot spot later, `optimize-metal` will
move it to GPU.

## Mask-count / per-step trajectory drift is OK

For confidence-threshold samplers (Fast-dLLM-style), GPU and CPU may
report slightly different per-step `n_changed` trajectories
(e.g. CPU `15 → 9 → 5 → 4 → 3 → 2 → 1 → 0` vs GPU `15 → 10 → 5 → 4
→ 3 → 2 → 1 → 0`) because fp32 reductions reorder differently on
the GPU. The final emitted tokens still match exactly, because the
threshold-plus-force-fill logic is robust to small softmax noise.

This is the strongest single argument for the skill's "token
equivalence, not per-step trace equivalence" acceptance criterion.
If you've matched the C reference's tokens but the printed
trajectory differs slightly, you're done — don't chase the
trajectory.

## MoE down_proj reshape trick

The per-(token, expert) inner loop in `src-cpu`'s MoE down_proj path
collapses cleanly into a single `linear_q8_gather` dispatch by
reshaping `(L, K_top)` → `(L' = L*K_top, K_top' = 1)`. The same gather
kernel handles both `gate_proj` / `up_proj` (real `K_top`) and
`down_proj` (`K_top' = 1`). One kernel, three call sites — no need for
a separate "per-token-per-expert" kernel in the naive port.

## Two MoE paths — DO NOT introduce here

The fully-optimized `src-metal/` you may have seen in the wild has
both a "qmv4" (decode) path and a "sorted-gather" (prefill) path for
MoE. **The naive port does not have those.** Use the same kernel for
`Lq = 1` and `Lq > 1`. `optimize-metal` will add the prefill fast
path later. Resist all temptation.

## Per-kernel correctness via `--dump`

A pattern that scales:

1. Add a `--dump <dir>` flag to the C reference. When `forward()`
   runs, it writes every intermediate it produces to
   `<dir>/<kernel>_<layer>.bin` in the same `.bin` format as
   `tools/dump_ref.py`.

2. Each Metal kernel test:
   - Loads its inputs from the dump dir.
   - Runs the Metal kernel.
   - Loads the expected output from the dump dir.
   - Compares within tolerance.

This means every Metal kernel test runs in milliseconds (no full
forward pass needed), and `src-cpu` is the single source of truth for
"what is the expected output of this kernel".

These per-kernel tests are *optional* — skip them if end-to-end token
match passes and your iteration loop is < 2 min. Worth adding only
when you need to bisect a divergence.

## Tolerance

Same as for C ↔ Python (~1e-2 abs in bf16). Metal GPU reductions may
reorder differently from CPU, so don't expect bit-exact. As long as
max-abs-error stays in tolerance you're good.

## When tokens match but only for a few steps

If end-to-end matches for the first N tokens then diverges: that's
pure numerical drift, fine for this skill. The pieces are correct.

Acceptance is "matches the C reference for the validation prompt".
Pick a validation prompt + max-tokens combo where they DO match
(a short prompt with 16 tokens usually works), and call it done.

## Suggested commit log

One commit per kernel, then one for stitching. Example shape (kernel
names will depend on your model — these are from a GPT-OSS-style MoE):

```
src-metal: metal shim + smoke test
src-metal: embed_gather_bf16 metal port + test
src-metal: rmsnorm_bf16 metal port + test
src-metal: linear_bf16 metal port + test
src-metal: rope_bf16 metal port + test
src-metal: sdpa_bf16 metal port + test
src-metal: topk_softmax_bf16 metal port + test
src-metal: mxfp4_linear_gather_bf16 metal port + test
src-metal: swiglu_bf16 metal port + test
src-metal: expert_mix_bf16 metal port + test
src-metal: residual_add_bf16 metal port + test
src-metal: argmax_bf16 metal port + test
src-metal: end-to-end metal forward + AR decode (N/N tokens match src-cpu)
```

For a non-MoE dense model, drop the `topk_softmax`, `*_gather`, and
`expert_mix` lines.
