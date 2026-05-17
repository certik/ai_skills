# Pre-flight pitfalls checklist

Read this when:
- Before declaring a kernel "passes the test".
- Before declaring `--validate-forward` green.
- Before declaring end-to-end tokens match.
- A bug is mysterious and the deep references didn't point at it
  obviously — walking this list often surfaces a forgotten gotcha.

Each item below is a quick reminder; the deep explanation lives in the
linked reference. Items marked **CRITICAL** are by far the most common
sources of "kernel-passes-but-tokens-don't-match" surprises.

## Setup / oracle

- **CRITICAL: PyTorch CPU SDPA defaults to FLASH**, not MATH, and FLASH
  gives different bf16 results. Force `SDPBackend.MATH` in oracle dumps
  AND in any end-to-end Python comparison run. → `pytorch-gotchas.md`
- **`-Wall -Wextra` clean**: hidden bugs love to live in `int`/`size_t`
  width mismatches and missing-prototype warnings. Get a clean build
  before declaring a kernel "validated".
- **JSON parsing performance**: define `JSMN_PARENT_LINKS` before
  including `jsmn.h`. Without it, parsing a 250k-entry `tokenizer.json`
  takes ~40 s (O(N²)); with it, ~40 ms (1000× speedup).
- **Centralize JSON, don't re-invent it per caller**: every caller
  doing its own substring-search JSON walker breaks on multi-modal
  configs with duplicate key names. Route everything through
  `utils/json`.

## Kernels

- **CRITICAL: Missing per-element bf16 round-trips**: causes 1–2 ULP
  drift per kernel even when "the math is right". → `precision-and-tolerance.md`
- **W transpose**: HF safetensors typically stores Linear weights as
  `[out_features, in_features]`. So `Y = X @ W.T` becomes
  `Y[m,n] = Σ_k X[m,k] * W[n,k]`. Get this wrong and EVERY linear is off.
- **GQA head grouping**: for grouped-query attention, q head `h` reads
  KV head `h / (Nq/Nkv)`. Easy to off-by-one.
- **Causal mask + sliding window + sinks**: get all three right. Sink
  contributes one extra logit per head with no value (it just adds to
  the denominator of the softmax). → `architectures.md`
- **bf16 round-to-nearest-even**: naive `(x >> 16)` truncation will
  cause systematic small errors. Use round-to-nearest-even
  (`utils/bf16.h`). → `precision-and-tolerance.md`
- **RoPE yarn scaling**: yarn has an extra `mscale` multiplier on top
  of the rotation — easy to miss. Match the reference exactly.
- **RoPE freq formula**: `inv_freqs[k] = 1 / (base ** (2k / D))` for
  `k ∈ [0, D/2)`. With `partial_rotary_factor < 1.0` use
  `D = D_rot = head_dim * partial_rotary_factor`, NOT `head_dim`.

## Loader / safetensors

- **`safetensors.c` `__metadata__` key**: `__metadata__` is 12 chars,
  not 11. The fixed template has this corrected.
- **Skipping vision tower / MTP heads**: multi-modal and
  "multi-token-prediction" checkpoints have many unused arrays
  (tensors). The Python reference's `sanitize` drops them; the C side
  may have to walk the archive but ignore them. Filter by name prefix
  (`vision_tower.`, `mtp.`, etc.). → `mlx-gotchas.md`
- **`sanitize` adds `+1.0` to RMSNorm weights**: some `sanitize` paths
  shift all RMSNorm weights by +1 when MTP weights are present in the
  archive (Qwen 3.5 family). The C loader must replicate this.
- **`conv1d.weight` axis order**: HF stores Conv1d weights as
  `[C_out, C_in/groups, kernel]`. MLX `sanitize` may
  `moveaxis(2, 1)` to put the kernel dim in the middle. Confirm via
  the safetensors header shape.

## Tokenizer / chat template

- **Tokenizer special tokens**: HF `tokenizer.json` has special tokens
  in `added_tokens` — make sure `build_tokenizer.c` includes them
  (the starter does this automatically from the JSON).
- **Tokenizer `MAGIC` length**: the C reader expects 12 bytes; if the
  Python writer writes only 11 you'll silently load garbage. The
  fixed template now writes 12.
- **Stop token**: pick the right stop token from the reference's
  generation config. For gpt-oss it's `<|return|>` (id 200002), not
  `<|endoftext|>`. For Qwen it's `<|im_end|>` (or `<|endoftext|>`,
  depending on the chat template). → `pytorch-gotchas.md`
- **Chat template token IDs**: the chat-template scaffolding is part
  of the *input* to the model. Before debugging anything, dump the
  prompt IDs from both the Python reference and your C builder and
  assert byte-for-byte equality. Many problems are actually
  tokenization mismatches.
- **`vocab_size > tokenizer vocab`**: many models pad the embedding to
  a power of 2 or a multiple of 64/128. The extra rows are unused;
  the C `argmax` over the full `vocab_size` is correct as long as you
  never emit a row index past the tokenizer's known set during decode.
- **Pre-tokenizer regex** varies per tokenizer family. The starter
  `tokenizer.c` is tuned for o200k_harmony; for llama / qwen / mistral
  the alternations are different (notably digit handling and how
  `\p{L}\p{M}+` is split). Swap `pretokenize()` per
  `tokenizer.json`'s `pre_tokenizer.pattern`. → `tokenizer.md`

## Forward orchestration

- **SSM state across forward calls**: prefill leaves the state with
  prompt content; the AR decode loop continues from there. Don't
  reset state between prefill and decode. → `architectures.md`
- **Final norm + lm_head**: only the last position's logits are
  needed for sampling. Apply `model.norm` and `lm_head` to just
  `x[-1, :]` to save `(L-1)/L` of the work. The starter
  `main.c.template` does this. (Bidirectional/diffusion models don't
  get this shortcut.)

## Sampler-specific (Fast-dLLM-style block diffusion)

- **Within-block vs full-sequence logit shift**: the shift is
  **always relative to the actual logits array's row indexing**, not
  relative to absolute sequence position. → `samplers.md`
- **`shift_rows_down1` must iterate backwards** so you don't clobber
  `row[i-1]` before reading it. → `samplers.md`
- **Forgetting `n_changed == 0 → break`** in confidence-threshold
  samplers. → `samplers.md`
- **Fresh cache (`cache_reset`) at the start of each block** in
  Fast-dLLM. → `samplers.md`
- **`cache_position = None` vs `cache.offset`**: use a sentinel
  `eff_pos = (cache_position >= 0) ? cache_position : cache->offset`.
  → `samplers.md`
- **Per-layer cache offset must advance in lockstep** (not per-layer).
  → `samplers.md`
