# MLX-specific gotchas

Read this when:
- Your reference is MLX (`mlx_lm/models/<name>.py`).
- Writing `tools/dump_ref.py` for an MLX backend.
- The model is quantized in MLX format.
- A kernel matches PyTorch but fails MLX (or vice versa).

MLX is the preferred reference for this skill because the codebase is
small and easy to read. But it has a few footguns:

1. **No `forward_hook` mechanism**. To dump per-op intermediates you
   must manually walk the forward pass in Python, mirroring the
   reference module's `__call__`. See `starter/tools/dump_ref.py.template`
   for a worked example.

2. **`mx.eval(...)`**. MLX is lazy; without `mx.eval` (or
   `np.array(...)`, which forces evaluation) you'll OOM on big models.
   After each layer in the manual walk, call `mx.eval(x)`.

3. **`scales` / `biases` dtype matches the model's compute dtype**, not
   necessarily `float32`. If you experiment with `mx.quantize(x, ...)`
   on an `f32` array, scales come back `f32`; but a serialized bf16
   model's `scales` are `bf16`. Always inspect the safetensors header
   to confirm.

4. **`mx.argpartition` doesn't sort**. The top-K indices from a router
   come back in implementation-defined order. When comparing top-K
   between C and MLX, compare **sets**, not sequences. Order only
   matters if the downstream op is non-commutative.

5. **`mx.softmax(..., precise=True)`** forces fp32 reduction. Default
   may be lower precision. Match in C with fp32 accumulators.

6. **`cast_predicate`** (`mlx_lm/models/<name>.py`): lists parameters
   that should *not* be cast to the model's compute dtype (e.g.,
   `A_log` in SSMs is sometimes excluded). In the safetensors archive
   the dtype may still be bf16; the cast happens at use-time inside
   the Python code (e.g., `A_log.astype(mx.float32)`). Mirror this in
   your C kernel.

7. **`quant_predicate`**: lists which parameters get which quantization
   settings. May force certain weights to a different bit-width than
   the model default.

8. **`Model.sanitize(weights)`**: applied at load time. May rename keys
   (e.g., `model.*` → `language_model.model.*` for multi-modal wrappers),
   transpose `conv1d.weight` axes, or add `+1.0` to RMSNorm weights.
   When loading directly from safetensors in C, you must replicate the
   effects of `sanitize`. **Read the model's `sanitize` carefully.**

   - **`sanitize` adds `+1.0` to RMSNorm weights**: some `sanitize`
     paths shift all RMSNorm weights by +1 when MTP weights are
     present in the archive (Qwen 3.5 family). The C loader must
     replicate this.
   - **`conv1d.weight` axis order**: HF stores Conv1d weights as
     `[C_out, C_in/groups, kernel]`. MLX `sanitize` may
     `moveaxis(2, 1)` to put the kernel dim in the middle. Confirm
     via the safetensors header shape.
   - **Skipping vision tower / MTP heads**: multi-modal and
     "multi-token-prediction" checkpoints have many unused tensors.
     `sanitize` drops them; the C side may have to walk the archive
     but ignore them. Filter by name prefix (`vision_tower.`, `mtp.`,
     etc.).

9. **`mx.fast.rope`** with `freqs=...` uses an explicit per-half-dim
   frequency table, not the implicit `1 / base^(2k/D)` form. The two
   are equivalent, but the C side needs to precompute the same table:
   `inv_freqs[k] = 1 / base ** (2k / D_rot)`.
