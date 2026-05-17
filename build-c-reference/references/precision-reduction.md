# Reduced precision (Quantization) formats

Read this when:
- The reference uses MXFP4 (openai-style, gpt-oss family).
- The reference uses MLX-style affine 4-bit or 8-bit precision reduction.
- You're sizing the dequant kernel for a MoE `gate_up_proj` / `down_proj`.

Other formats (GGUF Q4_K, AWQ, GPTQ, ...) follow the same pattern: read
the reference's loader, replicate the dequant inline with the matmul,
do not materialize a dequantized weight buffer.

## MXFP4 (gpt-oss family)

Openai-style MXFP4: e2m1 packed 2/byte + e8m0 per-32 block scale.

```c
static const float fp4_lut[16] = {
    0,  0.5,  1,  1.5,  2,  3,  4,  6,
   -0,-0.5, -1, -1.5, -2, -3, -4, -6
};
static inline float e8m0_scale(uint8_t b) {
    return b == 0xFF ? 0.0f : ldexpf(1.0f, (int)b - 127);
}
// per row: walk K_BLOCKS = K/32 blocks, decode 32 floats × scale,
// accumulate dot product with X.
```

The format details (group size, packing, exponent bias) are all model-
specific — check the reference's loader.

## MLX-style affine — 4-bit and 8-bit

MLX uses a different family of precision-reduction formats. For a
Linear of shape `(N, K)` with affine reduced precision at `B` bits and
`group_size` `G`, the safetensors stores **three** arrays:

```
<base>.weight  : uint32  [N, K * B / 32]   -- packs (32 / B) elements per uint32
<base>.scales  : <model dtype>  [N, K / G]
<base>.biases  : <model dtype>  [N, K / G]
```

For `B = 8`, `G = 64`:
- `weight` is `[N, K / 4]` uint32 — each `uint32` packs 4 little-endian
  `uint8`s (low byte = first element).
- `scales`, `biases` are `[N, K / 64]` in the model's native dtype
  (`bfloat16` for bf16 models, `float32` for fp32 models — **not always
  fp32**; check the actual safetensors!).

Dequant for element `[n, k]`:

```c
uint8_t  q = (W[n, k/4] >> (8 * (k % 4))) & 0xff;
float    x = (float)q * scale[n, k/64] + bias[n, k/64];
```

For `B = 4`, `G = 64`: same layout, but `weight` is `[N, K/8]` uint32
packing 8 nibbles per uint32 (low nibble = first element). The 4-bit
unsigned value `q ∈ [0, 15]` then dequants identically.

### Quirks worth remembering

- Even small linears (e.g., outputs of 1 or 32) use the reduced-precision
  representation in MLX models. Don't assume "1-dim outputs" means f32.
- MLX's `mx.quantized_matmul` and `mx.gather_qmm` dequant inline; never
  materialize a dequantized weight buffer in the C reference.
- `Embedding.weight` is in the reduced-precision form too in MLX models
  — use the same dequant in your `embed_gather` kernel.
- The MLX inference code may force certain weights to a specific
  reduced-precision format (`quant_predicate` in
  `mlx_lm/models/<name>.py`). Read it.
