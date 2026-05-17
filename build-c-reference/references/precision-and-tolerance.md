# Numerical precision and tolerance

Read this when:
- Writing or debugging any kernel (especially `rmsnorm`, `silu/swiglu`,
  `rope`, `sdpa`).
- A per-kernel test fails by 1–4 ULPs even though the math looks right.
- The `--validate-forward` argmax check disagrees with the Python oracle.
- You're choosing the per-kernel tolerance for your test harness.
- Deciding whether to enable `-ffast-math`.

## Contents

1. Per-element bf16 rounding patterns ("hidden casts") — the #1 cause of
   small mismatches.
2. Tolerance settings per kernel and per stage.
3. End-to-end logit drift — why argmax stability, not raw equality, is the
   acceptance metric.
4. `-ffast-math` warning.
5. bf16 round-to-nearest-even.

## Per-element bf16 rounding patterns ("hidden casts")

This is the **#1 cause of 1–4 ULP per-element kernel mismatches**. When
the reference is written in PyTorch / MLX / etc., each `tensor op tensor`
in the model dtype (bf16) **rounds the intermediate to bf16** even when
the surrounding math looks like one fused fp32 expression. A naive C
kernel that runs the whole expression in fp32 and rounds only at the
final store will be 1–2 ULP off per element.

The canonical examples in a vanilla decoder block:

| Reference code                        | What it really does (bf16 storage)               | Naive C bug                                                    |
|---------------------------------------|--------------------------------------------------|----------------------------------------------------------------|
| `w * x_normalized.to(input_dtype)`    | cast `x_normalized` to bf16, then `w * bf16`     | `y = w * x_normalized` (no cast) ⇒ 1 ULP off                   |
| `F.silu(gate) * up`                   | `silu(g)` in bf16, then `bf16 * bf16`            | `y = silu(g_f32) * u` ⇒ 1 ULP off                              |
| `cos.to(query.dtype); q*cos + ...`    | cast cos/sin to bf16 first, then per-mul rounds  | use fp32 cos/sin ⇒ 1–2 ULP off                                 |
| `(q*cos) + (rotate_half(q)*sin)`      | each multiply rounds to bf16, then add rounds    | `f32_to_bf16(a*c - b*s)` fuses ⇒ 1–2 ULP off                   |
| `softmax(logits, dtype=fp32).to(bf16)`| softmax in fp32, then cast back to bf16 before V | run softmax fp32 and use fp32 probs in V matmul ⇒ small drift  |

The fix is mechanical: wherever the reference would have a `.to(dtype)`
or a `bf16 op bf16` boundary, insert `bf16_to_f32(f32_to_bf16(v))` in
the C kernel. Examples:

```c
// rmsnorm: weight multiply happens in input dtype (bf16).
float v   = bf16_to_f32(x[d]) * rrms;
float vb  = bf16_to_f32(f32_to_bf16(v));        // <- mirror .to(input_dtype)
float out = bf16_to_f32(W[d]) * vb;
y[d] = f32_to_bf16(out);

// silu_mul: silu(g) rounds to bf16 before * up.
float g = bf16_to_f32(gate[i]);
float u = bf16_to_f32(up[i]);
float s = bf16_to_f32(f32_to_bf16(siluf_(g))); // <- mirror F.silu(g) returning bf16
out[i]  = f32_to_bf16(s * u);

// RoPE: cos/sin cast to bf16; each per-mul rounds; then add rounds.
float c  = bf16_to_f32(f32_to_bf16(cosf(pos * inv_freqs[k])));
float s  = bf16_to_f32(f32_to_bf16(sinf(pos * inv_freqs[k])));
float ac = bf16_to_f32(f32_to_bf16(a * c));   // bf16 mul
float bs = bf16_to_f32(f32_to_bf16(b * s));   // bf16 mul
row[k]   = f32_to_bf16(ac - bs);              // bf16 add
```

Whether you need this for *every* kernel or only some depends on how
strict your tolerance is. For per-kernel oracle matching at `max|d| ≈ 0`
(exact bit-equivalent), insert all of them. If you're willing to live
with 1–2 ULP per element and let drift accumulate, skip them — but
expect the global logit drift to be 2–3× larger after L layers.

## Other sources of numerical drift

bf16 accumulation order matters less than you'd think because most
reductions are linear and bf16 only has 8 mantissa bits. But:

- For `gate = sigmoid(alpha * gate_raw) * gate_raw` vs the SiLU/SwiGLU
  used by the reference — these can produce different bit patterns even
  algebraically equivalent. Match the reference's actual code.
- For RMSNorm, the order is `sum(x^2)/D → sqrt → rsqrt → multiply` —
  reference exact form may matter.
- For RoPE with yarn / mscale, the multiplication order between
  `cos/sin`, `mscale`, and the input affects the last mantissa bit.
  Match the reference.

## Tolerance setting for bf16

bf16 has 7 mantissa bits + an implicit leading 1, so its **relative
precision** is roughly `2^-7 ≈ 7.8e-3`. Absolute error scales with
magnitude:

| `|x|` range       | bf16 ULP (approx)  |
|-------------------|--------------------|
| `[0.125, 0.25)`   | `0.001`            |
| `[0.5, 1)`        | `0.004`            |
| `[1, 2)`          | `0.008`            |
| `[4, 8)`          | `0.03`             |
| `[16, 32)`        | `0.125`            |
| `[64, 128)`       | `0.5`              |

A flat absolute tolerance of `1e-2` is fine for outputs whose magnitude
stays under ~1, but it'll spuriously fail on RMSNorm outputs in the
tens. Use `tol_eff = abs_tol + rel_tol * |want|` in your tester (the
`test_kernels.c.template` does this). Recommended `tol`:

- `0.005` for elementwise ops on inputs `|x| <= 1` (one bf16 ULP).
- `0.02 – 0.05` after a single matmul with K in the thousands.
- `0.05 – 0.1` after several composed ops (e.g., SDPA + gate + o_proj).
- `1e-5` for fp32 paths.

If a test fails by a small margin and the values are large, suspect
the *tolerance*, not the kernel.

## End-to-end logit drift (the "1 ULP per kernel × N layers" rule)

Even with **every** per-kernel test passing at `max|d| ≤ 1` bf16 ULP,
the final logits after `N` layers can drift by `O(1)`. On Dream 7B
(28 layers, kernels at `max|d| ≤ 0.008`) the global raw-logit `max|d|`
is `~0.8`. This is **expected**: per-element error compounds through
each matmul-and-add.

The correct E2E acceptance metric is **argmax stability**, not raw
logit equality: for each position `p`, does the C `argmax(logits[p,:])`
equal the Python reference's argmax at the same position? On Dream 7B
all 25/25 positions match despite raw-logit drift, because the winning
logit's margin over the runner-up is far larger than the drift.

For the `--validate-forward` test (Phase 5), aim for **100% argmax
positions match**. Do not gate on raw-logit `max|d|` — it's
uninformative and will fail at a tolerance that allows real bugs to
slip through. If even one argmax mismatches, dig: most likely one
kernel has a bf16 round-trip missing (see "Per-element bf16 rounding
patterns" above).

## `-ffast-math` warning

Adding `-ffast-math` may speed up the C run by 1.3–2× but it allows the
compiler to reassociate fp ops and assume no NaN/Inf. Test that
end-to-end token output still matches before enabling. If even one token
diverges, **drop the flag**.

## bf16 round-to-nearest-even

The naive `(x >> 16)` truncation of fp32 → bf16 causes systematic small
errors (always rounds toward zero). Use round-to-nearest-even — see
`utils/bf16.h`. Both directions:

- `bf16_to_f32(b)` is exact (zero-extend the mantissa).
- `f32_to_bf16(f)` rounds the discarded low 16 bits to nearest-even.

This matters because nearly every kernel does at least one `f32 → bf16`
store and the error from "always round down" accumulates over a forward
pass.
