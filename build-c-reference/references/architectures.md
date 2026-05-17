# Modern-architecture patterns

Read this when:
- Doing Phase 1 reconnaissance on a non-vanilla architecture.
- Implementing kernels for attention variants (GQA, sinks, sliding
  window, partial RoPE, q/k_norm, output gate).
- The reference uses MoE, a shared expert, or `norm_topk_prob`.
- The reference is a hybrid linear-attention / SSM model (Mamba,
  GatedDeltaNet, RWKV, ...).
- The reference is bidirectional (discrete-diffusion / masked LM).
- Deciding how to draw kernel boundaries vs the reference.

## Contents

1. Choosing kernel boundaries
2. Order of operations vs reference
3. Decoder-block variants you'll encounter
4. MoE specifics
5. Sliding-window attention
6. Hybrid linear-attention / SSM models
7. Bidirectional attention (for masked LMs)
8. Tied/untied lm_head; vocab padding; final norm + lm_head shortcut

## Choosing kernel boundaries

The rule is **structure kernels EXACTLY as the reference**. That means: if
the reference does

```python
gate, up = self.gate_up_proj(x)       # one Linear with 2*intermediate output
mid = self.swiglu(gate, up)
```

then you have **two** kernels: `gate_up_linear` and `swiglu`. Even if
fusing them would be faster, that's a job for `optimize-metal`.

If, however, the reference's quantized linear is fundamentally a
"gather expert weights + dequant + matmul" combined op (as in MLX's
`mxfp4_qmm`), follow that: keep dequant inline with the matmul, do not
materialize a bf16 weight buffer.

Reuse the same kernel for different call sites of the same op (e.g., one
`linear_bf16` used for q_proj / k_proj / v_proj / o_proj / router /
lm_head). One C function per **distinct** math op, not per call site.

## Order of operations vs reference

Maintain reference order even when it's "wasteful":

- RoPE on Q and RoPE on K can run in either order in principle, but the
  reference picks one — do the same.
- KV cache writes happen in a specific position relative to RoPE — match.
- Residual additions happen at a specific spot — match.

## Decoder-block variants you'll encounter

The basic transformer "decoder block" (RMSNorm → QKV linear → RoPE →
SDPA → o_proj → residual → RMSNorm → SwiGLU → residual) is increasingly
just the *starting* template. Newer models add:

- **Grouped-query attention (GQA)**: `num_kv_heads << num_attention_heads`.
  In the C reference, just one SDPA kernel that maps q-head `h` to kv-head
  `h / (Nq / Nkv)`. Easy to off-by-one — sanity-check with `Nq == Nkv` first.
- **Attention output gate** (Qwen 3.5, gpt-oss-mini, ...): `q_proj`
  outputs `2 * num_heads * head_dim`; half is Q, half is a per-head
  output gate. After SDPA: `out = sigmoid(gate) * out` before `o_proj`.
- **Partial RoPE** (`partial_rotary_factor < 1.0`): rotate only the
  first `D_rot = head_dim * partial_rotary_factor` dims of each head;
  leave the rest untouched. Common in long-context models.
- **Per-head q_norm / k_norm**: Qwen-style RMSNorm applied to each
  attention head independently after the projections. Reuse the
  standard `rmsnorm` kernel with `M = L * H, D = head_dim`.
- **Sliding-window attention** on some layers: pass a `window` parameter
  to SDPA; clamp the lower bound on the key index per query. (More
  below.)
- **Attention sinks**: an extra per-head logit that contributes only to
  the softmax denominator (no V). Pass a `sinks` pointer to SDPA. Get
  all three of {causal mask, sliding window, sinks} right together.
- **MoE with shared expert**: in addition to top-K experts, every token
  also passes through a shared expert and (optionally) a scalar
  shared-expert gate `sigmoid(gate1) * shared_out`. Total
  `y = mix(experts) + shared_gate * shared_out`.
- **`norm_topk_prob`**: some models renormalize the top-K router scores
  to sum to 1 after argpartition; others don't. Check the reference.
- **Tied vs untied lm_head**: `tie_word_embeddings: true` means
  `logits = embed_tokens.as_linear(x)`. Otherwise it's a separate
  `lm_head` weight.

### RoPE flavors and gotchas

- **RoPE freq formula**: `inv_freqs[k] = 1 / (base ** (2k / D))` for
  `k ∈ [0, D/2)`. With `partial_rotary_factor < 1.0` use
  `D = D_rot = head_dim * partial_rotary_factor`, NOT `head_dim`.
- **RoPE yarn scaling**: yarn has an extra `mscale` multiplier on top
  of the rotation — easy to miss. Match the reference exactly.

### W transpose for Linear

HF safetensors typically stores Linear weights as
`[out_features, in_features]`. So `Y = X @ W.T` becomes
`Y[m,n] = Σ_k X[m,k] * W[n,k]`. Get this wrong and EVERY linear is off.

## MoE specifics

Naive MoE for the C reference: just loop over `K_TOP` selected experts
per token, no bucketing. It will be slow (~`32x` slower than dense),
but that's fine. The Metal port (and especially `optimize-metal`) will
add the sorted-gather grouped-MoE fast path.

## Sliding-window attention

Even-numbered layers use a sliding window of e.g. 128 tokens; odd layers
use full attention. In the C reference, this is just
`win_lo = max(0, lq_abs - W + 1)` inside the SDPA kernel. No need for a
separate "ring KV cache" — keep the full KV cache for simplicity. The
Metal port can optimize this.

## Hybrid linear-attention / SSM models

Many recent small-but-capable models (Mamba, Jamba, Falcon-Mamba,
Qwen 3.5, RWKV, ...) interleave standard softmax attention with one of
several "linear attention" variants:

- **Mamba / Mamba-2**: selective state-space model with `A`, `B`, `C`,
  `Δ` and a state matrix `h ∈ R[hidden, d_state]` per layer.
- **GatedDeltaNet** (Qwen 3.5): a delta-rule recurrence with state
  `S ∈ R[Hv, Dv, Dk]`; per step `S ← S * g + k ⊗ ((v - S k) * β)`,
  output `y = S q`. Plus an in-projection conv1d, q/k normalize-and-scale,
  and a gated RMSNorm.
- **RWKV**: time-mix + channel-mix with a carried `wkv` state.

What they have in common (and what the C reference needs to handle):

1. **Per-layer state buffers persisted across forward() calls**:
   - depthwise conv1d state (`(kernel-1) * conv_dim` per layer)
   - SSM state (`O(Hv * Dv * Dk)` or `O(hidden * d_state)` per layer)
   Allocate once; reset to zero between independent generations.
2. **A causal recurrence per token** that's awkward to vectorize. Just
   loop over `Lq` tokens in C; the optimized GPU port will rewrite this.
3. **A "compute_g" or similar nonlinearity** wrapping `A_log`, `dt_bias`,
   `softplus`, etc. Promote to fp32 inside this kernel; the reference
   usually does.
4. **Output normalization gated by an input projection** (e.g.,
   `silu(z) * rms_norm(y, weight)`): two kernels, not fused.
5. **SSM state must persist between prefill and decode**. Prefill leaves
   the state with prompt content; the AR decode loop continues from
   there. Don't reset state between prefill and decode.

## Bidirectional attention (for masked LMs)

Discrete-diffusion and masked LMs (Dream, LLaDA, MDLM, ...) use BERT-
style bidirectional attention. SDPA collapses to `softmax(QK/√D) @ V`
with no mask, no sinks, no window. The whole "AR decode loop" goes away
— see `references/samplers.md` for the diffusion sampler driver.

Forward kernels themselves are identical to a vanilla decoder (just
`is_causal=False` in SDPA). What changes is the **outer driver loop**
in `main.c`.

## Tied lm_head, vocab padding, final norm shortcut

- **Tied lm_head**: if `tie_word_embeddings: true` in `config.json`, you
  don't load a separate `lm_head` weight — use `embed_tokens.weight`
  with the same dequant as the embedding kernel.
- **`vocab_size > tokenizer vocab`**: many models pad the embedding to
  a power of 2 or a multiple of 64/128. The extra rows are unused; the
  C `argmax` over the full `vocab_size` is correct as long as you never
  emit a row index past the tokenizer's known set during decode.
- **Final norm + lm_head shortcut**: only the last position's logits are
  needed for sampling. Apply `model.norm` and `lm_head` to just
  `x[-1, :]` to save `(L-1)/L` of the work. The starter
  `main.c.template` does this. (For bidirectional / diffusion models you
  need the full sequence's logits, so this shortcut doesn't apply.)
