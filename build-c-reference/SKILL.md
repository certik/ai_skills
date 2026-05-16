---
name: build-c-reference
description: >
  Build a pure-C, single-threaded CPU reference implementation of an LLM
  inference engine, starting from a working Python reference (MLX preferred,
  else PyTorch/HF transformers). The C reference is intentionally simple
  (no SIMD, no threads), uses the model's native precision (bf16 / fp8 /
  mxfp4 / etc.) with fp32 accumulators, mirrors the reference's kernels
  1:1 (no fusion), and is validated kernel-by-kernel against the Python
  oracle. The output is a standalone `./src-cpu/` directory that loads
  HF safetensors directly, includes its own BPE tokenizer + chat template,
  and produces the same generated tokens as the reference for a fixed
  prompt under greedy decoding. The output is meant to be the starting
  point for `port-c-to-metal` (and later `optimize-metal`).
  Triggers: build c reference, c reference, src-cpu, port to c, pure-c
  port, c inference reference, c llm port, cpu reference, slow reference.
---

# build-c-reference — pure-C CPU reference port of an LLM

Take a working Python reference of an LLM (MLX, HF transformers, PyTorch,
etc.) and a HF-safetensors weight directory, and produce a standalone
`./src-cpu/` that runs the same model in plain C99 on a single CPU core,
with **bit-equivalent-or-very-close** output for the same prompt under
greedy decoding.

The output is **not** trying to be fast. It is the reference and the spec.
Speed comes later (see `port-c-to-metal` and `optimize-metal`).

## When to use

Use this skill when:

- You have a working Python LLM inference (MLX/PyTorch/HF) and want a
  pure-C reference you can later port to any GPU / NPU / TPU platform.
- You want a simple, debuggable, single-threaded C99 implementation that
  matches the reference numerically.
- You need a "spec" of the model that's small enough to read end-to-end.

**Do not** use this skill when:

- You don't have a working Python reference. The skill requires one as
  the numerical oracle. Stop and report this to the user.
- You need a fast CPU runtime — use a vendor library (llama.cpp, ggml,
  MLX) instead.

## Pipeline

```
build-c-reference  →  port-c-to-metal  →  optimize-metal
   (this skill)        (Metal: 1:1)         (Metal: optimized)
```

This skill produces the C reference. The next two skills take it to a
high-performance Apple-GPU Metal implementation. The C reference is also
the starting point for ports to CUDA, ROCm, TPU, etc. (each its own
future skill).

## Prerequisites

- A working **Python reference implementation** of the target model.
  MLX-LM preferred (smaller surface, fewer dependencies). PyTorch / HF
  transformers acceptable.
- A **HF safetensors weight directory** locally (the `model.safetensors`
  or sharded `model-*-of-*.safetensors` + `config.json` + `tokenizer.json`).
- A **C99 compiler** (clang or gcc). `-O2` is the default; `-ffast-math`
  may be used cautiously **only if** it doesn't change per-kernel outputs
  beyond tolerance.
- Python deps to run the reference + dump oracle tensors (numpy, mlx /
  torch / transformers as required by the reference).

## Inputs to gather

Ask the user for (if not already provided / discoverable):

1. **Repo path** containing the Python reference (default: the current
   working directory).
2. **Model directory** (HF safetensors layout); used both by the Python
   oracle and the C binary at runtime.
3. **Binary name** for the resulting C executable. Choose a
   model-specific name (e.g., `llama3-cpu`, `qwen2-cpu`) so multiple
   model ports can coexist in the same workspace, or a generic name
   (e.g., `llm-cpu`, `infer-cpu`) if you prefer. Avoid naming it after
   the reference model (`gpt-oss`, `llama3`) — keep it project-/binary-
   specific.
4. **Sample prompt** for end-to-end validation (default:
   `"Hello, who are you?"` with greedy decoding, max 16 tokens).
5. **System / chat-template metadata** if not obvious from the reference
   (e.g., harmony reasoning level for gpt-oss, llama3 instruct format
   for llama3, etc.).
6. **Output directory** (default: `./src-cpu/` in the repo root).

## Output layout

```
./src-cpu/
├── main.c             # CLI; reads config.json; calls forward() per token
├── kernels.h          # one C function per distinct math op
├── kernels.c          # naive scalar implementations
├── safetensors.c      # HF safetensors mmap loader (drop-in)
├── safetensors.h
├── tokenizer.c        # BPE tokenizer (drop-in)
├── tokenizer.h
├── tokenizer.bin      # built by tools/build_tokenizer.py
├── <chattmpl>.c       # chat-template builder (e.g., harmony.c for gpt-oss)
├── <chattmpl>.h
├── bf16.h             # bf16 ↔ fp32 helpers (drop-in)
├── Makefile
├── refs/              # oracle dumps from Python (per-kernel inputs/outputs)
│   ├── input_ids.bin
│   ├── x_after_embed.bin
│   ├── x_after_rmsnorm_0.bin
│   ├── ...
└── tests/
    ├── load_ref.h         # .bin loader + bf16-aware err_check
    └── test_kernels.c     # one test fn per kernel; CLI-selectable
```

Plus:
```
./tools/
├── build_tokenizer.py  # tokenizer.json → tokenizer.bin (drop-in)
└── dump_ref.py         # runs Python reference, dumps refs/*.bin
```

## Starter files (in this skill's `starter/`)

Drop-in (copy verbatim — model-agnostic):

- `safetensors.c` / `safetensors.h` — HF safetensors mmap loader. Handles
  single-file and sharded layouts, BF16/F16/F32/Uxx/Ixx dtypes.
- `tokenizer.c` / `tokenizer.h` — BPE tokenizer engine. Reads
  `tokenizer.bin`. The BPE merging algorithm is generic, but the **pre-
  tokenizer regex is tuned for o200k_harmony** (the GPT-4o-class
  tokenizer used by gpt-oss). For other tokenizers (llama, mistral,
  qwen, ...) you may need to swap the pre-tokenizer rules in
  `pretokenize()`. The special-token list is fully data-driven from
  the `.bin`.
- `tools/build_tokenizer.py` — converts HF `tokenizer.json` →
  `tokenizer.bin`. **NB**: contains a hard-coded `HARMONY_SPECIALS` list
  for gpt-oss. Replace this with your model's special tokens (usually
  derivable from `added_tokens` in the HF tokenizer.json).
- `bf16.h` — `bf16_to_f32` / `f32_to_bf16` helpers.

Templates (read, then customize):

- `main.c.template` — CLI skeleton: argparse, config.json loader, weight
  load, tokenizer load, prompt encode, prefill, AR decode, stats.
- `kernels.h.template` — kernel signature pattern (one C function per
  distinct math op). Includes a checklist of common modern-architecture
  ops (output gate, partial RoPE, q/k_norm, SSM step, etc.).
- `kernels.c.template` — naive scalar reference kernels plus commented
  snippets for two common quantization formats (MXFP4 and MLX 8-bit affine).
- `Makefile.template` — `make src-cpu` builds `./<BIN>`.
- `tools/dump_ref.py.template` — Python oracle dumper. Includes both a
  PyTorch / HF path (using `forward_hook`s) and an MLX path (manual
  forward walk, since MLX has no hook mechanism).
- `tests/test_kernels.c.template` — consolidated test harness with a
  `TESTS[]` table and a bf16-aware (`abs + rel * |want|`) tolerance
  helper. Run a single kernel with `./test_kernels rmsnorm` etc.

## Procedure

### Phase 1: Reconnaissance — understand the reference

1. Read the Python reference's model file (e.g. `mlx_lm/models/<model>.py`
   or the HF `modeling_*.py`).

2. Identify the model architecture by reading the reference's `forward`
   methods. Build a mental list of the **distinct math operations** in
   order, e.g.:
   - `embed_gather` (one-time at start)
   - per layer:
     - `rmsnorm` (input)
     - `linear` (q_proj, k_proj, v_proj)
     - `rope` (q, k)
     - `sdpa` (with sinks? sliding window? GQA?)
     - `linear` (o_proj)
     - `residual_add`
     - `rmsnorm` (post-attn)
     - router-specific: `linear` (router), `topk_softmax`
     - MoE: `quant_linear_gather` (gate_up_proj), `swiglu` (or activation),
       `quant_linear_gather` (down_proj), `expert_mix`
     - `residual_add`
   - final `rmsnorm`
   - `linear` (lm_head)
   - `argmax` (greedy)

3. Read `config.json` to extract dimensions: `num_hidden_layers`,
   `hidden_size`, `num_attention_heads`, `num_key_value_heads`,
   `head_dim`, `intermediate_size`, `num_local_experts`,
   `num_experts_per_tok`, `vocab_size`, `sliding_window`,
   `rope_theta` / `rope_scaling`, etc.

4. Identify model-specific details by reading the reference and asking
   the user when unclear:
   - **Quantization format**: bf16? f16? MXFP4 (e2m1+e8m0)? FP8? Per-channel
     scales? Group size?
   - **RoPE flavor**: standard? llama-style? yarn? long-rope?
   - **Attention variant**: MHA / GQA / MQA; sliding window every N
     layers? Attention sinks?
   - **Activation**: SwiGLU / GeGLU / GELU? Clamping (e.g., gpt-oss
     clamps gate at 7.0)?
   - **Normalization**: RMSNorm? LayerNorm? `eps`?
   - **Chat template / prompt format**: which one (chatml, harmony, llama
     instruct, mistral, ...)? Special token IDs?
   - **Stop tokens** for AR decode loop.

5. Identify the **safetensors tensor names** for each weight by reading
   the reference's `__init__` / `state_dict` keys.

6. Commit a `src-cpu/README.md` capturing this reconnaissance.

### Phase 2: Scaffold — get loader + tokenizer + CLI working

1. Create `src-cpu/`. Copy drop-in starters: `safetensors.{c,h}`,
   `tokenizer.{c,h}`, `bf16.h`. Customize `tools/build_tokenizer.py`'s
   special-token list for your model and run it to produce
   `src-cpu/tokenizer.bin`.

2. Author `<chattmpl>.{c,h}` (e.g., `harmony.{c,h}`) that builds the
   prompt token sequence. **Tip**: have the Python reference output the
   numeric token IDs it produces for the same `(system, user, reasoning)`
   tuple, and write the C builder to match byte-for-byte. Add a small
   test that compares C and Python token sequences.

3. Author `main.c` from `main.c.template`:
   - argparse `--prompt`, `--system`, `--max-tokens`, `--model`
   - Load `config.json` into a `cfg_t` struct
   - Open safetensors archive
   - Load tokenizer; build prompt; print prompt_ids
   - Stub `forward()` that just returns argmax of a zero vector
   - Stub AR loop

4. Author `Makefile` from `Makefile.template` (replace `<BIN>` with
   your chosen executable name from Inputs). Build and run:
   ```
   make
   ./<BIN> --prompt "Hello" --max-tokens 1 --model /path/to/model
   ```
   **Smoke-test checklist** (all must pass before moving to Phase 3):
   - [ ] Build succeeds with `-Wall -Wextra` clean.
   - [ ] Program prints the model dims read from `config.json`.
   - [ ] Program prints the list of tensors found in safetensors and
         their dtypes (add a `--list-tensors` debug flag if helpful).
   - [ ] Program prints the prompt token IDs from the chat-template
         builder and they match what the Python reference produces for
         the same `(system, user)`.
   - [ ] No segfault.

5. **Commit**: `src-cpu: scaffold loader + tokenizer + CLI`.

### Phase 3: Build the Python oracle

Author `tools/dump_ref.py` (start from
`starter/tools/dump_ref.py.template`):

1. Load the Python reference model and tokenizer.
2. Encode the validation prompt.
3. Run the forward pass on layer 0 only (initially) and dump the
   intermediate tensors at every op boundary. Use a small binary format
   with a magic header:
   ```
   "LLMTNSR1" (8 bytes) | u32 dtype | u32 ndim | i64[ndim] shape | data
   ```
4. Dump to `src-cpu/refs/`:
   ```
   input_ids.bin
   x_after_embed.bin
   x_after_rmsnorm_0.bin
   q_proj_0.bin
   k_proj_0.bin
   v_proj_0.bin
   q_after_rope_0.bin
   k_after_rope_0.bin
   attn_out_0.bin
   o_proj_0.bin
   ...
   ```

5. **Commit**: `tools/dump_ref.py + initial oracle dump`.

### Phase 4: Implement kernels one at a time

This is the bulk of the work. For each distinct math op in the reference's
forward pass, in the order it appears:

#### 4a. Define the kernel signature in `kernels.h`

One C function per distinct math op. Reuse the same function for
different call sites of the same op (e.g., one `linear_bf16` used for
q_proj / k_proj / v_proj / o_proj / router / lm_head).

Example signatures (gpt-oss-shaped):
```c
void embed_gather_bf16(const int32_t* ids, const bf16* E,
                       bf16* Y, uint32_t M, uint32_t D);

void rmsnorm_bf16(const bf16* X, const bf16* W, bf16* Y,
                  uint32_t M, uint32_t D, float eps);

void linear_bf16(const bf16* X, const bf16* W, const bf16* B,
                 bf16* Y, uint32_t M, uint32_t K, uint32_t N);

void rope_yarn_inplace_bf16(bf16* X, const float* inv_freqs,
                            uint32_t L, uint32_t H, uint32_t D,
                            uint32_t q_offset, float mscale);

void sdpa_with_sinks_bf16(const bf16* Q, const bf16* K, const bf16* V,
                          const bf16* sinks, bf16* OUT,
                          uint32_t Lq, uint32_t Lk,
                          uint32_t Nq, uint32_t Nkv, uint32_t D,
                          uint32_t window, uint32_t q_offset, float scale);

void topk_softmax_bf16(const bf16* logits, int32_t* out_idx, float* out_w,
                       uint32_t L, uint32_t E, uint32_t K);

void mxfp4_linear_gather_bf16(const bf16* X,
                              const uint8_t* blocks, const uint8_t* scales,
                              const bf16* bias, const int32_t* expert_ids,
                              bf16* Y, ...);

void swiglu_bf16(const bf16* gate, const bf16* up, bf16* out,
                 uint32_t N, float alpha, float limit);

void expert_mix_bf16(const bf16* per_expert_out, const float* topk_w,
                    bf16* out, uint32_t L, uint32_t K_top, uint32_t H);

void residual_add_bf16(bf16* x, const bf16* d, uint32_t N);

void argmax_bf16(const bf16* logits, int32_t* out_idx, uint32_t N);
```

**Rules**:
- One function per *distinct* op (not per call site).
- Plain C, no shim, no buffer/pipeline abstraction. Pointers + sizes.
- Native model precision in storage (bf16 if model is bf16); fp32 in
  accumulators where it changes the result.
- **No fusion** beyond what the reference itself does. If the reference
  has separate `gate_up_proj` + `swiglu`, you do too. If the reference
  has a fused `quant_linear+gather` (MLX does, for MoE — never
  materializes a dequantized buffer), follow that fusion.
- **No prefill-vs-decode special paths**. Same kernel for `Lq=1` and
  `Lq>1`. The Metal port may split them later.

#### 4b. Implement the kernel in `kernels.c`

Naive scalar implementation. Examples:

```c
// Y[m,n] = B[n] + Σ_k X[m,k] * W[n,k]    (note W is row-major [N,K])
void linear_bf16(const bf16* X, const bf16* W, const bf16* B,
                 bf16* Y, uint32_t M, uint32_t K, uint32_t N) {
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            float acc = B ? bf16_to_f32(B[n]) : 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                acc += bf16_to_f32(X[m*K + k]) * bf16_to_f32(W[n*K + k]);
            }
            Y[m*N + n] = f32_to_bf16(acc);
        }
    }
}
```

**Style guidelines**:
- One layer of nested loops per logical tensor index.
- fp32 accumulator for any reduction.
- Use `bf16_to_f32` / `f32_to_bf16` (in `bf16.h`); do not introduce
  arch-specific intrinsics.
- Don't try to be clever. The Metal port will rearrange this anyway.

#### 4c. Write a per-kernel correctness test

In `tests/test_<kernel>.c`:

```c
int main(void) {
    // 1. Load inputs from refs/*.bin (use a small loader helper).
    // 2. Call the C kernel.
    // 3. Compare against refs/<output>.bin element-by-element.
    // 4. Print max-abs-error, mean-abs-error.
    // 5. Exit 0 if max-abs-error < tol; nonzero otherwise.
}
```

A typical bf16 tolerance is `1e-2` element-wise abs (because bf16 has
~3 decimal digits of mantissa). For fp32 paths, `1e-5` is reasonable.
Reductions can drift more — accept up to `~K * eps_per_op` for large K.

#### 4d. Iterate the kernel until it passes

If the test fails:
- Print the first few mismatching elements.
- Check tensor layout / strides against the reference.
- Check accumulator precision. Move to fp32 if you were using bf16.
- Check special cases (sinks, sliding window, the +1 in SwiGLU's
  `(u+1)`, the clamping, the scale on RoPE for yarn, etc.).
- **Compare math literally with the reference Python**. Often a hidden
  scaling factor or transpose is the culprit.

#### 4e. Commit per-kernel

```
src-cpu: <kernel_name> + test (max|d|=<tol> vs ref)
```

This makes rollback trivial and gives you a clear log of progress.

#### 4f. Iterate over all ops

Walk the reference's forward pass and repeat 4a–4e for every distinct
op. The order matters — implement in *forward-pass order* so you can
chain dumps later: layer 0 input → layer 0 output → layer 1 input → ...

### Phase 5: Stitch into a forward() function

1. In `main.c`, write `forward(int q_off, int Lq)` (or however your model
   parameterizes it) that:
   - For each layer, calls every kernel in reference order, threading
     activations through workspace buffers.
   - Manages per-layer KV cache (mmap'd or malloc'd, sized to
     `MAX_CTX * N_KVHEADS * HEAD_DIM * sizeof(bf16)`).

2. Add the AR decode loop:
   - Embed prompt → run `forward(0, Lp)` for prefill → argmax
   - Loop: embed one id → run `forward(Lp+i, 1)` for decode → argmax →
     emit token → break on stop.

3. End-to-end run with the same prompt as the Python reference:
   ```
   ./<BIN> --prompt "Hello, who are you?" --max-tokens 16
   ```
   Compare generated token IDs against the reference. They must match.

4. If they don't match: dump intermediates from the C run (e.g., add a
   `--dump-c-refs` flag that writes the same `refs/*.bin` files as the
   Python oracle does), and binary-diff against the Python refs. The
   first divergence localizes the bug.

5. **Commit**: `src-cpu: end-to-end forward + AR decode (N/N tokens match ref)`.

### Phase 6: Acceptance

- [ ] `./<BIN> --prompt "<validation_prompt>" --max-tokens N` produces
      the **same N token IDs** as the Python reference under greedy decoding.
- [ ] All per-kernel tests in `tests/` pass.
- [ ] Build is `make src-cpu` with no warnings (`-Wall -Wextra`).
- [ ] `src-cpu/` is standalone (no link to the Python reference or to
      any external runtime besides libc and libm).

When all boxes are ticked, the skill is done.

## Tips

### Choosing kernel boundaries

The user-supplied rule is **structure kernels EXACTLY as the reference**.
That means: if the reference does

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

### Order of operations vs reference

Maintain reference order even when it's "wasteful":

- RoPE on Q and RoPE on K can run in either order in principle, but the
  reference picks one — do the same.
- KV cache writes happen in a specific position relative to RoPE — match.
- Residual additions happen at a specific spot — match.

### Numerical drift

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

### `-ffast-math` warning

Adding `-ffast-math` may speed up the C run by 1.3–2× but it allows the
compiler to reassociate fp ops and assume no NaN/Inf. Test that
end-to-end token output still matches before enabling. If even one token
diverges, **drop the flag**.

### MoE specifics

Naive MoE for the C reference: just loop over `K_TOP` selected experts
per token, no bucketing. It will be slow (~`32x` slower than dense),
but that's fine. The Metal port (and especially `optimize-metal`) will
add the sorted-gather grouped-MoE fast path.

### Sliding-window attention

Even-numbered layers use a sliding window of e.g. 128 tokens; odd layers
use full attention. In the C reference, this is just `win_lo = max(0,
lq_abs - W + 1)` inside the SDPA kernel. No need for a separate "ring
KV cache" — keep the full KV cache for simplicity. The Metal port can
optimize this.

### Quantization (MXFP4 example)

For openai-style MXFP4 (e2m1 packed 2/byte + e8m0 per-32 block scale):

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

### Quantization (MLX-style affine — 4-bit and 8-bit)

MLX uses a different family of quantization formats. For a Linear of
shape `(N, K)` quantized affinely at `B` bits with `group_size` `G`,
the safetensors stores **three** tensors:

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

Quirks worth remembering:
- Even small linears (e.g., outputs of 1 or 32) are quantized in MLX
  models. Don't assume "1-dim outputs" means f32.
- MLX's `mx.quantized_matmul` and `mx.gather_qmm` dequant inline; never
  materialize a dequantized weight buffer in the C reference.
- `Embedding.weight` is quantized too in MLX-quantized models — use the
  same dequant in your `embed_gather` kernel.
- The MLX inference code may force certain weights to a specific quant
  (`quant_predicate` in `mlx_lm/models/<name>.py`). Read it.

### Modern-architecture patterns you'll likely encounter

The basic transformer "decoder block" (RMSNorm → QKV linear → RoPE →
SDPA → o_proj → residual → RMSNorm → SwiGLU → residual) is increasingly
just the *starting* template. Newer models add:

- **Grouped-query attention (GQA)**: `num_kv_heads << num_attention_heads`.
  In the C reference, just one SDPA kernel that maps q-head `h` to kv-head
  `h / (Nq / Nkv)`.
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
  to SDPA; clamp the lower bound on the key index per query.
- **Attention sinks**: an extra per-head logit that contributes only to
  the softmax denominator (no V). Pass a `sinks` pointer to SDPA.
- **MoE with shared expert**: in addition to top-K experts, every token
  also passes through a shared expert and (optionally) a scalar
  shared-expert gate `sigmoid(gate1) * shared_out`. Total `y = mix(experts)
  + shared_gate * shared_out`.
- **`norm_topk_prob`**: some models renormalize the top-K router scores
  to sum to 1 after argpartition; others don't. Check the reference.
- **Tied vs untied lm_head**: `tie_word_embeddings: true` means
  `logits = embed_tokens.as_linear(x)`. Otherwise it's a separate
  `lm_head` weight.

#### Hybrid linear-attention / SSM models

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

### MLX-specific gotchas

MLX is the preferred Python reference for this skill because the
codebase is small and easy to read. But it has a few footguns:

1. **No `forward_hook` mechanism**. To dump per-op intermediates you
   must manually walk the forward pass in Python, mirroring the
   reference module's `__call__`. See `tools/dump_ref.py.template` for
   a worked example.
2. **`mx.eval(...)`**. MLX is lazy; without `mx.eval` (or `np.array(...)`,
   which forces evaluation) you'll OOM on big models. After each layer
   in the manual walk, call `mx.eval(x)`.
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
9. **`mx.fast.rope`** with `freqs=...` uses an explicit per-half-dim
   frequency table, not the implicit `1 / base^(2k/D)` form. The two
   are equivalent, but the C side needs to precompute the same table:
   `inv_freqs[k] = 1 / base ** (2k / D_rot)`.

### Multi-modal config.json parsing

For multi-modal models the relevant LM dimensions live inside a
nested object:

```json
{
  "model_type": "qwen3_5_moe",
  "text_config":   { "num_hidden_layers": 40, "hidden_size": 2048, ... },
  "vision_config": { "num_hidden_layers": 27, "hidden_size": 1152, ... }
}
```

If you grep flatly for `num_hidden_layers` you'll get the wrong block.
Implement a scoped JSON walker that descends into `text_config` first.
The starter `main.c.template` shows the pattern (`find_section` +
`find_key_in_section`).

Some models also place the architecture-defining keys (e.g.,
`rope_parameters`, `quantization`) at multiple levels of nesting.
Always confirm by `print(json.load(open('config.json')))` first.

### Tolerance setting for bf16

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

## Common pitfalls

- **W transpose**: HF safetensors typically stores Linear weights as
  `[out_features, in_features]`. So `Y = X @ W.T` becomes `Y[m,n] = Σ_k
  X[m,k] * W[n,k]`. Get this wrong and EVERY linear is off.
- **GQA head grouping**: for grouped-query attention, q head `h` reads
  KV head `h / (Nq/Nkv)`. Easy to off-by-one.
- **Causal mask + sliding window + sinks**: get all three right. Sink
  contributes one extra logit per head with no value (it just adds to
  the denominator of the softmax).
- **bf16 round-to-nearest-even**: naive `(x >> 16)` truncation will
  cause systematic small errors. Use round-to-nearest-even (see
  `bf16.h`).
- **RoPE yarn scaling**: yarn has an extra `mscale` multiplier on top of
  the rotation — easy to miss. Match the reference exactly.
- **RoPE freq formula**: `inv_freqs[k] = 1 / (base ** (2k / D))` for
  `k ∈ [0, D/2)`. With `partial_rotary_factor < 1.0` use `D = D_rot =
  head_dim * partial_rotary_factor`, NOT `head_dim`.
- **Tokenizer special tokens**: HF `tokenizer.json` has special tokens in
  `added_tokens` — make sure `build_tokenizer.py` includes them.
- **Tokenizer `MAGIC` length**: the C reader expects 12 bytes; if the
  Python writes only 11 you'll silently load garbage. The fixed
  template now writes 12.
- **Stop token**: pick the right stop token from the reference's
  generation config. For gpt-oss it's `<|return|>` (id 200002), not
  `<|endoftext|>`. For Qwen it's `<|im_end|>` (or `<|endoftext|>`,
  depending on the chat template).
- **Chat template token IDs**: the chat-template scaffolding is part of
  the *input* to the model. Before debugging anything, dump the prompt
  IDs from both the Python reference and your C builder and assert
  byte-for-byte equality. Many problems are actually tokenization
  mismatches.
- **`vocab_size > tokenizer vocab`**: many models pad the embedding to
  a power of 2 or a multiple of 64/128. The extra rows are unused; the
  C `argmax` over the full `vocab_size` is correct as long as you never
  emit a row index past the tokenizer's known set during decode.
- **Pre-tokenizer regex** varies per tokenizer family. The starter
  `tokenizer.c` is tuned for o200k_harmony; for llama / qwen / mistral
  the alternations are different (notably digit handling and how
  `\p{L}\p{M}+` is split). Swap `pretokenize()` per `tokenizer.json`'s
  `pre_tokenizer.pattern`.
- **`safetensors.c` `__metadata__` key**: `__metadata__` is 12 chars,
  not 11. The fixed template has this corrected.
- **Skipping vision tower / MTP heads**: multi-modal and "multi-token-
  prediction" checkpoints have many unused tensors. The Python
  reference's `sanitize` drops them; the C side may have to walk the
  archive but ignore them. Filter by name prefix (`vision_tower.`,
  `mtp.`, etc.).
- **`sanitize` adds `+1.0` to RMSNorm weights**: some `sanitize` paths
  shift all RMSNorm weights by +1 when MTP weights are present in the
  archive (Qwen 3.5 family). The C loader must replicate this.
- **`conv1d.weight` axis order**: HF stores Conv1d weights as
  `[C_out, C_in/groups, kernel]`. MLX `sanitize` may `moveaxis(2, 1)`
  to put the kernel dim in the middle. Confirm via the safetensors
  header shape.
- **SSM state across forward calls**: prefill leaves the state with
  prompt content; the AR decode loop continues from there. Don't reset
  state between prefill and decode.
- **Final norm + lm_head**: only the last position's logits are needed
  for sampling. Apply `model.norm` and `lm_head` to just `x[-1, :]` to
  save `(L-1)/L` of the work. The starter `main.c.template` does this.
- **`-Wall -Wextra` clean**: hidden bugs love to live in `int`/`size_t`
  width mismatches and missing-prototype warnings. Get a clean build
  before declaring a kernel "validated".

## Commit strategy

Make one commit per phase and per kernel:

- `src-cpu: scaffold loader + tokenizer + CLI` (Phase 2)
- `tools/dump_ref.py + initial oracle dump` (Phase 3)
- `src-cpu: embed_gather_bf16 + test` (Phase 4 per kernel)
- `src-cpu: rmsnorm_bf16 + test` ...
- ...
- `src-cpu: forward() + AR decode (N/N tokens match ref)` (Phase 5)

This lets the next skill (`port-c-to-metal`) consult the per-kernel
boundaries when porting and roll back surgically if anything regresses.

## Hand-off to port-c-to-metal

The hand-off contract:

1. `./src-cpu/` exists and builds with `make src-cpu`.
2. `./src-cpu/<BIN> --prompt "<validation_prompt>" --max-tokens N`
   produces the same N tokens as the Python reference.
3. Every kernel is a plain C function in `kernels.{c,h}` with clear
   pointer-arity inputs / outputs.
4. `src-cpu/refs/` contains per-op oracle dumps usable by the Metal
   port for kernel correctness tests.
5. Tokenizer and chat-template are model-agnostic infrastructure that the
   Metal port can copy/symlink rather than re-author.

When this contract is met, run the `port-c-to-metal` skill.
