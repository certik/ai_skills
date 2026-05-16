---
name: build-c-reference
description: >
  Build a pure-C, single-threaded CPU reference implementation of an LLM
  inference engine, starting from a working Python reference (MLX preferred,
  else PyTorch/HF transformers). The C reference is intentionally simple
  (no SIMD, no threads), uses the model's native precision (bf16 / fp8 /
  mxfp4 / etc.) with fp32 accumulators, mirrors the reference's kernels
  1:1 (no fusion), and is validated kernel-by-kernel against the Python
  oracle. The output is a standalone `./csrc-cpu/` directory that loads
  HF safetensors directly, includes its own BPE tokenizer + chat template,
  and produces the same generated tokens as the reference for a fixed
  prompt under greedy decoding. The output is meant to be the starting
  point for `port-c-to-metal` (and later `optimize-metal`).
  Triggers: build c reference, c reference, csrc-cpu, port to c, pure-c
  port, c inference reference, c llm port, cpu reference, slow reference.
---

# build-c-reference — pure-C CPU reference port of an LLM

Take a working Python reference of an LLM (MLX, HF transformers, PyTorch,
etc.) and a HF-safetensors weight directory, and produce a standalone
`./csrc-cpu/` that runs the same model in plain C99 on a single CPU core,
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
3. **Sample prompt** for end-to-end validation (default:
   `"Hello, who are you?"` with greedy decoding, max 16 tokens).
4. **System / chat-template metadata** if not obvious from the reference
   (e.g., harmony reasoning level for gpt-oss).
5. **Output directory** (default: `./csrc-cpu/` in the repo root).

## Output layout

```
./csrc-cpu/
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
    └── test_kernel_*.c  # per-kernel correctness tests vs refs/
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
  distinct math op).
- `Makefile.template` — `make csrc-cpu` builds `./gptoss-cpu`.
- `tools/dump_ref.py.template` — Python oracle dumper using `MAGIC` +
  `dtype` + `shape` header format readable from C tests.

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

6. Commit a `csrc-cpu/README.md` capturing this reconnaissance.

### Phase 2: Scaffold — get loader + tokenizer + CLI working

1. Create `csrc-cpu/`. Copy drop-in starters: `safetensors.{c,h}`,
   `tokenizer.{c,h}`, `bf16.h`. Customize `tools/build_tokenizer.py`'s
   special-token list for your model and run it to produce
   `csrc-cpu/tokenizer.bin`.

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

4. Author `Makefile` from `Makefile.template`. Build and run:
   ```
   make csrc-cpu
   ./csrc-cpu/gptoss-cpu --prompt "Hello" --max-tokens 1
   ```
   At this point the program should load weights and tokenize the
   prompt without crashing (output garbage is OK).

5. **Commit**: `csrc-cpu: scaffold loader + tokenizer + CLI`.

### Phase 3: Build the Python oracle

Author `tools/dump_ref.py` (start from
`starter/tools/dump_ref.py.template`):

1. Load the Python reference model and tokenizer.
2. Encode the validation prompt.
3. Run the forward pass on layer 0 only (initially) and dump the
   intermediate tensors at every op boundary. Use a small binary format
   with a magic header:
   ```
   "GPTOSSF1" (8 bytes) | u32 dtype | u32 ndim | i64[ndim] shape | data
   ```
4. Dump to `csrc-cpu/refs/`:
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
csrc-cpu: <kernel_name> + test (max|d|=<tol> vs ref)
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
   ./gptoss-cpu --prompt "Hello, who are you?" --max-tokens 16
   ```
   Compare generated token IDs against the reference. They must match.

4. If they don't match: dump intermediates from the C run (e.g., add a
   `--dump-c-refs` flag that writes the same `refs/*.bin` files as the
   Python oracle does), and binary-diff against the Python refs. The
   first divergence localizes the bug.

5. **Commit**: `csrc-cpu: end-to-end forward + AR decode (N/N tokens match ref)`.

### Phase 6: Acceptance

- [ ] `./gptoss-cpu --prompt "<validation_prompt>" --max-tokens N` produces
      the **same N token IDs** as the Python reference under greedy decoding.
- [ ] All per-kernel tests in `tests/` pass.
- [ ] Build is `make csrc-cpu` with no warnings (`-Wall -Wextra`).
- [ ] `csrc-cpu/` is standalone (no link to the Python reference or to
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
- **Tokenizer special tokens**: HF `tokenizer.json` has special tokens in
  `added_tokens` — make sure `build_tokenizer.py` includes them.
- **Stop token**: pick the right stop token from the reference's
  generation config. For gpt-oss it's `<|return|>` (id 200002), not
  `<|endoftext|>`.

## Commit strategy

Make one commit per phase and per kernel:

- `csrc-cpu: scaffold loader + tokenizer + CLI` (Phase 2)
- `tools/dump_ref.py + initial oracle dump` (Phase 3)
- `csrc-cpu: embed_gather_bf16 + test` (Phase 4 per kernel)
- `csrc-cpu: rmsnorm_bf16 + test` ...
- ...
- `csrc-cpu: forward() + AR decode (N/N tokens match ref)` (Phase 5)

This lets the next skill (`port-c-to-metal`) consult the per-kernel
boundaries when porting and roll back surgically if anything regresses.

## Hand-off to port-c-to-metal

The hand-off contract:

1. `./csrc-cpu/` exists and builds with `make csrc-cpu`.
2. `./csrc-cpu/gptoss-cpu --prompt "<validation_prompt>" --max-tokens N`
   produces the same N tokens as the Python reference.
3. Every kernel is a plain C function in `kernels.{c,h}` with clear
   pointer-arity inputs / outputs.
4. `csrc-cpu/refs/` contains per-op oracle dumps usable by the Metal
   port for kernel correctness tests.
5. Tokenizer and chat-template are model-agnostic infrastructure that the
   Metal port can copy/symlink rather than re-author.

When this contract is met, run the `port-c-to-metal` skill.
