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
├── safetensors.{c,h}  # HF safetensors mmap loader (drop-in)
├── tokenizer.{c,h}    # BPE tokenizer engine (mostly drop-in; swap regex)
├── tokenizer.bin      # built by ./build_tokenizer (pure-C, no Python)
├── build_tokenizer.c  # tokenizer.json → tokenizer.bin (drop-in for byte-BPE)
├── <chattmpl>.{c,h}   # model-specific chat-template builder
├── utils/             # reusable, model-agnostic helpers (no model knowledge)
│   ├── bf16.h         # bf16 ↔ fp32 helpers (RN-even)
│   ├── jsmn.h         # vendored single-header JSON tokenizer (MIT)
│   ├── json.{c,h}     # Python-like JSON reader on top of jsmn
│   ├── utf8.{c,h}     # UTF-8 encode/decode (one codepoint)
│   ├── bytebuf.{c,h}  # growable byte buffer (Python's `bytearray`)
│   └── iofile.{c,h}   # read-only file mmap helper
├── Makefile
├── refs/              # oracle dumps from Python (per-kernel inputs/outputs)
│   ├── input_ids.bin
│   ├── x_after_embed.bin
│   ├── x_after_rmsnorm_0.bin
│   └── ...
└── tests/
    ├── load_ref.h         # .bin loader + bf16-aware err_check
    └── test_kernels.c     # one test fn per kernel; CLI-selectable
```

Plus the **only** remaining Python tools, kept outside `src-cpu/`:

```
./tools/
├── dump_ref.py            # runs the Python reference, dumps refs/*.bin
└── run_ref_<sampler>.py   # runs the Python reference's sampler with
                           # deterministic config — produces token IDs
                           # to diff against C output for Phase 5/6.
```

Note: `src-cpu/` is fully Python-free. The inference build needs only
clang + libc + libm. Python is used only to (a) regenerate oracle dumps
in `refs/` for the per-kernel correctness tests, and (b) produce a
reference token sequence for end-to-end token-equivalence checks.

## Starter files (in this skill's `starter/`)

### Drop-in, model-agnostic (copy verbatim)

The `utils/` subdirectory is a small "Python-stdlib equivalent in C"
toolkit shared by `safetensors.c`, `tokenizer.c`, `build_tokenizer.c`,
and `main.c`. None of these files contain any model knowledge:

- `utils/bf16.h` — `bf16_to_f32` / `f32_to_bf16` (round-to-nearest-even).
- `utils/jsmn.h` — vendored single-header JSON tokenizer (MIT, Serge Zaitsev).
- `utils/json.{c,h}` — small Python-like wrapper: `json_open_file`,
  `json_open_mem`, `json_get`, `json_path` (dotted), `json_obj_iter`,
  `json_arr_iter`, `json_int_or`, `json_double_or`, `json_str_view`,
  `json_str_utf8`, `json_str_codepoints`. **NB**: defines
  `JSMN_PARENT_LINKS` before including jsmn — without this, parsing
  large objects (like a 250k-entry `tokenizer.json::model.vocab`) is
  O(N²) and takes ~40 s; with it, it's O(N) and takes ~40 ms (1000×
  speedup). See "Common pitfalls".
- `utils/utf8.{c,h}` — `utf8_encode` / `utf8_decode` (one codepoint each).
- `utils/bytebuf.{c,h}` — growable buffer: `bb_append`, `bb_append_u32`
  (little-endian), `bb_write_file`. Aborts on OOM (same policy as
  Python's `bytearray` raising `MemoryError`).
- `utils/iofile.{c,h}` — `iofile_mmap_ro` / `iofile_close`. Replaces
  the open/fstat/mmap/close dance in every caller.

Then the model-shared model-format readers:

- `safetensors.{c,h}` — HF safetensors mmap loader. Handles single-file
  and sharded layouts, BF16/F16/F32/U8/I8/U16/I16/U32/I32/U64/I64. Uses
  `utils/json` + `utils/iofile` internally.
- `tokenizer.{c,h}` — byte-level BPE engine. Reads `tokenizer.bin`.
  The BPE merging algorithm is fully generic; only the **pre-tokenizer
  regex** in `pretokenize()` may need swapping (the default ships an
  ASCII subset that matches Qwen 3.5 / Llama 3; for o200k_harmony /
  Mistral / sentencepiece-based families, rewrite the `match_*` helpers
  per `tokenizer.json::pre_tokenizer.pattern`).
- `build_tokenizer.c` — converts HF `tokenizer.json` → `tokenizer.bin`
  in pure C. Drop-in for any **byte-level BPE** family (Qwen, Llama 3,
  GPT-2, o200k, gpt-oss). For sentencepiece / unigram tokenizers (older
  Llama, Mistral, T5) the vocab-key decoding step is different — copy
  + rewrite the `decode_vocab_key` helper.

### Templates (read, then customize)

- `main.c.template` — CLI skeleton: argparse, config.json loader (via
  `json_path` so multi-modal `text_config` nesting is one line), weight
  load, tokenizer load, prompt encode, prefill, AR decode (with live
  text streaming + final summary), stats.
- `kernels.h.template` — kernel signature pattern (one C function per
  distinct math op). Includes a checklist of common modern-architecture
  ops (output gate, partial RoPE, q/k_norm, SSM step, etc.).
- `kernels.c.template` — naive scalar reference kernels plus commented
  snippets for two common quantization formats (MXFP4 and MLX 8-bit
  affine).
- `Makefile.template` — `make` builds `./<BIN>`; `make tokenizer` builds
  the C tokenizer-bin generator; `make tests` runs the per-kernel
  correctness checks.
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

1. Create `src-cpu/`. Copy the drop-in starters wholesale:
   ```
   utils/{bf16.h, jsmn.h, json.{c,h}, utf8.{c,h}, bytebuf.{c,h}, iofile.{c,h}}
   safetensors.{c,h}
   tokenizer.{c,h}
   build_tokenizer.c
   tests/load_ref.h
   ```
   These contain no model knowledge — only the tokenizer's pre-tokenizer
   regex may need swapping later (see "Common pitfalls").

2. Build and run the C tokenizer-bin generator on your model's
   tokenizer files:
   ```
   make tokenizer MODEL_DIR=/path/to/model
   ```
   This produces `src-cpu/tokenizer.bin` (a compact, mmap'd blob the
   runtime tokenizer reads). For **byte-level BPE** tokenizers (Qwen,
   Llama 3, GPT-2, gpt-oss/o200k) `build_tokenizer.c` is drop-in
   provided the model ships a `tokenizer.json`. For sentencepiece-based
   families, rewrite `decode_vocab_key`.

   **Older HF tokenizer format** (Qwen ≤ 2.5, Llama 1, some older
   community models): no `tokenizer.json`. Instead, two files:
   - `vocab.json`: flat `{token_string: id}` (GPT-2 byte-encoded keys)
   - `added_tokens.json`: flat `{name: id}` for special tokens

   Modify `build_tokenizer.c` to read BOTH and merge them: emit one
   entry per id, with `added_tokens` ids getting empty BPE strings so
   the merge engine never picks them as merge candidates. Total id
   range is `max(vocab.json ids, added_tokens.json ids) + 1`. The
   tiktoken trick (merge priority == vocab id) still works — no
   `merges.txt` needed.

3. Author `<chattmpl>.{c,h}` (e.g., `harmony.{c,h}`, `qwen_chat.{c,h}`,
   `llama3_chat.{c,h}`) that builds the prompt token sequence. **Tip**:
   have the Python reference output the numeric token IDs it produces
   for the same `(system, user, reasoning)` tuple, and write the C
   builder to match byte-for-byte. Add a small test that compares C and
   Python token sequences.

4. Author `main.c` from `main.c.template`:
   - argparse `--prompt`, `--system`, `--max-tokens`, `--model`
   - Load `config.json` via `json_open_file` + `json_get` + `json_int_or`
     (the template handles multi-modal `text_config` nesting in one line)
   - Open safetensors archive
   - Load tokenizer; build prompt; print prompt_ids
   - Stub `forward()` that just returns argmax of a zero vector
   - Stub AR loop

5. Author `Makefile` from `Makefile.template` (replace `<BIN>` with
   your chosen executable name from Inputs). Build and run:
   ```
   make MODEL_DIR=/path/to/model
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

6. **Commit**: `src-cpu: scaffold loader + tokenizer + CLI`.

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
   logits.bin              # full [L, vocab] logits — for Phase 5 validate-forward
   argmax_per_pos.bin      # per-position argmax — easier acceptance metric
   ```

5. **For the PyTorch backend**, wrap both the full forward and any
   manual-replay SDPA calls in `sdpa_kernel([SDPBackend.MATH])`. The
   CPU default `FLASH_ATTENTION` backend disagrees with the textbook
   math on bf16 by `mean|d|~0.004 / max|d|~0.9`, which will cause your
   per-kernel SDPA test to fail at 0.9 absolute even when your kernel
   is correct. See "PyTorch / HF gotchas".

6. **For ops not natural to hook** (RoPE'd q/k, raw SDPA output, fused
   silu*up): the PyTorch `forward_hook` only fires at module
   boundaries. Add a small **"manual layer-0 replay"** section that
   reuses the captured `q_proj_0` etc., re-applies RoPE / SDPA / silu
   manually, and saves the intermediates. The starter template shows
   the MLX equivalent (which always requires manual walking).

7. **Commit**: `tools/dump_ref.py + initial oracle dump`.

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
- Use `bf16_to_f32` / `f32_to_bf16` (in `utils/bf16.h`); do not introduce
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
   - For **non-causal / diffusion-style models**, there is no KV cache:
     `forward(L)` runs over the entire sequence each call. See
     "Discrete-diffusion / masked LMs" below.

2. Add the AR decode loop (or diffusion sampler):
   - Embed prompt → run `forward(0, Lp)` for prefill → argmax
   - Loop: embed one id → run `forward(Lp+i, 1)` for decode → argmax →
     emit token → break on stop.

3. **Add a `--validate-forward REFS_DIR` mode BEFORE running E2E**:
   - Load `refs/input_ids.bin` (saved by the oracle dumper).
   - Run `embed_gather` then `forward(L)` over those ids.
   - Per-position, compute argmax over `logits_bf16[p, :]` and compare
     against `refs/argmax_per_pos.bin`. Also report global max|d| vs
     `refs/logits.bin`.
   - This catches "all kernels pass individually but `forward()` is
     stitched wrong" bugs in **one forward** (tens of seconds) instead
     of waiting for a multi-minute end-to-end run that produces
     mysterious wrong tokens.
   - Acceptance: N/N argmax positions match. Raw logits max|d| may be
     0.5–1.5 after 28+ layers — that's drift, not a bug, as long as
     argmax is preserved. See "Tolerance setting for bf16".

4. End-to-end run with the same prompt as the Python reference:
   ```
   ./<BIN> --prompt "Hello, who are you?" --max-tokens 16
   ```
   Compare generated token IDs against the reference. They must match.

5. **Write a small `tools/run_ref_<sampler>.py`** that runs the Python
   reference with **identical sampler config** (T=0, greedy, no
   sampling-rng-dependent algs) and prints the generated ids. This is
   what you diff against your C output. Two pitfalls:
   - Some references quietly use a non-deterministic backend (e.g.
     PyTorch CPU's default `SDPBackend.FLASH_ATTENTION` for bf16, which
     diverges from textbook math by mean|d|≈0.004 / max|d|≈0.9 — see
     "PyTorch CPU SDPA backend gotcha"). Force the deterministic /
     textbook backend in this Python helper.
   - Some sampler algs (e.g. `alg=origin` in Dream's diffusion) draw
     from `torch.rand` and can't be matched in C. Pick a deterministic
     alg (Dream: `alg=entropy`).

6. If tokens don't match: dump intermediates from the C run (e.g., add
   a `--dump-c-refs` flag that writes the same `refs/*.bin` files as
   the Python oracle does), and binary-diff against the Python refs.
   The first divergence localizes the bug. Most often: a missing bf16
   round-trip in one of the per-element kernels — see "Numerical
   drift".

7. **Budget**: naive scalar bf16 on a single CPU core runs the
   bf16-7B-class forward at ~1–2 GFLOPS/s after `bf16_to_f32` overhead.
   For e.g. Dream 7B at L=25 that's ~85 s/forward, ~15 min for an
   8-step diffusion. **Don't be surprised** if a full diffusion or AR
   loop takes 10+ minutes per run — it's the price of "no SIMD, no
   threads, naive scalar". `port-c-to-metal` and `optimize-metal` will
   cut this to under a second.

8. **Commit**: `src-cpu: end-to-end forward + decode (N/N tokens match ref)`.

### Phase 6: Acceptance

- [ ] `./<BIN> --prompt "<validation_prompt>" --max-tokens N` produces
      the **same N token IDs** as the Python reference under greedy decoding.
- [ ] All per-kernel tests in `tests/` pass.
- [ ] Build is `make src-cpu` with no warnings (`-Wall -Wextra`).
- [ ] `src-cpu/` is standalone (no link to the Python reference or to
      any external runtime besides libc and libm).

When all boxes are ticked, the skill is done.

## Tips

### The `utils/` library — Python-stdlib equivalents in C

The starter `utils/` directory exists because four different C files in
the project (`safetensors.c`, `tokenizer.c`, `build_tokenizer.c`,
`main.c::load_config`) all need to do the same four things:

1. read a file by name (`open` / `fstat` / `mmap` / `close`),
2. grow a byte buffer and write it back out (`bytearray` + `struct.pack`),
3. parse + navigate a JSON document, and
4. encode / decode UTF-8 codepoints.

If every caller does these by hand you get four subtly different mini-
parsers (the substring-search JSON walker bites you on multi-modal
configs; the manual mmap dance gets the error path wrong; the BPE byte
encoder reinvents UTF-8). Centralising them is a one-time ~700-LOC
investment that the rest of `src-cpu/` reads as if it were Python:

```c
// "with open(path) as f: cfg = json.load(f)" + "cfg['text_config']['rope_theta']"
json_doc_t* doc = json_open_file(path, &err);
double theta = json_double_or(
    json_path(json_root(doc), "text_config.rope_parameters.rope_theta"), 1e4);

// "buf = bytearray(); buf += MAGIC; buf += struct.pack('<I', n); open(out,'wb').write(buf)"
bytebuf_t out = BYTEBUF_INIT;
bb_append    (&out, MAGIC, MAGIC_LEN);
bb_append_u32(&out, n);
bb_write_file(&out, out_path);

// "with open(path,'rb') as f: data = mmap.mmap(f.fileno(), 0, mmap.PROT_READ)"
iofile_t f;
iofile_mmap_ro(path, &f, &err);
// ... f.data, f.size ...
iofile_close(&f);
```

Treat `utils/` as a fixed dependency of the project. Do *not* leak any
model-specific knowledge into it: that's how you keep the port to a new
model from being a fork of the whole tree.

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

#### Per-element bf16 rounding patterns ("hidden casts")

This is the **#1 cause of 1–4 ULP per-element kernel mismatches**.
When the reference is written in PyTorch / MLX / etc., each `tensor op
tensor` in the model dtype (bf16) **rounds the intermediate to bf16**
even when the surrounding math looks like one fused fp32 expression. A
naive C kernel that runs the whole expression in fp32 and rounds only
at the final store will be 1–2 ULP off per element.

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
- **Bidirectional attention** (`is_causal=False`): discrete-diffusion
  and masked LMs (Dream, LLaDA, MDLM, ...) use BERT-style bidirectional
  attention. SDPA collapses to `softmax(QK/√D) @ V` with no mask, no
  sinks, no window. The whole "AR decode loop" goes away — see
  "Discrete-diffusion / masked LMs" below.

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

#### Discrete-diffusion / masked LMs

Discrete-diffusion LMs (Dream, LLaDA, MDLM, ...) replace the AR decode
loop with an iterative denoising sampler. Important consequences for
the C reference:

1. **Bidirectional attention** (`is_causal=False`). No causal mask, no
   sliding window, no sinks. SDPA is just `softmax(QK/√D) @ V`. No
   q_offset, no per-token attention window — every position attends to
   every position.
2. **No KV cache**. Each denoising step re-runs the full forward over
   `[L = prompt_len + max_new_tokens]`. Allocate workspaces once at
   size `L` and clobber them per step.
3. **Mask-token padding**. The initial sequence is
   `[prompt_ids..., MASK, MASK, ...]` where MASK is a special id
   (e.g., 151666 for Dream). Each step replaces some masks with the
   model's predictions.
4. **Logits-shift trick** (per `generation_utils._sample` in many of
   these models): the logits used at position `p` are actually the row
   computed for position `p-1` (`logits = cat(logits[:,:1], logits[:,:-1], dim=1)`).
   In C, mirror this by scoring position `p` with row `(p == 0 ? 0 : p - 1)`.
5. **Confidence-based scheduling**. Each step picks the top-K most
   confident masked positions and unmasks them. Common confidence
   metrics: `maskgit_plus` (top-1 prob), `topk_margin` (top1 − top2),
   `entropy` (negative entropy). At `temperature=0` all three are
   deterministic — pick one (e.g., `entropy`) for E2E validation.
6. **Skip RNG-dependent algs**. Some samplers offer an `alg=origin` or
   `alg=random` mode that draws from `torch.rand`; you cannot match
   these in C without reimplementing PyTorch's Philox / Mersenne RNG.
   Stick to deterministic algs.
7. **Validation prompt has to leave room for output**. If
   `max_new_tokens=8` is too few and the model decides everything is
   `<eos>`, your "matching" output is `[eos]*8` on both sides — still
   valid bit-equivalent proof, but trivially so. Bump to ≥16 to see a
   real response.

The forward kernels are identical to a vanilla decoder (just
`is_causal=False` in SDPA). What changes is the **outer driver loop**
in `main.c`:

```c
// Pseudocode for a diffusion sampler driver.
for (i = 0; i < steps; i++) {
    n_mask = count(x == MASK);
    if (!n_mask) break;
    embed_gather(x, ..., x_buf);
    forward(L);                          // bidirectional, full sequence
    for each p with x[p] == MASK:
        src = (p == 0) ? 0 : p - 1;
        (id, conf) = score_row(logits[src], alg);
        cand_id[p] = id; cand_conf[p] = conf;
    n_transfer = (i == steps-1) ? n_mask : (int)(n_mask * (1 - s/t));
    topk_indices = topk_desc(cand_conf, n_transfer);
    for each idx in topk_indices: x[idx] = cand_id[idx];
}
```

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

### PyTorch / HF-specific gotchas

The PyTorch / HF transformers stack is the more common fallback when
no MLX port exists. Its footguns:

1. **`SDPBackend.FLASH_ATTENTION` on CPU is not bit-equivalent to
   the textbook math** for bf16 inputs. On CPU, PyTorch's default for
   `torch.nn.functional.scaled_dot_product_attention(..., is_causal=False)`
   with bf16 q/k/v is `FLASH_ATTENTION`, which disagrees with
   `SDPBackend.MATH` by `mean|d| ≈ 0.004`, `max|d| ≈ 0.93` on a typical
   7B model. Your naive C SDPA implements the textbook math, so:

   ```python
   from torch.nn.attention import SDPBackend, sdpa_kernel
   with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
       out = model(input_ids=ids, ...)  # oracle dump
   ```

   Apply this both in `tools/dump_ref.py` (so the oracle matches what
   your C SDPA computes) AND in `tools/run_ref_<sampler>.py` (so the
   reference's generated tokens are comparable to your C output).
   Without this, your per-kernel SDPA test will fail by ~0.9 absolute
   (real divergence, not precision drift) and end-to-end tokens may
   differ.

2. **`AutoModel` may wrap the base class**. For HF models, the
   `architectures` field in `config.json` tells you which class
   `AutoModel.from_pretrained` returns. For LM-headed wrappers
   (`<X>ForCausalLM`, `<X>ForMaskedLM`, etc.) the returned object has
   the `lm_head` AND its `.forward()` returns a `*Output` dataclass with
   a `.logits` field — not `.last_hidden_state`. Check `type(model)`
   and read the model's `.forward` signature once.

3. **`num_logits_to_keep=0` slices the full sequence**, not zero. Many
   HF causal-LM heads default to `num_logits_to_keep=0`, which is then
   used as `hidden[:, -0:, :]`. Because `-0 == 0` in Python, this is
   `hidden[:, 0:, :]` = the full sequence. So the default returns
   all-position logits, not just the last. (No action needed; just
   don't be confused.)

4. **Older HF tokenizer files (Qwen ≤ 2.5, llama-1)**: don't have
   `tokenizer.json`. Instead they ship two files:
   - `vocab.json`: flat `{token_string: id}`, where strings are
     GPT-2-byte-encoded (printable ASCII passes through; control bytes
     map to U+0100..U+0142).
   - `added_tokens.json`: flat `{name: id}` for special tokens.

   `build_tokenizer.c` must read BOTH and merge into the
   `tokenizer.bin` format the runtime expects. Total id range is
   `max(vocab.json ids, added_tokens.json ids) + 1` (HF tokenizers
   reserve special-token ids past `len(vocab)` and may leave gaps).
   For ids owned by `added_tokens.json`, emit an empty BPE entry so
   the merge engine never picks them as merge candidates. **Do not**
   need `merges.txt` for any byte-level BPE — the tiktoken trick
   (merge priority == vocab id) works as long as the vocab was built
   in merge order, which is true for all Qwen / Llama / GPT-2-family
   tokenizers.

5. **`forward_hook` payload shape varies**. For most simple
   `nn.Linear` / `nn.LayerNorm` modules the hook output is a single
   tensor. For attention blocks the output may be a tuple of
   `(attn_output, attn_weights, past_kv)`. Always check
   `isinstance(out, tuple)` and pick `out[0]`:

   ```python
   def hook(name):
       def f(_m, _i, out):
           t = out[0] if isinstance(out, tuple) else out
           captures[name] = t.detach().float().cpu().numpy()
       return f
   ```

6. **`torch_dtype=torch.bfloat16` on CPU is slow**. A 7B bf16 forward
   over L=25 tokens takes ~15–60 s on a modern CPU (depending on how
   well the BLAS path handles bf16). Don't be surprised. The whole
   point of this skill is the C reference; speed comes later.

7. **`model.config` vs `tokenizer.special_tokens_map`**: the eos used
   by `model.generate()` (= `model.generation_config.eos_token_id`) is
   not always the same as `tokenizer.eos_token_id`. For Dream they're
   both 151643; for some Qwen variants the generation eos is
   `<|im_end|>` (151645) while the tokenizer eos is `<|endoftext|>`
   (151643). The **C reference's stop condition** must match whatever
   the Python sampler actually stops on. Read the relevant
   `generation_utils.py` / `_sample` path.

8. **`trust_remote_code=True` loads model code from the model dir**.
   For models like Dream whose architecture isn't in mainline
   transformers, `modeling_<name>.py` lives in the HF model directory
   (alongside `config.json`) and is imported on load. To find the
   class definitions referenced by `dump_ref.py.template` (e.g.,
   `apply_rotary_pos_emb`, `repeat_kv`), use:

   ```python
   cls = type(model)
   modeling_mod = sys.modules[cls.__module__]
   apply_rotary_pos_emb = getattr(modeling_mod, "apply_rotary_pos_emb")
   repeat_kv             = getattr(modeling_mod, "repeat_kv")
   ```

9. **`do_sample=False` + `temperature=0.0` warning is benign**.
   HF prints "temperature is set to 0.0 but do_sample=False" but the
   generation_config in some shipped models has both set, so suppress
   or ignore the warning. The effective behaviour with `do_sample=False`
   is greedy, regardless of temperature.

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
With the starter `utils/json.h` this is just:

```c
json_doc_t* doc = json_open_file(path, &err);
json_val_t  root = json_root(doc);
json_val_t  lm   = json_get(root, "text_config");
if (json_is_null(lm)) lm = root;       // text-only model

int n_layers = (int)json_int_or(json_get(lm, "num_hidden_layers"), 32);
float theta  = (float)json_double_or(
    json_path(lm, "rope_parameters.rope_theta"), 10000.0);
```

`json_get` short-circuits safely through missing keys, so chained
accesses on absent paths yield the default rather than crashing.

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

#### End-to-end logit drift (the "1 ULP per kernel × N layers" rule)

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
patterns").

## Common pitfalls

- **PyTorch CPU SDPA defaults to FLASH**, not MATH, and FLASH gives
  different bf16 results. Force `SDPBackend.MATH` in oracle dumps AND
  in any end-to-end Python comparison run. See "PyTorch / HF gotchas".
- **Missing per-element bf16 round-trips**: causes 1–2 ULP drift per
  kernel even when "the math is right". See "Numerical drift /
  Per-element bf16 rounding patterns".
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
  `utils/bf16.h`).
- **RoPE yarn scaling**: yarn has an extra `mscale` multiplier on top of
  the rotation — easy to miss. Match the reference exactly.
- **RoPE freq formula**: `inv_freqs[k] = 1 / (base ** (2k / D))` for
  `k ∈ [0, D/2)`. With `partial_rotary_factor < 1.0` use `D = D_rot =
  head_dim * partial_rotary_factor`, NOT `head_dim`.
- **Tokenizer special tokens**: HF `tokenizer.json` has special tokens in
  `added_tokens` — make sure `build_tokenizer.c` includes them (the
  starter does this automatically from the JSON).
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
- **JSON parsing performance**: if you use `jsmn` (the starter does, in
  `utils/json.c`), **define `JSMN_PARENT_LINKS` before including it**.
  Without it, jsmn back-walks the entire token array on every `,` inside
  an object — O(N²) on container size. For a 250k-entry
  `tokenizer.json::model.vocab` that's ~40 s of pure overhead;
  with `JSMN_PARENT_LINKS` defined it drops to ~40 ms (1000× speedup).
  The cost is 4 extra bytes per token (~2 MB extra for the vocab).
- **Centralize JSON, don't re-invent it per caller**: it's tempting to
  write a one-off "good enough" substring-search JSON walker every time
  (`strstr` for the key, `strchr` for `:`, `atoi`). This breaks for any
  nested object with duplicate key names (every multi-modal config) and
  for any key whose value contains a literal `:`. Build one JSON reader
  (the starter's `utils/json` is ~500 LOC) and route everything through
  it — `safetensors.c` shard headers, `config.json`, `tokenizer.json`,
  and any future model registry files.

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

1. `./src-cpu/` exists and builds with `make` (no Python in the inference
   build path — only `tools/dump_ref.py` for regenerating the oracle).
2. `./<BIN> --prompt "<validation_prompt>" --max-tokens N` produces the
   same N tokens as the Python reference.
3. Every kernel is a plain C function in `kernels.{c,h}` with clear
   pointer-arity inputs / outputs.
4. `src-cpu/refs/` contains per-op oracle dumps usable by the Metal
   port for kernel correctness tests.
5. The `src-cpu/utils/` library (json, jsmn, utf8, bytebuf, iofile, bf16),
   the tokenizer + tokenizer-bin builder, the safetensors reader, and the
   per-kernel test harness are model-agnostic infrastructure the Metal
   port can copy / symlink rather than re-author.

When this contract is met, run the `port-c-to-metal` skill.
