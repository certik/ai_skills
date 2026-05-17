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

Take a working Python reference of an LLM (MLX, HF transformers,
PyTorch, etc.) and a HF-safetensors weight directory, and produce a
standalone `./src-cpu/` that runs the same model in plain C99 on a
single CPU core, with **bit-equivalent-or-very-close** output for the
same prompt under greedy decoding.

The output is **not** trying to be fast. It is the reference and the
spec. Speed comes later (see `port-c-to-metal` and `optimize-metal`).

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
high-performance Apple-GPU Metal implementation. The C reference is
also the starting point for ports to CUDA, ROCm, TPU, etc. (each its
own future skill).

## Prerequisites

- A working **Python reference implementation** of the target model.
  MLX-LM preferred (smaller surface, fewer dependencies). PyTorch / HF
  transformers acceptable.
- A **HF safetensors weight directory** locally (the `model.safetensors`
  or sharded `model-*-of-*.safetensors` + `config.json` + `tokenizer.json`).
- A **C99 compiler** (clang or gcc). `-O2` is the default; `-ffast-math`
  may be used cautiously **only if** it doesn't change per-kernel outputs
  beyond tolerance (see `references/precision-and-tolerance.md`).
- Python deps to run the reference + dump oracle arrays (tensors) (numpy,
  mlx / torch / transformers as required by the reference).

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
├── cache.{c,h}        # OPTIONAL: KV cache module (only for AR or
│                      # Fast-dLLM-style cached samplers; not needed for
│                      # the naive diffusion sampler)
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
├── refs-<sampler>/    # OPTIONAL: per-step E2E oracle for a second
│                      # sampler (e.g. Fast-dLLM block diffusion).
│                      # Contains per-(block,step) `x[]` snapshots.
└── tests/
    ├── load_ref.h         # .bin loader + bf16-aware err_check
    └── test_kernels.c     # one test fn per kernel; CLI-selectable
```

Plus the **only** remaining Python tools, kept outside `src-cpu/`:

```
./tools/
├── dump_ref.py               # per-op oracle for kernel tests
├── run_ref_<sampler>.py      # E2E oracle: prints generated ids only
└── dump_ref_<sampler>.py     # OPTIONAL per-step `x` snapshots (one
                              # file per sampler with a non-trivial
                              # orchestration to debug, e.g. Fast-dLLM)
```

Note: `src-cpu/` is fully Python-free. The inference build needs only
clang + libc + libm. Python is used only to (a) regenerate oracle dumps
in `refs/` for the per-kernel correctness tests, and (b) produce a
reference token sequence for end-to-end token-equivalence checks.

## Starter files (in this skill's `starter/`)

The starter ships two kinds of files. The **drop-in** files contain no
model knowledge — copy them verbatim:

- `utils/{bf16.h, jsmn.h, json.{c,h}, utf8.{c,h}, bytebuf.{c,h}, iofile.{c,h}}`
  — Python-stdlib equivalents in C (mmap, JSON, UTF-8, bytebuf, bf16
  conversion). Treat as a fixed dependency; do not leak model knowledge
  into it. For the rationale, available functions, and the critical
  `JSMN_PARENT_LINKS` define (without which `tokenizer.json` parsing is
  1000× slower), see `references/utils-library.md`.
- `safetensors.{c,h}` — HF safetensors mmap loader. Handles single-file
  and sharded layouts, BF16/F16/F32/U8/I8/U16/I16/U32/I32/U64/I64.
- `tokenizer.{c,h}` — byte-level BPE engine. Reads `tokenizer.bin`. The
  BPE merging algorithm is generic; only the **pre-tokenizer regex** in
  `pretokenize()` may need swapping for non-Qwen / non-Llama families
  (`references/tokenizer.md`).
- `build_tokenizer.c` — converts HF `tokenizer.json` → `tokenizer.bin`
  in pure C. Drop-in for any byte-level BPE family (Qwen, Llama 3,
  GPT-2, o200k, gpt-oss). For sentencepiece / unigram or older HF
  formats (Qwen ≤ 2.5, Llama 1), see `references/tokenizer.md`.
- `tests/load_ref.h` — `.bin` loader + bf16-aware `err_check`.

The **template** files are starting points you customize per model:

- `main.c.template` — CLI skeleton: argparse, `config.json` loader (via
  `json_path` so multi-modal `text_config` nesting is one line), weight
  load, tokenizer load, prompt encode, prefill, AR decode (with live
  text streaming + final summary), stats.
- `kernels.h.template` — kernel signature pattern (one C function per
  distinct math op). Includes a checklist of common modern-architecture
  ops (output gate, partial RoPE, q/k_norm, SSM step, etc.).
- `kernels.c.template` — naive scalar reference kernels plus commented
  snippets for two common precision-reduction (quantization) formats
  (MXFP4 and MLX 8-bit affine).
- `Makefile.template` — `make` builds `./<BIN>`; `make tokenizer`
  builds the C tokenizer-bin generator; `make tests` runs the
  per-kernel correctness checks.
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
   `rope_theta` / `rope_scaling`, etc. For multi-modal models the
   relevant LM dims live in a nested `text_config` — see
   `references/pytorch-gotchas.md` ("Multi-modal config.json parsing").

4. Identify model-specific details by reading the reference and asking
   the user when unclear:
   - **Precision-reduction format**: bf16? f16? MXFP4 (e2m1+e8m0)? FP8?
     Per-channel scales? Group size? → `references/quantization.md`
   - **RoPE flavor**: standard? llama-style? yarn? long-rope? →
     `references/architectures.md`
   - **Attention variant**: MHA / GQA / MQA; sliding window every N
     layers? Attention sinks? Output gate? Partial RoPE? Per-head
     q_norm / k_norm? → `references/architectures.md`
   - **Activation**: SwiGLU / GeGLU / GELU? Clamping (e.g., gpt-oss
     clamps gate at 7.0)?
   - **Normalization**: RMSNorm? LayerNorm? `eps`?
   - **Hybrid linear-attention / SSM** layers? (Mamba, GatedDeltaNet,
     RWKV, ...) → `references/architectures.md`
   - **Bidirectional** (`is_causal=False`, discrete-diffusion / masked
     LM)? → `references/samplers.md`
   - **Chat template / prompt format**: which one (chatml, harmony,
     llama instruct, mistral, ...)? Special token IDs?
   - **Stop tokens** for AR decode loop. Note that
     `model.generation_config.eos_token_id` ≠ `tokenizer.eos_token_id`
     in some models — see `references/pytorch-gotchas.md`.

5. Identify the **safetensors array names** for each weight by reading
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
   These contain no model knowledge — only the tokenizer's
   pre-tokenizer regex may need swapping later (see
   `references/tokenizer.md`).

2. Build and run the C tokenizer-bin generator on your model's
   tokenizer files:
   ```
   make tokenizer MODEL_DIR=/path/to/model
   ```
   This produces `src-cpu/tokenizer.bin`. For older HF formats
   (`vocab.json` + `added_tokens.json` instead of `tokenizer.json`) or
   sentencepiece families, see `references/tokenizer.md`.

3. Author `<chattmpl>.{c,h}` (e.g., `harmony.{c,h}`, `qwen_chat.{c,h}`,
   `llama3_chat.{c,h}`) that builds the prompt token sequence. **Tip**:
   have the Python reference output the numeric token IDs it produces
   for the same `(system, user, reasoning)` tuple, and write the C
   builder to match byte-for-byte. Add a small test that compares C
   and Python token sequences.

4. Author `main.c` from `main.c.template`:
   - argparse `--prompt`, `--system`, `--max-tokens`, `--model`
   - Load `config.json` via `json_open_file` + `json_get` +
     `json_int_or` (the template handles multi-modal `text_config`
     nesting in one line)
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
   - [ ] Program prints the list of arrays found in safetensors and
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
   intermediate arrays at every op boundary. Use a small binary
   format with a magic header:
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
   is correct. See `references/pytorch-gotchas.md`.

6. **For ops not natural to hook** (RoPE'd q/k, raw SDPA output, fused
   silu*up): the PyTorch `forward_hook` only fires at module
   boundaries. Add a small **"manual layer-0 replay"** section that
   reuses the captured `q_proj_0` etc., re-applies RoPE / SDPA / silu
   manually, and saves the intermediates. The starter template shows
   the MLX equivalent (which always requires manual walking — MLX has
   no hook mechanism).

7. **Commit**: `tools/dump_ref.py + initial oracle dump`.

### Phase 4: Implement kernels one at a time

This is the bulk of the work. For each distinct math op in the
reference's forward pass, **in the order it appears**, walk through 4a–4e.
Walking in forward-pass order lets you chain dumps later: layer 0
input → layer 0 output → layer 1 input → ...

#### Two rules that govern every kernel

- **Structure kernels EXACTLY as the reference.** If the reference does
  `gate, up = self.gate_up_proj(x); mid = self.swiglu(gate, up)`, you
  have **two** kernels: `gate_up_linear` and `swiglu`. Even if fusing
  them would be faster, that's a job for `optimize-metal`. If, however,
  the reference's reduced-precision linear is fundamentally a "gather
  expert weights + dequant + matmul" combined op (as in MLX's
  `mxfp4_qmm`), follow that — keep dequant inline with the matmul. Reuse
  the same kernel for different call sites of the same op (e.g., one
  `linear_bf16` used for q_proj / k_proj / v_proj / o_proj / router /
  lm_head). **One C function per distinct math op, not per call site.**

- **Maintain reference order even when it's "wasteful".** RoPE on Q
  and RoPE on K can run in either order in principle, but the reference
  picks one — do the same. KV cache writes happen in a specific
  position relative to RoPE — match. Residual additions happen at a
  specific spot — match.

See `references/architectures.md` for a tour of common modern variants
(GQA, attention sinks, output gate, partial RoPE, per-head q_norm/k_norm,
MoE+shared expert, sliding window, hybrid SSM, bidirectional).

#### 4a. Define the kernel signature in `kernels.h`

See `starter/kernels.h.template` for the full signature pattern and a
worked checklist of common modern-architecture ops (one function per
distinct op: `embed_gather`, `rmsnorm`, `linear`, `rope_yarn_inplace`,
`sdpa_with_sinks`, `topk_softmax`, `mxfp4_linear_gather`, `swiglu`,
`expert_mix`, `residual_add`, `argmax`, ...).

**Signature rules**:
- One function per *distinct* op (not per call site).
- Plain C, no shim, no buffer/pipeline abstraction. Pointers + sizes.
- Native model precision in storage (bf16 if model is bf16); fp32 in
  accumulators where it changes the result.
- **No fusion** beyond what the reference itself does.
- **No prefill-vs-decode special paths**. Same kernel for `Lq=1` and
  `Lq>1`. The Metal port may split them later.

#### 4b. Implement the kernel in `kernels.c`

Naive scalar implementation. Example:

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
- One layer of nested loops per logical array index.
- fp32 accumulator for any reduction.
- Use `bf16_to_f32` / `f32_to_bf16` (in `utils/bf16.h`); do not
  introduce arch-specific intrinsics.
- Don't try to be clever. The Metal port will rearrange this anyway.

For reduced-precision linears (MXFP4, MLX affine) see
`references/quantization.md`.

For **per-element bf16 round-trips that mirror the reference's hidden
`.to(dtype)` casts** — the most common source of small numerical
mismatches — see `references/precision-and-tolerance.md`. This is the
#1 cause of kernels that look right but fail tolerance by 1–4 ULPs.

#### 4c. Write a per-kernel correctness test

In the consolidated `tests/test_kernels.c` (one function per kernel,
registered in the `TESTS[]` table):

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
Reductions can drift more — accept up to `~K * eps_per_op` for large
K. For a magnitude-aware tolerance table and the
`tol = abs + rel * |want|` formula used by the starter, see
`references/precision-and-tolerance.md`.

#### 4d. Iterate the kernel until it passes

If the test fails:
- Print the first few mismatching elements.
- Check array layout / strides against the reference.
- Check accumulator precision. Move to fp32 if you were using bf16.
- Check special cases (sinks, sliding window, the +1 in SwiGLU's
  `(u+1)`, the clamping, the scale on RoPE for yarn, etc.).
- **Check for missing bf16 round-trips** — see
  `references/precision-and-tolerance.md`.
- **Compare math literally with the reference Python**. Often a hidden
  scaling factor or transpose is the culprit.

#### 4e. Commit per-kernel

```
src-cpu: <kernel_name> + test (max|d|=<tol> vs ref)
```

This makes rollback trivial and gives you a clear log of progress.

#### 4f. Iterate over all ops

Walk the reference's forward pass and repeat 4a–4e for every distinct
op.

### Phase 5: Stitch into a forward() function

1. In `main.c`, write `forward(int q_off, int Lq)` (or however your
   model parameterizes it) that:
   - For each layer, calls every kernel in reference order, threading
     activations through workspace buffers.
   - Manages per-layer KV cache (mmap'd or malloc'd, sized to
     `MAX_CTX * N_KVHEADS * HEAD_DIM * sizeof(bf16)`).
   - For **non-causal / diffusion-style models**, the simplest reference
     has no KV cache: `forward(L)` runs over the entire sequence each
     call. But "diffusion = no cache" is not a law — see
     `references/samplers.md` for the cached Fast-dLLM variant (much
     faster, also the realistic deployment target).

2. Add the AR decode loop (or diffusion sampler — see
   `references/samplers.md` for the diffusion / Fast-dLLM block
   diffusion drivers):
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
     argmax is preserved. See
     `references/precision-and-tolerance.md` ("End-to-end logit
     drift").

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
     PyTorch CPU's default `SDPBackend.FLASH_ATTENTION` for bf16 —
     see `references/pytorch-gotchas.md`). Force the deterministic /
     textbook backend in this Python helper.
   - Some sampler algs (e.g. `alg=origin` in Dream's diffusion) draw
     from `torch.rand` and can't be matched in C. Pick a deterministic
     alg (Dream: `alg=entropy`).

6. If tokens don't match: dump intermediates from the C run (e.g., add
   a `--dump-c-refs` flag that writes the same `refs/*.bin` files as
   the Python oracle does), and binary-diff against the Python refs.
   The first divergence localizes the bug. Most often: a missing bf16
   round-trip in one of the per-element kernels — see
   `references/precision-and-tolerance.md`.

7. **Budget**: naive scalar bf16 on a single CPU core runs the
   bf16-7B-class forward at ~1–2 GFLOPS/s after `bf16_to_f32` overhead.
   For e.g. Dream 7B at L=25 that's ~85 s/forward, ~15 min for an
   8-step diffusion. **Don't be surprised** if a full diffusion or AR
   loop takes 10+ minutes per run — it's the price of "no SIMD, no
   threads, naive scalar". `port-c-to-metal` and `optimize-metal` will
   cut this to under a second.

8. **Commit**: `src-cpu: end-to-end forward + decode (N/N tokens match ref)`.

### Phase 6: Acceptance

Before declaring done, walk `references/pitfalls.md` for the
pre-flight checklist. Then check:

- [ ] `./<BIN> --prompt "<validation_prompt>" --max-tokens N` produces
      the **same N token IDs** as the Python reference under greedy
      decoding.
- [ ] All per-kernel tests in `tests/` pass.
- [ ] Build is `make src-cpu` with no warnings (`-Wall -Wextra`).
- [ ] `src-cpu/` is standalone (no link to the Python reference or to
      any external runtime besides libc and libm).

When all boxes are ticked, the skill is done.

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

1. `./src-cpu/` exists and builds with `make` (no Python in the
   inference build path — only `tools/dump_ref.py` for regenerating
   the oracle).
2. `./<BIN> --prompt "<validation_prompt>" --max-tokens N` produces
   the same N tokens as the Python reference.
3. Every kernel is a plain C function in `kernels.{c,h}` with clear
   pointer-arity inputs / outputs.
4. `src-cpu/refs/` contains per-op oracle dumps usable by the Metal
   port for kernel correctness tests.
5. The `src-cpu/utils/` library (json, jsmn, utf8, bytebuf, iofile,
   bf16), the tokenizer + tokenizer-bin builder, the safetensors
   reader, and the per-kernel test harness are model-agnostic
   infrastructure the Metal port can copy / symlink rather than
   re-author.

When this contract is met, run the `port-c-to-metal` skill.

## References

Deep dives are kept in `references/`. Read them on demand:

| File | Read when |
|---|---|
| `references/precision-and-tolerance.md` | Writing or debugging any kernel; per-kernel test fails by 1–4 ULPs; choosing tolerance; argmax-stability acceptance; deciding on `-ffast-math`. |
| `references/architectures.md` | Phase 1 reconnaissance on non-vanilla architectures; implementing GQA / sinks / sliding window / partial RoPE / q,k_norm / output gate / MoE / shared expert / SSM (Mamba / GatedDeltaNet / RWKV) / bidirectional attention. |
| `references/quantization.md` | The reference uses MXFP4 (gpt-oss) or MLX-style affine 4-bit / 8-bit precision reduction. |
| `references/samplers.md` | Phase 5 driver loops; discrete-diffusion / masked LM samplers; Fast-dLLM-style block diffusion with DualKVCache; supporting multiple samplers in the same binary. |
| `references/mlx-gotchas.md` | The reference is MLX (no `forward_hook`; `mx.eval`; `sanitize`; `cast_predicate`; `quant_predicate`; `mx.fast.rope`). |
| `references/pytorch-gotchas.md` | The reference is PyTorch / HF (FLASH vs MATH SDPA; AutoModel wrappers; `num_logits_to_keep`; `trust_remote_code`; multi-modal `config.json`). |
| `references/tokenizer.md` | Older HF tokenizer format (`vocab.json` + `added_tokens.json`); sentencepiece / unigram families; pre-tokenizer regex; chat templates. |
| `references/pitfalls.md` | Phase 6 pre-flight checklist; debugging mysterious failures; quick "did I forget anything?" sanity pass. |
| `references/utils-library.md` | Understanding the `utils/` design philosophy; adding helpers; the `JSMN_PARENT_LINKS` performance trap. |
