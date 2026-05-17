---
name: port-c-to-metal
description: >
  Port a working pure-C CPU reference LLM implementation (produced by the
  build-c-reference skill, living in ./src-cpu/) to an Apple-GPU Metal
  implementation in ./src-metal/. The port introduces a small Metal C shim
  (gpu_ctx / gpu_buf / gpu_pipeline / gpu_cmdbuf), translates
  each C kernel to a 1:1 Metal Shading Language kernel (initially naive,
  1 thread per output element), and stitches the kernels together with a
  Metal-dispatching main.c that mirrors the C reference's forward()
  schedule. Correctness is validated kernel-by-kernel against C-reference
  oracle dumps, then end-to-end against the C reference's generated
  tokens. This skill does NOT optimize — that's the next skill,
  optimize-metal. The goal here is correctness on GPU. Triggers: port to
  metal, metal port, port c to gpu, port to apple gpu, c to metal,
  metal naive, gpu correctness.
---

# port-c-to-metal — Apple-GPU Metal port of a C LLM reference

Take a working pure-C CPU reference (`./src-cpu/`, produced by
`build-c-reference`) and produce a working Apple-GPU Metal port
(`./src-metal/`) that generates the same tokens as the C reference for the
same prompt.

This skill is **only** about correctness on the GPU. Optimization is
the job of `optimize-metal`. Kernels here can be the most naive possible
(one thread per output element); we will iterate them up later.

## When to use

After `build-c-reference` has produced a working `./src-cpu/`. Use this
skill to:

- Set up the Metal infrastructure (shim, MSL JIT compilation,
  command-buffer dispatch).
- Translate each C kernel 1:1 into a Metal kernel.
- Verify each Metal kernel against the C reference (or the C-reference
  oracle dumps).
- Get end-to-end GPU inference matching C-reference tokens.

**Do not** use this skill to optimize. Resist all temptation. Naive 1:1
ports first; optimize after.

## Pipeline

```
build-c-reference  →  port-c-to-metal  →  optimize-metal
                       (this skill)
```

## The C reference is the spec

Throughout this skill: **`./src-cpu/` is the ground truth**, not the
original Python reference. The Python reference may already differ from
the C reference by small numerical amounts (different reduction orders,
fp32 vs bf16 accumulators, etc.); that's expected and was tolerated in
`build-c-reference`. The Metal port must match the C reference within
tolerance, NOT the Python reference. Always validate kernels by feeding
inputs through C first, then comparing Metal outputs to C outputs.

If you find yourself comparing Metal output to Python output, stop — you
are creating a moving target.

## Prerequisites

- `./src-cpu/` exists, builds, and passes end-to-end token match.
- macOS / Apple Silicon (M1+).
- `clang` with `-framework Metal -framework Foundation -framework MetalKit`.
- The same model directory used by `src-cpu`.

## Inputs to gather

1. Path to `./src-cpu/` (default: sibling of this skill's working dir).
2. Path to the HF model directory (same as `src-cpu` expects).
3. **Binary name** for the Metal executable. Typically the same root
   as the CPU binary but without the `-cpu` suffix (e.g., `llama3-cpu`
   → `llama3`) so the two coexist.
4. The validation prompt + expected token IDs from the C reference.

## Output layout

```
./src-metal/
├── main.c                  # mirrors src-cpu/main.c, dispatches Metal kernels via shim
├── metal_shim.h            # C API for Metal (drop-in from starter/)
├── metal_shim.m            # Obj-C implementation (drop-in from starter/)
├── kernel_concat.{c,h}     # concatenate kernels/*.metal into one MSL source (drop-in)
├── kernels/                # one .metal file per C kernel — names match the C source
│   ├── embed_gather.metal
│   ├── rmsnorm.metal
│   ├── linear.metal
│   ├── rope.metal
│   ├── sdpa.metal
│   ├── ...
│   └── argmax.metal
├── safetensors.{c,h}       # symlink or copy from src-cpu
├── tokenizer.{c,h}         # symlink or copy from src-cpu
├── tokenizer.bin           # symlink or copy
├── <chattmpl>.{c,h}        # symlink or copy
└── (Makefile targets added to the root Makefile, not a new one — see Phase 1.5)
```

Plus `tests/test_metal_<kernel>.c` for per-kernel validation against the
C reference output (optional; see `references/kernel-patterns.md`).

The exact kernel list depends on the model. A dense transformer needs
roughly `embed_gather / rmsnorm / linear / rope / sdpa / swiglu /
residual_add / argmax`. An MoE adds `topk_softmax`, a gather-linear
variant for expert weights, and `expert_mix`.

## Starter files (in this skill's `starter/`)

Drop-in (copy verbatim):

- `metal_shim.h` / `metal_shim.m` — minimal C-callable Metal wrapper:
  device init, MSL JIT compilation, pipelines (one per kernel name),
  buffers (`gpu_buf_new`, `gpu_buf_new_from`,
  `gpu_buf_wrap_nocopy` for zero-copy mmap weights), command buffers
  with batched dispatches.
- `kernel_concat.{c,h}` — concatenates the per-kernel `.metal` files into
  a single MSL source string at startup; lets you keep one `.metal` file
  per kernel for readability without paying for many JIT compilations.

Customize:

- `main.c.template` — naive 1:1 skeleton for a dense AR transformer
  (RMSNorm + GQA + RoPE + SwiGLU + LM head): `cfg_t` + JSON loader,
  per-tensor weight buffers, params arena, `d_*` dispatch helpers,
  `forward(q_off, Lq)`, prefill + decode loop. Distilled from real
  ports; search the file for `PLACEHOLDER` to find what to adapt.
- `Makefile.template` — standalone Makefile, only useful if the repo has
  no existing root Makefile. Usually you'll extend the root Makefile
  instead (Phase 1.5).
- `rmsnorm.metal.template`, `linear_naive.metal.template` — two worked
  examples of the "naive 1:1" Metal kernel style. Use as templates for
  your own kernels.

## Procedure

### Phase 1: Set up the Metal infrastructure

1. `mkdir -p src-metal/kernels src-metal/tests src-metal/utils`. Copy
   drop-ins: `metal_shim.{h,m}`, `kernel_concat.{c,h}`.

2. Bring shared files in from `src-cpu`. **`cp -a` is easier than `ln -s`** —
   symlinks have a relative-path gotcha (see `references/pitfalls.md`).
   If you use symlinks, run them from inside `src-metal/`:

   ```sh
   cd src-metal
   for f in safetensors.{c,h} tokenizer.{c,h} <chattmpl>.{c,h} tokenizer.bin; do
       ln -sf ../src-cpu/$f
   done
   cd utils
   for f in ../../src-cpu/utils/*.{c,h}; do ln -sf "$f" $(basename "$f"); done
   ```

   Verify with `head -1 src-metal/safetensors.h`.

3. Author a minimal `src-metal/kernels/noop.metal` containing a 1-line kernel:

   ```metal
   kernel void noop_inc(device uint* x [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
       x[id] += 1;
   }
   ```

4. Author `tests/test_shim.c` that **compiles every `.metal` file you
   plan to write AND looks up every named kernel function**, in addition
   to dispatching `noop_inc`. This catches MSL syntax errors and
   `[[buffer(K)]]` index mistakes in <1s — *before* you spend 30s+
   building `main.c` and waiting for a model load. As you add more
   kernels in Phase 2, append them to the `paths[]` and `kernels[]`
   arrays in this test; rerun before every commit.

5. Build with the Metal frameworks and run the test. It must pass before
   you touch real kernels.

6. **Commit**: `src-metal: metal shim + smoke test`.

### Phase 1.5: Makefile — extend, don't replace

If the repo already has a root Makefile (typical after `build-c-reference`),
**add new metal targets to it** rather than creating a separate
`src-metal/Makefile`. Both binaries should coexist (e.g. `qwen35-cpu`
and `qwen35`). Pattern:

```make
# ---------- Apple-GPU Metal port (naive correctness build) ------------------
METAL_BIN        := <BIN>
METAL_CFLAGS     := -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Isrc-metal
METAL_OBJCFLAGS  := -O2 -Wall -Wextra -Wno-unused-parameter -fobjc-arc -ObjC -Isrc-metal
METAL_LDFLAGS    := -lm -framework Foundation -framework Metal -framework MetalKit

METAL_C_SRCS := src-metal/main.c src-metal/kernel_concat.c \
                src-metal/safetensors.c src-metal/tokenizer.c \
                src-metal/<chattmpl>.c \
                src-metal/utils/json.c src-metal/utils/iofile.c \
                src-metal/utils/bytebuf.c src-metal/utils/utf8.c
METAL_M_SRCS := src-metal/metal_shim.m
METAL_OBJS   := $(METAL_C_SRCS:.c=.o) $(METAL_M_SRCS:.m=.o)

metal: $(METAL_BIN)
$(METAL_BIN): $(METAL_OBJS)
	$(CC) $^ -o $@ $(METAL_LDFLAGS)
src-metal/%.o: src-metal/%.c
	$(CC) $(METAL_CFLAGS) -c $< -o $@
src-metal/%.o: src-metal/%.m
	$(CC) $(METAL_OBJCFLAGS) -c $< -o $@

METAL_TEST_BINS := src-metal/tests/test_shim
metal-tests: $(METAL_TEST_BINS)
	@for t in $^; do echo "==> $$t" && ./$$t || exit 1; done
src-metal/tests/test_%: src-metal/tests/test_%.c src-metal/metal_shim.o src-metal/kernel_concat.o
	$(CC) $(METAL_CFLAGS) $^ -o $@ $(METAL_LDFLAGS)

metal-run: $(METAL_BIN) src-cpu/tokenizer.bin
	./$(METAL_BIN) --prompt "<validation_prompt>" --max-tokens N --model $(MODEL_DIR)
```

Extend the existing `clean:` rule and `.gitignore` rather than
introducing new ones.

### Phase 2: Port each C kernel 1:1 to MSL

For each kernel in `src-cpu/kernels.{c,h}`, in **forward-pass order**:

#### 2a. Write the MSL kernel

Naive translation: **one thread per output element**, no tiling, no
threadgroup memory, no SIMD-group ops, no vector loads. Pass scalar parameters via a
small "dims" `constant` struct in a single `[[buffer(K)]]` slot.

Example (RMSNorm) — note how `eps` is bundled into the dims struct:

```metal
// rmsnorm.metal
#include <metal_stdlib>
using namespace metal;

struct rmsnorm_params { uint M; uint D; float eps; };

kernel void rmsnorm_bf16(device const bfloat*        X       [[buffer(0)]],
                         device const bfloat*        W       [[buffer(1)]],
                         device bfloat*              Y       [[buffer(2)]],
                         constant rmsnorm_params&    p       [[buffer(3)]],
                         uint                        m       [[thread_position_in_grid]])
{
    if (m >= p.M) return;
    uint D = p.D;
    const device bfloat* xp = X + m * D;
    device bfloat*       yp = Y + m * D;

    float sq = 0.0f;
    for (uint d = 0; d < D; ++d) {
        float v = float(xp[d]);
        sq += v * v;
    }
    float rrms = rsqrt(sq / float(D) + p.eps);
    for (uint d = 0; d < D; ++d) {
        yp[d] = bfloat(float(xp[d]) * rrms * float(W[d]));
    }
}
```

`starter/rmsnorm.metal.template` and `starter/linear_naive.metal.template`
show this style end-to-end. See `references/msl-tips.md` for the full
"naive style" checklist, threadgroup-size guidance, and Apple-GPU facts
(bf16 support, page size, JIT speed).

#### 2b. Wire the dispatch in main.c

For each Metal kernel, you need three things at the call site:

1. A `gpu_pipeline*` looked up once at startup
   (`gpu_pipeline_for(ctx, "rmsnorm_bf16", &err)`).
2. A per-dispatch slice in a **params arena** Metal buffer (see below).
   **Do NOT reuse a single param buffer across dispatches in the same
   cmdbuf** — the GPU reads buffer contents at *execution* time
   (post-commit), so all dispatches would see the last value written.
   Full discussion: `references/pitfalls.md`.
3. A `gpu_cmdbuf_dispatch(...)` call with `grid = (M, 1, 1)`,
   `tg = (1, 1, 1)` (naive — one thread per output row).

**Params-arena pattern** (drop into your `main.c`):

```c
static gpu_buf* g_params_buf;
static size_t   g_params_off, g_params_cap;

static gpu_arg_buf push_params(const void* p, size_t n) {
    size_t off = (g_params_off + 15u) & ~(size_t)15;   // 16-byte align
    if (off + n > g_params_cap) { fprintf(stderr, "arena overflow\n"); exit(1); }
    memcpy((char*)gpu_buf_contents(g_params_buf) + off, p, n);
    g_params_off = off + n;
    return (gpu_arg_buf){ g_params_buf, off };
}
static void reset_params(void) { g_params_off = 0; }

// In alloc_buffers():
g_params_cap = 4u << 20;   // 4 MB headroom
g_params_buf = gpu_buf_new(ctx, g_params_cap);

// At the top of forward():  reset_params();
// At each dispatch site:
//   struct { uint32_t M, D; float eps; } p = { Lq, H, rms_eps };
//   gpu_arg_buf args[] = { WARG(x_buf), TARG("...weight"), WARG(h_buf), push_params(&p, sizeof p) };
//   gpu_cmdbuf_dispatch(cb, pso_rmsnorm, args, 4, Lq, 1, 1, 1, 1, 1);
```

Budget: ~40 layers × ~30 dispatches × ~64 B ≈ 80 KB per forward(). 4 MB
is comfortable headroom.

If you call `gpu_cmdbuf_commit_wait` *inside* `forward()` (e.g. to
host-sync between a prefill chunk and a per-token decode step), you
can also `reset_params()` at that point — once the GPU has finished
the previous cmdbuf, those arena slots are free to overwrite. Simpler
in practice is to just reset once at the top of `forward()` and size
the arena for the whole pass.

#### 2c. Validate the kernel against the C reference

The cleanest validator is to have the C reference run with a `--dump`
flag that writes every intermediate it produces to `src-metal/refs/`,
then a small per-kernel test that loads inputs from the dump, runs the
Metal kernel, and compares to the dumped output. Pattern + tolerance
in `references/kernel-patterns.md`.

If the kernel fails:
- Compare element-by-element with the **C** reference, not the Python
  ref. We only care that GPU matches CPU within tolerance.
- Common bugs: wrong grid size, `[[buffer(K)]]` index doesn't match
  the dispatch arg order, `bfloat` vs `bfloat16` type-name drift,
  stride confusion in 3D buffers. See `references/pitfalls.md`.

#### 2d. Commit per kernel

```
src-metal: <kernel_name> 1:1 metal port + test
```

### Phase 3: Stitch into a Metal main.c

1. Author `src-metal/main.c`. Structure: same as `src-cpu/main.c` but
   every C-function call becomes a `gpu_cmdbuf_dispatch(...)`.

2. **Weights: allocate one Metal buffer per tensor** via
   `gpu_buf_new_from(ctx, tensor.data, tensor.nbytes)` at load time.
   Zero-copy shard wrapping is almost always blocked by Metal's
   offset-alignment rule (this is the #1 silent-failure trap — see
   `references/pitfalls.md`). Total RAM usage ≈ model size; on Apple
   Silicon unified memory this is what `mmap` would page in anyway.
   After all tensors are copied, you can `st_close(arch)` to free the
   mmap virtual range if you want.

   Convenient pattern: index per-tensor buffers by
   `(t - st_at(arch, 0))` so `T_arg(t)` returns
   `{ g_tensor_bufs[idx], 0 }` — always offset 0, always naturally
   aligned.

3. Allocate workspace buffers (`x_buf`, `h_buf`, KV caches, etc.) via
   `gpu_buf_new(ctx, size)`. They live in shared (unified) storage and
   are directly host-readable via `gpu_buf_contents` — useful for the
   host-side bookkeeping ops below.

4. `forward(q_off, Lq)`:
   - Open a `gpu_cmdbuf`. Reset the params arena.
   - For each kernel, push params and dispatch.
   - Commit + wait at the end of `forward()`.

   For ops that mix GPU compute with **host-side scalar / bookkeeping
   work** (e.g. KV-cache `memcpy` at the right offset, conv1d state
   shift, one-off scalar broadcasts), the naive pattern is:
   commit_wait the current cmdbuf, do the host work on
   `gpu_buf_contents(...)`, open a fresh cmdbuf, continue. Specific
   templates for stateful / SSM / MoE kernels in
   `references/kernel-patterns.md`.

5. AR loop:
   - Embed prompt → `forward(0, Lp)` → argmax → emit
   - For each token: embed → `forward(Lp + i, 1)` → argmax → emit →
     break on stop.

6. Run on the validation prompt. Compare generated tokens with the C
   reference. They should match.

7. If they don't match: dump the layer-0 intermediates from BOTH the
   Metal run and the C run for the same input, and binary-diff. The
   first divergent kernel is the bug.

8. **Commit**: `src-metal: end-to-end metal forward + AR decode (N/N tokens match src-cpu)`.

### Phase 4: Acceptance

- [ ] `make <BIN>` builds `./<BIN>` with `-Wall -Wextra` clean.
- [ ] `./<BIN> --prompt "<validation_prompt>" --max-tokens N` runs on
      the GPU (`metal device: <Apple ...>` printed at startup).
- [ ] Generated N tokens match the C reference (`./src-cpu/<BIN>`)
      exactly.
- [ ] Phase-1 smoke test (`metal-tests`) still passes (all kernels
      compile + look up).
- [ ] Per-kernel tests in `tests/` pass — **optional**. Skip them if
      end-to-end token match passes and your iteration loop is < 2 min
      (rebuild + run). Only worth adding when you actually need to
      bisect a divergence.

When all the required boxes are ticked, the skill is done.

## References

Read the relevant file when you hit the situation it covers; don't
preload them all.

- **`references/pitfalls.md`** — silent-failure traps and weird-symptom
  lookup: per-tensor offset alignment, param buffer reuse, symlink
  paths, KV cache offsets, etc. Read this first if a kernel "passes"
  but end-to-end output is garbage.
- **`references/msl-tips.md`** — MSL / Apple-GPU specifics: the
  naive-style checklist, threadgroup sizing, command-buffer strategy,
  bf16 support, Apple Silicon facts.
- **`references/kernel-patterns.md`** — recurring shapes for specific
  kernel types: stateful (conv1d / SSM), host-side scalar broadcasts,
  MoE down_proj reshape, the `--dump` correctness scaffolding,
  tolerance discussion, suggested commit log.

## Hand-off to optimize-metal

The hand-off contract:

1. `./src-metal/` exists, builds with `make <BIN>`.
2. `./src-metal/<BIN> --prompt "<validation_prompt>" --max-tokens N` runs on
   the GPU and produces the same N tokens as `./src-cpu/<BIN>`.
3. Every kernel is one `.metal` file in `src-metal/kernels/` with a naive
   1-thread-per-output implementation.
4. Per-kernel correctness tests exist in `tests/` (or the `--dump`-based
   pattern is wired up so they can be added quickly).
5. The C reference (`src-cpu`) remains the ground truth for kernel
   outputs and end-to-end tokens. Do not modify it during this skill.

When this contract is met, run the `optimize-metal` skill.
