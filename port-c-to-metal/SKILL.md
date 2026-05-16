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
├── metal_shim.h            # C API for Metal (drop-in)
├── metal_shim.m            # Obj-C implementation (drop-in)
├── kernel_concat.{c,h}     # concatenate kernels/*.metal into one MSL source (drop-in)
├── kernels/                # one .metal file per C kernel
│   ├── embed_gather.metal
│   ├── rmsnorm.metal
│   ├── linear.metal
│   ├── rope.metal
│   ├── sdpa.metal
│   ├── topk_softmax.metal
│   ├── mxfp4_linear_gather.metal
│   ├── swiglu.metal
│   ├── expert_mix.metal
│   ├── residual_add.metal
│   └── argmax.metal
├── safetensors.{c,h}       # symlink or copy from src-cpu
├── tokenizer.{c,h}         # symlink or copy from src-cpu
├── tokenizer.bin           # symlink or copy
├── <chattmpl>.{c,h}        # symlink or copy
└── Makefile
```

Plus `tests/test_metal_<kernel>.c` for per-kernel validation against the
C reference output.

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

- `Makefile.template`
- `main.c.template` — same flow as `src-cpu/main.c` but uses Metal
  buffers, pipelines, and a command buffer per forward() pass.
- `kernels/<kernel>.metal.template` — one naive MSL kernel as a worked
  example.

## Procedure

### Phase 1: Set up the Metal infrastructure

1. `mkdir -p src-metal/kernels src-metal/tests src-metal/utils`. Copy
   drop-ins: `metal_shim.{h,m}`, `kernel_concat.{c,h}`.

2. Bring shared files in from src-cpu. **`cp -a` is easier than `ln -s`** —
   symlinks have a relative-path gotcha (the path is relative to the
   symlink's location, not the cwd you ran `ln` from). If you must use
   symlinks, run them from inside `src-metal/`:

   ```sh
   cd src-metal
   for f in safetensors.{c,h} tokenizer.{c,h} qwen_chat.{c,h} tokenizer.bin; do
       ln -sf ../src-cpu/$f
   done
   cd utils
   for f in ../../src-cpu/utils/*.{c,h}; do ln -sf "$f" $(basename "$f"); done
   ```

   (Verify with `head -1 src-metal/safetensors.h` — if you get "No such
   file", the link is broken.)

3. Author a minimal `src-metal/kernels/noop.metal` containing a 1-line kernel:
   ```metal
   kernel void noop_inc(device uint* x [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
       x[id] += 1;
   }
   ```

4. Author a `tests/test_shim.c` that **compiles every `.metal` file you
   plan to write AND looks up every named kernel function**, in addition
   to dispatching `noop_inc`. This catches MSL syntax errors and
   `[[buffer(K)]]` index mistakes in <1s — *before* you spend 30s+
   building main.c and waiting for a model load. As you add more kernels
   in Phase 2, append them to the `paths[]` and `kernels[]` arrays in
   this test; rerun before every commit.

5. Build with the Metal frameworks and run the test. It must pass before
   you touch real kernels.

6. **Commit**: `src-metal: metal shim + smoke test`.

### Phase 1.5: Makefile — extend, don't replace

If the repo already has a root Makefile (typical after `build-c-reference`),
**add new metal targets to it** rather than creating a separate
`src-metal/Makefile`. Both binaries should coexist (`qwen35-cpu` and
`qwen35`, etc.). Pattern:

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

Remember to extend the existing `clean:` rule and `.gitignore` rather
than introducing new ones.

### Phase 2: Port each C kernel 1:1 to MSL

For each kernel in `src-cpu/kernels.{c,h}`, in **forward-pass order**:

#### 2a. Write the MSL kernel

Naive translation: **one thread per output element**, no tiling, no
threadgroup memory, no SIMD-group operations. Read inputs, compute,
write output. Pass scalar parameters via a small "dims" `device const`
buffer (you can also use kernel function-constant attributes, but the
buffer approach matches how the optimization phase will use it).

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

**Style guidelines for the naive port:**
- Match the C kernel's loop nest exactly.
- Use `bfloat` (Apple's bf16 type) for storage, `float` for accumulators.
- **Bundle all dims AND scalars (eps, scale, alpha, limit) into one
  `constant struct` per kernel**, in a single `[[buffer(K)]]` slot.
  Separate per-scalar buffers waste buffer slots and complicate the
  dispatch site.
- One thread per output row (or output element if outputs are scalar).
- **Do not** use `threadgroup memory`, `simdgroup_*`, vector loads, or
  any tiling. Save those for `optimize-metal`.
- **Do not** combine kernels. If C has `linear` and `swiglu` separately,
  Metal does too.
- For trivially-sequential reductions (e.g. argmax over the whole vocab),
  a single-thread kernel (`if (i != 0u) return;`) is fine for the naive
  port. `optimize-metal` will parallelise.

#### 2b. Wire the dispatch in main.c

For each Metal kernel, you need three things at the call site:

1. A `gpu_pipeline*` looked up once at startup
   (`gpu_pipeline_for(ctx, "rmsnorm_bf16", &err)`).
2. A per-dispatch slice in a **params arena** Metal buffer (see below).
   **Do NOT reuse a single param buffer across dispatches in the same
   cmdbuf** — the GPU reads buffer contents at *execution* time
   (post-commit), so all dispatches would see the last value written.
3. A `gpu_cmdbuf_dispatch(...)` call with grid=`(M, 1, 1)`,
   threadgroup=`(1, 1, 1)` (naive — one thread per output row).

**Params-arena pattern** (drop into your main.c):

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
is comfortable headroom. If you commit_wait inside forward() you can
reset the arena there too, but easier to just reset once per forward().

#### 2c. Validate the kernel against the C reference

The cleanest validator is to have the C reference run with `--dump`
flag and write the same kernel's input AND output into `src-metal/refs/`.
Then a small test:

```c
// tests/test_metal_rmsnorm.c
int main(void) {
    // 1. gpu_init(msl, &err); load rmsnorm pipeline.
    // 2. ref_t* X    = ref_load("src-metal/refs/rmsnorm_X.bin");      // f32 from Python OR bf16 from C
    //    ref_t* W    = ref_load("src-metal/refs/rmsnorm_W.bin");
    //    ref_t* Yref = ref_load("src-metal/refs/rmsnorm_Y.bin");
    // 3. Wrap as Metal buffers, run kernel, copy out.
    // 4. Compare against Yref with bf16 tolerance.
}
```

**Tolerance**: same as for C ↔ Python (~1e-2 abs in bf16). Metal GPU
reductions may reorder differently, so don't expect bit-exact. As long
as max-abs-error stays in tolerance you're good.

If the kernel fails:
- Compare element-by-element with the C reference, not the Python ref
  (they may already differ by small reduction-order amounts; we only
  care that GPU matches CPU within tolerance).
- Common bugs:
  - Wrong grid size or threadgroup size (kernel runs N times not M).
  - `[[buffer(K)]]` index doesn't match the dispatch's arg order.
  - `bfloat` vs `bfloat16` (Apple's type name has changed; use whichever
    your Metal stdlib has).
  - Forgot to multiply by stride when indexing into 3D buffers.

#### 2d. Commit per kernel

```
src-metal: <kernel_name> 1:1 metal port + test
```

### Phase 3: Stitch into a Metal main.c

1. Author `src-metal/main.c` from `main.c.template`. Structure: same as
   `src-cpu/main.c` but every C-function call becomes a
   `gpu_cmdbuf_dispatch(...)`.

2. **Weights: allocate one Metal buffer per tensor** via
   `gpu_buf_new_from(ctx, tensor.data, tensor.nbytes)` at load time.
   See "Why per-tensor weight buffers (not wrap_nocopy)" below — zero-copy
   shard wrapping is almost always blocked by Metal's offset-alignment
   rule. Total RAM usage ≈ model size; on Apple Silicon unified memory
   this is what `mmap` would page in anyway. After all tensors are copied,
   you can `st_close(arch)` to free the mmap virtual range if you want.

   Convenient pattern: index per-tensor buffers by `(t - st_at(arch, 0))`
   so `T_arg(t)` returns `{ g_tensor_bufs[idx], 0 }` (always offset 0,
   always naturally aligned).

3. Allocate workspace buffers (`x_buf`, `h_buf`, KV caches, etc.) via
   `gpu_buf_new(ctx, size)`. They live in shared (unified) storage and
   are directly host-readable via `gpu_buf_contents` — useful for the
   host-side bookkeeping ops below.

4. forward(q_off, Lq):
   - Open a `gpu_cmdbuf`. Reset the params arena.
   - For each kernel, push params and dispatch.
   - Commit + wait at the end of forward.

   For ops that mix GPU compute with **host-side scalar/bookkeeping work**
   (e.g. KV-cache memcpy at the right offset, conv1d state shift,
   one-off scalar broadcasts), the naive pattern is: commit_wait the
   current cmdbuf, do the host work directly on `gpu_buf_contents(...)`
   memory, open a fresh cmdbuf, continue. Don't try to make every
   side-quest a kernel; the optimisation skill will tidy this up.

5. AR loop:
   - Embed prompt → forward(0, Lp) → argmax → emit
   - For each token: embed → forward(Lp+i, 1) → argmax → emit → break
     on stop.

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
      (rebuild + run). Only worth adding when you actually need to bisect
      a divergence.

When all the required boxes are ticked, the skill is done.

## Tips

### Why per-tensor weight buffers (not `wrap_nocopy`)

`gpu_buf_wrap_nocopy` on an mmap'd safetensors shard *sounds* great
(zero-copy unified memory) but **almost never works** for per-tensor
access, because:

- safetensors tightly packs tensor bytes: the `data_offsets` from the
  JSON header place tensors at arbitrary byte boundaries (e.g. offset
  2417203021 for a uint32 weight, 5118533389 for a bf16 scales tensor).
- Metal `setBuffer:offset:atIndex:` requires the offset to be a multiple
  of 4 bytes *and* naturally aligned to the kernel's pointer dtype. An
  offset of 2417203021 (mod 4 = 1) fails silently with garbage reads —
  symptom: embed output values around 1e35.

Symptoms of this bug:
- `./<BIN>` generates all-zero token IDs.
- First layer of embed/RMSNorm outputs have nonsensical magnitudes
  (~1e35 or all-identical values).

**Solution: per-tensor `gpu_buf_new_from` at load time.** Each tensor
becomes its own Metal buffer; every dispatch uses offset 0. Total RAM
≈ model size; the copy takes a few seconds at memory-bandwidth speed
(~5s for 35 GB on M4 Max). On Apple Silicon this is the same RAM
`mmap` would have lazily faulted in anyway.

If you absolutely need zero-copy weights for memory reasons, your
alternatives are: (a) rewrite every weight-reading kernel to take a
`device const uchar*` raw-byte pointer plus a 4-byte-aligned byte
offset uniform, and reconstruct the typed pointer in-kernel; (b)
pre-pass through the model and emit one Metal buffer per shard with
tensors re-laid-out at dtype-aligned offsets. Both are
optimisation-skill work — for the naive port, just copy.

### Useful facts (not bugs but worth knowing)

- **Apple Silicon page size is 16 KB** (not 4 KB). Matters if you do
  page-alignment math.
- **PROT_READ mmap is fine** for `newBufferWithBytesNoCopy` reads — the
  GPU happily reads from read-only host memory. (Moot if you take the
  per-tensor-copy advice above.)
- **`tg=(1,1,1)` is accepted** by Apple GPUs (M1/M2/M3/M4). No need to
  bump to a SIMD group for the naive port.
- **Apple `bfloat` works out of the box** at `MTLLanguageVersion3_1` on
  M2+; no need for the ushort emulation path on modern Xcode.
- **JIT compile is fast** — all ~20 kernel functions in one MSL string
  compile in <1s on M4 Max. Don't worry about persisting metallibs in
  this skill.

### bf16 in MSL

Apple Metal supports `bfloat` (or `bfloat16`, depending on Xcode
version) since Metal 3 / M2-era. If your Metal version doesn't have
`bfloat`, cast manually:

```metal
inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }
inline ushort f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    uint lsb  = (u >> 16) & 1u;
    uint bias = 0x7fffu + lsb;
    return ushort((u + bias) >> 16);
}
```

### Threadgroup sizes

For the naive port, use `(1, 1, 1)` threadgroup and a grid of
`(M, 1, 1)`. Yes, this is terrible — one threadgroup per output. It
will run, and `optimize-metal` will fix it.

If the Metal compiler refuses a 1-thread threadgroup, bump to `(32, 1, 1)`
(one SIMD group) and have threads `>= M` early-out. Still naive, still
correct.

### Dispatching per-layer

Each kernel call inside forward() does:

```c
gpu_arg_buf args[] = { ABUF(X_buf), ABUF(W_buf), ABUF(Y_buf), ABUF(dims_buf), ABUF(eps_buf) };
gpu_cmdbuf_dispatch(cmdbuf, pso_rmsnorm, args, 5,
                       grid_x, grid_y, grid_z,
                       tg_x,  tg_y,  tg_z);
```

For a naive RMSNorm where one thread does one row: `grid=(Lq,1,1)`,
`tg=(1,1,1)`.

### Command buffer strategy (naive)

For each `forward()` call: one `gpu_cmdbuf_new`, dispatch every
kernel sequentially, then `commit_wait` at the end. Per-kernel barriers
happen automatically with Metal's compute encoder unless you opt into
concurrent encoding (don't do that yet).

### Kernels that update state in place

Some kernels (e.g. `conv1d_depthwise_causal_bf16`) both read input AND
update a carried `state` buffer. A naive parallel kernel hits a
write-after-read hazard across threads. Cleanest naive port: **split
into "compute on GPU" + "update state on host"** (memcpy / memmove on
`gpu_buf_contents(state)` between two cmdbufs). Optimisation skill can
later replace the host shift with a dedicated kernel.

### Host-side scalar broadcasts

For one-off ops that don't decompose cleanly into a kernel — e.g.
"sigmoid of a per-row scalar gate × a per-row vector", or "add two
host-side intermediates" — just commit_wait, do them on the host with
`gpu_buf_contents(...)`, and open a fresh cmdbuf. Naive is fine; the
optimisation skill will reshape these as broadcast kernels.

### MoE down_proj reshape trick

The per-(token, expert) inner loop in src-cpu's MoE down_proj path
collapses cleanly into a single `linear_q8_gather` dispatch by
reshaping `(L, K_top)` → `(L' = L*K_top, K_top'=1)`. The same gather
kernel handles both `gate_proj/up_proj` (real K_top) and `down_proj`
(K_top'=1). One kernel, three call sites — no need for a separate
"per-token-per-expert" kernel in the naive port.

### Per-kernel correctness tests

A pattern that scales:

1. C reference grows a `--dump <dir>` flag that, when forward() runs,
   writes every intermediate it produces to `<dir>/<kernel>_<layer>.bin`
   in the same .bin format as `tools/dump_ref.py`.

2. Each Metal kernel test:
   - Loads its inputs from the dump dir.
   - Runs the Metal kernel.
   - Loads the expected output from the dump dir.
   - Compares within tolerance.

This means every Metal kernel test runs in milliseconds (no full forward
pass needed), and `src-cpu` is the single source of truth for "what is
the expected output of this kernel".

### When tokens match but only for a few steps

If end-to-end matches for the first N tokens then diverges: that's
pure numerical drift, fine for this skill. The pieces are correct.
Acceptance is "matches the C reference for the validation prompt".
Pick a validation prompt + max-tokens combo where they DO match (a
short prompt with 16 tokens usually works), and call it done.

### Two MoE paths — DO NOT introduce here

The src-metal/ in this repo has both a "qmv4" (decode) and a "sorted-gather"
(prefill) MoE path. **The naive port does not have those**. Use the
same kernel for `Lq=1` and `Lq>1`. The optimization skill will add the
prefill fast path later.

## Common pitfalls

- **Per-tensor offset alignment**: see "Why per-tensor weight buffers
  (not `wrap_nocopy`)" above. This is the #1 silent-failure trap.
- **Symlink relative paths**: `ln -sf` resolves the target relative to
  the *symlink's* directory, not your cwd. Run `ln` from inside
  `src-metal/` (or use `cp -a`).
- **Param buffer reuse within a cmdbuf**: the GPU reads buffer
  contents at execution time (post-commit), so reusing one param
  buffer across dispatches in the same cmdbuf gives every dispatch
  the *last* value written. Use the bump-arena pattern (one slice per
  dispatch, reset per forward()).
- **Buffer arg index mismatch**: `[[buffer(0)]]` in the .metal file
  must line up with `args[0]` in the dispatch. Missing one slot
  silently shifts every following arg. The Phase-1 smoke test catches
  most of these by attempting to load every named pipeline.
- **Stride confusion**: HF weights are `[N, K]` row-major. In MSL,
  `W[n*K + k]` reads row `n`. Easy to swap.
- **bfloat type name**: older Metal had no native bf16 — emulate via
  `ushort` + `as_type` casts. Test on your Xcode version first.
- **Forgot to barrier between dispatches**: Metal's compute encoder
  serializes by default; you only need explicit barriers if you use
  the *concurrent* encoder. Don't.
- **KV cache offsets**: when writing K/V at position `q_off`, write to
  `k_cache[li] + q_off * Nkv * D`, not `q_off * Nq * D`. The K cache
  has Nkv heads, not Nq.
- **SDPA scores need bounded thread-local storage**: declare
  `float scores[MAX_CTX]` with `MAX_CTX` set well above your model's
  `max_ctx`. Apple Metal allows fairly large per-thread arrays — for
  the naive port this is simpler than threadgroup/global scratch.

## Commit strategy

One commit per kernel, then one for stitching:

```
src-metal: metal shim + smoke test
src-metal: embed_gather_bf16 metal port + test
src-metal: rmsnorm_bf16 metal port + test
src-metal: linear_bf16 metal port + test
src-metal: rope_bf16 metal port + test
src-metal: sdpa_bf16 metal port + test
src-metal: topk_softmax_bf16 metal port + test
src-metal: mxfp4_linear_gather_bf16 metal port + test
src-metal: swiglu_bf16 metal port + test
src-metal: expert_mix_bf16 metal port + test
src-metal: residual_add_bf16 metal port + test
src-metal: argmax_bf16 metal port + test
src-metal: end-to-end metal forward + AR decode (N/N tokens match src-cpu)
```

## Hand-off to optimize-metal

The hand-off contract:

1. `./src-metal/` exists, builds with `make <BIN>`.
2. `./src-metal/<BIN> --prompt "<validation_prompt>" --max-tokens N` runs on
   the GPU and produces the same N tokens as `./src-cpu/<BIN>`.
3. Every kernel is one `.metal` file in `src-metal/kernels/` with a naive
   1-thread-per-output implementation.
4. Per-kernel correctness tests exist in `tests/`.
5. The C reference (`src-cpu`) remains the ground truth for kernel
   outputs and end-to-end tokens. Do not modify it during this skill.

When this contract is met, run the `optimize-metal` skill.
