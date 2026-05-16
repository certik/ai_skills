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

1. `mkdir -p src-metal/kernels`. Copy drop-ins: `metal_shim.{h,m}`,
   `kernel_concat.{c,h}`.

2. Copy/symlink `safetensors.{c,h}`, `tokenizer.{c,h}`, `tokenizer.bin`,
   and your `<chattmpl>.{c,h}` from `src-cpu/`. (Symlinks keep the two
   trees in sync. Copies are fine if you prefer fully-independent
   directories.)

3. Author a minimal `src-metal/kernels/noop.metal` containing a 1-line kernel:
   ```metal
   kernel void noop_inc(device uint* x [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
       x[id] += 1;
   }
   ```

4. Author a tiny `tests/test_shim.c` that:
   - Reads `noop.metal` into a string (or uses `kernel_concat` to slurp
     `src-metal/kernels/*.metal`).
   - Calls `gpu_init(msl_source, &err)`.
   - Allocates a small buffer, dispatches `noop_inc`, reads back, checks
     each element incremented by 1.

5. Build with the Metal frameworks and run the test. It must pass before
   you touch real kernels.

6. **Commit**: `src-metal: metal shim + smoke test`.

### Phase 2: Port each C kernel 1:1 to MSL

For each kernel in `src-cpu/kernels.{c,h}`, in **forward-pass order**:

#### 2a. Write the MSL kernel

Naive translation: **one thread per output element**, no tiling, no
threadgroup memory, no SIMD-group operations. Read inputs, compute,
write output. Pass scalar parameters via a small "dims" `device const`
buffer (you can also use kernel function-constant attributes, but the
buffer approach matches how the optimization phase will use it).

Example (RMSNorm):

```metal
// rmsnorm.metal
#include <metal_stdlib>
using namespace metal;

struct rmsnorm_dims { uint M; uint D; };

kernel void rmsnorm_bf16(device const bfloat*       X       [[buffer(0)]],
                         device const bfloat*       W       [[buffer(1)]],
                         device bfloat*             Y       [[buffer(2)]],
                         constant rmsnorm_dims&     dims    [[buffer(3)]],
                         constant float&            eps     [[buffer(4)]],
                         uint                       m       [[thread_position_in_grid]])
{
    if (m >= dims.M) return;
    uint D = dims.D;
    const device bfloat* xp = X + m * D;
    device bfloat*       yp = Y + m * D;

    float sq = 0.0f;
    for (uint d = 0; d < D; ++d) {
        float v = float(xp[d]);
        sq += v * v;
    }
    float rrms = rsqrt(sq / float(D) + eps);
    for (uint d = 0; d < D; ++d) {
        yp[d] = bfloat(float(xp[d]) * rrms * float(W[d]));
    }
}
```

**Style guidelines for the naive port:**
- Match the C kernel's loop nest exactly.
- Use `bfloat` (Apple's bf16 type) for storage, `float` for accumulators.
- Pass dimensions in a `constant struct` buffer; pass scalars (eps,
  scale, alpha, limit) in their own small buffers.
- One thread per output row (or output element if outputs are scalar).
- **Do not** use `threadgroup memory`, `simdgroup_*`, vector loads, or
  any tiling. Save those for `optimize-metal`.
- **Do not** combine kernels. If C has `linear` and `swiglu` separately,
  Metal does too.

#### 2b. Wire the dispatch in main.c

For each Metal kernel, you need three things at the call site:

1. A `gpu_pipeline*` looked up once at startup
   (`gpu_pipeline_for(ctx, "rmsnorm_bf16", &err)`).
2. Small param buffers for `dims` / scalars (allocate once, refill in
   place per call).
3. A `gpu_cmdbuf_dispatch(...)` call with grid=`(M, 1, 1)`,
   threadgroup=`(1, 1, 1)` (naive — one thread per output row).

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

2. Allocate weights as Metal buffers via `gpu_buf_wrap_nocopy` on the
   mmap'd safetensors regions (zero-copy on Apple Silicon — works as
   long as the host pointer is page-aligned, which `mmap` guarantees).

3. Allocate workspace buffers (`x_buf`, `h_buf`, KV caches, etc.) via
   `gpu_buf_new(ctx, size)`.

4. forward(q_off, Lq):
   - Open a `gpu_cmdbuf`.
   - For each kernel, dispatch with grid sized by `Lq * ...`.
   - Commit + wait at the end of forward.

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
- [ ] Per-kernel tests in `tests/` pass.

When all boxes are ticked, the skill is done.

## Tips

### Zero-copy weights

`gpu_buf_wrap_nocopy(ctx, host_ptr, bytes)` makes the mmap'd
safetensors region directly accessible from the GPU on Apple Silicon
(unified memory). This avoids a 14 GB copy at startup. The host pointer
must be page-aligned and the size a multiple of the page size; `mmap`
gives you both.

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

- **Buffer arg index mismatch**: `[[buffer(0)]]` in the .metal file
  must line up with `args[0]` in the dispatch. Missing one slot
  silently shifts every following arg.
- **Stride confusion**: HF weights are `[N, K]` row-major. In MSL,
  `W[n*K + k]` reads row `n`. Easy to swap.
- **Page alignment for wrap_nocopy**: if you compute a pointer that's
  not page-aligned (e.g., into the middle of a safetensors blob), use
  `gpu_buf_new_from` (copy) instead. Or use the buffer-and-offset
  form: keep the whole shard as the buffer and pass `offset = tensor's
  byte offset` in the `gpu_arg_buf`.
- **bfloat type name**: older Metal had no native bf16 — emulate via
  `ushort` + `as_type` casts. Test on your Xcode version first.
- **Forgot to barrier between dispatches**: Metal's compute encoder
  serializes by default; you only need explicit barriers if you use
  the *concurrent* encoder. Don't.
- **KV cache offsets**: when writing K/V at position `q_off`, write to
  `k_cache[li] + q_off * Nkv * D`, not `q_off * Nq * D`. The K cache
  has Nkv heads, not Nq.

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
