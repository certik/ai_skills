# MSL / Apple GPU tips

Reference notes about Metal Shading Language and Apple Silicon GPUs
that are useful when porting kernels but not needed for the procedural
spine. Skim once; come back when something looks off.

## Naive style

The naive port style is **one thread per output element**, no tiling,
no threadgroup memory, no SIMD-group operations. Read inputs, compute,
write output. Match the C kernel's loop nest exactly.

- Use `bfloat` for storage, `float` for accumulators.
- Bundle all dims AND scalars (`eps`, `scale`, `alpha`, `limit`) into
  ONE `constant struct` per kernel in a single `[[buffer(K)]]` slot.
  Separate per-scalar buffers waste buffer slots and complicate the
  dispatch site.
- Do not combine kernels. If C has `linear` and `swiglu` separately,
  Metal does too.
- For trivially-sequential reductions (e.g. argmax over the whole vocab),
  a single-thread kernel (`if (i != 0u) return;`) is fine. `optimize-metal`
  will parallelise.

`starter/rmsnorm.metal.template` and `starter/linear_naive.metal.template`
are worked examples of this style.

## Threadgroup sizes

For the naive port, use `(1, 1, 1)` threadgroup and a grid of
`(M, 1, 1)`. Yes, this is terrible — one threadgroup per output. It
will run, and `optimize-metal` will fix it.

If the Metal compiler ever refuses a 1-thread threadgroup, bump to
`(32, 1, 1)` (one SIMD group) and have threads `>= M` early-out.
Still naive, still correct.

## Command buffer strategy (naive)

For each `forward()` call: one `gpu_cmdbuf_new`, dispatch every
kernel sequentially, then `commit_wait` at the end. Per-kernel
barriers happen automatically with Metal's compute encoder unless
you opt into concurrent encoding (don't, yet).

## Dispatch boilerplate

```c
gpu_arg_buf args[] = {
    ABUF(X_buf), ABUF(W_buf), ABUF(Y_buf), push_params(&p, sizeof p)
};
gpu_cmdbuf_dispatch(cmdbuf, pso_rmsnorm, args, 4,
                    grid_x, grid_y, grid_z,
                    tg_x,   tg_y,   tg_z);
```

For a naive RMSNorm where one thread does one row:
`grid = (M, 1, 1)`, `tg = (1, 1, 1)`.

## Useful Apple-GPU facts

- **Apple Silicon page size is 16 KB** (not 4 KB). Matters if you do
  page-alignment math.
- **`PROT_READ` mmap is fine** for `newBufferWithBytesNoCopy` reads —
  the GPU happily reads from read-only host memory. (Moot if you take
  the per-tensor-copy advice in `references/pitfalls.md`.)
- **`tg = (1, 1, 1)` is accepted** by Apple GPUs (M1/M2/M3/M4). No
  need to bump to a SIMD group for the naive port.
- **Apple `bfloat` works out of the box** at `MTLLanguageVersion3_1`
  on M2+; no need for the `ushort` emulation path on modern Xcode.
- **JIT compile is fast** — ~20 kernel functions in one MSL string
  compile in <1s on M4 Max. Don't worry about persisting `.metallib`
  artifacts in this skill.
