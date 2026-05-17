# Pitfalls — silent failures and "weird symptoms" lookup

These are the things that, if you miss them, will eat a day of debugging.
SKILL.md flags the high-impact ones inline; this file has the full
explanations. If a Metal kernel is producing nonsense, scan the
"Symptoms" lines first.

## Per-tensor offset alignment (the #1 silent-failure trap)

**Symptoms**
- `./<BIN>` generates all-zero token IDs, or all-identical tokens.
- First layer of embed / RMSNorm outputs have nonsensical magnitudes
  (around `1e35`, or all-identical values).
- No crash, no warning — just garbage downstream.

**Why**

`gpu_buf_wrap_nocopy` on an mmap'd safetensors shard *sounds* great
(zero-copy unified memory) but almost never works for per-tensor access:

- safetensors tightly packs tensor bytes. The `data_offsets` from the
  JSON header place tensors at arbitrary byte boundaries (e.g. offset
  2417203021 for a uint32 weight, 5118533389 for a bf16 scales tensor).
- Metal `setBuffer:offset:atIndex:` requires the offset to be a multiple
  of 4 bytes *and* naturally aligned to the kernel's pointer dtype. An
  offset of 2417203021 (mod 4 = 1) fails silently with garbage reads.

**Fix**

Per-tensor `gpu_buf_new_from` at load time — every tensor becomes its
own Metal buffer; every dispatch uses offset 0. Total RAM ≈ model size;
the copy takes a few seconds at memory-bandwidth speed (~5s for 35 GB
on M4 Max). On Apple Silicon unified memory this is the same RAM
`mmap` would have lazily faulted in anyway.

Convenient pattern: index per-tensor buffers by `(t - st_at(arch, 0))`
so `T_arg(t)` returns `{ g_tensor_bufs[idx], 0 }` — always offset 0,
always naturally aligned.

**If you really need zero-copy weights for memory reasons**

Both options below are `optimize-metal` work; do not introduce here.

- Rewrite every weight-reading kernel to take a `device const uchar*`
  raw-byte pointer plus a 4-byte-aligned byte offset uniform, and
  reconstruct the typed pointer in-kernel.
- Pre-pass through the model and emit one Metal buffer per shard with
  tensors re-laid-out at dtype-aligned offsets.

## Param buffer reuse within a cmdbuf

**Symptoms**
- Per-kernel tests pass, but end-to-end output is wrong.
- All dispatches in a layer behave as if they used the same dims
  (e.g. wrong M / D / scale on every dispatch except the last).

**Why**

The GPU reads buffer contents at *execution* time (post-commit), not at
dispatch-encode time. If you reuse a single Metal buffer for params
across dispatches in the same cmdbuf, every dispatch reads whatever
was last written — usually the last layer's values.

**Fix: bump-arena pattern**

One buffer per `forward()`, sliced per dispatch, reset at the top of
`forward()`. See the snippet in SKILL.md Phase 2b. Budget ≈ 80 KB per
forward() for a typical 40-layer model; 4 MB headroom is comfortable.

## Symlink relative paths

**Symptoms**
- `head -1 src-metal/safetensors.h` says "No such file or directory"
  even though `ls -l` shows the symlink.
- Build fails with confusing missing-include errors.

**Why**

`ln -sf TARGET LINK` resolves `TARGET` relative to the *symlink's*
directory, not the cwd you ran `ln` from. So
`ln -sf ../src-cpu/safetensors.h src-metal/safetensors.h` (run from
the repo root) creates a link whose target is `../src-cpu/...`
relative to `src-metal/`, which becomes `src-cpu/safetensors.h` —
which happens to work. But:
`ln -sf src-cpu/safetensors.h src-metal/safetensors.h` (also from
repo root) makes a link to `src-cpu/safetensors.h` *relative to
src-metal/* = `src-metal/src-cpu/safetensors.h`. Broken.

**Fix**

Either `cp -a` (simplest), or run `ln -s` from inside `src-metal/`:

```sh
cd src-metal
for f in safetensors.{c,h} tokenizer.{c,h} <chattmpl>.{c,h} tokenizer.bin; do
    ln -sf ../src-cpu/$f
done
```

Verify with `head -1 src-metal/safetensors.h` — if you get "No such
file", the link is broken.

## Buffer arg index mismatch

The `[[buffer(K)]]` index in the `.metal` file must line up exactly
with the position in the dispatch's `args[]`. Missing or extra slot
silently shifts every following arg.

**Prevention**: the Phase-1 smoke test (`test_shim.c`) that compiles
every MSL file and looks up every named kernel function catches the
class of bugs where you mistyped a kernel name or have a syntax
error. It does *not* catch arg-order mistakes — those show up as
garbage at first-use. Mitigation: when adding a new dispatch site,
read the kernel's `[[buffer(K)]]` annotations top-to-bottom and write
the `args[]` in the same order.

## Stride confusion

HF weights are `[N, K]` row-major. In MSL, `W[n*K + k]` reads row `n`.
Easy to swap when porting from C if the C code used a transposed view.

## KV cache offsets

When writing K/V at position `q_off`, write to
`k_cache[li] + q_off * Nkv * D`, not `q_off * Nq * D`. The K cache
has `Nkv` heads (key-value), not `Nq` (query). With GQA (Nkv < Nq)
this is the difference between correct and very wrong.

## SDPA scores need bounded thread-local storage

For the naive port, declare `float scores[MAX_CTX]` inside the
SDPA kernel with `MAX_CTX` set well above your model's `max_ctx`.
Apple Metal allows fairly large per-thread arrays. This is simpler
than threadgroup or global scratch buffers — `optimize-metal` will
replace it with online softmax later.

## bfloat type name

Older Metal versions (pre-3) lack native `bfloat`. If your Xcode is
old, emulate via `ushort` + `as_type` casts:

```metal
inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }
inline ushort f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    uint lsb  = (u >> 16) & 1u;
    uint bias = 0x7fffu + lsb;
    return ushort((u + bias) >> 16);
}
```

Modern Xcode (Metal 3 / `MTLLanguageVersion3_1`, M2+) supports
`bfloat` natively — prefer it.

## Concurrent encoder barriers

Metal's compute encoder serializes dispatches by default. You only
need explicit barriers if you opt into the **concurrent** encoder.
Don't, for the naive port.

## Build-system: extend, don't replace

If the repo already has a root Makefile (typical after
`build-c-reference`), add new `metal` targets to it rather than
introducing `src-metal/Makefile`. Both binaries should coexist
(e.g. `qwen35-cpu` and `qwen35`). Also extend the existing
`clean:` rule and `.gitignore` rather than creating new ones.
