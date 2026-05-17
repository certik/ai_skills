# Sliding-window / rotating KV cache (catalog section D4)

## What

For models with sliding-window attention (e.g., gpt-oss has window=128
on every other layer), the K/V cache for those layers does NOT need to
hold the full context — only the window. Use a **rotating fixed-size
KV cache** of size `W` (the window) instead of `MAX_CTX`.

## Why

- Memory: saves `(MAX_CTX - W) * Nkv * D * 2 * sizeof(bf16)` per sliding
  layer. For gpt-oss (12 sliding layers, MAX_CTX=1024, W=128,
  Nkv=8, D=64, bf16): saves `12 × (1024-128) × 8 × 64 × 2 × 2 ≈ 22 MB`.
- Bandwidth: SDPA only reads `W` K/V positions per query, not `Lk`.
  Even with full-cache layout the kernel ignored the older positions,
  but the cache writes still touched cold pages.

## How

Allocate K/V cache as `[W, Nkv, D]`. When writing position `q_off`,
write to slot `q_off % W`. In the SDPA kernel, iterate `lk` from
`win_lo` to `lk_hi` *in absolute coords*, but index the cache with
`lk % W`:

```metal
for (int lk_abs = win_lo; lk_abs <= lk_hi; ++lk_abs) {
    uint slot = uint(lk_abs) % W;
    const device bfloat* krow = K_cache + (slot * Nkv + hkv) * D;
    // ... use krow as before
}
```

## When to apply

If the model has sliding-window attention. The reference implementation
likely already does this (MLX-LM does for gpt-oss).

## Speedup

- 1.05–1.20× decode for the sliding layers (mostly from reduced cache
  writeback bandwidth).
- Free memory savings.

## Commits

- b09f12b — `mlx-cpp: rotating fixed-size KV cache for sliding layers`
- fc59ccf — `mlx-cpp: pipeline decode + chunked sliding cache (99 tok/s)`

## Subtleties

- The window can extend before position 0 — clamp `win_lo` to 0.
- When `lk_hi < W - 1` (early in decode), some slots in the ring are
  unwritten — make sure your initial allocation is zero (or the kernel
  ignores positions `> lk_hi`).
- This optimization is orthogonal to (and composable with) multi-SG
  SDPA, online softmax, and SDPA sinks.
