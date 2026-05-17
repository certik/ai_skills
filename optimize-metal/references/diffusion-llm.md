# Diffusion-LLM optimization notes

The catalog and the empirical attack order were largely written from
**autoregressive decoder** ports (Qwen3.6, gpt-oss). Diffusion-LLM
samplers (Dream, LLaDA, "fastdllm" style block-diffusion) have a
different forward structure, and that changes which optimizations
matter and in what order.

Read this file when your `./src-metal/` is a diffusion LLM and you've
just finished the kernel-level wins. The end-game looks different from
an autoregressive port.

## What's different

A diffusion LLM doesn't generate one token at a time. Instead:

1. Initialize a sequence `x = [prompt; mask, mask, ..., mask]` of fixed
   length `L`.
2. Process the gen region in blocks of length `BL` (e.g., 16). For each
   block (4 blocks for a 64-token gen):
   - **Prefetch pass**: forward the entire `L`-long context once, fresh
     KV cache. Read ONE logits row (the one for `block_start`) to fill
     the first token of the block.
   - **Refine loop**: re-run forward with `Lq = BL` (block-local),
     using the cached KV plus replacing positions in `[block_start,
     block_start+BL)` of the cache. After each refine step, score the
     softmax confidence of every mask position in the block and unmask
     the high-confidence ones. Repeat until all `BL` positions are
     unmasked (or `steps_per_block` is exhausted).

So in numbers, for the validation prompt that drove this skill (Dream
7B, `L=64+25, BL=16, steps=64`):

| Phase     | Calls            | M (Lq) | What runs                                        |
|-----------|------------------|--------|--------------------------------------------------|
| Prefetch  | `n_blocks` = 4   | L = 89 | Full forward over the whole sequence             |
| Refine    | ~35 (early-exit) | BL = 16 | Block-local forward with KV-replace              |
| host_post | every step       | n/a    | Per-position softmax + argmax + confidence over V |

Refine is the long pole (~80% of GPU time). Prefetch is ~20%.

## Profiling diffusion runs

Tracking only `wall` and `gpu_busy` is insufficient — you also need
**`host_post`** as a separate accumulator, because the host loop that
does `softmax + argmax + confidence` for every mask position is
roughly:

```
for each refine step:
    for i in [0, BL):
        for v in [0, V):              # V = 152064 for Dream
            probs[v] = exp(logits[i, v] - max_logit)
        # then argmax + threshold check
```

That's `BL × V × 3` scalar ops per refine, all on the CPU. In one
session we saw `host_post = 0.25 s` out of `wall = 2.90 s` — **8.6%
of total wall, completely invisible in `gpu_busy`**. Optimizing
kernels can't reach this. Always log it separately:

```c
double t = now_sec();
double g = gpu_time_get();
forward_cached(...);
prefetch_wall += now_sec() - t;
prefetch_gpu  += gpu_time_get() - g;

double th = now_sec();
... host softmax / argmax / threshold ...
host_post_wall += now_sec() - th;
```

Then your stats line is `wall = gpu_busy + host_post + (encode_gap)`,
where `encode_gap = wall - gpu_busy - host_post` is the residual
host-encode overhead — usually small once H1/H2 are in.

## What does NOT help (relative to autoregressive ports)

### A4 (2-deep cmdbuf pipeline) is mostly inert

For autoregressive decode, A4 wins because the host can encode `cb_{t+1}`
while the GPU executes `cb_t`. For diffusion refine, this **fails**:

- Each refine step's input ids depend on `select_confident_updates(...)`
  on the previous step's logits.
- The host therefore can't start encoding the next refine until the
  current refine has returned its argmax/confidence to host memory.

A4 can still save the prefetch wall-gpu gap (~0.08 s in one port) but
that's not where the bottleneck is. Don't reach for A4 on a diffusion
LLM unless prefetch dominates.

### Decode/prefill split (G) is replaced by prefetch/refine split

A diffusion LLM never does true "decode" (single-token-at-a-time). The
useful split is by *Lq value*: prefetch (`Lq=L=89` for Dream-7B
validation prompt) vs refine (`Lq=BL=16`). Make sure your GEMM tile is
right for BOTH. See "dual-tile dispatch" below.

## What DOES help (in order)

### Step 1: GPU per-row softmax + argmax + confidence (F3)

The single biggest diffusion-LLM win, period. Replace the host loop
above with one Metal kernel:

```
input:  logits   bf16 [BL, V]
output: out      {int idx; float conf}[BL]
```

One TG of 256 threads per row, two-pass:
1. `max + argmax over V` (SG-wide `simd_max`, then cross-SG via TG-mem).
2. `sum_exp(logit - max)` over V. `conf = 1 / sum_exp`.

The output is `BL * 8 bytes = 128 bytes`, tiny. The host then only
applies the threshold logic and writes back the chosen ids.

In one port this dropped `host_post` from 0.25 s → 0.00 s and the
wall from 2.90 s → 2.63 s on the same prompt. See catalog F3, pattern
`patterns/softmax_argmax_per_row.metal`.

### Step 2: Prune unused output rows in prefetch lm_head (H3)

The prefetch forward does `lm_head(h_buf[0..L-1])` even though the host
only reads ONE row of the output (the row at `block_start - 1`, used
to fill the first token of the next block). For `L=89`, `V=152064`,
that's 88 rows of compute and 88 rows of `W` streaming that get
discarded.

Pass an `lm_head_only_row` hint to `forward_cached`; on that path,
dispatch lm_head with `M=1` and an `x_off_bytes` offset into `h_buf`.
The kernel writes one row to `logits_buf[0]`, the host reads row 0.

Save: ~30 ms on prefetch (small but free). See catalog H3.

### Step 3: Dual-tile GEMM dispatch by M

Prefetch (M=89) is compute-bound and benefits from `BM=32 TM=4`
(halves the number of M row-bands, halves W-streaming). Refine (M=16)
stays at `BM=16 TM=2` (BM=32 wastes 50% of the threads and Cs TG-mem).

The simplest layout: two PSOs (`pso_gemm` and `pso_gemm_bm32`), one
kernel file each, `d_linear` picks based on `M > 16`. Each kernel file
defines its tile constants and is otherwise identical algorithm.

Buffer padding: with BM=32 in play, ALL workspace buffers (`x_buf`,
`h_buf`, `attn_buf`, etc.) must be padded to `(L + 31) & ~31` rows so
the BM=32 row-band loads are in-buffer. If you forget this and a load
runs off the end, you get nondeterministic garbage past the first ~30
tokens.

### Step 4: Fused residual into o_proj / down_proj (B5)

Same as autoregressive. Drops 56 `residual_add` dispatches per
forward. Token-identical on Dream-7B in our test (bf16 round point
happened to coincide).

### Step 5: Concurrent encoder + barrier surgery (A5)

Same as autoregressive. q/k/v are independent, gate/up are independent,
rope_Q/rope_K are independent, cache_write_K/cache_write_V are
independent. Wrap each independent group in a `gpu_cmdbuf_barrier`
fence.

### Step 6 (BL ≥ 32 only): Aspect-ratio dispatch for down_proj (C4)

At small BL (e.g., 16) the refine M is ≤16 and everything routes to
the BM=16 path, so this step is a no-op. At BL ≥ 32 refine starts
hitting `M > 16` and the dual-tile dispatch (Step 3) sends EVERYTHING
to BM=32. That's right for gate_proj, up_proj, qkv, o_proj, lm_head —
all of which have `N ≥ K`. It's WRONG for down_proj.

down_proj has `K = I = 18944, N = H = 3584` — the only LLaMA-style MLP
matmul with K > N. At BM=32 BN=64 you get N/BN = 56 TGs, which on
M4 Max (~80 TG slots) leaves the memory subsystem starved (measured
125 GB/s effective vs gate_proj's 314 GB/s on the SAME matrix size).

Fix: extend the dual-tile dispatch with an aspect-ratio check.

```c
int use_bm32 = (M > 16) && (K <= N);   // K > N → BM=16 even at M>16
gpu_pipeline* pso = use_bm32 ? pso_gemm_bm32 : pso_gemm;
```

On Dream-7B BL=32 this drops down_proj from 1082us → 741us per call
(-31%), refine from 73ms → 63ms per step, and wall from **1.72 s →
1.52 s (12% faster, beats MLX 1.56 s)**. See gotcha #39 and catalog
C4 for the full analysis.

## End-to-end validation

Diffusion LLMs are particularly easy to validate because the entire
output sequence is bounded by `L`. Generate the full sequence and
compare token-by-token against `src-cpu/` and MLX. The "first N
tokens" rule is essentially "all generated tokens" here.

## Worked example: Dream-7B on M4 Max (this skill's reference port)

| Stage                           | Wall (s) | gpu_busy | host_post |
|---------------------------------|----------|----------|-----------|
| Naive port (1-thread argmax+softmax host) | ~10  | ~3       | ~5        |
| After B1, SG-per-row reductions | ~5       | ~3       | ~1        |
| After H1 (glue kernels)         | ~4       | ~2.7     | ~0.5      |
| After C1+C2 (MMA GEMM)          | 3.40     | 2.70     | 0.25      |
| After A5 (concurrent encoder)   | 3.04     | 2.71     | 0.25      |
| After C2 dual-tile (BM=32/16)   | 3.04     | 2.71     | 0.25      |
| After B5 (fused residual)       | 3.04     | 2.72     | 0.25      |
| After H3 (single-row lm_head)   | 3.04     | 2.69     | 0.25      |
| After BK=16 GEMM inner loop     | 2.90     | 2.54     | 0.25      |
| **After F3 (GPU softmax+argmax)** | **2.63** | **2.53** | **0.00**  |
| MLX target                      | 2.74     | —        | —         |

Final: 4% faster than MLX (`2.63 s` vs `2.74 s`). F3 alone closed half
the remaining gap.

### Follow-up: BL=32 with max_tokens=32, where down_proj dominates

A second, shorter benchmark (`max-new-tokens=32 --steps=24 --block-length=32`)
exposed a different bottleneck — **down_proj alone took 38.5% of refine
GPU time** because at BM=32 BN=64 it produced only 56 TGs (vs gate_proj's
296), under-occupying the GPU's memory subsystem.

| Stage                                            | Wall (s) | Notes                            |
|--------------------------------------------------|----------|----------------------------------|
| All optimizations above (≤ Step 5)               | 1.72     | refine 73 ms/step, down_proj 1082 us/call (125 GB/s) |
| **After C4 (down_proj K>N → BM=16 routing)**     | **1.52** | refine 63 ms/step, down_proj 741 us/call (-31%) |
| MLX target                                       | 1.56     | —                                |

Final on this bench: **3% faster than MLX** (1.52 s vs 1.56 s),
single-line dispatch change. This is the lesson that prompted gotcha
#39 / catalog C4 and made aspect-ratio routing a permanent step in
the recipe (Step 6 above).

**Diagnostic note**: this win would have stayed hidden without KPROF.
The wall gap to MLX was only ~10% (not 2×, despite a misleading `tok/s`
ratio in the surface output — see "Profiling diffusion runs" above),
and the bottleneck wasn't one of the usual suspects. Always KPROF
before optimizing past the empirical attack order; per-kernel attribution
is what makes "this matmul is twice as slow as the matmul of the same
size" visible.
