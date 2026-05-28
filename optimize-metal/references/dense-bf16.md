# Dense bf16 LLMs — attack order

**Applies to**: Gemma 3, Gemma 4 (E4B/E2B), Llama bf16, Mistral bf16,
and any other modern dense (non-MoE, non-quantized) LLM whose every
linear is one bf16 read of W times an x.

**Why a separate attack order**: a dense bf16 LLM has *no quantization
headroom* — every linear is at or near peak DRAM bandwidth in
isolation. The default MoE / quantized attack order in SKILL.md still
applies up to a point, but several late-pipeline steps that pay off on
Qwen3.6 do nothing here, and one obscure gotcha (per-layer
`embed_gather` on Gemma 3/4 architectures) is the single largest
late-stage win.

## The order

0. **A9 parallel pread weight loader, B1 SG-per-output GEMV, D1 SDPA
   SG-per-head, H1 glue kernels.** Same as the default MoE order.
1. **Scan KPROF for ANY per-layer kernel with `us_per_call > 50`.** On
   modern architectures (Gemma 3 / 4) the most common culprit is the
   **per-layer `embed_gather`** that the naive port left at 1 thread
   per row. At D=10752 bf16 × 42 layers × 12 tokens it's ~25 ms/token
   of pure latency. TG-per-row + 128-thread cooperative bfloat4 copy
   gives +4-5% wall (single biggest late-stage win on Gemma 4 E4B).
   Full kernel + dispatch in `patterns/embed_gather_per_layer.metal`;
   diagnosis recipe in gotcha #57.
2. **F4 multi-SG rmsnorm at decode** (M=1, large D). Same as MoE.
3. **D3 multi-SG SDPA with online merge.** Same as MoE.
4. **B5 fuse residual into LAST linear epilogue.** Same as MoE.
5. **STOP — check the DRAM-BW ceiling** (gotcha #59). If you're at
   ≥ 95% of `(per-token weight bytes) / (peak GB/s)`, further
   optimization is futile without quantization.

## Skip on dense bf16

- **A8 concurrent encoder** — every parallel chain is already at
  ~100% DRAM peak; concurrent dispatches share the same ceiling.
  Tokens correct but ~0% gain or slight regression from the
  `autobar=false` overhead. (gotcha #58)
- **B6 uint4 amortized (s,b)** — there are no `(scale, bias)` to
  amortize. The kernel doesn't apply.
- **B4 TG-mem X-share** — gotcha #56's "regresses when L1/L2
  already serves X" applies even harder when W is already at
  peak.
- **K_OUT=8 in GEMV** — already proven neutral on q8/uint4 and
  now on bf16 too. Stay at K_OUT=4. (gotcha #60)

## Worked example — Gemma 4 E4B on M4 Max

Naive port: 5.6 tok/s. After B1/D1/H1 + per-layer-embed_gather fix +
F4 + D3 + B5: **49.7 tok/s short / 45.6 tok/s long, beating MLX
48.2 / 44.1 by +3.1% / +3.4%**.

At 99-108% of theoretical DRAM ceiling — no more wins available
without quantization.

## Why per-layer embed_gather is so painful

Gemma 3 and 4 add a per-layer additional embedding to the hidden
state at every layer. The naive port-c-to-metal keeps the gather as
1 thread per row (gotcha #17 family, the TG=(1,1,1) pattern). For a
plain elemwise kernel that's a small constant overhead; here it's
amplified by D=10752 × L=42 layers × tokens, so what would be a 9 µs
kernel becomes 600 µs.

The fix is mechanical: TG-per-row, 128 threads cooperate on the row,
loads as `bfloat4` at 8-byte-aligned addresses. The embed dim is
always 4-divisible in mainstream architectures so alignment is free.

See `patterns/embed_gather_per_layer.metal` for the kernel (both
flat-table `[V, D_pli]` and per-layer-interleaved `[V, L, D_pli]`
variants) and the dispatch snippet.
