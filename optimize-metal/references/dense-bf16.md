# Dense bf16 LLMs — attack order

**Applies to**: Gemma 3, Gemma 4 (E4B/E2B), Llama bf16, Mistral bf16,
and any other modern dense (non-MoE, non-quantized) LLM whose every
linear is one bf16 read of W times an x.

**Why a separate menu**: dense BF16 lacks affine dequantization work, but
that does not prove every linear is bandwidth-saturated. Start with
pinned source/actual dispatch and the profile. Several steps that pay off on
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
4. **B5 round-preserving residual epilogue**, if native rounding allows it.
5. **Assess remaining headroom**, not a hard weight-byte ceiling. Nominal
   weight bytes/peak GB/s omits reuse/cache, actual traffic and instructions.
   Read gotcha #59 and `measurement-and-diagnosis.md`.

## Skip on dense bf16

- **A8 concurrent encoder** often loses to shared-bandwidth contention in
  dense BF16; the Gemma example had no gain. Check actual chain headroom
  before testing, rather than assuming all dense kernels saturate DRAM.
- **B6 uint4 amortized (s,b)** — there are no `(scale, bias)` to
  amortize. The kernel doesn't apply.
- **B4 TG-mem X-share** — gotcha #56's "regresses when L1/L2
  already serves X" applies even harder when W is already at
  peak.
- **K_OUT=8 in GEMV** was neutral/regressive on covered q8/BF16 cases.
  Start with native geometry or K_OUT4, not an unbounded sweep. Re-test
  only with a shape/device/profile reason. (gotcha #60)

## Worked example — Gemma 4 E4B on M4 Max

Naive port: 5.6 tok/s. After B1/D1/H1 + per-layer-embed_gather fix +
F4 + D3 + B5: **49.7 tok/s short / 45.6 tok/s long, beating MLX
48.2 / 44.1 by +3.1% / +3.4%**.

The reported99–108% of a nominal weight-byte estimate reflects cache/reuse
or imperfect assumptions; it does not prove no further wins are possible.

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
