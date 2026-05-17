# Parallel independent kernel chains — the LATE-decode killer win

## Pattern

When you've already pulled the obvious decode wins (qmv4, parallel
argmax, multi-SG SDPA, glue kernels, 2-deep pipeline) and you're stuck
~5% short of MLX, the remaining gap is almost always **serialization
of independent computation chains**.

A typical MoE layer has two completely independent subgraphs that share
ONLY their input (post_attn_rmsnorm output):

```
              h_buf  (post_attn rmsnorm)
              /             \
       MoE chain         shared-expert chain
       --------          --------------------
       router            shared.gate_proj
       softmax_topk      shared.up_proj
       moe.gate_gather   shared.silu_mul
       moe.up_gather     shared.down_proj   →  sd_buf
       moe.silu_mul      shared_expert_gate →  sg_buf
       moe.down_gather
       expert_mix        →  moe_buf
              \             /
              shared_combine_add  (x += moe + sigmoid(sg)*sd)
```

If you wrote the encoder naïvely you have ~9 `dispatch_bar`s in serial.
Each barrier serializes the GPU. But the two chains have no data
dependency until shared_combine_add. They should run in parallel.

## Implementation

With `MTLDispatchTypeConcurrent` encoder, drop barriers between dispatches
that have NO data hazard, and insert barriers only at fan-in stages.

Restructured:

```c
// Stage 1: fan out — 5 independent dispatches share input (h_buf)
dispatch    (moe.gate_proj_gather)        // writes gp_buf
dispatch    (moe.up_proj_gather)          // writes up_buf
dispatch    (shared.gate_proj)            // writes sgp_buf
dispatch    (shared.up_proj)              // writes sup_buf
dispatch_bar(shared_expert_gate)          // writes sg_buf  +BAR

// Stage 2: silu_mul on BOTH chains (independent)
dispatch    (moe.silu_mul)                // reads gp_buf, up_buf
dispatch_bar(shared.silu_mul)             // reads sgp_buf, sup_buf  +BAR

// Stage 3: down_proj on BOTH chains (independent)
dispatch    (moe.down_proj_gather)        // writes dn_buf
dispatch_bar(shared.down_proj)            // writes sd_buf  +BAR

// Stage 4: expert_mix (depends on dn_buf only; sd_buf already done)
dispatch_bar(expert_mix)                  // writes moe_buf  +BAR

// Stage 5: fan in — combine everything
dispatch_bar(shared_combine_add)          // x_buf += moe + sg*sd
```

The 5 fan-out dispatches in stage 1 launch concurrently. On a memory-
bandwidth-bound model they share the GPU's bandwidth, but each one
keeps the GPU busy while another would otherwise be waiting at a
barrier.

## Real result on Qwen3.6-35B-A3B (M4 Max)

This single restructure: **66.9 → 69.5 tok/s @ n=210** (+3.9%, MLX parity).
Biggest single win late in the optimization series. No new kernels,
no template tricks, no fusion — just barriers removed where no hazard
exists.

## How to spot this opportunity

Look for code like:

```
dispatch_bar(stage_A_op_1)
dispatch_bar(stage_A_op_2)
dispatch_bar(stage_A_op_3)
dispatch_bar(stage_B_op_1)   ← chain B starts; does it depend on
dispatch_bar(stage_B_op_2)     anything in chain A?
dispatch_bar(stage_B_op_3)
dispatch_bar(combine A and B outputs)
```

If chain B reads only inputs that were ready BEFORE chain A started,
the two chains are independent. Interleave them, replace bars with
plain dispatches except at the genuine fan-in barriers.

## When this does NOT help

- If chain A and B are wildly different sizes (one ~10x the other),
  the small one finishes idle while the big one runs anyway — no win.
- If your GPU is already saturated on chain A alone (rare on decode,
  common on prefill).
- If chains share output buffers — you have a hazard, not independence.

## Diagnostic — measure first

If `gpu_busy ≈ wall` (within a few %), you ARE GPU-bound and removing
serial barriers WILL help if hazards permit.

If `gpu_busy << wall`, you have CPU-encode overhead — fix that first
with the 2-deep pipeline (A4).

## Commit reference

`d813cf5 src-metal: parallel MoE main chain + shared-expert chain — 66.9 → 69.5 tok/s @ n=210 (MLX parity)`
