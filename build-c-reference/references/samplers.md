# Samplers: AR decode, diffusion, Fast-dLLM

Read this when:
- Building the outer driver loop in `main.c` (Phase 5).
- The model is autoregressive (standard `model.generate(...)` style).
- The model is a discrete-diffusion / masked LM (Dream, LLaDA, MDLM,
  ...).
- The model supports a cached/block-diffusion variant like Fast-dLLM.
- You need to add a second sampler to an existing C reference without
  forking the tree.

## Contents

1. AR (autoregressive) decode
2. Discrete-diffusion / masked LMs (naive sampler)
3. Fast-dLLM-style block diffusion with DualKVCache
4. Multiple samplers in the same binary
5. Pitfalls specific to confidence-threshold and block-diffusion drivers

## AR (autoregressive) decode

The standard loop:

- Embed prompt → run `forward(0, Lp)` for prefill → argmax last position.
- Loop: embed one id → run `forward(Lp+i, 1)` for decode → argmax →
  emit token → break on stop.

`forward(q_off, Lq)` parameterizes the absolute starting position
(`q_off`) and the number of query positions in this call (`Lq`). The
KV cache is sized to `MAX_CTX * N_KVHEADS * HEAD_DIM * sizeof(bf16)`,
mmap'd or malloc'd at start.

**Same kernel for `Lq=1` and `Lq>1`.** The Metal port may split prefill
and decode later; in the C reference there is one code path.

**Stop tokens** must match the Python reference's actual stop set, not
just `tokenizer.eos_token_id` — see `references/pytorch-gotchas.md`
("model.config vs tokenizer.special_tokens_map").

## Discrete-diffusion / masked LMs (naive sampler)

Discrete-diffusion LMs (Dream, LLaDA, MDLM, ...) replace the AR decode
loop with an iterative denoising sampler. Important consequences for
the C reference:

1. **Bidirectional attention** (`is_causal=False`). No causal mask, no
   sliding window, no sinks. SDPA is just `softmax(QK/√D) @ V`. No
   q_offset, no per-token attention window — every position attends to
   every position.
2. **No KV cache**. Each denoising step re-runs the full forward over
   `[L = prompt_len + max_new_tokens]`. Allocate workspaces once at
   size `L` and clobber them per step.
3. **Mask-token padding**. The initial sequence is
   `[prompt_ids..., MASK, MASK, ...]` where MASK is a special id
   (e.g., 151666 for Dream). Each step replaces some masks with the
   model's predictions.
4. **Logits-shift trick** (per `generation_utils._sample` in many of
   these models): the logits used at position `p` are actually the row
   computed for position `p-1`
   (`logits = cat(logits[:,:1], logits[:,:-1], dim=1)`).
   In C, mirror this by scoring position `p` with row
   `(p == 0 ? 0 : p - 1)`.
5. **Confidence-based scheduling**. Each step picks the top-K most
   confident masked positions and unmasks them. Common confidence
   metrics: `maskgit_plus` (top-1 prob), `topk_margin` (top1 − top2),
   `entropy` (negative entropy). At `temperature=0` all three are
   deterministic — pick one (e.g., `entropy`) for E2E validation.
6. **Skip RNG-dependent algs**. Some samplers offer an `alg=origin` or
   `alg=random` mode that draws from `torch.rand`; you cannot match
   these in C without reimplementing PyTorch's Philox / Mersenne RNG.
   Stick to deterministic algs.
7. **Validation prompt has to leave room for output**. If
   `max_new_tokens=8` is too few and the model decides everything is
   `<eos>`, your "matching" output is `[eos]*8` on both sides — still
   valid bit-equivalent proof, but trivially so. Bump to ≥16 to see a
   real response.

Driver skeleton:

```c
// Pseudocode for a diffusion sampler driver.
for (i = 0; i < steps; i++) {
    n_mask = count(x == MASK);
    if (!n_mask) break;
    embed_gather(x, ..., x_buf);
    forward(L);                          // bidirectional, full sequence
    for each p with x[p] == MASK:
        src = (p == 0) ? 0 : p - 1;
        (id, conf) = score_row(logits[src], alg);
        cand_id[p] = id; cand_conf[p] = conf;
    n_transfer = (i == steps-1) ? n_mask : (int)(n_mask * (1 - s/t));
    topk_indices = topk_desc(cand_conf, n_transfer);
    for each idx in topk_indices: x[idx] = cand_id[idx];
}
```

## Fast-dLLM-style block diffusion with DualKVCache

A common second-generation sampler for the same diffusion model
(Fast-dLLM, parallel-decoding speculative diffusion, etc.) trades
"flat full-sequence forwards" for **block diffusion with a DualKVCache**.
Per-token compute drops by ~30–50% and the algorithm is much more
amenable to GPU scheduling. The kernels are unchanged; only the driver
+ cache layer is new.

What changes vs the naive sampler:

1. **`DualKVCache`**: per-layer post-RoPE K and V buffers sized
   `[max_len, Nkv, D]`, with a shared `offset` across all layers and
   **two write modes**:
   - `append` (`replace=0`): write the current block's K/V at
     `[offset .. offset+Lq)`; advance offset by Lq.
   - `slice-replace` (`replace=1`): write at
     `[cache_position .. cache_position+Lq)`; offset unchanged.
2. **MLX semantics for `cache_position`**: when the Python reference
   uses `rope_offset = cache.offset if cache_position is None else
   cache_position`, mirror that in C with a sentinel:
   `int eff_pos = (cache_position >= 0) ? cache_position : cache->offset;`.
   Then RoPE the block at `eff_pos` and write K/V at `eff_pos`.
3. **Block loop**:
   ```c
   for (b = 0; b < num_blocks; b++) {
       block_start = prompt_len + b * BL;
       block_end   = block_start + BL;
       cache_reset(cache);                  // FRESH cache per block
       // (a) prefetch: full-sequence forward, fills cache from scratch
       embed_gather(x, embed, x_buf, L);
       forward_cached(L, cache, /*pos=*/ 0, /*replace=*/ 0);
       shift_rows_down1(logits, L, V);
       if (x[block_start] == MASK) {
           x[block_start] = argmax(softmax(logits[block_start]));
       }
       // (b) up to steps_per_block iterations of block-local refinement
       for (s = 0; s < steps_per_block; s++) {
           if (no_masks_in_block(x, block_start, block_end)) break;
           embed_gather(&x[block_start], embed, x_buf, BL);
           forward_cached(BL, cache, block_start, /*replace=*/ 1);
           shift_rows_down1(logits, BL, V);        // WITHIN-BLOCK shift
           n_changed = select_confident_updates_greedy(
               x, block_start, BL, logits, MASK, V, threshold);
           if (n_changed == 0) break;              // no-progress guard
       }
   }
   ```
4. **`select_confident_updates_greedy` (one block):**
   - For each position in the block, compute `argmax`, `max_prob`.
   - Replace every **masked** position whose `max_prob >= threshold`.
   - Also force-replace the single **highest-confidence masked
     position** regardless of threshold (so each step makes progress
     even when the model is uncertain).
   - Return the number of positions actually changed.
5. **Why `cache_reset` per block?** Once tokens in earlier blocks have
   been filled in, the K/V cache from before those fills is stale.
   Cheapest correct fix is to throw it away and reprefetch. (Smarter:
   incrementally update only the previous block's K/V, but harder to
   match the reference bit-for-bit.)
6. **Within-block vs full-sequence logit shift**: `_shift_logits`
   (`cat(logits[:,:1], logits[:,:-1], dim=1)`) is applied to whatever
   shape comes out of the forward. In the prefetch call that's the
   full `[L, V]` array (tensor); in block-local calls it's `[BL, V]` and
   the shift is **within the block**, not relative to the absolute
   sequence. Implement it as a row-shift over the actual array, NOT
   as a "use absolute position p-1" trick.
7. **Performance**: the speedup grows with `num_blocks`. For
   `num_blocks=1` (one big block) you still save ~40% from the
   block-local GEMMs being smaller than the full-sequence GEMMs, but
   the cache itself is barely amortized. For `num_blocks=4+` the
   cache pays off properly.

The DualKVCache is itself a small standalone module (~80 LOC in C);
keep it in `cache.{c,h}` so the Metal port can carry it over directly.
Cache stores **post-RoPE** K/V (matches MLX); preserve that contract
through the Metal port — if you instead cache pre-RoPE K and apply
RoPE during SDPA, you have to change the cache layout too.

## Multiple samplers in the same binary

Once you have one working C reference, it's common to discover the
target model has **multiple sampler families** in active deployment
(e.g., upstream HF generation vs Fast-dLLM block diffusion;
auto-regressive vs speculative decoding; greedy vs MTP). Don't fork
the C tree; layer the new sampler on top of the same kernels and
expose `--sampler` on the CLI.

Pattern:

```
./<bin> --sampler upstream  --alg entropy --max-new-tokens N --steps N
./<bin> --sampler fastdllm  --block-length BL --threshold 0.9 --steps N --max-new-tokens N
```

Implementation rules:

1. **All samplers share the same `kernels.{c,h}`**. Only `main.c`
   grows: a second `forward_cached(...)` if the new sampler needs a
   different signature, plus the new sampler function.
2. **Default to the simplest sampler** so existing regression runs
   keep passing without flag changes.
3. **Each sampler gets its own Python oracle script** under `tools/`,
   each with its own `refs-<sampler>/` directory if per-step dumps
   help debug:
   ```
   tools/dump_ref.py               # per-op oracle for kernel tests
   tools/run_ref_<sampler>.py      # E2E oracle, prints generated ids
   tools/dump_ref_<sampler>.py     # optional: per-step `x` snapshots
                                   # for diff-based sampler debug
   ```
4. **Use cross-reference oracle agreement as a stronger signal**:
   if the same C kernels produce tokens that match BOTH a PyTorch
   reference (sampler A) AND an MLX reference (sampler B), that's
   stronger evidence the kernels are correct than either reference
   alone. (Bonus: MLX and PyTorch have independent bf16 numerics, so
   any C bug would have to coincidentally match both.)
5. **You don't need per-op oracle dumps for the second sampler**.
   Token-level (and ideally per-step `x[]`-level) oracle is enough,
   because the kernels are already validated. The new code path is
   pure orchestration. The per-step `x` dumps are valuable purely
   for binary-diff debugging when the token-level diff fails.
6. **`(void)flag_unused_by_active_sampler;`** in `main()` after parsing
   so `-Wall -Wextra` stays clean when one sampler ignores another's
   args.

## Pitfalls specific to confidence-threshold and block-diffusion drivers

- **`shift_rows_down1` must iterate backwards**: a naive in-place
  `for (i = 1; i < n; i++) row[i] = row[i-1]` propagates row 0 into
  every row. Iterate `for (i = n-1; i >= 1; i--) memmove(row[i],
  row[i-1])` so you don't clobber `row[i-1]` before reading it.
- **Forgetting `n_changed == 0 → break`** in confidence-threshold
  samplers (Fast-dLLM `_compiled_select_confident_updates_greedy`):
  the algorithm always force-fills the single max-conf masked
  position, so it WILL make progress as long as masks remain. But if
  the model's confidences saturate (e.g., all remaining masks have
  the same predicted id), each step writes the same answer and you
  loop forever. The MLX reference has this guard; mirror it in C.
- **Fresh cache (`cache_reset`) at the start of each block** in
  Fast-dLLM. After earlier blocks were filled in, the KV cache that
  was built from the original (mask-padded) sequence is stale for
  later blocks. The cheapest correct fix is to reset and re-prefetch.
- **`cache_position = None` vs `cache.offset`**: the MLX cache idiom
  `rope_offset = cache.offset if cache_position is None else
  cache_position` translates to C as a sentinel:
  `eff_pos = (cache_position >= 0) ? cache_position : cache->offset`.
  Missing this and always using `cache->offset` breaks block-local
  refinement because each refinement call must RoPE+write at the
  block's absolute position, not at the end-of-cache.
- **Per-layer cache offset must advance in lockstep**: in a DualKVCache
  where all layers share one offset, advance it ONCE per forward call
  (after the per-layer loop), not per-layer (or you'll get
  inconsistent `Lk` values across layers in the same SDPA call).
  Easiest: pass the new `Lk = replace ? cache->offset : eff_pos + Lq`
  computation INSIDE the per-layer loop and only mutate `cache->offset`
  after the loop completes.
- **Within-block vs full-sequence logit shift**: the
  `cat(logits[:,:1], logits[:,:-1], dim=1)` shift in the Python
  reference is applied to the actual `logits` array that comes out
  of the call — full sequence for the prefetch (`[L, V]`), one block
  for refinement (`[BL, V]`). The shift is **always relative to that
  array's own row indexing**, NOT relative to the absolute sequence
  position. Implement it as a row-shift over the array, not as a
  "use absolute position p-1" trick.
