# Case study: a dense hybrid model, from days of tuning to a reusable fast path

This is historical evidence, not a tile/formula/performance prescription
for the next model. A 27B q8 checkpoint with a misleading directory name
declared `qwen3_5` in config: dense FFN, 48 recurrent +16 full-attention layers,
group64 unsigned affine q8, BF16 activations and FP32 recurrent gate/state.

## What consumed days

1. Portability/compiler work advanced while the handwritten C+Metal engine
   was still too slow to serve as its performance blueprint. Optional IEEE
   diagnostics and unrelated compiler failures expanded the critical path.
2. Early tuning preserved scalar-C reduction order instead of studying the
   actual pinned MLX bodies/options/graph. Independent q8 outputs, packed
   loads and RMS stores were useful, but only reached roughly 5 tok/s.
3. “MLX-inspired” qmv/MMA/attention substitutes produced incremental wins
   while remaining far behind native. A long-context profile shifted attention
   to 64% prefill/92% decode, showing why short-model assumptions failed.
4. Full traces, repeated broad testing and interrupted timing campaigns
   obscured the decisive question: what does the real reference execute?
   Service failure also required recovery from saved source/artifacts.

The user then insisted on literal source and actual runtime behavior.
The reusable correction is to do that study and one unchanged-kernel pilot
early, distinguish the C diagnostic from the fast blueprint, complete
compatible native families and postpone downstream compiler tuning.
Aim for a few bounded phases/hours; do not promise a fixed duration.

## What worked

- Pinned MLX0.31.2 / MLX-LM0.31.3 source and installed files; observed real
  dispatches, live ABI, MSL3.2 Safe/Precise options and lazy generation.
  Machine was Apple M4 Max64GiB, macOS15.8; SDK26.2 was **not** the runtime OS.
- An unchanged native qmv pilot matched **311,856 outputs bit-for-bit** to
  MLX. Sixty-three differed from C but passed unchanged C linear bounds.
  This directly falsified “native speed requires serial C order.”
- Ported native affine factoring, BF16 qmm/split reducers/RMS/activation
  round points, multiplicative FP32 g, persistent recurrent time loop,
  attention/RoPE/KV/glue, sampler, cache-only prefix and queued lookahead.
  Production remained pure C + Objective-C Metal glue, not libMLX inference.
- A long hybrid mismatch was localized past sampler and head by identical-
  logits/hidden replay. Hidden differed upstream. Finishing native math
  removed it, but the **first earlier cause was not isolated**.
- Review caught header/sidecar dependencies, incomplete ID gates, invalid
  clock acceptance, >8192 workspace sizing, >4096 looped softmax and
  GQA6 M2–M5 vector dispatch. Focused threshold controls fixed these before
  final all-workload timing.
- Preserved the scalar C80 native-comparison stress with only94/96 positions
  and16/17 predictions matching, a baseline failure.
  New native all-workload success never retroactively repaired that result.
- Reused completed evidence, kept per-family deltas and eventually sealed
  source plus benchmark-to-commit mapping. The lesson is to persist such
  buildable phase boundaries earlier, not wait for days of measurements.

## Audited pre-residency baseline cohort

Fresh-process A=MLX/B=C+Metal ABBABAAB, N4 each per workload:
**24 samples,13,328 generated IDs**, all requested native IDs verified.
All rates are tokens/s; cells are **C / MLX medians**.

| Prompt / generated | Conventional prefill | Decode | Actual child wall, s |
|---|---:|---:|---:|
|18 /65|14.2481 /61.2749|16.5298 /16.5560|7.6940 /10.6999|
|2048 /65|205.4146 /230.4752|16.1762 /16.2120|17.2552 /19.0691|
|29 /1536|23.7556 /96.7974|15.1392 /15.2559|105.4185 /107.1877|

Decode was within1%; wall was28.09%,9.51%,1.65% lower. **Conventional
prefill still failed the within5% target.** Native's conventional clock
excluded wiring entry; C charged first-use cost. Ready-to-first ratios
were descriptive because C ready already included workspaces while native
still allocated cache/prewire. This is not unqualified parity, formal
population noninferiority, ISA equality or measured bandwidth saturation.
Native GPU busy was unavailable/null. All interrupted attempts stayed
separate; completed unrelated cohorts were not rerun.

## Open hypotheses, not lessons claimed as successes

A short sample showed about8ms encode/commit stamping then929ms until
GPU execution. That is not evidence of slow CPU encoding. Source inspection
found a native ResidencySet/global-limit lifecycle with allocatedSize
admission, pending sets, queue attachment/commit and sync/restore; no tensor
reread/page-touch/forward/wait at entry. A tracked-hazard-preserving
prototype built and passed focused lifetime checks. At an **initial
checkpoint**, Battery Power blocked all full-model probes (0/3 launched).
AC was subsequently restored and bounded probes resumed; their results
were pending when this case was recorded. No later performance or causal
result is claimed here. The reviewed source baseline was
persisted and publication verified where authorized; the prototype remained
an unaccepted experiment.

Native Shared/Untracked allocation, candidate Shared/Tracked allocation,
loader/cache and submission granularity still differed. Nominal full
weight bytes were not measured memory-controller traffic. Scalar-C80 failure,
earlier mismatch causality, physical memory peaks and downstream Fortran
real-model/GPU acceptance remained separate limitations.

## Carry-forward checklist

Freeze → inspect literal source/dispatch → unchanged pilot → whole compatible
family → threshold/lifetime/identical-input gates → all workload IDs →
one final counterbalanced cohort → diagnose setup separately → persist.
Keep manifests/handoffs compact, failure states honest and deferred work
off the performance critical path.
