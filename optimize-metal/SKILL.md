---
name: optimize-metal
description: >
  Optimize a working pure C/Objective-C Apple-GPU Metal LLM implementation
  (./src-metal/) toward pinned MLX prefill, decode and end-to-end performance,
  normally within 5% or faster. Use whenever asked to optimize Metal, match
  MLX, tune GPU kernels, reduce startup latency, or speed up autoregressive,
  hybrid recurrent/MoE, dense BF16 or diffusion LLM inference. Start with
  literal reference source and observed dispatch, not MLX-like substitutes;
  preserve an independent C oracle and diagnose token conflicts honestly.
  Catalog includes SIMD GEMV/qmv4, MMA prefill, online/multi-SG SDPA,
  recurrent time loops, GPU samplers/glue, completion-owned pipelines,
  concurrent chains and parallel pread/residency. Validate same-input
  kernels and full workload IDs, then use bounded counterbalanced timing.
---

# optimize-metal — a source-first path to competitive inference

Make a working Metal inference engine fast without spending days tuning
the wrong numerical graph. The normal objective is pinned-reference
prefill and decode throughput within 5% (or faster), with competitive
startup-inclusive wall time. Agree on workloads and margins first.
An unmet metric remains unmet even if another metric wins.

**Read [pitfalls](references/pitfalls.md) first.** Then use this fast path;
load deeper references only for the current decision.

## Keep two references, with different jobs

- **Frozen scalar C:** independent diagnostic oracle, fixed primitive
  bounds, exact conversion checks, saved inputs/state/reset evidence.
  Leave it untouched. It does not require optimized GPU arithmetic to
  preserve serial C summation, Kahan reduction or forced no-FMA flags.
- **Pinned performance reference:** actual MLX/MLX-LM source, kernels,
  dtype/round points, compiler options, dispatch, lazy demand and
  generation/lifetime behavior. Match this when the goal is MLX parity.
  A hand-written “MLX-like” kernel is not a literal-source port.

Same-input C tolerance, exact native output/ID parity, finiteness,
protocol counts and reset are separate gates. Record PASS/FAIL/NOT_RUN
and scope separately; a new native success never repairs an older C failure.
Do not loosen bounds or regenerate an oracle to accept an optimization.
An independently demonstrated oracle defect needs a separately versioned
correction, not an optimization-side waiver.

If the starting implementation is broken, make a bounded correctness
repair via `port-c-to-metal` before performance work. During optimization,
diagnose newly introduced mismatches here; do not dismiss them as drift.
When building a portability/compiler pipeline, finish the fast reference
before tuning a downstream implementation to reproduce a slow one.

## Phase 0 — freeze the contract and resource window

Create one compact local manifest and phase handoff, using the project's
existing tools/environment (Pixi where required; no alternate installation
or model upload). Gather:

1. Source commits/dirty fingerprints, package and installed-source versions,
   compiler flags, executable and **actually loaded** shader/library hashes,
   generator inputs/sidecars, OS runtime, SDK, device/GPU and memory.
2. Model config/index/tokenizer/template identities, tensor shapes/dtypes,
   packing/group size and exclusions. The model directory name is not
   architecture evidence; a dense FFN may still have recurrent attention.
3. Exact raw input IDs, greedy/tie/normalization/stop policy, requested and
   emitted counts, consumed continuation inputs, cache bounds and reset.
4. Short chat (normally at least 65 generated tokens), roughly 2K prefill,
   and sustained decode (roughly 1.5K generated tokens), unless the user
   requests a different set. Record gaps instead of substituting short
   surrogates for long workloads.
5. Cold/fresh-process versus warm-resident scope, timers, acceptance margins,
   cooldown/run order, diagnostic budget and agreed quiet/resource guards.

Use one GPU/full-model job at a time and one writer/build/test owner per
tree. Budget CPU, RSS/free memory, disk and trace throughput explicitly.
Choose known-compute/load guards before timing; a project may require a
strict process-name guard, but that is not universal proof of contention.
Do not kill unrelated jobs or silently relax guards. Read-only work can
proceed elsewhere; check that “unit” tests do not secretly launch GPU work.

## Phase 1 — study the real reference before tuning

Read [native source and parity](references/native-source-parity.md).

1. Obtain the **exact licensed source release** matching installed packages.
   Confirm model/cache/generation files and native binary provenance.
2. Build a source-to-runtime replacement table for real shapes:
   source anchor/hash → PSO/body/options → live ABI/strides → selector/grid/TG
   → round points → owner lifetime → saved identical-input evidence.
3. Observe representative real operator/model dispatches with bounded,
   untimed instrumentation. A name in a metallib or capability boolean
   does not prove production use. Keep observer/capture hooks out of timing.
4. Port one unchanged hot kernel as a pure C + Objective-C Metal pilot;
   MLX runs separately as oracle, not as a linked inference runtime.
   Compare entire outputs on actual shapes and unchanged C bounds.
5. If the pilot proves source/options/ABI compatibility, migrate the
   **compatible semantic family** (including reducers/helpers/selectors),
   rather than testing a succession of slower serial-order approximations.
   Preserve licensed bodies/notices and explicit, reviewable host adapters.

**Decision branches**

| Evidence | Next action |
|---|---|
| Existing source/identical-input proofs match current pins and shapes | Reuse them; do not repeat migration or pilot work. |
| Already validated competitive or faster-than-MLX engine | Profile the remaining requested goal. Do not rewrite solely for literalness unless the user requires it. |
| Native pilot works, serial-C output differs within fixed bounds | Keep both facts; proceed with native math, not C-order substitutes. |
| Pilot fails on identical inputs | Check packing, options, ABI, strides and round points before model runs. |
| Hybrid engine fails long IDs | Stop performance promotion; use the bounded first-divergence protocol below. Complete known nonnative families with separate proofs; do not invent a cause. |
| Source/runtime is unavailable or unsupported | State the limit; use measured catalog adaptations with explicit nonnative scope, not a claim of literal parity. |
| Native math and IDs pass, prefill wall still loses | Diagnose timer/setup/first-submit behavior before rewriting fast kernels. |

The catalog is a fallback/tuning menu, not an attack order that overrides
observed native routing. Tile sizes, group size, GQA thresholds, precise
softmax fallbacks and recurrent helper math come from this reference.

## Phase 2 — cheap validation, then bounded whole-model controls

For each changed family:

Audit actual typed fixture ranges before timing; native/candidate agreement
alone can hide a shared degenerate input. Exercise successful controller
admission and independent request receipts before expensive model runs.
Reuse existing minimal harness helpers rather than building a new framework.

1. Check complete identical-input outputs against pinned native kernels;
   run unchanged C tolerance/conversion checks independently. Include
   final normalization and vocabulary head, not just internal layers.
2. Cover selector thresholds on both sides, M/N/K tails, causal offsets,
   GQA, strides, aliasing, special values, guards and completed buffer reuse.
   Exercise the **production selector/helper**, not a copied test predicate.
3. Check state across calls and reset. Completion owns constant/token arenas,
   scratch and retired KV buffers; CPU handle replacement is not GPU release.
4. Run the smallest short production control, then the requested long
   controls once the relevant shape/lifetime gates pass. Require all
   workload-specific native IDs, actual counts/schema/dtypes, finiteness,
   successful completion and the stated reset gate before promotion.

Avoid a days-long scalar-C large-prefill capture merely to profile a naive
kernel. Use frozen small/real-input captures and bounded native operator
proofs first; mark missing whole-C evidence UNKNOWN.
Stream large comparisons, retain hashes and sparse first-divergence slices,
and use redzones/reuse controls rather than dumping every tensor repeatedly.
Keep producer disk backpressure and cleanup ownership bounded.

### When outputs disagree

Read [measurement and diagnosis](references/measurement-and-diagnosis.md).
Verify identities first, then use **identical teacher-forced tokens and
state** up to the first divergence:

- Native sampler on candidate logits isolates sampling.
- Native head on candidate normalized hidden isolates head arithmetic.
- Compare the first upstream hidden/cache/state difference, replaying the
  relevant primitive on identical inputs.

Equal tokens for 16 steps, matching failure counts, or small accumulated
whole-model differences do not prove “numerical drift.” Check capacity,
dispatch thresholds, rounding, queue hazards and ownership. Record what
was exonerated and what remains unlocalized. Preserve the original failed
receipt even if completing the native migration later removes the mismatch.

## Phase 3 — profile the remaining gap and change one cause

Read [profiling](references/profiling.md) and the matching catalog entry.
Measure wall plus valid GPU interval coverage per phase; use null when
native GPU time is unavailable. Add host sampling, setup, encoding,
commit-to-GPU-start and final sync/restore clocks as needed.

- GPU interval coverage near wall → investigate actual hot shapes/kernels.
- Low coverage → **unattributed gap**, not automatically slow CPU encoding.
  Separate host breaks, sampler, setup, queue delay, residency/paging and waits.
- KPROF serializes/perturbs execution: use it for bounded attribution,
  not acceptance throughput or an exact uninstrumented cost model.

State one hypothesis, predicted metric, focused checks and stop/pivot
criterion before editing. A practical default is one representative profile,
one pilot and at most two failed diagnostic approaches before changing
the hypothesis; allocate a different budget explicitly when justified.
Screen operator geometry cheaply, not with a full model for every tile.
Use the latest matched profile, not an earlier bottleneck. Batch short
operator calls when needed, retaining every interval and the declared sample
unit. Ordinary build/path/pre-dispatch repairs are not failed GPU experiments;
budget CPU jobs concurrently and keep actual safety/correctness gates firm.
Small wins below measured variability are INCONCLUSIVE, not automatically
good or bad. Native parity is a gate, not proof no further speed is possible.

### Useful general catalog (retain these tools, not old-model assumptions)

| Family | Candidate techniques / references |
|---|---|
| A — scheduling/startup | Batched buffers; persistent/constant rings; completion-owned 2-deep pipeline; concurrent independent chains; parallel pread; measured residency lifecycle. |
| B — GEMV | SG-per-output, aligned bfloat4, qmv4 register reuse, packed uint4/group-metadata amortization, conditional TG X sharing, round-preserving residual epilogues. |
| C — GEMM | BF16-staged SIMD-group MMA; shape/aspect-ratio tile sweeps; native split-K and typed reducers. |
| D — attention | SG score/output work, online/multi-SG merge, native vector/two-pass/generic-GEMM selectors, ILP and model-defined sliding KV. |
| E — MoE | Per-token qmv, sorted-gather prefill GEMM, gate/up/SwiGLU and combine/scatter where licensed semantics permit. |
| F — reductions/sampler | Cooperative/multi-SG RMS, parallel argmax/top-K, exact rounded normalization; diffusion per-row softmax/argmax/confidence. |
| G/H — graph/boundaries | Shape-selected decode/prefill paths, GPU glue, merge dispatch chains, demand-based suffix/head/row pruning and per-layer embedding gather. |
| I — recurrent/SSM | SG-per-state-row/dimension; persistent native time-loop kernels and correctly typed carry/gate/state. |

Detailed entries, empirical speedups, negative results and original commit
attribution remain in [catalog](references/catalog.md) and `patterns/`.
Patterns are inspiration, not drop-ins or native-equivalence evidence.
For dense BF16 read [dense-bf16](references/dense-bf16.md); for diffusion
read [diffusion-llm](references/diffusion-llm.md). Load
[decode/prefill split](references/decode-prefill-split.md),
[parallel chains](references/parallel-chains.md), or
[sliding KV](references/sliding-kv-cache.md) only when relevant.

## Phase 4 — one honest final measurement cohort

Before acceptance, read the [measurement protocol](references/measurement-and-diagnosis.md).

- Pin source, binary, package, loaded libraries, generated sidecars and input
  identities. Header/sidecar changes must rebuild the right products.
- Separate untimed observers/correctness probes from the timing cohort.
  **KPROF must be absent, not `0`**, after environment activation; verify
  the real child's environment before model initialization.
- Use a predeclared counterbalanced cohort (e.g. ABBABAAB, four samples per
  engine), with all samples, median/spread and paired ratios retained.
  Best-of is supplementary only. Do not pool interrupted attempts, replace
  bad samples, or rerun already-successful unrelated cohorts.
- Compare actual work: prefix demand, generated/consumed counts, head calls,
  lookahead and final synchronization. Derive numerator/timer conventions
  from the pinned generator; typically autoregressive decode is G−1.
- Declare subsequent-request timing separately: load once, report conditioning,
  then time reset/cache/conversion/wiring on both engines. Use a same-window
  reference/old/candidate comparison when both readiness and promotion matter.
- Report conventional prefill, setup-inclusive ready-to-first/TTFT, decode
  and **true outer child wall** separately. Document excluded work at every
  ready/timer boundary; moving work outside a timer is not a speedup.
- Stop and retain the attempt on a declared gate violation; obtain a new
  disposition/window rather than relaxing the gate silently.

Nominal weight bytes ÷ advertised bandwidth is only an analytical estimate.
It is not measured memory-controller traffic or a hard stopping condition.
Cache/reuse, metadata, state/KV traffic, instructions and scheduling matter;
an apparent rate above that estimate refutes its assumptions, not physics.

## Persist early; finish with an explicit scope

Checkpoint buildable validated source at phase boundaries, before a long
benchmark campaign. Commit/push only where authorized; do not defer all
persistence for days or turn diagnostic checkpoints into “accepted speedup”
commits. Prefer one coherent semantic-family change per buildable commit,
with separate math, runtime/setup and measurement changes. Keep useful
revertible catalog experiments and negative evidence.

Maintain a compact source manifest, lesson/decision log and durable handoff:
source/binary mapping, passed gates, preserved failures, artifact locations,
next bounded action and explicit resource release. Recover completed work
from saved receipts after service failures instead of rebuilding/rerunning.
Use fresh specialized tasks only when context/objective genuinely changes
and delegation is authorized; avoid constant status polling and sprawling
coordinator traces. Defer unrelated compiler/platform bugs as reproducible,
deduplicated reports; do not build new kinds/backends before the reference
is fast.

Finish when all agreed correctness and performance gates pass, or report
an explicit blocked/partial disposition with the unmet metric and next test.
Document actual commands, identities, full-ID scope, timer boundaries,
median/spread/pairs, limitations and deferred hypotheses in project PERF
documentation. Do not call partial decode parity “optimization complete.”
An explicitly chosen research milestone can retain a promising candidate
with caveats, but failed frozen acceptance gates remain FAIL. A milestone
does not by itself lift an existing downstream-work hold; an explicit
user/coordinator disposition can do so.

## Deeper material

- [Native source/parity protocol](references/native-source-parity.md) —
  source/runtime table, numerical/graph/lifetime gates, boundary tests.
- [Measurement/diagnosis](references/measurement-and-diagnosis.md) —
  first divergence, clocks, cohort acceptance and residency causal probes.
- [Distilled hybrid-model case study](references/native-parity-case-study.md) —
  why the old approach took days, what worked, audited results and open questions.
- [Debugging recipes](references/debugging.md), [pitfalls](references/pitfalls.md)
  and [numbered gotchas](references/gotchas.md) — focused follow-up, not a
  requirement to reread the archive or rerun broad suites for every change.
