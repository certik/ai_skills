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

## Bounded residency result and remaining hypotheses

A short sample showed about8ms encode/commit stamping then929ms until
GPU execution. That is not evidence of slow CPU encoding. Source inspection
found a native ResidencySet/global-limit lifecycle with allocatedSize
admission, pending sets, queue attachment/commit and sync/restore; no tensor
reread/page-touch/forward/wait at entry. A tracked-hazard-preserving prototype
then passed three diagnostics: short17/reset, short65 and2K/65 exact
IDs/counts/lifetimes. First-submit gaps collapsed, but explicit residency
entry moved cost before conventional prefill, not out of user-visible time.

A separate same-window **C-baseline versus C-prototype short65 ABBA screen**
had four valid samples, N2 per engine, with all65 IDs/counts/source/power
checks passing. It is not the24-sample MLX cohort above. Lower duration is
better; ratios below are prototype/baseline:

| Metric | Ratio of medians | Screen disposition |
|---|---:|---|
| Outer child wall |1.0775|slower; both adjacent ratios1.120/1.033|
| Ready-to-first |1.1236|slower|
| Decode duration |1.0005|nearly unchanged|

**LOSS at this bounded screening scope:** retain the faster production
baseline and archive the functional prototype locally, without promoting it.
This is not a formal/N4/all-workload regression or proof residency is
universally harmful. The original diagnostics' restore cost was tiny
(about0.7ms short and3.85ms2K); no teardown, physical page-in or JIT cause
was established. Next diagnose the retained baseline's startup and
first-command timeline, not blindly rewrite its loader or hazard policy.

Native Shared/Untracked allocation, candidate Shared/Tracked allocation,
loader/cache and submission granularity still differed. Nominal full
weight bytes were not measured memory-controller traffic. Scalar-C80 failure,
earlier mismatch causality, physical memory peaks and downstream Fortran
real-model/GPU acceptance remained separate limitations.

## Subsequent-request scope and the QMM64 experiment

A matched-byte resource-count/parent-view pilot remained inconclusive, so
object count did not justify a production loader/consolidation rewrite.

A new comparison loaded once, reported the complete conditioning request, then
timed a reset/new request from **before** cache/reset/conversion/wiring on both
engines. “Resident” meant retained model, not physical page pinning. Native
created a fresh cache; C retained/zeroed KV. These real setup/stride/count
differences were disclosed, not artificially erased; ingress JSON was parsed
once per child. The original cold cohort above was never retimed.

The two-arm study covered 16 children, 32 requests and 2,080 exact IDs. Short
second-request prefill was 80.07 versus 67.74 tokens/s, but 2K was 181.34 versus
196.10: the 2K median gate **failed**, and other 2K gates had losing pairs.
A good short result was not a substitute for long-workload readiness.

Review caught earlier-request evidence fallback, omitted clock/calibration
fields, stable-but-wrong power mode and an undefined pre-child power receipt
in successful admission. Omission and complete mocked-admission tests repaired
the harness before the campaign. The lesson is to reuse/test the full success
path, not build more framework or repeat costly runs to discover tooling errors.

### Profile the current bottleneck, then validate meaningful inputs

Normal matched 2K C coverage was about 99.7%; reset took roughly 14ms, uncovered
generation about 30ms and initial submit delay about 2ms. In losing matched
pairs, C GPU union alone exceeded native's entire request, so those small
host/setup delays could not explain the excess without any native GPU counter.
The next isolated profile attributed 94.53% to q8 QMM/folds and about 1.6% to
SDPA. Its elapsed intervals included stalls and changed synchronization/cache
behavior; they did not prove arithmetic, frequency or retained-KV causality.

One hot shape (M=2047,N=17408,K=5120; 126 calls, 4.181s isolated warm elapsed)
motivated a native-body 64×64×32 QMM specialization. Template order was BM/BK/BN,
so 64,32,64; WM/WN 2×2, TG128, BK32 and BF16 affine/native arithmetic stayed
fixed. Staging doubled 5120→10240 bytes; grid groups fell 34816→8704. This was
an exact-shape experiment, not a general compiler tile rule.

An initial unsigned-negation bug wrapped bias to 4,194,304 and BF16 staging
collapsed quantization variation. Both kernels agreed, but those timings were
unrepresentative and could not justify promotion. Corrected typed range/sign/distinct/
dequant-span/midpoint audits preceded fresh timing. Complete actual-shape
native output scans, M=2048 tails, small K=64/no-split C cases, guards/reuse and
actual narrow/off-target/split/fail-closed production paths passed without a
days-long scalar 2K oracle. Live complete scans were not advertised as independently
saved full payload comparisons.

Single-call screens remained inconclusive. A fresh declared study used eight
batches of 32 serial completed calls, no inner idle/log/hash/fill, retaining all
256 intervals including first calls. It gave 6.58–10.63% paired reductions,
median 8.19%. The sample unit was eight batches (N4/arm), not 128 independent
calls per arm; this promising operator result was still not a model gain.

### Three-arm acceptance and public information limits

Protected model diagnostics passed before a same-window MLX/old-C/candidate
comparison with balanced triplets, own binaries/assets and explicit within-block
pairs, sometimes nonadjacent. Readiness versus MLX and promotion versus old C
were separate frozen decisions. The 24-child/48-request study verified 3,120 IDs:

| 2K subsequent-request prefill | Median tokens/s |
|---|---:|
|MLX|185.040938|
|Old C|182.843067|
|QMM64 candidate|194.002953|

The candidate median was 6.10% above old C and 4.84% above MLX, yet **both overall
gates failed** from cold-cost medians and losing pairs. A valid severe sample
(112.709 prefill, 8.351 decode tokens/s; 58.993s child) remained included.
Correct outputs and a good median did not override the agreed regressions.

A bounded public-only baseline/candidate two-request diagnostic verified 260
IDs but did **not reproduce** the severe slowdown. Own-task VM/event/rusage
and public thermal/QoS/Metal evidence had documented units and sampling limits;
about one million cold decompressions near first GPU/driver execution were
correlation, not cause. Warm zero-decompression prefixes still had longer GPU
elapsed. Logical compressed-ledger bytes were not physical residency.
Public counters exposed timestamps, not frequency/traffic/utilization; declined
admin access was respected. The information limit did not justify excluding
the old sample, promoting the candidate or rerunning until a win. Readiness
and promotion remained failed; no Fortran transition followed.

## Carry-forward checklist

Freeze → inspect literal source/dispatch → unchanged pilot → whole compatible
family → meaningful typed-input/threshold/lifetime gates → stable all-call
operator sampling → protected model IDs → one appropriate final comparison →
diagnose only the remaining bounded question → persist.
Keep manifests/handoffs compact, failure states honest and deferred work
off the performance critical path.
