# Measurement and first-divergence diagnosis

Use this for late token mismatches, apparent prefill/setup wins, noisy
comparisons and first-submit stalls. A small bounded experiment should
answer one causal question before another whole-model campaign.

## Contents

- [First divergence](#1-first-divergence-ladder)
- [Timers and subsequent requests](#2-timer-dictionary)
- [Clock and instrumentation semantics](#3-validate-clocks-before-believing-profiles)
- [Harness, operator sampling and comparisons](#4-acceptance-cohort-not-best-of-hunting)
- [Residency probes](#5-residencyfirst-submit-causal-probe)
- [Bandwidth estimates](#6-bandwidth-claims)
- [Public telemetry limits](#7-public-telemetry-and-information-limits)

## 1. First-divergence ladder

1. Verify source/binary/loaded-library/options/input/cache identities, tensor
   schema/shape/dtype and full requested ID scope. Preserve failed output.
2. Use identical teacher-forced inputs and state up to the first difference.
   Separate trajectory accumulation from an identical-input primitive test.
3. Replay native sampler on candidate logits. If it chooses the candidate
   ID, sampler differences do not explain that disagreement.
4. Replay native final head on candidate normalized hidden. Exact logits
   exonerate that head invocation, not the upstream hidden generator.
5. Compare first upstream hidden/cache/state differences in forward order;
   replay the implicated primitive with identical typed inputs/state.
6. Check round points, selectors, causal/stride offsets, capacity arithmetic,
   barriers, owner retirement and actual JIT options before proposing causes.

Record “sampler/head exonerated; upstream unlocalized” when that is all the
evidence says. Completing remaining native math may restore IDs without
identifying the first old cause. Do not invent drift, stack, TRANSFER,
paging or FMA explanations. C tolerances passing on isolated inputs are
not permission to waive a mandatory full-ID gate.

For large models, stream comparisons and retain first-difference/sparse
slices plus hashes; guard allocation tails and reuse. Budget tensor spool,
disk headroom, producer backpressure, deadline and owned-child cleanup.
Requested bytes, cumulative trace disk, cumulative allocations, sampled RSS
and physical live peaks are different metrics; shared memory is not two
independent host+VRAM allocations.

## 2. Timer dictionary

Define endpoints and included work in the manifest **before measuring**:

| Metric | Required boundary |
|---|---|
| Outer child wall | Retain actual supervisor monotonic Popen-start, Popen-return and blocking-wait/reap endpoints, not just elapsed duration; includes spawn/import/init/output/teardown, excludes supervisor/environment activation/cooling if so declared. |
| Startup | Separate config/tokenizer, libraries/PSOs, weight reads, workspace/cache allocation, prewire and residency entry. |
| Ready-to-first / TTFT | Precisely define “ready”; list excluded allocations/cache/wiring/warmup work on both engines. |
| Conventional prefill | Follow the pinned reference's convention; report setup excluded by that convention separately. |
| Decode | Exact emitted-count numerator and first-token/final-sync/lookahead endpoints from the native generator. |
| Restore/exit | Final sync, residency restoration, cache/report/teardown lifetimes when excluded by inner clocks. |
| GPU coverage | Valid completed intervals on a compatible monotonic clock; interval union clipped to phase, or null if unavailable. |

Save child-main, ready, first-token and final-sync endpoints too, with clock
identity and compatibility/calibration for joining supervisor/child clocks.
A correct child duration alone cannot reconstruct common spawn-to-first.
Matched `CLOCK_UPTIME_RAW` endpoints enabled that comparison in the residency
screen; Unix stamps are not a substitute without clock calibration.

For autoregressive generation, first output is normally the prefill argmax;
decode often uses `(G−1)/decode_seconds`. Confirm exact pinned behavior:
hidden lookahead can mean more executed forwards than yielded tokens.
For diffusion, requested slots and actual EOS-emitted tokens can differ;
state both throughput numerators rather than conflating slots with tokens.

Fresh process is not boot-cold disk/JIT cache. Warm-resident and fresh-first-use
results belong in different cohorts. Added load time + warmed inference is
an estimate, not measured outer wall; medians of phases are not additive.
One engine's ready may already include workspaces while the other allocates
cache/prewire afterward. Ready ratios are then descriptive, not exact
work-matched prefill parity.

Moving wiring/warmup/page-touch outside conventional prefill can improve
that number with zero benefit to the user. Charge it to setup-inclusive
ready/outer wall. Never subtract independently computed medians to manufacture
parity.

The residual outside a reported inner body includes pre-main/import **and**
later output/teardown; do not label it isolated teardown. Inspect source scope
before interpreting coarse labels such as `pread` or `Metal_JIT`: they may
include allocation, worker scheduling, context or residency work. Diagnose
parallel stages through interval unions/critical path, not summed durations
or subtraction of independent medians.

### Subsequent requests are a separate service scope

Load one model, report a complete real conditioning request, then reset/start
a new request. On **both** engines start before cache/reset, input conversion
and wiring. Parse ingress JSON once per child, not inside only one engine's
timed loop; required native tensor conversion still belongs inside its request.
Preserve actual cold spawn, both requests, audit/inter-request/exit and whole
child costs. Never retime an old cold cohort as this new scope.

“Resident” means logical same-process model retention, not physically pinned
pages. Native fresh cache versus C retained/zeroed KV is a real implementation
difference: disclose capacities/strides/growth/fill counts and charge setup,
rather than force artificial identical work. Short,2K and sustained scopes
remain separate; one fast second short request does not establish long parity.

## 3. Validate clocks before believing profiles

Keep CPU encoding/commit stamps and GPU uptime endpoints on compatible
clocks. Persist encode-start, post-commit submitted, GPU-start/end,
completion, token-ready and sync bounds. Reject nonfinite metrics,
unknown/zero timestamps, impossible order and incomplete owners; NaN can
silently pass a naïve inequality. Validate GPU start against **its owner's
encode-start** with a declared clock tolerance.

The submitted CPU stamp can be **after** commit; GPU execution can therefore
legitimately begin before that stamp. Do not reject that valid case.
Overlapping command buffers and lookahead can straddle TTFT:

- Sum owner durations only as owner-attribution totals.
- Sort/merge intervals and clip to TTFT or the relevant phase for busy union.
- Do not sum overlaps into fabricated exclusive phase time.
- Native GPU time unavailable means null, not zero or an invented estimate.

KPROF per-dispatch commit/wait changes scheduling and timing; cross-profile
absolute deltas are not proof of a production cost reduction. Use focused
attribution to choose the next experiment, then measure uninstrumented.

Metal's public `kernelStartTime/kernelEndTime` describes **CPU-driver scheduling
elapsed**, not GPU arithmetic or CPU-busy time. `GPUStartTime/GPUEndTime` is
execution elapsed including stalls. Read supported properties after actual
completion on compatible clocks. Empty command wrappers can report tiny or
nonsensical values; preserve raw/unknown/excluded status instead of fabricating
a parent-prefix interval from them.

Reconcile actual dispatches by loaded PSO instance as well as text name: several
PSOs can share one label. A purely offline parser correction can reuse successful
raw GPU data if exact counts/intervals reconcile; keep its failed parser/log
and do not replace a failed timing sample.

Use the latest matched profile to bound the gap. If C's valid GPU interval
union alone exceeds native's **entire request** in a losing matched pair,
small reset/host/uncovered delays cannot be the main excess; native GPU
counters are unnecessary for that bound. This does not identify which kernel,
stall or frequency effect caused it. Do not subtract unrelated medians, or
keep tuning yesterday's SDPA/startup bottleneck after current QMM dominates.

## 4. Acceptance cohort, not best-of hunting

Use a cheap short/operator triage screen first. Once source and correctness
are stable, preregister one counterbalanced final cohort, e.g. A=MLX/B=C,
ABBABAAB with N4 each for each requested workload. Declare numeric
materiality margins, timing scope, power/thermal/quiet and memory/disk gates,
cooling and stop policy beforehand. Small N4 observations are descriptive,
not formal statistical noninferiority.

- Pin source commits/dirty maps, binary, generated sidecars, loaded shader
  libraries, runtime options, installed packages and benchmark adapters.
  Check header and grouped-output dependencies to avoid stale products.
- Keep observers/capture/trace/profiling/Python injection out of real children.
  Scrub hooks **after** environment activation; KPROF is absent, not `0`.
  Verify before MLX/model initialization. Do not rebuild inside timing.
- Run exact raw IDs, greedy/template/EOS/count policy and genuine workload
  lengths; require every requested native ID, not just a 17-token prefix.
- Retain every sample, median/min/max/spread and paired ratios. Best-of is a
  supplementary statistic, never the sole acceptance argument. A universal
  “2% is noise / 3% is real” rule is not justified.
- On a declared gate failure, stop and retain a separate interrupted attempt.
  Do not pool its completed samples into a new cohort, replace a failing
  sample, or rerun already-successful unrelated cohorts.
- Guard before/during/after and record sampling limits. A named idle compiler
  process can violate an agreed strict guard without proving distortion;
  neither claim “it ruined timing” nor waive it silently. Arrange an authorized
  quiet window; never kill unrelated work.

Report source-bound scope and every unmet metric. One GPU interval gain,
short-workload speedup or ID control is not whole-model parity.

### Validate admission before paying for a cohort

Reuse existing controllers with the minimum gates for the declared study:

- Require independent IDs, counters, clocks and records for **every** request.
  Never fill missing request fields from a top-level/earlier request dictionary.
  Inherit only an explicit immutable identity allowlist.
- Test omissions as well as wrong values. A baseline absence contract covers
  nested request/forward records too, including forbidden counters set to 0.
  A new variant requires its loaded identity and actual per-forward dispatches.
- Mock successful complete controller admission for every arm, not only helper
  validators/imports. Capture immediate pre-child power in the launch scope,
  not an out-of-scope cooldown local; test admission and saved assessment.
- Gate ordered perf **and** uptime endpoints, identities/calibrations, and the
  agreed active power mode, not merely an unchanged arbitrary mode. Derive
  manifest/handoff/runtime caps from one authority so copied metadata cannot
  silently contradict execution limits.

### Stable operator sampling without selecting the fastest calls

For short variable calls, declare batches long enough to discriminate the
effect. Keep identical serial dispatch/commit/completion policy and workload/
cache scope in both arms, with no inner idle, logging, fills or hashes.
Retain every interval, including each batch's first call. The batch—not every
correlated inner call—is the sample unit. Report host/setup/outer costs too.
A changed batch duration is a new declared study: preserve prior inconclusive
runs separately, do not pool them or infer a frequency cause from timing alone.
Operator gains still require protected model and final workload comparisons.

### Readiness and promotion are different decisions

When both matter, compare reference, old implementation and candidate in one
balanced same-window study. Bind each arm's own cwd/binary/assets and variant
contract; declare chronological directions and within-block pairs explicitly,
including nonadjacent pairs. The example triplet order is not a universal rule.
Freeze candidate/reference readiness and candidate/old promotion gates
separately before measurement. Good medians or functional correctness do not
override cold-cost/pair regressions. Keep severe valid samples; a later normal
diagnostic cannot exclude them or repair their acceptance result. A user may
choose a qualified research milestone without retroactively lowering criteria.

## 5. Residency/first-submit causal probe

A large wall–GPU gap is not automatically encoding overhead. Separate:
setup/ready → residency entry → generation start → encode-start → commit →
GPU-start/end → token-ready → final sync → residency restore → outer exit.
An 8ms encode followed by a 929ms post-submit wait calls for a driver/queue/
residency causal probe, not another CPU encoder rewrite.

First read the exact pinned allocator/queue/generation source. In the
illustrative MLX macOS15+/Metal3 path:

- A persistent residency set is requested and attached to relevant queues.
- Membership/admission uses `MTLAllocation.allocatedSize`, including pending
  and cached allocations; the global capacity is the recommended working set,
  not just logical model nbytes or model-only weight priority.
- Entry adjusts membership/limit and commits; there is no explicit tensor
  reread, page-touch loop, forward, mlock or residency-completion wait.
- Exit synchronizes streams and restores the old limit.
- Free-to-cache is not physical destruction/removal. Distinguish logical
  bytes, buffer length, allocatedSize, set footprint and process RSS.

These source facts do not prove physical page-in latency or a speedup.
For the first prototype, change **only** residency lifecycle, keeping math,
Shared/tracked hazard mode, barriers, loader, queue/submission and work counts
unchanged. Native Shared/Untracked versus candidate Shared/Tracked remains
a separate difference, not permission to remove hazards during this probe.
Keep entry in setup-inclusive ready/wall and check restoration, parallel
allocation/free and in-flight retired owners. An unsupported API/runtime
must fail explicitly.

Use one short/reset/full-ID diagnostic then a relevant longer control if
needed; no new sustained/N4 series until the causal question and source
review warrant it. Report before/after first-submit delay **and** entry/
ready/exit/outer clocks. A prototype without returned full-model results is
pending, even if source or tiny lifecycle tests pass.

If a bounded same-window screen shows collapsed submit gaps but worse
setup-inclusive ready/child time, retain the faster baseline and archive
the functional prototype. Do not repeat broad cohorts to obtain a win or
generalize an N2 short screen to all residency/workloads. Next localize the
baseline's startup/first-command timeline; tiny restore time and coarse
body residuals do not establish teardown, page-in or JIT causality.

## 6. Bandwidth claims

Label weight-bytes/advertised-GB/s calculations “weight-only estimates.”
Recompute traffic for actual demand, metadata, split partials, KV/state and
cache/reuse. Nominal effective bytes/time is not measured memory-controller
traffic; rates above advertised bandwidth imply reused/cache-served bytes or
bad assumptions. Use counters or a controlled access-pattern/timing study
before claiming saturation. Decode parity with MLX alone is neither a
bandwidth-floor proof nor an impossibility theorem for further optimization.

## 7. Public telemetry and information limits

Respect declined administrator access: no sudo, privileged powermetrics,
private API workaround or settings change. Use supported own-task
TASK_VM_INFO/TASK_EVENTS_INFO, rusage and public NSProcessInfo/Metal telemetry
with documented units and phase-bracketed snapshots. Logical compressed-ledger
bytes are not physical compressed storage or GPU residency; process/global
counters and sampled gauges have different scopes.
Equal logical/allocated byte footprints and page-rounded address spans are
not equal resident-page evidence. If a short child ends before resource polling
observes it, zero sampled RSS means NOT SAMPLED, not a verified zero peak.

Public Metal capabilities may expose only timestamps, with stage sampling
supported but dispatch sampling unsupported. Leave frequency, traffic and
utilization UNKNOWN when unavailable. Nominal thermal/default QoS/power
samples do not exclude unsampled clock/stall/global activity; timestamp duration
is not a frequency measurement. Decompression near first GPU/driver execution
is temporal correlation, not causality; a warm request with no decompressions
can still have slower GPU elapsed. If a severe slowdown is not reproduced,
retain its historical result and stop at the information limit instead of
rerunning until a win appears.
