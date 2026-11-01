# Literal native source and parity protocol

Use this when matching MLX, migrating a hybrid engine, or deciding whether
an “equivalent” formula is actually the reference. Keep the scalar C oracle
independent. This protocol concerns licensed source/runtime equivalence on
declared shapes, not universal bit identity or machine-instruction equality.

## 1. Freeze what actually runs

Pin MLX and MLX-LM source commits/releases, installed Python model/cache/
generation files, extension/dylib/metallib hashes, compiler and build flags,
OS runtime, SDK and device architecture. Distinguish AOT build metadata from
observed JIT options. SDK support does not prove runtime availability.
Record configuration, sanitizer transformations, parameter dtype/layout and
model exclusions; names such as “dense” do not identify attention structure.
Equal total projection shape does not distinguish head-interleaved query/gate
values from two global halves; verify the actual reshape, offsets and strides.

Obtain full matching public source locally; preserve upstream licenses and
copyright notices when vendoring. Keep original bodies byte-identical where
possible. Expand includes/templates with a small reproducible generator;
do not hide arithmetic edits in it. Hash inputs, expanded outputs and
identity sidecars, including every loaded library. Partition translation
units when original function-constant names/indices collide rather than
renaming native arithmetic blindly.

Use pure C orchestration and a C ABI to Objective-C Metal glue when that is
the deliverable. Do not silently link libMLX/C++ inference to manufacture
parity. Existing dependencies and the platform SDK own setup.

## 2. One replacement table, not a sprawling source tour

For each production family, record this row:

| Field | Required evidence |
|---|---|
| Source | Exact file/function/commit/hash; MIT or other license obligations. |
| Runtime | Observed PSO/body, device path, JIT options/function constants; AOT facts separately. |
| Input/output | Dtypes, packing/group size, shapes, strides, offsets, aliases and intermediate stores. |
| ABI | Live buffer/constant indices, fixed-width fields, alignment/`offsetof`, units. |
| Dispatch | Full selector and thresholds; `dispatchThreads` total extents versus `dispatchThreadgroups` group counts; grid/TG/shared bytes. |
| Demand | Which prefix/head/helper operations execute and which lazy suffixes do not. |
| Lifetime | Owner/view/donation, arena/slot, dependencies, retirement and completion. |
| Proof | Entire identical-input native outputs, fixed C bounds, guards/reuse, actual production-helper use. |
| Status | Native / adapted / legacy / unsupported, plus known test gaps. |

Use bounded operator calls and a representative model diagnostic to observe
real dispatches/options/strides. A precompiled function inventory proves
availability, not execution. Stale encoder argument slots are not live
shader dependencies; read the current declared ABI. Public wait counters
do not count every language-runtime/event wait.

Keep instrumentation out of throughput runs. Sparse counters/source-bound
receipts are usually more useful than multi-GiB captures.
Preflight exact generated asset paths and loaded-library selection before GPU
work; a filename typo is not a kernel failure. When counters change a C struct,
update the matching FFI array size/layout too. Variant evidence must show actual
PSO/source/ABI/grid and dispatch counts, not just capability labels.

## 3. Pilot, then complete compatible families

Start with one hot unchanged kernel on actual K/N/M and saved typed inputs.
Compare the full output with actual pinned native execution, plus frozen C
bounds. If source/options/ABI agree, replace its compatible family, including
split reducers, activations, copy/layout helpers and selectors. Keep per-family
deltas/proofs, not one full benchmark after each individual dispatch.

Critical numerical audit questions:

- Are codes signed or unsigned? Is affine bias separate from ordinary output
  bias? What is the actual group size? A uint32 holds four q8 codes; uint4
  holds sixteen. **Group64 is four uint4**, not one. For q4/mxfp4, inspect the
  specific packing/exponent format instead of reusing q8 arithmetic.
- Does native decode factor `scale*dot(code,x) + bias*sum(x)` instead of
  forming per-weight products? Does prefill stage BF16 dequantized weights?
  Where are split-K partials rounded and which typed strided reducer follows?
- Does RMS round normalized values to BF16 before multiplying learned weight?
  Are sigmoid/SiLU/softplus intermediates BF16 or FP32? “Fewer rounding steps”
  is not automatically the reference.
- Is recurrence g an FP32 **multiplicative gate**, a log-decay, or something
  else? Trace producer and consumer together; do not add another exp to a
  gate already exponentiated. Preserve state precision and native time loops.
- Are Safe/Precise options effective in the actual runtime/cache identity?
  Fast-disabled does not mean contraction-off, nor prohibit explicit native
  `fast::exp`, FMA or `simd_sum`. Keep native expressions/options as observed,
  not the diagnostic C flags.

These are questions to answer from source; the hybrid case study's formulas
are not universal model recipes. Passing a literal pilot establishes only
the covered shapes/values/options, not AOT/ISA equality.

### Source-faithful specialization after native parity

An unchanged native body can expose a useful new tile specialization without
rewriting its arithmetic. Inspect template parameter order: the illustrative
`affine_qmm_t` takes **BM, BK, BN**, so a 64×64×32 tile is encoded 64,32,64.
Retain declared WM/WN, threads, BF16 staging, affine grouping, native BlockMMA
and effective Safe/Precise options. Larger staging and accumulator pressure
can erase a theoretical grid reduction; shape counts alone prove no speedup.

Start with one latest-profile hot shape, not a model-wide tile rule. Test the
actual narrow production selector, off-target/tail/decode/split paths, missing
PSO failure and unchanged reducers. Small shapes do not necessarily use the
same split policy: choose and verify genuinely no-split C fixtures separately.
Full native actual-shape output/reuse checks plus small fixed C cases can
avoid a days-long scalar large-prefill oracle. Then require protected model
IDs/state/counts and per-request/per-forward variant counters before promotion.
Keep baseline and candidate assets/builds isolated and map each committed tree
to its measured binary; exact-shape experimental selectors are not general
compiler heuristics without new proof.

## 4. Port the lazy graph, sampler and lifetimes too

Reference speed includes work that a naive engine unnecessarily performs.
Inspect the actual generator rather than a synchronized “balanced” substitute:

- Cache-only prefix P−1 followed by the last token when the reference does so.
  Demand can prune final-layer Q/output/MLP and final norm/head while retaining
  K/V/cache writes; do not prune recurrent/cache state needed by later tokens.
- Exact sampling sequence, normalization dtype, tie-breaking and token dtype.
  BF16 logsumexp scalar followed by BF16 subtraction can change ties: real
  argmax invariance under subtracting a scalar does not survive rounding.
- Queued lookahead before current yield, one unemitted final forward if native
  requires it, final synchronization and reset/exit behavior.
- Actual emitted versus consumed counts, heads/samplers/cache positions and
  result schema/dtype. Count at real dispatch sites; labels alone are not proof.

Two-deep queues require completion-owned token/output/constant arenas and
scratch. Reuse after completion, not after encoding or yielding. Retain
retired KV storage through queued copies and consumers. Descriptor and data
uniqueness both matter for donation; views keep their parent alive.
Compact recurrent carry when native copies it: a small tail view can otherwise
retain a large prefix allocation. Track buffer owner/offset, shape/strides,
logical bytes, span-elements, physical length and allocatedSize separately.

Submission accounting also has a source-defined unit and identity. The
illustrative pinned g16s policy deduplicates by MTLBuffer pointer and sums
mixed-dtype address-span **elements**, not logical bytes; its predicate
`ops>50 || (sum_span_elements>>20)>50` is not a50MiB limit. Do not “correct”
such units into a new batching heuristic when claiming native parity.

## 5. Boundary sweep before a full model

Use the actual production selector and workspace-requirements function.
Test both sides of each threshold, relevant cross-products and checked
capacity arithmetic, including view-offset subtraction. Include:

- Small-M vector versus matrix selection, K/N alignment, M/N tails, split
  count/divisibility and native BF16 reducers.
- Attention query count × GQA eligibility, causal small prefixes, offsets,
  sequence/head strides, long softmax and one-/two-pass boundaries.
- Workspace query/head/block/dimension axes; block count must come from
  the same native selector used by dispatch, not a fixed constant.
- Multi-chunk prefix decomposition, grow/trim and completion-owned KV reuse.
- Entire Y outputs, partials where available, finiteness, readonly inputs,
  redzones and two completed reuses. Do not extend a core-only guard claim
  to untested full-block helpers.

Use nonuniform attention and BF16-midpoint/nondefault-epsilon fixtures.
Uniform inputs or exactly representable values can hide scaling, round-point
and configuration-wiring bugs even when shapes and short IDs pass.

Audit **actual typed** input min/max, signs, distinct values and staged dequant
span before timing. Unsigned subtraction/negation can wrap an affine bias so
far that BF16 staging removes quantization variation: native and candidate
then agree on unrepresentative data. Cast before signed subtraction/negation;
isolate intentional modular PRNG arithmetic. Include nonzero biases and
midpoints, and check output variation as well as finiteness. A fixture repair
requires a fresh declared timing study; do not reuse its invalid timings.

**Illustrative pinned g16s/D256/GQA6 boundaries:** vector eligibility
`M<=8 && M*GQA<=32` means M1–M5, causal for M>1; M6+ uses native generic
QK/precise-softmax/PV GEMM, not automatically flash attention. Lk1024 begins
two-pass attention; blocks64 at1024,128 through8192,256 above8192,512 above32768,
1024 above65536. Softmax axes above4096 use the looped specialization.
Sweep 1023/1024/1025,4096/4097,8192/8193 and later boundaries with appropriate
query counts. Re-derive all of these for another release/device/model.

## 6. Promotion and evidence scope

Require complete workload-specific native IDs, actual protocol counts,
successful completion, finiteness and the declared reset gates. Keep exact
conversion, fixed C tolerance and native ID gates independent.
Make validators fail closed on missing/duplicate checks, nonfinite values,
truncated payloads, dtype/shape mismatch and nonzero/fatal/interrupted exits.
Do not treat a boolean failure code1 as success via a numeric `<=1` check.
Teacher-forced primitive comparison uses **identical state/input**; a whole
divergent trajectory does not inherit primitive bounds.

Persist inputs/reference hashes, outputs or scoped comparison receipts,
source/binary/options/ABI identity and untested boundaries. If a live checker
did not save candidate bytes, call it producer evidence, not an independent
offline byte comparison. Maintain historical failed results and distinguish
metadata-only changes, source-equivalent changes and actual binary identity.
Do not rerun already-exonerated head/sampler probes or broad unrelated suites.
