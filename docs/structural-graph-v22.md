# SALT-HLS structural graph V2.2

**Latest structural representation on `main`: V2.2 RC3 — technically frozen;
independent adjudication pending.** V2.1 remains a frozen historical baseline. Explicit `--revision 2.2` selects schema
`salt-hls-mlir-structural-v2.2`. There is no compatible learned tensor encoder
or checkpoint. Production V1 models, memory, routing and refinement stay frozen.

Latest RC3 artifacts: [manifest](graph-release-v22-rc3.json),
[archive](../GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz)
and [checksum](graph-release-v22-rc3.sha256). The preserved
[RC2 manifest](graph-release-v22.json) and V2.1 release remain historical baselines.
RC3 uses the same schema, with explicit `semantic_revision: rc3`. Old files
without that marker retain their original query semantics. New generation can
select `--semantic-revision rc2` to reproduce the historical behavior.

## One compiler graph

V2.2 extends the [existing representation](structural-graph-v2.md), retaining
its six categories: operations, SSA values, constants/auxiliary features,
control regions, memory objects and action points.

```mermaid
flowchart TD
  F[Function] --> R[Real regions and blocks]
  R --> O[Operations and loops]
  O -->|defines| V[SSA values and block arguments]
  V -->|operand position| O
  O -->|concrete read/write| M[Memory object]
  V -->|storage-preserving view| M
  C[Exact call UID] -->|argument/result binding| V
  A[Action point Lk] -->|one verified target| O
  A -->|one verified target| M
```

The directed edge vocabulary remains the existing control/region, SSA,
loop-state, memory-access/view/alias/dependence, call/return and action-target
vocabulary. Full globals are retained as symbol-keyed compiler definitions;
live global objects also retain their definition attributes. Canonical facts
contain no neural reverse edges. Compact graphs losslessly fold auxiliary
feature tokens, preserving actual constants, SSA values and compiler evidence.

No requested FPGA, clock, resource budget, objective, QoR or optimizer-selected
pragma assignment enters generation. Compiler module attributes such as the
host data layout describe frontend lowering, not a deployment FPGA.

## Semantic changes

**Action identity.** A matching native `mailohls.action_id` or exact compiler
declaration location is required. RC3 enumerates native candidates before selection; it does not ask the V1 heuristic to choose a target first. Exact annotations outrank declaration matches, uniqueness and rooted storage are required, and shapes only filter inconsistent declaration candidates. Shared source-use lines are context, not
allocation identity. Static/global binding requires an exact function/declaration location and a unique compiler-symbol storage root. This can identify a declared static object even when V2.1 cannot recognize its class-lowered accesses.

Both external Kalman metadata variants register L7 (`k_arr`, source line 63)
and L8 (`calc_temp_arr`, line 64). Their frozen IR has no independently bound
allocation for either. V2.1 assigned L8 to the L6-tagged `p_arr_1` allocation
at line 62, using overlapping use lines 74/80. V2.2 rejects that unsupported
assignment. The surviving `arith.divf` at line 76 and scalar uses at lines 78/80
do not prove separate L7/L8 memory. No transformation receipt proves elimination:
the registrations remain unresolved. Development metadata never registered
these two labels. Absolute `Lk`/slot `k-1` identities are never renumbered.

The same issue occurs in `rodinia-hotspot-2-pipeline` and
`rodinia-hotspot-3-unroll`: L1–L4 and L6 were mapped to the allocation tagged
L5 (`temp_center`, declaration line 27). Their own declarations are at different
locations; shared stencil-use lines do not prove identity. Those ten mappings
are also rejected. All 606 registrations remain accounted for, with 592 matched
and 14 unresolved. This is a correction, not an attempt to maximize coverage.

**Storage roots.** Native function-local LLVM GEP, pointer bitcast/address-space
cast and Polygeist memref/pointer conversions preserve underlying storage.
They do not prove equal addresses, overlap or dependence. Exact base/index
operands are retained. Integer-to-pointer conversions and loaded pointer
fields needing invocation-aware reaching-store/context analysis remain conservative. RC3 proves exact-address loads from nonescaping local pointer cells only when all reaching stores are covered by same-block order or structured conditional branches. Overwrites replace earlier definitions. Helper escapes, unsupported CFG paths and uncertain aliases leave candidate sets incomplete. Types,
names, equal shapes, declaration order and global may-alias closure are not
storage-identity proofs. Select/conditional alternatives stay possible unless
all alternatives establish the same unique object.

Concrete accesses survive unresolved roots; call effects remain summaries.
Memref subview offsets are relative to their source view, while reinterpret
casts use underlying-storage offsets. Static/dynamic offsets, sizes, strides,
types and original attributes are retained. LLVM access records additionally
retain pointer-expression chains and their induction-loop dependencies. Empty
direct memref-index fields on LLVM accesses are not independence evidence.

Pointer path tags describe storage derivation, not a normalized byte address.
In the pinned descriptor ABI, `polygeist.memref2pointer` uses the aligned pointer
plus the source descriptor offset (or pointer/address-space casts in its pointer
ABI). The source view and indices remain available; a path tag never implies
zero offset. Composed addresses and element overlap remain unproven.

**Calls.** Bindings retain exact call UID, caller/callee, operand/result position
and actual/formal SSA identities. Returned-memory reasoning is invocation
scoped. Shared helpers do not merge caller arrays. Recursive, external and
incomplete calls remain conservative. RC3 substitutes ambiguous actual roots lazily at each exact invocation. Both A and B remain possible, each with its call UID, actual/formal position and compiler witnesses. `root_candidates_complete` distinguishes a closed set from known alternatives with unresolved paths. Completeness is relative to the represented invocation: an unbound formal remains symbolic, not an enumeration of every caller allocation. Arguments are not combined into a Cartesian product; unrelated callers are not merged. LUD L2–L5 is checked in context
`lud_diagonal:op5` and rejected in the empty helper context.

**CFG and SSA.** The pinned Python Operation wrapper lacks `successors`.
The legacy compatibility helper silently returns an empty list. V2.2 uses
the existing `mlirOperationGetSuccessor` C API and independently checks actual
successor blocks/positions. Unused block arguments formerly removed by V1
cleanup retain their native ownership before cleanup and become canonical
`declares_argument` edges. No synthetic value or active action is invented.

For memory-valued SCF/Affine for-results, the initial buffer also reaches the
result when there are zero iterations. V2.2 records this conditional
`loop_zero_trip_result` SSA fact alongside the yielded/backedge buffer. If they
differ and execution count is not established, the result has possible roots,
not a false unique yielded root. SCF while-results retain the separate
condition-forwarding semantics. This additional adversarial correction is in
RC2; the earlier local candidate remains preserved and superseded.

**Dependence evidence.** RAW/WAR/WAW direction, scope, intervals and certainty
remain distinct. No-alias and explicit native no-dependence conclusions remain
negative analysis records, never positive connectivity. Unknown or an absent
edge is not independence. An interval is not an exact distance.

AES's 43,236 possible dependencies are distinct outcomes: 23,124 involve
different potentially aliased SSA storage, 20,032 fail affine analysis and 80
involve non-affine accesses. Root equality does not compose access maps or
prove independence. V2.2 retains every dependence record and does not sample,
deduplicate or promote these outcomes. The native analyzer stops at the first
successful depth; other carrying depths remain unverified.

All 20,032 AES affine failures lack a common block within their function affine
scope. This follows the pinned `getCommonBlockInAffineScope` rule, checked
against the native-validated block/region ownership: 9,984 pairs each occur in
`ByteSub_ShiftRow` and `InversShiftRow_ByteSub`; the remaining 64 involve
`encrypt`/`decrypt` loop/body paths. The pinned dependence routine returns
Failure at this CFG boundary. This explains an analysis limitation, not an
absence of dependence. `details/AES_failure_contexts.json` preserves the groups
and exact native source references.

Native view records previously inherited a C++ `Block*` map's address order.
V2.2 sorts this unordered list by SSA identity, retaining its entire multiset
and multiplicities. Semantic operand order and native alias/dependence outcomes
are unchanged. The original compiler binary and V2.1 archive are untouched.

**Numerical fidelity.** Source template spellings are lexical evidence, not
AST-verified substitution. Native SSA/element types are separately checked.
Unknown source/compiler correspondence, rounding, overflow and template
defaults stay unresolved. AP analysis stubs are not bit-accurate Vitis types;
an `ap_uint<256>` source spelling does not make an `i64` lowering hardware-exact.

## Validation and release boundaries

The original 63 source/metadata/header/raw-MLIR snapshots are authenticated:
55 development kernels and eight external snapshots, including all three
Kalman contexts. JPEG, Rosetta and two GEMV variants stay excluded. Current
working-tree sources cannot silently enter this cohort. The original raw MLIR
is reused exactly, isolating graph changes from frontend changes.

An independent live-MLIR walk checks every function operation, SSA value,
operand/result occurrence, type, block argument, region ownership, immediate
loop hierarchy, CFG successor, call/return binding and loop-state forwarding.
Global symbols/attributes are checked independently. Native memory-effect
occurrences are compared with concrete accesses independently of root guesses.
Portable checks validate witness fingerprints and evidence chains. Corrupted
operands, call UIDs, roots, certainty, directions and carrying scopes are rejected.
Both native generation runs and canonical/compact public queries must agree.

This is not exhaustive mathematical proof of native dependence analysis.
Complete mechanical classification plus two SHA-256-selected examples per
result/reason/RAW-WAR-WAW stratum produce draft review worksheets. Neither these
nor the previous real-source annotations are independently adjudicated ground
truth. Publication-level precision/recall is unavailable.

The initial 63-case run found nine unordered-view determinism failures; it is
preserved in `GNN_branch/MLIR_graphs/v22_corpus/`. The first corrected pass is
preserved in `validated/`; the current RC2 run is in `validated_rc2/`. Its `validation.json`, detailed
`comparison.csv`, `comparison.json`, `details/` evidence and
`V22_SEMANTIC_AUDIT.md` distinguish all pass/fail/pending cases and release gates.
Candidate packaging never declares the representation final.

## Preserved RC2 corpus comparison

All **63/63** corrected snapshots passed input authentication, native occurrence
checks, repeated native generation, exact compact reconstruction and public
query equivalence. No snapshot was omitted. Aggregate changes:

| Property | Frozen V2.1 | V2.2 candidate |
|---|---:|---:|
| Canonical nodes | 30,874 | 30,634 |
| Canonical edges | 115,316 | 115,783 |
| Registered / matched actions | 606 / 604 | 606 / 592 |
| Unresolved registrations | 2 | 14 |
| Native operation / SSA entities | 9,730 / 7,657 | 9,730 / 7,670 |
| CFG successor edges | 0 | 43 |
| Proven concrete-access relationships | 2,813 | 2,973 |
| Possible / unresolved access relationships | 0 / 182 | 32 / 6 |
| Proven / possible dependencies | 1,911 / 51,659 | 1,911 / 51,659 |
| Native negative dependence/alias conclusions | 150,439 | 150,439 |
| Duplicate native dependence query records | 0 | 0 |

Of the original 182 unresolved access occurrences, 160 now have proven storage
chains, 16 have possible alternatives and six remain unresolved loaded-pointer
accesses in Kalman helper functions. The 16 alternatives produce 32 edges;
they do not create new physical accesses. There are 2,821 affine/memref
load/store operations plus 174 LLVM load/store operations, totaling 2,995
concrete occurrences in both versions. The earlier 2,821 count excluded LLVM.

All 241 removed memory-object nodes were unresolved placeholders replaced by
supported root evidence. No established allocation/global/formal identity was
removed. Twelve unsupported action nodes were removed and thirteen native SSA
arguments restored, explaining the net node change. New storage/view and CFG
facts explain edge growth; no dependence edge was removed or promoted.

The native walk also checks 13,712 operand occurrences, 6,205 result occurrences,
1,089 blocks, 318 loop-state occurrences and 21 global definitions. This is
implementation consistency plus native evidence, not a proof of every analysis
result. The final 63-case generation/validation run took approximately 513 seconds,
including two native generations per snapshot, excluding subsequent tests.

The final broad graph/audit/domain/refinement suite passed **224 tests**;
the focused RC2 suite passed **36** (included in that total). Native analysis
regressions passed with the original binary. The requested full worktree suite
also ran: **122 passed, two failed** in pre-existing uncommitted external
recovery tests outside this cohort: Rosetta expects 12 matches but the retained
V2.1 path finds 11; the JPEG marker harness references static/nonexistent symbols
as externs and does not link. These were reported, not deleted or weakened.

Gates A (reproducibility), B (mechanically supported structure/fixtures) and D
(complete registration accounting) pass. Gate C still needs independent
adjudication of the draft semantic/dependence worksheets; Gate E is partial
because of the reported recovery-test failures. The disposition is
**V2.2 RELEASE CANDIDATE — remaining evidence required**, not FINAL.

After packaging, the extracted archive was independently checked with its own
archived Python code: **63/63** canonical/compact graphs passed validation,
native-witness consistency, reconstruction and every public action/loop query,
without loading native MLIR bindings. The final RC2 portable read-only check took
100.44 seconds. Every bundled checksum passed.

## RC3 changes and review boundary

The evidence-first resolver fixes a coverage failure on Rosetta's static
`z_buffer`: explicit RC3 has 12/12 targets; V2.1 retains its historical 11/12.
The recovery workload remains outside the frozen 63. JPEG's recovery failure
was its test harness: `ReadBuf` has file-local static linkage and the two table
names must include `comps_info`. Including the benchmark in the wrapper's
translation unit fixes the harness without changing the benchmark algorithm.
Both original/recovered implementations are compared, including 65,536 words.

RC3 adds native-verified adversarial MLIR fixtures for dominating/overwritten
pointer stores, two conditional stores, missing branch initialization, helper
writes, uncertain cell aliases, nested calls with two possible caller roots,
SCF/Affine conditional results, SCF while and zero/positive-count for-results.
The for-root analysis deliberately retains a sound union of initial/yielded
buffers even for known counts; it is not exact last-iteration alias analysis.
The complete/incomplete set flag does not establish element overlap.

AES analysis is unchanged. No common affine-scope block is recovered by bypassing
the pinned checker's precondition. No unknown is turned into `no_dependence`.
The first-successful-depth limitation and all distinct RAW/WAR/WAW facts remain.
The `memref2pointer` storage tag now explicitly mentions the source descriptor
offset; it still does not claim normalized addresses or equal elements.

The core counts remain the same as RC2: 30,634 nodes, 115,783 edges, 592/606 matched registrations, six unknown concrete access relationships and 51,659 possible dependencies. No corpus public-query proven relationship was lost, and the new ambiguous-call handling adds no relationship in this cohort; its improvement is established by adversarial fixtures.

RC3 validation completed **267 tests with no failures or skips** (53 V2.2, 124 graph/audit/domain/recovery, 90 refinement/device/selection), plus the unchanged native-analysis regression script. All 63 snapshots passed two separate-interpreter native generations and canonical/compact query checks. The final release graphs match the earlier complete RC3 pass byte for byte.

The RC3 audit package contains `V22_RC3_SEMANTIC_AUDIT.md`,
`V22_ACTION_BINDING_AUDIT.md` (all 14 RC2 registrations),
`V22_POINTER_ROOT_AUDIT.md` (all six RC2 unknown accesses),
`V22_INTERPROCEDURAL_AUDIT.md`, `V22_RC2_VS_RC3_COMPARISON.csv` and
`V22_REGRESSION_REPORT.md`, with machine-readable witnesses in `details/`.
Two native generations use separate interpreters. The failed initial RC3 check
is preserved: it compared live Python tuples with JSON lists. Canonical JSON
comparison checks the actual serialization contract, with a regression test.

Mechanical native checks, independently specified adversarial assertions and
source-compiled fixtures are separate evidence levels. Machine-produced real
program/dependence review worksheets are **not human adjudication**. Gate C
remains open until an independent reviewer resolves those records. An RC label
does not imply a known incorrect proven relationship or hardware-exact arithmetic.

## Reproduction

Use project Python/NetworkX for orchestration and the pinned native Python/
Polygeist environment for MLIR. No GPU, model loading, training or HLS runs.

```bash
python -B -m pytest -q tests/test_mlir_graph_v22.py
python -B -m pytest -q tests/test_mlir_graph_v2.py tests/test_structural_audit.py tests/test_v21_refinement.py tests/test_directive_domains.py
python -B -m experiments.structural_audit.finalize_v22 --mode diagnose
python -B -m experiments.structural_audit.finalize_v22 --mode generate --repeat-native
python -B -m experiments.structural_audit.finalize_v22 --mode summarize
python -B -m experiments.structural_audit.finalize_v22 --mode package
```

Override `--build`, `--native-python`, `--output-dir` or `--timeout` as needed.
Changed code/input/compiler identities require a fresh output directory.
`V22_RECOVERY_INPUTS.json` separately authenticates the local JPEG/Rosetta/GEMV regression inputs; they are not extra snapshots in the frozen cohort. The bundled `V22_PORTABLE_VALIDATION.py` checks extracted graphs using archived code and project Python without native MLIR bindings.

The default fresh RC3 output is `GNN_branch/MLIR_graphs/v22_corpus/validated_rc3_release/`. Resume authenticates completed outputs. Failures do not omit later snapshots.
The candidate package contains sources, metadata, raw MLIR, both graph views,
generation/validation code, compiler identities and checksums. The original
V2.1 archive/checksum is never overwritten. Transitive system include closure
and exact Vitis arithmetic are not claimed.

For individual sources, the existing CLI accepts:

```bash
python -m GNN_branch.mlir_graph_v2 --revision 2.2 --semantic-revision rc3 --source path/to/kernel.cpp --metadata path/to/kernel_info.txt --cgeist /path/to/pinned/cgeist --output /fresh/graph.json
python -m GNN_branch.mlir_graph_v2 --mode validate --input /fresh/graph.json
python -m GNN_branch.mlir_graph_v2 --mode compact --input /fresh/graph.json --output /fresh/compact.json
```

## Future integration contract

| Fields | Meaning | Later consumer |
|---|---|---|
| Operations/types/SSA/regions/shapes/indexing | Compiler facts | Separate tensor encoder |
| Typed accesses/views/aliases/dependencies and certainty | Evidence-bearing relationships | Refinement queries and evidence-aware encoding |
| Call UID, action Lk, SSA IDs, locations, hashes | Alignment/provenance | Matching/audit; exclude raw IDs as learned shortcuts |
| Possible call-scoped roots/completeness, unresolved-action masks, failures, unknown trip counts | Uncertainty rather than zero | Explicit missingness/certainty features |
| Source/lowered numerical fidelity | Limits on hardware interpretation | Prevent unsupported width/cost claims |
| FPGA, objective, clock, budget, candidate pragmas | Separate request/design inputs | Later conditioning/ranking; never overwrite compiler facts |

Neural reverse edges and feature IDs belong in reproducible derived tensor
views. A matched-capacity, fixed-split V1/V2.2 learning comparison is a later
experiment, after semantic evidence and regression issues are addressed.

Semantic references: official [LLVM](https://mlir.llvm.org/docs/Dialects/LLVM/),
[MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/),
[Func](https://mlir.llvm.org/docs/Dialects/Func/),
[SCF](https://mlir.llvm.org/docs/Dialects/SCFDialect/) and
[Affine](https://mlir.llvm.org/docs/Dialects/Affine/) documentation. The pinned
source APIs, rather than newer documentation syntax alone, determine this build.

## Final verification and consumer contract

The [release review](V22_FINAL_RELEASE_REVIEW.md) and
[independent-adjudication worksheet](v22-independent-adjudication.json) separate
new checks from inherited RC3 receipts. The immutable RC3 archive and checksum
are preserved. No corpus graph was regenerated for this verification.

A native adversarial fixture exposed a **query-only** completeness defect:
an incomplete root set forwarded through loop SSA state could lose its
incomplete flag. The current `_candidate_set_complete()` follows incoming
memory-valued `loop_carried` facts as well as view references, checks local
unknown/incomplete evidence before closing cycles, and does not change graph
facts. The archived RC3 query implementation remains historical; consumers
needing completeness flags must use the corrected checkout implementation
identified by the review's source hash. This is not a new graph revision.

The following interface is the downstream contract, conditional on independent
adjudication. It does not certify hardware arithmetic or prediction quality.

| Field or derived input | Required interpretation |
|---|---|
| `action_point_coverage.records` | Authoritative registration registry: all registered actions remain here, including unresolved targets. Active `action_point` nodes are only the matched subset. |
| `Lk` / absolute slot | Slot is `k-1`; preserve holes. Never renumber or silently truncate actions. |
| Missing compiler target | Structural evidence is missing; directive legality is still decided by source metadata and the separate TCAD policy. Do not delete a legal directive or fabricate a target. |
| Structural masks | Derive registration, target-presence and relation-evidence masks separately. An unresolved target contributes no claimed structural proof. |
| `evidence_status` | Keep `proven`, `possible`, `not_established` separate. Interpret relation polarity too: proven `no_alias` is negative evidence, not positive connectivity. Missing edges do not prove independence. |
| Call-scoped roots | Retain every supported alternative, exact invocation binding and `root_candidates_complete`. A complete possible set is not must-alias; incomplete known alternatives do not exhaust storage. Unbound formals remain symbolic. |
| Native source / SSA / call IDs and hashes | Alignment and provenance keys, not arbitrary learned categorical features or kernel-identity shortcuts. |
| Trip counts / numerical types | Keep value, knownness, compiler proof and source metadata distinct. A known zero trip count is different from unknown. Stub-lowered widths are not hardware-exact types. |
| Reverse message edges | Derive only in a separate tensor view with explicit direction metadata; never insert them into canonical facts. |
| FPGA / resources / clock / objective / QoR / selected directives | Separate request or design-conditioning inputs. They must not overwrite canonical compiler semantics. |

No tensor encoder, training, target conditioning, production routing or
refinement behavior is implemented by this review. Human adjudication remains
open; a generated worksheet never closes that gate.
