# V2.2 final release review

**V2.2 RC3 — technically frozen; independent adjudication pending.**

Reviewed base: `cc00a6e2acd7e87eba9076c12def1e7afb70dc91`, branch
`feature/v22-structural-semantic-finalization`. This is a review of immutable RC3
artifacts and one narrowly corrected consumer query, not a new graph release.
The original [manifest](graph-release-v22-rc3.json), [checksum](graph-release-v22-rc3.sha256)
and 63-graph archive are unchanged. No full-corpus generation, native-analysis
rebuild, HLS experiment, neural encoding, training or refinement was performed.

## What was independently executed in this review

| Check | Newly executed result | Boundary |
|---|---|---|
| Archive and all 538 bundled content hashes | PASS | SHA-256 and safe extraction; historical RC2/V2.1 release bytes also compared with Git |
| Exact cohort/source/metadata/headers/raw MLIR | 63/63 | Compared with the authenticated V2.1 release index; 55 development + 8 external snapshots, not a changed cohort |
| Portable archived-code validation | 63/63 | No native MLIR imported; reconstruction, witness consistency and public queries, not fresh compiler analysis |
| Current consumer validity and canonical/compact queries | 63/63 | Frozen canonical payloads unchanged by queries; all original registered identities retained |
| Archived versus corrected public action/loop/views queries | 63/63; 380 loop actions | No public-query differences on this cohort; native adversarial case below does differ |
| Live native operation/SSA/CFG/state occurrence walks | 9/9 diagnostic snapshots | GAN, covariance, AES, LUD, two Hotspot variants and three Kalman contexts; no compiler alias/dependence engine rerun |
| Full regression suite | 271 passed, zero failed/skipped | Newly executed; includes native fixtures and JPEG/Rosetta recovery tests |
| Additional defensive/worksheet checks | 5 passed | Includes two tests added after the full run; overlap with the full suite is not counted as extra independent observations |

The current suite has 273 distinct tests. The final focused V2.2 file run passed **59 tests**, including the two added
after the full run, and is recorded separately in the machine receipts; counts must not be added together
as if repeated tests were independent samples. The PyTorch NVML warning is an
environment warning, not a skipped test or a GPU validation claim.

`experiments/structural_audit/results/v22/final_review/` contains this run's
`verification.json`, `snapshot_verification.json`, `portable_receipt.json`,
`all_public_query_comparison.json`, native witness tables, regression logs and
recovery provenance. The verification tool had an initial native-context
cleanup failure; that attempt is preserved and is not counted as a successful
native check. Correct context teardown was applied only to the review tool;
all nine final native checks exited successfully.

## Evidence inherited from RC3, not re-executed as corpus generation

The archive records two separate-interpreter generations for each of the 63
snapshots, native occurrence receipts for all 63, 267 passing RC3 regressions,
and the earlier compatibility/native-analysis checks. Their hashes, identities
and recorded outcomes were checked here. **Two full native generations and the
267-test RC3 run are inherited receipts, not newly performed work.** Newly
executed compatibility tests include historical RC2 topology/query reproduction
and the existing V2.0/V2.1 regression suite.

The corpus still has 30,634 nodes, 115,783 edges, 606 registrations, 592 matched
compiler targets and 13 source-only labels. Its 2,995 concrete access occurrences
produce 2,973 proven, 32 possible and six not-established access relationships
(alternative roots can produce multiple relationships per occurrence). There
are 1,911 proven and 51,659 possible dependencies. Native negative conclusions
remain 150,439; they are not inferred from missing edges.

## Demonstrated query defect and correction

A native-verified handwritten MLIR fixture selects between a known allocation
and an unresolved external return, then forwards that partial set through an
`scf.for` iteration argument/backedge/result cycle. RC3's private
`_candidate_set_complete()` followed view references but not incoming memory
`loop_carried` facts. It therefore incorrectly reported the forwarded candidate
set as complete. The archived query shows `root_candidates_complete=true` at
the helper call; the corrected checkout reports `false`, preserving `possible`.

The minimal fix follows existing memory-valued loop-state inputs, rejects
explicit not-established root evidence, and checks local incompleteness before
closing a visited cycle. No canonical node, edge, relation type, root identity,
certainty or source/metadata contract was changed. Four cyclic/missing-evidence
unit cases and a native/public-query/compact-view regression cover this path.
The failing-before and passing-after logs are preserved.

**Consumer boundary:** the archived RC3 query code is historical and still has
this off-corpus completeness defect. Use the corrected checkout query helper
for completeness-sensitive consumers; do not relabel archived code as fixed.
The canonical artifact release can be technically frozen with that documented
consumer correction. No graph-changing release or corpus regeneration was
required. The worksheet includes both archived and corrected outputs on the
same native adversarial graph.

## Independent-adjudication package

[The editable worksheet](v22-independent-adjudication.json) contains **60 bounded
cases**, all pending, plus authenticated coverage/receipt references for every
one of the 63 snapshots. There has been **no human adjudication**. Each case
provides source expressions/line/hash, raw MLIR identity, native operation and
operand witnesses where available, expected semantics and their justification,
actual typed evidence, a machine-only diagnostic classification, and the
remaining evidence needed to close the case.

It covers GAN hierarchy/access/negative bindings and SSA; exact LUD invocation
and empty-context negatives; all ten rejected Hotspot mappings; all four
external Kalman unresolved registrations and six unknown pointer accesses;
covariance's triangular bound and dependence case; AES proven/possible/failure/
non-affine and must/may/no-alias cases plus CFG; strided/overlapping views;
conditional, incomplete loaded-pointer and loop-state fixtures. Handwritten
MLIR fixtures are explicitly identified, and their C++ annotation stubs are
not represented as compiler translations. Full embedded IR supports review.

The existing 1,002 machine-produced dependence samples remain archived draft
references. The bounded selection here uses fixed source cases and deterministic
lexicographic native-record selection, never QoR. It is not exhaustive proof
of memory dependencies. No precision/recall, significance or ground-truth
accuracy claims are made from these unreviewed examples.

For each case, fill `reviewer_disposition` (`pending`, `agree`, `disagree`,
`unresolved`), `reviewer_identity`, ISO `review_date`, `supporting_evidence` and
`reviewer_notes`. A changed disposition alone is not independent validation.
The tool preserves completed review fields and refuses to silently reuse a
completed review when its evidence changes. Source/IR paths resolve relative
to the extracted RC3 archive; fixture IR is embedded directly in the worksheet.

## Recovery-workload isolation

The archive's `V22_RECOVERY_INPUTS.json` and this run's
`recovery_provenance.json` authenticate 13 current recovery input files and the
separately bundled historical JPEG reference. Original JPEG and GEMV metadata
bytes remain recoverable from `4244ef7:path` (JPEG differential tests also use
`6863ce0`). Rosetta's source is unchanged; its added `ap_int.h` is explicitly an
analysis surrogate, absent before RC3, not a vendor replacement.

GEMV's prepared 32x32 source is selected only through an explicit source
override, retaining the original source hash. Its original metadata describes
12/16 loop values and differs from the recovered 32/32 metadata. These are
separate source/metadata identities, not silently repaired versions of the
original workload. Source/action validation rejects mismatched label bindings;
authenticated reuse checks full source, metadata and local-header hashes.

The existing synthesis materializer excludes local `ap_fixed.h`/`ap_int.h` and
requires authenticated genuine vendor headers. No synthesis behavior was
changed. Direct unmanaged compilation in a directory containing an analysis
header is outside this guarantee. Structural analysis compatibility, selected
host-GCC JPEG differential tests, and vendor-HLS equivalence are distinct:
**vendor equivalence is not established**. JPEG, Rosetta and GEMV remain outside
the exact frozen 63-snapshot cohort.

## Remaining limitations and release gates

- All 14 registrations remain compiler-unresolved: external Kalman L7/L8 in two
  variants; Hotspot L1–L4/L6 in pipeline/unroll variants. No unique surviving
  identity or transformation receipt establishes elimination. No target is guessed.
- Six Kalman proxy loaded-pointer accesses still need invocation-aware,
  field-sensitive memory-state evidence. Local reaching-store fixtures do not
  justify resolving these formal-cell contents.
- AES retains 43,236 distinct possible dependencies: 23,124 storage-alias
  outcomes, 20,032 affine failures and 80 non-affine outcomes. The 20,032 failures
  lack a common affine-scope block. Native first-successful-depth evidence is
  incomplete over other depths. Unknown does not mean no dependence.
- Static for-result storage alternatives may remain conservatively broader
  than exact last-iteration aliasing. View roots do not prove element overlap.
- Source template spellings and stub-lowered numerical types do not establish
  bit-accurate Vitis arithmetic, default rounding or hardware costs.

A (artifact reproducibility/readability), B (supported structural invariants),
D (complete coverage/uncertainty accounting) and E (regression isolation) pass
within the declared scope. Gate C remains **open**: an independent reviewer must
adjudicate the bounded references, resolve disagreements or approve explicit
limitations, and record identity/date/evidence. If a new graph-changing defect
is found, authenticate a corrected release and rerun all 63; do not amend RC3.

The [consumer contract](structural-graph-v22.md#final-verification-and-consumer-contract)
requires absolute `Lk` alignment, registered/unresolved masks, independent
source legality, typed certainty/polarity, call-context completeness, provenance
IDs rather than arbitrary learned labels, explicit unknown-value masks,
tensor-only reverse edges, and separate hardware/objective/QoR conditioning.
No tensor encoder is implemented. Recommend a **conditional semantic freeze**
for downstream interface engineering, not a FINAL label or a learning claim.

## Reproduction

From the repository root, use project Python with NetworkX/PyTorch and the
pinned native installation. A fresh owned test directory gives stable fixture
paths without touching prior outputs:

```bash
V22_REVIEW_PY=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
V22_REVIEW_TMP=$(mktemp -d /tmp/mailohls-v22-review.XXXXXX)
"$V22_REVIEW_PY" -B -m pytest -q -p no:cacheprovider tests --basetemp "$V22_REVIEW_TMP/tests"
"$V22_REVIEW_PY" -B -m experiments.structural_audit.review_v22 \
  --fixture-root "$V22_REVIEW_TMP/tests/v22_native_ir0" \
  --output-dir /fresh/v22-release-review \
  --worksheet /fresh/v22-independent-adjudication.json
```

These commands validate existing release graphs, not regenerate the corpus.
The bundled `V22_PORTABLE_VALIDATION.py` separately validates extracted archives
using their own frozen code without native MLIR. The review additionally
compares current and archived queries; a passing archive-only check cannot
establish that the archived completeness helper handles the new adversarial case.

## Fingerprints and case index

RC3 archive SHA-256: `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.

Corrected query helper SHA-256: `39aab15de84fecb46d108df194a6ebcad9e0db9963fba7bd155dbdcd036aba7c`.

Portable read-only check: 63/63, 116.85 seconds.

| Case | Kernel | Reviewer disposition |
|---|---|---|
| `afa1bdfea894:loop_array_access:L3:L11` | GAN | pending |
| `afa1bdfea894:loop_array_access:L2:L11` | GAN | pending |
| `afa1bdfea894:loop_array_access:L1:L4` | GAN | pending |
| `afa1bdfea894:loop_array_access:L2:L9` | GAN | pending |
| `afa1bdfea894:loop_array_access:L3:L9` | GAN | pending |
| `afa1bdfea894:nearest_child_loop:L5:L6` | GAN | pending |
| `afa1bdfea894:nearest_child_loop:L10:L11` | GAN | pending |
| `afa1bdfea894:nearest_child_loop:L5:L9` | GAN | pending |
| `e3ce197df78b:nearest_child_loop:L1:L2` | covariance | pending |
| `e3ce197df78b:nearest_child_loop:L3:L4` | covariance | pending |
| `e3ce197df78b:nearest_child_loop:L6:L7` | covariance | pending |
| `e3ce197df78b:nearest_child_loop:L5:L7` | covariance | pending |
| `e3ce197df78b:dependence_forward:L2:L4` | covariance | pending |
| `e3ce197df78b:dependence_forward:L1:L5` | covariance | pending |
| `41c8975cabb3:loop_array_access:L1:L4` | serrano-kalman-filter | pending |
| `41c8975cabb3:loop_array_access:L2:L13` | serrano-kalman-filter | pending |
| `41c8975cabb3:loop_array_access:L1:L13` | serrano-kalman-filter | pending |
| `41c8975cabb3:loop_array_access:L5:L11` | serrano-kalman-filter | pending |
| `41c8975cabb3:nearest_child_loop:L10:L11` | serrano-kalman-filter | pending |
| `e35f0c4cd5d4:loop_array_access:L5:L8` | rodinia_lud_1_tiling_0 | pending |
| `e35f0c4cd5d4:nearest_child_loop:L1:L2` | rodinia_lud_1_tiling_0 | pending |
| `e35f0c4cd5d4:loop_array_access:L5:L2` | rodinia_lud_1_tiling_0 | pending |
| `e35f0c4cd5d4:helper_empty_context_negative` | rodinia_lud_1_tiling_0 | pending |
| `e35f0c4cd5d4:loop_array_access:L19:L45` | rodinia_lud_1_tiling_0 | pending |
| `7c11a20afd22:registration:L1` | rodinia-hotspot-3-unroll | pending |
| `7c11a20afd22:registration:L2` | rodinia-hotspot-3-unroll | pending |
| `7c11a20afd22:registration:L3` | rodinia-hotspot-3-unroll | pending |
| `7c11a20afd22:registration:L4` | rodinia-hotspot-3-unroll | pending |
| `7c11a20afd22:registration:L6` | rodinia-hotspot-3-unroll | pending |
| `96b4d608969f:registration:L7` | serrano-kalman-filter | pending |
| `96b4d608969f:registration:L8` | serrano-kalman-filter | pending |
| `b90908f83f76:registration:L7` | serrano-kalman-filter | pending |
| `b90908f83f76:registration:L8` | serrano-kalman-filter | pending |
| `c448680ca9be:registration:L1` | rodinia-hotspot-2-pipeline | pending |
| `c448680ca9be:registration:L2` | rodinia-hotspot-2-pipeline | pending |
| `c448680ca9be:registration:L3` | rodinia-hotspot-2-pipeline | pending |
| `c448680ca9be:registration:L4` | rodinia-hotspot-2-pipeline | pending |
| `c448680ca9be:registration:L6` | rodinia-hotspot-2-pipeline | pending |
| `41c8975cabb3:loaded_pointer:op:_ZN7ap_uintILi256EE11range_proxyaSEm:op1` | serrano-kalman-filter | pending |
| `41c8975cabb3:loaded_pointer:op:_ZNK7ap_uintILi256EE11range_proxycvmEv:op1` | serrano-kalman-filter | pending |
| `96b4d608969f:loaded_pointer:op:_ZN7ap_uintILi256EE11range_proxyaSEm:op1` | serrano-kalman-filter | pending |
| `96b4d608969f:loaded_pointer:op:_ZNK7ap_uintILi256EE11range_proxycvmEv:op1` | serrano-kalman-filter | pending |
| `b90908f83f76:loaded_pointer:op:_ZN7ap_uintILi256EE11range_proxyaSEm:op1` | serrano-kalman-filter | pending |
| `b90908f83f76:loaded_pointer:op:_ZNK7ap_uintILi256EE11range_proxycvmEv:op1` | serrano-kalman-filter | pending |
| `883c223fddcf:dependence:proven_affine_dependence` | chstone-aes | pending |
| `883c223fddcf:dependence:aliased_storage_not_canonicalized` | chstone-aes | pending |
| `883c223fddcf:dependence:affine_analysis_failure` | chstone-aes | pending |
| `883c223fddcf:dependence:non_affine_access` | chstone-aes | pending |
| `883c223fddcf:alias:must_alias` | chstone-aes | pending |
| `883c223fddcf:alias:may_alias` | chstone-aes | pending |
| `883c223fddcf:alias:no_alias` | chstone-aes | pending |
| `883c223fddcf:cfg_successor` | chstone-aes | pending |
| `afa1bdfea894:ssa_def_use` | GAN | pending |
| `e3ce197df78b:dynamic_trip_count` | covariance | pending |
| `native_fixture:strided_views` | adversarial_native_fixture | pending |
| `native_fixture:overlapping_views` | adversarial_native_fixture | pending |
| `native_fixture:conditional_if` | adversarial_native_fixture | pending |
| `native_fixture:slot_incomplete` | adversarial_native_fixture | pending |
| `native_fixture:memory_state` | adversarial_native_fixture | pending |
| `native_fixture:incomplete_loop_forwarding` | adversarial_native_fixture | pending |
