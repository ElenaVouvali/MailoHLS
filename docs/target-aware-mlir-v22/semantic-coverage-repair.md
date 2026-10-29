# V2.2 semantic coverage repair audit

## Scope and provenance

This audit uses the authenticated 63-snapshot release
`salt-hls-graphs-v22-20261027-rc3` and the source/toolchain state at the start of
this repair branch (`research/v22-semantic-coverage-repair`, based on
`21ed70492e74e167354c1cb21d5a499e2f434014`). The archive SHA256 is
`feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`. Every
member hash in `docs/graph-release-v22-rc3.json` was checked, and the current
consumer-side graph authentication accepted all 63 canonical snapshots. No
archive, graph payload, manifest, target-overlay payload, measurement, or
checkpoint was changed.

The corpus counts below count a source snapshot once per released graph file.
Some kernels intentionally have multiple graph snapshots; the Stencil2D,
GEMM, AES, LUD, and Covariance examples below identify the exact snapshot so
the counts are not mistaken for unique source-program counts.

## Audit method

The audit walks each unique `(memory-access operation, operand index)` in every
authenticated canonical graph. It distinguishes compiler-recorded affine maps
from maps accepted by the tensorizer, verifies SSA operand positions using the
current graph query, and reports accesses visible from registered, matched
actions. Dependence counts come from the graph's recorded native-analysis
records. Common-loop counts are read from the relation evidence. “Queries
skipped after a first positive result” is a count derived from the native
depth loop's documented range and `dependence_depth`; it is not newly run
analysis and is not a claim that skipped depths have any particular outcome.

View ranks are measured separately from view vectors. A shape rank being known
does not make its offsets or strides known.

The machine-readable per-snapshot report, including graph hashes, action
access coverage, dependence reasons, and view ranks, is
[`semantic-coverage-repair.json`](semantic-coverage-repair.json).

## Corpus-wide results

| Measure | Authenticated count | Interpretation |
|---|---:|---|
| Unique memory-access operations | 2,995 | Unique within each snapshot by operation and memory operand. |
| Compiler-recorded affine maps | 2,741 | Map text and compiler status are present. |
| Verified SSA/map input bindings | 2,719 | Exact SSA values and operand positions match the compiler graph. |
| Current V4 supported additive maps | 2,719 | All verified input bindings are represented; at most two map results were observed. |
| Missing affine evidence | 254 | The compiler did not provide a usable map for these accesses. |
| Quasi-affine maps outside current parser | 22 | 21 maps contain `floordiv`, 17 contain `mod`; these counts overlap. No `ceildiv` was observed. |
| Maps over four inputs / over three results | 0 / 0 | Observed maxima are four inputs and two results. The old result cap remains unchanged. |
| Conflicting map witnesses | 0 | No authenticated operation had conflicting access-map evidence. |
| View mappings | 757 | Source/result shape rank is at most three. |
| View mappings with static offsets/sizes/strides vectors | 0 | V2.2 records these vectors as unknown for this corpus; no dimension vector is truncated. |
| View feature dimension overflow | 0 | No recorded vector exceeds the encoder's four-dimension representation. |

The former `MAX_AFFINE_INPUTS = 4` cap was not excluding any RC3 graph
access: no access has more than four verified inputs. However, each operand is
already represented by its own verified `operand` edge, so a fixed input cap
was unnecessary and could reject future authenticated maps without saving
tensor width. The cap is removed. The per-result coefficient channels remain
fixed at three because the release maximum is two; increasing them would add
no coverage in this corpus.

The 22 quasi-affine maps need an operator-aware expression representation.
They must not be coerced into the existing linear coefficient/offset fields.
This repair leaves them unknown to the neural affine-feature path. Examples
include three `AddRoundKey_InversMixColumn` accesses in AES loop action `L6`
with nested `floordiv` expressions, and one `SubByte` map containing `mod`.
The KNN coalescing snapshot has one such access visible in each of `L2`, `L3`,
and `L4`; LavaMD snapshots expose four such accesses to loop actions. These
are registered, compiler-matched actions, so the gap is real, but correct
encoding needs a separately designed AST/operator feature block. The existing
analysis parser does not establish a lossless fixed-dimensional encoding for
these expressions. The number of such expressions is not justification for
pretending they are linear.

## Representative action coverage

| Snapshot | Action(s) | Affine/access evidence |
|---|---|---|
| `machsuite-stencil2d-624ee88fef89.json` | `L1`, `L2` loop actions: 3 accesses each; `L3`, `L4`: 2 each | 3/3 maps have verified bindings; maximum is 4 inputs, 1 result. One native pair is proven independent; four pairs remain unknown because storage identity is not canonicalized. |
| `machsuite-gemm-ncubed-8692fbcbe7b2.json` | `L1`, `L2`: 3 accesses; `L3`: 2 | 3/3 maps are supported; maximum 2 inputs. One native pair is proven independent; four are unknown due to noncanonicalized storage aliases. |
| `machsuite-gemm-blocked-87bab6d97f84.json` | `L1`–`L4`: 4 accesses; `L5`: 3 | 4/4 maps are supported; maximum 4 inputs, 1 result. Three dependences are proven and four pairs remain alias-unknown. |
| `covariance-e3ce197df78b.json` | `L1`–`L7` (matched loop actions) | 18/18 accesses have verified maps, maximum 2 inputs/results. Dependence outcomes: 38 proven, 38 proven no-dependence, 148 unknown alias pairs. |
| `rodinia_lud_1_tiling_0-e35f0c4cd5d4.json` | Matched loop actions `L1`–`L45` | 50/54 accesses have verified affine maps; four lack compiler map evidence. Outcomes: 54 proven dependence, 39 proven no-dependence, 44 proven no-alias/not applicable, 48 alias-unknown. This audit does not add invocation-specific cloned facts. |
| `chstone-aes-883c223fddcf.json` | `L6` loop action: 8 accesses | Five maps are supported and three `floordiv` maps are explicitly unsupported. Dependence outcomes: 363 proven, 12,443 proven no-dependence, 1,250 not applicable, 43,156 unknown. |
| `serrano-kalman-filter-96b4d608969f.json` | `L1`–`L13` where matched; `L3`/`L4` accesses include unresolved compiler evidence | 21/27 accesses have verified maps; six lack affine evidence. Existing unresolved action registrations and pointer-derived accesses remain unresolved. The three Kalman release snapshots have the same aggregate findings. |

These action counts are graph-derived relevance counts, not dynamic access
counts or hardware demands. An action may cover a static operation nested in
its loop context; that does not imply one execution per iteration or
concurrent execution.

## Dependence-depth audit

The V2.2 RC3 native patch is `toolchain/patches/0009-mailohls-analysis-python.patch`.
The locked upstream sources are Polygeist
`77c04bb2a7a2406ca9480bcc9e729b07d2c8d077` with LLVM project gitlink
`26eb4285b56edd8c897642078d91f16ff0fd3472`; the local patched build reports
LLVM 18.0.0. The patch queries depths from 1 through `common + 1`. A failed
depth sets a failure flag and iteration continues. A positive result records
that exact depth and components, then stops. If there was no positive result,
one or more failures yield `unknown`; only all successful no-dependence
results produce `no_dependence`.

Across 63 snapshots, recorded pair outcomes total:

| Outcome | Count |
|---|---:|
| Proven dependence | 1,911 |
| Proven no-dependence | 55,173 |
| Not applicable (including proven no-alias and non-affine cases) | 101,439 |
| Unknown | 45,486 |

Among the unknown records, 25,454 are `aliased_storage_not_canonicalized`
and 20,032 are `affine_analysis_failure`. The release also records 6,173
`non_affine_access` and 95,266 `proven_no_alias` not-applicable records.
There are 1,409 dependence graph relationships with more than one common
affine-loop level. For 385 positive relationships among those, the native
loop exited at its first positive query; the recorded positive depth is valid,
but the later depth results are absent. Based on that query range, 863
subsequent depth queries were not run. Their outcomes are unknown, not
independent.

Examples of the observed multi-level relationships include Stencil2D (four
multi-common-depth relationships), GEMM-blocked (seven), Covariance (44), LUD
(55), AES (15), and Kalman (seven per released snapshot). This is evidence that
depth-indexed coverage could be useful. It is not safe to recover those
outcomes by editing the archived JSON or changing the meaning of its existing
single-depth fields.

### Why no supplementary native depth run is included

The correct next step would be an additive, versioned native-analysis build
that emits one record per query depth, with `dependence`, `no_dependence`, and
`failure` kept distinct and exact source-access witnesses. It must use the
locked MLIR/Polygeist APIs and preserve the old release unchanged. The current
RC3 artifact does not contain the omitted results, and no such supplementary
native build/evidence artifact was produced in this repair. Therefore the
tensorizer continues consuming only the existing RC3 evidence. It does not
infer the unqueried depths or reinterpret failures as independence.

The AES affine failures are particularly unsafe to bypass: 20,032 native
queries fail because the accesses do not share the required affine analysis
block/scope. The analyzer's preconditions are not relaxed here. The 23,124
storage-alias unknowns and 80 non-affine outcomes remain unknown as recorded.
For Kalman, six pointer-derived memory accesses lack stronger memory-root and
call-context evidence; no roots or dependencies are fabricated, and the
unresolved `L7`/`L8` registrations are untouched.

## Implementation and schema

`GNN_branch/semantic_tensor_features_v22.py` now accepts any number of
compiler-verified affine map inputs. Each input remains associated with its
exact SSA operand, ordinal, dimension/symbol role, and result coefficients on
the existing operand edge. No node, edge, or feature channel is added. The
three-result coefficient limit remains and marks genuine overflow unknown.
The additive linear parser remains unchanged; `mod`, `floordiv`, `ceildiv`,
unsupported expressions, incomplete bindings, and conflicting witnesses are
not encoded as linear expressions.

The semantic edge metadata is now
`salt-hls-semantic-edge-features-v2`; the tensor schema is
`salt-hls-target-tensors-v4-semantic-wide-affine-experimental`. The schema
change records the altered input-count acceptance contract. Tensor dimensions
are unchanged: the same 108 semantic edge channels and action feature width
remain, the same two `TransformerConv` layers are used, and action memory
remains 128D with absolute `Lk -> k-1` alignment. V3 remains the identifier
for prior experimental tensors and pilot results. No result is represented as
compatible with old V3 checkpoint weights.

Because the authenticated corpus has no map with more than four inputs, this
change adds no RC3 semantic values and no tensor channels. Tensor memory is
therefore unchanged for these release graphs. On the AES snapshot at ZCU104,
10 ns, a direct V3-source versus V4 tensorization comparison produced identical
25 tensor fields and `60,297,588` tensor bytes in both cases; measured CPU
tensorization times were 22.7475 s (V3) and 22.4770 s (V4) for this run. This
single CPU measurement is not a training-throughput claim, and no GPU memory
claim is made. The 63 graphs were not regenerated.

## Verification and reproduction

Run the authenticated audit after extracting the existing archive to a
temporary directory using the repository's release verification instructions:

```bash
PYTHONPATH=. python scripts/graphs/audit_semantic_coverage.py \
  --release-dir /path/to/extracted-v22-rc3 \
  --output /tmp/semantic-coverage-audit.json
```

Focused semantic/tensor regressions:

```bash
python -m pytest -q \
  tests/test_target_graph_semantic_tensor_v22.py \
  tests/test_action_demand_tensor_v22.py
```

The repair run used the repository's `/home/elvouvali/.hls-llm` environment:
12 tests passed. The authenticated audit accepted all 63 snapshots and all
member SHA256 values. The archive SHA256 remained
`feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
The separate MLIR V2.2 graph test file passed 59 tests. The combined graph and
overlay test invocation reported 96 passed and 21 setup errors: those overlay
tests require `pathlib.Path.is_relative_to`, which is unavailable in the local
environment's Python 3.8.10. The errors occurred before test bodies while
opening the authenticated archive; the independent corpus audit authenticated
all archive members with the same codebase's file hashing. This compatibility
issue is not changed here. The native supplementary-depth build, GNN training,
and synthesis were not run.

References: the [MLIR Affine dialect documentation](https://mlir.llvm.org/docs/Dialects/Affine/)
specifies affine dimensions/symbols and the restricted semantics of
`mod`/`floordiv`/`ceildiv`; the [Affine dependence API](https://mlir.llvm.org/doxygen/AffineAnalysis_8h_source.html)
documents interval-valued dependence components. These current online pages
are explanatory references; implementation behavior is governed by the
repository's pinned LLVM/Polygeist sources and the RC3 patch above.
