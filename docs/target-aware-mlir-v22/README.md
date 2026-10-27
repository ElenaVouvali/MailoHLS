# SALT-HLS MLIR V2.2: compiler graph and target-aware extension

This document explains two parts of the framework: the V2.2 RC3 graph, which
records what the compiler knows about a program, and a separate **target
overlay**, which adds FPGA capacities, the requested clock and resource budgets.
The overlay connects relevant program entities to possible hardware resources.
For example, a multiply may connect to DSP and LUT resources, but these links
do not say which implementation HLS will choose or how many resources it needs.

The sections below retain the exact schemas, evidence rules, measurements and
reproduction commands. A few terms are useful before reading them:

| Term | Meaning here |
|---|---|
| MLIR / dialect | The compiler's intermediate representation / a group of operations for a particular purpose, such as loops or memory |
| Operation / SSA value | A compiler instruction / one result or block argument used by instructions; SSA means each value has one definition |
| Region / block | A body owned by an operation / a sequence of operations with arguments and control-flow connections |
| Memory root / view | The underlying allocation, global or symbolic parameter / a reference into storage, possibly with offsets and strides |
| Alias / dependence | Two references may refer to overlapping storage / one access or computation may constrain another's execution |
| Action point | A source loop or array marked `Lk` where the framework supports an HLS directive |
| Call context | The particular helper invocation, including its argument bindings; two calls to the same helper are different contexts |
| Witness / provenance | The specific operation, operand or analysis record supporting a claim / the inputs, tool versions and hashes needed to trace it |
| Canonical / compact graph | The authoritative graph / a smaller view from which that graph can be restored exactly |
| Tensor encoder | Future code that converts graph attributes and relationships into numerical arrays for a GNN |

## Baseline documentation checkpoint (Phase A)

This section was written and saved **before implementing hardware enrichment**.
The documentation-only commit preserves this original description independently.
Inspected checkout: `d488681f68e58ab1b1e666f6911f5257d6462dc2` (`main`),
with pre-existing working-tree deletions preserved. Work branch:
`research/target-aware-mlir-v22`. The compiler schema is
`salt-hls-mlir-structural-v2.2`, semantic revision `rc3`, selected explicitly by
`--revision 2.2 --semantic-revision rc3`. V2.1 is historical.

**RC3 is technically frozen; independent human semantic review is pending.**
Automated checks establish consistency but do not replace that review. The
review worksheet has 60 pending cases. See [final review](../V22_FINAL_RELEASE_REVIEW.md) and
[V2.2 semantics](../structural-graph-v22.md). This work neither regenerates nor
amends that graph release.

### Authenticated baseline and implementation identities

Archive: `GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz`.
SHA-256: `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
The [release manifest](../graph-release-v22-rc3.json) authenticates 538 bundled
files and 63 snapshots (55 GNΩSIS development kernels and eight external
snapshots). Three Kalman source/metadata contexts are distinct snapshots.
JPEG, Rosetta and two GEMV contexts are outside this frozen cohort.

The pinned frontend is Polygeist `77c04bb2a7a2406ca9480bcc9e729b07d2c8d077`,
with llvm-project gitlink `26eb4285b56edd8c897642078d91f16ff0fd3472` and the
nine ordered repository patches in [toolchain/README](../../toolchain/README.md).
Recorded binary SHA-256:

| Implementation | SHA-256 |
|---|---|
| `cgeist` | `29271acd6b365b0d828a637e908e5aa74eee2c21d402e64a8139f550e03fbc89` |
| `_mailohls_analysis` | `4d4932f962968118da6f8faee07402b62b2a99b29e996c53ede9939baba71e98` |
| Archived V2 builder | `ae12ae568905d3375f145e31ccab4cab73dcf16a6cb10aef1c50f49fe6d1ea5b` |
| Archived semantic annotation | `6167ab0bcf84f8aab0e7a60d7e896f546c8430f98165f166031bf95b1eb81d8d` |
| Archived V2.2 extension | `522135be66558874905bd133b2d6396514c0437482788bfe2989d895a389c855` |
| Toolchain lock | `9b46a4dd124a4e92cb99fdc87414e8e026bf2fe43ebdadeb56e6923a9a5451b3` |

Use the **current checkout's** `GNN_branch/action_point_relations.py` to query
the graphs. Its corrected `_candidate_set_complete()` follows memory references
passed between loop iterations. It checks for missing root alternatives before
treating a revisited node as fully explored. The helper bundled in the archive
has a defect exposed by a test outside the released corpus: it can report an
incomplete root set as complete. The current correction changes query behavior,
not RC3 graph bytes. The verification report below records the code hashes and
validation results.

### Source to native analysis to graph

1. `load_kernel_info_actions()` joins source `Lk` markers with `kernel_info.txt`.
   These identifiers are source registrations, not inferred optimization results.
2. Pinned `cgeist` emits textual MLIR with source locations. A representative
   authenticated command uses `-function=<top> -S -O0 -scal-rep=0
   -print-debug-info -memref-fullrank -raise-scf-to-affine`. Exact arguments,
   source/header/metadata/raw-MLIR hashes are in each graph's provenance.
3. Official MLIR Python bindings parse and verify the module. The compiled
   `_mailohls_analysis.analyze_module()` supplies effects, view reachability,
   aliases, affine dependence outcomes and trip counts. Memory effects describe
   reads/writes; affine analysis reasons about supported loop bounds and index
   expressions. Production generation
   rejects missing native analysis; there is no approximate fallback.
4. `mlir_graph_v2.generate()` uses `_PreciseCoverageBuilder`, then
   `build_from_result()`, `mlir_semantics.annotate_semantics()` and
   `mlir_semantics_v22.annotate()`. Native operation/SSA ownership, successor
   occurrences and state forwarding are independently checked against the native
   MLIR objects before those objects are released. The pinned wrapper uses the
   `mlirOperationGetSuccessor` C API to read the actual control-flow graph (CFG)
   successors: the blocks to which an operation can transfer control.
5. The canonical `networkx.MultiDiGraph` is validated and serialized.
   `compact_view()` makes an optional, losslessly reconstructible view.
6. Public `get_action_point_relations()`, `get_loop_semantics()` and
   `action_point_views()` derive action relationships on demand from that graph.
   `generate_mlir_v2_corpus.py` manages the kernel inventory and artifact reuse;
   its default is still V2.1. It is not an alternative V2.2 analysis engine. The RC3 release
   uses authenticated frozen raw MLIR through the V2.2 finalization workflow.

The represented dialects include Affine, SCF, MemRef, Arith, Math, Func and
remaining CF/LLVM/Polygeist operations. Operations are not restricted to a
preselected arithmetic vocabulary: their exact names, attributes, operand/result
types, locations and ownership are retained. LLVM pointer operations matter
when C++ objects are lowered to pointer-based operations. Full operation text
and source declarations are retained for inspection. A declaration copied from
source text does not establish how the compiler implemented a C++ template or
whether it matches Vitis hardware arithmetic exactly.

### Compiler entities: the six node categories

| Category | Identifier / attributes | Interpretation |
|---|---|---|
| `operation` | `op:<function>:op<ordinal>`; function operations also use `:function`; `op_uid`, `text`, `mlir_attrs`, types, source location, `parent_operation`, `loop_context` | Actual compiler operation, including functions, loops, accesses and calls |
| `ssa_value` | `value:<ssa_id>`; operand/result/block ownership, `value_type`, `lowered_numerical_type` | Exact result or block argument; a view is a value, not a second allocation |
| `constant_or_feature` | `feature:<legacy_id>`; `feature_role`, `feature_kind`, token | Semantic literal or auxiliary shape/access/trip token; only some auxiliary leaves compact |
| `control_region` | `module`, `region:<owner>:<index>`, `block:<function>:<block>`; `control_kind` | Actual region/block nesting, not synthetic loop membership |
| `memory_object` | `memory:<ssa_id>`, `memory:global:<symbol>`, `memory:unresolved:<ssa_id>`; `root_kind`, root SSA values, shape/type | Allocation, exact global, symbolic formal parameter or explicitly unresolved storage identity |
| `action_point` | `action:Lk`; `action_id`, `absolute_index`, `action_kind`, `supported_directives`, source binding | Matched registered source directive site with one exact compiler target |

Shapes, scalar constants, affine maps (compiler expressions describing indices),
source dimensions, lowered widths and compiler-proven trip counts remain
separate attributes. A loop's metadata value may describe its upper bound
rather than its actual iteration count. Both values are kept when they disagree.

### Complete compiler edge vocabulary

Every canonical edge carries `relation_type`, `evidence_status`, `access_path`
and `is_reverse=false`. The table describes **all 22 implemented categories**
in `mlir_graph_v2.EDGE_KINDS`. Abbreviations: O operation, V SSA value,
F constant/feature, C control region, M memory object, A action point.
Examples are schematic relation illustrations, not additional benchmark facts.

| Relation | Endpoints | Evidence / meaning | Example; possible HLS relevance |
|---|---|---|---|
| `owns_region` | O → C(region) | Proven actual region owner | `affine.for → body region`; scope of pipelining |
| `contains_block` | C(region) → C(block) | Proven native block membership | `body → ^bb0`; structured/CFG boundaries |
| `contains_operation` | C(block/module) → O | Proven native membership; module owns function ops | `^bb0 → arith.addf`; operation placement |
| `declares_argument` | C(block) → V | Proven native block-argument owner, including restored unused arguments | `^bb0 → induction variable`; loop state |
| `next_operation` | O → O | Proven lexical block order, not a scheduling decision | `load → add`; execution context |
| `control_flow` | O → O | Native successor or retained compiler control role; RC3 successor targets first op of successor block | `cf.br → successor first op`; feasible control paths |
| `defines` | O → V | Exact native result occurrence/index | `arith.mulf → %p`; compute/dataflow |
| `operand` | V → O | Exact native operand occurrence/index | `%p → arith.addf`; producer/consumer chains |
| `literal` | F → O | Semantic immediate associated with compiler constant | `2 → arith.constant`; fixed bounds/constants |
| `feature` | F → O/V (associated compiler entity) | Retained auxiliary compiler feature occurrence | `shape token → memref value`; compactable information |
| `loop_carried` | V → V | Exact init/iter/yield/result/condition forwarding role and loop; zero-trip path retained separately | `%init → iter_arg`; recurrence constraints |
| `loop_child` | O(loop) → O(loop) | Proven immediate compiler loop nesting | `outer → inner`; unroll scope |
| `calls` | O(call) → O(function) | Proven matched body/arity or explicitly possible resolution | `call helper → helper`; interprocedural context |
| `actual_to_formal` | V(actual) → V(formal) | Exact call UID, operand/argument index, type/body checks | `A → helper.%p @ call1`; no union of unrelated callers |
| `returns_to` | V(return operand) → V(call result) | Initially possible, exact call/result index; separate invocation-scoped root reasoning may establish roots | `helper.%p → call1.%r`; returned storage |
| `references` | V(view) → V(base) | Direct compiler-supported storage/view forwarding; pointer path is not normalized address equality | `subview → A`; storage identity versus indexing |
| `alias` | V ↔ V (one canonically ordered edge) | Native must/may/partial/no-alias; `symmetric=true`; must/no are proven, may/partial possible | `%a no_alias %b`; negative evidence, never positive connectivity |
| `resolves_to` | V(memory) → M | Proven unique root, possible alternatives or not-established placeholder | `%view → allocation`; storage interpretation |
| `memory_access` | O(concrete access) → M | Exact native effect operand; root certainty preserved; read/write/readwrite | `affine.load → A`; memory demand, not port count |
| `depends_on` | O(consumer/access B) → O(producer/access A) | Native RAW/WAR/WAW record, carrying loop, bounds and certainty | `load B → preceding store A`; dependence constrains transformations |
| `targets` | A → O(loop)/M | Proven exact source/compiler binding; one target per active action | `L5 → loop`; action alignment |
| `analysis_summary` | F → V, V → V, or O → M (retained summary endpoints) | Explicit possible view reachability/call/legacy summary; not concrete access proof | `call → formal root`; conservative effects |

`proven` means compiler evidence supports the stated relationship within the
documented analysis limits. `possible` keeps alternatives that the analysis
cannot rule out. `not_established` means proof is missing; it does not mean false.
Native negative conclusions (`no_alias`, `no_dependence`) remain identifiable
negative evidence. Missing edges never prove independence. A dependence-distance
interval gives bounds, not an exact iteration distance. The native analyzer
stops at the first loop depth where it establishes a dependence; it does not
prove that no other enclosing loop carries one. Different relation types and
distinct native outcomes between the same operations are all kept.

### Memory, calls, loops and uncertainty

Root resolution starts from allocations, globals and formal parameters within
each function. It follows direct views, native must-alias evidence and supported
SSA forwarding. Must-alias is a definite alias conclusion; may-alias alone
cannot establish a unique root. Subview offsets are relative to the source
view; reinterpret casts use underlying-storage offsets. Offsets, sizes, strides,
SSA indices, affine maps and their induction dependencies remain available.
For LLVM accesses the pointer-expression chain and induction dependencies are
retained. An LLVM access can calculate its address through pointer operations,
so an empty memref-index list does not mean it uses the same address each time.

`llvm.getelementptr`, pointer casts and Polygeist pointer/memref conversions
can preserve the underlying storage root. `memref2pointer` uses the memref's
aligned pointer plus its descriptor offset, or a pointer cast required by the
calling convention. Sharing a root does not establish equal addresses or
overlapping elements. When a pointer is itself loaded from memory, resolution
requires evidence about which stores could have supplied it. If the pointer
escapes to unanalyzed code, control flow is unsupported, or aliasing is uncertain,
the root set remains incomplete. Finding one or more roots does not prove that
all alternatives have been found.

Calls retain exact call UIDs, actual/formal positions and return positions.
Public queries substitute caller roots only when examining that verified
invocation. They do not merge arrays from unrelated callers or generate every
combination of argument alternatives. Recursive/external calls remain uncertain
where evidence is insufficient. A symbolic formal parameter is not a proven
local on-chip buffer. `root_candidates_complete` says whether all alternatives
are accounted for in the particular invocation being examined.

Loop hierarchy includes unlabeled loops. Public queries can find the nearest
labeled enclosing or nested loop without deleting the intervening loops.
Native trip counts stay separate from source metadata. Init, yield, backedge
and result forwarding are explicit;
memory-valued for-results retain initial-buffer zero-trip alternatives, and
while-results follow values passed by the loop condition. This distinguishes
SSA state passed between iterations from dependencies through memory. Absence
of a dependence proof does not establish independence.

### Action registration and absolute alignment

`Lk` comes from source/metadata and always means absolute slot **`k-1`**.
Matched loop actions bind through exact compiler source evidence. Array actions
enumerate native `mailohls.action_id` or exact declaration-location candidates,
require a unique storage origin, and use shape only as a consistency filter.
Two objects sharing source-use lines, names or shapes are not thereby the same
allocation.
Unresolved registrations get coverage records and diagnostics, not guessed
`targets` edges or fabricated active nodes.

The release contains **606 metadata registrations, 592 matched action points,
14 unresolved registrations and 13 source-only labels**. There are 380 matched
loop actions. Source-only means labeled but unregistered; unresolved means
registered without a supported surviving compiler target. Neither implies
compiler elimination. Source legality is independent of graph matching.

The 14 unresolved cases are external Kalman L7/L8 in two metadata variants (4)
and Hotspot pipeline/unroll L1–L4/L6 (10). The development metadata **frozen in
RC3** did not register L7/L8. The later
[Kalman registration update](../structural-graph-v22.md#updated-development-kalman-registration)
registers both to match the repaired CSV, while retaining unresolved compiler
targets. The corrected source/metadata match the already-released snapshot
`serrano-kalman-filter-96b4d608969f.json`; use that file and the default RC3
manifest with the overlay CLI. No duplicate graph release is needed, and the
historical corpus measurements above remain unchanged. Consumers must carry
separate registered/matched/unresolved
masks: Boolean flags stating what is known for each slot. They must never remove
gaps or renumber active slots. The existing production memory uses
`[max_slots,128]`, absolute labels, active masks and relation masks;
multiscale export also seals checkpoint, graph, split and normalization hashes.
These conventions guide a future interface; matching the array dimensions alone
does not make V2.2 compatible with the old checkpoint.

### Canonical and compact views

`compact_view()` folds only single-neighbor auxiliary `memory_shape`,
`affine_access`, and `loop_trip_count` feature leaves whose incident relations
are all `feature`. Folded tokens are attached to the neighbor, while complete
original node attributes and keyed edges are saved in `compaction_mapping`.
`expand_compact()` removes the fold annotations and restores the exact canonical
payload. In other words, it removes a feature-only leaf from the visible topology
while keeping enough information to restore it exactly. Operations, SSA values,
semantic constants, regions/blocks, memory objects, action points and their
semantic relations survive intact.

The existing guarantees are exact canonical reconstruction and equal public
action views, all-evidence action queries and loop semantics. Smaller topology
does not by itself demonstrate improved learned QoR prediction. A consumer must
not attach irreplaceable semantics to compacted auxiliary node IDs.

### Serialization and validation

`graph_payload()` sorts node IDs and keyed edges and copies metadata;
`canonical_json()` sorts keys, uses compact separators and rejects NaN.
`add_relation()` gives each fact an edge key using the first 24 hex digits of
SHA-256 over its endpoints and complete attributes. Identical facts are stored
once; different facts between the same nodes remain separate edges.
`write_graph()` refuses to overwrite different bytes.
Graph identity hashes schema/provenance/builder identity, while release file
hashes authenticate the complete payload. Neither is a learned feature.

`validate()` checks node/edge vocabulary, exact action targets, memory-access
endpoints, call bindings, semantic annotations, RC3 evidence consistency and
connected semantic levels. Native correspondence checks actual compiler
occurrences; portable validation verifies their preserved witnesses without
loading native MLIR. Portable validation is not fresh compiler analysis.

### Existing architecture diagrams

```mermaid
flowchart TD
  SRC["C/C++ + Lk registrations + sealed local headers"] --> FE["Pinned cgeist: Affine/SCF/MemRef/Arith/Func + surviving LLVM/CF"]
  FE --> IR["Verified native MLIR module"]
  IR --> NA["Native effects, aliases, dependences, views, trip counts"]
  IR --> O["operation"]
  O -->|owns_region| R["control_region: region"]
  R -->|contains_block| B["control_region: block"]
  B -->|contains_operation| O
  B -->|declares_argument| V["ssa_value"]
  O -->|defines| V
  V -->|operand| O
  V -->|loop_carried / references / alias| V
  V -->|resolves_to| M["memory_object"]
  O -->|memory_access / analysis_summary| M
  O -->|depends_on / calls / loop_child / control_flow| O
  V -->|actual_to_formal / returns_to| V
  F["constant_or_feature"] -->|literal / feature| O
  A["action_point: matched Lk"] -->|targets| O
  A -->|targets| M
  NA --> G["Canonical V2.2: all entities + evidence + provenance"]
  O --> G
  V --> G
  M --> G
  A --> G
  G --> C["Compact: auxiliary feature leaves folded with reconstruction map"]
  G --> Q["Current public action/loop queries"]
  C --> Q
```

Simpler compiler-design view:

```mermaid
flowchart LR
  S["Source program + marked loops/arrays"] --> I["Compiler IR: operations, values and control"]
  I --> G["Graph: who computes, accesses, calls and depends on whom"]
  G --> C["Optional lossless compact view"]
  G --> A["Lk → exact loop or array; uncertainty retained"]
  C --> A
```

### Authenticated examples (read-only neighborhoods)

* **Simple loops / stencil:** `spcl_example_03-fdea84258953.json` has 95 nodes,
  162 edges. `action:L5 --targets--> op:Stencil2D:op11`, an `affine.for` with
  compiler count 998 despite source metadata value 999. Its nested
  `op:Stencil2D:op14 --memory_access(read)--> memory:Stencil2D:op1:r0`
  joins L5 to array action L2. The exact access edge key is
  `e7e9a1c56d44f21b01b41ede`; the native affine index uses the inner induction
  variable. This establishes an access, not a memory-port bottleneck.
* **Memory/dependence-heavy AES:** `chstone-aes-883c223fddcf.json` has 4,204
  nodes and 53,788 edges. `action:L1 --targets--> op:encrypt:op95`.
  `op:encrypt:op100 --depends_on(RAW)--> op:encrypt:op102` has supporting key
  `620fee8fe6e022ee428831b0`, carrying loop `op:encrypt:op95` and distance
  bounds 1..15, not an exact distance. AES has 43,236 possible dependence
  records: 23,124 alias-related, 20,032 affine-analysis failures at missing
  common affine-scope blocks, and 80 non-affine outcomes. Uncertainty survives.
* **Interprocedural LUD:** `rodinia_lud_1_tiling_0-e35f0c4cd5d4.json` has
  1,479 nodes and 2,790 edges. L2 targets `op:diagonal_load:op2`; a public
  memory-access relationship to L5 is proven only in invocation context
  `op:lud_diagonal:op5`, with exact actual/formal witnesses. The same join is
  absent when explicitly querying the empty helper context. Sharing a helper
  does not join different callers' arrays.
* **Unresolved Kalman:** `serrano-kalman-filter-96b4d608969f.json` retains
  L7/L8 as unresolved coverage records. Surviving scalar arithmetic and shared
  source-use lines cannot prove separate arrays. No `action:L7` or `action:L8`
  target is invented. Six loaded-pointer access relationships across the
  Kalman cohort remain `not_established` even though the accesses themselves
  exist. Hotspot's ten unresolved registrations follow the same policy.

### Baseline limitations and acceptance boundary

RC3 has 30,634 canonical nodes, 115,783 edges and 2,995 concrete access
occurrences. Those accesses produce 2,973 proven, 32 possible and six
not-established relationships (alternative roots explain non-unit counts).
There are 1,911 proven and 51,659 possible dependencies; 150,439 native negative
conclusions are retained. No graph count is a QoR label.

Independent semantic adjudication remains open. Fourteen registrations and
some pointer roots remain unresolved. Static root sets for loop results can
include alternatives that a particular execution will not take. Sharing a root
does not prove element overlap, and dependence analysis is not complete.
Simplified analysis headers for arbitrary-precision (AP) types, and declarations
read from source text, do not establish hardware-exact widths, rounding,
overflow or cost. System headers included indirectly are not all covered by
recorded hashes. FPGA characteristics are absent from canonical compiler facts.
No V2.2 tensor encoder is trained or implemented; production
GNN weights consume V1 feature schema `mailohls-mlir-features-v13-conditioned-targets`
(403 node features, 82 edge features in the maintained training launcher).

Phase-A acceptance requires authenticated archive/member hashes, existing
validators, exact reconstruction and equal current public queries on all 63
canonical/compact pairs. Newly executed receipts and any environment failures
are added here before enrichment begins. Historical final-review test counts
are inherited evidence, not tests newly executed by this task.

### Newly executed Phase-A acceptance

All 538 bundled content hashes and the archive SHA-256 passed. Archived-code
portable validation passed **63/63**, with no native MLIR loaded (117.00 s).
Current-code validation, exact canonical reconstruction, action views,
all-evidence action queries and loop-query equivalence passed **63/63**
(153.84 s). Queries left canonical payloads unchanged. Existing focused graph
tests passed **117/117**, zero failures/skips (104.25 s), including native fixture
subprocesses using the existing pinned compiler. This is not full native corpus
regeneration. All protected source/release/checkpoint hashes and the pre-existing
tracked change status remained unchanged. No target overlay code existed yet.

The compact corpus has 23,268 nodes and 108,417 edges; exact reconstruction
retains every folded node and edge. See [machine receipt](baseline-verification.json)
for per-snapshot counts, coverage, protected hashes and exact commands.

## Motivation and research boundary

GNΩSIS covers two devices (ZCU104 and U200) and three measured periods
(3.33/5/10 ns). This limits the evidence available to learn device/clock
generalization. Its [maintainer dataset card](https://huggingface.co/datasets/aferikoglou/GNWSIS)
and [upstream repository](https://github.com/aferikoglou/GNWSIS) describe the
larger dataset; that publication-level inventory is distinct from this
repository's processed records and 63-snapshot graph release.

The hypothesis is that hardware capacities, explicit constraints and links from
relevant program entities to resources will help a future GNN learn useful
patterns. This includes testing a new device such as U50 or a clock period such
as 6.67 ns. **No generalization or QoR improvement
has been established by this task.** Device properties are inputs; directive
assignments, measured latency/utilization, Pareto ranks and adaptation labels
are never read by the overlay. Resource ratios are conditioning values, not
synthetic QoR labels.

## Target architecture: a separate hardware overlay (Approach C)

`G_joint(x,d,c,B) = G_V2.2(x) ∪ H_d ∪ R_PH(x,d,c,B)`.
Here `x` is the program, `d` the device, `c` the requested clock period and `B`
the resource budget. `G_V2.2` is the unchanged program graph, `H_d` contains
hardware nodes, and `R_PH` contains the links between program and hardware.
The separate overlay composes an existing graph with five hardware nodes and
justified relationships. Compile once; reuse the same authenticated program
graph for different targets, clocks and budgets. No hardware attribute is
injected into an original compiler node or edge.

```mermaid
flowchart LR
  C["C/C++ + action registrations"] --> ML["Pinned MLIR V2.2 RC3"]
  ML --> P["Canonical / compact program graph"]
  H["Verified FPGA profile + numeric clock + resource budget"] --> E["Implemented target overlay + validator"]
  P --> E
  E --> J["Serializable target-aware MultiDiGraph"]
  J -.-> G["FUTURE: V2.2 heterogeneous tensor encoder + trained GNN"]
  G -.-> M["FUTURE: target-aware, action-aligned memory [max_slots,128]"]
  M -.-> X["Existing Stage-2 cross-attention; FUTURE integration"]
  X -.-> L["Existing frozen Stage-1 latency LLM"]
  L -.-> D["Future predicted directives"]
```

Dashed paths are **not implemented in this task**. Existing Stage-2 code shows
how structural memory can reach the LLM, but its trained components are not
compatible with the new graphs without further work.

### What the Stage-1 checkpoint path means

`mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1` is an
**existing LLM adapter directory**, not an MLIR graph, a V2.2 GNN checkpoint or
an RC3 release requirement. [FINAL_ARTIFACTS.json](../../FINAL_ARTIFACTS.json)
selects it for the `PARETO_LATENCY` objective. Its saved
[training contract](../../mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1/training_contract.json)
records seed 123, the DeepSeek-Coder 6.7B base model and
`--disable_structural_memory`. The directory holds the learned LoRA adapter
(small updates loaded alongside the base LLM), tokenizer files and training
records. `best_custom_stage1` is the selected checkpoint directory; the repeated
`final_final` is a historical run name, not a graph-version identifier.

Keep this directory unchanged and backed up if you want to reproduce the
existing latency model or use it as a fixed baseline. **No action on this
checkpoint is needed to generate, inspect or validate RC3 graphs or target
overlays.** For later experiments, first build and evaluate a separate V2.2
encoder; only then test how its action vectors enter Stage-2 while holding this
Stage-1 baseline fixed. This is a proposed experimental control, not a claim
that the adapter already understands V2.2.

The adapter weights and tokenizer files exist in the inspected local checkout,
but the weights are Git-ignored and are not tracked in this repository. A fresh
GitHub clone therefore does not provide the complete model. Before future LLM
inference, obtain the full adapter/tokenizer directory and its base model, and
check the files against the recorded
[checkpoint hashes](baseline-verification.json). Preserve the saved tokenizer
and training contract: the contract records the historical v2 directive-domain
policy, which must not be silently reinterpreted as a newer policy. None of
these model files is loaded by the graph commands below.

## Hardware specifications and independent verification

The authoritative new registry is
[`GNN_branch/targets/fpga_profiles.json`](../../GNN_branch/targets/fpga_profiles.json),
schema `salt-hls-fpga-profiles-v1`, registry/profile version `1.0.0`.
`validate_registry()` checks positive integer capacities, units, meaning,
missingness, source references, vendor-excerpt hashes and exact excerpt/profile
agreement. The four resource keys match the existing LLM and V1 GNN conventions.
These are **physical full-part capacities**: all resources on the FPGA chip.
A Vitis platform shell may reserve some of them for infrastructure, and a user
may impose a smaller budget. Neither changes the chip's physical capacity.

| Property / unit | ZCU104 | Alveo U200 | Alveo U50 | Meaning and evidence |
|---|---|---|---|---|
| Part identifier | `xczu7ev-ffvc1156-2-e` | `xcu200-fsgd2104-2-e` | `xcu50-fsvh2104-2-e` | Exact experiment parts; independently read vendor `UserAssignments/Part`; board identities corroborated by UG1267/UG1289/UG1371 |
| Family | Zynq UltraScale+ MPSoC | UltraScale+ | UltraScale+ | Documented family; no inferred subtype embedding |
| BRAM / 18-Kibit RAM equivalents | 624 | 4,320 | 2,688 | Exact vendor `AvailableResources/BRAM_18K`; **not** 36-Kbit physical block count |
| DSP / slices | 1,728 | 6,840 | 5,952 | Exact vendor device enumeration; DS891/DS962/DS965 corroboration |
| LUT / LUTs | 230,400 | 1,182,240 | 871,680 | Exact vendor enumeration; public U200/U50 tables round counts |
| FF / flip-flops | 460,800 | 2,364,480 | 1,743,360 | Exact vendor enumeration; public U200/U50 tables round counts |
| Platform-available resources / same units | unknown | unknown | unknown | No platforminfo/XPFM shell allocation supplied; explicit `null` |
| User-requested budget / same units | input or unknown | input or unknown | input or unknown | Separate deployment data; zero is a known zero, not missing |
| Clock period / ns | arbitrary finite positive input | same | same | Deployment constraint, not an achieved clock or device capacity |
| Distributed-memory LUT allocation / LUTs | unknown | unknown | unknown | SLICEM subset availability unestablished; never equal to total LUT capacity |
| External-memory architecture | PS/PL DDR4 | DDR4 | HBM2 | Board-level fact only; kernel binding, available bandwidth and latency unknown |

**Independent capacity check actually performed:** only the `Part` and
`AreaEstimates/AvailableResources` sections of existing vendor report XML were
read on Coroni. No new tool run or synthesized `Resources`/performance labels
were used. Exact report paths, the four-field excerpts and normalized excerpt
SHA-256s are committed in the registry's `vendor_zcu104`, `vendor_u200` and
`vendor_u50` source records. This verifies exact capacities beyond copying the
GNΩSIS tuples. Vendor tool-version fields were not read and are not asserted.
The existing U50 adaptation config agrees; its training/measurement rows were
not used. The standalone `u50_device_specs.json` mentioned in that experiment's
README is absent in this checkout, so it was not claimed as inspected evidence.

Primary specification references (accessed for this task):

* [DS891 v1.11.1, 2025-03-18](https://docs.amd.com/api/khub/documents/sbPbXcMUiRSJ2O5STvuGNQ/content),
  EV resource table, p.8: ZU7EV has 312 block RAM blocks and the exact LUT/FF/DSP
  counts above. [UG1267 v1.1, 2018-10-09](https://docs.amd.com/v/u/en-US/ug1267-zcu104-eval-bd)
  establishes the ZCU104 board/device association.
* [DS962 v1.7, 2023-06-23](https://docs.amd.com/r/en-US/ds962-u200-u250/Alveo-Product-Details)
  lists U200 resources; [UG1289 v1.1, 2023-06-13](https://docs.amd.com/r/en-US/ug1289-u200-u250-reconfig-accel/Card-Features)
  identifies its XCU200-2FSGD2104E device.
* [DS965 v1.8, 2023-06-23](https://docs.amd.com/r/en-US/ds965-u50/Product-Details)
  lists 1,344 **36-Kbit** blocks, rounded LUT/FF counts and 5,952 DSPs.
  [UG1371 v1.1, 2023-06-15](https://docs.amd.com/r/en-US/ug1371-u50-reconfig-accel/Card-Features)
  establishes UltraScale+ XCU50 and HBM architecture.

Later document revisions above describe the same named physical devices. They
are used for architecture/capacity facts, not to change the pinned compiler or
Vitis 2021.1 operator semantics. Initial guessed AMD PDF endpoints were
inaccessible; the linked working official pages/API documents and vendor
excerpts are the evidence actually used.

### Primitive capabilities: supported, unknown and intentionally unmodeled

[UG573 v1.14, 2025-11-18, Block RAM Summary](https://docs.amd.com/r/en-US/ug573-ultrascale-memory-resources/Block-RAM-Summary)
applies to UltraScale architecture-based devices, including these UltraScale+
parts. One 36-Kbit block splits into two independent 18-Kbit halves. The overlay
records 18,432 bits per RAMB18 equivalent, 36,864 bits per physical block,
two halves, and maximum per-half port widths of 18 bits in true dual-port mode
and 36 bits in simple dual-port mode. True dual-port allows reads/writes through
either port; simple dual-port separates the read and write ports. These describe
primitive configurations,
not the ports or capacity achieved by a benchmark array. Primitive bit capacity
includes parity storage; usable array payload depends on the chosen configuration.
Hence 624/4,320/2,688
BRAM18K equivalents correspond to 312/2,160/1,344 physical 36-Kbit blocks.

[UG579 v1.11, 2021-08-30, DSP48E2 functionality, p.16](https://docs.amd.com/api/khub/documents/pTysoma4TYgNH95BrY1Sbw/content)
documents integer multiply/accumulate/add capability, with 27×18-bit multiplier
inputs and a 48-bit arithmetic logic unit (ALU). These describe a physical DSP
block, not the cost of every compiler multiply. They are **not** a model for
mapping a lowered MLIR operation to one DSP.

[UG574 v1.6, 2025-01-22, Distributed RAM Applications](https://docs.amd.com/r/en-US/ug574-ultrascale-clb/Distributed-RAM-Applications)
supports LUT-based storage relevance. Only a subset of LUTs provides distributed
memory; its deployment availability is explicitly unknown. The overlay does
not assume a size threshold, exact packing, memory latency, bandwidth, or a
resource demand formula. UltraRAM support is documented for the devices but
an UltraRAM graph/model is intentionally outside version 1, as are external
interfaces, their kernel bindings, and physical interconnect topology.

### Deployment semantics

The API accepts `clock_period_ns`, `budgets` and optional `platform_available`
maps using `BRAM_18K/DSP/LUT/FF`. The CLI exposes `--clock-period-ns`,
`--available-bram18k`, `--available-dsp`, `--available-lut`, `--available-ff`
and optional `--platform-resources-json`. A platform allocation is the resource
pool available after platform infrastructure has reserved its share; a user
budget can restrict that pool further.

For each resource, the overlay uses the user budget when supplied. Otherwise,
it uses the supplied platform allocation. If neither is known, it uses the
full physical capacity as an explicit convention, not as a measured amount of
free space. Requests cannot exceed a supplied platform allocation or physical
capacity. Counts must be nonnegative integers; fractional/boolean/nonfinite
counts are rejected. Periods must be finite positive numbers. A zero available
DSP budget is valid; an absent budget is `null` with `known=false` in the
request field. The physical capacity never changes. Providing allocation counts
does not imply that every other property of the platform is known.

`deployment` stores raw clock, units, requested/platform/effective counts,
availability fractions and their basis. The FPGA node holds the raw requested
clock. The fraction is effective availability divided by the physical capacity;
it is not a measured utilization. Resource compatibility can remain present at
zero available budget because it describes physical capability/relevance,
not feasibility of a selected implementation.

[UG1399 **2021.1**, 2021-06-16, Specifying the Clock Frequency](https://docs.amd.com/r/2021.1-English/ug1399-vitis-hls/Specifying-the-Clock-Frequency)
describes the requested clock as a synthesis constraint. Final placement/routing
delays are unavailable at that point; emitting a design does not guarantee timing
closure. This overlay makes an even narrower statement: a user constraint is
attached, with **no timing calculation or assumed clock uncertainty**.
[UG1393 2021.1 platforminfo](https://docs.amd.com/r/2021.1-English/ug1393-vitis-application-acceleration/platforminfo-Utility)
can report platform clocks, allocations and memory; it is a future source of
deployment allocation metadata, not proof that the full part is free in a shell.

## New graph schema and preservation contract

Composition/export schema: **`salt-hls-target-enriched-v22-v1`**. The original
schema remains **`salt-hls-mlir-structural-v2.2`** inside `program_metadata`.
The original V2.2 validator accepts compiler graphs only, so it intentionally
rejects a graph with the added hardware schema. Use the dedicated
`target_graph_v22.validate()` for that graph. It removes the overlay, validates
the restored program, rebuilds the overlay from the current code, FPGA profile
and deployment inputs, and compares the complete result. This detects changed
capacities, inputs, certainty flags, evidence, code identities or compiler facts.
It checks the overlay's construction rules; it does not rerun an independent
compiler analysis or perform human semantic review.

Original node IDs, edge keys and attributes are copied without changes.
Original graph metadata is kept under `program_metadata`; the outer metadata
describes the combined graph. `strip_overlay()` restores the exact original
`graph_payload()`, including canonical/compact metadata and evidence. The
program's nodes and edges are not duplicated in a hidden metadata field.

### Hardware nodes and identifiers

| Category | Exact identifiers | Attributes |
|---|---|---|
| `hardware_target` | `hardware:fpga` | Physical part/family/profile version; identity sources; raw clock and unit/constraint meaning; board memory architecture with unknown kernel binding; platform-known mask |
| `hardware_resource` | `hardware:resource:DSP`, `hardware:resource:BRAM_18K`, `hardware:resource:LUT`, `hardware:resource:FF` | Resource class; typed physical capacity; effective available capacity; available fraction; separate platform/request properties with `known` masks; documented capability properties |

Every numerical property has a unit and meaning. Known device properties cite
their sources; deployment numbers record whether they came from the user, a
platform allocation or the full-capacity convention. Optional unknown values
stay `null` with explicit known/unknown flags. No missing value becomes zero.
Compiler graphs containing reserved `hardware:` IDs or overlay markers are
rejected. There is one node per resource **class**, not per physical primitive.

### New edge relations

Every added edge has `_target_overlay=true` and a deterministic
`target:<24-hex-hash>` key. These relation names are separate from V2.2 facts:

| Relation | Endpoints | Derivation and interpretation |
|---|---|---|
| `hardware:belongs_to` | resource → FPGA | Documented class/capacity belongs to physical target; no program mapping |
| `target:arithmetic_compatible` | arithmetic operation → DSP/LUT | Conservative scalar operator whitelist and native operand/result types; possible implementation, not actual binding |
| `target:storage_compatible` | exact local memory object → BRAM/LUT/FF | Verified local scalar memref allocation; possible storage choices, not selected packing or partitioning |
| `target:loop_resource_relevance` | exact loop operation → resource | Compiler ownership/proven calls or public-query memory witnesses connect loop contents to an existing compatibility edge |
| `target:action_resource_relevance` | matched action point → resource | Exact `targets` edge plus supported loop/storage relevance; candidate directive values absent |
| `target:requested_clock` | matched loop action → FPGA | Exact loop target participates in requested global clock constraint; no achievable II or timing assertion |

Each edge answers three **separate questions**: what does the compiler establish,
what hardware capability is documented, and is an actual implementation known?
These are stored in three evidence fields:

```json
{
  "compiler_evidence": {
    "status": "proven",
    "nodes": ["<original compiler node ID>"],
    "edges": ["<original supporting edge key>"],
    "witnesses": ["<when derived: query/call/access receipts>"]
  },
  "hardware_evidence": {"status": "documented", "sources": ["<registry source ID>"]},
  "compatibility": {
    "status": "possible",
    "actual_binding": "unknown",
    "interpretation": "<precise relation-specific statement>"
  }
}
```

This illustrates the field layout; it is not a literal benchmark edge. Compiler
status is `proven`, `possible`, `not_established`, or `not_applicable`, as supported
by the underlying evidence. Resource membership has no program-side claim, so
that field is `not_applicable`. Implementation compatibility is always
`possible`; membership and clock facts use `not_applicable` because they do not
map a computation to hardware.

For example, a multiply can be a **proven compiler operation** and DSP support
can be a **documented device capability**, while implementing that multiply
with DSPs remains only **possible**. The overlay never combines these fields
into a single `proven` hardware-binding claim. If a loop's memory relevance is
uncertain, the corresponding action point's relevance remains uncertain too.

### Identities, serialization and provenance

Metadata retains original `program_graph_identity`, an exact
`program_payload_sha256`, profile ID/version/content hash, registry content hash,
numeric deployment context, implementation version and hashes of the overlay,
current query helper, V2 builder and semantic helpers. `target_context_identity`
hashes profile/registry/deployment independently of the source graph;
`overlay_identity` additionally includes program payload and implementation.
Canonical/compact inputs describe the same program semantics but have different
payload hashes because their serialized representations differ.

The CLI checks each input file's hash against the release manifest. When
validating a combined graph, it restores the program and checks its bytes
against the same recorded identity. Handwritten API test graphs may omit release
authentication; they can pass schema checks but **must not be described as
authenticated release graphs**.
The registry contains source IDs/URLs and exact capacity excerpts; referenced
registry bytes are sealed by its hash. Hashes, arbitrary IDs and source paths
are provenance, not proposed learned features.

`serialized()` uses the existing canonical sorted JSON conventions, rejecting
NaN; `write_graph()` validates and refuses different existing output bytes.
The underlying V2.2 writer/validator/schema are unchanged. No timestamps or
machine-dependent absolute output paths enter deterministic graph payloads.

## Deriving program–hardware relationships

### Arithmetic and compute relevance

Version 1 recognizes `arith.addi/subi/muli` when both operands and the result
have the same scalar integer type. It recognizes `arith.addf/subf/mulf/divf`
when operands and results have the same scalar f16/f32/f64 type. Add, subtract
and multiply connect to DSP and LUT resources; floating division connects to
LUT only. These are the implemented compatibility rules, not a complete account
of all hardware implementations. Index arithmetic, vectors, unsupported
operations and missing/inconsistent types get no arithmetic-compatibility links;
the exclusions are counted in diagnostics.

[UG1399 **2021.1** BIND_OP](https://docs.amd.com/r/2021.1-English/ug1399-vitis-hls/pragma-HLS-bind_op)
documents fabric/DSP arithmetic alternatives and automatic synthesis selection.
Its floating implementations can require multiple hardware primitives. The
overlay records only possible compatibility; it does not apply BIND_OP, use its
latency ranges or estimate implementation cost.
The graph retains lowered types and graph-level numerical-fidelity evidence:
`hardware_exact=false/null`, source correspondence `not_established`. A multiply
produced using simplified analysis headers therefore gets no inferred DSP count,
claim that it fits one DSP's width, or throughput estimate.

To relate a loop to compute resources, the overlay finds operations belonging
to its `loop_context` and follows **proven** calls into resolved helpers. It
keeps the exact call chain and supporting edge keys. Recursive or unresolved
calls do not become proof of a particular computation. This traversal reads
existing compiler facts; it does not run another alias, dependence or root analysis.

### Memory and storage relevance

Storage compatibility requires `root_kind=allocation`, a unique defining
`memref.alloc`/`memref.alloca` and a proven direct root resolution. The memory
must have at least one dimension, all sizes must be statically known, and its
elements must be ordinary scalar integer/f16/f32/f64 values. Its three
possible realizations are BRAM, distributed LUT storage, and FF-based storage.
These are general alternatives, independent of candidate pragmas. No complete
partition directive, demand, fit, port count or banking decision is assumed.

Version 1 adds no storage-compatibility links for formal/external parameters,
globals, views considered as separate objects, or unresolved roots. It also
excludes allocations with pointers, aggregates, zero rank, dynamic dimensions
or unsupported types. Their compiler facts remain intact and the exclusions
are counted. This intentionally limited coverage avoids treating every `memref`
argument as a local BRAM buffer. An established access through a view can still
reach an eligible allocation; the view is not a new physical memory.

Memory relevance reads the corrected `get_loop_semantics()` and
`get_action_point_relations(..., 'all')` outputs. The latter preserves verified
invocation contexts such as LUD helper-to-caller arrays. Supporting records keep
the concrete access keys, roots, memory operands, indices, views, call bindings,
certainty and `root_candidates_complete`. A possible access to an eligible
allocation can support a possible resource-relevance link. An incomplete root
set is never treated as a proven unique root. Accesses with no established root
remain in query summaries without creating unsupported resource links. Shared
roots do not prove overlapping accesses.

### Directive actions, dependencies and constraints

The original supported action kinds remain PIPELINE/UNROLL for loops and
ARRAY_PARTITION for arrays. Action-resource edges refer to the exact original
`targets` key plus supporting compatibility/relevance evidence. They do not use
a chosen directive value, such as an unroll factor or partition type.
Array dimensions remain in the original source binding and memory node;
access maps, index dependencies and invocation contexts remain in original
edges/public query witnesses. No bank-conflict proof is synthesized.

`action_annotations` records absolute slot, exact target, supported directive
names, all-evidence query hash/counts and (for loops) native trip-count metadata,
SSA-state keys, dependence keys/certainty counts and access evidence counts.
`dependence_completeness=not_established` means the analysis does not establish
that it found every dependence. `achievable_ii=null` means there is no estimate
of the initiation interval: the cycles between starting consecutive iterations.
All original dependence details remain on the program graph. `no_alias` is
counted as negative evidence and is never used to create resource connectivity.
The `action_slots` table preserves full absolute label spans and separate
registered/matched/unresolved/source-only masks, including unresolved trailing
registrations. No active or inactive slot is renumbered.

### How a future GNN could use these inputs

A future heterogeneous GNN (one that handles different node and edge types)
could pass capacity and budget information to the operations, local memories and
matched actions connected to each resource. This is the intended inductive
bias: provide relevant hardware context rather than ask the model to learn it
from a device name alone. The model could then learn how computations, memory
accesses and dependencies relate to a requested clock. U50 can use the same
four resource classes and numerical properties without requiring a new learned
device-ID embedding. Whether this helps predictions remains to be tested.

The graph stores raw properties. A future encoder may rescale capacities with
`log2(1+capacity)`, keep the available fractions, and express the clock as
`log2(clock_period_ns/10)`. Logarithmic scaling reduces the difference in scale
between large capacities and smaller numerical inputs. Use fixed documented scales or statistics
fitted on the training split only. Unknown values need explicit masks before
any placeholder numbers are inserted. No encoder or normalization is implemented
here, and scaling alone does not guarantee accuracy on new targets or clocks,
physical feasibility or improved QoR.

## Relationship to MLIR and prior research

* [MLIR DLTI](https://mlir.llvm.org/docs/Dialects/DLTIDialect/) defines
  `TargetDeviceSpecAttr` for optional device properties and `TargetSystemSpecAttr`
  for a collection of identified devices. This supports the separation of target
  metadata conceptually; using modern DLTI is **future work**, not a prerequisite
  or a claim about availability in this pinned older compiler.
* [Data Layout Modeling](https://mlir.llvm.org/docs/DataLayout/) addresses type
  storage sizes/alignment and scoped queries. Data layout does not establish
  FPGA scheduling, primitive binding or DSP cost. No layout-derived cost is added.
* Official [Affine](https://mlir.llvm.org/docs/Dialects/Affine/),
  [MemRef](https://mlir.llvm.org/docs/Dialects/MemRef/),
  [SCF](https://mlir.llvm.org/docs/Dialects/SCFDialect/) and
  [Func](https://mlir.llvm.org/docs/Dialects/Func/) explain loop/map, storage/view,
  structured-control and call abstractions. Current web docs are conceptual
  references; exact implemented operations and evidence are governed by the
  pinned native implementation and authenticated snapshots described above.
* Ye et al., **ScaleHLS**, HPCA 2022,
  [DOI 10.1109/HPCA53966.2022.00060](https://doi.org/10.1109/HPCA53966.2022.00060),
  [author paper, sections IV and V-E](https://arxiv.org/html/2107.11673v4):
  retain multiple IR abstraction levels. Adopted here: preserving loop, memory
  and operation levels. Its transformation/DSE/analytical estimation machinery
  is not adopted; this overlay has no candidate configuration or analytical QoR
  model. The paper does not establish this extension's hardware mappings.
* Ferretti et al., **Graph Neural Networks for High-Level Synthesis Design Space
  Exploration**, TODAES 28(2), article 25 (online 2022),
  [DOI 10.1145/3570925](https://doi.org/10.1145/3570925):
  hybrid control/dataflow graphs support learned latency/resource surrogates
  conditioned on directives, with adaptation experiments. Adopted here only as
  motivation for explicit typed program relationships. This pragma-free overlay
  neither predicts costs nor inherits their learning/generalization results.
* Ferikoglou et al., **GNΩSIS: Lessons Learned in Generating a High-Level
  Synthesis Dataset**, TACO 2026, [maintainer publication information](https://github.com/aferikoglou/GNWSIS#publication)
  and [dataset documentation](https://huggingface.co/datasets/aferikoglou/GNWSIS):
  motivates separating measured configurations/labels from static program and
  target inputs. The overlay imports none of the design-point CSV labels.
* Sohrabizadeh et al., **Robust GNN-Based Representation Learning for HLS**,
  ICCAD 2023, [DOI 10.1109/ICCAD57390.2023.10323853](https://doi.org/10.1109/ICCAD57390.2023.10323853),
  [author manuscript](https://par.nsf.gov/servlets/purl/10539416), already cited
  in `HARP-rep/README.md`: hierarchical representation and separation of program
  from transformations are relevant design principles. This work preserves
  existing compiler hierarchy and separates hardware in an overlay; it does not
  add HARP's learned pragma transformation or copy its feature dimensions.

Research references justify representation choices, not FPGA primitive-binding
proofs. Operator/storage compatibility is justified only by the compiler facts
and the explicitly cited vendor capabilities. No custom dialect, new compiler,
additional dependence solver or speculative analytical model is introduced.

## Reproduction (CPU only)

Use Python **3.10.20**, NetworkX **3.4.2** and pytest **9.1.1**, as tested here.
The overlay needs only Python and NetworkX; pytest is for tests. It does not
import PyTorch or native MLIR. Existing graph tests additionally exercise the
existing pinned native environment through their established subprocess fixtures.
No new compiler or toolchain dependency is installed.

All commands below run from the repository root. Choose fresh `/tmp` output
paths when rerunning; writers refuse to overwrite different artifacts.

```bash
PY=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python

# Existing outer release checksum; does not modify the archive.
sha256sum -c docs/graph-release-v22-rc3.sha256

# Authenticate every bundled content hash and create an owned inspection copy.
"$PY" -B -m GNN_branch.target_graph_v22 authenticate \
  --archive GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz \
  --output-dir /tmp/salt-v22-inspection

# Existing archived portable verification tool; independent historical consumer.
"$PY" -B /tmp/salt-v22-inspection/V22_PORTABLE_VALIDATION.py \
  /tmp/salt-v22-inspection /tmp/salt-v22-portable-receipt.json

# Materialize a target graph from the authenticated canonical stencil snapshot.
"$PY" -B -m GNN_branch.target_graph_v22 compose \
  --graph /tmp/salt-v22-inspection/canonical/spcl_example_03-fdea84258953.json \
  --target zcu104 --clock-period-ns 10 --output /tmp/stencil-zcu104-10.json

# New device, continuous unseen clock, restricted resource budget; same program.
"$PY" -B -m GNN_branch.target_graph_v22 compose \
  --graph /tmp/salt-v22-inspection/compact/spcl_example_03-fdea84258953.json \
  --target u50 --clock-period-ns 6.67 --available-bram18k 1344 \
  --available-dsp 1024 --available-lut 435840 --available-ff 871680 \
  --output /tmp/stencil-u50-6p67-budget.json

"$PY" -B -m GNN_branch.target_graph_v22 compose \
  --graph /tmp/salt-v22-inspection/canonical/spcl_example_03-fdea84258953.json \
  --target u200 --clock-period-ns 10 --output /tmp/stencil-u200-10.json

"$PY" -B -m GNN_branch.target_graph_v22 validate --graph /tmp/stencil-zcu104-10.json
"$PY" -B -m GNN_branch.target_graph_v22 inspect --graph /tmp/stencil-u50-6p67-budget.json

"$PY" -B -m pytest -q tests/test_target_graph_v22.py
"$PY" -B -m pytest -q tests/test_mlir_graph_v2.py tests/test_mlir_graph_v22.py

# Reauthenticate and test all 63 canonical/compact pairs; extra devices on 5 cases.
"$PY" -B -m GNN_branch.target_graph_v22 corpus \
  --archive GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz \
  --output /tmp/salt-target-corpus-validation.json
```

For a known platform allocation, `--platform-resources-json` accepts a JSON
object such as `{"BRAM_18K":300,"DSP":800,"LUT":100000,"FF":200000}`.
It must be genuine deployment metadata; the overlay does not discover or invent
a platform allocation. These values remain separate from a stricter user budget.
The API exposes `enrich`, `validate`, `strip_overlay`, `overlay_signature`,
`read_graph`, and `write_graph` for future CPU dataset preparation.
Run existing public compiler queries on `strip_overlay(joint)`, not directly on
the independently typed composed graph. No existing consumer is silently adapted.

## Recorded verification results and before/after comparison

The results in this section come from implementing and validating the overlay,
before the later readability edit. The linked JSON reports preserve the exact
commands and results; editing this README does not rerun those experiments.

Phase A's original documentation is independently preserved in commit `afe131c`.
The focused overlay tests pass **58/58** (14.27 s), zero failures/skips. Tests
cover real authenticated graphs plus explicitly hand-authored consumer units;
the latter are not represented as native compiler evidence. The first test runs
exposed new archive-reader assumptions about the names and self-checksum of
the bundled manifest files. The reader was corrected to match the frozen
archive; no historical test, checksum, archive or validator was relaxed.

Full-corpus validation passed **63/63 canonical/compact pairs**, zero failures,
zero blocked cases, in **1,026.85 s**. It authenticated the entire release,
validated both program views, checked exact canonical reconstruction, generated
and validated both overlays, compared their derived semantics, repeated
canonical enrichment byte-for-byte, and restored both original payloads exactly.
The additional U200/U50 contexts passed **10/10**: both devices on stencil,
AES, LUD, Hotspot pipeline and external Kalman. Together with ZCU104's full pass,
all three profiles were exercised on those five snapshots.

Machine receipts: [baseline](baseline-verification.json),
[full corpus](corpus-validation.json),
[tested CLI commands and exact example edges](reproduction-verification.json),
and [final preservation checks](final-verification.json).
All reported executions are CPU representation checks; no training, GPU model
inference, native full-corpus regeneration, HLS synthesis or QoR evaluation ran.

### Measured before/after statistics: ZCU104, 10 ns, full-capacity convention

| Authenticated snapshot | Original nodes / edges | Added nodes / edges | Enriched nodes / edges | Matched / registered actions |
|---|---:|---:|---:|---:|
| SPCL stencil `fdea84258953` | 95 / 162 | 5 / 54 | 100 / 216 | 6 / 6 |
| AES `883c223fddcf` | 4,204 / 53,788 | 5 / 83 | 4,209 / 53,871 | 6 / 6 |
| LUD `e35f0c4cd5d4` | 1,479 / 2,790 | 5 / 424 | 1,484 / 3,214 | 46 / 46 |
| Hotspot pipeline `c448680ca9be` | 695 / 1,505 | 5 / 164 | 700 / 1,669 | 14 / 19 |
| External Kalman `96b4d608969f` | 541 / 1,006 | 5 / 123 | 546 / 1,129 | 11 / 13 |
| **All 63 canonical snapshots** | **30,634 / 115,783** | **315 / 6,655** | **30,949 / 122,438** | **592 / 606** |

Hotspot's five unresolved labels and Kalman's L7/L8 remain unresolved. Across
the whole release, all 14 unresolved registrations and 13 source-only labels
retain their original identities. The original 51,659 possible dependencies
and six not-established concrete-access relationships remain unchanged.

Added relations across the full canonical corpus:

| Relation | Count |
|---|---:|
| `hardware:belongs_to` | 252 |
| `target:arithmetic_compatible` | 2,044 |
| `target:storage_compatible` | 891 |
| `target:loop_resource_relevance` | 1,226 |
| `target:action_resource_relevance` | 1,862 |
| `target:requested_clock` | 380 |

All 6,655 added edges cite documented hardware evidence. Their compiler-evidence
fields contain 6,403 `proven` entries and 252 `not_applicable` entries for
hardware membership. The conservative linking rules did not need to add any
edge with possible/unknown compiler evidence in this cohort. Separate synthetic
tests do exercise incomplete root sets and confirm that their uncertainty is
preserved.

The implementation-compatibility fields tell a different story: there are
**6,023 possible compatibilities**, 632 non-mapping facts and **zero proven
actual bindings**. None of these results establishes which resources HLS will
choose. Existing compiler uncertainty remains in the underlying graphs.

Serialized enriched canonical payloads total **199,637,711 bytes**. The stencil
is 221,861 bytes; AES is 67,826,368 bytes because its compiler evidence is
retained. The base ZCU104 canonical/compact/repeat checks take about 0.69 s for
the stencil and 286.86 s for AES; total corpus time also includes extra-profile
validation, authentication and serialization. These exhaustive timings are not
encoder throughput or synthesis runtimes. No graph was omitted to improve them.

The exact overlay implementation SHA-256 is
`d528c62be0cad9b8eec46898de7d3f6873d28bdf46d8d6d59a49401009dddfdf`;
the current consumer query hash is
`39aab15de84fecb46d108df194a6ebcad9e0db9963fba7bd155dbdcd036aba7c`.
The canonical registry-content hash is
`d005abae28531fed2f43b0cae68a5c3d9f86a0d641a48423c01f05516a418269`.
Receipts record file hashes separately from canonical-content hashes, environment
versions, base/documentation commits and exact commands. Runtime/host receipts
are outside deterministic graph content.

The implementation is independently preserved in commit `1b97487` after the
baseline documentation commit. Final checks confirm all **65 protected file
hashes** unchanged, including release files and the Stage-1 checkpoint; the
current pinned `cgeist`, analysis extension and lock hashes also match the
release. All **386 pre-existing tracked changes** are preserved. `main` still
points to `d488681f68e58ab1b1e666f6911f5257d6462dc2`; all task commits are on
`research/target-aware-mlir-v22`. These statements describe the recorded
implementation run. The branch was subsequently pushed to GitHub; it was not
merged into `main`. No validation was blocked in that run's environment;
independent human semantic review remains pending.

### Exact before/after neighborhood

These links were materialized from authenticated SPCL stencil bytes, not drawn
from an imagined mapping. Original facts remain:

```text
action:L5 --targets--> op:Stencil2D:op11
action:L2 --targets--> memory:Stencil2D:op1:r0
op:Stencil2D:op14 --memory_access(read)--> memory:Stencil2D:op1:r0
```

New links (all mapping/relevance edges are possible implementations):

```text
op:Stencil2D:op16 --target:arithmetic_compatible--> hardware:resource:DSP
  key target:42df60bbb9447c9001c46590; native arith.addf, f32 operands/result
memory:Stencil2D:op1:r0 --target:storage_compatible--> hardware:resource:BRAM_18K
  key target:f6e3f177a2d783f7c9057615; exact local allocation, unknown actual binding
op:Stencil2D:op11 --target:loop_resource_relevance--> hardware:resource:DSP
  key target:a6788899bc63eda549fc532a; contained arithmetic witnesses
action:L5 --target:action_resource_relevance--> hardware:resource:DSP
  key target:f81a66a491bb8058c3b573ac; original loop target and relevance edge
action:L2 --target:action_resource_relevance--> hardware:resource:BRAM_18K
  key target:0d684b1349a1fb53072b2602; original array target and storage edge
```

```mermaid
flowchart LR
  L5["action:L5"] -->|"original targets"| Loop["op:Stencil2D:op11"]
  L2["action:L2"] -->|"original targets"| Mem["memory:Stencil2D:op1:r0"]
  Load["op:Stencil2D:op14"] -->|"original concrete read"| Mem
  Add["op:Stencil2D:op16 arith.addf"] -.->|"possible arithmetic compatibility"| DSP["hardware:resource:DSP"]
  Loop -.->|"resource relevance"| DSP
  L5 -.->|"action relevance"| DSP
  Mem -.->|"possible storage compatibility"| BRAM["hardware:resource:BRAM_18K"]
  L2 -.->|"action relevance"| BRAM
  DSP -->|"documented membership"| FPGA["hardware:fpga; requested 10 ns"]
  BRAM -->|"documented membership"| FPGA
```

This neighborhood intentionally omits other verified nodes/edges for readability.
It expresses relevance and possibility, not DSP count, achieved latency,
memory-port pressure or a recommended directive assignment.

## Limitations and the next integration steps

This is a graph representation with possible resource relevance. It does not
implement a V2.2 tensor encoder, GNN training, learned target-aware action
embeddings, Stage-2 integration, hardware-exact resource binding, full HLS
scheduling, timing closure, synthesis-verified bank conflicts, or new QoR
measurements. Physical capacity is not deployment feasibility. Unsupported
operators, formal/global/dynamic/aggregate storage and uncertain roots deliberately
receive fewer links. All RC3 numerical and semantic limitations remain, including
pending independent adjudication and 14 unresolved registrations.

The proposed next learning task is a **new V2.2 target-aware heterogeneous GNN**,
subject to the semantic-review boundary below. It requires these explicit steps:

1. Define versioned tensor schemas for the six compiler categories, two overlay
   categories, 22 compiler relation kinds and six overlay relation kinds. Keep
   certainty, negative polarity, root completeness and missingness distinct.
   If reverse message-passing edges are needed, construct them only in tensor
   space with separate types; never relabel them compiler facts.
2. Encode static typed operations/types, loops, index/view/state relationships,
   resource capacities, availability and numeric clock. Fit normalization only
   on training data. Exclude source IDs, provenance hashes, query fingerprints,
   candidate pragmas and QoR labels from input features of this static
   program/target encoder. A new device must not
   require a learned device embedding.
3. Produce target-aware action vectors at exact slot `k-1`, with separate
   registered/matched/unresolved masks and explicit handling of missing targets.
   Preserve sparse and trailing slots. The conceptual memory shape is
   `[max_slots,128]` (normally 64 slots); shape alone is insufficient for
   compatibility: relation masks, labels, identities and provenance must agree.
4. Train and evaluate that new encoder under a declared family/target split;
   only afterward export new memory and integrate with existing Stage-2
   cross-attention, initially freezing the preserved Stage-1 latency adapter.

The current V1 GNN uses 403 node-feature dimensions and 82 edge-feature
dimensions. Its weights and structural memory are not compatible merely because
a future V2.2 encoder produces vectors of a familiar size. Before making learning
claims, independent semantic reviewers must review the RC3 worksheet or
explicitly approve the documented uncertainty boundaries. Neither the overlay
implementation nor this documentation edit closes that review.
