# GNOSIS: recovery of the 17 canonically ineligible kernels

Recovery branch: `research/gnosis-17-kernel-recovery-20261113`, isolated at
`worktrees/gnosis-17-kernel-recovery`, from GNOSIS FiLM commit
`5fa470038b93427f24e2582a075c1eafe74eef35`. No model training or synthesis ran.

**Combined result: 54/55 canonical kernels qualify; the remaining KNN kernel
qualifies only as a separately versioned exact measured-source dataset.**
The first recovery, frozen at `3053f674f88a3562796e96d9b9cecd6be3ec09fd`, raised
eligibility from 38 to 50. The compiler investigation below authenticates four
more kernels and 15 more canonical action histories under a distinct Vitis-2021.1
correspondence contract. Together, 16 additional canonical kernels / 29 action
histories are recovered. The untouched CSV/APL projection still qualifies **38**.
Canonical coalescing KNN retains one unproven action: its current effective
32-bit representation cannot inherit measurements of the original 512-bit design.
No training or HLS measurement campaign ran; frontend compilation alone ran.

## Evidence and root cause

The previous history/CSV-header searches were correct but did not inspect the
original SQLite design vectors. At pinned GNOSIS commit
[`8888db17`](https://github.com/aferikoglou/GNWSIS/tree/8888db17a6f344307de0a27cb431582f3f6fd093),
the six device/clock databases for every affected kernel retain full directive
vectors and QoR. All **102 databases / 196 selected source, metadata and mapping
artifacts** were retrieved by pinned Git blob identity; locked Kalman databases
were neither retrieved nor parsed. All **85,118 measured rows** join by original
version counter, device and requested clock. Counts and identities are bijective;
all six QoR fields and already mapped directives match. Numeric formatting alone
uses tolerance `1e-12`. There are **zero** mismatched QoR values, mismatched mapped
directives, clamped indices, missing design keys, or imputed states.

The evidence chain is available under
[`Data/provenance/gnosis_17_recovery_20261113`](../Data/provenance/gnosis_17_recovery_20261113/):

- `recovery.json`: every action's source line, statement, type, metadata record,
  original DSE vector index, observed states/counts, six database identities,
  source/header comparisons, and metadata before/after hashes.
- `states/*.jsonl.gz`: original design version/device/clock, SQLite rowid and key,
  recovered states, and a digest binding the six original QoR fields.
- `upstream_blobs.json`, `code_blobs.json`, `upstream_code/`: pinned provenance and
  exact exporter/generator code, with upstream licences retained.
- `before/`: original manifests, metadata, and source/header/mapping/measurement
  hashes. The canonical originals remain available at the base commit as well.

Full original artifacts remain separately under the ignored local path
`artifacts/gnosis_17_recovery_20261113/upstream/`; they are not substituted for the
canonical sources. Derived sidecars do not masquerade as original CSV columns.

### Why the fields disappeared

1. **Loops:** GNOSIS `loopAnalyzer.py` builds feature mappings from the optimized
   `src_info.json` loop inventory, which omits all 28 affected loop labels.
   Source existence and DSE intent therefore diverge from compiler-feature
   presence. In `directivesManipularor.py`, `get_directives_action_point_representation`
   decodes each full vector, but catches `KeyError` and skips labels absent from
   that mapping. Existing global unused `NDIR` columns are not evidence for these
   actions. None were repurposed to assign a state.
2. **LUD arrays:** `_get_application_array_map` in `arrayAnalyzer.py` keys arrays
   by **variable name**, losing scope. L39/L40 overwrite the earlier L20/L21
   declarations of `top_buffer`/`left_buffer`. These earlier arrays are used in
   `lud_perimeter`; they are not dead declarations or copies of the later arrays.
   Their independently varying original vector positions recover their states.
3. **CFD metadata:** four loop records disappeared from three MailoHLS metadata
   files, independently of the upstream APL projection loss. Their exact records
   survive in MailoHLS `25dfef26` and the pinned GNOSIS source bundle. They have
   been restored without changing any existing action definition.

The dictionaries distinguish the historical ZCU104/3.33 ns search space (array
limit 1024) from the other contexts (128). The upstream exporter's out-of-range
clamping fallback is **not** used. The original sequential label inserter's
positions are checked against every original source label. The generator code
is corroborating evidence, not a licence to rerun DSE or synthesize missing data.

## Per-kernel decisions

“Recovered” below means every original measured configuration has a matched
state for the listed actions. “Eligible + sidecar” always requires the explicit
join contract below. None of the 30 actions was proven fixed or outside DSE:
all have multiple recorded states in the original databases.

| Kernel | Missing actions | Root cause | Evidence found | Repair | Remaining uncertainty | Final eligibility |
|---|---|---|---|---|---|---|
| `machsuite-sort-radix` | L6 | `init_1` bucket-clearing loop omitted from optimized feature map | 11-entry vectors; 5,561 matching rows; exact source/headers | Recover L6 states | No unknown state; no new physical qualification | Eligible + sidecar |
| `rodinia-knn-1-tiling` | L8 | 2-iteration `LOAD_INPUTQUERY` loop absent from feature map | 9-entry vectors; 4,506 matching rows; exact source/headers | Recover L8 | Same limitation | Eligible + sidecar |
| `rodinia-knn-2-pipeline` | L8 | Same source-level loading action, separate measured variant | 9-entry vectors; 4,900 matching rows; exact source/headers | Recover its own L8 states | No values copied from tiling variant | Eligible + sidecar |
| `rodinia-knn-3-unroll` | L9 | Loading loop omitted from optimized feature map | 10-entry vectors; 5,462 matching rows; exact source/headers | Recover L9 | No unknown state | Eligible + sidecar |
| `rodinia-knn-4-doublebuffer` | L11 | Loading loop omitted from optimized feature map | 12-entry vectors; 4,317 matching rows; exact source/headers | Recover L11 | No unknown state | Eligible + sidecar |
| `rodinia-knn-5-coalescing` | L12 | Loading loop projection loss; changed AP representation | 13-entry vectors; 5,676 matched rows; Vitis original `i512` versus canonical `i32` | Exact measured-source bundle, complete join and separate compiler graph | Original run defines unavailable; new graph is AP simulation representation, not a historical numeric memory | **Canonical ineligible; original-source eligible** |
| `rodinia_cfd_step_factor_1_tiling_0` | L5 | Outer tile loop absent from feature map and current metadata | 5-entry vectors; 3,617 matching rows; historical `L5,loop,1` | Restore exact metadata; recover L5 states | Regenerate affected RC3 metadata binding | Eligible + sidecar |
| `rodinia_cfd_step_factor_2_pipeline_0` | L5, L9 | Single-iteration compute/tile loops absent from feature map and metadata | 9-entry vectors; 3,010 matching rows; historical `L5,loop,1`, `L9,loop,1` | Restore both metadata records; recover both states | `.cpp` defines `PARA_FACTOR=1` after the header; do not infer 16 from the header; RC3 binding changes | Eligible + sidecar |
| `rodinia_cfd_step_factor_3_unroll_0` | L9 | Outer tile loop absent from feature map and metadata | 9-entry vectors; 4,944 matching rows; historical `L9,loop,1` | Restore metadata; recover L9 | Existing L5 remains the distinct 16-iteration action; RC3 binding changes | Eligible + sidecar |
| `rodinia_lud_1_tiling_0` | L20, L21 | Name-only array map overwrites these scopes with L39/L40 | 46-entry vectors; 5,760 matching rows; exact source/headers | Recover each array's own vector position | Structural consumers must preserve scope and action ID, not just variable name | Eligible + sidecar |
| `rodinia_lavaMD_1_tiling_0` | L8 | `WIPE_LOCAL_POS_O` initialization loop absent from feature map | 15-entry vectors; 5,685 matching rows from its own restored CSV | Recover L8; source differs only in documented whitespace after labels | Existing RC3 source/metadata mismatch remains | Eligible + sidecar |
| `rodinia_lavaMD_1_tiling_1` | L8 | Same projection loss, flattened-array source variant | 15-entry vectors; 5,739 matching rows; exact source/headers | Recover its own L8 | No cross-variant join; existing RC3 mismatch remains | Eligible + sidecar |
| `rodinia_lavaMD_2_pipeline_0` | L8 | Same initialization-loop projection loss | 15-entry vectors; 5,695 matching rows; exact source/headers | Recover its own L8 | Existing RC3 mismatch remains | Eligible + sidecar |
| `rodinia_dilate_2_pipeline_0` | L5, L6, L12 | Initialization/load/shift loops omitted; unused header dependencies removed | 20-entry vectors; 5,600 matched rows; complete HLS LLVM module and expanded source body identical | Attach its own histories through compiler-correspondence v2 | Bounded reconstructed Vitis environment; no original per-run flag receipt | **Eligible + v2** |
| `rodinia_dilate_3_pipeline_0` | L5, L6, L12, L25, L27, L29, L31, L33, L35 | Initialization/load/shift/storage loops omitted; same unused includes | 35-entry vectors; 5,760 matched rows; independently identical complete module/body | Attach its own nine histories through v2 | Same environment bound; no cross-variant directive transfer | **Eligible + v2** |
| `spcl_example_03` | L3, L4 | Buffer-loading loops omitted; linkage-only source adaptation | 6-entry vectors; 5,142 matched rows; complete HLS module identical under explicit top-symbol correspondence | Attach its own histories through v2 | HLS top binding only; no claim of host ABI equivalence | **Eligible + v2** |
| `spcl_example_05` | L5 | Result-storage loop omitted; linkage-only adaptation | 5-entry vectors; 3,744 matched rows; independently identical module under its top-symbol correspondence | Attach its own L5 through v2 | Same restriction; unused `Entry` prototype only reformatted | **Eligible + v2** |

## Counts and supervision contract

| Category | Before | After |
|---|---:|---:|
| Canonical kernels | 55 | 55 |
| Authenticated complete source-action configurations | 38 | **54** |
| Complete using original CSV/APL alone | 38 | **38** |
| Ineligible/explicitly partial canonical kernels | 17 | **1** |
| Canonical unproven actions | 30 | **1** |
| Train / validation / locked-test complete kernels | 34 / 3 / 1 | **49 / 4 / 1** |
| Original measured rows changed | 0 | **0** |

**30 historical action histories** are genuinely reconstructed for original
upstream inputs. The first **14**, covering **59,196 configurations in 12 kernels**,
transfer under exact hashes or the narrow label-spacing rule. Another **15**,
covering **20,246 configurations in four kernels**, transfer only under the new
compiler correspondence. KNN's one history / **5,676 configurations** stays bound
to its original bundle. **Zero** fixed/non-DSE declarations, defaults or new
measurements are introduced. The earlier v1 receipt and its original decisions
remain byte-identical; the new receipt explicitly extends them.

The first recovery contract is `canonical-csv-apl-plus-authenticated-sqlite-states-v1`:

```python
from Preprocessing.gnosis_recovery import complete_rows

for record in complete_rows("machsuite-sort-radix"):
    measured = record["measurement"]          # original QoR and context
    actions = record["source_action_states"]  # every source action, by stable L-ID
```

The join checks source/header/metadata/CSV identities, unique original design
keys, complete action coverage and per-row QoR association. It rejects source
mismatches, missing/duplicate rows or states, changed inputs, and locked-test
access. It never fills missing values. The canonical manifest names the required
contract and sidecar; its separate `raw_csv_complete_source_configuration_eligible`
flag prevents conflating these views.

Join **before** directive canonicalization, aggregation or Pareto selection.
Old partial-cache rows can aggregate designs that differ in an omitted action;
their row indices cannot substitute for original version/device/clock identities.

The legacy cache and FiLM bank serializer do **not** join these sidecars. They
remain explicitly CSV-projection-only, with their original **37 non-test kernels**
eligible for that serializer. No existing FiLM bank, model, checkpoint contract,
normalization, or experiment result was updated. A new joined bank is a separate
reviewable experiment, not a repaired version of the existing 720-update A/B run.

## Graph and historical-input integrity

No source body, header, APL mapping, or raw CSV was changed. The only original
input corrections are the four historical metadata rows in three CFD files;
their before/after hashes and original versions are recorded. No stored hash was
rewritten to make a graph or checkpoint pass. The RC3 mismatch list grows from
**7 to 10**, adding those three CFD variants; the existing LavaMD mismatches
remain. The previous seven also include flux, baseline LavaMD, baseline
Pathfinder and locked Kalman. Graph releases and historical GNN weights remain intact.

The read-only `graph_action_audit.json` checks the original RC3 archive member
hashes, source bindings, and absolute `Lk -> k` action-node indices for all 30
actions. Six recovered kernels (radix, four KNN variants, LUD) already have matching
source/metadata fingerprints and seven compiler-matched recovered action nodes.
These node indices remain 1-based; any existing absolute memory-slot convention
`Lk -> k-1` remains unchanged. No action was renumbered.
LUD L20/L21 bind to `lud_perimeter`, independently of L39/L40. The three corrected
CFD variants have four `not_registered` actions with no action nodes in the frozen
archive. LavaMD has action nodes but mismatched source/metadata provenance. These
are different failure modes. No numerical historical-GNN or GMT memory is newly
qualified by this audit. Matching RC3 source/metadata hashes alone did not
establish the five measured header identities; the compiler evidence below
resolves four correspondences while preserving the KNN source distinction.

All missing loop labels exist in exact original sources and metadata (with the
four recovered CFD records), but not in the old optimized loop-feature inventory.
Do not invent GNN/GMT nodes, vectors, no-directive states, or action bindings for
compiler-eliminated loops. Correct target coverage does not prove memory-slot
coverage. LUD needs distinct scope-aware bindings for L20/L21 versus L39/L40.

Older preprocessed snapshots with fewer columns remain frozen replay inputs;
they neither supply missing states nor invalidate the new SQLite evidence.
The isolated cache is regenerated from unchanged raw measurements, and its
eligibility flags continue to describe the partial projection it actually stores.

## Verification and files

Executed checks and their captured output are recorded in
`Data/provenance/gnosis_17_recovery_20261113/validation_commands.json`; final integrity
checks are in `verification.json`. Re-run without GPU or synthesis:

The first recovery passed 34 regression tests. The follow-up passes **42 tests
and 19 subtests**, including the pytest placeholder cases; canonical and cache
verification pass. Cache totals remain 54 tables / 152,560 unique design rows /
174,958 known measured raw rows / 94,102 CSV-projection-complete rows. All 54 cache
table hashes match their pre-recovery values. The first cache-generation attempt
correctly refused absent ignored tables in the new worktree; 54 registered copies
were materialized by their existing hashes and the full generation/check rerun
passed. That initial refusal is retained in the command log.

```bash
python -m Preprocessing.gnosis_recovery --verify
python -m Preprocessing.gnosis_compilation_recovery --verify
python -m Preprocessing.consolidate_data verify
python -m Preprocessing.consolidate_data verify-cache
python -m pytest Preprocessing/tests/test_canonical_data.py \
  Preprocessing/tests/test_canonical_preprocessing.py \
  Preprocessing/tests/test_final_experiment_contract.py \
  Preprocessing/tests/test_placeholder_contract.py \
  Preprocessing/tests/test_gnosis_recovery.py \
  Preprocessing/tests/test_gnosis_compilation_recovery.py -q
```

The new regression cases cover vector bounds/clamping, executable-pickle
rejection, source/sidecar drift, missing/duplicate designs or action states,
incorrect QoR association, incompatible source reuse, locked-test access, and
keeping sidecar-only recovery out of the legacy bank builder. Historical recovery
itself is reproducible with `Preprocessing.gnosis_recovery --upstream PATH` from
the isolated base plus supplied recovery code/provenance inputs; it refuses to
overwrite an existing recovery receipt.

Exact changed paths are listed in the companion `changed_files.json`. They comprise:
three CFD `kernel_info.txt` files; `Data/manifest.json`; the isolated
`Data/preprocessed_CSVS/manifest.json`; `Preprocessing/consolidate_data.py`;
new `Preprocessing/gnosis_recovery.py` and its focused tests;
`experiments/gnosis_stage1_ab/rebuild_non_test.py`'s CSV-only selection guard;
the recovery provenance/sidecars; `Data/README.md`; and this report. Existing
CSV cache table hashes, raw measurements, sources, mappings, family split and
locked-test inputs are checked separately from these intentional changes.

## Follow-up: the five remaining source/header bundles

Full evidence is in
[`gnosis_5_compilation_20261113`](../Data/provenance/gnosis_5_compilation_20261113/).
Every original source/header/metadata file is retained by the pinned upstream
Git blob and SHA-256. `source_changes.patch` records all original/canonical
differences; `correspondence.json` binds each bundle, CSV, sidecar, expanded body,
LLVM module and newly derived graph. Before-manifests and protected-input hashes
are retained independently. The parent recovery receipt and its 17 sidecars
are not rewritten.

### Compiler environment and limits

The archived generator documents Ubuntu 18.04.6, Python 3.6.9 and Vitis 2021.1.
It emits `set_top <kernel_info first line>` and `add_files <source>` with **no
explicit `-cflags`**, sets device/clock, disables automatic loop pipelining with
`config_compile -pipeline_loops 0`, and sets array partition threshold 0 /
throughput-driven off. Directive insertion follows the same stable source labels
used by the recovered vectors. Original per-measurement compiler invocations,
environment variables, installation build IDs and injected defines were not
retained. They cannot be retroactively authenticated.

Coroni's installed Vitis HLS 2021.1 frontend reports Clang 7.0.0. A retained
completed HLS experiment's `autopilot.flow.log` supplies an observed invocation:
`-fhls -std=gnu++14 -target fpga64-xilinx-linux-gnu`, the vendor GCC-6.2 toolchain,
AP/autopilot include paths and forced `autopilot_ssdm_op.h`, disabled exceptions,
thread-safe statics and cxa-atexit, and defines `__VITIS_HLS__`, `AESL_SYN`,
`__SYNTHESIS__`, `__HLS_SYN__`, `__DSP48E2__`. Its macro dump does **not** define
`__VIVADO_HLS__`. This is corroboration from the installed tool, **not an original
GNOSIS log**. The evidence records the full command/log hash, compiler and
disassembler hashes, installed AP header hashes, 42 successful preprocessing /
macro / LLVM emission / disassembly commands and their outputs. No `vitis_hls`,
`csynth_design`, export, implementation or new QoR measurement was run.

The v2 contract is deliberately bounded to the reconstructed Vitis-2021.1 HLS
compilation. It establishes compiler correspondence, not identical historical
files, equivalent host executables or newly replayed timing/QoR. Whole modules
are compared without removing declarations, instructions, types, HLS annotations
or attributes. Only `ModuleID` / `source_filename` paths and the enumerated SPCL
top symbols are normalized. The expanded source bodies must also match.

### Exact differences and decisions

| Kernel | Effective compilation differences | Evidence and decision |
|---|---|---|
| `spcl_example_03` | `extern "C"` added to declaration and definition of `Stencil2D`; N=M=T=1000, float widths, arrays, loops, labels and body unchanged | Original `_Z9Stencil2DPKfPf` maps uniquely to canonical `Stencil2D`. Expanded tokens and complete HLS modules match after that symbol correspondence. Same source-level `set_top Stencil2D`, one definition, no overload/wrapper. **Canonical eligible under v2; original eligible.** |
| `spcl_example_05` | `extern "C"` added to `MatrixMultiplication` prototype/definition; prototype formatting and unused `Entry` declaration formatting change; N=K=M=1024 and double widths unchanged | Original `_Z20MatrixMultiplicationPKdS0_Pd` maps uniquely to canonical `MatrixMultiplication`. All other module bytes match after path normalization. `set_top MatrixMultiplication` remains uniquely bound. **Canonical eligible under v2; original eligible.** |
| `rodinia_dilate_2_pipeline_0` | Source byte-identical; `dilate.h` comments `<iostream>` and adds EOF newline; `mc.h` comments `MARS_WIDE_BUS_TYPE`, `ap_int.h` and the three `mars_wide_bus*.h` includes | Original wide-bus headers are recovered exactly. Expanded complete source body and complete HLS LLVM module match without symbol changes. No synthesized function uses these AP/wide-bus definitions. Float widths, signatures, pragmas, loops and PARA_FACTOR=32 unchanged. **Canonical eligible under v2; original eligible.** |
| `rodinia_dilate_3_pipeline_0` | Same header-only differences; its own source is byte-identical, with PARA_FACTOR=8 | Independently identical expanded body/module; no values copied from pipeline-2. GRID_ROWS=544, GRID_COLS=512, TILE_ROWS=32, TILE_COLS=128, STREL=5x5 and radius=2 preserved in both variants. **Canonical eligible under v2; original eligible.** |
| `rodinia-knn-5-coalescing` | Header replaces unconditional coalescing `ap_uint<512>` with an `__VIVADO_HLS__` branch; false branch uses `uint32_t` and whole-word `GET_RANGE`/`SET_RANGE`. Source replaces two `.range()` operations with those macros | Installed HLS flags select the false branch: original module uses `i512*`, canonical uses `i32*`; loads/stores, extraction/insertion and resulting helper inlining differ. **Canonical ineligible; exact original-source dataset eligible separately.** |

For KNN, the original extraction/insertion spans 32 bits at offsets 0..480,
covering a 512-bit word; WIDTH_FACTOR=16 in both headers does **not** preserve
512-bit storage in the canonical false branch. The true macro branch expands to
the same `.range(hi,lo)` operations with harmless parentheses and the same unsigned
32-bit payload; a separately labelled counterfactual preprocessing probe confirms
that branch can be selected. There is no original flag evidence authorizing it
for the canonical bundle, and its existing compiler representation must not be
silently reinterpreted. No macro, source, header or graph identity was patched.

Each Dilate module compares every helper plus `workload`, not merely a selected
loop. SPCL compares the complete module, including signature/array-dimension HLS
annotations. Thus the original action-labelled bodies are preserved before the
historical generator inserts their respective measured directives. Every state
still joins its own version/device/clock and QoR; no measurement is transferred
between variants and no original measured directive was changed.

### Versioned datasets, compiler graphs and contracts

| Explicit contract / view | Eligible kernels | Interpretation |
|---|---:|---|
| `canonical-csv-apl-v1` alone | **38/55** | Original complete CSV/APL projection |
| Frozen `canonical-csv-apl-plus-authenticated-sqlite-states-v1` cohort at `3053f67` | **50/55** | Original 12-kernel recovery remains reproducible |
| Above plus `canonical-csv-apl-plus-vitis2021.1-source-correspondence-v2` | **54/55** | Four additional canonical kernels, 15 histories, 20,246 measured configurations |
| Above plus `gnosis-measured-source-8888db17-sqlite-states-v1` for KNN | **55 distinct kernel/source versions** | 54 canonical versions plus one original measured KNN version; **not 55 canonical kernels** |

`gnosis_recovery.complete_rows` reports v2 explicitly on the four newly qualified
kernels and rejects canonical coalescing KNN. Its exact measured-source dataset
is accessed separately:

```python
from Preprocessing.gnosis_compilation_recovery import complete_original_rows
for row in complete_original_rows("rodinia-knn-5-coalescing"):
    source_version = row["source_version"]  # GNWSIS-8888db17/rodinia-knn-5-coalescing
    actions = row["source_action_states"]   # all 13 authenticated states
```

This view references, without copying or editing, the original CSV / 5,676
configurations and L12 sidecar. Each row names the measured bundle, source version,
new graph and original-source contract. Existing train membership is unchanged;
canonical KNN is not included alongside it. No unresolved historical state remains
for the original bundle; its equivalence to canonical KNN remains rejected.

All five **original** bundles now have separate, compressed Polygeist v7 GEXF,
MLIR and native compiler-analysis artifacts. Their action counts are 13 / 20 /
35 / 6 / 5, all matched by exact compiler source location, with no action fallback.
Compiler, generator, native extension, source, metadata, flags and transitive
header hashes are recorded. KNN's genuine AP simulation path preserves each word
as 8 x 64-bit storage; it is **not native HLS i512 IR**. Four array-scope warnings
and conservative unresolved memory effects remain explicit. No fake AP header,
scalar substitute, source rewrite or broad compiler change was introduced.

These newly derived graphs do not replace RC3 graphs or authenticate numerical
historical-GNN/GMT embeddings. The existing RC3 audit has source/metadata matches
and all 15 additional action nodes compiler-matched for the four canonical
recoveries. Its KNN source match still refers to the incompatible canonical
representation. A separate measured KNN structural memory must be generated and
validated against its exact AP graph and the frozen encoder's input contract.
Do not reuse canonical KNN embeddings. The pre-existing ten RC3 mismatch cases
from the first recovery remain unchanged.

Follow-up validation and the exact changed-file inventory are in this new
provenance directory's `validation_commands.json`, `integrity.json` and
`changed_files.json`. Both recovery verifiers, canonical verification and existing
cache verification pass. All 54 cache table hashes and all protected original
input hashes are unchanged, including all 55 CSVs, sources, headers, mappings,
split membership and the earlier restored metadata. Full cache regeneration
changed only its manifest's current registry/code hashes and four explanatory
authenticated-eligibility flags; its raw-projection flags remain false for these
four. The existing FiLM serializer therefore still selects 37 non-test kernels.
The initial test attempt used an interpreter without NumPy/pandas/pytest; its
import failures are retained, followed by the successful existing environment's
42-test run. Focused new tests reject altered original blobs/parent receipts,
changed IR semantics and canonical KNN reuse while validating the separate join.

Follow-up changes are confined to `Data/manifest.json`, the isolated cache manifest,
`Preprocessing/consolidate_data.py`, a small shared join extraction in
`Preprocessing/gnosis_recovery.py`, new compiler-correspondence code/tests, two
frontend/evidence scripts under `experiments/gnosis_recovery`, the new provenance
directory, `Data/README.md`, and this report. No canonical source/header, measured
CSV, APL mapping, model, checkpoint, split or original receipt changed.

## Recommendation for final FiLM + historical-GNN/GMT Stage-2

**The four additional canonical kernels can safely become eligible for complete
directive supervision under the explicit v2 correspondence contract.** Together
with the first recovery, this permits 15 additional training kernels and one
validation kernel (radix sort), after
the explicit sidecar join is implemented in a newly versioned supervision bank
and its targets/domains are validated. Keep the existing family split; filter
unknown/invalid QoR and enforce per-request budget feasibility using the same
declared training protocol. Fit any new normalization only on training families.

Before structural training, regenerate or otherwise authenticate the affected
graph/action-memory interfaces, validate every supervised action's slot/mask
semantics (including eliminated loops and LUD scopes), and enforce the historical
GNN/GMT/checkpoint identities. Source-action eligibility alone is not Stage-2
memory qualification. Do not modify old checkpoint contracts to bypass those checks.
Adding training kernels and radix validation targets changes cohort and
selection comparability: use a new run/bank ID and report that change, or hold the
old cohort fixed for a matched comparison.

**KNN requires the separately versioned measured source.** Its original directive
supervision and source graph are recovered; its canonical equivalence and
historical-GNN/GMT numerical memory are not qualified. No kernel is left without
an authenticated original source-action dataset, but canonical KNN remains
excluded. Do not merge the new graphs into a historical structural checkpoint
contract or supervision bank automatically.

**There is no reason to delay Stage-2 for further source recovery.** Use the
already qualified cohort, or the 54 canonical source-eligible kernels after the
new bank/graph/memory contract checks above. Treat original-source KNN as a
separate follow-up unless its encoder/memory interface is qualified independently.
No automatic bank expansion, training, synthesis or push was performed.
