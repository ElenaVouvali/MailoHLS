# GNOSIS: recovery of the 17 canonically ineligible kernels

Recovery branch: `research/gnosis-17-kernel-recovery-20261113`, isolated at
`worktrees/gnosis-17-kernel-recovery`, from GNOSIS FiLM commit
`5fa470038b93427f24e2582a075c1eafe74eef35`. No model training or synthesis ran.

**Result: 12 additional canonical kernels and 14 action histories are recovered.**
Complete source-action eligibility rises from **38 to 50 of 55**, provided the new
authenticated SQLite-state sidecars are joined. The untouched CSV/APL projection
alone still qualifies **38**. Five kernels/16 canonical action states remain
unresolved because their source/header equivalence to the measured upstream
version is unproven. This is not a relaxation of the eligibility test.

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
| `rodinia-knn-5-coalescing` | L12 | Loading loop projection loss; canonical source/header adaptation | 13-entry vectors; 5,676 matching rows; original L12 recovered | Preserve historical evidence only | Canonical `.range()` replacement and `__VIVADO_HLS__`-conditional 512-bit/32-bit representations are not proven experimentally equivalent | **Ineligible** |
| `rodinia_cfd_step_factor_1_tiling_0` | L5 | Outer tile loop absent from feature map and current metadata | 5-entry vectors; 3,617 matching rows; historical `L5,loop,1` | Restore exact metadata; recover L5 states | Regenerate affected RC3 metadata binding | Eligible + sidecar |
| `rodinia_cfd_step_factor_2_pipeline_0` | L5, L9 | Single-iteration compute/tile loops absent from feature map and metadata | 9-entry vectors; 3,010 matching rows; historical `L5,loop,1`, `L9,loop,1` | Restore both metadata records; recover both states | `.cpp` defines `PARA_FACTOR=1` after the header; do not infer 16 from the header; RC3 binding changes | Eligible + sidecar |
| `rodinia_cfd_step_factor_3_unroll_0` | L9 | Outer tile loop absent from feature map and metadata | 9-entry vectors; 4,944 matching rows; historical `L9,loop,1` | Restore metadata; recover L9 | Existing L5 remains the distinct 16-iteration action; RC3 binding changes | Eligible + sidecar |
| `rodinia_lud_1_tiling_0` | L20, L21 | Name-only array map overwrites these scopes with L39/L40 | 46-entry vectors; 5,760 matching rows; exact source/headers | Recover each array's own vector position | Structural consumers must preserve scope and action ID, not just variable name | Eligible + sidecar |
| `rodinia_lavaMD_1_tiling_0` | L8 | `WIPE_LOCAL_POS_O` initialization loop absent from feature map | 15-entry vectors; 5,685 matching rows from its own restored CSV | Recover L8; source differs only in documented whitespace after labels | Existing RC3 source/metadata mismatch remains | Eligible + sidecar |
| `rodinia_lavaMD_1_tiling_1` | L8 | Same projection loss, flattened-array source variant | 15-entry vectors; 5,739 matching rows; exact source/headers | Recover its own L8 | No cross-variant join; existing RC3 mismatch remains | Eligible + sidecar |
| `rodinia_lavaMD_2_pipeline_0` | L8 | Same initialization-loop projection loss | 15-entry vectors; 5,695 matching rows; exact source/headers | Recover its own L8 | Existing RC3 mismatch remains | Eligible + sidecar |
| `rodinia_dilate_2_pipeline_0` | L5, L6, L12 | Initialization/load/shift loops absent from optimized feature map | 20-entry vectors; 5,600 matching rows; original states recovered | Preserve historical evidence only | Top source matches, but `dilate.h` removes `<iostream>` and `mc.h` disables AP/wide-bus definitions/includes; full header equivalence not proven | **Ineligible** |
| `rodinia_dilate_3_pipeline_0` | L5, L6, L12, L25, L27, L29, L31, L33, L35 | Initialization, shift, boundary loading/blanking and storage loops omitted from feature map | 35-entry vectors; 5,760 matching rows; all nine original histories recovered | Preserve historical evidence only | Same header-identity restriction; do not transfer values from pipeline-0 or repair source semantics | **Ineligible** |
| `spcl_example_03` | L3, L4 | Buffer-loading loops absent from feature map | 6-entry vectors; 5,142 matching rows; both original states recovered | Preserve historical evidence only | Canonical source/header adds `extern "C"`; numerical-body similarity alone does not authenticate historical compilation identity | **Ineligible** |
| `spcl_example_05` | L5 | Result-storage loop absent from feature map | 5-entry vectors; 3,744 matching rows; original state recovered | Preserve historical evidence only | Source/header linkage edits; no authenticated same-version measurements supplied | **Ineligible** |

## Counts and supervision contract

| Category | Before | After |
|---|---:|---:|
| Canonical kernels | 55 | 55 |
| Authenticated complete source-action configurations | 38 | **50** |
| Complete using original CSV/APL alone | 38 | **38** |
| Ineligible/explicitly partial canonical kernels | 17 | **5** |
| Canonical unproven actions | 30 | **16** |
| Train / validation / locked-test complete kernels | 34 / 3 / 1 | **45 / 4 / 1** |
| Original measured rows changed | 0 | **0** |

**30 historical action histories** are genuinely reconstructed for original
upstream inputs; **14** of them, covering **59,196 original configurations in 12
kernels**, are transferable to the current canonical inputs under exact hashes
or the existing narrow label-spacing rule. The remaining **16 histories** are
authenticated only for different upstream source/header bundles. They are not
counted as recovered canonical states. **Zero** fixed/non-DSE declarations,
default assignments, or new measurements are introduced.

The new contract is `canonical-csv-apl-plus-authenticated-sqlite-states-v1`:

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
qualified by this audit. Even a matching RC3 source/metadata hash cannot establish
the measured header identity of the five source-incompatible kernels.

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

Final results: **34 regression tests pass**; canonical verification and cache
verification pass. Cache totals remain 54 tables / 152,560 unique design rows /
174,958 known measured raw rows / 94,102 CSV-projection-complete rows. All 54 cache
table hashes match their pre-recovery values. The first cache-generation attempt
correctly refused absent ignored tables in the new worktree; 54 registered copies
were materialized by their existing hashes and the full generation/check rerun
passed. That initial refusal is retained in the command log.

```bash
python -m Preprocessing.gnosis_recovery --verify
python -m Preprocessing.consolidate_data verify
python -m Preprocessing.consolidate_data verify-cache
python -m unittest Preprocessing.tests.test_canonical_data \
  Preprocessing.tests.test_canonical_preprocessing \
  Preprocessing.tests.test_final_experiment_contract \
  Preprocessing.tests.test_placeholder_contract \
  Preprocessing.tests.test_gnosis_recovery -v
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

## Recommendation for final FiLM + historical-GNN/GMT Stage-2

**Yes: 11 additional training kernels and one validation kernel (radix sort) can
be included under the authenticated joined complete-action contract**, after
the explicit sidecar join is implemented in a newly versioned supervision bank
and its targets/domains are validated. Keep the existing family split; filter
unknown/invalid QoR and enforce per-request budget feasibility using the same
declared training protocol. Fit any new normalization only on training families.

Before structural training, regenerate or otherwise authenticate the affected
graph/action-memory interfaces, validate every supervised action's slot/mask
semantics (including eliminated loops and LUD scopes), and enforce the historical
GNN/GMT/checkpoint identities. Source-action eligibility alone is not Stage-2
memory qualification. Do not modify old checkpoint contracts to bypass those checks.
Adding a training kernel and adding radix validation targets change cohort and
selection comparability: use a new run/bank ID and report that change, or hold the
old cohort fixed for a matched comparison. Keep the five source-incompatible
kernels excluded until their exact measured source/header identity is independently
resolved. No automatic incorporation or training was performed.
