# Canonical SALT-HLS measurements

Use [manifest.json](manifest.json) as the single dataset entry point. It identifies
the live raw measurements in `Data/CSVS`, their labeled source, APL mapping,
action metadata, SHA-256 hashes, family split membership, unknown action states,
and compatibility with the frozen RC3 graph release. GNN and LLM experiments
should refer to this registry rather than choose an arbitrary older data folder.

**13 November recovery:** [the 17-kernel audit](../docs/gnosis_17_kernel_recovery.md)
recovers 29 canonical action histories for 16 kernels from original SQLite vectors
and authenticated compiler correspondences. Complete source-action eligibility is
now 54/55 **with the explicit sidecar/correspondence joins**, while CSV/APL-only
eligibility remains 38/55. The original recovery at `3053f67` remains frozen at 50.
The manifest records both flags and each required contract. Coalescing KNN remains
canonically ineligible: its measured 512-bit source has a separately versioned
complete dataset and compiler graph, accessed only with
`Preprocessing.gnosis_compilation_recovery.complete_original_rows`. Original
measurements, sources and mappings are unchanged; four CFD
metadata records are restored with explicit before/after provenance.

`Preprocessing.gnosis_recovery.complete_rows` provides the strict complete-action
join; `python -m Preprocessing.gnosis_recovery --verify` checks its evidence.
`python -m Preprocessing.gnosis_compilation_recovery --verify` checks the four
Vitis-2021.1-bounded correspondences and the separate original-source KNN dataset.
Legacy CSV cache tables and the existing FiLM bank are not relabeled or expanded.
Do not use the new complete-action flag with a consumer that ignores its contract
and sidecar. Graph/memory qualification is separate from this source-state recovery.

There is one shared, reproducible derived cache: `Data/preprocessed_CSVS`.
The existing maintained preprocessor creates it for all measured device/clock
contexts. It retains finite measured resource demands at and above capacity,
uses the paper's existing latency and arithmetic-mean resource-area conventions,
canonicalizes directive spellings, and aggregates repeated measurements.
Recorded constant no-directive columns are retained explicitly; absent columns
are never created.
Unknown/invalid measurements remain in raw data and are counted in the registry;
they are not labeled resource-infeasible. Requested clock does not establish
achieved timing qualification.

```bash
# Standard library only: verify all canonical inputs and graph identities.
python -m Preprocessing.consolidate_data verify

# Requires the existing numpy/pandas preprocessing dependencies.
python -m Preprocessing.consolidate_data tables
python -m Preprocessing.consolidate_data verify-cache

# Regression checks, including locked-test protection and row identity.
python -m unittest Preprocessing.tests.test_canonical_data -v
```

The shared cache contains training/development tables only. Its manifest uses
`salt-hls-canonical-derived-v1` and explicitly records incomplete source-action
coverage. It cannot be passed off as an old SFT preprocessing manifest: doing
so would let the historical JSONL converter silently invent no-directive values
for unknown actions. Building complete training candidates must require
`complete_source_configuration_eligible`, or use separately authenticated applied
states. Do not infer those states from missing columns or optimized-away loops.

Reuse the locked split's **kernel/family membership**, not its old row indices,
dataset hash, fitted area floor or normalization statistics for the new cache.
Refit normalization from training families only. The locked Kalman test's rows
are not read by consolidation, indexing or shared-cache generation.

## Restored measurements

All 15 fetched GitHub branches and reachable Git history were searched. The
[search receipt](provenance/history_search_20261110.json) records exact refs,
commits and mapping states. These measurements were recovered from commit
`25dfef26f3e9d8a4f6356e672e8c552563a76a7b`:

| Kernel | Restored measured columns |
|---|---|
| CFD flux baseline | `Array_1` → `L1` |
| lavaMD baseline | `Array_2` → `L1` |
| lavaMD tiling-0 | `Array_6` → `L1`; original source-associated CSV |
| lavaMD tiling-1 | `Array_6` → `L1` |
| lavaMD pipeline-0 | `Array_1` → `L3`, `Array_6` → `L1` |
| Pathfinder baseline | `OuterLoop_1` → `L1`, `InnerLoop_1_1` → `L2` |

The [restoration receipt](provenance/restoration_20261110.json) records every
original/restored file hash, prior Git blob, source/header check and row check.
For five kernels, all design keys `(Version, Device, numeric clock)` and shared
measurements/directives match, allowing only numeric formatting tolerance
`1e-12`. Their original files restore recorded directive values without imputation.

The tiling-0 CSV had instead been replaced by a byte-identical copy of tiling-1
in `4f3c6ba`, despite distinct source layouts. It is restored as a whole from its
own original source-associated dataset: 5,685 rows replace the incorrect 5,739-row
copy. The corresponding source and local headers match the original, permitting
only whitespace differences immediately after source action labels. Source
identities, existing action IDs and common metadata definitions are preserved.

`restore` is reproducible and idempotent. It refuses unexpected edits and checks
all proposed files before writing:

```bash
python -m Preprocessing.consolidate_data history  # search fetched refs/history
python -m Preprocessing.consolidate_data restore
python -m Preprocessing.consolidate_data index    # explicitly register reviewed changes
python -m Preprocessing.consolidate_data verify
```

## Earlier gap assessment (10 November) and historical replay

The following describes the earlier CSV/mapping-only search. The linked
13 November SQLite audit supersedes its state-recoverability conclusion, while
preserving its source-identity, snapshot and graph-compatibility restrictions.

No historical mapping supplies sort-radix `L6`, or any of the other 26 previously
reported unmeasured source actions. The registry also checks labeled source
directly: four further source labels lack current metadata and measurements in
three CFD step-factor variants. Their old metadata records do not recover their
applied directive states. These gaps remain explicit, not filled with defaults.

The [latest recheck](provenance/action_state_recheck_20261110.json) compared all
55 newly supplied CSVs under `Data/CSVS/CSVS` against the canonical measurements:
every file is byte-identical. Canonical source, metadata and APL mapping hashes
are unchanged; 30 action states across 17 kernels remain unproven. The duplicate
CSV directory is not an additional canonical dataset.

Sort-radix `L6` is the `init_1` loop in
[sort.c, line 45](ApplicationDataset/machsuite-sort-radix/sort.c#L45). It clears
the 2,048-element bucket array and is called by `ss_sort` at line 85. The loop is
registered in `kernel_info.txt`, but has no measured CSV-column mapping in its
APL file. Source existence does not establish the directives applied per design.

The older `LLM_branch/Data/preprocessed_CSVS`, `GNN_branch/Data/preprocessed_CSVS`
and experiment artifact copies are **frozen historical inputs**, not competing
canonical datasets. They remain in place to replay existing checkpoints and
locked experiments. In particular, the current Kalman raw CSV already contains
the originally restored array columns, while its locked historical preprocessing
snapshot omits them; this is recorded without changing or evaluating the test.
Snapshot references remain registered when their files are not locally
materialized; restore those files from the artifact store only for historical
replay. Canonical verification checks a snapshot's hash when the file is present.

The six restored metadata files change RC3 action inventory hashes. The registry
lists seven graphs requiring regeneration, including the preexisting Kalman
metadata/release mismatch. Existing graph releases, checkpoints,
memory banks, normalization and XAttn weights remain historical artifacts. They
must not be relabeled as trained on this repaired dataset. Regenerate and validate
the affected RC3 graphs and candidate interfaces before training a new cohort;
the existing paper loader's identity checks intentionally reject a mixture.

The published legacy GNN workflow is reproducible at
`20028e7` on `research/final-architecture-v3-20261110`. Repaired-data training and
any downstream LLM/synthesis improvement are not established by this restoration.
