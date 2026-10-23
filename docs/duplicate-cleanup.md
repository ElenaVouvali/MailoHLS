# Duplicate-file cleanup on main

The exact path mappings, hashes, experiment identities, and winning evaluation
steps are recorded in [`deduplication-manifest.json`](deduplication-manifest.json).
Every consolidated file was byte-identical to its retained canonical copy.

| Category | Change |
| --- | --- |
| Nested legacy GNN modules | Removed 10 duplicate files from `src/mailohls_gnn_training/`; the identical modules remain directly in `src/`. No tracked caller referenced the nested directory. |
| Stage-3 validation exports | Removed 11 tracked copies; the selected `stage3_adp_semantic_action_s789_rep2` run retains the shared validation dataset. The manifest maps each other run to it. |
| Historical benchmark copies | Replaced 110 exact copies in `old_harp_graphs/` with relative symlinks to `Data/ApplicationDataset/`. Remaining historical files are unchanged. Relative include layouts continue to work. |
| Winning-step metrics | Removed 27 per-step exports identical to the same run's `best_selection_metrics.json`. The checkpoint copy, winning-step association, and all different step evaluations remain. |

## Local run caches and resuming old runs

Existing local validation exports and per-step metrics are retained as ignored
runtime files. They are no longer tracked, and future training may regenerate
them. The DPO selection-cache loader expects its per-run validation path, so a
fresh clone should materialize the mapped file before resuming an old run with
`--reuse_selection_cache`. Copy the canonical dataset to the original path from
the manifest; create the parent directory if needed. Use an independent copy,
since training can overwrite its debug exports. Model weights are supplied
separately as before.

## Preserved material

The cleanup retains the differing historical implementations, all checkpoint
provenance snapshots, raw measurement tables, standalone canonical benchmark
headers, and both HARP baseline trees. `mlir-target` shares this organized
baseline. Earlier layouts remain in Git history, and `llvm-ir` preserves the
original LLVM implementation.

Git already stores identical blobs once. This cleanup reduces duplicate paths
and repeated content in new checkouts; it does not claim to reduce historical
GitHub storage by the same amount.
