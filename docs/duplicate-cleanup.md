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
headers, and the HARP baseline source and measurements. `mlir-target` preserves
the organized baseline before the HARP consolidation below. Earlier layouts
remain in Git history, and `llvm-ir` preserves the original LLVM implementation.

Git already stores identical blobs once. This cleanup reduces duplicate paths
and repeated content in new checkouts; it does not claim to reduce historical
GitHub storage by the same amount.

## HARP baseline consolidation

On 2026-10-26, `HARP-paper-rep/` was consolidated into `HARP-rep/`. A byte-content
comparison found 807 identical files and two differing shared files. The former
configuration already exists at `HARP-rep/src/config.py.original`; the original
compiler script is preserved at `HARP-rep/dse_database/clang_script.sh.original`.
The only files exclusive to the removed tree were two generated PyTorch cache
metadata files, totaling 862 bytes, which were discarded. Its models, graphs,
measurements and results all have identical copies in `HARP-rep/`.

The retained tree also includes the external-kernel experiments and unseen-kernel
checkpoints. No callers outside the HARP trees referenced either directory;
the repository map and upstream provenance now point to `HARP-rep/`. Removing
the duplicate directory reclaims approximately 576 MiB in this checkout.

## Legacy entry points (2026-10-31 maintenance branch)

Four additional exact copies now delegate to their canonical implementation:
`src/data.py` → `GNN_branch/data.py`, `src/cpp_to_gnn_emb.py` →
`GNN_branch/cpp_to_gnn_emb.py`, `src/filter_jsonl.py` →
`Preprocessing/filter_jsonl.py`, and `src/inference_qLoRA.py` →
`LLM_branch/inference/inference_qLoRA.py`. Original source is recoverable at
`7c6844f`; canonical code is unchanged. The compatibility entries preserve legacy
module names, script guards and caller-selected sibling imports. Other `src/`
files and historical checkpoint snapshots are retained. See the
[maintenance report](REPOSITORY_CONSOLIDATION_REPORT.md) for exact measurements
and deferred artifact cleanup.
