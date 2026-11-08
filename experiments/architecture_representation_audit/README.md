# Architecture representation audit package

This package records a CPU-first, frozen-model investigation. It does not train Stage-1/GNN/Stage-2, modify model checkpoints, use the GPU, change the production decoder, or touch the active Stage-2 worktree.

## Reproduce Phase 1 / frozen GNN extraction

The audit uses the checked-in RC3 archive and V4b family/data audit. The frozen V4b checkpoint remains external at the manifested run directory. Check its SHA against `experiments/v22_gnn_training/v4b/receipts_v3/artifact_manifest.json` before use.

The extraction environment was an isolated `/tmp` package path with Python from the existing `mailohls-llm-v2` environment, PyTorch 2.6.0, and PyG 2.6.1. No shared environment was changed. Reproduction commands:

```bash
python experiments/architecture_representation_audit/audit_cohort.py
PYTHONPATH=/tmp/mailohls-pyg-audit-site:. CUDA_VISIBLE_DEVICES= \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  experiments/architecture_representation_audit/extract_gnn_cpu.py
```

The first script authenticates the archive and maps each measured graph to the frozen family split. The second uses CPU only, extracts intermediate tensors, runs the documented in-memory input-group ablations, and writes large arrays under `/tmp/mailohls-gnn-audit-activations/`. The script records mapping and evidence overflow errors rather than repairing/truncating them. Compact row-level results are in `results/cohort_manifest.json` and `results/gnn_extraction_manifest.json`.

Rebuild the two compact figures with:

```bash
MPLCONFIGDIR=/tmp/mailohls-mpl-cache \
  /home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python \
  experiments/architecture_representation_audit/plot_results.py
```

`R0`–`R6` activation files are transient diagnostic data and intentionally excluded from Git. Their per-file hashes are tied to graph identities in the compact manifest; checkpoint provenance and hash are included. The extraction script uses explicit two CPU threads and no CUDA device.

## Report status

- Completed: authenticated multi-family cohort, RC3 compiler-label coverage, frozen V4b forward extraction, global diagnostic feature-group sensitivity, graph-to-evidence distances, and C-cap/mapping coverage.
- Static audit: tensor/compiler schemas, action-level compression, existing attention interface.
- Reused GEMV-only result: previous 32-layer Stage-1 summary; cross-kernel Stage-1 probes remain pending.
- Not run: Stage-1 GPU trace, new model training, learned Stage-2 evidence interventions, fusion-layer pilots, synthesis/QoR measurement, and historical 4-layer + JK training.

See `ARCHITECTURE_FINDINGS.md` for the integrated conclusions and evidence levels.
