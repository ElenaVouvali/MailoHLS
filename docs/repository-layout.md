# Repository versions and generated artifacts

`main` continues the full history of `stage2-analysis-refactor` and includes its
previously uncommitted implementation changes. `llvm-ir` preserves the original
LLVM-based training path at the `MailoHLS_paper_v1` snapshot (`c9b2f43`). That
historical snapshot also contains experimental MLIR builders, but its active
GNN entry point imports the original `data` backend. No historical source has
been removed merely because it uses a different IR.

The repository has three branches: `llvm-ir` for the original implementation,
`mlir-target` for the preserved organized MLIR + target-conditioning baseline,
and `main` for further framework extensions. After the shared file, launcher,
and artifact cleanup, `main` and `mlir-target` were synchronized to the same
commit. Develop future extensions on `main` and keep `mlir-target` at that
baseline. The initial pre-cleanup snapshot (`26db8ec`) remains in Git history.
No additional worktree is required. The reorganization does not rewrite
history or immediately shrink GitHub's historical repository storage.

## What belongs in Git

Local refinement has one maintained package, `experiments/refinement/`, and one
entry point, `python -m experiments.refinement.experiment`. Its four modules
separate configurations, compiler evidence, synthesis and experiment execution.
Do not add numbered experiment packages or duplicate policy implementations
for each research update. Extend the existing modules and tests instead.

The shared directive-domain module contains current TCAD factors and explicit
checkpoint compatibility. Serialized policy/schema identifiers remain stable
because existing checkpoints and artifacts refer to them; they are not separate
implementation directories. Superseded experiment launchers were removed from
the active tree. Exact historical code is recoverable through Git history and
an ignored local source archive, while original measurements remain unchanged.

Generated refinement reports, source snapshots, graphs, receipts and HLS
projects belong under ignored `experiments/refinement/results/`. The user guide
is `experiments/refinement/README.md`; it documents the current interface rather
than a sequence of historical research milestones.

Keep implementation, tests, source patches, compiler locks, kernel sources,
measurement tables, evaluation results, experiment configurations, family splits,
checkpoint contracts, and source provenance. The upstream HARP source and
measurement databases are consolidated in `HARP-rep/` with their license;
upstream revisions are recorded in `third-party-provenance.json`. The original
paper configuration and compiler script are retained as `.original` files;
see `duplicate-cleanup.md` for the consolidation details.

Generated weights (`.pt`, `.pth`, `.safetensors`, `.bin`), encoder/tensor caches,
intermediate `checkpoint-*` output, repeated tokenizer files, logs, Python
caches, duplicate archives, and accidental empty CLI-output files are excluded.
Existing small tracked contracts/configurations in runtime directories remain
available for provenance. Regenerating tensors or training should not stage
large artifacts automatically.

## Local models and recovery

Selected models, structural memories, measurements, and experiment outputs are
kept locally during consolidation. Obsolete binary files in `old_*` directories,
intermediate checkpoint output, optimizer state, and caches may be cleaned;
source files in those directories are preserved.

Local recovery material is outside the tracked tree in
`.local-archive/reorganization-20261023/`: a Git bundle of the original refs,
working-tree patches, file manifests, and cleanup records. This directory is
ignored. It is a recovery copy, not a development worktree or another branch.

## Selected model runs

The active `mailohls_runs/` directory keeps the five selected checkpoint folders
and family split listed in `FINAL_ARTIFACTS.json`. Selected `best_custom_stage*`
directories remain intact. Their directive registries, validation budget banks,
training contracts, and selected Stage-3 datasets/pairs are retained for
reproducibility. Intermediate checkpoints and duplicate last-step model exports
are outside the active directory.

Two supporting components are retained even though the final-artifact manifest
does not inventory them: `stage2_initial_states/` is required by the maintained
Stage-2 training launcher, and the selected model, cases, calibration and metrics
under `auto_clock_v8_final/` support the documented automatic-clock evaluation.
Generated clock-feature caches are archived. Do not remove these dependencies
merely because they are absent from `FINAL_ARTIFACTS.json`.

Unselected runs and intermediate files were moved to the ignored recovery copy
`.local-archive/mailohls-runs-20261026T132025Z/`. Its `manifest.json` inventories
all original files, records moved paths and retained-file SHA256 hashes, and its
`moves.jsonl` records each completed move. The active directory shrank from
approximately 8.6 GiB to 438 MiB. Archiving is reversible and does not reclaim
that disk space; model weights and historical experiments have not been
permanently deleted. The final-artifact manifest remains unchanged.

To restore the original layout, first ensure that no new run occupies an
archived path, then run:

```bash
python .local-archive/mailohls-runs-20261026T132025Z/restore.py
```

The recovery script refuses to overwrite existing paths. Historical run metadata
also remains in Git history. This cleanup does not automatically commit or push
the working-tree removals of previously tracked experiment metadata.
