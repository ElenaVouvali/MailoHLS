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
measurement databases are preserved with their license; upstream revisions are
recorded in `third-party-provenance.json`.

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
