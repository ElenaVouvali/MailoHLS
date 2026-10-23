# Repository versions and generated artifacts

`main` continues the full history of `stage2-analysis-refactor` and includes its
previously uncommitted implementation changes. `llvm-ir` preserves the original
LLVM-based training path at the `MailoHLS_paper_v1` snapshot (`c9b2f43`). That
historical snapshot also contains experimental MLIR builders, but its active
GNN entry point imports the original `data` backend. No historical source has
been removed merely because it uses a different IR.

Only these two development branches are needed. Earlier commits remain in Git
history; the reorganization does not rewrite history or immediately shrink
GitHub's historical repository storage.

## What belongs in Git

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
