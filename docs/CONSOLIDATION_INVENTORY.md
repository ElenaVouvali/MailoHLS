# Tracked-tree and branch review

This dry run covers the existing index at foundation
`3bd533b5c489a0ad7487c32d0fccfcdaf6657180`: **2,896 tracked paths**, about
**368.6 MiB** of current worktree bytes at those paths. New consolidation files
are excluded from that count. The README had already been shortened in the
worktree. Exact counts, largest files and branch SHAs are preserved in the
[compact inventory](evidence/receipts/consolidation_inventory.json).

The read-only audit writes the full path/reference inventory to a caller-selected
new JSON file. It neither edits the index nor reads ignored model weights:

```bash
python scripts/maintenance/audit_final_architecture.py --output /tmp/salt-tree-review-new.json
```

## Candidate directories

| Path | Tracked files | Tracked bytes | Matching tracked source/config lines |
|---|---:|---:|---:|
| `old_GNN_checkpoint` | 1 | 15,510 | 0 |
| `old_harp_graphs` | 234 | 1,235,989 | 110 |
| `HARP-rep` | 345 | 4,630,593 | 407 |
| `Checkpoints` | 32 | 4,228,094 | 1,770 |
| `LIFT_perf` | 47 | 300,697 | 1 |
| `LIFT_pareto` | 41 | 274,660 | 0 |
| `Histogram` | 11 | 221,031 | 5 |
| `mailohls_runs` | 46 | 22,665,870 | 415 |
| `mailohls_u50_adaptation` | 173 | 4,314,092 | 26 |
| `asplos_results` | 24 | 590,943 | 3 |
| `asplos_paper` | 1 | 870 | 0 |
| `paper_plots` | 8 | 144,851 | 14 |
| `Notebooks` | 3 | 1,676,500 | 0 |
| `artifacts` | 342 | 77,813,853 | 522 |

Reference counts are literal `git grep` line counts over tracked Python, shell,
JSON, YAML and TOML files. They include historical code and manifests. They do not
measure current execution or prove that a path without matches is unused.

**Retain every listed directory in this phase.** Concrete dependencies include
[the maintained memory export launcher](../scripts/training/export_gnn_memory.sh)
using `Checkpoints`, [the production registry](../FINAL_ARTIFACTS.json) using
`Checkpoints` and `mailohls_runs`, and
[the graph corpus generator](../GNN_branch/generate_mlir_v2_corpus.py) using a
checkpoint provenance split. `old_harp_graphs` contains source/header snapshots;
`old_GNN_checkpoint` contains a training configuration rather than a removable
weight payload. LIFT directories contain generated designs and headers whose
baseline provenance must be checked. Notebook/table files without matches still
need research-owner review. HARP attribution and license remain necessary.

The largest files include the three frozen compiler archives (about 13–14 MB
each), three paper-GNN prediction banks (about 11.4 MB each), registered Stage-3
debug predictions (about 9.6 MB), and target-coverage evidence (about 7.8 MB).
File size alone is not a removal criterion. No host artifact preservation or
signed-off path-removal list has been established here.

## Branch content inventory

The live remote-head query confirmed 17 branches; full SHAs are in
[the remote snapshot](evidence/receipts/remote_heads_20261111.tsv). The table uses
matching local remote-tracking refs and compares each to the foundation HEAD.
Commit counts are graph ancestry counts, not counts of independent ideas or
proof of content integration. Path counts compare the complete trees.

| Remote branch | Commits outside foundation ancestry | Different paths | Disposition |
|---|---:|---:|---|
| `llvm-ir` | 1 | 2,903 | Preserve untouched |
| `main` | 7 | 662 | Preserve until supervisor sign-off |
| `mlir-target` | 0 | 3,153 | Preserve untouched |
| `research/final-architecture-decision-20261109` | 0 | 400 | Preserve research provenance |
| `research/final-architecture-v3-20261110` | 0 | 0 | Foundation; preserve |
| `research/gan-stage1-tcad-20261110` | 5 | 432 | Report curated; code remains research |
| `research/gemv-stage1-full-depth` | 30 | 873 | Preserve pending content review |
| `research/gnn-refinement` | 0 | 512 | Preserve research provenance |
| `research/paper-gnn-latency-final-20261110` | 3 | 101 | Reports curated; local completion separately preserved |
| `research/rc3-fusion-foundation` | 0 | 469 | Preserve source/contract lineage |
| `research/rc3-stage2-multikernel-c` | 0 | 458 | Preserve source/contract lineage |
| `research/rc3-structural-memory-audit` | 1 | 518 | Preserve source/contract lineage |
| `research/refinement-mechanism` | 6 | 681 | Preserve pending content review |
| `research/stage1-gnn-representation-audit` | 1 | 487 | Preserve pending content review |
| `research/stage1-qor-preference` | 30 | 892 | Preserve measured-quality lineage |
| `research/stage1-refinement` | 29 | 835 | Preserve pending content review |
| `research/stage1-target-film-20261110` | 14 | 491 | Neutral report curated; no default promotion |

Eleven refs contain commits outside the foundation ancestry. Even zero such
commits does not authorize deletion: source links, historical trees, worktrees
and checkpoint resumes can still depend on a ref. Nine local worktrees are
registered. The new `final-architecture` branch is local and preserved.

No branch, runtime output or tracked candidate was removed. No merge, squash,
history rewrite, archive bundle or remote mutation was performed. Any later
removal needs a signed-off explicit path/ref list, verified artifact custody,
relevant tests and current job/worktree checks on the affected hosts.
