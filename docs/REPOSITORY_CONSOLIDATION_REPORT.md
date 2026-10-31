# Repository consolidation report

Status: **SIMPLIFICATION_PARTIALLY_IMPLEMENTED**. High-confidence source deduplication,
artifact verification and navigation improvements are implemented. Large scientific
artifacts and unique historical implementations remain preserved; their removal
is deferred to independently verified archival publication and compatibility review.

## Preservation and baseline

Maintenance base: `7c6844f7ca26b53cfb0d6b54c25aba95f61b950a`, the published
consolidation branch, 12 commits beyond main `d488681f68e58ab1b1e666f6911f5257d6462dc2`.
Worktree: `/tmp/MailoHLS-simplification`; branch
`maintenance/repository-simplification-20261031`.
Original checkout remains at `72b42ae`, with 78 staged additions and 386 deletions.
Its porcelain status and binary index diff were recorded outside Git at
`/tmp/mailohls-original-status-20261031.txt` and
`/tmp/mailohls-original-index-20261031.patch`. This task neither commits nor
changes that original index. The earlier independent bundle remains at
`/tmp/mailohls-consolidation-preflight.bundle`; linked worktrees do not replace it.
The process namespace exposed no research process details; no training output was
modified, no process signaled, and no GPU or synthesis workload launched.

## Before/after and exact changes

The committed base contains 3,723 files, 566,713,851 bytes, 455 Python files,
472 duplicate blob groups and 1,140 redundant paths (124,624,848 repeated bytes).
These numbers count paths in the Git tree, including symlinks; they are not Git
object-store usage. Comparison is against the consolidation base, not older main.
Final measurements are recorded below after committing the implementation.

Four exact copies in `src/` now delegate to their maintained source:

| Legacy entry | Canonical source | Safety evidence |
|---|---|---|
| `src/data.py` | `GNN_branch/data.py` | Byte-identical before change; same legacy module namespace retained. |
| `src/cpp_to_gnn_emb.py` | `GNN_branch/cpp_to_gnn_emb.py` | Byte-identical; caller sys.path and API preserved. |
| `src/filter_jsonl.py` | `Preprocessing/filter_jsonl.py` | Byte-identical; CLI guard and caller imports preserved. |
| `src/inference_qLoRA.py` | `LLM_branch/inference/inference_qLoRA.py` | Byte-identical; canonical implementation and model settings unchanged. |

No standalone historical `src/` implementation was deleted. Execution in the
legacy namespace preserves class pickle module names; it deliberately does not
redirect unrelated LLVM-era sibling imports. Tests check both import identities
and script execution. Canonical files are byte-unchanged. Original copies are
recoverable at the base commit. This removes approximately 2,450 duplicate
implementation lines rather than creating another active package.

Added: standard-library artifact CLI and three graph-release registry entries,
committed-tree/branch/blob/AST inventory tooling, focused compatibility/security
tests, architecture/reproduction/experiment/developer guides and development
status. README is shorter and points to those guides. Repository layout now
distinguishes intended permanent branches from actual research refs. `.gitignore`
adds only generated maintenance reports. No directory migration was performed.

## Scientific content and deferred candidates

All datasets, family splits, selected checkpoint metadata, `FINAL_ARTIFACTS.json`,
V2.2 archives/manifests, profiles, baseline sources and measured reports are
unchanged from the base. The Kalman CSV/APL/metadata edits remain excluded as
decided in the previous consolidation. No experimental adapter is promoted.

| Candidate | Tracked size / paths | Classification and blocker |
|---|---|---|
| `Checkpoints/` | 111,172,323 bytes / 556 files, 165 Python files | REQUIRED_PROVENANCE; source snapshots may contain uncommitted run-specific behavior. Retain until per-run source archives and config contracts are independently verified. |
| `plots/ta_synthesis/` | 28,650,754 bytes / 296 files | EXTERNAL_ARTIFACT_CANDIDATE; repeated headers coexist with directive/source variants and measured reports. Retain pending exact materialization manifests and published archives. |
| `artifacts/final_salt_probes_20260907/` | 4,251,340 bytes / 107 files | Same provenance constraint; no blanket source replacement. |
| `Data/GAN/`, `evaluation_set/` | 735,093 / 2,475,566 bytes | PROTECTED canonical source/input relationships; repeated copies alone do not establish interchangeable execution paths. |
| `GNN_branch/MLIR_graphs/` | 77,116,474 bytes / 114 files | Frozen compiler releases; retained and checksum-verified. |
| `artifacts/gnn_datasets/`, `LLM_branch/Data/` | 31,208,382 / 32,578,259 bytes | Dataset contracts/splits; retained without relabeling or normalization. |
| HARP generated graphs | 58,104,236 bytes / 381 files | Baseline reproduction dependency; upstream provenance and local modifications retained. |

`refinement_search` already imports configurations, experiment and synthesis
utilities from `experiments/refinement`; creating another shared HLS layer would
add indirection without eliminating verified duplication. Target contracts for
production and graph profiles have different versioned evidence requirements;
they were not forcibly unified. Root utilities and result files retain their
paths because no compatibility benefit justifies another move. Launchers with
different experimental parameters remain distinct scientific configurations.

## Artifact migration and dependency audit

`configs/artifacts.json` binds immutable Git commit URLs, SHA256 and size for
V2.1, RC2 and RC3 graph archives. `scripts/artifacts.py` lists/verifies offline,
downloads to a temporary file then publishes without overwrite, and restores
validated regular-file/directory archives to a fresh location. Links, traversal
and special members are rejected. No large artifact is removed. The full catalog
remains on its published catalog branch; no GitHub Release is asserted here.

Full reports are outside Git in `/tmp/mailohls-simplification-before/`:
`repository-inventory.json`, `branch-inventory.json`, `duplicate-inventory.json`,
`dependency-inventory.json`, `artifact-inventory.json`. Reproduce them with
`python scripts/maintenance/inventory.py --ref 7c6844f --output /tmp/fresh-inventory`.
The inventory records every tracked path, blob, SHA256, size, owner, path-based
protection classification and AST import/dynamic-call evidence, and every remote
branch's differing file blobs and unique commits. Automated classifications are
review aids; static import searches do not prove absence of runtime, checkpoint
or historical dependencies. Unknown regeneration/archive/dependency claims stay
explicitly unverified. Shell/config/checkpoint paths were manually reviewed for
the four delegates; the remaining large groups are not certified safe to delete.

## Branch disposition

All heads below were refreshed from origin before isolation. Unique commits are
relative to origin/main; differing file counts are relative to the consolidation
base and include documentation/ancestry differences, not just missing features.
No branch or tag was deleted or rewritten. Published branch heads themselves
preserve unintegrated work; earlier annotated archive tags remain intact.

| Branch | Head | Unique commits / differing files | Purpose / integration | Disposition / blocker |
|---|---|---:|---|---|
| `docs/framework-development-status-20261031` | `5dbede9fb576fa1e171b42f1ebd3c608032d074a` | 1 / 6 | Development navigation; Status doc integrated and qualified | retire after merge; No current runtime probe |
| `feature/v22-structural-semantic-finalization` | `cc00a6e2acd7e87eba9076c12def1e7afb70dc91` | 0 / 10 | Frozen graph semantics; Compiler release already integrated | retire only after merge/dependency review; Verify preservation tag |
| `graphs/latest-v22-catalog` | `42f52df75c73496f4c21b7809415cf4e0d64b787` | 6 / 2226 | Complete generated graph catalog; Generator integrated; full catalog excluded | retain; No independent complete Release verified |
| `llvm-ir` | `0b5581b2b22becf1d20f476e22e92ea3fd533a64` | 1 / 34 | Historical LLVM baseline; Protected unchanged | retain permanently; Never update |
| `main` | `d488681f68e58ab1b1e666f6911f5257d6462dc2` | 0 / 5 | Production and authenticated RC3; Base ancestor | integration target; None |
| `maintenance/repository-consolidation-clean-20261029` | `7c6844f7ca26b53cfb0d6b54c25aba95f61b950a` | 12 / 0 | Earlier 12-commit integration; Exact maintenance base | retire after reviewed merge; PR not merged |
| `mlir-target` | `bf82c648273a5cd5eb30363059a3809e7b3979db` | 0 / 937 | Historical MLIR baseline; Protected unchanged | retain permanently; Never update |
| `research/hotspot-v22-graph-export-20261028` | `4a308a0ceb84b7ccfad743697576c08fe2a59d49` | 1 / 8 | Authenticated Hotspot export; Archive preserved on published branch | retain until artifact migration; Not a new source correction |
| `research/stage1-directive-learning-repair` | `72b42ae5e95b40d7cd57f3eade9b4b4cdb258731` | 10 / 10 | Tokenizer/recovery and Kalman data edits; Correctness/recovery integrated; data excluded | retain pending data version review; Kalman dataset identity changes |
| `research/stage1-full-data-recovery` | `f577a1ad342b21cb6eb9d2e65596389207e64e7d` | 17 / 12 | Full-data Stage-1 recovery; Separate active research; not imported | retain; Separate data/checkpoint contract |
| `research/stage1-measured-quality` | `0b592a9e0186dc033b42f7c52b0e7bab6276cc93` | 19 / 13 | Measured validation assessment; Separate research; not imported | retain; Incomplete run; not production model |
| `research/target-aware-mlir-v22` | `84e78a42b473469a96c2cb0a7994b380359ea0fc` | 4 / 6 | Hardware overlay and documentation; Integrated in base | retire only after merge; Preservation tag and PR review |
| `research/target-graph-readiness` | `982c72da716179c95cad4dda044a32ceb1ca7058` | 5 / 4 | Experimental tensor pilot; Integrated in base | retire only after merge; CONDITIONAL GO remains unchanged |
| `research/v22-action-hardware-demand` | `7c231f4b60871b91421ecaeb8683873fd1861c61` | 14 / 7 | Static/candidate workload descriptors; Later research not imported | retain; Ordered semantic/training integration required |
| `research/v22-gnn-resumable-20261030` | `bf93db9e159ad55c4f096fa89e8b230c73516ff5` | 24 / 24 | Atomic resume and paired training protocol; Latest training infrastructure, not imported | retain; Last receipt blocked by GPU contention |
| `research/v22-semantic-coverage-repair` | `26c22474bfcfd26d7ee55c51772b1e708bbe3ec5` | 17 / 14 | Affine/dependence coverage audit; Separate supplementary audit | retain; Do not rewrite historical RC3 |
| `research/v22-semantic-tensor-enrichment` | `21ed70492e74e167354c1cb21d5a499e2f434014` | 16 / 11 | V3 semantic tensor encoder; Later research not imported | retain; Pilot schema differs; separate tests/contracts |
| `research/v22-target-aware-gnn-training` | `43431de1e77efaf42692db3cfece7773541bb37c` | 20 / 18 | First supervised trainer; Superseded in part by resumable branch | retain pending follow-up integration; No completed result |

These counts do not authorize branch deletion. Protected `llvm-ir` and
`mlir-target` retain their exact SHA. Later V3 encoder and resumable trainer
changes should be integrated in their own ordered, tested PR; maintenance does
not merge an active experimental training contract incidentally.

## Validation and final integration

Baseline: 85 overlay/readiness/tokenizer tests passed in 40.53 seconds.
New compatibility and artifact tests: 11 passed in 0.52 seconds.
RC3 offline verification and restoration to a fresh temporary directory passed;
archive SHA256 remains `feb2c2d469bf6672248ad0f3a88099ac9627b3b10b8e139a0fab050d532a1b73`.
Broader suite, clean-checkout smoke, publication and final measurements follow
below as executed evidence. No GPU, full model training or Vitis synthesis is
part of this maintenance validation.

## Prioritized next PRs

1. Review and merge the existing consolidation plus this maintenance work.
2. Integrate action-demand, V3 semantic tensorization and resumable trainer in
   dependency order, preserving their schema versions and original pilot results.
3. Publish/download/verify the complete catalog Release before retiring its branch.
4. Audit checkpoint snapshot source archives per run, then repeated synthesis
   source materialization. Potential upper bounds above are not promised savings.
5. Version any Kalman dataset corrections separately, with checkpoint compatibility
   notes and unchanged original datasets.

### Executed regression evidence

- Complete CPU suite (`CUDA_VISIBLE_DEVICES='' python -m pytest -q tests LLM_branch/tests`):
  **468 passed, 21 skipped, 0 failed**, 408.18 seconds. Collection preceded one
  additional path-escape regression; the focused final maintenance suite includes
  that case and passed all 11 tests. A thread-limited rerun records skip reasons.
- Changed/new Python and production GNN/Stage-1/Stage-2/Stage-3 entry-point syntax
  passed. Maintained training/reproduction shell scripts passed `bash -n`.
- New documentation relative links passed, and no broken symlinks were found.
- Public HTTPS RC3 fetch into `/tmp/mailohls-public-restore-20261031` passed size
  and SHA256 verification; local verified extraction passed in a fresh directory.
  `gh release list` returned no releases. Catalog externalization remains deferred.
- Original porcelain/index patch comparison passed byte-for-byte; the staged
  patch SHA256 is `dbd65d851ef0bf35d222b3ad4ccc05992894d0d76ca481cbb88aa4bcb0bd5a17`.
- `git bundle verify /tmp/mailohls-consolidation-preflight.bundle` passed again.
- Zero diff against base for Data, selected runtime metadata, checkpoint sources,
  FPGA profiles, graph archives and graph-release manifests. No runtime weights
  were migrated. Full checkpoint inference and Vitis execution are unexecuted;
  isolated environments still need their documented external assets.

### Measurement snapshot

At implementation/documentation commit `009e6b9`: 3,733 paths, 566,677,814 bytes,
458 Python files, 468 duplicate groups and 1,136 repeated paths. Relative to the
consolidation base this is 10 extra documentation/tool/test paths, 36,037 fewer
tracked bytes and four fewer exact implementation copies. Final report additions
slightly change bytes; final working-tree measurements are recorded below.
Main alone has 3,696 paths, 565,985,079 bytes and 441 Python files. Comparing this
branch to main also includes the 12 already-prepared consolidation commits;
that is a different scope from the simplification-only measurement.

No claim of smaller Git history or object storage is made. New guides and tests
increase file count while reducing redundant implementation responsibility.

Clean-checkout smoke: a fresh `git archive` export (no `.git`, runtime caches or
weights) passed **96 tests in 45.00 seconds** for artifact/delegate, graph-overlay,
tensor readiness and tokenizer behavior, plus RC3 verification and artifact CLI
help. Installed Python dependencies were reused; no environment installation or
model download was performed. Before/after five-part inventories are available
outside Git under `/tmp/mailohls-simplification-before/` and
`/tmp/mailohls-simplification-after/` (the latter at `009e6b9`).

Draft PR: https://github.com/ElenaVouvali/MailoHLS/pull/2 . The new maintenance
branch is published. `main` is not merged, all original research branches remain,
and no branch retirement is authorized by this PR. Implementation commits are
`b75a2e9` (code/tooling), `009e6b9` (guides), and `d13719b` (validation receipt).
The final audit commit is the branch/PR head, discoverable via `git rev-parse HEAD`.

Final reviewed tree: 3733 files; 566683112 bytes; 458 Python files; 468 duplicate groups; 1136 redundant paths.

Manifest resolution in the clean export: the selected family split, memory path
and selected Stage-1/2/3 metadata directories exist. The selected GNN `.pth` is
not supplied in source; directory existence does not establish weight availability.
Full checkpoint deserialization/inference therefore remains an external-asset
check, not a passed test here. The 21 skipped checks concern unavailable historical
or external graph/source fixtures and the explicitly CUDA-only FP16 smoke.

Local-only `maintenance/repository-consolidation` is an alias at main `d488681`;
local `feature/v22-structural-semantic-finalization` also points to main while its
remote head remains `cc00a6e`. They introduce no missing source. Other research
local heads match their published refs. Inventory tooling records both local and
remote refs explicitly; detached worktrees are preserved in the original worktree
audit and are never treated as disposable branches. Final inventories can be
regenerated with `--ref HEAD`; report scope and commit are embedded in each run.

Final thread-limited CPU rerun: **469 passed, 21 skipped, 0 failed in 405.90 s**,
including the added path-escape case. Command:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -rs tests LLM_branch/tests`.
Skips: 20 unavailable historical/external/compiled fixtures and one CUDA-only
FP16 smoke. PyTorch Geometric tests ran; PyG was not a missing dependency.
Full logs remain `/tmp/mailohls-simplification-cpu-bounded.log`; final inventory
output is `/tmp/mailohls-simplification-reviewed/`. Current installed project
Python/PyG dependencies were reused without upgrades.
