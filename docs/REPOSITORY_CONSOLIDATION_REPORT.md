# Final artifact consolidation and compatibility report

Current branch: `maintenance/mailohls-final-consolidation`, isolated at
`/tmp/MailoHLS-final-consolidation`, based on PR #2 commit
`4d707ca8079a79e4bbd22dff0b6849d7f7332cd3`. PR #2 and its branch are unchanged.
The earlier maintenance receipt is linked below; its statements about blocked
Release publication and only 31 KB savings describe the previous phase.

## Substantive artifact migration

The complete chain was executed: tracked Git blobs → deterministic archive →
SHA256 seal → GitHub Release → independent public HTTPS download → checksum
verification → original-path restoration → graph/table reader validation →
removal of only the authenticated indexed copies.

| Group | Indexed files externalized | Indexed bytes externalized | Preserved in source |
|---|---:|---:|---|
| Historical checkpoint evidence | 524 | 106,944,229 | Selected `gnn_final_absolute_direct_rank_s123` source/configuration/schema/split/metrics; `FINAL_ARTIFACTS.json` unchanged. |
| Generated HARP graphs | 436 | 100,961,929 | Baseline source, local modifications, license and generation scripts. |
| Materialized synthesis evidence | 391 | 32,860,763 | Python/shell tools, guides, canonical benchmark sources and manifests. |
| Derived training CSVs | 109 | 63,705,355 | Raw measurements, source/APL mapping, JSON schemas/manifests and family splits. |
| Total | **1,460** | **304,472,276** | Every removed byte is available through the authenticated restore contract. |

This reduces the latest tracked tree; it does not shrink Git history or reclaim
space occupied by retained local copies. Removal used an exact, reviewed path
list with `git rm --cached` in the isolated worktree. The original dirty checkout
and all local experiment outputs remain untouched. Source-copies retained in this
worktree are ignored runtime artifacts, not evidence that a clean clone works;
a separate clean-export restoration check is required and reported below.

Published Releases:

- [Complete catalog](https://github.com/ElenaVouvali/MailoHLS/releases/tag/archive/v22-graph-catalog-20261029), source `42f52df`: the pre-existing 218 MB archive has SHA256 `bba4b870b568da18265567797eb9610762d357825f1aa6507d85b4b3f4cb36cd`; all **2,218** checksums passed after independent download and extraction. All 63 snapshots and 1,512 overlays are preserved. The catalog branch was not deleted.
- [Research evidence v1](https://github.com/ElenaVouvali/MailoHLS/releases/tag/artifacts/research-evidence-v1-20261031), source `4d707ca`: four sealed trees, public `SHA256SUMS` and `ARTIFACTS.json`. Packages contain only already-public tracked blobs; no ignored/private model weights were read. HARP's existing BSD license is included. No new benchmark license is granted.

[Artifact migration manifest](../configs/artifact-migration.json) records every
externalized path, group, source commit, archive identity, policy and retained
contracts. [Artifact registry](../configs/artifacts.json) records actual published
URLs, SHA256, size, schemas, dependencies and original restoration roots. Raw Git
URLs for frozen compiler archives are explicitly not treated as independent
archival replacements.

## Compatibility review and implemented repairs

- PR #2's special-token correction checks token-to-ID identity, rejecting a
  permutation even when vocabulary size and ID set match. Training records an
  explicit mapping. Legacy reconstruction follows the recorded token order;
  checkpoint weights, tokenizer identities and directive domains are unchanged.
- Its Stage-1 launcher adds environment overrides with the same original learning
  rate defaults (`1.5e-5`, `5e-6`). No training settings changed in this mission.
- Its graph overlay/readiness code remains experimental and does not claim V1
  checkpoint interchangeability or improved target generalization. No graph,
  target capacity, model architecture or learning objective changed here.
- The existing four `exec` compatibility delegates were retained. Eight further
  byte-identical implementations now use the same namespace-preserving pattern:
  `cpp_to_gexf_deterministic`, `filter_pt`, `gexf_to_pt_zero`, `parameter`,
  `pragma_free_pts`, `pt_to_gnn_emb`, `result`, `server`. Canonical implementations
  stay unchanged. Their original bodies remain at the immutable source commit.
  [Every src file is classified](legacy-entrypoints.json); nonidentical code is
  retained, not declared obsolete from an import search. There are twelve delegates;
  all other source files retain distinct historical behavior.
- Namespace/CLI/pickle fixtures cover all twelve delegates. Actual legacy
  `result.Result` and `parameter.DesignParameter` pickle round trips also passed.
- PR #2 extraction's check-then-rename could replace a concurrently created empty
  destination. Extraction now reserves the destination exclusively and publishes
  regular files with non-overwriting links. It preserves Git executable bits.
- The same artifact module now packages committed blobs atomically and restores
  into existing roots. It verifies archive/member/schema/source identity, rejects
  symlink traversal (including internal links), refuses conflicting content or
  executable modes, and leaves identical files alone. A multi-file restore is
  resumable/idempotent but not a filesystem transaction; concurrent conflicts may
  leave a subset of verified files. It never silently overwrites user work.

## Validation and reproduction

Initial targeted PR review baseline: **30 passed in 5.43 s**. Initial new artifact
and delegate tests: **22 passed in 1.75 s**. Actual restored consumer report:
1,514 verified paths across four packages, 186 graphs (95,045 nodes / 278,887
edges), 179 CSV files (689,286 data rows), and 431 JSON files. These use the same
NetworkX/CSV/JSON readers as the existing consumers; this is not checkpoint
inference, new synthesis or a QoR claim. The selected `.pth` was not part of the
tracked source and remains separately supplied.

```bash
python scripts/artifacts.py fetch --artifact checkpoint-evidence-v1
python scripts/artifacts.py restore --artifact checkpoint-evidence-v1
# Repeat for harp-generated-graphs-v1, synthesis-evidence-v1,
# derived-training-tables-v1 when required by the selected experiment.
python -m scripts.maintenance.validate_restored_research --root . --report /tmp/restored-consumers.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -rs tests LLM_branch/tests
```

The full CPU and clean-source results, final metrics and publication status are
recorded below. No GPU training, HLS synthesis, production memory
migration or Stage-2 integration is performed.

## Remaining independent candidates and blockers

- `Data/CSVS/`: protected raw scientific input. No independently reviewed dataset
  redistribution/version license was established; no CSV, label or split is removed.
- Frozen `GNN_branch/MLIR_graphs/`: retained as required historical compiler releases.
  Catalog publication does not authorize changing their provenance or semantics.
- Selected checkpoint source/metadata remains tracked; weights remain local and
  were not part of any package. Shared initial states and automatic-clock assets
  remain protected. No experimental adapter is promoted.
- Unique `src/` behavior remains available. No new context-sensitive compiler/GNN
  implementation or generic workflow abstraction is introduced. Existing
  `refinement_search` already shares configuration, materialization, report parsing
  and synthesis helpers with `refinement`; no unnecessary utility layer was added.
- Later Stage-1/V3 GNN research branches retain their own contracts and outputs.
  No branch retirement or PR #2 merge is performed. Protected historical refs stay
  exactly at their original SHA. Branch dispositions in the earlier receipt remain
  review recommendations, not deletion authorization.

## Earlier audit and branch records

The [unaltered PR #2 report](https://github.com/ElenaVouvali/MailoHLS/blob/4d707ca8079a79e4bbd22dff0b6849d7f7332cd3/docs/REPOSITORY_CONSOLIDATION_REPORT.md)
contains the original full branch matrix, checkpoint/hash audit and previous
measurements. All existing branch heads remain unchanged in this mission;
only this new maintenance branch and the evidence preservation tag are added.
Catalog preservation status is updated above: the Release is verified and the
branch still remains for review. Local original status/index identity was checked
again against `/tmp/mailohls-final-original-status.txt` and
`/tmp/mailohls-final-original-index.patch` and passed byte-for-byte.

## Final mission validation and current integration status

- Full CPU suite: **480 passed, 21 skipped, 0 failed in 402.51 s**. One additional
  migration-policy guard added after collection is covered by the final focused
  suite: **23 passed**, including all twelve delegates, pickle/CLI behavior,
  deterministic packaging, traversal/links, concurrent destination creation,
  executable bits, idempotency and conflicting-file protection.
- The repaired packager regenerated the real HARP archive with the **same SHA256**
  as the published asset, confirming deterministic packaging of committed blobs.
- Clean export `9396168` actually omits the 1,460 externalized paths. From that
  export, four public HTTPS downloads and restores passed again; the complete
  graph/table consumer report is identical to the first independent restoration.
  See [machine-readable evidence](artifact-migration-validation.json).
- Protected sources/contracts have zero diff from PR #2: `FINAL_ARTIFACTS.json`,
  raw Data, family splits, selected checkpoint source/metadata, profiles, frozen
  compiler archives, training/inference implementations and model architecture.
  New code is artifact plumbing and compatibility delegates only.
- Maintained launcher syntax, changed Python compilation and maintained relative
  documentation links pass. No dependency installation/upgrade was performed.
- New preservation tag `artifacts/research-evidence-v1-20261031` points to original
  source `4d707ca`; the existing catalog tag was used without rewriting it. The
  complete catalog's archival gate now passes, but its branch remains untouched
  pending separate retirement review. All other branch dispositions retain their
  previous dependencies; no research branch was integrated or deleted here.
- PR #2 remains an open draft at its original head. This new maintenance branch
  carries the final migration; its draft PR and final clean-checkout smoke receipt
  are recorded below. Nothing is merged into main.

The final size comparison uses the committed source tree, excluding restored
ignored caches. Extra manifest/receipt bytes are included. Publication of assets
is not a claim that Git history has shrunk or that scientific results improved.

## Final result

**SIMPLIFICATION_IMPLEMENTED_AND_VALIDATED** for the source/artifact boundary and
compatibility changes in this mission. Full paired GNN learning, checkpoint
inference and FPGA synthesis remain separate research work, not maintenance
validation results.

Clean-source smoke: **108 passed in 39.86 s**, after fresh public downloads and
restoration from the clean export. Frozen RC3 verification passed there as well.
No invisible model/tensor cache or original checkout source files were required;
the already-installed Python dependencies were reused.

The new draft PR is [#3](https://github.com/ElenaVouvali/MailoHLS/pull/3); PR #2 is not modified or merged. The source
implementation commits are `42859ad` and `9396168`; the final report commit is
the published branch/PR head. No branch is retired and no history is rewritten.

| Metric | PR #2 baseline | Final source tree |
|---|---:|---:|
| Tracked files | 3,733 | 2,277 |
| Tracked bytes | 566,683,112 | 262,321,258 |
| Python files (including historical snapshots) | 458 | 304 |
| Exact duplicate groups | 468 | 311 |
| Repeated file paths | 1,136 | 584 |

Python-file reductions primarily move historical snapshots into the sealed
release; they are not claims that unique algorithms were eliminated. Eight exact
active copies were separately consolidated through compatibility delegates.

Restore instructions are in [reproducibility](reproducibility.md); source and
artifact ownership rules are in [development](development.md). Remaining work is
review/merge of the prepared changes, followed by separately reviewed integration
of active research branches. Raw data externalization and branch retirement are
not prerequisites to using the new artifact interface.

The independently downloaded catalog also passed the existing catalog `unpack`
consumer and current compiler/target-overlay readers for the Rodinia dilate
snapshot at ZCU104/10ns/full, with exact program restoration. No corpus was
regenerated. The process namespace exposed only this tool session; no claim of
live research-job inactivity was made, and no process was signaled.
