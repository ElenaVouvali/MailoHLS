# MailoHLS repository consolidation

This is the historical 2026-10-29 audit receipt. Subsequent verified Release
publication and tracked artifact migration are recorded in
[the final consolidation report](REPOSITORY_CONSOLIDATION_REPORT.md). Statements
below about unavailable Release publication describe that original audit date.

## Audit snapshot

Audit date: 2026-10-29. Main began at `d488681f68e58ab1b1e666f6911f5257d6462dc2`.
The protected branches were `llvm-ir` `0b5581b2b22becf1d20f476e22e92ea3fd533a64`
and `mlir-target` `bf82c648273a5cd5eb30363059a3809e7b3979db`.
Other observed local heads: semantic finalization `d488681f68e58ab1b1e666f6911f5257d6462dc2`,
target overlay `84e78a42b473469a96c2cb0a7994b380359ea0fc`, readiness
`982c72da716179c95cad4dda044a32ceb1ca7058`, Stage-1 recovery
`72b42ae5e95b40d7cd57f3eade9b4b4cdb258731`, and graph catalog
`42f52df75c73496f4c21b7809415cf4e0d64b787`. The remote semantic-finalization
head was `cc00a6e2acd7e87eba9076c12def1e7afb70dc91`; its local branch had
already converged to main. Remote branch heads must be rechecked before cleanup.

The original checkout at `/home/elvouvali/MailoHLS` remains on
`research/stage1-directive-learning-repair`, HEAD
`72b42ae5e95b40d7cd57f3eade9b4b4cdb258731`. It had 464 staged paths: 78
Hotspot graph/source additions and 386 deletions, no unstaged changes and no
untracked files. Its exact porcelain status and staged binary patch are backed
up outside the repository at `/tmp/mailohls-original-status-v2.txt` and
`/tmp/mailohls-original-staged.patch`; their SHA256 hashes are recorded in the
work log for this task. The original index and files were not changed.

The 3.1 GB bundle `/tmp/mailohls-consolidation-preflight.bundle` passed
`git bundle verify`; it contains all 22 original refs, including the research
branch heads and stash. A linked worktree is not an independent object backup.
Available disk at audit: 120 GB on `/home`, 63 GB on `/tmp`.

## Integration decisions

The clean worktree is `/tmp/MailoHLS-consolidation-clean`, branch
`maintenance/repository-consolidation-clean-20261029`, based exactly on the
audited main commit. All work below is confined to that worktree.

Integrated as separate commits:

- `afe131c`, `1b97487`, `658ff4c`, `84e78a4`: frozen V2.2 RC3 documentation,
  FPGA overlay implementation, profiles, tests and verification reports.
- `982c72d`: experimental target graph tensor encoder, tests and the original
  `CONDITIONAL GO` readiness report.
- `506eb03`: checkpoint tokenizer-ID correctness fix and regression test.
- `d2cb43d`, `10ce2da`, `2c57182`: isolated Stage-1 recovery tools, tests and
  reports. They remain experimental; no adapter is promoted.

The Kalman CSV, action map and `kernel_info.txt` changes from `a9cb717` were
excluded. They alter training examples and dataset hashes. Their independent
versioning and compatibility review remain required before integration.

## Branch-only file audit

The comparison is path- and content-based against current `main` and the
integration worktree, not just commit ancestry:

| Branch | Original head | Unique work and disposition | Preservation | Retirement after merge? |
| --- | --- | --- | --- | --- |
| `feature/v22-structural-semantic-finalization` | remote `cc00a6e2acd7e87eba9076c12def1e7afb70dc91`; local ref had converged to main | No path differences from main at the remote head; final V2.2 release and adjudication documents already on main. | Annotated tag above; original commits remain in the bundle. | Yes, once its remote PR/dependencies are checked. |
| `research/target-aware-mlir-v22` | `84e78a42b473469a96c2cb0a7994b380359ea0fc` | Builder, profile registry, overlay tests, full technical guide and four verification receipts are integrated. No dataset or graph archive changed. | Annotated tag above. | Yes, after PR merge and ref verification. |
| `research/target-graph-readiness` | `982c72da716179c95cad4dda044a32ceb1ca7058` | Experimental tensor encoder, forward/reverse typed relation readiness, action-aligned memory tests, config and report integrated. Report retains its conditional-go result and does not claim improved generalization. | Annotated tag above. | Yes, after PR merge; keep experiment files on main. |
| `research/stage1-directive-learning-repair` | `72b42ae5e95b40d7cd57f3eade9b4b4cdb258731` | Tokenizer correction and test plus recovery scripts, controls, reports and tests integrated. Experimental checkpoints remain unpromoted and are not committed as production weights. | Annotated tag above; original checkout patch and bundle also retained. | Yes, only after Kalman data changes are separately resolved. |
| `graphs/latest-v22-catalog` | `42f52df75c73496f4c21b7809415cf4e0d64b787` | Contains 63 snapshots, 1,512 enriched graphs, source/MLIR review pages, manifests, receipts and catalog tooling. Generated variants remain outside the source tree. Tooling is integrated; complete catalog remains on branch pending Release upload. | Annotated tag, remote branch, locally prepared and checksum-verified archive. | **No** until an independently downloadable complete Release asset is verified. |

Unique excluded Stage-1 files are `Data/ApplicationAPLMapping/serrano-kalman-filter.txt`,
`Data/ApplicationDataset/serrano-kalman-filter/kernel_info.txt`,
`Data/CSVS/serrano-kalman-filter.csv`, and `docs/graph-update-kalman.json`.
The changes register L7/L8 and fill corresponding label columns, but change
canonical dataset inputs and hashes. Existing checkpoint/training reproduction
must continue using its original data version; do not cherry-pick these changes
without a versioned dataset/split contract and verified correspondence audit.

The other catalog-branch-only files are generated per-kernel graph JSON gzip
files, per-context hardware graphs, and per-snapshot review material under
`graphs/latest/`. Their complete content is classified as generated research
artifact (retain in release/branch), not source code to merge into main. Reusable
code and target profiles are integrated above. The branch's index, manifest and
checksums remain the provenance for the prepared archive.

The feature branch has no unique file content relative to main at its verified
remote head. The catalog branch's apparent deleted Stage-1 files are ancestry
differences; those files remain present in this integration branch. No unique
baseline source or measurement was removed.

The full graph catalog remains accessible on the preserved
`graphs/latest-v22-catalog` branch. No GitHub Release asset was created, so the
branch must not be deleted. The catalog builder is included in this
consolidation worktree; its copy helper no longer reads generated catalog
review files. It passed syntax and CLI-help checks, but a full expensive rebuild
was deliberately not run. A deterministic archive was made from the existing
branch and verified against all 2,218 catalog checksums and the tar member paths.
Archive: `/tmp/salt-hls-v22-target-graph-catalog-20261029.tar.gz`; SHA256
`bba4b870b568da18265567797eb9610762d357825f1aa6507d85b4b3f4cb36cd` (218 MB).
It is prepared locally, not published. `gh auth status` reports the configured
GitHub token is invalid. After re-authenticating, upload with:

```bash
gh release create archive/v22-graph-catalog-20261029 \
  /tmp/salt-hls-v22-target-graph-catalog-20261029.tar.gz \
  --repo ElenaVouvali/MailoHLS \
  --title "MailoHLS V2.2 target-aware graph catalog" \
  --notes "Authenticated V2.2 RC3 compiler graphs and target-enriched catalog. Verify the included graphs/latest/SHA256SUMS. Independent RC3 semantic adjudication remains pending."
```

## Preservation tags

Annotated local tags were created at original heads:

- `archive/v22-semantic-finalization-20261029`
- `archive/target-aware-mlir-v22-20261029`
- `archive/target-graph-readiness-20261029`
- `archive/stage1-directive-learning-repair-20261029`
- `archive/v22-graph-catalog-20261029`

Remote publication and dereferenced SHA verification succeeded. No research
branch has been deleted. `llvm-ir` and `mlir-target` have not been modified.

## Validation and remaining work

On this consolidation branch, graph, target overlay, tensor-readiness,
tokenizer and Stage-1 recovery suites ran: **183 passed, 32 skipped**. Eleven
skips were missing authenticated historical/external graph fixtures; twenty-one
were missing optional `torch_geometric`. Directive-domain suites passed
**29 tests**. Python compilation and catalog CLI-help checks passed. The source
branches previously recorded 756 catalog contexts validating with zero failures
and 176 graph/overlay tests passing. No full graph regeneration, model training
or synthesis was run during consolidation.

The branch-content comparison is complete for the five listed research
branches. Open PR dependencies and a finer exact-duplicate audit across paper
figures/root-level outputs remain; no ambiguous files were removed or moved.
The root utilities `fix_mailohls_static_domain_teacher_eval.py` and
`import_gnosis_lud0.py` had no matches by basename in maintained docs/scripts/
experiments/plots callers, and remain in place. `ASPLOS_experiments.jsonl`,
`LLM_target_conditioning.csv` and `hls_results_all.csv` remain at their existing
paths because no safe canonical replacement was established. `FINAL_ARTIFACTS.json`
and dataset/graph release files have unchanged hashes. Branch cleanup remains
deferred until PR merge and external catalog publication.

The original dirty-worktree snapshot remains byte-identical to the captured
status and patch hashes. The clean maintenance branch
`maintenance/repository-consolidation-clean-20261029` is published at
`8f14ade0f801dfa96002566864cb9639f9108b54` (before this final documentation
update). GitHub PR creation was rejected by
the connected integration with HTTP 403. `main` remains at
`d488681f68e58ab1b1e666f6911f5257d6462dc2`; no merge or branch retirement has
occurred. Branch retirement is deferred until the remaining audit and artifact
publication work is complete.
