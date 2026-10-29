# MailoHLS repository consolidation

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

The full graph catalog remains accessible on the preserved
`graphs/latest-v22-catalog` branch. No GitHub Release asset was created, so the
branch must not be deleted. The catalog builder is included in this
consolidation worktree; its copy helper no longer reads generated catalog
review files. It passed syntax and CLI-help checks, but a full expensive rebuild
was deliberately not run. The published catalog branch remains the complete
catalog copy.

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

The branch audit is not yet complete: open PR dependencies, full branch-only
file inventories, root utility/result canonical locations, duplicate figure
checks, checkpoint/data hash baselines, release publication and branch cleanup
remain. Do not retire branches until each selected artifact is on main or in a
verified durable release and preservation tags are verified remotely.

The original dirty-worktree snapshot remains byte-identical to the captured
status and patch hashes. The clean maintenance branch
`maintenance/repository-consolidation-clean-20261029` is published at
`a082b614a64a065220ba3963e0ad2065bdc5d774`. GitHub PR creation was rejected by
the connected integration with HTTP 403. `main` remains at
`d488681f68e58ab1b1e666f6911f5257d6462dc2`; no merge or branch retirement has
occurred. Branch retirement is deferred until the remaining audit and artifact
publication work is complete.
