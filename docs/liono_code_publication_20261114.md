# Liono code publication — 2026-11-14

This branch collects the new source packages on top of the evaluated FiLM/GMT
Stage-2 branch. It preserves the existing research branches and frozen checkpoint
contracts. Publication does not launch training, inference, synthesis or deployment.

| Area | Published code |
|---|---|
| Nine-kernel inference and Coroni synthesis | `experiments/stage2_eval_compat9/` |
| Kalman/Rosetta compiler repair | `experiments/stage2_eval_compat9/compiler_repair/` |
| Routing ablation trainer/inference | `experiments/routing_ablation_20261114/` |
| Routing experiment setup and comparison | `routing_ablation_package/` |
| Compact AES/JPEG prompt views and synthesis tooling | `coroni_eval9_toolkit/` |
| Checkpoint-300 budgeted refinement | `final300_refinement/` |
| U50 few-shot experiment | `u50_final300_fewshot/` |
| Saved compact-view prediction JSONL | `inference/` |
| Davinci deployment adapter and preparation | `experiments/film_gmt_stage2_final/davinci_deployment/` |
| Coroni ChatGPT guide and tested report parser | `docs/coroni_chatgpt_synthesis_guide.md`, `experiments/coroni_synthesis_guide/` |

The source files were copied byte-for-byte from their existing Liono locations.
[`provenance/liono_code_publication_20261114.json`](provenance/liono_code_publication_20261114.json)
records original paths, repository destinations and SHA-256 identities. This is a
new publication inventory; no old authentication digest was rewritten to accept
changed artifacts. Historical source snapshots and receipts retain their original
absolute execution paths and their version-specific requirements.

The separate experiment branches also preserve their latest local source:

* `research/stage2-evaluated-20261113`: routing-ablation trainer and inference.
* `research/gnosis-stage1-film-ab-20261112`: GAN V4 audit, synthesis preparation,
  and plotting changes.
* `research/historical-mlir-gmt-gemv`: direct full-run launcher, validation progress
  logging and explicit skip-validation control.
* `research/gemv-structural-explainability`: final-artifact refinement and integration.
* `research/gnosis-17-kernel-recovery-20261113`: authenticated recovery history.
* `archive/liono-final-architecture-20261114`: preserves the previously unpublished
  final-architecture representation diagnostic commit without replacing the newer
  remote `final-architecture` branch.

The [compiler repair report](kalman_rosetta_compat9_v2.md) and
[deployment report](film_gmt_stage2_davinci_deployment.md) are historical receipts
for the qualified runs described in them. Compiler repair remains an explicitly
versioned representation shift. The deployed execution code/checkpoint contract
is pinned to `6ee1390`; this publication commit does not change that identity.
The scripts' expected execution layout and source/artifact identities remain
mandatory when reproducing their experiments.

Model checkpoints, optimizer state, generated GEXFs/memory tensors, full deployment
mirrors, native compiler binaries/libraries, raw logs, caches and editor backups
remain local. Small compiler-pass source snapshots, hash receipts and parser XML
fixtures are included. The malformed uncommitted edit to
`paper_plots/SoTA_experiments.jsonl` (two JSON objects joined on one line) is left
untouched and is not part of this publication. Research results are not edited.

Validation used offline tests and source syntax/hash checks. Coroni toolkit tests,
refinement tests, U50 offline tests and the historical-environment GEMV tests pass.
GEMV's trained-artifact tests require the authenticated local artifacts; the
ordinary LLM environment lacks `torch_geometric`, so those tests were executed in
the existing historical GNN environment. Package README files distinguish offline
checks from physical synthesis/model experiments, which this publication did not run.

The original worktrees, deployment mirrors and scratch shell-output files remain
available locally. Gitlink/submodule-pointer changes in the root checkout are not
staged; research sources are published on their actual branches.
