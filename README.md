# MailoHLS — original LLVM IR version

This branch preserves the original MailoHLS implementation at the
`MailoHLS_paper_v1` snapshot (`c9b2f43`, 17 July 2026). Its active graph pipeline
uses LLVM IR, ProGraML graphs, and GNN embeddings for the LLM pragma generator.
Experimental MLIR source present in that historical snapshot is also preserved.

The current optional [CG-BDR v3 refinement mechanism](refinement/v3/README.md)
adds bounded, constraint-grounded HLS feedback after an initial directive
prediction. It preserves scheduling variants, attributes diagnostics to source
actions and LLVM anchors, proposes small repairs, and retains the best measured
feasible incumbent. Its [first campaign](refinement/v3/first_campaign/README.md)
contains six verified baseline reuses and one new Jacobi synthesis: the candidate
regressed and the baseline was retained. No V3 latency gain is claimed.
The [v2 prototype](refinement/v2/README.md) and
[earlier six-kernel version](refinement/README.md) retain their separate evidence
and metric contracts. Refinement uses at most four evaluations per kernel,
including baseline, and does not retrain Stage-3 DPO or change the original
LLVM/model pipeline.

| Branch | Version |
| --- | --- |
| [`llvm-ir`](https://github.com/ElenaVouvali/MailoHLS/tree/llvm-ir) | Original LLVM IR / ProGraML implementation (this branch). |
| [`main`](https://github.com/ElenaVouvali/MailoHLS/tree/main) | Current MLIR graphs, structural memory, target conditioning, and subsequent training/evaluation work. |

Use `git switch main` or `git switch llvm-ir` in one checkout. Commit or stash
source changes first.

## Original layout

- `GNN_branch/`: LLVM-to-graph conversion, GNN model, and embedding extraction.
- `LLM_branch/train/main_GNN.py`, `train_GNN.py`: original GNN training entry points;
  the data backend is `GNN_branch/data.py`.
- `LLM_branch/train/`: SFT/DPO and structure-conditioned LLM training.
- `LLM_branch/inference/`: inference and evaluation scripts.
- `preprocessing/`, `notebooks/`: original data preparation and notebooks.
- `requirements.txt`: historical environment dependencies.
- `refinement/v3/`: current refinement mechanism, deterministic ranker, semantic
  attribution, Coroni workflow, offline tests, checkpoint replay and first-run evidence.
- `refinement/v2/`: earlier post-inference refinement prototype, semantic facts,
  scheduling-aware repair contracts, offline tests and five-kernel evidence.
- `refinement/`: earlier controller, LLVM graph preparation helper, offline
  tests and authenticated six-kernel evidence, preserved for reproducibility.

This is an archival implementation. Some original scripts contain machine-specific
paths; configure those paths and provide the original data/checkpoints before
running them. Generated checkpoints, weights, logs, and caches are not committed.
The historical implementation remains unchanged. The optional refinement module
is a later addition; its measured scope and limitations are documented explicitly.
