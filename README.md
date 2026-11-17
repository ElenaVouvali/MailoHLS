# MailoHLS — original LLVM IR version

This branch preserves the original MailoHLS implementation at the
`MailoHLS_paper_v1` snapshot (`c9b2f43`, 17 July 2026). Its active graph pipeline
uses LLVM IR, ProGraML graphs, and GNN embeddings for the LLM pragma generator.
Experimental MLIR source present in that historical snapshot is also preserved.

The optional [CG-BDR refinement mechanism](refinement/README.md) adds bounded,
compiler-grounded HLS feedback after an initial directive prediction. It is the
exact controller tested on six kernels with Vitis HLS 2021.1: at most four HLS
evaluations per kernel, including the initial configuration. Its measured
campaign, native report audit, usage and offline tests are included separately
from the historical model and graph pipeline. It does not retrain Stage-3 DPO.

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
- `refinement/`: tested post-inference CG-BDR controller, optional LLVM graph
  preparation helper, offline tests and authenticated six-kernel evidence.

This is an archival implementation. Some original scripts contain machine-specific
paths; configure those paths and provide the original data/checkpoints before
running them. Generated checkpoints, weights, logs, and caches are not committed.
The historical implementation remains unchanged. The optional refinement module
is a later addition; its measured scope and limitations are documented explicitly.
