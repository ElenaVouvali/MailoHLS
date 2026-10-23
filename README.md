# MailoHLS

MailoHLS learns to generate HLS pragma configurations from kernel structure,
optimization objectives, target FPGA, clock period, and available resources.
The current implementation uses MLIR semantic graphs and GNN structural memory
to condition an LLM. The paper also refers to this system as SALT-HLS.

## Versions and ongoing development

| Branch | Implementation |
| --- | --- |
| [`main`](https://github.com/ElenaVouvali/MailoHLS/tree/main) | Active development of further framework extensions, starting from the MLIR + target-conditioning implementation. |
| [`mlir-target`](https://github.com/ElenaVouvali/MailoHLS/tree/mlir-target) | Preserved MLIR + target-conditioning baseline at `26db8ec`: multiscale GNN memory, target-aware SFT, structural cross-attention, DPO, clock selection, and device adaptation. |
| [`llvm-ir`](https://github.com/ElenaVouvali/MailoHLS/tree/llvm-ir) | Original LLVM IR / ProGraML / GNN implementation, based on the `MailoHLS_paper_v1` snapshot (`c9b2f43`). |

Use one checkout and switch with `git switch main`, `git switch mlir-target`,
or `git switch llvm-ir`. Develop extensions on `main` and keep `mlir-target`
as the preserved baseline. Commit or stash source changes before switching. The
previous `stage2-analysis-refactor` branch is incorporated into `main`; a
separate `MailoHLS-next` worktree is no longer required.

## Implementation map

| Path | Purpose |
| --- | --- |
| `Data/` | Kernel sources, action mappings, and original HLS measurements. |
| `Preprocessing/` | Target-aware preprocessing, SFT examples, and family splits. |
| `GNN_branch/` | MLIR graph construction, GNN training, and memory export. |
| `LLM_branch/common/` | Prompt contracts, structural attention, and device-adaptation checkpoints. |
| `LLM_branch/train/` | SFT and DPO training. |
| `LLM_branch/inference/`, `LLM_branch/clock_adapt/` | Generation, evaluation, and automatic clock selection. |
| `scripts/reproduction/`, `scripts/training/`, `scripts/evaluation/` | Maintained preparation, training, and evaluation launchers. |
| `scripts/experiments/`, `scripts/legacy/` | Distinct experiment presets and archived LLVM-era launchers. |
| `FINAL_ARTIFACTS.json` | Selected checkpoint and memory paths, including pending objectives. |
| `HARP-rep/`, `HARP-paper-rep/` | Vendored baseline source, licenses, measurements, and local reproduction changes. |

Historical source and checkpoint provenance code are retained. Generated model
weights, tensor caches, intermediate checkpoints, tokenizer copies, training
logs, and temporary files are excluded from Git.

## Setup and reproduction

Install the dependencies from `requirements.txt` in an isolated environment.
MLIR graph generation additionally requires the pinned Polygeist toolchain;
follow [`toolchain/README.md`](toolchain/README.md). The detailed pipeline guide
is [`docs/mlir-pipeline.md`](docs/mlir-pipeline.md).

The shared experiment preparation entry point is:

```bash
bash scripts/reproduction/prepare_final_experiment.sh
```

Inspect the paths and parameters in `scripts/training/train_gnn.sh`,
`scripts/training/train_stage1.sh`, and `scripts/training/train_stage2.sh`
before launching training. Model weights and processed tensors must be generated
or supplied separately; cloning this repository does not download them.

See [`docs/shell-scripts.md`](docs/shell-scripts.md) for launcher choices and
previous-to-current paths, [`docs/repository-layout.md`](docs/repository-layout.md) for the cleanup
policy and [`docs/paper-draft.md`](docs/paper-draft.md) for the manuscript notes.

Duplicate artifact mappings and local-cache compatibility are documented in
[`docs/duplicate-cleanup.md`](docs/duplicate-cleanup.md).
