# MailoHLS

MailoHLS learns to generate HLS pragma configurations from kernel structure,
optimization objectives, target FPGA, clock period, and available resources.
The current implementation uses MLIR semantic graphs and GNN structural memory
to condition an LLM. The paper also refers to this system as SALT-HLS.

## Versions and ongoing development

| Branch | Implementation |
| --- | --- |
| [`main`](https://github.com/ElenaVouvali/MailoHLS/tree/main) | Active development of further framework extensions, starting from the MLIR + target-conditioning implementation. |
| [`mlir-target`](https://github.com/ElenaVouvali/MailoHLS/tree/mlir-target) | Preserved organized MLIR + target-conditioning baseline: multiscale GNN memory, target-aware SFT, structural cross-attention, DPO, clock selection, and device adaptation. |
| [`llvm-ir`](https://github.com/ElenaVouvali/MailoHLS/tree/llvm-ir) | Original LLVM IR / ProGraML / GNN implementation, based on the `MailoHLS_paper_v1` snapshot (`c9b2f43`). |

Use `main` for current development. The other two branches preserve the baseline
implementations. Keep local source changes safe before switching branches;
one checkout is sufficient.

## Implementation map

| Path | Purpose |
| --- | --- |
| `Data/` | Kernel sources, action mappings, and original HLS measurements. |
| `evaluation_set/` | External evaluation kernels, directive placeholders, and action metadata. |
| `paper_plots/` | Consolidated measured results used by paper plotting and analysis. |
| `Preprocessing/` | Target-aware preprocessing, SFT examples, and family splits. |
| `GNN_branch/` | MLIR graph construction, GNN training, and memory export. |
| `LLM_branch/common/` | Prompt contracts, structural attention, and device-adaptation checkpoints. |
| `LLM_branch/train/` | SFT and DPO training. |
| `LLM_branch/inference/`, `LLM_branch/clock_adapt/` | Generation, evaluation, and automatic clock selection. |
| `experiments/refinement/` | Optional compiler-verified local directive experiments, synthesis, measurement reuse and attribution. |
| `experiments/refinement_search/` | Optional budgeted feedback controller comparing coordinate, composition, structural, hybrid and random search. |
| `experiments/structural_audit/` | Read-only source, graph, action-memory and model-interface diagnostics. |
| `scripts/reproduction/`, `scripts/training/`, `scripts/evaluation/` | Maintained preparation, training, and evaluation launchers. |
| `scripts/experiments/`, `scripts/legacy/` | Distinct experiment presets and archived LLVM-era launchers. |
| `FINAL_ARTIFACTS.json` | Selected checkpoint and memory paths, including pending objectives. |
| `HARP-rep/` | Consolidated HARP baseline source, license, measurements, and local reproduction changes. |

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

## Optional local refinement

The standard framework predicts a complete directive configuration in one pass.
The optional local experiment tests small legal changes using stored predictions
and measured HLS feedback:

```bash
python -m experiments.refinement.experiment --mode prepare
python -m experiments.refinement.experiment --mode run --vitis-hls /path/to/vitis_hls
python -m experiments.refinement.experiment --mode summarize
```

Preparation does not launch synthesis. Read
[`experiments/refinement/README.md`](experiments/refinement/README.md) for compiler
prerequisites, targets, measurement reuse, outputs and scientific limitations.
Generated projects and reports remain outside Git; source/measurement hashes and
explicit synthesis budgets preserve reproducibility.

For sequential decisions driven by previous synthesis results, use the
[`budgeted refinement controller`](experiments/refinement_search/README.md).
It reuses the same configuration, compiler-evidence and HLS interfaces, adds five
deterministic search policies, and separates logical evaluation budgets from
physical report reuse. Neither experiment runs the LLM or GNN during refinement.
The standard one-pass training and inference entry points remain unchanged.

To inspect compiler relationships and their path into GNN features, action memory
and attention routing without synthesis or model inference, use the
[`structural representation audit`](experiments/structural_audit/README.md).

An explicitly selected [MLIR structural graph V2](docs/structural-graph-v2.md)
adds precise call bindings, unified memory relations and a lossless compact view.
Its explicit `--revision 2.1` extends return, indexing, loop-state and coverage evidence.
It is a research representation alongside the production graph; existing GNN
weights, structural memory and inference continue to use V1.

Directive domains are maintained in one shared module,
`LLM_branch/common/directive_domains.py`. Existing model entry points retain
their checkpoint-compatible policy. The explicit `tcad` policy adds bounded
powers of two and divisors, with dimension-aware array validation. To inspect
the opt-in model workflow without loading a model or starting training:

```bash
python -m LLM_branch.common.directive_domain_workflow --workflow train --domain-policy tcad
```

Execution requires `--execute` and a policy record. Evaluating an existing
checkpoint with expanded domains additionally requires `--allow-domain-shift`;
its trained policy is never silently relabeled. Future training requires an
audited compatible dataset. Existing checkpoints and `FINAL_ARTIFACTS.json`
remain unchanged.

For measured objective-response figures, use `plots/objective_response/objective_response.py`.
