# MailoHLS

MailoHLS is a research pipeline for **structure-aware HLS design-space
optimization**. It combines two complementary models:

- an edge-aware GNN learns kernel structure and predicts latency/area from a
  compiler-derived MLIR graph and a pragma configuration;
- an LLM generates a complete legal pragma assignment for a kernel, objective,
  target FPGA, clock period, and resource budget.

The intended final system uses action-aligned GNN representations as structural
memory for the LLM. The repository is a research prototype: every implemented
stage is identified below, and unfinished integration work is stated explicitly.

## Current status

| Stage | State | Main artifact |
|---|---|---|
| Raw HLS measurements | Available | `Data/CSVS/` |
| GNN/LLM QoR preprocessing | Implemented | `Preprocessing/data_preprocess.py` |
| MLIR semantic graphs | Validated for 55/55 kernels | `GNN_branch/MLIR_graphs/` |
| PyTorch Geometric dataset | Implemented | `GNN_branch/MLIR_dataset/` |
| GNN QoR training | Implemented; final checkpoint not published yet | `GNN_branch/train_GNN.py` |
| Target-aware SFT JSONL | Implemented | `Preprocessing/create_jsonl.py` |
| Directive-only SFT | Implemented | `LLM_branch/train/train_SFT_xattn_new.py` |
| GNN structural-memory export | Pending alignment with the current graph/checkpoint schemas | — |
| Cross-attention SFT and inference | Trainer implemented; end-to-end validation pending | — |

```mermaid
flowchart TD
    A["C/C++ kernels"] --> C["MLIR semantic graphs"]
    B["HLS measurements"] --> D["Target-specific QoR tables"]
    C --> E["Edge-aware GNN"]
    D --> E
    E --> F["Action-aligned structural memory"]
    D --> G["Target-aware SFT examples"]
    F --> H["LLM pragma generator"]
    G --> H
    H --> I["HLS validation and Pareto evaluation"]
```

## Repository map

```text
Data/
├── ApplicationDataset/          labeled C/C++ sources and kernel_info.txt
├── ApplicationAPLMapping/       synthesis-CSV column -> source action ID
├── ApplicationInformation.csv   top function and source-file metadata
└── CSVS/                        raw multi-device/multi-clock HLS measurements

Preprocessing/
├── data_preprocess.py           validated GNN/LLM QoR preprocessing
└── create_jsonl.py              deterministic target-aware SFT builder

GNN_branch/
├── mlir_graph_gen.py            one source kernel -> one semantic GEXF graph
├── generate_mlir_dataset.py     reproducible 55-kernel graph driver
├── mlir_data.py                 GEXF + QoR points -> PyG samples
├── model.py                     edge-aware TransformerConv architecture
├── train_GNN.py                 training, model selection, and evaluation
├── main_GNN.py                  command-line entry point
└── MLIR_graphs/                 published graphs, audit MLIR, logs, manifest

LLM_branch/train/
└── train_SFT_xattn_new.py       target-aware Stage-1/Stage-2 SFT trainer
```

Older HARP graph builders, SFT scripts, and inference scripts are retained as
research history. The files named above define the current path.

## 1. Environment

Create a Python environment:

```bash
python3 -m venv .hls-llm
source .hls-llm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` records the development environment. PyTorch/CUDA wheels may
need to be selected for the local GPU before installing the remaining packages.

MLIR graph generation additionally requires the project’s patched Polygeist
build, Python 3.11 MLIR bindings, and `_mailohls_analysis` extension. The PyPI
package named `mlir` is unrelated. Point all commands at one build:

```bash
export POLYGEIST_BUILD="$HOME/tools/Polygeist/build-mailohls-assertions"
export CGEIST="$POLYGEIST_BUILD/bin/cgeist"
export MLIR_PYTHON="$HOME/.mlir-python311/bin/python"
export MLIR_PYTHON_ROOT="$POLYGEIST_BUILD/tools/mlir/python_packages/mlir_core"
export PYTHONHASHSEED=0
export PYTHONPATH="$MLIR_PYTHON_ROOT"

"$CGEIST" --help | grep mailohls-action-manifest
"$MLIR_PYTHON" - <<'PY'
from mlir import ir
from mlir._mlir_libs import _mailohls_analysis
print("MLIR bindings:", ir.__file__)
print("MailoHLS analysis:", _mailohls_analysis.__file__)
PY
```

The compiler-side action preservation, Mem2Reg correction, and analysis binding
are currently maintained in the associated Polygeist working tree. A release
artifact must pin or publish that patch set; the Python repository alone cannot
reproduce graph generation.

## 2. Preprocess HLS measurements

`Data/CSVS/` is the immutable measurement source. The action metadata chain is:

```text
synthesis CSV column -> ApplicationAPLMapping -> kernel_info.txt -> source label
```

An active directive is accepted only when that chain is complete.

### GNN labels: one hardware target

The graph represents the kernel and pragma configuration; therefore the GNN
uses one device/clock so identical inputs do not receive conflicting QoR labels.

```bash
python Preprocessing/data_preprocess.py \
  --mode gnn \
  --device xczu7ev-ffvc1156-2-e \
  --clock-period-ns 10.0 \
  --force
```

### LLM labels: all measured targets

The target-aware LLM retains every device and clock and computes Pareto weights
within each hardware target:

```bash
python Preprocessing/data_preprocess.py --mode llm --force

python Preprocessing/create_jsonl.py --force
```

The second command writes:

- `artifacts/llm/mailohls_sft.jsonl`: complete directive assignments plus
  device, clock, QoR, utilization, and a compact source key;
- `artifacts/llm/mailohls_sft.sources.json`: each source template stored once;
- `artifacts/llm/mailohls_sft.manifest.json`: source/table/output hashes and
  per-kernel target counts.

The JSONL builder does not choose objectives or winning points. Selection occurs
inside each data split in the trainer, preventing held-out leakage.

## 3. Generate MLIR graphs

```bash
PYTHONHASHSEED=0 PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
  --force --keep-mlir
```

Accept a run only when `GNN_branch/MLIR_graphs/generation_manifest.csv` validates
all 55 configured kernels. The GEXF files are directed multigraphs; parallel
edges intentionally preserve distinct relations between the same nodes.

| Relation | Information represented |
|---|---|
| Structure/control | functions, regions, blocks, order, loops, and calls |
| SSA data flow | operands/results, block arguments, and loop-carried values |
| Memory | allocations, views, exact aliases, pairwise uncertain aliases, effects, and accesses |
| Dependences | proven affine RAW/WAR/WAW and explicitly marked conservative fallbacks |
| HLS actions | source-grounded pipeline, unroll, and array-partition scopes |

Every declared action must map exactly once by source location. The production
generator does not invent graph loops or use nearest-location/order fallbacks.

## 4. Build the PyG cache and train the GNN

The first run after any graph/feature change must include `--force_regen`:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONHASHSEED=0 python GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --force_regen \
  --epoch_num 200 \
  --random_seed 123 \
  --experiment_name gnn_selection_seed123 \
  --standardize_targets \
  --kernel_balanced_loss \
  --kernel_uniform_sampling \
  --scheduler plateau \
  --warmup_epochs 3 \
  --plateau_patience 4 \
  --early_stopping_patience 25 \
  --val_kernels machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0 \
  --test_kernels serrano-kalman-filter
```

This is a **model-selection run**: the held-out test remains locked because the
command omits `--evaluate_test`. Subsequent seeds reuse the cache and omit
`--force_regen`. Report at least five seeds and grouped kernel-family folds;
initialization seeds do not replace structural folds.

After hyperparameters and the epoch budget are frozen, refit on train+validation
and open the test exactly once:

```bash
PYTHONHASHSEED=0 python GNN_branch/main_GNN.py \
  --dataset mlir --subtask train \
  --random_seed 123 \
  --experiment_name gnn_final_refit_seed123 \
  --standardize_targets \
  --kernel_balanced_loss \
  --kernel_uniform_sampling \
  --scheduler cosine \
  --final_refit --final_refit_epochs <FROZEN_EPOCH_COUNT> \
  --evaluate_test \
  --val_kernels machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0 \
  --test_kernels serrano-kalman-filter
```

The GNN reports inverse-transformed physical latency/area metrics, per-kernel
macro averages, and Kendall tau-b. The selected model must beat both the
training-mean baseline and the seeded untrained model on every validation target
before test evaluation is permitted.

## 5. Train the target-aware LLM

### Stage 1: directive-only baseline

This stage is runnable after creating the JSONL and does not require GNN memory:

```bash
python LLM_branch/train/train_SFT_xattn_new.py \
  --dataset artifacts/llm/mailohls_sft.jsonl \
  --objective ALL \
  --run_mode single \
  --disable_structural_memory \
  --device_mode known \
  --device_token_dropout 0 \
  --top_k 1 \
  --auto_frequency_fraction 0 \
  --output_dir checkpoints/sft_stage1 \
  --save_split_json artifacts/llm/family_split.json \
  --seed 123
```

The default production task uses a specified clock. The prompt conditions on
device identity, exact clock, available resources, and optimization objective;
the target contains one deterministic value for every legal directive site.

### Optional automatic-clock task

The trainer can also ask the model to choose the best **measured** clock for the
same kernel/device/resource budget. It uses `<CLK=AUTO>`, lists the supported
clock candidates, and emits `selected_clock_period_ns` before the directives:

```bash
# Ablation only until held-out clock-selection accuracy/regret is reported.
--auto_frequency_fraction 0.15 --min_auto_clock_count 2
```

This is an additional conditioned task, not a replacement for specified-clock
examples. The default fraction is zero, so existing training is unchanged. Do
not let the model invent a continuous clock: it may select only a candidate
supported by the measured design space. Before adopting this mode, verify that
specified-clock QoR does not regress and report held-out clock-choice accuracy,
QoR regret, and synthesis feasibility.

### Stage 2: structural-memory conditioning

Stage 2 should begin only after the current GNN checkpoint can be exported as
one deterministic, zero-pragma, action-aligned memory pack per kernel. The
export must record graph/feature/checkpoint hashes and reject missing action
slots. That exporter is the principal remaining integration gate; this README
does not present the two-stage command as reproducible until it exists.

## Reproducibility and publication checklist

- Pin MailoHLS, the patched Polygeist/LLVM revisions, Python, MLIR bindings,
  PyTorch, Transformers, PEFT, CUDA, and the FPGA toolchain.
- Preserve preprocessing, graph, feature, split, and checkpoint manifests.
- Use kernel-family-disjoint train/validation/test partitions.
- Fit encoders and target statistics on training kernels only.
- Never tune on the held-out test kernel or synthesis results.
- Report mean ± sample standard deviation across seeds and family folds.
- Evaluate directive legality/action coverage, resource feasibility, physical
  QoR, ranking, Pareto hypervolume/regret, and synthesis success.
- Treat automatic-clock selection and new-device adaptation as separate
  ablations until each has a dedicated held-out evaluation.

## Key references

- [MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/)
- [MLIR affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [MLIR side-effect interfaces](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)
- [PyTorch reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [T5 unified text-to-text formulation](https://www.jmlr.org/papers/v21/20-074.html)