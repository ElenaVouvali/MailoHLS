MailoHLS

MailoHLS is a structure-aware framework for high-level synthesis (HLS) design-space optimization. It combines:

a compiler/GNN path that represents each C/C++ kernel as a deterministic MLIR semantic graph and learns latency/area behavior; and

an LLM path that predicts pragma assignments for an optimization objective, target device, clock period, and resource budget.

The repository currently contains a validated 55-kernel MLIR graph dataset and an experimental GNN training pipeline. The unified target-aware SFT trainer is implemented, but the current GNN-to-LLM memory exporter and SFT JSONL builder still need to be aligned with the new MLIR schemas before the complete LLM pipeline is reproducible.

Pipeline status

Stage

Purpose

Status

Raw measurements

Multi-device, multi-frequency synthesis results

Available for 55 kernels

QoR preprocessing

Build GNN- or LLM-specific supervision tables

Implemented

MLIR graph generation

Capture compiler structure, data flow, memory, dependences, and HLS actions

Validated for 55/55 kernels

MLIR tensor dataset

Convert GEXF graphs and design points to compact PyG samples

Implemented

GNN training

Predict latency/area and learn structure-aware representations

Experimental; v11 training in progress

MLIR structural-memory export

Convert the selected GNN checkpoint into action-aligned LLM memory

Pending schema-aligned exporter

SFT JSONL construction

Build target-aware prompts and complete pragma targets

Pending schema-aligned builder

Target-aware SFT

Two-stage LoRA plus cross-attention over GNN memory

Trainer implemented; end-to-end validation pending

Inference/HLS validation

Generate legal directives and evaluate by synthesis

Pending alignment with the unified trainer

flowchart TD
    A["C/C++ kernels and synthesis CSVs"] --> B["QoR preprocessing"]
    A --> C["cgeist and MLIR analysis"]
    C --> D["Semantic GEXF graphs"]
    B --> E["PyG design-point dataset"]
    D --> E
    E --> F["GNN QoR model"]
    F --> G["Action-aligned structural memory"]
    B --> H["Target-aware SFT JSONL"]
    G --> I["LLM cross-attention SFT"]
    H --> I
    I --> J["Legal pragma plan and HLS validation"]

Repository layout

Data/
├── ApplicationDataset/          # sources, headers, and kernel_info.txt
├── ApplicationAPLMapping/       # CSV directive column -> action ID
├── ApplicationInformation.csv   # graph-generation metadata
└── CSVS/                        # raw multi-target synthesis measurements

Preprocessing/
├── data_preprocess.py           # current GNN/LLM QoR preprocessing
└── create_jsonl.py              # legacy SFT builder; not current production path

GNN_branch/
├── mlir_graph_gen.py            # one kernel -> one semantic GEXF
├── generate_mlir_dataset.py     # validated 55-kernel graph driver
├── MLIR_graphs/                 # graphs, audit MLIR, logs, and manifest
├── mlir_data.py                 # GEXF + pragma/QoR -> compact PyG dataset
├── model.py                     # edge-aware TransformerConv model
├── train_GNN.py                 # training, selection, and physical metrics
├── main_GNN.py                  # GNN entry point
└── summarize_seed_runs.py       # aggregate final multi-seed results

LLM_branch/
├── train/train_SFT_xattn_new.py # unified target-aware SFT trainer
├── train/                        # older SFT/DPO experiments
└── inference/                    # older inference experiments; alignment pending

Checkpoints/                      # small diagnostic logs/configuration snapshots

Files with absolute /home/... defaults or HARP-specific tensor names are retained as research history unless this README explicitly selects them as the current path.

1. Install the Python environment

Create a virtual environment and install the repository requirements:

python3 -m venv .hls-llm
source .hls-llm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

MLIR graph generation additionally requires the audited Polygeist build, its Python 3.11 bindings, and the MailoHLS MLIR-analysis extension. These are not provided by the unrelated PyPI package named mlir.

2. Preprocess QoR measurements

Data/CSVS/ is the immutable multi-device source of truth.

GNN supervision

The GNN uses one reference target so an identical kernel/pragma configuration cannot have conflicting labels from different devices or frequencies:

python Preprocessing/data_preprocess.py \
  --mode gnn \
  --device xczu7ev-ffvc1156-2-e \
  --clock-period-ns 10.0 \
  --force

This writes 55 tables and preprocessing_manifest.json under GNN_branch/Data/preprocessed_CSVS/. Repeated measurements of the same effective pragma configuration are aggregated.

LLM supervision

The LLM retains all measured devices and clock periods, and computes target-local candidate weights:

python Preprocessing/data_preprocess.py --mode llm --force

This writes to LLM_branch/Data/preprocessed_CSVS/. These CSVs are an intermediate artifact, not yet the JSONL consumed by the unified SFT trainer. See Remaining integration gates.

3. Generate MLIR semantic graphs

Point every component at the same Polygeist/MLIR build:

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
print("MLIR bindings: OK")
print("MailoHLS analysis:", _mailohls_analysis.__file__)
PY

Generate or validate all configured graphs:

PYTHONHASHSEED=0 PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
  --force --keep-mlir

A run is complete only when GNN_branch/MLIR_graphs/generation_manifest.csv accounts for all 55 kernels and every output is present and valid. A repeated run may report skipped for unchanged validated outputs.

Each GEXF is a directed multigraph; parallel edges are intentional.

Graph element

Meaning

Operation/value nodes

MLIR operations, SSA values, block arguments, constants, and types

Structural edges

Regions, blocks, control order, loop nesting, and calls

SSA edges

Def-use, operands/results, and loop-carried values

Memory edges

Allocations, views, exact aliases, pairwise uncertain aliases, effects, and accesses

Dependence edges

Proven affine RAW/WAR/WAW or explicitly marked conservative uncertainty

Action nodes

PIPELINE, UNROLL, and ARRAY_PARTITION scopes grounded in source and MLIR

Action IDs from kernel_info.txt must map exactly once through source locations into MLIR. The production generator does not use nearest-location, loop-order, or synthetic-loop fallbacks.

4. Build the tensor cache and train the GNN

--force_regen validates the graph and preprocessing manifests, fits categorical encoders on training kernels only, and rebuilds GNN_branch/MLIR_dataset/all_kernels/.

Model-facing pragma values use compact encodings:

partition type: none/cyclic/block/complete -> 0/1/2/3;

unroll and partition factors: log2(1 + factor);

pipeline II and partition dimension: raw numeric values.

Use kernel-disjoint splits. The current reference experiment reserves Kalman for final testing:

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONHASHSEED=0 python GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --force_regen \
  --epoch_num 200 \
  --random_seed 123 \
  --experiment_name gnn_v11_full_seed123 \
  --val_kernels machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0 \
  --test_kernels serrano-kalman-filter

Do not add --force_regen to subsequent seeds unless the graph or feature schema changed.

Checkpoint selection uses the worst normalized validation target: each target loss is divided by the deterministic training-mean baseline, then the maximum ratio is minimized. A score below 1.0 means both latency and area beat their corresponding constant baseline. The displayed summed validation loss is diagnostic and is not expected to decrease monotonically.

Early stopping restores the best validation-selected checkpoint and opens the held-out test split exactly once. Do not interrupt a healthy run merely because the latest validation loss increased; let the configured patience stop it. If an interrupted run must be continued, use the same command and configuration with --resume_training.

Before a final reported run, add an initialized-model sanity gate: the selected trained checkpoint must beat both the constant predictor and the seeded untrained model on every validation target. If it does not, stop before opening the held-out test set and revise the training configuration using validation data only.

Evaluation is reported after inverse transformation in physical units (latency_ms and area_score), both across all design points and as per-kernel macro averages. Kendall tau-b measures ranking quality. Latency and area RMSE are not combined because they have different units.

Publication protocol

Freeze the graph version, preprocessing manifest, split, and hyperparameters before final testing. Run at least five independent initializations without regenerating the tensor cache:

VAL=machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0
TEST=serrano-kalman-filter

for SEED in 456 789 2025 4096; do
  PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python GNN_branch/main_GNN.py \
    --dataset mlir \
    --subtask train \
    --epoch_num 200 \
    --random_seed "$SEED" \
    --experiment_name "gnn_v11_full_seed_${SEED}" \
    --val_kernels "$VAL" \
    --test_kernels "$TEST"
done

Summarize the held-out metrics:

python GNN_branch/summarize_seed_runs.py \
  --runs-glob 'logs/gnn_v11_full_seed_*/run1/test_physical_metrics.csv' \
  --output logs/gnn_v11_seed_summary.csv

Multiple seeds estimate initialization variance; they do not replace several predefined kernel-family folds.

5. GNN representation used by the LLM

The GNN embeds MLIR nodes and relations, propagates information with edge-aware TransformerConv layers, injects pragma values at mapped action scopes, applies post-pragma message passing, and pools program and structural-scope channels.

Two representations must remain distinct:

structural memory: one zero-pragma, action-aligned memory pack per kernel for LLM conditioning; and

design-point embedding: the pragma-conditioned representation used by the QoR predictor.

forward_embed() is not automatically pragma-free. A structural-memory exporter must explicitly zero pragma tensors/masks, verify one slot per declared action, and record the exact checkpoint and schema hashes.

6. Unified target-aware SFT

LLM_branch/train/train_SFT_xattn_new.py is the current trainer. Its intended production mode is:

Stage 1: train one objective-conditioned LoRA adapter to emit deterministic directive RHS values.

Stage 2: initialize from the best Stage-1 adapter and train gated cross-attention over action-aligned MLIR structural memory.

The production objective is ALL, with known-device conditioning and one target candidate per prompt. Once the two integration gates below are complete, the intended command is:

python LLM_branch/train/train_SFT_xattn_new.py \
  --dataset artifacts/llm/mailohls_sft.jsonl \
  --memory_dir artifacts/llm/mlir_structural_memory \
  --objective ALL \
  --run_mode two_stage \
  --split_mode family \
  --top_k 1 \
  --device_mode known \
  --device_token_dropout 0 \
  --require_pragma_free_memory \
  --stage1_output_dir checkpoints/sft_stage1 \
  --stage2_output_dir checkpoints/sft_stage2 \
  --save_split_json artifacts/llm/family_split.json \
  --seed 123

Do not run this as a reported experiment until the JSONL and structural-memory manifests pass strict validation. resource_dropout_ablation is an ablation, not the production path.

New-device adaptation should use a small measured calibration set and a residual adapter. The current <DEV=UNKNOWN> path is not publication-ready because a newly added special token must have a deliberately trained embedding, not only a resized but frozen embedding matrix.

Remaining integration gates

Complete these before launching SFT:

Replace the legacy JSONL builder. It must read LLM_branch/Data/preprocessed_CSVS/, use repository-relative CLI paths, emit device, clock, latency, area, all four utilization fields, and a complete deterministic assignment for every source action. It must write a manifest with input hashes and split-independent row counts.

Replace the legacy HARP memory exporters. Export directly from the current MLIR PyG cache and the validation-selected v11 checkpoint. Verify feature dimensions, zero pragma injection, action-slot completeness, deterministic output, and checkpoint/graph/schema hashes.

Align inference with the unified trainer. Load the same tokenizer, Stage-1 adapter, Stage-2 cross-attention state, objective tokens, device policy, structural-memory schema, and constrained RHS decoder.

Make special-token training explicit. Use PEFT selective trainable-token support (or an equivalently saved delta) so only the added schema/device tokens change and their learned state is restored by Stage 2 and inference.

Add end-to-end tests. Cover preprocessing -> graph -> PyG -> memory -> SFT example -> inference, including missing/mismatched action IDs and schema hashes.

Evaluate by synthesis. Report syntactic validity, action coverage, resource feasibility, predicted and measured QoR, Pareto quality/hypervolume, and generalization by unseen kernel family, device, and clock.

Until gates 1–3 are complete, Preprocessing/create_jsonl.py, GNN_branch/build_harp_memory.py, GNN_branch/pragma_free_pts.py, GNN_branch/pt_to_gnn_emb.py, and the older inference scripts should be treated as legacy references rather than a reproducible production chain.

Reproducibility checklist

Pin MailoHLS, Polygeist, LLVM, Python, PyTorch, Transformers, and PEFT versions.

Record compiler, MLIR binding, graph-generator, graph-schema, feature-schema, preprocessing, and checkpoint hashes.

Keep PYTHONHASHSEED, CUDA determinism settings, split files, and manifests.

Use strict deterministic algorithms; --allow_nondeterministic is debugging only.

Regenerate tensors after any graph, preprocessing, or feature-schema change.

Never use the held-out test result for hyperparameter or checkpoint selection.

Report physical-unit point and per-kernel macro metrics, Kendall tau-b, and mean +/- sample standard deviation across seeds/folds.

PyTorch documents that exact reproducibility is not guaranteed across releases or platforms even when random sources are controlled; reported experiments should therefore pin the complete software/hardware environment.