MailoHLS

MailoHLS learns HLS quality of results (QoR) from a kernel's compiler structureand its pragma configuration. The current GNN path converts each C/C++ kernelinto one deterministic MLIR semantic graph, reuses that graph across thekernel's design points, and predicts latency and area. A later LLM stage cancombine these representations with device- and frequency-aware objectives.

Pipeline status

Stage

Purpose

Status

QoR preprocessing

Build one reference-target supervision table for the GNN

Implemented

MLIR graph generation

Capture control, SSA, loops, calls, memory, dependences, and HLS actions

Implemented

MLIR tensor dataset

Convert GEXF graphs and pragma/QoR points to compact PyG tensors

Implemented

GNN training

Learn kernel- and pragma-aware QoR representations

Experimental

LLM integration

Add target-aware reasoning and optimization

Next stage

Repository layout

Data/
├── ApplicationDataset/          # source, headers, and kernel_info.txt
├── ApplicationAPLMapping/       # CSV directive column -> action mapping
├── ApplicationInformation.csv   # graph-generation metadata
└── CSVS/                        # raw multi-target synthesis measurements

Preprocessing/
└── data_preprocess.py           # GNN and LLM preprocessing modes

GNN_branch/
├── mlir_graph_gen.py            # one C/C++ kernel -> one semantic GEXF
├── generate_mlir_dataset.py     # validated 55-kernel graph driver
├── MLIR_graphs/                 # generated graphs and manifest
├── mlir_data.py                 # GEXF + pragma/QoR -> compact PyG dataset
├── model.py                     # edge-aware TransformerConv model
├── train_GNN.py                 # training, selection, and physical metrics
├── main_GNN.py                  # training/inference entry point
└── summarize_seed_runs.py       # mean ± standard deviation across seeds

1. Prepare single-target GNN labels

Raw CSVs remain the multi-device source of truth. GNN preprocessing selects onereference target, canonicalizes equivalent directives, and aggregates repeatedmeasurements of the same effective pragma configuration:

device: xczu7ev-ffvc1156-2-e

clock period: 10.0 ns (100 MHz)

python Preprocessing/data_preprocess.py \
  --mode gnn \
  --device xczu7ev-ffvc1156-2-e \
  --clock-period-ns 10.0 \
  --force

The command writes 55 tables and a manifest underGNN_branch/Data/preprocessed_CSVS/. Do not replace Data/CSVS/.

2. Generate MLIR semantic graphs

Use cgeist, the MLIR Python bindings, and the MailoHLS compiler-analysisextension from the same Polygeist build:

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

Generate all configured graphs:

PYTHONHASHSEED=0 PYTHONPATH="$MLIR_PYTHON_ROOT" \
"$MLIR_PYTHON" GNN_branch/generate_mlir_dataset.py \
  --force --keep-mlir

Generation is valid only whenGNN_branch/MLIR_graphs/generation_manifest.csv reports success for all 55kernels. Action labels from kernel_info.txt must map exactly once to sourceand MLIR; the generator does not use nearest-location or loop-order guesses.

Each GEXF is a directed multigraph. Parallel edges are intentional.

Element

Semantics

Operation/value nodes

MLIR operations, SSA values, block arguments, constants, and types

Structural edges

Regions, blocks, control order, loop nesting, and calls

SSA edges

Def-use, operands/results, and loop-carried values

Memory edges

Allocations, views, exact aliases, uncertain alias pairs, effects, and accesses

Dependence edges

Proven affine RAW/WAR/WAW or explicitly marked conservative uncertainty

Action nodes

PIPELINE, UNROLL, and ARRAY_PARTITION scopes grounded in source and MLIR

3. Build tensors and train

--force_regen validates graph/preprocessing manifests, fits categoricalencoders on training kernels only, and rebuildsGNN_branch/MLIR_dataset/all_kernels/.

Pragma magnitudes use compact model-facing values:

partition kind: none/cyclic/block/complete -> 0/1/2/3;

unroll and partition factor: log2(1 + factor);

pipeline II and partition dimension: raw values.

Use kernel-disjoint splits. This example reserves Kalman for final testing andthree unrelated kernels for validation:

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONHASHSEED=0 python GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --force_regen \
  --epoch_num 200 \
  --random_seed 123 \
  --experiment_name asplos_gnn_seed_123 \
  --val_kernels machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0 \
  --test_kernels serrano-kalman-filter

Training and checkpoint selection minimizeMSE(log2(latency)) + MSE(log2(area)). Evaluation is reported after inversetransformation in physical units (latency_ms and area_score), both over allpoints and as a per-kernel macro average. Kendall tau-b reports ranking quality.There is no combined latency/area RMSE because those targets have differentunits. The held-out test set is evaluated once from the best validationcheckpoint.

4. Multi-seed final experiment

Freeze the graph version, preprocessing, split, and hyperparameters beforeopening the test set. Build the tensor cache once with seed 123, then run atleast five independent initializations without --force_regen:

VAL=machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0
TEST=serrano-kalman-filter

for SEED in 456 789 2025 4096; do
  PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python GNN_branch/main_GNN.py \
    --dataset mlir \
    --subtask train \
    --epoch_num 200 \
    --random_seed "$SEED" \
    --experiment_name "asplos_gnn_seed_${SEED}" \
    --val_kernels "$VAL" \
    --test_kernels "$TEST"
done

Summarize mean ± sample standard deviation:

python GNN_branch/summarize_seed_runs.py \
  --runs-glob 'logs/asplos_gnn_seed_*/run1/test_physical_metrics.csv' \
  --output logs/asplos_gnn_seed_summary.csv

For a publication, repeat the protocol with several predefinedkernel-disjoint folds; multiple seeds measure initialization variance but donot replace evaluation on multiple unseen kernel families.

What the learned representation contains

The GNN first embeds MLIR nodes and relations, propagates information withedge-aware TransformerConv layers, injects pragma values only at their mappedaction scopes, performs one post-pragma message-passing step, and pools programand structural-scope channels. Its prediction embedding is thereforekernel- and pragma-conditioned, not a pure kernel-only vector.

For the later LLM stage, distinguish two products:

a structural kernel embedding captured before pragma injection; and

a design-point embedding captured after pragma injection.

Do not describe forward_embed() as pragma-free unless extraction explicitlyuses the pre-pragma state or a validated all-zero pragma configuration.

Reproducibility checklist

Pin the MailoHLS and Polygeist commits and record compiler/binding hashes.

Keep PYTHONHASHSEED, split lists, preprocessing manifest, and graph manifest.

Use strict deterministic algorithms; --allow_nondeterministic is debuggingonly and must not be used for reported results.

Regenerate tensors after graph, preprocessing, or feature-schema changes.

Never use the Kalman test result for hyperparameter or checkpoint selection.

Report physical-unit point metrics, per-kernel macro metrics, Kendall tau-b,and mean ± standard deviation across seeds.