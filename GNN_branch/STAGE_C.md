# Stage C: measured-reference delta training

Stage B established that the model can fit pragma responses, but its learned
kernel center does not calibrate unseen kernels. Stage C removes that
underdetermined branch. For each kernel it synthesizes one directive-neutral
point with the same Vitis/device/clock as the labels, then learns

`delta = log2(design QoR) - log2(neutral-reference QoR)`.

The final prediction is `measured neutral reference + predicted delta` in log2
space. This is a one-reference calibrated predictor, not a strict zero-shot
absolute-QoR predictor.

## 1. Reproduce the Stage B diagnosis without touching test

```bash
cd ~/MailoHLS
python GNN_branch/analyze_validation_offsets.py \
  Checkpoints/gnn_v14_stage_b_full_s123/run1/best_val_predictions.csv \
  --output-dir Checkpoints/gnn_v14_stage_b_full_s123/validation_offset_audit
```

The output is diagnostic only. `oracle_kernel_offset_log2` is computed from
validation labels and must never become a model input.

## 2. Smoke-test neutral synthesis and Stage C overfit

Activate Vitis HLS 2021.1 first, then synthesize only the tiny-overfit kernel:

```bash
python GNN_branch/generate_neutral_baselines.py \
  --kernels machsuite-gemm-blocked \
  --output GNN_branch/baselines/neutral_smoke.csv \
  --work-dir neutral_baseline_build_smoke
```

The script rejects source files containing searched PIPELINE, UNROLL, or
ARRAY_PARTITION pragmas, records source/tool/device/clock provenance, and uses
the same zero-utilization floor as `Preprocessing/data_preprocess.py`.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python -u GNN_branch/main_GNN.py \
  --dataset mlir --subtask train --target perf area \
  --target_mode reference_delta \
  --baseline_manifest GNN_branch/baselines/neutral_smoke.csv \
  --loss smooth_l1 --smooth_l1_beta 0.5 \
  --standardize_targets --kernel_balanced_loss \
  --tiny_overfit --tiny_overfit_kernel machsuite-gemm-blocked \
  --tiny_overfit_num_samples 64 --tiny_overfit_epochs 300 \
  --lr 1e-3 --random_seed 123 \
  --experiment_name gnn_v15_stage_c_tiny_s123
```

Proceed only if both target losses collapse on the identical train/validation
subset and the exported predictions satisfy `predicted_log2 = baseline_log2 +
predicted_delta_log2` (within floating-point tolerance).

## 3. Build the development manifest

This performs 54 independent C-synthesis runs: 51 training kernels plus the
three declared validation kernels. The locked Kalman test kernel is excluded
by default.

```bash
python GNN_branch/generate_neutral_baselines.py \
  --output GNN_branch/baselines/neutral_vitis_2021_1.csv \
  --work-dir neutral_baseline_build
```

Do not use `--allow-test-kernels` during model development. Commit the manifest,
not the large `neutral_baseline_build/` projects.

## 4. Run the full Stage C validation experiment

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python -u GNN_branch/main_GNN.py \
  --dataset mlir \
  --subtask train \
  --target perf area \
  --target_mode reference_delta \
  --baseline_manifest GNN_branch/baselines/neutral_vitis_2021_1.csv \
  --loss smooth_l1 \
  --smooth_l1_beta 0.5 \
  --epoch_num 60 \
  --lr 3e-5 \
  --random_seed 123 \
  --experiment_name gnn_v15_stage_c_reference_delta_s123 \
  --standardize_targets \
  --kernel_balanced_loss \
  --kernel_uniform_sampling \
  --samples_per_kernel_per_epoch 128 \
  --scheduler plateau \
  --warmup_epochs 0 \
  --plateau_patience 2 \
  --early_stopping_patience 8 \
  --checkpoint_objective absolute \
  --val_kernels machsuite-sort-radix,rodinia_pathfinder_0_baseline_0,rodinia_pathfinder_4_doublebuffer_0 \
  --test_kernels serrano-kalman-filter
```

Do not add `--decompose_targets`, center/response auxiliary weights, or
`--evaluate_test`. This run succeeds only if one validation checkpoint beats
the measured neutral-reference (zero-delta) baseline for both targets. Until
that happens, the test remains locked and Stage C is not qualified.

No GEXF or cached PT regeneration is required: the baseline and delta tensors
are attached at runtime, leaving the graph encoding unchanged.
