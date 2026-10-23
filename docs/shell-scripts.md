# Shell launchers

Run maintained launchers with Bash from any working directory. They resolve the
repository from their own location; relative data/output paths refer to the
repository root. Training commands launch real jobs. Inspect their settings and
choose fresh output directories before running them.

| Purpose | Entry point |
| --- | --- |
| Prepare measurements, SFT data, and family split | `scripts/reproduction/prepare_final_experiment.sh` |
| Train the selected absolute-QoR GNN profile | `scripts/training/train_gnn.sh` |
| Train target-aware Stage 1 | `scripts/training/train_stage1.sh` |
| Train structural Stage 2 | `scripts/training/train_stage2.sh` |
| Export selected GNN structural memory | `scripts/training/export_gnn_memory.sh` |
| Contract-driven Stage 3 | `LLM_branch/inference/run_stage3_contract.sh` |
| Fixed-clock/target adaptation to U50 | `experiments/u50_adaptation/run_device_adapt.sh` |
| Locked automatic-clock evaluation | `scripts/evaluation/evaluate_auto_clock_test.sh` |
| Structural gate parity check | `scripts/evaluation/run_gate_parity_screen.sh` |

The memory export defaults now match the selected checkpoint and output family
in `FINAL_ARTIFACTS.json`. Existing output is protected; set `BASE_OUT` to a
fresh path when repeating an export. U50 adaptation uses the local
`mailohls_u50_adaptation/` assets by default; set `U50_ADAPT_ROOT` to use a
different directory and `MAILOHLS_STAGE1` to choose a parent adapter.

## Experiment presets

- `scripts/experiments/gnn/`: alternate GNN losses, heads, and multitarget runs.
- `scripts/experiments/representation/`: paired HARP/MLIR and HARP-GNOSIS profiles.
- `scripts/experiments/stage3/`: AREA/LATENCY trials and ADP replication queue.
- `scripts/experiments/templates/`: earlier reproducibility profiles with their
  original hyperparameters. These are distinct experiments, not aliases for
  the selected training profile.
- `scripts/asplos_ablation/` and `experiments/asplos_memory_ablation/`: ablations.

Do not infer production status from a filename containing `final`. Consult
`FINAL_ARTIFACTS.json`; some AREA/LATENCY Stage-2 variants remain experiments.

## Historical and upstream scripts

`scripts/legacy/inference/` and `scripts/legacy/src/` preserve the old LLVM-era
launchers. Their original machine-specific data/model paths and command-line
contracts are retained for provenance; configure the historical environment
before running them. Four identical historical copies are relative symlink
aliases. `src/clang_script.sh` likewise aliases `GNN_branch/clang_script.sh` so
existing compiler callers continue to work.

Toolchain scripts, upstream HARP scripts, and compiler helpers remain beside
the code they support. No upstream baseline experiment parameters were changed.
The organized MLIR baseline is preserved on `mlir-target`; pre-cleanup layouts
remain available in Git history. The original LLVM baseline stays on `llvm-ir`.

## Previous paths

Existing shell commands should use the replacements below. No redundant
root-level wrappers are retained.

| Previous path | Current path |
| --- | --- |
| `run_final_mailohls_gnn_absolute_direct.sh` | `scripts/training/train_gnn.sh` |
| `run_final_mailohls_stage1.sh` | `scripts/training/train_stage1.sh` |
| `run_final_mailohls_stage2_specified.sh` | `scripts/training/train_stage2.sh` |
| `export_final_mailohls_gnn_memory.sh` | `scripts/training/export_gnn_memory.sh` |
| `run_final_mailohls_gnn.sh` | `scripts/experiments/gnn/train_reference_delta_hardware.sh` |
| `run_final_mailohls_gnn_absolute_rankdense.sh` | `scripts/experiments/gnn/train_absolute_rankdense.sh` |
| `run_final_mailohls_gnn_reference_delta.sh` | `scripts/experiments/gnn/train_reference_delta_pairwise.sh` |
| `run_final_mailohls_gnn_reference_delta_direct.sh` | `scripts/experiments/gnn/train_reference_delta_direct.sh` |
| `run_final_mailohls_gnn_multitarget_proven_rankdense.sh` | `scripts/experiments/gnn/train_multitarget_rankdense.sh` |
| `run_final_mailohls_gnn_multitarget_structural.sh` | `scripts/experiments/gnn/train_multitarget_structural.sh` |
| `run_stage3_area_s789.sh` | `scripts/experiments/stage3/train_area_s789.sh` |
| `run_stage3_latency_s123.sh` | `scripts/experiments/stage3/train_latency_s123.sh` |
| `run_stage3_overnight_queue.sh` | `scripts/experiments/stage3/run_replication_queue.sh` |
| `run_harp_gnosis_paper_profile.sh` | `scripts/experiments/representation/train_harp_gnosis.sh` |
| `run_paired_representation_gnn.sh` | `scripts/experiments/representation/train_paired_gnn.sh` |
| `evaluate_auto_clock_test.sh` | `scripts/evaluation/evaluate_auto_clock_test.sh` |
| `scripts/reproduction/run_final_gnn.sh` | `scripts/experiments/templates/train_gnn.sh` |
| `scripts/reproduction/run_final_stage1.sh` | `scripts/experiments/templates/train_stage1.sh` |
| `scripts/reproduction/run_gate_parity_screen.sh` | `scripts/evaluation/run_gate_parity_screen.sh` |
| `mailohls_u50_adaptation/run_device_adapt.sh` | `experiments/u50_adaptation/run_device_adapt.sh` |
| `LLM_branch/inference/change_base_model.sh` | `scripts/legacy/inference/change_base_model.sh` |
| `LLM_branch/inference/deepseek_1.3b_DPO.sh` | `scripts/legacy/inference/deepseek_1.3b_DPO.sh` |
| `LLM_branch/inference/designsplit_2_stages_3_goals.sh` | `scripts/legacy/inference/designsplit_2_stages_3_goals.sh` |
| `LLM_branch/inference/dpo_designsplit_2_stages_3_goals.sh` | `scripts/legacy/inference/dpo_designsplit_2_stages_3_goals.sh` |
| `LLM_branch/inference/machsuite_holdout_2_stages_3_goals.sh` | `scripts/legacy/inference/machsuite_holdout_2_stages_3_goals.sh` |
| `LLM_branch/inference/random_design_split_inference_GAN.sh` | `scripts/legacy/inference/random_design_split_inference_GAN.sh` |
| `LLM_branch/inference/random_design_split_inference_all.sh` | `scripts/legacy/inference/random_design_split_inference_all.sh` |
| `LLM_branch/inference/serrano_holdout_2_stages_3_goals.sh` | `scripts/legacy/inference/serrano_holdout_2_stages_3_goals.sh` |
| `LLM_branch/inference/serrano_holdout_inference_all.sh` | `scripts/legacy/inference/serrano_holdout_inference_all.sh` |
| `src/change_base_model.sh` | `scripts/legacy/src/change_base_model.sh` |
| `src/deepseek_1.3b_DPO.sh` | `scripts/legacy/src/deepseek_1.3b_DPO.sh` |
| `src/machsuite_holdout_2_stages_3_goals.sh` | `scripts/legacy/src/machsuite_holdout_2_stages_3_goals.sh` |
| `src/random_design_split_inference_GAN.sh` | `scripts/legacy/src/random_design_split_inference_GAN.sh` |
| `src/random_design_split_inference_all.sh` | `scripts/legacy/src/random_design_split_inference_all.sh` |
| `src/random_designsplit_3stages_3goals.sh` | `scripts/legacy/src/random_designsplit_3stages_3goals.sh` |
| `src/serrano_holdout_inference_ablation.sh` | `scripts/legacy/src/serrano_holdout_inference_ablation.sh` |
| `src/serrano_holdout_inference_all.sh` | `scripts/legacy/src/serrano_holdout_inference_all.sh` |
| `src/unseen_machsuite_3stages_3goals.sh` | `scripts/legacy/src/unseen_machsuite_3stages_3goals.sh` |
| `src/unseen_machsuite_inference_all.sh` | `scripts/legacy/src/unseen_machsuite_inference_all.sh` |
| `src/unseen_serrano_3stages_3goals.sh` | `scripts/legacy/src/unseen_serrano_3stages_3goals.sh` |
| `src/unseen_serrano_inference_serrano_ablation.sh` | `scripts/legacy/src/unseen_serrano_inference_serrano_ablation.sh` |
| `src/unseen_serrano_inference_serrano_stage3.sh` | `scripts/legacy/src/unseen_serrano_inference_serrano_stage3.sh` |
