#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
OBJECTIVE="${OBJECTIVE:-PARETO_ADP}"
case "$OBJECTIVE" in
  PARETO_ADP|PARETO_AREA|PARETO_LATENCY) ;;
  *)
    echo "Unsupported objective: $OBJECTIVE" >&2
    exit 2
    ;;
esac
OBJECTIVE_TAG="${OBJECTIVE#PARETO_}"
OBJECTIVE_TAG="${OBJECTIVE_TAG,,}"
OUT="${OUT:-mailohls_runs/stage1_final_final_${OBJECTIVE_TAG}_s123}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing Stage-1 output: $OUT" >&2
  exit 2
fi
# if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
#   echo "Tracked git tree is dirty. Commit the final patch before this run." >&2
#   git status --short --untracked-files=no >&2
#   exit 2
# fi
CUDA_VISIBLE_DEVICES="$GPU" \
python -u -m LLM_branch.train.train_SFT_xattn_new \
  --run_mode single \
  --disable_structural_memory \
  --objective "$OBJECTIVE" \
  --dataset artifacts/llm/mailohls_sft.jsonl \
  --split_json mailohls_runs/mailohls_final_family_split_s123.json \
  --minimum_validation_families 3 \
  --minimum_test_families 1 \
  --application_dataset_dir Data/ApplicationDataset \
  --model deepseek-ai/deepseek-coder-6.7b-base \
  --model_revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912 \
  --top_k 1 \
  --device_mode known \
  --device_token_dropout 0 \
  --resource_budget_mode random \
  --random_budgets_per_case 16 \
  --random_budget_min_frac 0.05 \
  --min_feasible_candidates_per_budget 3 \
  --candidate_pool_per_objective 24 \
  --budget_target_max_duplicates 4 \
  --auto_frequency_fraction 0 \
  --goal_domination_penalty 0.25 \
  --goal_max_dominated_gap 0.12 \
  --min_supervised_sites 2 \
  --min_site_coverage 0.85 \
  --directive_loss_weighting inverse_sqrt_frequency \
  --value_loss_weight 1 \
  --ce_loss_weight 1 \
  --candidate_loss_weight 0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules attention \
  --lora_dropout 0.10 \
  --lora_weight_decay 0.01 \
  --lr_lora 1.5e-5 \
  --lr_embed 5e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_grad_norm 1 \
  --max_length 7168 \
  --batch_size 1 \
  --grad_accum 8 \
  --gradient_checkpointing \
  --num_workers 0 \
  --loss_chunk_t 128 \
  --family_sampling_power 0.5 \
  --selection_num_val_kernels 0 \
  --selection_cases_per_kernel_device 4 \
  --selection_candidate_batch_size 4 \
  --selection_eval_steps 50 \
  --eval_steps 50 \
  --save_steps 50 \
  --early_stopping_patience 3 \
  --eval_on_start \
  --best_dir_name best_custom_stage1 \
  --epochs 2 \
  --max_steps -1 \
  --seed 123 \
  # --require_clean_git \
  --output_dir "$OUT"
