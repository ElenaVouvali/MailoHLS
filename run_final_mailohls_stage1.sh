#!/usr/bin/env bash
set -euo pipefail

# Final MailoHLS Stage-1 run.
# Run from the MailoHLS repository root AFTER:
#   1) applying apply_mailohls_stage1_final_patch.py
#   2) passing the LLM tests
#   3) committing all tracked code changes
#
# This deliberately keeps the previous promising Stage-1 optimization setup.
# The only experimental changes are validation/reproducibility improvements:
#   - 4 distinct-target validation cases per kernel/device/clock when available
#   - batched candidate scoring for validation only
#   - teacher-forced MRR/top-1 diagnostics
#   - cascade + budget-counterfactual diagnostics
#   - one-effective-epoch early-stop floor
#   - canonical source-derived directive-domain hash
#
# Family sampling remains the old replacement=False behavior.
# Candidate-ranking TRAINING loss remains disabled.
# AUTO-clock training remains disabled.

OUT="mailohls_runs/stage1_final_mailohls_s123_v2"
LOG="${HOME}/stage1_final_mailohls_s123_v2.log"

if [[ -e "${OUT}" ]]; then
  echo "Refusing to reuse existing output directory: ${OUT}" >&2
  echo "Choose a new output directory for a scratch run." >&2
  exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked git tree is dirty. Commit or restore tracked changes first:" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u -m LLM_branch.train.train_SFT_xattn_new \
  --run_mode single \
  --disable_structural_memory \
  --objective PARETO_ADP \
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
  --auto_frequency_fraction 0 \
  --goal_domination_penalty 0.25 \
  --goal_max_dominated_gap 0.12 \
  --min_supervised_sites 2 \
  --min_site_coverage 0.85 \
  --directive_loss_weighting uniform \
  --value_loss_weight 1 \
  --ce_loss_weight 1 \
  --candidate_loss_weight 0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules attention \
  --lora_dropout 0.10 \
  --lora_weight_decay 0.01 \
  --lr_lora 3e-5 \
  --lr_embed 1e-5 \
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
  --selection_eval_steps 100 \
  --eval_steps 100 \
  --save_steps 100 \
  --early_stopping_patience 3 \
  --eval_on_start \
  --best_dir_name best_custom_stage1 \
  --epochs 3 \
  --max_steps 1200 \
  --seed 123 \
  --require_clean_git \
  --output_dir "${OUT}" \
  2>&1 | tee "${LOG}"
