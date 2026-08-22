#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

seed="${MAILOHLS_SEED:-123}"
objective="${MAILOHLS_OBJECTIVE:-PARETO_ADP}"
dataset="${MAILOHLS_SFT_DATASET:-$repo_root/artifacts/llm/mailohls_sft.jsonl}"
split="${MAILOHLS_FAMILY_SPLIT:-$repo_root/mailohls_runs/stage1_final_family_split_s${seed}.json}"
registry="${MAILOHLS_DOMAIN_REGISTRY:-$repo_root/artifacts/llm/directive_domain_registry.json}"
output="${MAILOHLS_STAGE1_OUTPUT:-$repo_root/mailohls_runs/stage1_final_${objective,,}_s${seed}}"

for artifact in "$dataset" "$split" "$registry"; do
  if [[ ! -f "$artifact" ]]; then
    printf 'Required Stage-1 artifact does not exist: %s\n' "$artifact" >&2
    exit 1
  fi
done

python -u -m LLM_branch.train.train_SFT_xattn_new \
  --run_mode single \
  --disable_structural_memory \
  --objective "$objective" \
  --dataset "$dataset" \
  --split_json "$split" \
  --directive_domain_registry_json "$registry" \
  --model deepseek-ai/deepseek-coder-6.7b-base \
  --model_revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912 \
  --top_k 1 \
  --device_mode known \
  --device_token_dropout 0 \
  --resource_budget_mode random \
  --random_budgets_per_case 16 \
  --random_budget_min_frac 0.01 \
  --min_feasible_candidates_per_budget 3 \
  --candidate_pool_per_objective 24 \
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
  --lora_dropout 0.05 \
  --lr_lora 5e-5 \
  --lr_embed 5e-5 \
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
  --selection_cases_per_kernel_device 2 \
  --selection_candidate_batch_size 2 \
  --selection_eval_steps 100 \
  --eval_steps 100 \
  --save_steps 100 \
  --early_stopping_patience 3 \
  --eval_on_start \
  --best_dir_name best_custom_stage1 \
  --epochs 3 \
  --max_steps 1200 \
  --seed "$seed" \
  --output_dir "$output" \
  "$@"
