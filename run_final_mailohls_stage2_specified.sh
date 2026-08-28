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

GPU="${CUDA_VISIBLE_DEVICES:-0}"
STAGE1="${STAGE1:-mailohls_runs/stage1_final_final_${OBJECTIVE_TAG}_s123/best_custom_stage1}"
MEMORY="${MEMORY:-artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned}"
OUT="${OUT:-mailohls_runs/stage2_specified_${OBJECTIVE_TAG}_epoch9_s123}"

INIT_REF="${INIT_REF:-mailohls_runs/stage2_initial_states/post_self_attention_residual_s123.json}"

for required in \
  "$STAGE1/adapter_model.safetensors" \
  "$STAGE1/training_contract.json" \
  "$MEMORY/memory_manifest.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required Stage-2 input: $required" >&2
    exit 2
  fi
done
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing Stage-2 output: $OUT" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU" \
python -u -m LLM_branch.train.train_SFT_xattn_new \
  --run_mode single \
  --objective "$OBJECTIVE" \
  --dataset artifacts/llm/mailohls_sft.jsonl \
  --split_json mailohls_runs/mailohls_final_family_split_s123.json \
  --minimum_validation_families 3 \
  --minimum_test_families 1 \
  --application_dataset_dir Data/ApplicationDataset \
  --model deepseek-ai/deepseek-coder-6.7b-base \
  --model_revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912 \
  --init_adapter_dir "$STAGE1" \
  --memory_dir "$MEMORY" \
  --require_pragma_free_memory \
  --structural_routing compiler_relational \
  --structural_fusion_placement post_self_attention_residual \
  --mem_dim -1 \
  --max_slots 64 \
  --every_n_layers 8 \
  --xattn_heads 4 \
  --xattn_dim_head 64 \
  --xattn_ff_mult 1 \
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
  --lr_lora 0 \
  --lr_embed 0 \
  --lr_xattn 1e-4 \
  --lr_gate 2e-4 \
  --lr_ff 0 \
  --lr_gate_ff 0 \
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
  --selection_eval_steps 177 \
  --eval_steps 177 \
  --save_steps 177 \
  --early_stopping_patience 3 \
  --no-eval_on_start \
  --early_stopping_min_step 177 \
  --best_dir_name best_custom_stage2 \
  --epochs 3 \
  --max_steps -1 \
  --seed 123 \
  --require_clean_git \
  --output_dir "$OUT" \
  --initial_state_reference "$INIT_REF" 
