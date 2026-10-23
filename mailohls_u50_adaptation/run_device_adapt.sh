#!/usr/bin/env bash
set -euo pipefail
cd /home/elvouvali/MailoHLS/MailoHLS-next
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SEED=${SEED:-123}
ROOT=/home/elvouvali/mailohls_u50_adaptation
BASE=mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1
python -u -m LLM_branch.train.train_SFT_xattn_new \
  --run_mode single \
  --disable_structural_memory \
  --objective PARETO_LATENCY \
  --dataset "$ROOT/artifacts/u50_10ns_sft.jsonl" \
  --split_json "$ROOT/splits/u50_adaptation_split.json" \
  --minimum_validation_families 3 \
  --minimum_test_families 1 \
  --application_dataset_dir Data/ApplicationDataset \
  --model deepseek-ai/deepseek-coder-6.7b-base \
  --model_revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912 \
  --device_mode device_adapt \
  --adapt_device xcu50-fsvh2104-2-e \
  --device_specs_json "$ROOT/u50_device_specs.json" \
  --init_adapter_dir "$BASE" \
  --device_token_dropout 0 \
  --resource_budget_mode none \
  --auto_frequency_fraction 0 \
  --top_k 1 \
  --ce_loss_weight 1 \
  --candidate_loss_weight 0.10 \
  --candidate_sites_per_sample 2 \
  --candidate_negatives_per_site 2 \
  --candidate_min_sample_fraction 0 \
  --min_supervised_sites 2 \
  --min_site_coverage 0.85 \
  --lora_r 2 \
  --lora_alpha 4 \
  --lora_target_modules attention \
  --lora_dropout 0.10 \
  --lora_weight_decay 0.01 \
  --lr_lora 5e-6 \
  --lr_embed 0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.10 \
  --max_grad_norm 1 \
  --max_length 7168 \
  --batch_size 1 \
  --grad_accum 4 \
  --gradient_checkpointing \
  --num_workers 0 \
  --loss_chunk_t 128 \
  --family_sampling_power 0.5 \
  --selection_num_val_kernels 0 \
  --selection_cases_per_kernel_device 1 \
  --selection_candidate_batch_size 4 \
  --selection_eval_steps 5 \
  --eval_steps 5 \
  --save_steps 5 \
  --early_stopping_patience 3 \
  --eval_on_start \
  --epochs 20 \
  --max_steps 40 \
  --best_dir_name best_custom_stage1 \
  --seed "$SEED" \
  --output_dir "$ROOT/runs/u50_latency_seed${SEED}"
