#!/usr/bin/env bash
# Run from this LLVM worktree. Runtime weights/data stay outside the repository.
set -euo pipefail
: "${LLVM_RUN_ROOT:?Set LLVM_RUN_ROOT to the prepared runtime directory}"
: "${LLVM_BASE_MODEL:?Set LLVM_BASE_MODEL to the pinned cached DeepSeek snapshot}"
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU by UUID}"
LLVM_PYTHON="${LLVM_PYTHON:-python}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=123
export OMP_NUM_THREADS=4
mkdir -p "$LLVM_RUN_ROOT/logs"
exec 9>"$LLVM_RUN_ROOT/training.lock"
flock -n 9 || { echo 'This reconstruction is already running.' >&2; exit 1; }

common=(
  --dataset "$LLVM_RUN_ROOT/dataset.jsonl"
  --memory_dir "$LLVM_RUN_ROOT/memory"
  --model "$LLVM_BASE_MODEL"
  --objective PARETO_LATENCY_EXTREME
  --split_json "$LLVM_RUN_ROOT/split.json"
  --split_mode random_design --split_seed 123 --stratify_by_kernel
  --top_k 6 --goal_domination_penalty 0.25 --goal_max_dominated_gap 0.12
  --score_weight_min 0.6 --score_weight_power 1.0
  --min_supervised_sites 2 --min_site_coverage 0.85
  --selection_num_val_kernels 6 --max_length 7168
  --batch_size 1 --grad_accum 8 --num_workers 2
  --group_by_length --gradient_checkpointing
  --mem_dim 64 --max_slots 64 --every_n_layers 8
  --xattn_heads 4 --xattn_dim_head 64 --xattn_ff_mult 1 --seed 123
)

run_sft_stage() {
  local name="$1"; shift
  if [[ -f "$LLVM_RUN_ROOT/$name.complete.json" ]]; then
    echo "Skipping completed $name"
    return
  fi
  if compgen -G "$LLVM_RUN_ROOT/$name/checkpoint-*" >/dev/null; then
    echo "Incomplete $name has optimizer checkpoints. Resume with the native trainer's --resume_from_checkpoint; refusing to overwrite it." >&2
    exit 1
  fi
  if [[ "$name" == stage2 ]]; then
    test -f "$LLVM_RUN_ROOT/memory_rebuild_manifest.json" || {
      echo 'Rebuild and authenticate the LLVM memory bank before Stage 2.' >&2; exit 1;
    }
  fi
  "$LLVM_PYTHON" -u LLM_branch/train/train_SFT_xattn.py \
    "${common[@]}" --run_mode single --precision auto \
    --epochs 4 --eval_steps 40 --save_steps 40 --save_total_limit 2 \
    --output_dir "$LLVM_RUN_ROOT/$name" "$@" \
    >"$LLVM_RUN_ROOT/logs/$name.log" 2>&1
  "$LLVM_PYTHON" experiments/llvm_latency_reconstruction/record_checkpoint.py \
    --checkpoint "$LLVM_RUN_ROOT/$name" --output "$LLVM_RUN_ROOT/$name.complete.json"
}

run_sft_stage stage1 --disable_harp --best_dir_name best_custom_stage1 \
  --lr_lora 5e-5 --lr_embed 5e-5 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05

run_sft_stage stage2 --best_dir_name best_custom_stage2 \
  --init_adapter_dir "$LLVM_RUN_ROOT/stage1/best_custom_stage1" \
  --lr_lora 0 --lr_embed 0 --lr_xattn 1e-4 --lr_gate 2e-4 --lr_ff 0 --lr_gate_ff 0

if [[ ! -f "$LLVM_RUN_ROOT/stage3.complete.json" ]]; then
  "$LLVM_PYTHON" -u LLM_branch/train/train_DPO_harp_xattn.py \
    "${common[@]}" --sft_script LLM_branch/train/train_SFT_xattn.py \
    --stage1_adapter_dir "$LLVM_RUN_ROOT/stage1/best_custom_stage1" \
    --stage2_harp_xattn_path "$LLVM_RUN_ROOT/stage2/best_custom_stage2/harp_xattn.pt" \
    --output_dir "$LLVM_RUN_ROOT/stage3" \
    --epochs 1 --max_steps 60 --eval_steps 60 --save_steps 60 --save_total_limit 2 \
    --dpo_chosen_top_k 3 --dpo_hard_window 2 \
    --dpo_hard_negatives_per_chosen 2 --dpo_medium_negatives_per_chosen 1 \
    --dpo_min_score_gap 0.03 --dpo_hard_gap_max 0.15 --dpo_medium_gap_max 0.35 \
    --dpo_min_primary_rel_gain 0.02 --dpo_min_edit_distance 1 \
    --require_same_supervised_schema --beta 0.1 --label_smoothing 0 --sft_alpha 0 \
    --lr_lora 0 --lr_embed 0 --lr_xattn 5e-5 --lr_gate 2e-5 --lr_ff 0 --lr_gate_ff 0 \
    --train_xattn_dpo --train_attn_gate_dpo \
    >"$LLVM_RUN_ROOT/logs/stage3.log" 2>&1
  "$LLVM_PYTHON" experiments/llvm_latency_reconstruction/record_checkpoint.py \
    --checkpoint "$LLVM_RUN_ROOT/stage3" --output "$LLVM_RUN_ROOT/stage3.complete.json"
fi
