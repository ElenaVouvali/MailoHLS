#!/usr/bin/env bash
set -euo pipefail
: "${LLVM_RUN_ROOT:?Set LLVM_RUN_ROOT}"
: "${LLVM_BASE_MODEL:?Set LLVM_BASE_MODEL}"
LLVM_PYTHON="${LLVM_PYTHON:-python}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for stage in stage1 stage2 stage3; do
  test -f "$LLVM_RUN_ROOT/$stage.complete.json"
done
test -f "$LLVM_RUN_ROOT/external_memory_manifest.json"
mkdir -p "$LLVM_RUN_ROOT/predictions"
"$LLVM_PYTHON" -u LLM_branch/inference/eval_stage1_stage2_stage3.py \
  --stage stage3 --model "$LLVM_BASE_MODEL" \
  --adapter_dir "$LLVM_RUN_ROOT/stage3/best_custom_stage3" \
  --lora_adapter_dir "$LLVM_RUN_ROOT/stage1/best_custom_stage1" \
  --harp_xattn_path "$LLVM_RUN_ROOT/stage3/best_custom_stage3/harp_xattn.pt" \
  --memory_dir "$LLVM_RUN_ROOT/external_memory" --mem_dim 64 --max_slots 64 \
  --every_n_layers 8 --xattn_heads 4 --xattn_dim_head 64 --xattn_ff_mult 1 \
  --input_jsonl experiments/llvm_latency_reconstruction/cases.jsonl \
  --candidate_bank_dataset "$LLVM_RUN_ROOT/stage1/selected_debug/train_selected_pareto_latency_extreme.jsonl" \
  --save_rhs_candidate_bank_json "$LLVM_RUN_ROOT/predictions/rhs_candidate_bank.json" \
  --max_prompt_tokens 7168 --candidate_max_prefix_tokens 0 --score_reduction mean \
  --num_samples 1 --sample_seed 123 --debug_topk 3 \
  --output_jsonl "$LLVM_RUN_ROOT/predictions/llvm_latency_four_kernels.jsonl" \
  >"$LLVM_RUN_ROOT/logs/four_kernel_inference.log" 2>&1
