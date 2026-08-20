#!/usr/bin/env bash
set -euo pipefail

: "${MAILOHLS_DATA:?set MAILOHLS_DATA}"
: "${MAILOHLS_SPLIT:?set MAILOHLS_SPLIT}"
: "${MAILOHLS_DOMAINS:?set MAILOHLS_DOMAINS}"
: "${MAILOHLS_MEMORY:?set MAILOHLS_MEMORY}"
: "${MAILOHLS_STAGE1:?set MAILOHLS_STAGE1}"
: "${ORIGINAL_THETA:?set ORIGINAL_THETA}"
: "${BAKED_THETA:?set BAKED_THETA}"
: "${PARITY_OUTPUT_ROOT:?set PARITY_OUTPUT_ROOT}"

for path in "$MAILOHLS_DATA" "$MAILOHLS_SPLIT" "$MAILOHLS_DOMAINS" \
  "$ORIGINAL_THETA" "$BAKED_THETA"; do
    test -s "$path" || { echo "Missing or empty file: $path" >&2; exit 2; }
done
test -d "$MAILOHLS_MEMORY"
test -d "$MAILOHLS_STAGE1"
mkdir -p "$PARITY_OUTPUT_ROOT"

run_screen() {
    local name="$1" gate_scale="$2" theta="$3"
    local output="$PARITY_OUTPUT_ROOT/$name"
    local log="$PARITY_OUTPUT_ROOT/$name.run.log"
    test ! -e "$output" && test ! -e "$log" || {
        echo "Refusing to overwrite $output or $log" >&2
        return 2
    }
    CUDA_VISIBLE_DEVICES="${SCREEN_GPU:-0}" \
    python -u -m LLM_branch.train.train_SFT_xattn_new \
      --run_mode single --objective PARETO_ADP \
      --dataset "$MAILOHLS_DATA" --split_json "$MAILOHLS_SPLIT" \
      --directive_domain_registry_json "$MAILOHLS_DOMAINS" \
      --memory_dir "$MAILOHLS_MEMORY" \
      --model deepseek-ai/deepseek-coder-6.7b-base \
      --model_revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912 \
      --init_adapter_dir "$MAILOHLS_STAGE1" \
      --init_structural_xattn_from "$theta" --require_pragma_free_memory \
      --selection_eval_only --structural_fusion_placement post_decoder_residual \
      --structural_routing exact_slot --every_n_layers 8 --xattn_heads 4 \
      --xattn_dim_head 64 --xattn_ff_mult 1 --candidate_loss_weight 0 \
      --lr_lora 0 --lr_embed 0 --lr_xattn 0 --lr_gate 0 --lr_ff 0 --lr_gate_ff 0 \
      --structural_gate_scale "$gate_scale" --structural_memory_value_scale 1 \
      --max_length 7168 --batch_size 1 --num_workers 0 \
      --selection_num_val_kernels 0 --selection_cases_per_kernel_device 16 \
      --selection_candidate_batch_size 1 --best_dir_name best_custom_stage2 \
      --save_selection_debug --seed 123 --output_dir "$output" 2>&1 | tee "$log"
}

run_screen original_scale32 32 "$ORIGINAL_THETA"
run_screen baked_scale1 1 "$BAKED_THETA"
