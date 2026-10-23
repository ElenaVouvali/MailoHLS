#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "[FATAL] rc=$rc line=$LINENO command=$BASH_COMMAND" >&2; exit $rc' ERR

ROOT="${MAILOHLS_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
GPU="${GPU:-0}"
OBJECTIVE="${OBJECTIVE:-PARETO_ADP}"
SEED12="${SEED12:-123}"
SEED3="${SEED3:-789}"

if [[ "$OBJECTIVE" != "PARETO_ADP" ]]; then
  echo "This ASPLOS architecture-ablation queue is intentionally locked to PARETO_ADP" >&2
  echo "because the current production Stage-3 artifact is PARETO_ADP." >&2
  exit 2
fi

cd "$ROOT"

DATA="artifacts/llm/mailohls_sft.jsonl"
SPLIT="mailohls_runs/mailohls_final_family_split_s123.json"
MEMORY="artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned"
PROD_S1="mailohls_runs/stage1_final_final_adp_s123/best_custom_stage1"
PROD_S2="mailohls_runs/stage2_specified_adp_epoch9_s123_v3/best_custom_stage2"
PROD_S3="mailohls_runs/stage3_adp_semantic_action_s789_rep2/best_custom_stage3"

MODEL67="deepseek-ai/deepseek-coder-6.7b-base"
REV67="ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912"
MODEL13="deepseek-ai/deepseek-coder-1.3b-base"
REV13="${MODEL13_REVISION:-c919139c3a9b4070729c8b2cca4847ab29ca8d94}"

LOGROOT="mailohls_runs/asplos_arch_ablation_logs"
mkdir -p "$LOGROOT" mailohls_runs/stage2_initial_states

# Experiment A: backbone capacity, fresh Stage 1 -> Stage 2 -> Stage 3.
S1_13="mailohls_runs/asplos_ablation_backbone_1p3b_stage1_adp_s123"
S2_13="mailohls_runs/asplos_ablation_backbone_1p3b_stage2_every8_adp_s123"
S3_13="mailohls_runs/asplos_ablation_backbone_1p3b_stage3_every8_adp_s789"
REF_13="mailohls_runs/stage2_initial_states/asplos_backbone_1p3b_every8_s123.json"
CACHE_13="mailohls_runs/stage2_data_cache_adp_s123.pt"

# Experiment B: 2x sparser structural fusion. Stage 1 is deliberately reused,
# because Stage 1 disables structural memory; retraining it would add a seed/noise
# confound while changing nothing about cross-attention placement.
S2_16="mailohls_runs/asplos_ablation_xattn_every16_stage2_adp_s123"
S3_16="mailohls_runs/asplos_ablation_xattn_every16_stage3_adp_s789"
REF_16="mailohls_runs/stage2_initial_states/asplos_xattn_every16_s123.json"
CACHE_16="mailohls_runs/stage2_data_cache_adp_s123.pt"

for f in "$DATA" "$SPLIT" "$MEMORY/memory_manifest.json" \
         "$PROD_S1/adapter_model.safetensors" "$PROD_S1/training_contract.json" \
         "$PROD_S2/training_contract.json" "$PROD_S3/training_contract.json"; do
  [[ -f "$f" ]] || { echo "Missing required input: $f" >&2; exit 2; }
done

# Provenance snapshot; do not silently require a clean tree because the current
# production launchers themselves permit dirty metadata, but preserve it.
{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "root=$ROOT"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "gpu=$GPU"
  echo "python=$(command -v python)"
  echo "objective=$OBJECTIVE"
  echo "model13=$MODEL13"
  echo "rev13=$REV13"
  echo "model67=$MODEL67"
  echo "rev67=$REV67"
  echo "git_status_begin"
  git status --short --untracked-files=no || true
  echo "git_status_end"
} | tee "$LOGROOT/provenance_$(date +%Y%m%d_%H%M%S).txt"

python - <<PY
import torch
from transformers import AutoConfig
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is not available in the active environment"
print("GPU:", torch.cuda.get_device_name(0))
for model, rev, exp_layers, exp_hidden in [
    ("$MODEL13", "$REV13", 24, 2048),
    ("$MODEL67", "$REV67", 32, 4096),
]:
    c = AutoConfig.from_pretrained(model, revision=rev, trust_remote_code=True)
    print(model, rev, "layers=", c.num_hidden_layers, "hidden=", c.hidden_size)
    assert c.num_hidden_layers == exp_layers
    assert c.hidden_size == exp_hidden
PY

stage1_complete() {
  [[ -f "$1/best_custom_stage1/training_contract.json" && \
     -f "$1/best_custom_stage1/adapter_model.safetensors" ]]
}
stage2_complete() {
  [[ -f "$1/best_custom_stage2/training_contract.json" && \
     -f "$1/best_custom_stage2/structural_xattn.pt" ]]
}
stage3_complete() {
  [[ -f "$1/best_custom_stage3/training_contract.json" && \
     -f "$1/best_custom_stage3/structural_xattn.pt" ]]
}
refuse_partial() {
  local out="$1" complete_kind="$2"
  [[ ! -e "$out" ]] && return 0
  case "$complete_kind" in
    s1) stage1_complete "$out" && return 0 ;;
    s2) stage2_complete "$out" && return 0 ;;
    s3) stage3_complete "$out" && return 0 ;;
  esac
  echo "Refusing partial/stale output: $out" >&2
  echo "Move/delete that directory, then relaunch the queue." >&2
  exit 2
}

run_stage1_13() {
  if stage1_complete "$S1_13"; then
    echo "[SKIP] complete 1.3B Stage-1: $S1_13"
    return
  fi
  refuse_partial "$S1_13" s1
  echo "[RUN] 1.3B Stage-1 -> $S1_13"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -u -m LLM_branch.train.train_SFT_xattn_new \
    --run_mode single \
    --disable_structural_memory \
    --objective "$OBJECTIVE" \
    --dataset "$DATA" \
    --split_json "$SPLIT" \
    --minimum_validation_families 3 \
    --minimum_test_families 1 \
    --application_dataset_dir Data/ApplicationDataset \
    --model "$MODEL13" \
    --model_revision "$REV13" \
    --top_k 1 \
    --device_mode known \
    --device_token_dropout 0 \
    --resource_budget_mode random \
    --random_budgets_per_case 16 \
    --random_budget_min_frac 0.05 \
    --min_feasible_candidates_per_budget 3 \
    --candidate_pool_per_objective 24 \
    --budget_target_max_duplicates 4 \
    --budget_invariant_max_duplicates 1 \
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
    --seed "$SEED12" \
    --output_dir "$S1_13" \
    2>&1 | tee "$LOGROOT/backbone_1p3b_stage1.log"
  stage1_complete "$S1_13" || { echo "1.3B Stage-1 did not produce a complete best checkpoint" >&2; exit 3; }
}

run_stage2() {
  local tag="$1" model="$2" rev="$3" stage1="$4" every="$5" out="$6" ref="$7" cache="$8"
  if stage2_complete "$out"; then
    echo "[SKIP] complete Stage-2 $tag: $out"
    return
  fi
  refuse_partial "$out" s2
  local cache_args=(--data_cache_path "$cache" --reuse_data_cache)
  echo "[RUN] Stage-2 $tag every_n_layers=$every -> $out"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -u -m LLM_branch.train.train_SFT_xattn_new \
    --run_mode single \
    --objective "$OBJECTIVE" \
    --dataset "$DATA" \
    --split_json "$SPLIT" \
    --minimum_validation_families 3 \
    --minimum_test_families 1 \
    --application_dataset_dir Data/ApplicationDataset \
    --model "$model" \
    --model_revision "$rev" \
    --init_adapter_dir "$stage1" \
    --memory_dir "$MEMORY" \
    --require_pragma_free_memory \
    --structural_routing compiler_relational \
    --structural_fusion_placement post_self_attention_residual \
    --mem_dim -1 \
    --max_slots 64 \
    --every_n_layers "$every" \
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
    --budget_invariant_max_duplicates 1 \
    --auto_frequency_fraction 0 \
    --goal_domination_penalty 0.25 \
    --goal_max_dominated_gap 0.12 \
    --min_supervised_sites 2 \
    --min_site_coverage 0.85 \
    --directive_loss_weighting inverse_sqrt_frequency \
    --value_loss_weight 1 \
    --ce_loss_weight 1 \
    --candidate_loss_weight 0 \
    --candidate_sites_per_sample 0 \
    --candidate_negatives_per_site 0 \
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
    --early_stopping_patience 3 \
    --early_stopping_min_step 177 \
    --selection_eval_steps 177 \
    --eval_steps 177 \
    --save_steps 177 \
    --no-eval_on_start \
    --best_dir_name best_custom_stage2 \
    --epochs 3 \
    --max_steps -1 \
    --seed "$SEED12" \
    "${cache_args[@]}" \
    --output_dir "$out" \
    --initial_state_reference "$ref" \
    2>&1 | tee "$LOGROOT/${tag}_stage2.log"
  stage2_complete "$out" || { echo "Stage-2 $tag did not produce a complete best checkpoint" >&2; exit 3; }
}

run_stage3() {
  local tag="$1" stage2_root="$2" out="$3"
  if stage3_complete "$out"; then
    echo "[SKIP] complete Stage-3 $tag: $out"
    return
  fi
  refuse_partial "$out" s3
  echo "[RUN] Stage-3 $tag -> $out"
  MAILOHLS_DATA="$DATA" \
  MAILOHLS_SPLIT="$SPLIT" \
  MAILOHLS_DOMAINS="$stage2_root/directive_domain_registry.json" \
  MAILOHLS_MEMORY="$MEMORY" \
  MAILOHLS_STAGE2="$stage2_root/best_custom_stage2" \
  MAILOHLS_STAGE3_OUT="$out" \
  STAGE3_MODE=train \
  STAGE3_GPU="$GPU" \
  STAGE3_SEED="$SEED3" \
  STAGE2_SEED="$SEED12" \
  STAGE3_MAX_STEPS=40 \
  STAGE3_EVAL_STEPS=20 \
  STAGE3_SELECTION_EVAL_STEPS=20 \
  STAGE3_SAVE_STEPS=20 \
  STAGE3_LOGGING_STEPS=5 \
  STAGE3_SELECTION_EARLY_STOPPING_PATIENCE=1 \
  STAGE3_SELECTION_EARLY_STOPPING_MIN_STEP=20 \
  STAGE3_PAIR_UNIT=semantic_action \
  STAGE3_SEMANTIC_ACTION_ONLY=1 \
  STAGE3_MIN_ACTION_DISTANCE=1 \
  STAGE3_MAX_ACTION_DISTANCE=1 \
  STAGE3_MIN_EDIT_DISTANCE=1 \
  STAGE3_MAX_EDIT_DISTANCE=0 \
  STAGE3_MAX_REFERENCE_MARGIN=0.05 \
  STAGE3_REQUIRE_CHOSEN_RANK0=1 \
  STAGE3_REUSE_PAIR_CACHE=0 \
  STAGE3_REUSE_SELECTION_CACHE=0 \
  STAGE3_BETA=0.5 \
  STAGE3_SFT_ALPHA=0.02 \
  STAGE3_LR_XATTN=5e-5 \
  STAGE3_LR_GATE=2e-5 \
  bash LLM_branch/inference/run_stage3_contract.sh \
    2>&1 | tee "$LOGROOT/${tag}_stage3.log"
  stage3_complete "$out" || { echo "Stage-3 $tag did not produce a complete best checkpoint" >&2; exit 3; }
}

verify_contract() {
  local label="$1" path="$2" exp_model="$3" exp_every="$4" exp_layers_csv="$5"
  python - "$label" "$path" "$exp_model" "$exp_every" "$exp_layers_csv" <<'PY'
import json, sys
label, path, exp_model, exp_every, exp_layers_csv = sys.argv[1:]
c = json.load(open(path))
assert c["model"] == exp_model, (label, c["model"], exp_model)
s = c["structural"]
assert int(s["every_n_layers"]) == int(exp_every), (label, s["every_n_layers"], exp_every)
expected = [int(x) for x in exp_layers_csv.split(",") if x]
actual = [int(x) for x in s["selected_xattn_layers_1based"]]
assert actual == expected, (label, actual, expected)
print(f"[VERIFY] {label}: model={c['model']} every={s['every_n_layers']} layers={actual}")
PY
}

# =========================
# A) 1.3B backbone ablation
# =========================
run_stage1_13
run_stage2 "backbone_1p3b_every8" "$MODEL13" "$REV13" \
  "$S1_13/best_custom_stage1" 8 "$S2_13" "$REF_13" "$CACHE_13"
verify_contract "1.3B Stage-2" "$S2_13/best_custom_stage2/training_contract.json" \
  "$MODEL13" 8 "8,16,24"
run_stage3 "backbone_1p3b_every8" "$S2_13" "$S3_13"
verify_contract "1.3B Stage-3" "$S3_13/best_custom_stage3/training_contract.json" \
  "$MODEL13" 8 "8,16,24"

# ===================================
# B) 6.7B x2-sparser XAttn ablation
# ===================================
run_stage2 "xattn_every16" "$MODEL67" "$REV67" \
  "$PROD_S1" 16 "$S2_16" "$REF_16" "$CACHE_16"
verify_contract "6.7B every16 Stage-2" "$S2_16/best_custom_stage2/training_contract.json" \
  "$MODEL67" 16 "16,32"
run_stage3 "xattn_every16" "$S2_16" "$S3_16"
verify_contract "6.7B every16 Stage-3" "$S3_16/best_custom_stage3/training_contract.json" \
  "$MODEL67" 16 "16,32"

python - <<PY
import json, subprocess
manifest = {
  "schema": "mailohls-asplos-architecture-ablation-v1",
  "git_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
  "objective": "$OBJECTIVE",
  "production_controls": {
    "backbone_6p7b_stage1": "$PROD_S1",
    "backbone_6p7b_full": "$PROD_S3",
    "xattn_every8_stage2": "$PROD_S2",
    "xattn_every8_full": "$PROD_S3"
  },
  "backbone_1p3b": {
    "model": "$MODEL13", "revision": "$REV13",
    "stage1": "$S1_13/best_custom_stage1",
    "stage2": "$S2_13/best_custom_stage2",
    "stage3": "$S3_13/best_custom_stage3",
    "every_n_layers": 8,
    "selected_xattn_layers_1based": [8,16,24]
  },
  "xattn_every16": {
    "model": "$MODEL67", "revision": "$REV67",
    "reused_stage1": "$PROD_S1",
    "stage2": "$S2_16/best_custom_stage2",
    "stage3": "$S3_16/best_custom_stage3",
    "every_n_layers": 16,
    "selected_xattn_layers_1based": [16,32]
  }
}
path = "mailohls_runs/asplos_architecture_ablation_manifest.json"
with open(path, "w") as f: json.dump(manifest, f, indent=2)
print("[DONE]", path)
PY

echo "[DONE] Both ASPLOS architecture ablations completed."
echo "1.3B Stage1-only: $S1_13/best_custom_stage1"
echo "1.3B full:        $S3_13/best_custom_stage3"
echo "6.7B every16:     $S3_16/best_custom_stage3"
