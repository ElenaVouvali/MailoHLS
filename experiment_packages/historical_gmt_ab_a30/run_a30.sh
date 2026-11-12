#!/usr/bin/env bash
# Matched Stage-2 A/B on a *dedicated A30*, with original historical MLIR GNN
# versus the same 64 action vectors + 3 compiler-only GMT summary slots/action.
set -euo pipefail

if [[ ! -f experiments/historical_mlir_gmt_gemv/historical_gmt_slots.py ]]; then
  echo 'ERROR: cd into MailoHLS worktree on research/historical-mlir-gmt-gemv' >&2; exit 2
fi
if [[ ! -f LLM_branch/train/train_SFT_xattn_new.py ]]; then
  echo 'ERROR: Missing maintained Stage-2 production trainer' >&2; exit 2
fi
ROOT="$PWD"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
PARENT="${PARENT:-$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run}"
ASSETS="${HISTORICAL_ASSETS_ROOT:-$HOME/MailoHLS}"
GEXF="${GEXF_DIR:-$ROOT/GNN_branch/MLIR_graphs}"
RUN_ID="${RUN_ID:-historical_gmt_stage2_ab_20261112}"
OUT="${OUT:-$ROOT/artifacts/historical_gmt_stage2_ab/$RUN_ID}"
GMT_STEPS="${GMT_STEPS:-600}"
STEPS="${STEPS:-300}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SEEDS="${SEEDS:-123,456}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
STAGE2_EVAL="${STAGE2_EVAL:-50}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# No environment rewrite, no GPU sharing, and never touch the V100/V100S.
if ! command -v nvidia-smi >/dev/null; then echo 'nvidia-smi missing' >&2; exit 2; fi
export A30_UUID
A30_UUID="$($PYTHON - <<'PY'
import os,subprocess,sys
s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
selected=[]
for line in s.splitlines():
    f=[x.strip() for x in line.split(',')]
    if len(f)<4:continue
    if 'A30' in f[1] and (not os.environ.get('A30_GPU_UUID') or f[0]==os.environ['A30_GPU_UUID']):
        selected.append(f)
if len(selected)!=1:
    raise RuntimeError(f'Expected exactly one A30, or set A30_GPU_UUID: {selected}')
uid,name,free,total=selected[0]
if int(total)<23000 or int(free)<19000:
    raise RuntimeError(f'Expected nearly idle 24 GiB A30, got {name} ({free}/{total} MiB)')
print(uid)
PY
)"
procs="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null || true)"
if printf '%s\n' "$procs" | grep -F "$A30_UUID"; then
 echo "ERROR: compute job already using A30 $A30_UUID (refusing contention)" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$A30_UUID"
$PYTHON - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'A30' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print('A30 VERIFIED:',torch.cuda.get_device_name(0),'VRAM MiB',torch.cuda.get_device_properties(0).total_memory//2**20)
PY

if [[ ! -f "$PARENT/adapter_model.safetensors" ]]; then
 echo "Missing authenticated Stage-1 parent: $PARENT" >&2; exit 2
fi
PARENT_SHA="$(sha256sum "$PARENT/adapter_model.safetensors" | cut -d' ' -f1)"
if [[ "$PARENT_SHA" != '28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164' ]]; then
 echo "ERROR: wrong frozen V4 reference parent: $PARENT_SHA" >&2; exit 2
fi
if [[ ! -f "$ASSETS/Checkpoints/gnn_final_absolute_direct_rank_s123/run1/val_structural_rank_model_state_dict.pth" ]];then
 echo "ERROR: copy original epoch-9 historical GNN checkpoint to $ASSETS before this run" >&2; exit 2
fi
if [[ -e "$OUT" ]]; then echo "ERROR: $OUT already exists; use a fresh RUN_ID" >&2; exit 2; fi
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source_commit.txt"
if [[ -s "$HERE/audit_v4_action_filter.patch" ]]; then
  cp "$HERE/audit_v4_action_filter.patch" "$OUT/"
fi
"$PYTHON" -m unittest discover -s experiments/historical_mlir_gmt_gemv -p 'test_historical_gmt_slots.py' -v \
  2>&1 | tee "$OUT/historical_tests.log"

# Audit only; test data are not loaded. The same Stage-2 dataset is used in A/B.
"$PYTHON" -m experiments.rc3_stage2_multikernel.audit --data "$PARENT/../train.jsonl" --out "$OUT/independent_data_audit" \
  2>&1 | tee "$OUT/data_audit.log"
"$PYTHON" "$HERE/prepare_banks.py" \
  --dataset "$OUT/independent_data_audit/stage2_pilot.jsonl" \
  --split "$OUT/independent_data_audit/split.json" \
  --graphs "$GEXF" --assets "$ASSETS" \
  --out "$OUT/prepared" --steps "$GMT_STEPS" --seed 123 \
  2>&1 | tee "$OUT/prepare.log"

if [[ "$PREPARE_ONLY" == 1 ]];then
  echo "PREPARED (not trained Stage-2): $OUT/prepared/PREPARATION.json"; exit 0
fi

DATA="$OUT/prepared/stage2_matched.jsonl"
SPLIT="$OUT/prepared/stage2_matched_split.json"
# Early conservative A30 preflight: fail closed on likely truncation rather than
# silently dropping compiler-action placeholders during training.
"$PYTHON" "$HERE/check_tokens.py" --data "$DATA" --model deepseek-ai/deepseek-coder-6.7b-base \
  --max-length "$MAX_LENGTH" | tee "$OUT/token_lengths.log"

IFS=',' read -r -a seed_array <<< "$SEEDS"
for seed in "${seed_array[@]}";do
  if [[ ! "$seed" =~ ^[0-9]+$ ]];then echo "Invalid seed: $seed" >&2; exit 2; fi
  INIT="$OUT/initial_structural_state_s${seed}.json"
  for arm in A B;do
    DIR="$OUT/seed_${seed}/${arm}"
    BANK="$OUT/prepared/memory_${arm}"
    mkdir -p "$DIR"
    echo "===== START seed=$seed arm=$arm A30=$A30_UUID ====="
    "$PYTHON" -m LLM_branch.train.train_SFT_xattn_new \
      --run_mode single --stage2_output_dir "$DIR" --output_dir "$DIR" \
      --dataset "$DATA" --split_json "$SPLIT" --memory_dir "$BANK" \
      --init_adapter_dir "$PARENT" --directive_domain_registry_json "$PARENT/directive_domain_registry.json" \
      --research_allow_stage1_subset \
      --model deepseek-ai/deepseek-coder-6.7b-base \
      --model_revision ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912 \
      --objective PARETO_LATENCY --device_mode known --resource_budget_mode none \
      --directive_loss_weighting uniform --family_sampling_power 0 \
      --minimum_validation_families 2 --minimum_test_families 0 \
      --max_slots 64 --mem_dim 128 --every_n_layers 8 \
      --xattn_heads 4 --xattn_dim_head 64 --xattn_ff_mult 1 \
      --structural_fusion_placement post_self_attention_residual \
      --structural_routing compiler_relational \
      --initial_state_reference "$INIT" \
      --lr_lora 0 --lr_embed 0 --lr_xattn 1e-4 --lr_gate 2e-4 --lr_ff 0 --lr_gate_ff 0 \
      --max_steps "$STEPS" --epochs 1 --eval_steps "$STAGE2_EVAL" --save_steps "$STAGE2_EVAL" \
      --selection_eval_steps "$STAGE2_EVAL" --seed "$seed" \
      --batch_size 1 --grad_accum 4 --max_length "$MAX_LENGTH" \
      --gradient_checkpointing --num_workers 0 --candidate_loss_weight 0 \
      --selection_num_val_kernels 0 --selection_cases_per_kernel_device 4 \
      2>&1 | tee "$DIR/training_stdout.log"
    echo "===== COMPLETE seed=$seed arm=$arm ====="
  done
done
"$PYTHON" "$HERE/summarize.py" --run "$OUT" | tee "$OUT/FINISH_STATUS.txt"
echo "COMPLETE: $OUT"
