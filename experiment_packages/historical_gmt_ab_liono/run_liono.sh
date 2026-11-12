#!/usr/bin/env bash
# Matched Stage-2 A/B on Liono's explicitly selected A2, with original historical MLIR GNN
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
PYTHON="${PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
CPU_PYTHON="${CPU_PYTHON:-/home/elvouvali/.hls-llm/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mailohls-mpl}"
PARENT="${PARENT:-$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run}"
ASSETS="${HISTORICAL_ASSETS_ROOT:-$HOME/MailoHLS}"
GEXF="${GEXF_DIR:-$ROOT/GNN_branch/MLIR_graphs}"
RUN_ID="${RUN_ID:-historical_gmt_stage2_ab_20261112}"
OUT="${OUT:-$ROOT/artifacts/historical_gmt_stage2_ab/$RUN_ID}"
GMT_STEPS="${GMT_STEPS:-600}"
STEPS="${STEPS:-300}"
MAX_LENGTH="${MAX_LENGTH:-7168}"
SEEDS="${SEEDS:-123,456}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
STAGE2_EVAL="${STAGE2_EVAL:-50}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# UUID-scoped idle check. This package never connects to Davinci.
export CUDA_VISIBLE_DEVICES="${LIONO_GPU_UUID:-GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35}"
"$PYTHON" - <<'GPUCHECK'
import os,subprocess
uid=os.environ['CUDA_VISIBLE_DEVICES']
rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True)
matched=[x.split(',') for x in rows.splitlines() if x.split(',')[0].strip()==uid]
assert len(matched)==1, 'Selected GPU missing'
f=[x.strip() for x in matched[0]]
assert 'A2'==f[1].split()[-1], f'Expected Liono A2: {f}'
assert int(f[2])>=14000, f'A2 has insufficient free memory: {f}'
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert not any(x.split(',')[0].strip()==uid for x in procs.splitlines()), 'A2 already has a compute job'
print('LIONO GPU VERIFIED:',f,flush=True)
GPUCHECK

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
"$CPU_PYTHON" -m unittest discover -s experiments/historical_mlir_gmt_gemv -p 'test_historical_gmt_slots.py' -v \
  2>&1 | tee "$OUT/historical_tests.log"

# Audit only; test data are not loaded. The same Stage-2 dataset is used in A/B.
"$CPU_PYTHON" -m experiments.rc3_stage2_multikernel.audit --data "$PARENT/../train.jsonl" --out "$OUT/independent_data_audit" \
  2>&1 | tee "$OUT/data_audit.log"
"$CPU_PYTHON" "$HERE/prepare_banks.py" \
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
# Early conservative A2 preflight: fail closed on likely truncation rather than
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
    echo "===== START seed=$seed arm=$arm GPU=$CUDA_VISIBLE_DEVICES ====="
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
