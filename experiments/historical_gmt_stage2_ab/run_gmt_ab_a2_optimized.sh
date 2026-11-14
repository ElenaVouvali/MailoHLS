#!/usr/bin/env bash
# Isolated, matched historical-GNN-versus-GMT Stage-2 A/B on Liono A2.
# Usage: bash run_gmt_ab_a2_optimized.sh smoke|full
# Never edits the maintained Python trainer, original launcher, or an existing run.
set -euo pipefail

MODE="${1:-}"
ROOT=/home/elvouvali/MailoHLS/worktrees/historical-mlir-gmt-gemv
ASSETS=/home/elvouvali/MailoHLS
HERE="$ROOT/experiments/historical_gmt_stage2_ab"
PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
CPU_PYTHON=/home/elvouvali/.hls-llm/bin/python
PARENT=/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run
A2_UUID=GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35
GEXF="$ROOT/GNN_branch/MLIR_graphs"
OUTPUT_BASE="$ROOT/artifacts/historical_gmt_stage2_ab"

case "$MODE" in
  smoke)
    RUN_ID=historical_gmt_ab_a2_smoke_fast_20261112
    STEPS=2
    GMT_STEPS=40
    SEEDS=123
    EVAL_EVERY=2
    CASE_CAP=1
    EVAL_START_FLAG=--no-eval_on_start
    ;;
  full)
    RUN_ID=historical_gmt_ab_a2_full_no_validation_20261112
    STEPS=300
    GMT_STEPS=600
    SEEDS=123,456
    EVAL_EVERY=150
    CASE_CAP=4
    EVAL_START_FLAG=--no-eval_on_start
    ;;
  *)
    echo "Usage: bash $0 smoke|full" >&2
    exit 2
    ;;
esac

cd "$ROOT"
if [[ ! -f experiments/historical_mlir_gmt_gemv/historical_gmt_slots.py || \
      ! -f LLM_branch/train/train_SFT_xattn_new.py ]]; then
  echo "ERROR: expected historical research worktree is incomplete." >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -x "$CPU_PYTHON" ]]; then
  echo "ERROR: required research Python interpreters are unavailable." >&2
  exit 2
fi

mkdir -p "$OUTPUT_BASE"
# Prevent two copies of the optimized launcher from overlapping even while
# the GPU is temporarily idle during CPU-only preparation/audit.
exec 9>"$OUTPUT_BASE/.optimized_a2_launcher.lock"
flock -n 9 || { echo "ERROR: another optimized A2 launcher is running." >&2; exit 2; }

OUT="$OUTPUT_BASE/$RUN_ID"
if [[ -e "$OUT" ]]; then
  echo "ERROR: output already exists: $OUT. Never overwrite a prior experiment." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$A2_UUID"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mailohls-mpl}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A2-only, fail-closed preflight. Never touch the independent V100S process.
"$PYTHON" - <<'PY'
import os, subprocess
uid = os.environ['CUDA_VISIBLE_DEVICES']
rows = subprocess.check_output(
    ['nvidia-smi','--query-gpu=uuid,name,memory.free',
     '--format=csv,noheader,nounits'], text=True)
matches = [[x.strip() for x in row.split(',')]
           for row in rows.splitlines() if row.split(',')[0].strip() == uid]
if len(matches) != 1 or not matches[0][1].endswith('A2'):
    raise RuntimeError(f'Configured A2 GPU not found: {matches}')
if int(matches[0][2]) < 14000:
    raise RuntimeError(f'A2 is not sufficiently free: {matches[0]}')
processes = subprocess.check_output(
    ['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],
    text=True)
if any(row.split(',')[0].strip() == uid for row in processes.splitlines()):
    raise RuntimeError('A2 has an active compute process: do not double-launch')
print('A2 preflight passed:', matches[0], flush=True)
PY

[[ -f "$PARENT/adapter_model.safetensors" ]] || {
  echo "ERROR: authenticated V4 parent checkpoint missing" >&2; exit 2;
}
ACTUAL_SHA=$(sha256sum "$PARENT/adapter_model.safetensors" | awk '{print $1}')
EXPECTED_SHA=28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || {
  echo "ERROR: V4 parent checksum mismatch: $ACTUAL_SHA" >&2; exit 2;
}
[[ -f "$ASSETS/Checkpoints/gnn_final_absolute_direct_rank_s123/run1/val_structural_rank_model_state_dict.pth" ]] || {
  echo "ERROR: historical epoch-9 GNN checkpoint missing" >&2; exit 2;
}

mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source_commit.txt"
git status --short > "$OUT/source_git_status.txt"
git diff --binary > "$OUT/source_dirty_diff.patch"
{
  printf 'mode=%s\nrun_id=%s\ngpu_uuid=%s\nsteps=%s\ngmt_steps=%s\n' \
    "$MODE" "$RUN_ID" "$A2_UUID" "$STEPS" "$GMT_STEPS"
  printf 'seeds=%s\neval_every=%s\ncase_cap=%s\neval_on_start=%s\n' \
    "$SEEDS" "$EVAL_EVERY" "$CASE_CAP" "$EVAL_START_FLAG"
  printf 'early_stopping_patience=0\ncontext_max_length=7168\nvalidation_skipped=true\n'
} > "$OUT/optimized_protocol.txt"
if [[ -s "$HERE/audit_v4_action_filter.patch" ]]; then
  cp "$HERE/audit_v4_action_filter.patch" "$OUT/"
fi

"$CPU_PYTHON" -m unittest discover \
  -s experiments/historical_mlir_gmt_gemv \
  -p test_historical_gmt_slots.py -v \
  2>&1 | tee "$OUT/historical_tests.log"

"$CPU_PYTHON" -m experiments.rc3_stage2_multikernel.audit \
  --data "$PARENT/../train.jsonl" \
  --out "$OUT/independent_data_audit" \
  2>&1 | tee "$OUT/data_audit.log"

"$CPU_PYTHON" "$HERE/prepare_banks.py" \
  --dataset "$OUT/independent_data_audit/stage2_pilot.jsonl" \
  --split "$OUT/independent_data_audit/split.json" \
  --graphs "$GEXF" --assets "$ASSETS" \
  --out "$OUT/prepared" --steps "$GMT_STEPS" --seed 123 \
  2>&1 | tee "$OUT/prepare.log"

DATA="$OUT/prepared/stage2_matched.jsonl"
SPLIT="$OUT/prepared/stage2_matched_split.json"
"$PYTHON" "$HERE/check_tokens.py" \
  --data "$DATA" --model deepseek-ai/deepseek-coder-6.7b-base \
  --max-length 7168 | tee "$OUT/token_lengths.log"

IFS=',' read -r -a seed_array <<< "$SEEDS"
for seed in "${seed_array[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
  INIT="$OUT/initial_structural_state_s${seed}.json"
  for arm in A B; do
    DIR="$OUT/seed_${seed}/${arm}"
    BANK="$OUT/prepared/memory_${arm}"
    mkdir -p "$DIR"
    echo "===== $(date -u +%FT%TZ) START mode=$MODE seed=$seed arm=$arm ====="
    "$PYTHON" -m LLM_branch.train.train_SFT_xattn_new \
      --run_mode single --stage2_output_dir "$DIR" --output_dir "$DIR" \
      --dataset "$DATA" --split_json "$SPLIT" --memory_dir "$BANK" \
      --init_adapter_dir "$PARENT" \
      --directive_domain_registry_json "$PARENT/directive_domain_registry.json" \
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
      --lr_lora 0 --lr_embed 0 --lr_xattn 1e-4 --lr_gate 2e-4 \
      --lr_ff 0 --lr_gate_ff 0 \
      --max_steps "$STEPS" --epochs 1 \
      --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" \
      --selection_eval_steps "$EVAL_EVERY" --seed "$seed" \
      --batch_size 1 --grad_accum 4 --max_length 7168 \
      --gradient_checkpointing --num_workers 0 --candidate_loss_weight 0 \
      --selection_num_val_kernels 0 \
      --selection_cases_per_kernel_device "$CASE_CAP" \
      --early_stopping_patience 0 --skip_validation "$EVAL_START_FLAG" \
      2>&1 | tee "$DIR/training_stdout.log"
    echo "===== $(date -u +%FT%TZ) COMPLETE seed=$seed arm=$arm ====="
  done
done

"$PYTHON" - "$OUT" "$SEEDS" <<'PY_VERIFY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
arms = []
for seed in sys.argv[2].split(','):
    for arm in ('A', 'B'):
        directory = out / f'seed_{seed}' / arm
        for filename in ('adapter_model.safetensors', 'structural_xattn.pt'):
            if not (directory / filename).is_file():
                raise SystemExit(f'Missing final checkpoint: {directory / filename}')
        arms.append({'seed': seed, 'arm': arm, 'status': 'TRAINING_COMPLETE'})
report = {'arms': arms, 'validation_skipped': True,
          'decision': 'NOT_EVALUATED', 'paired_differences': []}
(out / 'architecture_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
(out / 'FINISH_STATUS.txt').write_text('TRAINING_COMPLETE; validation skipped; no A/B decision\n')
print('Verified final checkpoints for all arms; validation skipped.')
PY_VERIFY
touch "$OUT/SUCCESS"
echo "COMPLETE: $OUT"
