#!/usr/bin/env bash
# Sequential V100-only comparison; never promotes a production adapter.
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
PYTHON=/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python
GPU_UUID=GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb
STUDY=artifacts/diagnostics/stage1_conditional_study_20261101
PREPARED=artifacts/diagnostics/stage1_full_data_recovery_data
BASELINE=artifacts/diagnostics/stage1_conditional_ce_full_validation_20261101/report.json
REFERENCE=mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1
mkdir -p "$STUDY"
if [[ -e "$STUDY/status" ]]; then
  echo "Study status already exists; refusing an accidental second launch" >&2
  exit 2
fi
status() {
  printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$1" "$(git rev-parse HEAD)" > "$STUDY/status.tmp"
  mv "$STUDY/status.tmp" "$STUDY/status"
}
on_exit() {
  code=$?
  if (( code != 0 )); then status "FAILED_exit_${code}"; fi
}
trap on_exit EXIT
status PREFLIGHT
if [[ -n "${BASELINE_EVAL_PID:-}" ]]; then
  # The first supervised comparison must finish before any new training.
  for ((minute=0; minute<240; minute++)); do
    if ! kill -0 "$BASELINE_EVAL_PID" 2>/dev/null; then break; fi
    sleep 60
  done
fi
"$PYTHON" - "$BASELINE" "$PREPARED" <<'PY'
import json, pathlib, sys
from experiments.stage1_recovery.prepare import sha
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
manifest = json.loads((pathlib.Path(sys.argv[2]) / 'manifest.json').read_text())
assert report['cases'] == manifest['counts']['val'] == 383
assert len(report['adapters']) == 2
assert report['prepared_manifest_sha256'] == sha(pathlib.Path(sys.argv[2]) / 'manifest.json')
assert manifest['counts']['train'] == 2187
PY
baseline_measured="$STUDY/baseline_measured_quality"
if [[ -e "$baseline_measured" ]]; then
  echo "Refusing to overwrite $baseline_measured" >&2
  exit 4
fi
status BASELINE_MEASURED_QOR
"$PYTHON" -B -m experiments.stage1_recovery.evaluate --prepared "$PREPARED" \
  --measured-quality-predictions \
  artifacts/diagnostics/stage1_conditional_ce_full_validation_20261101/adapter_0_field_greedy.json \
  artifacts/diagnostics/stage1_conditional_ce_full_validation_20261101/adapter_1_field_greedy.json \
  --output "$baseline_measured" > "$STUDY/baseline_measured.log" 2>&1
"$PYTHON" -B -m experiments.stage1_recovery.finalize \
  --full-run mailohls_runs/stage1_full_data_ce_recovery_s123 --prepared "$PREPARED" \
  --full-validation-report "$BASELINE" > "$STUDY/baseline_assessment.log" 2>&1
"$PYTHON" - "$BASELINE" "$baseline_measured/report.json" "$STUDY/baseline_summary.json" <<'PY'
import json, pathlib, sys
decisions = json.loads(pathlib.Path(sys.argv[1]).read_text())
quality = json.loads(pathlib.Path(sys.argv[2]).read_text())
names = ('legal_constant', 'historical_stage1', 'optimized_legacy_ce')
decision_rows = [decisions['baseline'], *(decisions['adapters'][str(i)]['field_greedy'] for i in range(2))]
quality_rows = list(quality['checkpoints'].values())
assert len(decision_rows) == len(quality_rows) == 3
summary = dict(schema='stage1-full-validation-baseline-comparison-v1', cases=383,
               dataset_sha256=decisions['dataset_sha256'],
               prepared_manifest_sha256=decisions['prepared_manifest_sha256'],
               no_test_labels_used=not decisions['heldout_test_used'], models={})
for name, decision, measured in zip(names, decision_rows, quality_rows):
    summary['models'][name] = dict(
        core_balanced=decision.get('core_decision_balanced_accuracy'),
        complete_action=decision.get('complete_action_accuracy'),
        exact_design=decision.get('exact_design_accuracy'),
        positive_pipeline_recall=decision.get('positive_reference_value_recall', {}).get('PIPE'),
        positive_unroll_recall=decision.get('positive_reference_value_recall', {}).get('UNROLL'),
        exact_measured_matches=measured['exact_measured_match_count'],
        measured_feasible=measured['matched_feasible_count'],
        mean_measured_regret=measured['mean_measured_regret'])
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2), flush=True)
PY
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -Fq "$GPU_UUID"; then
  echo "V100S has an active compute process; refusing to share it" >&2
  exit 3
fi
export PYTHON CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OBJECTIVE=PARETO_LATENCY
export STAGE1_LR_LORA=1e-4 STAGE1_LR_EMBED=2e-5
export STAGE1_BUDGET_INVARIANT_MAX_DUPLICATES=-1
export STAGE1_DOMAIN_REGISTRY=mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json
export STAGE1_RECOVERY_PREPARED="$PREPARED" STAGE1_DECISION_DIAGNOSTICS=1
export STAGE1_CONTRAST_WEIGHT=1 STAGE1_FULL_VALIDATION=0
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert 'V100S' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
import bitsandbytes, peft, transformers
print('GPU', torch.cuda.get_device_name(0), 'torch', torch.__version__,
      'transformers', transformers.__version__, 'peft', peft.__version__,
      'bitsandbytes', bitsandbytes.__version__, flush=True)
PY
for policy in decision decision_balanced; do
  smoke="mailohls_runs/stage1_conditional_${policy}_smoke_s123"
  full="mailohls_runs/stage1_conditional_${policy}_s123"
  validation="$STUDY/${policy}_full_validation"
  measured="$STUDY/${policy}_measured_quality"
  for path in "$smoke" "$full" "$validation" "$measured"; do
    if [[ -e "$path" ]]; then
      echo "Refusing to overwrite $path" >&2
      exit 4
    fi
  done
  status "SMOKE_${policy}"
  STAGE1_SUPERVISION_POLICY="$policy" STAGE1_MAX_STEPS=2 STAGE1_EVAL_ON_START=0 \
    OUT="$smoke" bash scripts/training/train_stage1.sh > "$STUDY/${policy}_smoke.log" 2>&1
  status "TRAIN_${policy}"
  STAGE1_SUPERVISION_POLICY="$policy" OUT="$full" \
    bash scripts/training/train_stage1.sh > "$STUDY/${policy}_train.log" 2>&1
  test -f "$full/adapter_model.safetensors"
  test -f "$full/best_custom_stage1/adapter_model.safetensors"
  status "VALIDATE_${policy}"
  "$PYTHON" -B -m experiments.stage1_recovery.evaluate --prepared "$PREPARED" \
    --full-validation --adapters "$REFERENCE" "$full/best_custom_stage1" \
    --output "$validation" > "$STUDY/${policy}_validation.log" 2>&1
  "$PYTHON" -B -m experiments.stage1_recovery.evaluate --prepared "$PREPARED" \
    --measured-quality-predictions "$validation/adapter_0_field_greedy.json" \
    "$validation/adapter_1_field_greedy.json" --output "$measured" \
    > "$STUDY/${policy}_measured.log" 2>&1
  "$PYTHON" -B -m experiments.stage1_recovery.finalize --full-run "$full" \
    --prepared "$PREPARED" --full-validation-report "$validation/report.json" \
    > "$STUDY/${policy}_assessment.log" 2>&1
  status "COMPLETE_${policy}"
done
status COMPLETE_BOTH
