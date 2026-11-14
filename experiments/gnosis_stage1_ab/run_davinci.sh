#!/usr/bin/env bash
# Launch the GNOSIS Stage-1 A/B run on the authenticated Davinci V100.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

REMOTE=${DAVINCI_SSH_TARGET:-evouvali@davinci.microlab.ntua.gr}
GPU_UUID=${DAVINCI_V100_UUID:-GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f}
RUN_ID=${RUN_ID:-gnosis_film_ab_20261112_v1}
ATTEMPT_ID=${DAVINCI_ATTEMPT_ID:-$(date +%Y%m%d_%H%M%S)}
REMOTE_HOME="$(ssh -o BatchMode=yes -o ConnectTimeout=12 "$REMOTE" 'printf %s "$HOME"')"
[[ "$REMOTE_HOME" == /home/evouvali ]] || { echo "Unexpected Davinci home: $REMOTE_HOME" >&2; exit 1; }
REMOTE_ROOT="$REMOTE_HOME/MailoHLS/worktrees/gnosis-stage1-film-ab-davinci/$RUN_ID/$ATTEMPT_ID"
LOCAL_OUTPUT="$ROOT/artifacts/gnosis_stage1_ab/$RUN_ID"
REUSE_REBUILD=1 # Davinci always verifies and reuses the dataset transferred below.
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe RUN_ID: $RUN_ID" >&2; exit 1; }
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe DAVINCI_ATTEMPT_ID: $ATTEMPT_ID" >&2; exit 1; }
[[ "$GPU_UUID" == GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f ]] || {
  echo "Refusing a noncanonical Davinci GPU UUID: $GPU_UUID" >&2; exit 1;
}
[[ -d "$SCRIPT_DIR" && -s Data/manifest.json ]] || { echo "Missing source package or Data/manifest.json" >&2; exit 1; }
[[ -d "$LOCAL_OUTPUT/gnosis_rebuilt" ]] || {
  echo "Build and authenticate the non-test dataset first: $LOCAL_OUTPUT/gnosis_rebuilt" >&2; exit 1;
}
REUSE_BANK=0
if [[ -e "$LOCAL_OUTPUT/bank" ]]; then
  [[ -s "$LOCAL_OUTPUT/bank/manifest.json" ]] || { echo "Existing local bank lacks a manifest; inspect it before launching" >&2; exit 1; }
  REUSE_BANK=1
fi

# This preflight happens on Davinci before any source transfer or training.
ssh -o BatchMode=yes -o ConnectTimeout=12 "$REMOTE" bash -s -- "$GPU_UUID" <<'REMOTE_PREFLIGHT'
set -euo pipefail
uuid=$1
[[ "$(hostname -s)" == davinci ]] || { echo "Remote host is not Davinci" >&2; exit 1; }
row=$(nvidia-smi -i "$uuid" --query-gpu=uuid,name,memory.used,memory.total --format=csv,noheader,nounits)
IFS=, read -r seen name used total <<<"$row"
seen=$(xargs <<<"$seen"); name=$(xargs <<<"$name")
used=$(xargs <<<"$used"); total=$(xargs <<<"$total")
[[ "$seen" == "$uuid" && "$name" == "Tesla V100-PCIE-32GB" && "$total" -ge 30000 ]] || {
  echo "Davinci GPU identity mismatch: $row" >&2; exit 1;
}
pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | awk -F, -v u="$uuid" 'index($1,u)==1 {print $2}')
if [[ -n "$pids" || "$used" -gt 2048 ]]; then
  echo "Davinci V100 is occupied (${used} MiB, PIDs: ${pids:-none}); no job launched" >&2; exit 1
fi
python="$HOME/MailoHLS/.venv/bin/python"
parent="$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run"
[[ -x "$python" && -s "$parent/adapter_model.safetensors" && -s "$parent/training_contract.json" ]] || {
  echo "Davinci Python environment or authenticated V4 parent is missing" >&2; exit 1;
}
"$python" - "$parent/training_contract.json" <<'PY'
import json, sys
import torch, transformers, peft, bitsandbytes
from transformers import AutoConfig
assert torch.cuda.is_available()
contract=json.load(open(sys.argv[1]))
AutoConfig.from_pretrained(contract['model'], revision=contract['model_revision'], local_files_only=True)
print('DAVINCI_RUNTIME_OK', torch.__version__, transformers.__version__, peft.__version__)
PY
echo "DAVINCI_PREFLIGHT_OK $uuid $name ${used}/${total} MiB"
REMOTE_PREFLIGHT

# Preserve and clear only the known failed first-step output: its saved
# checkpoint is from step 0, and its training log is empty. Any later progress
# or output with a different shape is left untouched.
if [[ -e "$LOCAL_OUTPUT/A" ]]; then
  if python - "$LOCAL_OUTPUT/A" <<'PY_STEP0'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); c=r/'checkpoint-0000'
allowed={'eval_0000.json','best.json','protocol.json','training_context_coverage.json','training.jsonl','checkpoint-0000'}
try:
    assert {p.name for p in r.iterdir()}==allowed
    assert json.loads((r/'best.json').read_text())['step']==0
    assert json.loads((c/'receipt.json').read_text())['step']==0
    assert (r/'training.jsonl').stat().st_size==0
    assert not (r/'completed.json').exists()
    assert all((c/p).is_file() for p in ('progress.pt','receipt.json','device_adapt/adapter_model.safetensors'))
except (AssertionError,KeyError,ValueError,OSError):
    raise SystemExit(1)
PY_STEP0
  then
    FAILED_A="$LOCAL_OUTPUT/A_failed_step0_$(date +%Y%m%d_%H%M%S)"
    [[ ! -e "$FAILED_A" ]] || { echo "Refusing to overwrite $FAILED_A" >&2; exit 1; }
    mv -- "$LOCAL_OUTPUT/A" "$FAILED_A"
    echo "Preserved failed first-step output as $FAILED_A"
  else
    echo "Refusing existing A output with possible training progress: inspect $LOCAL_OUTPUT/A" >&2; exit 1
  fi
fi
[[ ! -e "$LOCAL_OUTPUT/B" ]] || {
  echo "Refusing existing B output: inspect $LOCAL_OUTPUT/B before launching" >&2; exit 1;
}

# Transfer only the executable source set and the already-built non-test data.
ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$REMOTE_ROOT/LLM_branch' '$REMOTE_ROOT/Preprocessing' '$REMOTE_ROOT/experiments/gnosis_stage1_ab' '$REMOTE_ROOT/experiments/final_architecture_v3' '$REMOTE_ROOT/Data' '$REMOTE_ROOT/artifacts/gnosis_stage1_ab/$RUN_ID'"
rsync -a "$ROOT/LLM_branch/" "$REMOTE:$REMOTE_ROOT/LLM_branch/"
rsync -a "$ROOT/Preprocessing/" "$REMOTE:$REMOTE_ROOT/Preprocessing/"
rsync -a "$SCRIPT_DIR/" "$REMOTE:$REMOTE_ROOT/experiments/gnosis_stage1_ab/"
rsync -a "$ROOT/experiments/final_architecture_v3/" "$REMOTE:$REMOTE_ROOT/experiments/final_architecture_v3/"
rsync -a "$ROOT/Data/manifest.json" "$REMOTE:$REMOTE_ROOT/Data/manifest.json"
rsync -a "$LOCAL_OUTPUT/gnosis_rebuilt/" "$REMOTE:$REMOTE_ROOT/artifacts/gnosis_stage1_ab/$RUN_ID/gnosis_rebuilt/"
if [[ "$REUSE_BANK" == 1 ]]; then
  rsync -a "$LOCAL_OUTPUT/bank/" "$REMOTE:$REMOTE_ROOT/artifacts/gnosis_stage1_ab/$RUN_ID/bank/"
fi

set +e
ssh -o BatchMode=yes "$REMOTE" bash -s -- \
  "$REMOTE_ROOT" "$RUN_ID" "$GPU_UUID" "$REUSE_REBUILD" \
  "$REUSE_BANK" \
  "${STEPS:-720}" "${CONTEXTS:-240}" "${MAX_LENGTH:-6144}" \
  "${EVAL_EVERY:-60}" "${VAL_SFT:-32}" "${VAL_PAIRS:-24}" <<'REMOTE_RUN'
set -euo pipefail
root=$1; run_id=$2; uuid=$3; reuse=$4; reuse_bank=$5; steps=$6; contexts=$7; max_length=$8; eval_every=$9; val_sft=${10}; val_pairs=${11}
cd "$root"
export GNOSIS_REMOTE_EXEC=1 DAVINCI_V100_UUID="$uuid" RUN_ID="$run_id"
export REUSE_REBUILD="$reuse" REUSE_BANK="$reuse_bank" STEPS="$steps" CONTEXTS="$contexts" MAX_LENGTH="$max_length"
export EVAL_EVERY="$eval_every" VAL_SFT="$val_sft" VAL_PAIRS="$val_pairs"
export PYTHON="$HOME/MailoHLS/.venv/bin/python"
export PARENT="$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run"
bash experiments/gnosis_stage1_ab/run_v100s.sh
REMOTE_RUN
status=$?
set -e

# Return logs and results to Liono without touching unrelated output directories.
mkdir -p "$LOCAL_OUTPUT"
rsync -a --ignore-existing "$REMOTE:$REMOTE_ROOT/artifacts/gnosis_stage1_ab/$RUN_ID/" "$LOCAL_OUTPUT/"
exit "$status"
