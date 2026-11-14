#!/usr/bin/env bash
# Run inside MailoHLS worktree. No pre-existing outputs are overwritten.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${GNOSIS_REMOTE_EXEC:-0}" == 1 ]]; then
  REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif REPOSITORY_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPOSITORY_ROOT"
EXPERIMENT=experiments/gnosis_stage1_ab
# Keep the historical command safe for users who launch it on Liono: it now
# stages the small source/data bundle and executes on Davinci's pinned V100.
if [[ "${GNOSIS_REMOTE_EXEC:-0}" != 1 ]]; then
  exec bash "$EXPERIMENT/run_davinci.sh" "$@"
fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=${PYTHON:-$HOME/MailoHLS/.venv/bin/python}
RUN_ID=${RUN_ID:-gnosis_film_ab_20261112_v1}
OUTPUT="$PWD/artifacts/gnosis_stage1_ab/$RUN_ID"
REUSE_REBUILD=${REUSE_REBUILD:-0} # Retry this failed smoke without rebuilding 90,908 measurements
REUSE_BANK=${REUSE_BANK:-0}       # Reuse only a bank whose input/output hashes verify below
# The exact authenticated V4 Stage-1 checkpoint used in final-architecture GAN experiments.
# Do NOT fall back to stage1_final_final_latency_s123 or to a different adapter.
PARENT=${PARENT:-$HOME/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run}
EXPECTED_PARENT_SHA256=28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164
REGISTRY=${REGISTRY:-$PARENT/directive_domain_registry.json}
[[ -f "$REGISTRY" ]] || REGISTRY=$(dirname "$PARENT")/directive_domain_registry.json
STEPS=${STEPS:-720}
CONTEXTS=${CONTEXTS:-240}
MAX_LENGTH=${MAX_LENGTH:-6144}
EVAL_EVERY=${EVAL_EVERY:-60}
VAL_SFT=${VAL_SFT:-32}
VAL_PAIRS=${VAL_PAIRS:-24}
[[ -x "$PYTHON" ]] || { echo "Python env missing: $PYTHON" >&2; exit 1; }
[[ "$(hostname -s)" == davinci ]] || { echo "Refusing training outside Davinci (host=$(hostname -s))" >&2; exit 1; }
[[ -s "$PARENT/adapter_model.safetensors" && -s "$PARENT/training_contract.json" ]] || {
    echo "Missing authenticated V4 Stage-1 checkpoint at: $PARENT" >&2;exit 1;
}
ACTUAL_PARENT_SHA256="$(sha256sum "$PARENT/adapter_model.safetensors" | awk '{print $1}')"
if [[ "$ACTUAL_PARENT_SHA256" != "$EXPECTED_PARENT_SHA256" ]]; then
    echo "Wrong Stage-1 adapter at $PARENT: $ACTUAL_PARENT_SHA256" >&2; exit 1
fi
[[ -s "$REGISTRY" ]] || { echo "V4 domain registry missing: $REGISTRY" >&2; exit 1; }
echo "Using authenticated final-architecture V4 Stage-1: $PARENT (SHA256 $ACTUAL_PARENT_SHA256)"
# Verify the intended remote GPU before tests, dataset preparation, or writes.
GPU_UUID=${DAVINCI_V100_UUID:-GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f}
verify_target_gpu() {
  local row seen name used total pids
  row="$(nvidia-smi -i "$GPU_UUID" --query-gpu=uuid,name,memory.used,memory.total --format=csv,noheader,nounits)"
  IFS=, read -r seen name used total <<<"$row"
  seen="$(xargs <<<"$seen")"; name="$(xargs <<<"$name")"
  used="$(xargs <<<"$used")"; total="$(xargs <<<"$total")"
  [[ "$seen" == "$GPU_UUID" && "$name" == "Tesla V100-PCIE-32GB" && "$total" -ge 30000 ]] || {
    echo "Davinci target mismatch: $row" >&2; return 1;
  }
  pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | awk -F, -v u="$GPU_UUID" 'index($1,u)==1 {print $2}')"
  if [[ -n "$pids" || "$used" -gt 2048 ]]; then
    echo "Davinci V100 is occupied (${used} MiB, PIDs: ${pids:-none}); stop without interfering" >&2; return 1
  fi
  echo "Verified Davinci Tesla V100: $GPU_UUID (${used}/${total} MiB used)"
}
verify_target_gpu
if [[ "$REUSE_REBUILD" == 1 ]]; then
  [[ -d "$OUTPUT/gnosis_rebuilt" && -s "$OUTPUT/gnosis_rebuilt/mailohls_sft.jsonl" \
     && -s "$OUTPUT/gnosis_rebuilt/build_receipt.json" \
     && -s "$OUTPUT/gnosis_rebuilt/split_train_val.json" ]] || {
    echo "No complete earlier dataset build to reuse: $OUTPUT" >&2;exit 1;
  }
  [[ ( "$REUSE_BANK" == 1 || ! -e "$OUTPUT/bank" ) && ! -e "$OUTPUT/A" && ! -e "$OUTPUT/B" ]] || {
    echo "Refusing reuse: bank or model outputs already exist; investigate manually" >&2;exit 1;
  }
  # Guard against a stale or corrupted reused dataset without re-reading test.
  "$PYTHON" - "$OUTPUT/gnosis_rebuilt" <<'PYVERIFY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); x=json.loads((r/'build_receipt.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
for filename,field in [('mailohls_sft.jsonl','dataset_sha256'),('split_train_val.json','split_sha256')]:
    assert sha(r/filename)==x[field],(filename,'rebuild receipt mismatch')
print('REUSED_VERIFIED_NONTEST_DATA',x['included_kernels'],x['train_rows'],x['val_rows'])
PYVERIFY
else
  [[ ! -e "$OUTPUT" ]] || { echo "RUN_ID already exists: $OUTPUT" >&2;exit 1; }
fi
"$PYTHON" -m unittest discover -s "$EXPERIMENT" -p 'test_*.py' -v
if [[ "$REUSE_REBUILD" != 1 ]]; then
  mkdir -p "$OUTPUT"
  "$PYTHON" "$EXPERIMENT/rebuild_non_test.py" --out "$OUTPUT/gnosis_rebuilt" \
    --canonical-manifest Data/manifest.json
fi
if [[ "$REUSE_BANK" == 1 ]]; then
  [[ -s "$OUTPUT/bank/manifest.json" ]] || { echo "No complete prepared bank to reuse: $OUTPUT/bank" >&2; exit 1; }
  "$PYTHON" - "$OUTPUT" "$REGISTRY" <<'PYBANK'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); bank=root/'bank'; m=json.loads((bank/'manifest.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
inputs=[root/'gnosis_rebuilt/mailohls_sft.jsonl',root/'gnosis_rebuilt/mailohls_sft.manifest.json',
        root/'gnosis_rebuilt/mailohls_sft.sources.json',root/'gnosis_rebuilt/split_train_val.json',
        Path('Data/manifest.json'),Path(sys.argv[2])]
assert set(m['inputs_sha256'].values())=={sha(p) for p in inputs},'prepared bank input hashes do not match this run'
assert set(m['outputs_sha256'])=={'train_pairs.jsonl','train_sft.jsonl','val_pairs.jsonl','val_sft.jsonl'}
for name,digest in m['outputs_sha256'].items(): assert sha(bank/name)==digest,(name,'bank output hash mismatch')
print('REUSED_VERIFIED_BANK',m['counts'])
PYBANK
else
  "$PYTHON" "$EXPERIMENT/prepare_gnosis_ab.py" \
    --dataset "$OUTPUT/gnosis_rebuilt/mailohls_sft.jsonl" \
    --split "$OUTPUT/gnosis_rebuilt/split_train_val.json" \
    --canonical-manifest Data/manifest.json --registry "$REGISTRY" \
    --out "$OUTPUT/bank" --top-k 3 --max-budgets 10 --max-pairs-per-kernel 8
fi
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export TOKENIZERS_PARALLELISM=false
verify_target_gpu
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
p=torch.cuda.get_device_properties(0)
assert 'V100' in p.name.upper() and p.total_memory/2**30>=30,(p.name,p.total_memory)
assert torch.cuda.get_device_capability(0)[0]==7
assert p.name=='Tesla V100-PCIE-32GB',p.name
print('VERIFIED DAVINCI V100',p.name,p.total_memory)
PY
mkdir -p "$OUTPUT/logs"
for ARM in A B; do
  echo "=== Starting matched training arm $ARM ==="
  "$PYTHON" "$EXPERIMENT/train_gnosis_ab.py" \
    --parent "$PARENT" --bank "$OUTPUT/bank" \
    --out "$OUTPUT/$ARM" --arm "$ARM" \
    --steps "$STEPS" --train-contexts "$CONTEXTS" \
    --max-length "$MAX_LENGTH" --eval-every "$EVAL_EVERY" \
    --val-sft "$VAL_SFT" --val-pairs "$VAL_PAIRS" \
    --seed 123 --lr 8e-5 --film-lr 3e-4 --rank-budget .2 --rank-clock .05 \
    2>&1 | tee "$OUTPUT/logs/$ARM.log"
done
"$PYTHON" "$EXPERIMENT/compare_ab.py" --A "$OUTPUT/A" --B "$OUTPUT/B" \
  --output "$OUTPUT/architecture_comparison.json"
echo "COMPLETE: $OUTPUT/architecture_comparison.json"
