#!/usr/bin/env bash
# Safe CPU-only handoff for Stage-1 vNext. Does not train, synthesize, or modify drivers.
set -euo pipefail

if [[ "${1:-}" != "" && "${1:-}" != "--prepare" ]]; then
  echo "Usage: bash scripts/research/stage1_quality_cpu_gate.sh [--prepare]" >&2
  exit 64
fi
cd "$(git rev-parse --show-toplevel)"
if [[ "$(git branch --show-current)" != "research/stage1-refinement" && \
      "$(git branch --show-current)" != research/stage1-quality-sft-wip-* ]]; then
  echo "Wrong branch: $(git branch --show-current). Refusing to run." >&2
  exit 2
fi
PY="${PYTHON:-/home/elvouvali/miniforge3/envs/mailohls-llm-v2/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Documented Python is unavailable at $PY; set PYTHON=/path/to/python" >&2
  exit 3
fi
[[ -f experiments/stage1_quality_sft/__main__.py ]] || { echo 'Missing quality SFT package'; exit 4; }
[[ -f LLM_branch/tests/test_stage1_quality_sft.py ]] || { echo 'Missing quality SFT tests'; exit 4; }

mkdir -p "$HOME/stage1-vnext-backups"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$HOME/stage1-vnext-backups"
git status --short --branch | tee "$BACKUP/status-$STAMP.txt"
git diff --binary > "$BACKUP/unstaged-$STAMP.patch"
git diff --cached --binary > "$BACKUP/staged-$STAMP.patch"
git ls-files --others --exclude-standard -z | tar --null -czf "$BACKUP/untracked-$STAMP.tar.gz" -T -
echo "WIP backup saved in $BACKUP (stamp: $STAMP)"

"$PY" -m py_compile experiments/stage1_quality_sft/*.py \
  LLM_branch/tests/test_stage1_quality_sft.py
"$PY" -m unittest LLM_branch.tests.test_stage1_quality_sft -v

if [[ "${1:-}" == "--prepare" ]]; then
  PREP="/tmp/stage1_quality_sft_verified_$STAMP"
  echo "Creating NEW prepared result: $PREP"
  "$PY" -m experiments.stage1_quality_sft prepare --output "$PREP"
else
  PREP="${STAGE1_PREPARED:-/tmp/stage1_quality_sft_v1}"
fi

"$PY" scripts/research/stage1_quality_readiness.py \
  --root . --prepared "$PREP" --strict || {
    echo "BLOCKED: fix readiness findings before any GPU training." >&2
    exit 10
  }
echo "CPU handoff checks passed. Still require a real trainer-path integration test."
