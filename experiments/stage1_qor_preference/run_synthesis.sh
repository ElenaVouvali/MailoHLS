#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE="$ROOT/artifacts/stage1_qor_preference/v1_20261106/synthesis_bundle"
HOST=bonus24@coroni.microlab.ntua.gr
REMOTE=/home/bonus24/stage1_qor_dpo_latency_v1_20261106
test -s "$BUNDLE/manifest.json"
ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" 'if pgrep -x vitis_hls >/dev/null || pgrep -x vivado_hls >/dev/null; then echo "HLS_HOST_BUSY: existing synthesis preserved" >&2; exit 20; fi; test ! -e /home/bonus24/stage1_qor_dpo_latency_v1_20261106'
scp -r "$BUNDLE" "$HOST:$REMOTE"
ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" 'python3 /home/bonus24/stage1_qor_dpo_latency_v1_20261106/run_batch.py --bundle /home/bonus24/stage1_qor_dpo_latency_v1_20261106 --output /home/bonus24/stage1_qor_dpo_latency_v1_20261106/output --tool /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls --timeout 300'
