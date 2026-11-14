#!/usr/bin/env bash
# Copy the frozen nine-workload evaluation sources into the layout required by stage2_infer.py.
# Source/action metadata comes only from the pinned repository's evaluation_set.
set -euo pipefail
ROOT=${1:?Usage: bash prepare_stage2_eval_sources.sh MAILOHLS_WORKTREE OUTPUT_DIRECTORY}
OUT=${2:?Usage: bash prepare_stage2_eval_sources.sh MAILOHLS_WORKTREE OUTPUT_DIRECTORY}
ROOT=$(realpath "$ROOT")
[[ -d "$ROOT/evaluation_set" ]] || { echo "Missing evaluation_set at $ROOT" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing source staging directory: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

copy_kernel() {
  local name=$1 src=$2 meta=$3
  [[ -f "$src" && -f "$meta" ]] || { echo "Missing source/metadata for $name: $src / $meta" >&2; exit 2; }
  local ext=${src##*.} dst="$OUT/$name"
  mkdir "$dst"
  cp -- "$src" "$dst/source.$ext"
  cp -- "$meta" "$dst/kernel_info.txt"
  # Preserve source-local headers, including vendor-compatibility headers.
  local source_dir metadata_dir
  source_dir=$(dirname "$src"); metadata_dir=$(dirname "$meta")
  for directory in "$source_dir" "$metadata_dir" "$ROOT/evaluation_set/$name"; do
    [[ -d "$directory" ]] || continue
    for hdr in "$directory"/*.h "$directory"/*.hpp "$directory"/*.inc; do
      if [[ -f "$hdr" ]]; then
        [[ ! -f "$dst/$(basename "$hdr")" ]] || cmp -s "$hdr" "$dst/$(basename "$hdr")" || {
          echo "Conflicting header $(basename "$hdr") for $name" >&2; exit 2;
        }
        cp -n -- "$hdr" "$dst/"
      fi
    done
  done
  echo "[SOURCE] $name <- ${src#$ROOT/}"
}

copy_kernel GAN \
  "$ROOT/evaluation_set/GAN/code_to_memory_outputs/meta/GAN.cpp" \
  "$ROOT/evaluation_set/GAN/code_to_memory_outputs/meta/kernel_info.txt"
copy_kernel Kalman \
  "$ROOT/evaluation_set/serrano-kalman-filter/kernel.cpp" \
  "$ROOT/evaluation_set/serrano-kalman-filter/kernel_info.txt"
copy_kernel chstone-aes \
  "$ROOT/evaluation_set/chstone-aes/kernel.c" \
  "$ROOT/evaluation_set/chstone-aes/kernel_info.txt"
copy_kernel chstone-jpeg \
  "$ROOT/evaluation_set/chstone-jpeg/kernel.c" \
  "$ROOT/evaluation_set/chstone-jpeg/kernel_info.txt"
copy_kernel covariance \
  "$ROOT/evaluation_set/covariance/kernel.cpp" \
  "$ROOT/evaluation_set/covariance/kernel_info.txt"
copy_kernel jacobi-2d \
  "$ROOT/evaluation_set/jacobi-2d/kernel.cpp" \
  "$ROOT/evaluation_set/jacobi-2d/kernel_info.txt"
copy_kernel rosetta-3d-rendering \
  "$ROOT/evaluation_set/rosetta-3d-rendering/kernel.cpp" \
  "$ROOT/evaluation_set/rosetta-3d-rendering/kernel_info.txt"
copy_kernel syr2k \
  "$ROOT/evaluation_set/syr2k/kernel.cpp" \
  "$ROOT/evaluation_set/syr2k/kernel_info.txt"
copy_kernel trmm-opt \
  "$ROOT/evaluation_set/trmm-opt/kernel.cpp" \
  "$ROOT/evaluation_set/trmm-opt/kernel_info.txt"

python - "$OUT" <<'PY'
from pathlib import Path
import hashlib, json, re, sys
root=Path(sys.argv[1]); report={}
pat=re.compile(r'/\*\s*(L[1-9][0-9]*)\s*:\s*\*/')
for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    src=next(p for p in folder.iterdir() if p.name.startswith('source.'))
    info=(folder/'kernel_info.txt').read_text().splitlines()
    actions=[s.split(',')[0].strip().upper() for s in info[1:] if s.strip()]
    labels=pat.findall(src.read_text())
    if not actions or len(actions)>64 or len(set(actions))!=len(actions):
        raise ValueError(f'{folder.name}: invalid action IDs/count: {actions}')
    if set(actions)!=set(labels) or len(labels)!=len(set(labels)):
        raise ValueError(f'{folder.name}: source Lk labels differ from kernel_info.txt: expected={actions}, found={labels}')
    report[folder.name]={"source":src.name,"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
                         "metadata_sha256":hashlib.sha256((folder/'kernel_info.txt').read_bytes()).hexdigest(),
                         "actions":actions}
(root/'SOURCES_MANIFEST.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(f'[SOURCE AUDIT] {len(report)} kernels, {sum(len(v["actions"]) for v in report.values())} declared Lk actions')
PY
