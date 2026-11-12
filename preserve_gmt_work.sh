#!/usr/bin/env bash
# Preserve uncommitted research code on GitHub without touching running GPU jobs.
# Deliberately excludes large checkpoints, .pt, .pth, .safetensors, caches.
set -euo pipefail
ROOT="${MAILOHLS_ROOT:-$HOME/MailoHLS}"
HIST="$ROOT/worktrees/historical-mlir-gmt-gemv"
RC3="$ROOT/worktrees/gemv-structural-explainability"

git -C "$ROOT" fetch origin --prune
printf '\nWORKTREES\n'; git -C "$ROOT" worktree list

save_one() {
  local wt="$1" exp="$2" dst="$3" message="$4"
  if [[ ! -d "$wt/experiments/$exp" ]]; then
    echo "SKIP: missing worktree experiment $wt/experiments/$exp" >&2; return
  fi
  echo -e "\n=== $wt ==="
  git -C "$wt" status --short
  # Refuse to include unrelated pre-existing staged edits in the snapshot.
  while IFS= read -r staged; do
    if [[ -n "$staged" && "$staged" != "experiments/$exp/"* ]]; then
      echo "ABORT: unrelated file already staged in $wt: $staged" >&2
      echo "Preserve/review that staged work independently before rerunning." >&2
      return 2
    fi
  done < <(git -C "$wt" diff --cached --name-only)
  # Stage ONLY experiment content. Do not stage global source/checkpoint data.
  git -C "$wt" add -A -- "experiments/$exp"
  # Keep only reviewable source, plots, metadata. Unstage binary checkpoints.
  python3 - "$wt" <<'PY'
import pathlib, subprocess, sys
wt=sys.argv[1]
p=subprocess.run(['git','-C',wt,'diff','--cached','--name-only','-z'],capture_output=True,check=True).stdout
files=[x.decode('utf-8','surrogateescape') for x in p.split(b'\0') if x]
for rel in files:
    if not rel.startswith('experiments/'):
        continue
    f=pathlib.Path(wt)/rel
    forbidden=(f.suffix.lower() in {'.pt','.pth','.safetensors','.ckpt','.pkl','.pickle','.bin','.zip'}
               or f.name.startswith('.env')
               or any(w in f.name.lower() for w in ['secret','credential','password','api_key','id_rsa']))
    oversized=f.exists() and f.is_file() and f.stat().st_size>12*1024*1024
    if forbidden or oversized or any(x in f.parts for x in ['private','secrets']):
        subprocess.run(['git','-C',wt,'restore','--staged','--',rel],check=True)
        print('EXCLUDED FROM COMMIT:',rel)
PY
  git -C "$wt" diff --cached --check
  git -C "$wt" diff --cached --stat
  if ! git -C "$wt" diff --cached --quiet; then
    git -C "$wt" commit -m "$message"
  else
    echo "No new experiment changes to commit"
  fi
  # Protect existing remote history: push only if remote HEAD is ancestor.
  local remote="refs/remotes/origin/$dst"
  if git -C "$wt" show-ref --verify --quiet "$remote"; then
    if git -C "$wt" merge-base --is-ancestor "$remote" HEAD; then
      git -C "$wt" push origin "HEAD:refs/heads/$dst"
    else
      local backup="research/gmt-snapshot-$(date -u +%Y%m%dT%H%M%SZ)"
      echo "Remote $dst diverged. Preserving safe snapshot branch $backup instead."
      git -C "$wt" push origin "HEAD:refs/heads/$backup"
    fi
  else
    git -C "$wt" push origin "HEAD:refs/heads/$dst"
  fi
  echo "Saved commit: $(git -C "$wt" rev-parse HEAD)"
}

save_one "$HIST" historical_mlir_gmt_gemv research/historical-mlir-gmt-gemv 'Preserve historical MLIR GMT experiment implementation and diagnostics'
save_one "$RC3" gemv_structural_explainability final-architecture 'Preserve RC3 GMT experiment implementation and diagnostics'
echo -e '\nDONE. Model checkpoints remain local; back them up separately if needed.'
