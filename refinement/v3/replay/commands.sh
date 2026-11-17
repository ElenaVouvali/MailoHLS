#!/usr/bin/env bash
# Reproduce offline into a NEW output directory, from the repository root.
# Example: bash refinement/v3/replay/commands.sh /tmp/cg-bdr-v3-reproduced
set -eu
result_dir=${1:?Supply a new output directory}
mkdir "$result_dir"
run_dir="$result_dir/runs"
python3 -m unittest discover -s refinement/v3/tests -v > "$result_dir/tests-v3.txt" 2>&1
python3 -m unittest discover -s refinement/v2/tests -v > "$result_dir/tests-v2.txt" 2>&1
python3 -m unittest discover -s refinement/tests -v > "$result_dir/tests-legacy.txt" 2>&1
/usr/bin/python3.8 -m py_compile refinement/v3/*.py refinement/v3/tests/*.py
python3 refinement/v3/runner_v3.py audit > "$result_dir/archived-audit.json"
python3 refinement/v3/runner_v3.py prepare --kernels GAN jacobi-2d Kalman trmm-opt covariance syr2k --runs-root "$run_dir" --reuse-verified-baselines > "$result_dir/prepare.json"
/usr/bin/python3.8 refinement/v3/runner_v3.py preview --mode strict --kernels GAN jacobi-2d Kalman trmm-opt covariance syr2k --runs-root "$run_dir" --explain > "$result_dir/preview.json"
/usr/bin/python3.8 refinement/v3/runner_v3.py pilot --mode strict --kernels jacobi-2d GAN --runs-root "$run_dir" --no-hls > "$result_dir/pilot-no-hls.json"
/usr/bin/python3.8 refinement/v3/runner_v3.py report --runs-root "$run_dir" > "$result_dir/report.json"
/usr/bin/python3.8 refinement/v3/audit_evidence.py --bundles "$run_dir"/bundles/* > "$result_dir/audit.json"
# No actual pilot, SSH, Vitis or real evaluation reservation occurs in this script.
