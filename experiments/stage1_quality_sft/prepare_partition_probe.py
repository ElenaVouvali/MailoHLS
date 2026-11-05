"""Prepare tiny Vitis 2021.1 explicit-partition boundary/shape probes.

The script copies the already-used sealed-input runner; it does not implement
another synthesis runner. Probes are isolated from model and validation data.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from LLM_branch.common import mailohls_contract
from experiments.stage1_quality_sft.support import sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = ROOT / "artifacts/synthesis/quality_v4_gan_kalman_20261105/bundle/run_batch.py"
DEFAULT_OUT = ROOT / "artifacts/stage1_quality_sft/v4_paired_20261105/partition_probe_bundle"


def _source_1d(n):
    return f'''void partition_probe(int input[{n}], int *output) {{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
  int value = 0;
  for (int i=0; i<{n}; ++i) value += input[i];
  *output = value;
}}
'''


def _source_2d(dim):
    return f'''void partition_probe(int input[64][32], int *output) {{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ARRAY_PARTITION variable=input complete dim={dim}
  int value = 0;
  for (int i=0; i<64; ++i)
    for (int j=0; j<32; ++j) value += input[i][j];
  *output = value;
}}
'''


def prepare(output=DEFAULT_OUT, runner=DEFAULT_RUNNER):
    output, runner = Path(output), Path(runner)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite compiler probe bundle: {output}")
    if not runner.is_file():
        raise FileNotFoundError(f"Existing sealed-input runner not found: {runner}")
    output.mkdir(parents=True)
    shutil.copy2(runner, output / "run_batch.py")
    jobs_dir, jobs = output / "jobs", []
    jobs_dir.mkdir()
    source_cases = [("one_d_1024", _source_1d(1024)), ("one_d_1025", _source_1d(1025)),
                    ("two_d_dim0", _source_2d(0)), ("two_d_dim1", _source_2d(1)),
                    ("two_d_dim2", _source_2d(2))]
    device = "xczu7ev-ffvc1156-2-e"
    capacities = mailohls_contract.DEVICE_RESOURCES[device]
    for job_id, source in source_cases:
        job_dir = jobs_dir / job_id
        job_dir.mkdir()
        source_path = job_dir / "probe.cpp"
        source_path.write_text(source)
        jobs.append({"job_id": job_id, "kernel": "partition_compiler_probe", "arm": "policy_probe",
                     "request_number": len(jobs) + 1, "source": "probe.cpp", "top": "partition_probe",
                     "part": device, "clock_ns": 10.0, "capacities": capacities,
                     "budget": capacities, "materialized_source_sha256": sha(source_path),
                     "headers": {}, "probe_only": True})
    manifest = {"schema": "stage1-vitis-partition-probe-v1", "jobs": jobs,
                "settings": ["config_array_partition -complete_threshold 0 -throughput_driven off",
                             "config_compile -pipeline_loops 0"],
                "vendor_headers_not_bundled": ["ap_int.h", "ap_fixed.h"],
                "qor_policy": "Compiler boundary probe only; no model-selection or QoR claims.",
                "runner_sha256": sha(runner)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"jobs": len(jobs), "manifest_sha256": sha(output / "manifest.json"), "runner_sha256": sha(runner)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args = ap.parse_args()
    print(json.dumps(prepare(args.output, args.runner), indent=2))


if __name__ == "__main__":
    main()
