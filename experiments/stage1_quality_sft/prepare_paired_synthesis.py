"""Freeze paired V4 predictions for the existing sealed-input Vitis runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from LLM_branch.common import mailohls_contract
from LLM_branch.common.directive_domains import DIRECTIVE_RE
from experiments.refinement.synthesis import materialize
from experiments.stage1_quality_sft import core
from experiments.stage1_quality_sft.support import sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/stage1_quality_sft/v4_paired_20261105/synthesis_bundle"
DEFAULT_RUNNER = ROOT / "artifacts/synthesis/quality_v4_gan_kalman_20261105/bundle/run_batch.py"


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _normalized(path):
    out = {}
    for row in _rows(path):
        rid = row.get("context_id") or row.get("request_id")
        candidates = row.get("candidates") or []
        selected = row.get("best_unique_sample_id_by_model_score")
        candidate = next((item for item in candidates if item.get("sample_id") == selected), None)
        if rid is None or candidate is None:
            raise ValueError(f"Incomplete production inference row in {path}: {rid}")
        if rid in out:
            raise ValueError(f"Duplicate request in {path}: {rid}")
        out[rid] = candidate.get("canonical_prediction", candidate.get("prediction", ""))
    return out


def prepare(output_dir=DEFAULT_OUT, runner=DEFAULT_RUNNER, result_root=None):
    output_dir, runner = Path(output_dir), Path(runner)
    result_root = Path(result_root or output_dir.parent)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite synthesis bundle: {output_dir}")
    if not runner.is_file():
        raise FileNotFoundError(f"Existing sealed-input Vitis runner not found: {runner}")
    requests = {row["request_id"]: row for row in _rows(result_root / "frozen_validation_requests.jsonl")}
    subset = json.loads((result_root / "synthesis_subset.json").read_text())
    expected_ids = [row["request_id"] for row in subset["requests"]]
    if len(expected_ids) != 6 or len(set(expected_ids)) != 6 or any(r not in requests for r in expected_ids):
        raise ValueError("Frozen synthesis subset is not the authenticated six-request cohort")

    prediction_files = {}
    for arm in ("single_reference", "multi_reference"):
        for mode in ("legacy", "source_tool_v1"):
            path = result_root / "predictions" / f"{arm}_{mode}.jsonl"
            mapped = _normalized(path)
            if set(mapped) != set(requests):
                raise ValueError(f"Prediction request IDs do not match frozen cohort: {path}")
            prediction_files[(arm, mode)] = (path, mapped)

    output_dir.mkdir(parents=True)
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir()
    shutil.copy2(runner, output_dir / "run_batch.py")
    deduplicated = {}
    request_attributions = []
    for request_id in expected_ids:
        request = requests[request_id]
        source_path = (ROOT / request["source_file"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        kernel_dir = source_path.parent
        top = (kernel_dir / "kernel_info.txt").read_text().splitlines()[0].strip()
        if not top:
            raise ValueError(f"Missing HLS top identity for {request['kernel_name']}")
        template = request["input"]
        for arm, mode in (("single_reference", "legacy"), ("single_reference", "source_tool_v1"),
                          ("multi_reference", "legacy"), ("multi_reference", "source_tool_v1")):
            prediction_path, predictions = prediction_files[(arm, mode)]
            prediction = predictions[request_id]
            values = core.parse(prediction)
            normalized_target = "\n".join(f"{key} = {value}" for key, value in sorted(values.items()))
            materialized = materialize(template, normalized_target)
            source_fields = {match.group(1).upper() for match in DIRECTIVE_RE.finditer(template)}
            if len(values) != len(source_fields) or "auto{" in materialized.lower():
                raise ValueError(f"Incomplete or unresolved source assignment: {request_id}/{arm}/{mode}")
            source_bytes = materialized.encode()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            # Resource budget changes the request decision, not the physical
            # HLS implementation. Reuse the exact same source/device/clock/
            # assignment result across budget views and re-evaluate feasibility
            # against each attribution after parsing the one synthesis report.
            physical_key = (request["kernel_name"], request["source_key"], request["source_file"],
                            request["device"], round(float(request["clock_period"]), 2))
            design_key = (physical_key, source_hash)
            attribution = {"request_id": request_id, "arm": arm, "decoder": mode,
                           "prediction_sha256": sha(prediction_path), "prediction": prediction,
                           "budget_id": request["resource_budget_id"],
                           "budget_fractions": request["quality_budget_fractions"]}
            if design_key in deduplicated:
                deduplicated[design_key]["attributions"].append(attribution)
                request_attributions.append({"request_id": request_id, "arm": arm, "decoder": mode,
                                             "job_id": deduplicated[design_key]["job_id"], "reused": True,
                                             "budget_id": attribution["budget_id"],
                                             "budget_fractions": attribution["budget_fractions"]})
                continue
            identity_hash = hashlib.sha256((request_id + ":" + source_hash).encode()).hexdigest()[:14]
            job_id = f"v4_{len(deduplicated)+1:02d}_{identity_hash}"
            inputs = jobs_dir / job_id
            inputs.mkdir()
            (inputs / source_path.name).write_bytes(source_bytes)
            header_hashes = {}
            for file in sorted(kernel_dir.iterdir()):
                if (not file.is_file() or file.name == source_path.name or file.name == "kernel_info.txt"
                        or file.name in {"ap_int.h", "ap_fixed.h"}):
                    continue
                shutil.copy2(file, inputs / file.name)
                if file.suffix.lower() in {".h", ".hpp"}:
                    header_hashes[file.name] = sha(file)
            device = request["device"]
            capacities = mailohls_contract.DEVICE_RESOURCES[device]
            budget = {resource: int(round(capacities[resource] * request["quality_budget_fractions"][fraction]))
                      for resource, fraction in zip(mailohls_contract.RESOURCE_KEYS,
                          ("budget_frac_bram_18k", "budget_frac_dsp", "budget_frac_ff", "budget_frac_lut"))}
            job = {"job_id": job_id, "kernel": request["kernel_name"], "arm": arm,
                   "decoder": mode, "request_number": expected_ids.index(request_id) + 1,
                   "request_id": request_id, "source": source_path.name, "top": top,
                   "part": device, "clock_ns": float(request["clock_period"]),
                   "capacities": capacities, "budget": budget,
                   "materialized_source_sha256": source_hash,
                   "original_source_sha256": sha(source_path),
                   "source_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                   "headers": header_hashes,
                   "attributions": [attribution]}
            deduplicated[design_key] = job
            request_attributions.append({"request_id": request_id, "arm": arm, "decoder": mode,
                                         "job_id": job_id, "reused": False,
                                         "budget_id": attribution["budget_id"],
                                         "budget_fractions": attribution["budget_fractions"]})

    jobs = list(deduplicated.values())
    manifest = {"schema": "stage1-v4-paired-vitis-subset-v1", "selection_manifest_sha256": sha(result_root / "synthesis_subset.json"),
                "validation_manifest_sha256": sha(result_root / "manifest.json"),
                "source_tool_policy_sha256": sha(ROOT / "experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy.json"),
                "runner_sha256": sha(runner), "source_materialization": "experiments.refinement.synthesis.materialize; no clipping or repair",
                "settings": ["config_array_partition -complete_threshold 0 -throughput_driven off",
                             "config_compile -pipeline_loops 0"],
                "vendor_headers_not_bundled": ["ap_int.h", "ap_fixed.h"],
                "qor_policy": "successful authenticated Vitis csynth reports only; failures have unknown QoR; resource budgets assessed after synthesis",
                "request_attributions": request_attributions, "jobs": jobs}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"job_count": len(jobs), "attribution_count": len(request_attributions),
            "reuse_count": sum(row["reused"] for row in request_attributions),
            "manifest_sha256": sha(output_dir / "manifest.json"), "runner_sha256": sha(runner)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    ap.add_argument("--result-root", type=Path)
    args = ap.parse_args()
    print(json.dumps(prepare(args.output_dir, args.runner, args.result_root), indent=2))


if __name__ == "__main__":
    main()
