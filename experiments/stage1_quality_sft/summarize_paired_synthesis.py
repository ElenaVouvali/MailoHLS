"""Validate sealed Vitis results and expand reused jobs by request budget."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RESOURCE_FRACTIONS = {
    "BRAM_18K": "budget_frac_bram_18k",
    "DSP": "budget_frac_dsp",
    "FF": "budget_frac_ff",
    "LUT": "budget_frac_lut",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(manifest_path: Path, batch_report_path: Path, job_output_dir: Path | None = None) -> dict:
    manifest_path, batch_report_path = Path(manifest_path), Path(batch_report_path)
    manifest, report = json.loads(manifest_path.read_text()), json.loads(batch_report_path.read_text())
    manifest_hash = sha(manifest_path)
    if report.get("provenance", {}).get("manifest_sha256") != manifest_hash:
        raise ValueError("Vitis report does not authenticate the frozen synthesis manifest")
    if report.get("provenance", {}).get("runner_sha256") != manifest.get("runner_sha256"):
        raise ValueError("Vitis runner hash differs from the sealed bundle")
    jobs = {job["job_id"]: job for job in manifest.get("jobs", [])}
    results = report.get("results", [])
    by_id = {row.get("job_id"): row for row in results}
    if len(jobs) != len(results) or len(by_id) != len(results) or set(by_id) != set(jobs):
        raise ValueError("Vitis report is missing or duplicates a frozen job")

    attributions, physical_jobs = [], []
    for job_id, job in jobs.items():
        result = by_id[job_id]
        if result.get("kernel") != job.get("kernel") or result.get("part") != job.get("part"):
            raise ValueError(f"Synthesis identity mismatch for {job_id}")
        if float(result.get("clock_ns")) != float(job.get("clock_ns")):
            raise ValueError(f"Synthesis clock mismatch for {job_id}")
        metrics = result.get("metrics") if result.get("status") == "success" else None
        if result.get("status") == "success" and not metrics:
            raise ValueError(f"Successful synthesis lacks parsed QoR for {job_id}")
        if job_output_dir is not None:
            job_dir = Path(job_output_dir) / job_id
            receipt_path, log_path = job_dir / "receipt.json", job_dir / "synthesis.log"
            if not receipt_path.is_file() or not log_path.is_file():
                raise ValueError(f"Missing source/compiler receipt or log for {job_id}")
            receipt = json.loads(receipt_path.read_text())
            if sha(receipt_path) != result.get("receipt_sha256"):
                raise ValueError(f"Receipt hash mismatch for {job_id}")
            if receipt.get("job") != job or receipt.get("source_sha256") != job.get("materialized_source_sha256"):
                raise ValueError(f"Receipt does not bind the exact frozen source/configuration for {job_id}")
            source_path, tcl_path = job_dir / job["source"], job_dir / "run_hls.tcl"
            if (not source_path.is_file() or sha(source_path) != job["materialized_source_sha256"]
                    or not tcl_path.is_file() or sha(tcl_path) != receipt.get("tcl_sha256")):
                raise ValueError(f"Materialized source or Tcl hash mismatch for {job_id}")
            if sha(log_path) != result.get("log_sha256"):
                raise ValueError(f"Synthesis log hash mismatch for {job_id}")
            if result.get("status") == "success":
                xml_path = job_dir / "hls_project" / "solution1" / "syn" / "report" / f"{job['top']}_csynth.xml"
                if not xml_path.is_file() or sha(xml_path) != result.get("report_sha256"):
                    raise ValueError(f"csynth report does not match the recorded successful job {job_id}")
        physical_jobs.append({"job_id": job_id, "request_id": job["request_id"],
                              "kernel": job["kernel"], "arm": job["arm"],
                              "device": job["part"], "clock_ns": job["clock_ns"],
                              "source_sha256": job["materialized_source_sha256"],
                              "status": result["status"], "error": result.get("error"),
                              "runtime_seconds": result.get("runtime_seconds"),
                              "log_sha256": result.get("log_sha256"),
                              "report_sha256": result.get("report_sha256")})
        for attr in job.get("attributions", []):
            budget = {}
            fractions = attr.get("budget_fractions", {})
            for resource, fraction_name in RESOURCE_FRACTIONS.items():
                if fraction_name not in fractions:
                    raise ValueError(f"Missing {fraction_name} for {attr.get('request_id')}")
                budget[resource] = int(round(job["capacities"][resource] * fractions[fraction_name]))
            feasible = None
            if metrics is not None:
                feasible = all(metrics["resources"][key] <= budget[key] for key in RESOURCE_FRACTIONS)
            attributions.append({
                "request_id": attr["request_id"], "arm": attr["arm"],
                "decoder": attr["decoder"], "budget_id": attr["budget_id"],
                "job_id": job_id, "shared_synthesis": len(job.get("attributions", [])) > 1,
                "budget": budget, "status": result["status"],
                "qor": None if metrics is None else {
                    "latency_cycles": metrics["latency_cycles"],
                    "latency_at_requested_clock_ns": metrics["latency_at_requested_clock_ns"],
                    "estimated_clock_ns": metrics["estimated_clock_ns"],
                    "resources": metrics["resources"],
                    "resource_budget_feasible": feasible,
                    "fits_physical_device": metrics["fits_physical_device"],
                    "hls_estimated_clock_meets_request": metrics["hls_estimated_clock_meets_request"],
                    "timing_evidence": metrics["timing_evidence"],
                },
                "unknown_qor": metrics is None,
            })
    pairs = []
    by_request = {}
    for row in attributions:
        by_request.setdefault((row["request_id"], row["decoder"]), {})[row["arm"]] = row
    for (request_id, decoder), arms in sorted(by_request.items()):
        if {"single_reference", "multi_reference"} - set(arms):
            raise ValueError(f"Unpaired synthesis attribution for {request_id}/{decoder}")
        single, multi = arms["single_reference"], arms["multi_reference"]
        a, b = single["qor"], multi["qor"]
        pairs.append({
            "request_id": request_id, "decoder": decoder,
            "single_status": single["status"], "multi_status": multi["status"],
            "paired_latency_delta_ns_multi_minus_single": (
                None if a is None or b is None else
                b["latency_at_requested_clock_ns"] - a["latency_at_requested_clock_ns"]
            ),
            "both_measured_budget_feasible": bool(a and b and a["resource_budget_feasible"]
                                                  and b["resource_budget_feasible"]),
            "paired_resources": None if a is None or b is None else {
                key: b["resources"][key] - a["resources"][key] for key in RESOURCE_FRACTIONS
            },
            "interpretation": "HLS estimates for this exact paired source/target; not post-route timing or generalization evidence.",
        })
    return {
        "schema": "stage1-v4-paired-synthesis-summary-v1",
        "manifest_sha256": manifest_hash,
        "runner_sha256": manifest["runner_sha256"],
        "compiler": report.get("provenance", {}).get("version_output"),
        "executable_sha256": report.get("provenance", {}).get("executable_sha256"),
        "physical_job_count": len(physical_jobs),
        "request_attribution_count": len(attributions),
        "successful_job_count": sum(row["status"] == "success" for row in physical_jobs),
        "failed_job_count": sum(row["status"] != "success" for row in physical_jobs),
        "physical_jobs": physical_jobs,
        "attributions": attributions,
        "paired_comparisons": pairs,
        "unknown_qor_policy": "failed, timed-out, or incomplete synthesis has null QoR; no artificial penalty is assigned",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--batch-report", required=True, type=Path)
    ap.add_argument("--job-output-dir", required=True, type=Path,
                    help="Directory containing the authenticated per-job receipts/logs and HLS reports")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = summarize(args.manifest, args.batch_report, args.job_output_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("physical_job_count", "request_attribution_count",
                                                   "successful_job_count", "failed_job_count")}, indent=2))


if __name__ == "__main__":
    main()
