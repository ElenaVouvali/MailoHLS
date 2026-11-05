import json
import hashlib
from pathlib import Path

import pytest

from experiments.stage1_quality_sft.summarize_paired_synthesis import summarize


def _fixture(tmp_path: Path):
    caps = {"BRAM_18K": 100, "DSP": 100, "FF": 100, "LUT": 100}
    fractions = {"budget_frac_bram_18k": 0.5, "budget_frac_dsp": 0.5,
                  "budget_frac_ff": 0.5, "budget_frac_lut": 0.5}
    jobs = []
    results = []
    for arm, job_id in (("single_reference", "s"), ("multi_reference", "m")):
        jobs.append({"job_id": job_id, "request_id": "request-1", "kernel": "toy",
                     "arm": arm, "part": "part", "clock_ns": 10.0, "capacities": caps,
                     "source": "toy.c", "top": "toy",
                     "materialized_source_sha256": hashlib.sha256(f"source-{arm}".encode()).hexdigest(), "attributions": [{
                         "request_id": "request-1", "arm": arm, "decoder": "source_tool_v2",
                         "budget_id": "half", "budget_fractions": fractions,
                     }]})
        status = "success" if arm == "single_reference" else "failed"
        metrics = ({"latency_cycles": 9, "latency_at_requested_clock_ns": 90.0,
                    "estimated_clock_ns": 8.0,
                    "resources": {"BRAM_18K": 40, "DSP": 70, "FF": 30, "LUT": 60},
                    "fits_physical_device": True,
                    "hls_estimated_clock_meets_request": True,
                    "timing_evidence": "HLS estimate"} if status == "success" else None)
        results.append({"job_id": job_id, "kernel": "toy", "part": "part",
                        "clock_ns": 10.0, "status": status, "metrics": metrics,
                        "error": "compile error" if status == "failed" else None,
                        "runtime_seconds": 2, "log_sha256": None, "report_sha256": None})
    manifest = {"runner_sha256": "runner", "jobs": jobs}
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest))
    job_output = tmp_path / "jobs"
    for job, result in zip(jobs, results):
        directory = job_output / job["job_id"]
        (directory / "hls_project/solution1/syn/report").mkdir(parents=True)
        source_bytes = f"source-{job['arm']}".encode()
        tcl_bytes, log_bytes = b"tcl", f"log-{job['arm']}".encode()
        (directory / job["source"]).write_bytes(source_bytes)
        (directory / "run_hls.tcl").write_bytes(tcl_bytes)
        (directory / "synthesis.log").write_bytes(log_bytes)
        receipt = {"job": job, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                   "tcl_sha256": hashlib.sha256(tcl_bytes).hexdigest()}
        receipt_path = directory / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        result["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        result["log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
        if result["metrics"]:
            xml_path = directory / "hls_project/solution1/syn/report/toy_csynth.xml"
            xml_path.write_text("<report/>")
            result["report_sha256"] = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps({"provenance": {"manifest_sha256": hashlib.sha256(
        manifest_path.read_bytes()).hexdigest(), "runner_sha256": "runner", "version_output": "Vitis 2021.1"},
        "results": results}))
    return manifest_path, report_path, job_output


def test_synthesis_summary_recomputes_each_budget_and_preserves_unknown_qor(tmp_path):
    manifest, report, job_output = _fixture(tmp_path)
    summary = summarize(manifest, report, job_output)
    by_arm = {row["arm"]: row for row in summary["attributions"]}
    assert by_arm["single_reference"]["budget"]["LUT"] == 50
    assert by_arm["single_reference"]["qor"]["resource_budget_feasible"] is False
    assert by_arm["multi_reference"]["qor"] is None
    assert by_arm["multi_reference"]["unknown_qor"] is True
    assert summary["paired_comparisons"][0]["paired_latency_delta_ns_multi_minus_single"] is None


def test_synthesis_summary_rejects_wrong_manifest_identity(tmp_path):
    manifest, report, job_output = _fixture(tmp_path)
    data = json.loads(report.read_text())
    data["provenance"]["manifest_sha256"] = "wrong"
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not authenticate"):
        summarize(manifest, report, job_output)


def test_synthesis_summary_rejects_changed_generated_source(tmp_path):
    manifest, report, job_output = _fixture(tmp_path)
    (job_output / "s" / "toy.c").write_text("altered source")
    with pytest.raises(ValueError, match="Materialized source or Tcl hash mismatch"):
        summarize(manifest, report, job_output)
