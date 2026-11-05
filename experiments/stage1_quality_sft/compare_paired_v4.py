"""Assemble paired model and decoder-policy differences from authenticated reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("complete", "source_legal", "tool_admissible", "exact_measured_match",
           "measured_feasible", "near_optimal_match", "complete_action_accuracy",
           "measured_regret")


def load(report_dir):
    report_dir = Path(report_dir)
    result = {}
    for name in ("single_reference", "multi_reference", "production_latency"):
        for policy in ("legacy", "source_tool_v1"):
            key = f"{name}_{policy}"
            path = report_dir / f"{key}.json"
            if name == "production_latency" and not path.exists():
                continue
            row = json.loads(path.read_text())
            if row["model_name"] != key:
                raise ValueError(f"Report name mismatch in {path}")
            result[key] = row
    if not all(f"{name}_{policy}" in result
               for name in ("single_reference", "multi_reference")
               for policy in ("legacy", "source_tool_v1")):
        raise ValueError("Both V4 adapters need legacy and source/tool reports")
    production = {f"production_latency_{policy}" in result
                  for policy in ("legacy", "source_tool_v1")}
    if len(production) != 1:
        raise ValueError("Production baseline must have both decoder policies or neither")
    signatures = {(r["request_sha256"], r["metric_request_sha256"], r["candidate_pool_sha256"], r["requests"])
                  for r in result.values()}
    if len(signatures) != 1 or next(iter(signatures))[3] != 54:
        raise ValueError("Reports do not share the exact frozen ordered 54-request evaluation inputs")
    return result


def _per_id(report):
    rows = report["per_request"]
    if len(rows) != 54 or len({r["request_id"] for r in rows}) != 54:
        raise ValueError("Missing or duplicate per-request report rows")
    return {r["request_id"]: r for r in rows}


def _paired(left, right):
    a, b = _per_id(left), _per_id(right)
    if set(a) != set(b):
        raise ValueError("Paired reports have different request identities")
    rows = []
    for rid in a:
        changes = {}
        for metric in METRICS:
            av, bv = a[rid].get(metric), b[rid].get(metric)
            changes[metric] = None if av is None or bv is None else (float(bv) - float(av))
        rows.append({"request_id": rid, "left": a[rid], "right": b[rid], "right_minus_left": changes})
    summary = {}
    for metric in METRICS:
        vals = [row["right_minus_left"][metric] for row in rows if row["right_minus_left"][metric] is not None]
        summary[metric] = {"paired_n": len(vals), "mean_right_minus_left": sum(vals) / len(vals) if vals else None,
                           "improved_count": sum(v > 0 for v in vals) if metric not in ("measured_regret",) else sum(v < 0 for v in vals),
                           "worsened_count": sum(v < 0 for v in vals) if metric not in ("measured_regret",) else sum(v > 0 for v in vals)}
    return {"summary": summary, "per_request": rows}


def compare(report_dir, output=None):
    reports = load(report_dir)
    comparisons = {}
    for policy in ("legacy", "source_tool_v1"):
        comparisons[f"multi_minus_single_{policy}"] = _paired(
            reports[f"single_reference_{policy}"], reports[f"multi_reference_{policy}"])
    model_names = ["single_reference", "multi_reference"]
    if "production_latency_legacy" in reports:
        model_names.append("production_latency")
    for model in model_names:
        comparisons[f"source_tool_minus_legacy_{model}"] = _paired(
            reports[f"{model}_legacy"], reports[f"{model}_source_tool_v1"])
    payload = {"schema": "stage1-v4-paired-comparison-v1", "requests": 54,
               "production_baseline_status": ("included" if "production_latency_legacy" in reports else "pending_inference"),
               "selection": "No checkpoint selected or promoted; report differences on the exact paired requests.",
               "models": {key: {"prediction_sha256": value["prediction_sha256"],
                                "complete_rate": value["complete_assignment_rate"],
                                "source_legal_rate": value["source_legal_rate"],
                                "tool_admissible_rate": value["tool_admissible_rate"],
                                "measured_match_coverage": value["exact_measured_match_coverage"],
                                "feasible_matches": value["measured_feasible_count"],
                                "unknown_qor": value["unknown_qor"],
                                "near_optimal_count": value["near_optimal_match_count"],
                                "complete_action_accuracy": value["complete_action_accuracy"]}
                          for key, value in reports.items()},
               "comparisons": comparisons}
    if output:
        Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", type=Path, default=Path("artifacts/stage1_quality_sft/v4_paired_20261105/reports"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/stage1_quality_sft/v4_paired_20261105/reports/paired_comparison.json"))
    args = ap.parse_args()
    print(json.dumps(compare(args.report_dir, args.output)["models"], indent=2))


if __name__ == "__main__":
    main()
