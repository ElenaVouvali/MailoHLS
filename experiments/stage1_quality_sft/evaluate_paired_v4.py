"""Compare paired V4 and production Stage-1 inference against frozen QoR pools."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from LLM_branch.common import directive_domains
from LLM_branch.common.source_tool_policy import (
    load_policy, tuple_rejection, validate_complete_assignment, validate_source_assignment,
)
from experiments.stage1_quality_sft import core
from experiments.stage1_quality_sft.evaluate import evaluate as evaluate_qor
from experiments.stage1_quality_sft.inference import normalize_production_predictions
from experiments.stage1_quality_sft.support import load_registry, normalize_kname, sha


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "artifacts/stage1_quality_sft/v4_paired_20261105"
DEFAULT_POLICY = ROOT / "experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy.json"


def _read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _groups(actions, values):
    result = {}
    for action_id, action in actions.items():
        keys = ((f"AUTO{{_PIPE_{action_id}}}", f"AUTO{{_UNROLL_{action_id}}}")
                if action.kind == "loop" else
                (f"AUTO{{_ARRAY_T_{action_id}}}", f"AUTO{{_ARRAY_F_{action_id}}}", f"AUTO{{_ARRAY_D_{action_id}}}"))
        result[action_id] = (action.kind, keys, tuple(values.get(key) for key in keys))
    return result


def _macro(rows, field):
    vals = [float(r[field]) for r in rows if r.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def _summarize_group(rows):
    count = len(rows)
    return {
        "requests": count,
        "complete_rate": sum(r["complete"] for r in rows) / count if count else None,
        "source_legal_rate": sum(r["source_legal"] for r in rows) / count if count else None,
        "tool_admissible_rate": sum(r["tool_admissible"] for r in rows) / count if count else None,
        "exact_measured_match_rate": sum(r["measured_match"] for r in rows) / count if count else None,
        "measured_feasible_rate_all_requests": sum(r["measured_feasible"] is True for r in rows) / count if count else None,
        "unknown_qor_rate": sum(r["qor_unknown"] for r in rows) / count if count else None,
        "mean_regret_conditioned_on_feasible_match": _macro(
            [r for r in rows if r["measured_feasible"] is True], "measured_regret"),
        "within_1pct_conditioned_on_feasible_match": _macro(
            [r for r in rows if r["measured_feasible"] is True], "within_1pct"),
        "within_5pct_conditioned_on_feasible_match": _macro(
            [r for r in rows if r["measured_feasible"] is True], "within_5pct"),
        "near_optimal_rate": sum(r["near_optimal_match"] for r in rows) / count if count else None,
        "complete_action_accuracy_vs_selected_winner": _macro(rows, "complete_action_accuracy"),
    }


def _transition_report(requests, per_request):
    by_id = {r["request_id"]: r for r in per_request}
    pairs = {"device": defaultdict(list), "clock": defaultdict(list), "budget": defaultdict(list)}
    for left_index, left in enumerate(requests):
        for right in requests[left_index + 1:]:
            a, b = by_id[left["request_id"]], by_id[right["request_id"]]
            x, y = left["request_identity"], right["request_identity"]
            same_source = x[0:3] == y[0:3] and x[6] == y[6]
            relation = None
            if same_source and x[3] != y[3] and x[4] == y[4] and x[5] == y[5]:
                relation = "device"
            elif same_source and x[3] == y[3] and x[4] != y[4] and x[5] == y[5]:
                relation = "clock"
            elif same_source and x[3] == y[3] and x[4] == y[4] and x[5] != y[5]:
                relation = "budget"
            if relation:
                pairs[relation][(left["family"], left["physical_context"][0])].append((left, right, a, b))
    report = {}
    for relation, groups in pairs.items():
        rows = []
        for (family, kernel), comparisons in sorted(groups.items()):
            for left, right, a, b in comparisons:
                pred_a = dict(a["assignment"] or ())
                pred_b = dict(b["assignment"] or ())
                winner_a = a.get("selected_reference_assignment") or {}
                winner_b = b.get("selected_reference_assignment") or {}
                predicted_fields = sorted(key for key in set(pred_a) | set(pred_b)
                                          if pred_a.get(key) != pred_b.get(key))
                reference_fields = sorted(key for key in set(winner_a) | set(winner_b)
                                          if winner_a.get(key) != winner_b.get(key))
                correct_fields = sorted(set(predicted_fields) & set(reference_fields))
                rows.append({
                    "family": family, "kernel": kernel,
                    "request_ids": [left["request_id"], right["request_id"]],
                    "prediction_changed": a["assignment"] != b["assignment"] if a["assignment"] and b["assignment"] else None,
                    "predicted_changed_fields": predicted_fields,
                    "selected_winner_changed_fields": reference_fields,
                    "matched_changed_fields": correct_fields,
                    "changed_field_precision_vs_selected_winner": len(correct_fields) / len(predicted_fields) if predicted_fields else None,
                    "changed_field_recall_vs_selected_winner": len(correct_fields) / len(reference_fields) if reference_fields else None,
                    "both_measured_near_optimal": bool(a["near_optimal_match"] and b["near_optimal_match"]),
                    "both_measured_feasible": bool(a["measured_feasible"] is True and b["measured_feasible"] is True),
                    "exact_measured_match_both": bool(a["measured_match"] and b["measured_match"]),
                    "both_qor_unknown": bool(a["qor_unknown"] and b["qor_unknown"]),
                    "classification": (
                        "correctly_invariant_near_optimal" if a["assignment"] == b["assignment"] and a["near_optimal_match"] and b["near_optimal_match"] else
                        "measured_competitive_change" if a["assignment"] != b["assignment"] and a["near_optimal_match"] and b["near_optimal_match"] else
                        "measured_feasibility_violation" if a["measured_feasible"] is False or b["measured_feasible"] is False else
                        "unknown_or_not_supported"
                    )
                })
        report[relation] = {
            "pair_count": len(rows),
            "prediction_changed_count": sum(r["prediction_changed"] is True for r in rows),
            "correctly_invariant_near_optimal": sum(r["classification"] == "correctly_invariant_near_optimal" for r in rows),
            "measured_competitive_change": sum(r["classification"] == "measured_competitive_change" for r in rows),
            "measured_feasibility_violations": sum(r["classification"] == "measured_feasibility_violation" for r in rows),
            "unknown_or_unsupported": sum(r["classification"] == "unknown_or_not_supported" for r in rows),
            "per_pair": rows,
            "interpretation": "Descriptive paired contexts; no changed assignment is credited without exact measured near-optimal support.",
        }
    return report


def evaluate_prediction_file(name, prediction_path, *, output_dir=DEFAULT_DIR, policy_path=DEFAULT_POLICY):
    output_dir = Path(output_dir)
    frozen_path = output_dir / "frozen_validation_requests.jsonl"
    pool_path = output_dir / "candidate_pools.jsonl.gz"
    metric_path = output_dir / "metric_requests.jsonl.gz"
    normalized_path = output_dir / "predictions" / f"{name}_normalized.jsonl"
    normalize_production_predictions(Path(prediction_path), normalized_path)
    predictions = _read_jsonl(normalized_path)
    frozen = _read_jsonl(frozen_path)
    metric_requests = _read_jsonl(metric_path)
    pool_records = _read_jsonl(pool_path)
    pools = {p["candidate_pool_id"]: p["candidates"] for p in pool_records}
    if len(predictions) != 54 or [p["request_id"] for p in predictions] != [r["request_id"] for r in frozen]:
        raise ValueError(f"{name}: generated cases do not align exactly with the frozen ordered 54 requests")
    execution_path = output_dir / "inference_execution_manifest.json"
    execution = json.loads(execution_path.read_text()) if execution_path.is_file() else {}
    model_key = next((key for key in execution.get("models", {})
                      if name == key or name.startswith(key + "_")), None)
    default_registry = Path("/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/single_reference/run/directive_domain_registry.json")
    registry_path = Path(execution["models"][model_key]["registry_path"]) if model_key else default_registry
    registry = load_registry(registry_path)
    policy = load_policy(policy_path)
    actions_by_kernel, prediction_by_id = {}, {p["request_id"]: p for p in predictions}
    raw_rows = _read_jsonl(prediction_path)
    decoding_diagnostics = {}
    for row in raw_rows:
        rid = row.get("context_id") or row.get("request_id")
        selected_id = row.get("best_unique_sample_id_by_model_score")
        candidate = next((c for c in row.get("candidates", []) if c.get("sample_id") == selected_id), None)
        if rid:
            decoding_diagnostics[rid] = {
                "source_tool_contract": row.get("source_tool_contract"),
                "selected_policy_diagnostics": candidate.get("admissibility_policy") if candidate else None,
                "selected_candidate_id": selected_id,
            }
    selected_source_registry = {}
    decoded, source_legal_count, active_valid_count, invalid_factors = 0, 0, 0, 0
    for request in frozen:
        kernel = request["kernel_name"]
        if kernel not in actions_by_kernel:
            actions_by_kernel[kernel] = directive_domains.load_source_actions(kernel, ROOT / "Data/ApplicationDataset")
        action_map = actions_by_kernel[kernel]
        fields = {f"AUTO{{_{k}_{label}}}" for label, action in action_map.items()
                  for k in (("PIPE", "UNROLL") if action.kind == "loop" else ("ARRAY_T", "ARRAY_F", "ARRAY_D"))}
        pred = prediction_by_id[request["request_id"]].get("prediction", "")
        try:
            values = core.parse(pred)
            complete = bool(values) and set(values) == fields
            source_legal = False
            if complete:
                validate_source_assignment(values, action_map)
                source_legal = True
                source_legal_count += 1
            domain = registry.get(normalize_kname(kernel), {})
            saved_legal = complete and all(value in domain.get(lhs, []) for lhs, value in values.items())
            if complete:
                decoded += 1
            tool_legal = False
            tool_reason = None
            if complete:
                try:
                    validate_complete_assignment(values, action_map, policy)
                    tool_legal = True
                    active_valid_count += 1
                except ValueError as exc:
                    tool_reason = str(exc)
            for label, action in action_map.items():
                if action.kind != "array" or not complete:
                    continue
                typ, factor, dim = (values.get(f"AUTO{{_{k}_{label}}}", "") for k in ("ARRAY_T", "ARRAY_F", "ARRAY_D"))
                if dim == "0" and typ == "none":
                    continue
                reason = tuple_rejection((typ, factor, dim), action, policy)
                if reason and reason.startswith("factor_exceeds_selected_dimension_extent"):
                    invalid_factors += 1
            selected_source_registry[request["request_id"]] = {
                "complete": complete, "expected_field_count": len(fields),
                "actual_field_count": len(values), "source_legal": source_legal,
                "saved_checkpoint_domain_legal": saved_legal,
                "tool_admissible": tool_legal, "tool_rejection": tool_reason,
                "assignment": tuple(sorted(values.items())) if complete else None,
            }
        except ValueError as exc:
            selected_source_registry[request["request_id"]] = {
                "complete": False, "expected_field_count": len(fields), "actual_field_count": 0,
                "source_legal": False, "saved_checkpoint_domain_legal": False,
                "tool_admissible": False, "tool_rejection": str(exc), "assignment": None,
            }
    qor = evaluate_qor(metric_requests, predictions, pools, registry=None)
    pool_by_id = {p["candidate_pool_id"]: {c["candidate_id"]: c for c in p["candidates"]} for p in pool_records}
    per_request = []
    for request in metric_requests:
        result = selected_source_registry[request["request_id"]]
        pred = prediction_by_id[request["request_id"]]["prediction"]
        values = core.parse(pred) if result["complete"] else {}
        identity = tuple(sorted(values.items())) if values else None
        pool = pool_by_id[request["candidate_pool_id"]]
        match = next((c for c in pool.values() if tuple(sorted(c["assignment"].items())) == identity), None) if identity else None
        accepted = set(request["sensitivity"]["0.01"]["accepted_candidate_ids"])
        candidate_id = match["candidate_id"] if match else None
        best_id = request["best_candidate_id"]
        # Singleton-reference accuracy is defined only where the quality set
        # has one feasible near-optimal assignment.
        actions = actions_by_kernel[request["request_identity"][0]]
        winner = pool.get(best_id, {}).get("assignment", {})
        predicted_groups, winner_groups = _groups(actions, values), _groups(actions, winner)
        group_total = len(actions)
        group_correct = sum(predicted_groups[a][2] == winner_groups[a][2] for a in actions) if values and winner else 0
        by_action_kind = {}
        for kind in ("loop", "array"):
            ids = [a for a in actions if actions[a].kind == kind]
            by_action_kind[kind] = {
                "correct": sum(predicted_groups[a][2] == winner_groups[a][2] for a in ids) if values and winner else 0,
                "total": len(ids),
                "accuracy": (sum(predicted_groups[a][2] == winner_groups[a][2] for a in ids) / len(ids)
                             if values and winner and ids else None),
            }
        output_row = next(x for x in qor["per_request"] if x["request_id"] == request["request_id"])
        per_request.append({
            "request_id": request["request_id"], "kernel": request["request_identity"][0],
            "family": request["family"], "device": request["physical_context"][3],
            "clock_ns": request["physical_context"][4], "budget": request["budget"],
            **result,
            "exact_selected_winner_match": bool(candidate_id and candidate_id == best_id),
            "near_optimal_match": bool(candidate_id and candidate_id in accepted),
            "measured_match": output_row["exact_measured_match"],
            "measured_feasible": output_row["measured_feasible"],
            "measured_latency": output_row["latency"], "best_feasible_latency": output_row["best_feasible_latency"],
            "measured_regret": output_row["regret"], "within_1pct": output_row["within_1pct"],
            "within_5pct": output_row["within_5pct"], "qor_unknown": not output_row["exact_measured_match"],
            "complete_action_accuracy": group_correct / group_total if values and winner and group_total else None,
            "selected_reference_assignment": winner,
            "loop_action_accuracy": by_action_kind["loop"]["accuracy"],
            "array_action_accuracy": by_action_kind["array"]["accuracy"],
            "complete_action_counts": by_action_kind,
            "decoding_diagnostics": decoding_diagnostics.get(request["request_id"]),
        })
    families = sorted({r["family"] for r in per_request})
    result = {
        "schema": "stage1-v4-paired-inference-evaluation-v1", "model_name": name,
        "prediction_sha256": sha(prediction_path), "normalized_prediction_sha256": sha(normalized_path),
        "checkpoint_identity": execution.get("models", {}).get(model_key) if model_key else None,
        "saved_registry_sha256": sha(registry_path),
        "inference_execution_manifest_sha256": sha(execution_path) if execution_path.is_file() else None,
        "inference_code_sha256": sha(ROOT / "LLM_branch/inference/eval_stage1_stage2_stage3.py"),
        "source_tool_policy_code_sha256": sha(ROOT / "LLM_branch/common/source_tool_policy.py"),
        "source_action_registry_code_sha256": sha(ROOT / "LLM_branch/common/directive_domains.py"),
        "request_sha256": sha(frozen_path), "metric_request_sha256": sha(metric_path),
        "candidate_pool_sha256": sha(pool_path), "policy_sha256": policy.source_sha256,
        "requests": len(frozen), "complete_assignments": decoded,
        "complete_assignment_rate": decoded / len(frozen),
        "source_legal_count": source_legal_count, "source_legal_rate": source_legal_count / len(frozen),
        "source_illegal_count": len(frozen) - source_legal_count,
        "saved_registry_domain_legal_count": sum(r["saved_checkpoint_domain_legal"] for r in selected_source_registry.values()),
        "tool_rejected_complete_assignments": sum(r["complete"] and not r["tool_admissible"]
                                                   for r in selected_source_registry.values()),
        "missing_or_malformed_output_count": sum(not r["complete"] for r in selected_source_registry.values()),
        "duplicate_field_output_count": sum("Duplicate directive assignment" in str(r.get("tool_rejection"))
                                             for r in selected_source_registry.values()),
        "tool_admissible_count": active_valid_count, "tool_admissible_rate": active_valid_count / len(frozen),
        "invalid_partial_partition_factors": invalid_factors,
        "exact_selected_winner_match_count": sum(r["exact_selected_winner_match"] for r in per_request),
        "exact_complete_configuration_agreement": sum(r["exact_selected_winner_match"] for r in per_request) / len(per_request),
        "near_optimal_match_count": sum(r["near_optimal_match"] for r in per_request),
        "complete_action_accuracy": _macro(per_request, "complete_action_accuracy"),
        "exact_measured_match_coverage": qor["exact_match_coverage"],
        "unknown_qor": qor["unknown_qor"], "measured_feasible_count": qor["measured_feasible"],
        "measured_infeasible_count": qor["measured_infeasible"],
        "conditional_feasible_regret": qor["mean_latency_regret_feasible"],
        "conditional_within_1pct": qor["within_1pct"], "conditional_within_5pct": qor["within_5pct"],
        "family_macro": qor["family_macro"], "per_family": {}, "per_kernel": {},
        "per_device": {}, "per_clock": {}, "per_budget": {}, "per_target_stratum": {},
        "decision_metrics": {
            "loop_action_accuracy": _macro(per_request, "loop_action_accuracy"),
            "array_action_accuracy": _macro(per_request, "array_action_accuracy"),
            "core_positive_recall_singleton": qor["nondefault_recall_single_reference_requests"],
        },
        "target_response": _transition_report(metric_requests, per_request),
        "qor_source": qor["qos_source"], "unmatched_qor_policy": "unknown; no estimate assigned",
        "measurement_compiler_identity": "Unknown in the authenticated GNΩSIS row schema; exact source/device/clock/configuration matches are reported separately from new Vitis synthesis.",
        "per_request": per_request,
    }
    for field, target in (("family", "per_family"), ("kernel", "per_kernel"),
                          ("device", "per_device"), ("clock_ns", "per_clock"), ("budget", "per_budget")):
        keys = sorted({json.dumps(row[field], sort_keys=True) for row in per_request})
        for key in keys:
            group = [row for row in per_request if json.dumps(row[field], sort_keys=True) == key]
            label = json.loads(key)
            result[target][str(label)] = _summarize_group(group)
    for row in per_request:
        key = f"{row['device']}@{row['clock_ns']:g}ns/{json.dumps(row['budget'], sort_keys=True)}"
        result["per_target_stratum"].setdefault(key, []).append(row)
    result["per_target_stratum"] = {key: _summarize_group(rows)
                                    for key, rows in result["per_target_stratum"].items()}
    report_path = Path(output_dir) / "reports" / f"{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    result = evaluate_prediction_file(args.name, args.predictions, output_dir=args.output_dir, policy_path=args.policy)
    keys = ("requests", "complete_assignment_rate", "source_legal_rate", "tool_admissible_rate",
            "near_optimal_match_count", "exact_measured_match_coverage", "conditional_feasible_regret")
    print(json.dumps({k: result[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
