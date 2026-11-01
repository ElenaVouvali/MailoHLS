"""FGR-v1 CLI; delegates all accounting, authentication and synthesis to the frozen engine."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

from experiments.refinement_search import controller as ctl, policies as base
from . import POLICIES
from . import policies as fgr


def _code_hashes(original):
    result = original()
    package = Path(__file__).parent
    for path in sorted(package.glob("*.py")):
        result[str(path.relative_to(ctl.ROOT))] = ctl.c.sha256(path)
    return result


@contextmanager
def runtime():
    old_hashes, old_choose, old_names, old_history, old_candidate, old_summary, old_problem = ctl.code_hashes, base.choose, base.POLICIES, ctl._history_entry, ctl.physical_candidate, ctl.summarize_all, ctl.problem_for
    ctl.code_hashes = lambda: _code_hashes(old_hashes)
    def problem_for(manifest, shared):
        problem = old_problem(manifest, shared)
        return {**problem, "resource_budget": shared["context"]["resource_budget"],
                "device_capacities": shared["context"]["device_capacities"],
                "resource_keys": tuple(shared["context"]["resource_budget"])}
    ctl.problem_for = problem_for
    base.POLICIES = (*old_names, *POLICIES)
    base.choose = lambda policy, problem, history, incumbent, seed=123: (
        fgr.choose(policy, problem, history, incumbent, seed) if policy in POLICIES
        else old_choose(policy, problem, history, incumbent, seed))
    def history_entry(trace, record):
        row = old_history(trace, record)
        # These are the exact authenticated fields from the measurement already paid for.
        allowed = ("latency_cycles", "resources", "normalized_utilization", "area_proxy", "ADP",
                   "estimated_clock_ns", "feasible", "rankable", "synthesis_status", "comparison_context_id")
        return {**row, "measurement": {**row["measurement"],
                **{key: record["result"].get(key) for key in allowed}}}
    ctl._history_entry = history_entry
    def physical_candidate(manifest, shared, proposal):
        candidate = old_candidate(manifest, shared, proposal)
        # Proposal policy is sealed in shared inputs; physical identity continues
        # to use the unchanged engine implementation hash contract.
        from experiments.refinement import synthesis as hls
        kernel = shared["kernels"][manifest["kernel"]]
        candidate["evaluation_context_hash"] = ctl.c.object_hash({
            "source_context": hls.source_context(kernel), "signature": proposal["signature"],
            "search_code": old_hashes()})
        return candidate
    ctl.physical_candidate = physical_candidate
    def summarize_all(output):
        rows = old_summary(output)
        from experiments.refinement import configurations as c
        from experiments.refinement_search import controller as engine
        output = Path(output)
        details = []
        for manifest_path in sorted(output.glob("*/*/manifest.json")):
            manifest, shared = engine.load_run(manifest_path.parent)
            traces, records, history, _, terminal = engine.read_state(manifest_path.parent, manifest, shared)
            by_step = {r["step"]: r for r in records}
            baseline_measurement = history[0]["measurement"] if history else {}
            budget_rows = [row for row in rows if row["kernel"] == manifest["kernel"] and row["policy"] == manifest["policy"]]
            details.append({"kernel": manifest["kernel"], "policy": manifest["policy"], "seed": manifest["seed"],
                "terminal": terminal, "logical_queries": len(history),
                "single_queries": sum(len(r["reference_changed_actions"]) == 1 for r in history),
                "joint_queries": sum(len(r["reference_changed_actions"]) > 1 for r in history),
                "unique_candidates": len({r["signature"] for r in history}),
                "feasible_count": sum(r["measurement"].get("feasible") is True for r in history),
                "infeasible_count": sum(r["measurement"].get("feasible") is False for r in history),
                "outcomes": {status: sum(r["measurement"].get("synthesis_status") == status for r in history)
                    for status in ("success", "failed", "timeout", "unavailable")},
                "best_feasible_ADP": engine._best(history)["measurement"]["ADP"] if engine._best(history) else None,
                "budget_points": [{"budget": row["budget"], "coverage": row["status"],
                    "raw_ADP": row["best_ADP"], "normalized_ADP_ratio": row["best_ADP"] / baseline_measurement["ADP"]
                        if row["best_ADP"] is not None and baseline_measurement.get("ADP") else None,
                    "logical_queries": row["logical_evaluations"], "unique_candidates": row["unique_configurations"],
                    "new_physical_syntheses": row["new_physical_synthesis_runs"],
                    "authenticated_cache_reuses": row["authenticated_measurement_reuses"],
                    "recorded_synthesis_runtime_seconds": row["total_synthesis_runtime"],
                    "feasible_count": row["feasible_evaluations"], "failed_count": row["failures"],
                    "unavailable_count": row["unavailable"]} for row in budget_rows],
                "queries": [{"step": t["step"], "signature": t["candidate_signature"], "proposal_type": t["proposal_type"],
                    "selection_reason": t["selection_reason"], "changed_actions": t["changed_actions"],
                    "changed_fields": t["changed_fields"], "ADP": by_step[t["step"]]["result"].get("ADP"),
                    "feasible": by_step[t["step"]]["result"].get("feasible"),
                    "physical_synthesis_status": by_step[t["step"]].get("physical_synthesis_status"),
                    "runtime_seconds": by_step[t["step"]].get("physical_synthesis_runtime_seconds", 0),
                    "estimated_clock_ns": by_step[t["step"]]["result"].get("estimated_clock_ns"),
                    "detailed_diagnostics": "unavailable"} for t in traces if t["event_type"] == "query" and t["step"] in by_step]})
        c.write_json(output / "fgr_report.json", {"evidence_status": "only authenticated observations support hardware claims", "runs": details,
            "pool_comparison": "P0/P1/P2 use the same single_proposals + structurally verified structural_proposals generators; P2 also requires relation evidence for structural proposal generation."})
        return rows
    ctl.summarize_all = summarize_all
    try:
        yield
    finally:
        ctl.code_hashes, base.choose, base.POLICIES, ctl._history_entry, ctl.physical_candidate, ctl.summarize_all, ctl.problem_for = old_hashes, old_choose, old_names, old_history, old_candidate, old_summary, old_problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "replay", "run", "summarize"), default="prepare")
    parser.add_argument("--input-dir", default=str(ctl.DEFAULT_INPUT)); parser.add_argument("--output-dir", default=str(ctl.DEFAULT_OUTPUT.parent / "fgr_v1"))
    parser.add_argument("--measurements", action="append", default=[]); parser.add_argument("--kernels", nargs="+", default=["GAN", "covariance", "Kalman"])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES)); parser.add_argument("--execution", choices=("replay", "synthesis"), default="replay")
    parser.add_argument("--max-evaluations", type=int, default=8); parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fresh-synthesis", action="store_true"); parser.add_argument("--allow-batch", action="store_true")
    parser.add_argument("--limit", type=int); parser.add_argument("--vitis-hls", default="vitis_hls"); parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args(argv)
    args.kernels = list(dict.fromkeys("covariance" if k.lower() == "covariance" else k for k in args.kernels))
    try:
        with runtime():
            if args.mode == "prepare":
                ctl.prepare(args)
            elif args.mode == "summarize":
                ctl.summarize_all(Path(args.output_dir))
            else:
                if args.mode == "run" and len(args.kernels) * len(args.policies) > 1 and not args.allow_batch:
                    raise ValueError("Prospective batches require --allow-batch")
                for kernel in args.kernels:
                    for policy in args.policies:
                        path = Path(args.output_dir) / kernel / policy
                        manifest, shared = ctl.load_run(path)
                        expected = "replay" if args.mode == "replay" else "synthesis"
                        if manifest["execution"] != expected:
                            raise ValueError("Execution mode differs from prepared protocol")
                        ctl.search(path, executable=args.vitis_hls, timeout=args.timeout, limit=args.limit)
                ctl.summarize_all(Path(args.output_dir))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print("FGR-v1: " + str(exc), file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
