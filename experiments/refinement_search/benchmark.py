"""Sealed matched-budget evaluation of the existing refinement controller."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import combinations
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

from experiments.refinement import configurations as c, experiment as existing, synthesis as hls
from . import controller as ctl, policies as p

DEFAULT_OUTPUT = c.ROOT / "experiments/refinement/results/search/benchmark_r11"
MAIN = ("random", "coordinate", "compose", "structural", "hybrid")
VERIFIED = ("direct_verified", "indirect_verified")
VALIDATION = ("not_tested", "csim_pass", "csim_fail", "equivalence_verified")
CORE_FILES = ("__init__.py", "controller.py", "policies.py")
SMALL_GAIN_PCT = 0.5


def core_hashes():
    """Hash actual original engine files; the benchmark is sealed separately.

    R1 used a directory glob before this file existed. Excluding this wrapper
    preserves that exact contract without returning expected/stored hashes.
    No source, compiler, validator or controller hash is skipped.
    """
    result = existing.implementation_hashes()
    result.update({c.display_path(Path(__file__).parent / name): c.sha256(Path(__file__).parent / name)
                   for name in CORE_FILES})
    return result


def _edit_profile(problem, reference, proposal):
    profiles = []
    for label in proposal["reference_changed_actions"]:
        fields = [key for key in proposal["reference_changed_fields"] if key.endswith("_" + label + "}")]
        kinds = tuple(sorted(c.FIELD.fullmatch(key)[1] for key in fields))
        numerical = [abs(math.log2(max(1, int(proposal["assignment"][key])) / max(1, int(reference[key]))))
                     for key in fields if c.FIELD.fullmatch(key)[1] in ("UNROLL", "ARRAY_F")]
        profiles.append((problem["actions"][label].kind, kinds, sum(numerical)))
    return sorted(profiles)


def matched_proposals(problem, reference, seed=123):
    """Match source-only structural templates, without accessing their outcomes."""
    templates = p.structural_proposals(problem, reference, seed)
    if not templates:
        return []
    profiles = {_profile_key(_edit_profile(problem, reference, row)) for row in templates}
    relation_by_pair = {frozenset((r["action_i"], r["action_j"])): r for r in problem["relations"]}
    singles = p.single_proposals(problem, reference, seed)
    candidates = []
    for left, right in combinations(singles, 2):
        labels = left["reference_changed_actions"] + right["reference_changed_actions"]
        if len(set(labels)) != 2:
            continue
        relation = relation_by_pair.get(frozenset(labels))
        if relation and relation["verified_relation_class"] in VERIFIED:
            continue
        joined = sorted(_edit_profile(problem, reference, left) + _edit_profile(problem, reference, right))
        if _profile_key(joined) not in profiles:
            continue
        updates = {key: row["assignment"][key] for row in (left, right) for key in row["reference_changed_fields"]}
        row = p._proposal(problem, reference, {**reference, **updates}, "matched_control",
            "Two-action control matched to a source-only structural template; no verified relation is not proof of independence",
            control_label="no_verified_relation", relation_evidence=relation,
            compiler_relation_type=[])
        if row:
            candidates.append(row)
    result, seen = [], set()
    # Greedy one-to-one matching in the same deterministic template order.
    for template in templates:
        target = _edit_profile(problem, reference, template)
        target_novel = bool(template["domain_classification"]["changed_values_new_under_tcad"])
        eligible = []
        for row in candidates:
            if row["signature"] in seen:
                continue
            profile = _edit_profile(problem, reference, row)
            if [(a, b) for a, b, _ in profile] != [(a, b) for a, b, _ in target]:
                continue  # Exact action types, field families and counts.
            if bool(row["domain_classification"]["changed_values_new_under_tcad"]) != target_novel:
                continue
            distance = sum(abs(a[2] - b[2]) for a, b in zip(profile, target))
            eligible.append((distance, p._tie(seed, row["signature"]), row))
        if eligible:
            distance, _, row = min(eligible, key=lambda item: item[:2])
            result.append({**row, "matching_template_signature": template["signature"],
                "matching_template_actions": template["reference_changed_actions"],
                "matching_magnitude_distance_log2": distance,
                "matching_note": "Types, changed field families/counts and domain novelty match exactly; factor magnitude is nearest available, not necessarily equal"})
            seen.add(row["signature"])
    return result


def _profile_key(profile):
    return tuple((kind, fields) for kind, fields, _ in profile)


def choose_matched(problem, history, incumbent, seed):
    if not history:
        return p.choose("coordinate", problem, history, incumbent, seed)
    seen = {row["signature"] for row in history}
    return next((row for row in matched_proposals(problem, incumbent, seed) if row["signature"] not in seen), None)


def _cached_measurement(protocol, manifest, shared, proposal, own_root):
    """A hidden physical cache, consulted only inside the charged oracle call."""
    candidate = ctl.physical_candidate(manifest, shared, proposal)
    kernel = shared["kernels"][manifest["kernel"]]
    key = manifest["kernel"] + ":" + proposal["signature"]
    for directory in protocol["physical_cache_roots"]:
        root = c.repo_path(directory)
        if root.resolve() == Path(own_root).resolve():
            continue  # The engine owns durable reservation/recovery accounting here.
        journal = root / "physical/measurements.jsonl"
        if journal.is_file():
            for entry in reversed(c.read_jsonl(journal)):
                result = entry["result"]
                if entry["key"] != key or result.get("synthesis_status") != "success":
                    continue
                ctl._verify_result({"result": result}, candidate, kernel, shared["expected_tool_identity"])
                return {**result, "benchmark_reuse_origin": c.display_path(journal)}
    return None


@contextmanager
def runtime(protocol=None):
    """Scoped adapter; engine bytes remain unchanged and are strictly checked.

    The only selector extension is the explicit matched control. The original
    five selectors and all validation/accounting/execution functions are reused.
    """
    old_hashes, old_choose, old_policies, old_measure = ctl.code_hashes, p.choose, p.POLICIES, ctl.measure
    ctl.code_hashes = core_hashes
    p.POLICIES = (*MAIN, "matched")
    def choose(policy, problem, history, incumbent, seed=123):
        return choose_matched(problem, history, incumbent, seed) if policy == "matched" else old_choose(policy, problem, history, incumbent, seed)
    p.choose = choose
    def measure(manifest, shared, proposal, query_id, root, executable, timeout):
        result = _cached_measurement(protocol, manifest, shared, proposal, root) if protocol and not manifest["fresh_synthesis"] else None
        if result is not None:
            return result, {"physical_synthesis_status": "benchmark_authenticated_reuse", "new_physical_synthesis_runs": 0,
                "authenticated_measurement_reuses": 1, "physical_synthesis_runtime_seconds": 0.0}
        return old_measure(manifest, shared, proposal, query_id, root, executable, timeout)
    ctl.measure = measure
    try:
        yield
    finally:
        ctl.code_hashes, p.choose, p.POLICIES, ctl.measure = old_hashes, old_choose, old_policies, old_measure


def _initial_direction_audit(problem, seed):
    result = {}
    for name, rows in (("coordinate", p.single_proposals(problem, problem["baseline"], seed)),
                       ("structural", p.structural_proposals(problem, problem["baseline"], seed))):
        counts = {"increase": 0, "decrease": 0, "disable": 0}
        for row in rows:
            for field in row["reference_changed_fields"]:
                if c.FIELD.fullmatch(field)[1] not in ("UNROLL", "ARRAY_F"):
                    continue
                old, new = int(problem["baseline"][field]), int(row["assignment"][field])
                counts["disable" if new == 0 else "increase" if new > old else "decrease"] += 1
        result[name] = {"counts": counts, "first_four_signatures": [r["signature"] for r in rows[:4]]}
    return result


def prepare(args):
    output = c.repo_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark directory is nonempty; resume it or use a new protocol directory")
    if args.max_evaluations not in (1, 2, 4, 8):
        raise ValueError("Benchmark supports budgets 1, 2, 4, 8")
    output.mkdir(parents=True, exist_ok=True)
    runs, roots, direction_audit, baseline_contexts = [], [], {}, {}
    with runtime():
        for seed in args.seeds:
            adopted = c.repo_path(args.search_dir) if args.search_dir and seed == args.seeds[0] else None
            if adopted:
                for name in args.kernels:
                    for policy in MAIN:
                        path = adopted / name / policy
                        manifest, shared = ctl.load_run(path)
                        if (manifest["seed"], manifest["max_evaluations"], manifest["execution"]) != (seed, args.max_evaluations, args.execution):
                            raise ValueError("Adopted search differs from requested seed, budget or execution")
                roots.append(adopted)
                groups = [(adopted, MAIN, "main"), (output / f"control_seed_{seed}", ("matched",), "control")]
            else:
                groups = [(output / f"seed_{seed}", (*MAIN, "matched"), None)]
            for root, identities, group in groups:
                if root != adopted:
                    ctl.prepare(argparse.Namespace(input_dir=args.input_dir, output_dir=str(root), measurements=args.measurements,
                        kernels=args.kernels, policies=list(identities), max_evaluations=args.max_evaluations,
                        seed=seed, execution=args.execution, fresh_synthesis=False))
                    roots.append(root)
                for name in args.kernels:
                    for policy in identities:
                        path = root / name / policy
                        manifest, shared = ctl.load_run(path)
                        traces, records, history, pending, terminal = ctl.read_state(path, manifest, shared)
                        if pending:
                            raise ValueError("Finish or inspect the pending adopted query before freezing the benchmark")
                        kernel = shared["kernels"][name]
                        context = {**manifest["request"], "baseline_signature": manifest["baseline_signature"],
                            "source_context": hls.source_context(kernel), "expected_tool_identity": shared["expected_tool_identity"]}
                        if name in baseline_contexts and baseline_contexts[name] != context:
                            raise ValueError("Policies or seeds do not share identical source/baseline/deployment contexts")
                        baseline_contexts[name] = context
                        runs.append({"kernel": name, "policy": policy, "seed": seed, "group": group or ("control" if policy == "matched" else "main"),
                            "path": c.display_path(path), "manifest_sha256": c.sha256(path / "manifest.json"),
                            "initial_trace_sha256": c.sha256(path / "search_trace.jsonl"),
                            "initial_measurements_sha256": c.sha256(path / "measurements.jsonl"), "initial_logical_queries": len(history)})
                        if policy == "coordinate":
                            direction_audit[f"{seed}/{name}"] = _initial_direction_audit(ctl.problem_for(manifest, shared), seed)
        imports = []
        for directory in args.import_search:
            root = c.repo_path(directory)
            paths = list(root.glob("*/*/manifest.json"))
            if not paths:
                raise ValueError("Import search has no sealed policy contexts")
            for filename in paths:
                m, shared = ctl.load_run(filename.parent)
                ctl.read_state(filename.parent, m, shared)
                if hls.source_context(shared["kernels"][m["kernel"]]) != baseline_contexts[m["kernel"]]["source_context"]:
                    raise ValueError("Imported search source context differs")
                imports.append({"path": c.display_path(filename), "sha256": c.sha256(filename)})
            roots.append(root)
    worktree = subprocess.check_output(["git", "status", "--porcelain"], cwd=c.ROOT, text=True)
    protocol = ctl._sealed({"experiment": "matched-budget-refinement", "protocol_version": args.protocol_version,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip(),
        "initial_worktree_status": worktree, "benchmark_sha256": c.sha256(__file__), "core_sha256": core_hashes(),
        "execution": args.execution, "seeds": args.seeds, "kernels": args.kernels, "runs": runs,
        "budgets": [b for b in (1, 2, 4, 8) if b <= args.max_evaluations], "max_evaluations": args.max_evaluations,
        "maximum_main_logical_queries": len(args.kernels) * len(args.seeds) * len(MAIN) * args.max_evaluations,
        "maximum_control_logical_queries": len(args.kernels) * len(args.seeds) * args.max_evaluations,
        "contexts": baseline_contexts, "physical_cache_roots": list(dict.fromkeys(c.display_path(r) for r in roots)),
        "imported_search_manifests": imports, "initial_direction_audit": direction_audit,
        "rules": {"baseline_cost": 1, "reuse_logical_cost": 1, "failed_logical_cost": 1,
            "policy_order": list(MAIN) + ["matched"], "schedule": "round-robin one query per selected context",
            "composition_sibling_batch": p.BATCH_SIZE, "hybrid_structural_slots": [4, 8],
            "matched_control": "Exact action types/field families/counts/domain novelty; nearest log2 magnitude; no verified relation, not semantic independence",
            "direction_order": "Existing balanced lower/upper/disable order unchanged; lower-first resets can still bias short trajectories",
            "small_gain_reporting_pct": SMALL_GAIN_PCT, "missing_replay": "Unavailable; incomplete curve, never numerical QoR",
            "early_termination": "Carry only an explicitly exhausted, completely observed neighborhood",
            "functional_validation": "Separate final-design C simulation; three deterministic input patterns, not RTL equivalence",
            "timeouts_seconds": args.timeout}})
    c.write_json(output / "protocol.json", protocol)
    summarize(output)
    print(f"Sealed {len(runs)} contexts: at most {protocol['maximum_main_logical_queries']} main + {protocol['maximum_control_logical_queries']} control logical queries. No synthesis launched.")


def load_protocol(output):
    protocol = ctl._read_sealed(Path(output) / "protocol.json")
    if protocol["benchmark_sha256"] != c.sha256(__file__) or protocol["core_sha256"] != core_hashes():
        raise ValueError("Frozen benchmark or original engine changed; use a new protocol directory")
    with runtime(protocol):
        for run in protocol["runs"]:
            path = c.repo_path(run["path"])
            if c.sha256(path / "manifest.json") != run["manifest_sha256"]:
                raise ValueError("Search manifest changed")
            manifest, shared = ctl.load_run(path)
            if (manifest["seed"], manifest["policy"], manifest["kernel"], manifest["max_evaluations"], manifest["execution"]) != (run["seed"], run["policy"], run["kernel"], protocol["max_evaluations"], protocol["execution"]):
                raise ValueError("Search identity or budget differs from sealed protocol")
        for entry in protocol["imported_search_manifests"]:
            if c.sha256(c.repo_path(entry["path"])) != entry["sha256"]:
                raise ValueError("Imported search manifest changed")
    return protocol


def run(args):
    output = c.repo_path(args.output_dir); protocol = load_protocol(output)
    expected = "replay" if args.mode == "replay" else "synthesis"
    if protocol["execution"] != expected:
        raise ValueError("Requested execution differs from the sealed protocol")
    selected = [r for r in protocol["runs"] if (not args.kernels or r["kernel"] in args.kernels)
                and (not args.policies or r["policy"] in args.policies)
                and (args.groups == "all" or r["group"] == args.groups)]
    if not selected:
        raise ValueError("No selected benchmark contexts")
    if args.mode == "run" and len(selected) > 1 and not args.allow_batch:
        raise ValueError("Multiple prospective contexts require --allow-batch after preflight and smoke validation")
    done = 0
    with runtime(protocol):
        while True:
            progress = False
            for entry in selected:
                if args.limit is not None and done >= args.limit:
                    summarize(output); return
                path = c.repo_path(entry["path"]); manifest, shared = ctl.load_run(path)
                _, _, before, pending, terminal = ctl.read_state(path, manifest, shared)
                if terminal:
                    continue
                ctl.search(path, executable=args.vitis_hls, timeout=protocol["rules"]["timeouts_seconds"], limit=1)
                _, _, after, _, terminal = ctl.read_state(path, manifest, shared)
                done += len(after) - len(before)
                progress = progress or len(after) > len(before)
            summarize(output)
            if not progress:
                break


def _point(history, budget, terminal):
    reached = len(history) >= budget
    carried = terminal == "neighborhood_exhausted" and bool(history) and not reached
    rows = history[:budget] if reached else history if carried else []
    known = bool(rows) and all(r["measurement"]["synthesis_status"] != "unavailable" for r in rows)
    best = ctl._best(rows) if known else None
    return (best, "carried_early_termination" if carried and known else "complete" if reached and known else "incomplete")


def _validation_records(output):
    path = Path(output) / "functional/receipts.jsonl"
    result = {}
    if path.is_file():
        for row in c.read_jsonl(path):
            if row["status"] not in VALIDATION or row["status"] == "equivalence_verified":
                raise ValueError("Unsupported or unverified functional-validation claim")
            for filename, digest in row.get("evidence_sha256", {}).items():
                if c.sha256(c.repo_path(filename)) != digest:
                    raise ValueError("Functional validation evidence changed")
            if row["status"] != "not_tested" and not row.get("executed"):
                raise ValueError("Functional status lacks an executed validation")
            result[(row["kernel"], row["signature"])] = row
    return result


def _checked_state(path, manifest, shared):
    state = ctl.read_state(path, manifest, shared)
    traces, records, history, _, _ = state
    by_step = {row["step"]: row for row in records}
    for trace in traces:
        if trace["event_type"] != "query" or trace["status"] != "completed":
            continue
        result = by_step[trace["step"]]["result"]
        expected = {"measured_latency": result.get("latency_cycles"), "measured_resources": result.get("resources"),
                    "measured_ADP": result.get("ADP"), "feasibility": result.get("feasible")}
        if any(trace[key] != value for key, value in expected.items()):
            raise ValueError("Trace QoR differs from its authenticated measurement")
    return state


def summarize(output, plot=False):
    output = Path(output); protocol = load_protocol(output)
    validation = _validation_records(output)
    summaries, traces, measurements, curves = [], [], [], []
    with runtime(protocol):
        for entry in protocol["runs"]:
            path = c.repo_path(entry["path"]); manifest, shared = ctl.load_run(path)
            journal, records, history, pending, terminal = _checked_state(path, manifest, shared)
            queries = [r for r in journal if r["event_type"] == "query" and r["status"] == "completed"]
            baseline = history[0]["measurement"] if history and ctl._usable(history[0]["measurement"]) else None
            identity = {k: entry[k] for k in ("seed", "kernel", "policy", "group")}
            for trace in journal:
                traces.append({**trace, **identity})
            for record in records:
                key = (entry["kernel"], record["candidate_signature"])
                result = record["result"]
                gain = 100 * (baseline["ADP"] - result["ADP"]) / baseline["ADP"] if baseline and ctl._usable(result) else None
                measurements.append({**record, **identity, "revealed_to_policy": record["step"] <= len(history),
                    "improvement_over_baseline_pct": gain, "small_gain_below_0_5pct": gain is not None and 0 < gain < SMALL_GAIN_PCT,
                    "validation_status": validation.get(key, {}).get("status", "not_tested")})
            best = ctl._best(history)
            best_result = next((r["result"] for r in records if best and r["candidate_signature"] == best["signature"]), {})
            row = {**identity}
            for budget in (1, 2, 4, 8):
                point, coverage = _point(history, budget, terminal)
                row[f"ADP@{budget}"] = point["measurement"]["ADP"] if point else None
                row[f"coverage@{budget}"] = coverage
            valid8, _ = _point(history, 8, terminal)
            improvement = 100 * (baseline["ADP"] - valid8["measurement"]["ADP"]) / baseline["ADP"] if valid8 and baseline else None
            numerical = {"increase": 0, "decrease": 0, "disable": 0}
            composition_gain = 0.0
            for trace in queries:
                proposal = trace["proposal"]
                reference = next((h["assignment"] for h in history if h["signature"] == proposal["reference_signature"]), proposal["assignment"])
                for field in proposal["reference_changed_fields"]:
                    if c.FIELD.fullmatch(field)[1] in ("UNROLL", "ARRAY_F"):
                        before, after = int(reference[field]), int(proposal["assignment"][field])
                        numerical["disable" if after == 0 else "increase" if after > before else "decrease"] += 1
                if trace["proposal_type"] == "composition" and trace["incumbent_improved"] and baseline:
                    previous = next(h["measurement"]["ADP"] for h in history if h["signature"] == trace["incumbent_before"])
                    composition_gain += 100 * (previous - trace["measured_ADP"]) / baseline["ADP"]
            row.update({"improvement@8_pct": improvement, "small_gain_below_0.5pct": improvement is not None and 0 < improvement < SMALL_GAIN_PCT,
                "logical_HLS_calls": len(history) + int(pending is not None),
                "new_physical_syntheses": sum(r["new_physical_synthesis_runs"] for r in records),
                "authenticated_reuses": sum(r["authenticated_measurement_reuses"] for r in records),
                "total_HLS_runtime_seconds": sum(r["physical_synthesis_runtime_seconds"] for r in records),
                "controller_runtime_seconds": sum(r["controller_runtime_seconds"] for r in queries),
                "feasible": sum(ctl._usable(h["measurement"]) for h in history),
                "infeasible": sum(h["measurement"]["synthesis_status"] == "success" and h["measurement"].get("feasible") is False for h in history),
                "failed": sum(h["measurement"]["synthesis_status"] not in ("success", "unavailable") for h in history),
                "unavailable": sum(h["measurement"]["synthesis_status"] == "unavailable" for h in history),
                "single_action_proposals": sum(h["proposal_type"] in ("coordinate", "random") for h in history),
                "composed_proposals": sum(h["proposal_type"] == "composition" for h in history),
                "structural_proposals": sum(h["proposal_type"] == "structural" for h in history),
                "matched_control_proposals": sum(h["proposal_type"] == "matched_control" for h in history),
                "tcad_expanded_value_proposals": sum(bool(h["domain_classification"]["changed_values_new_under_tcad"]) for h in history),
                "incumbent_improvement_fraction": sum(r["incumbent_improved"] for r in queries) / max(1, len(history) - 1),
                "composition_increment_pct_of_baseline": composition_gain,
                "numerical_increases": numerical["increase"], "numerical_decreases": numerical["decrease"], "numerical_disables": numerical["disable"],
                "distinct_configurations": len({h["signature"] for h in history}), "early_termination": terminal == "neighborhood_exhausted",
                "termination": terminal or ("pending" if pending else "paused"), "best_observed_ADP": best_result.get("ADP"),
                "best_signature": best["signature"] if best else None, "best_latency": best_result.get("latency_cycles"),
                **{"best_" + resource: best_result.get("resources", {}).get(resource) for resource in c.RESOURCE_KEYS},
                "validation_status": validation.get((entry["kernel"], best["signature"] if best else None), {}).get("status", "not_tested")})
            summaries.append(row)
            for budget in range(1, protocol["max_evaluations"] + 1):
                point, coverage = _point(history, budget, terminal)
                curves.append({**identity, "budget": budget, "ADP": point["measurement"]["ADP"] if point else None,
                               "ratio": point["measurement"]["ADP"] / baseline["ADP"] if point and baseline else None, "coverage": coverage})
    c.write_jsonl(output / "search_traces.jsonl", traces)
    c.write_jsonl(output / "measurements.jsonl", measurements)
    ctl._write_csv(output / "summary.csv", summaries)
    lines = ["# Controlled matched-budget refinement pilot", "", f"Protocol {protocol['protocol_version']}; execution {protocol['execution']}; seeds {protocol['seeds']}.",
        "", "| Kernel | Policy | Seed | ADP@1 | ADP@2 | ADP@4 | ADP@8 | Improvement@8 % | Logical calls | New syntheses | HLS seconds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def display(value):
        return "unavailable" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
    for row in summaries:
        keys = ("kernel", "policy", "seed", "ADP@1", "ADP@2", "ADP@4", "ADP@8", "improvement@8_pct", "logical_HLS_calls", "new_physical_syntheses", "total_HLS_runtime_seconds")
        lines.append("| " + " | ".join(display(row[key]) for key in keys) + " |")
    lines += ["", f"Main logical queries: {sum(r['logical_HLS_calls'] for r in summaries if r['group']=='main')}/{protocol['maximum_main_logical_queries']}. "
        f"Additional matched-control queries: {sum(r['logical_HLS_calls'] for r in summaries if r['group']=='control')}/{protocol['maximum_control_logical_queries']}.",
        f"New physical syntheses: {sum(r['new_physical_syntheses'] for r in summaries)}; authenticated reuses: {sum(r['authenticated_reuses'] for r in summaries)}; "
        f"new HLS runtime: {sum(r['total_HLS_runtime_seconds'] for r in summaries):.3f} s; per-query controller overhead: {sum(r['controller_runtime_seconds'] for r in summaries):.3f} s.",
        f"Unavailable replay observations: {sum(r['unavailable'] for r in summaries)}; early-terminated searches: {sum(r['early_termination'] for r in summaries)}.",
        "", "## Interpretation and ablations", "",
        "A: Compare coordinate with compose at equal budgets. Compose also uses common-reference sibling sweeps, so the policy contrast is not a pure composition-toggle ablation. The CSV separately records the measured incremental incumbent gain contributed by explicit compositions.",
        "B: Compare structural with matched. Both change two actions; control templates match action types, changed fields and domain novelty exactly, with nearest available numerical magnitude. Incumbents may diverge, so actual comparisons are exploratory. No verified relationship does not prove independence. Matching scores and evidence remain in traces.",
        "C: The existing numerical order already offers decreases and increases; it was not retuned. Initial pool direction counts are sealed in the protocol, and observed direction counts are in the CSV. Lower-first ordering restarts after improvements, so a short trajectory may still favor decreases.",
        "D: Compare hybrid with compose, its exploitation-only counterpart. Reserved structural slots cost logical budget. A joint improvement is not automatically evidence that compiler guidance caused it.",
        "", "| Kernel | Seed | Comparison (positive favors second) | Additional gain % of baseline |", "| --- | --- | --- | ---: |"]
    for name in protocol["kernels"]:
        for seed in protocol["seeds"]:
            lookup = {r["policy"]: r for r in summaries if r["kernel"] == name and r["seed"] == seed}
            for first, second in (("coordinate", "compose"), ("matched", "structural"), ("compose", "hybrid")):
                a, z = lookup.get(first), lookup.get(second)
                gain = 100 * (a["ADP@8"] - z["ADP@8"]) / a["ADP@1"] if a and z and a["ADP@8"] is not None and z["ADP@8"] is not None and a["ADP@1"] else None
                lines.append(f"| {name} | {seed} | {first} → {second} | {display(gain)} |")
    lines += ["", f"Explicit compositions evaluated: {sum(r['composed_proposals'] for r in summaries)}; "
        f"sum of within-trajectory measured incumbent increments: {sum(r['composition_increment_pct_of_baseline'] for r in summaries):.6g} percentage points. "
        "This diagnostic sums repeated policy observations and is not an independent-sample effect estimate.",
        "", "## Finalists and validation", ""]
    for name in protocol["kernels"]:
        eligible = [r for r in summaries if r["kernel"] == name and r["ADP@8"] is not None]
        if eligible:
            winner = min(eligible, key=lambda r: r["ADP@8"])
            lines.append(f"{name}: best completed observed policy {winner['policy']}, ADP {winner['ADP@8']:.9g}, improvement {winner['improvement@8_pct']:.6g}%, "
                f"latency {winner['best_latency']}, resources { {k: winner['best_'+k] for k in c.RESOURCE_KEYS} }, validation {winner['validation_status']}.")
        else:
            lines.append(name + ": complete eight-evaluation comparison unavailable; research decision pending.")
    lines += ["", "Positive improvements below 0.5% are flagged as small, physically unvalidated gains; this convention is not a significance test. "
        "C simulation on finite test inputs is not RTL equivalence. No post-route timing, area or power benefit is established.",
        "Partial replay cannot rank policies. Plots leave unavailable prefixes blank; only explicit neighborhood exhaustion carries measured results forward. "
        "Cross-kernel aggregation uses ADP/baseline ratios with complete matched kernel coverage, never averaged raw ADP.",
        "Three development kernels and one seed cannot establish universal superiority. Large feasible gains are candidates for later physical validation; tiny gains need particular caution. "
        "Choose a learned selector only after separating individual, composition, structural and expanded-domain outcomes."]
    functional = list(validation.values())
    if functional:
        lines.append(f"Functional validation: {len(functional)} selected designs, {sum(r.get('runtime_seconds',0) for r in functional):.3f} seconds, separate from all selection-time HLS budgets.")
    if plot and any(r["ADP"] is not None for r in curves):
        _plot(output, protocol, curves)
        lines.append("Optimization curves: optimization_curves.png (and standalone PDF).")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summaries


def _plot(output, protocol, curves):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mailohls-refinement-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(protocol["kernels"]) + 1, figsize=(5 * (len(protocol["kernels"]) + 1), 4), squeeze=False)
    for index, name in enumerate(protocol["kernels"]):
        ax = axes[0, index]
        for policy in (*MAIN, "matched"):
            for seed in protocol["seeds"]:
                rows = [r for r in curves if r["kernel"] == name and r["policy"] == policy and r["seed"] == seed]
                ax.plot([r["budget"] for r in rows], [r["ratio"] if r["ratio"] is not None else math.nan for r in rows], marker="o", markersize=3, label=f"{policy}/{seed}")
        ax.set(title=name, xlabel="Logical HLS evaluations", ylabel="Best ADP / original SALT ADP")
        ax.set_xticks(protocol["budgets"]); ax.grid(alpha=.25)
    ax = axes[0, -1]
    for policy in (*MAIN, "matched"):
        for seed in protocol["seeds"]:
            values = []
            for budget in range(1, protocol["max_evaluations"] + 1):
                rows = [r for r in curves if r["policy"] == policy and r["seed"] == seed and r["budget"] == budget]
                complete = len(rows) == len(protocol["kernels"]) and all(r["ratio"] is not None for r in rows)
                values.append(sum(r["ratio"] for r in rows) / len(rows) if complete else math.nan)
            ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=f"{policy}/{seed}")
    ax.set(title="Mean normalized ratio (complete kernels)", xlabel="Logical HLS evaluations", ylabel="Mean ADP / baseline ratio")
    ax.set_xticks(protocol["budgets"]); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "optimization_curves.png", dpi=180); fig.savefig(output / "optimization_curves.pdf"); plt.close(fig)


def _testbench(name, top):
    common = '''#include <cstdio>
#include <cstdint>
#include <cstring>
static float sample(int pattern, int i) {
  if(pattern==0) return 0.0f;
  if(pattern==1) return i%2 ? -0.25f : 0.25f;
  uint32_t x=123u+uint32_t(i)*1664525u+1013904223u;
  return (float(x%1024u)-512.0f)/1024.0f;
}
'''
    if name == "GAN":
        declaration = f"void {top}(float*,float*);\n"
        body = f'''static float x[392], y[392];
for(int pattern=0;pattern<3;pattern++) {{
 for(int i=0;i<392;i++) x[i]=sample(pattern,i);
 {top}(x,y);
 if(std::fwrite(y,sizeof(float),392,f)!=392) return 3;
}}'''
        count, width, atol, rtol = 3 * 392, 4, 1e-6, 1e-6
    elif name == "covariance":
        declaration = f'extern "C" void {top}(int,int,double,double[100][80],double[80][80],double[80]);\n'
        body = f'''static double data[100][80], cov[80][80], mean[80];
for(int pattern=0;pattern<3;pattern++) {{
 for(int i=0;i<100;i++) for(int j=0;j<80;j++) data[i][j]=sample(pattern,i*80+j);
 {top}(80,100,100.0,data,cov,mean);
 if(std::fwrite(data,sizeof(double),8000,f)!=8000) return 3;
 if(std::fwrite(cov,sizeof(double),6400,f)!=6400) return 3;
 if(std::fwrite(mean,sizeof(double),80,f)!=80) return 3;
}}'''
        count, width, atol, rtol = 3 * (8000 + 6400 + 80), 8, 1e-9, 1e-9
    elif name == "Kalman":
        declaration = '#include "ap_int.h"\n' + f'extern "C" void {top}(ap_uint<256>[32768],ap_uint<256>[32768]);\n'
        body = f'''static ap_uint<256> input[32768], output[32768];
for(int pattern=0;pattern<3;pattern++) {{
 for(int i=0;i<262144;i++) {{
  float value=sample(pattern,i); uint32_t bits; std::memcpy(&bits,&value,4);
  input[i/8].range((i%8+1)*32-1,(i%8)*32)=bits;
 }}
 {top}(input,output);
 for(int i=0;i<262144;i++) {{
  uint32_t bits=output[i/8].range((i%8+1)*32-1,(i%8)*32).to_uint();
  float value; std::memcpy(&value,&bits,4);
  if(std::fwrite(&value,sizeof(float),1,f)!=1) return 3;
 }}
}}'''
        count, width, atol, rtol = 3 * 262144, 4, 1e-6, 1e-6
    else:
        raise ValueError("No functional testbench defined for this kernel")
    program = common + declaration + '\nint main(){\nFILE *f=std::fopen("validation_output.bin","wb");\nif(!f) return 2;\n' + body + '\nreturn std::fclose(f)==0 ? 0 : 4;\n}\n'
    return program, {"patterns": ["zero", "alternating +/-0.25", "seed-123 bounded deterministic integer sequence"],
                     "output_values": count, "element_bytes": width, "absolute_tolerance": atol, "relative_tolerance": rtol}


def _write_once(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.encode() if isinstance(data, str) else data
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Existing functional preparation differs: " + str(path))
    if not path.exists():
        path.write_bytes(payload)


def _compare_outputs(reference, candidate, plan):
    size = plan["output_values"] * plan["element_bytes"]
    if reference.stat().st_size != size or candidate.stat().st_size != size:
        raise ValueError("Functional test did not produce the expected number of outputs")
    fmt = "=f" if plan["element_bytes"] == 4 else "=d"
    first = struct.iter_unpack(fmt, reference.read_bytes()); second = struct.iter_unpack(fmt, candidate.read_bytes())
    maximum, mismatches = 0.0, 0
    for (left,), (right,) in zip(first, second):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Nonfinite C-simulation output")
        error = abs(left - right); maximum = max(maximum, error)
        mismatches += error > plan["absolute_tolerance"] + plan["relative_tolerance"] * abs(left)
    return {"maximum_absolute_error": maximum, "mismatches": mismatches}


def validate(args):
    output = c.repo_path(args.output_dir); protocol = load_protocol(output)
    rows = summarize(output); previous = _validation_records(output)
    receipts = list(previous.values())
    tool = hls.tool_info(args.vitis_hls) if args.execute_csim else None
    for name in protocol["kernels"]:
        if args.kernels and name not in args.kernels:
            continue
        eligible = [r for r in rows if r["kernel"] == name and r["ADP@8"] is not None]
        if not eligible:
            print(name + ": final-design validation pending complete measured budget point", flush=True); continue
        winner = min(eligible, key=lambda r: r["ADP@8"])
        key = (name, winner["best_signature"])
        if key in previous and previous[key]["status"] in ("csim_pass", "csim_fail"):
            continue
        entry = next(r for r in protocol["runs"] if all(r[k] == winner[k] for k in ("kernel", "policy", "seed")))
        with runtime(protocol):
            manifest, shared = ctl.load_run(c.repo_path(entry["path"]))
            traces, records, history, _, _ = ctl.read_state(c.repo_path(entry["path"]), manifest, shared)
            best = next(row for row in history if row["signature"] == winner["best_signature"])
            candidate = ctl.physical_candidate(manifest, shared, best)
        kernel = shared["kernels"][name]
        if tool and hls.tool_identity(tool) != shared["expected_tool_identity"]:
            raise ValueError("Functional tool/header identity differs from synthesis")
        directory = output / "functional" / (name + "-" + winner["best_signature"][:16])
        source = c.repo_path(kernel["source_provenance"]["snapshot_source"])
        _write_once(directory / "reference/source.cpp", source.read_bytes())
        _write_once(directory / "candidate/source.cpp", hls.materialize(c.repo_path(kernel["template_path"]).read_text(), candidate["target"]))
        for file in kernel["source_provenance"]["files"]:
            original = c.repo_path(file["snapshot"])
            if original.suffix.lower() not in (".h", ".hpp", ".hh", ".inc") or original.name in hls.VENDOR_HEADERS:
                continue
            for side in ("reference", "candidate"):
                _write_once(directory / side / original.relative_to(source.parent), original.read_bytes())
        testbench, plan = _testbench(name, kernel["top"])
        _write_once(directory / "testbench.cpp", testbench)
        tcl = ['set here [file dirname [file normalize [info script]]]', 'set headers [file join $::env(XILINX_HLS) include]']
        for side in ("reference", "candidate"):
            tcl += [f"open_project -reset csim_{side}", "set_top " + kernel["top"],
                f'add_files -cflags "-std=c++11 -I$headers" [file join $here {side} source.cpp]',
                'add_files -tb -cflags "-std=c++11 -I$headers" [file join $here testbench.cpp]',
                "open_solution -reset solution1 -flow_target vivado", "set_part " + hls.tcl_quote(c.PART),
                "create_clock -period 10 -name default", "csim_design -clean", "close_project"]
        tcl.append("exit")
        _write_once(directory / "run_csim.tcl", "\n".join(tcl) + "\n")
        receipt = {"kernel": name, "signature": winner["best_signature"], "status": "not_tested", "executed": False,
            "protocol_seal": protocol["seal"], "plan": plan, "runtime_seconds": 0.0,
            "source_sha256": c.sha256(source), "candidate_assignment": candidate["assignment"],
            "evidence_sha256": {c.display_path(path): c.sha256(path) for path in directory.rglob("*") if path.is_file()}}
        if args.execute_csim:
            import resource
            def stack_limit():
                _, hard = resource.getrlimit(resource.RLIMIT_STACK)
                resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
            number = 1
            while (directory / f"attempt_{number:03}").exists():
                number += 1
            attempt = directory / f"attempt_{number:03}"; attempt.mkdir()
            log = attempt / "csim.log"; start = time.monotonic(); error = None
            with log.open("w") as stream:
                process = subprocess.Popen([tool["executable"], "-f", str(directory / "run_csim.tcl")], cwd=attempt,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=stack_limit)
                try:
                    code = process.wait(timeout=protocol["rules"]["timeouts_seconds"])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(); code = -1; error = "C simulation timed out"
            receipt.update(executed=True, status="csim_fail", returncode=code, runtime_seconds=time.monotonic() - start,
                           tool_identity=hls.tool_identity(tool), stack_policy="Inherited hard stack limit for original and candidate")
            reference = attempt / "csim_reference/solution1/csim/build/validation_output.bin"
            tested = attempt / "csim_candidate/solution1/csim/build/validation_output.bin"
            try:
                if code != 0:
                    raise ValueError(error or "Vitis C simulation returned nonzero")
                comparison = _compare_outputs(reference, tested, plan)
                receipt.update(comparison, status="csim_pass" if comparison["mismatches"] == 0 else "csim_fail")
            except (ValueError, OSError) as exc:
                receipt["error"] = str(exc)
            receipt["evidence_sha256"].update({c.display_path(path): c.sha256(path) for path in (log, reference, tested) if path.is_file()})
        receipts = [r for r in receipts if (r["kernel"], r["signature"]) != key] + [receipt]
        c.write_jsonl(output / "functional/receipts.jsonl", receipts)
        print(name + ": " + receipt["status"], flush=True)
    summarize(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "replay", "run", "summarize", "validate"), default="prepare")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--input-dir", default=str(ctl.DEFAULT_INPUT))
    parser.add_argument("--search-dir", help="Adopt and resume the original five prepared searches for the first seed")
    parser.add_argument("--import-search", action="append", default=[])
    parser.add_argument("--measurements", action="append", default=[])
    parser.add_argument("--kernels", nargs="+")
    parser.add_argument("--policies", nargs="+", choices=(*MAIN, "composition", "matched"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--max-evaluations", type=int, default=8)
    parser.add_argument("--execution", choices=("replay", "synthesis"), default="replay")
    parser.add_argument("--protocol-version", default="1")
    parser.add_argument("--groups", choices=("main", "control", "all"), default="all")
    parser.add_argument("--allow-batch", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--vitis-hls", default="vitis_hls")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--execute-csim", action="store_true")
    args = parser.parse_args(argv)
    try:
        if sys.version_info >= (3, 12):
            raise ValueError("Use Python 3.8–3.11 for exact historical ADP arithmetic/authentication")
        args.seeds = list(dict.fromkeys(args.seeds))
        if args.kernels:
            args.kernels = list(dict.fromkeys("covariance" if k.lower() == "covariance" else k for k in args.kernels))
        if args.policies:
            args.policies = list(dict.fromkeys("compose" if k == "composition" else k for k in args.policies))
        if args.limit is not None and args.limit <= 0 or args.timeout <= 0:
            raise ValueError("Limits and timeouts must be positive")
        if args.mode == "prepare":
            args.kernels = args.kernels or ["GAN", "covariance", "Kalman"]
            prepare(args)
        elif args.mode in ("replay", "run"):
            run(args)
        elif args.mode == "validate":
            validate(args)
        else:
            summarize(c.repo_path(args.output_dir), args.plot)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print("Refinement benchmark: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
