"""Read-only scientific assessment of sealed refinement evidence; never runs HLS.

Authenticate/replay on the original numerical host. Portable verification checks
an exported seal and its input hashes, without claiming foreign-host replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys

from experiments.refinement import configurations as c, synthesis as hls
from experiments.refinement_search import benchmark as b, controller as ctl, policies as p

POLICIES = (*b.MAIN, "matched")
BUDGETS = (1, 2, 4, 8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_points(history, terminal, budgets=BUDGETS):
    baseline = history[0]["measurement"] if history else {}
    valid = ctl._usable(baseline) and baseline["ADP"] > 0
    points = []
    for budget in budgets:
        best, coverage = b._point(history, budget, terminal)
        adp = best["measurement"]["ADP"] if best else None
        ratio = adp / baseline["ADP"] if valid and adp is not None else None
        require(ratio is None or ratio <= 1, "Incumbent exceeds included baseline")
        points.append({"budget": budget, "ADP": adp, "ratio": ratio,
                       "improvement_pct": 100 * (1 - ratio) if ratio is not None else None,
                       "coverage": coverage})
    return points


def factorial(reference, a, z, joint):
    rows = (reference, a, z, joint)
    if not all(ctl._usable(row) for row in rows):
        return None
    if len({row.get("comparison_context_id") for row in rows}) != 1 or reference.get("comparison_context_id") is None:
        return None
    j0, ja, jb, jab = (row["ADP"] for row in rows)
    if j0 <= 0:
        return None
    return {"reference_A_B_AB_ADP": [j0, ja, jb, jab],
            "interaction_ADP": jab - ja - jb + j0,
            "joint_gain_fraction_of_reference": (min(j0, ja, jb) - jab) / j0}


def aggregate(contexts, kernels, policies=POLICIES):
    result = []
    for policy in policies:
        selected = [row for row in contexts if row["policy"] == policy]
        for budget in BUDGETS:
            values = [next(pt for pt in row["points"] if pt["budget"] == budget)["ratio"] for row in selected]
            complete = (len(selected) == len(kernels)
                        and {r["kernel"] for r in selected} == set(kernels)
                        and len({r["kernel"] for r in selected}) == len(selected)
                        and all(x is not None for x in values))
            wins = 0
            for row in selected:
                own = next(pt for pt in row["points"] if pt["budget"] == budget)["ratio"]
                peers = [r for r in contexts if r["kernel"] == row["kernel"] and r["policy"] in policies]
                ratios = [next(pt for pt in r["points"] if pt["budget"] == budget)["ratio"] for r in peers]
                if own is not None and {r["policy"] for r in peers} == set(policies) and all(v is not None for v in ratios):
                    wins += own == min(ratios)
            result.append({"policy": policy, "budget": budget, "complete": complete,
                "arithmetic_mean": statistics.mean(values) if complete else None,
                "geometric_mean": math.exp(statistics.mean(math.log(x) for x in values)) if complete and all(x > 0 for x in values) else None,
                "median": statistics.median(values) if complete else None,
                "kernels_improved": sum(x < 1 for x in values) if complete else None,
                "kernels_unchanged": sum(x == 1 for x in values) if complete else None,
                "kernels_best_including_ties": wins if complete else None})
    return result


def verify_functional(receipt, protocol, kernels):
    require(receipt["protocol_seal"] == protocol["seal"], "Functional protocol mismatch")
    require(receipt.get("executed") is True, "Functional status lacks execution")
    require(receipt["tool_identity"] == protocol["contexts"][receipt["kernel"]]["expected_tool_identity"], "Functional tool mismatch")
    kernel = kernels[receipt["kernel"]]
    require(c.signature(receipt["candidate_assignment"]) == receipt["signature"], "Functional assignment mismatch")
    evidence = receipt["evidence_sha256"]
    for filename, digest in evidence.items():
        require(c.sha256(c.repo_path(filename)) == digest, "Functional evidence changed: " + filename)
    def one(suffix):
        matches = [c.repo_path(f) for f in evidence if f.endswith(suffix)]
        require(len(matches) == 1, "Functional artifact missing/ambiguous: " + suffix)
        return matches[0]
    source = c.repo_path(kernel["source_provenance"]["snapshot_source"])
    require(receipt["source_sha256"] == c.sha256(source), "Functional reference identity mismatch")
    require(one("/reference/source.cpp").read_bytes() == source.read_bytes(), "Functional reference source mismatch")
    from LLM_branch.common import directive_domains as domains
    actions = domains.load_source_actions(receipt["kernel"], source.parent)
    materialized = hls.materialize(c.repo_path(kernel["template_path"]).read_text(), c.serialize(receipt["candidate_assignment"], actions))
    require(one("/candidate/source.cpp").read_text() == materialized, "Functional materialization mismatch")
    bench, plan = b._testbench(receipt["kernel"], kernel["top"])
    require(one("/testbench.cpp").read_text() == bench and receipt["plan"] == plan, "Functional testbench/plan mismatch")
    log = one("/csim.log").read_text(errors="replace")
    reference = one("/csim_reference/solution1/csim/build/validation_output.bin")
    candidate = one("/csim_candidate/solution1/csim/build/validation_output.bin")
    comparison = b._compare_outputs(reference, candidate, plan)
    require(comparison["mismatches"] == receipt.get("mismatches") and comparison["maximum_absolute_error"] == receipt.get("maximum_absolute_error"), "Functional comparison mismatch")
    require(receipt.get("returncode") == 0 and log.count("CSim done with 0 errors.") == 2,
            "Functional process/completion evidence failed")
    require(receipt["status"] == ("csim_pass" if comparison["mismatches"] == 0 else "csim_fail"), "Functional status mismatch")
    return {**receipt, "candidate_sha256": c.sha256(one("/candidate/source.cpp")),
            "testbench_sha256": c.sha256(one("/testbench.cpp")), "mismatch_count": comparison["mismatches"]}


def assess_context(entry, manifest, shared, state):
    journal, records, history, pending, terminal = state
    problem = ctl.problem_for(manifest, shared)
    identity = {key: entry[key] for key in ("kernel", "policy", "seed", "group")}
    by_sig = {row["signature"]: row for row in history}
    by_step = {row["step"]: row for row in records}
    queries, compositions, alternatives = [], [], []
    counts = Counter()
    for trace in (t for t in journal if t["event_type"] == "query" and t["status"] == "completed"):
        proposal = trace["proposal"]; measurement = by_step[trace["step"]]; result = measurement["result"]
        reference_row = by_sig[proposal["reference_signature"]]
        reference = reference_row["assignment"]
        before = by_sig.get(trace["incumbent_before"])
        expected_gain = bool(before and ctl._usable(result) and result["ADP"] < before["measurement"]["ADP"])
        require(trace["incumbent_improved"] == expected_gain, "Incorrect improvement flag")
        edits = {key: {"before": reference[key], "after": proposal["assignment"][key]}
                 for key in proposal["reference_changed_fields"]}
        for key, change in edits.items():
            kind = c.FIELD.fullmatch(key)[1]
            counts["field_" + kind] += 1
            if kind in ("UNROLL", "ARRAY_F"):
                direction = "disable" if int(change["after"]) == 0 else "increase" if int(change["after"]) > int(change["before"]) else "decrease"
                counts[direction] += 1
                if expected_gain:
                    counts["successful_" + direction] += 1
        new_values = proposal["domain_classification"]["changed_values_new_under_tcad"]
        introduced = {key: value for key, value in new_values.items() if key in edits}
        counts["expanded_configuration_queries"] += bool(new_values)
        counts["expanded_values_introduced_queries"] += bool(introduced)
        counts["improving_expanded_configuration_queries"] += bool(new_values) and expected_gain
        counts["improving_queries"] += expected_gain
        counts[proposal["proposal_type"]] += 1
        row = {**identity, "step": trace["step"], "signature": proposal["signature"],
            "query_id": trace["query_id"], "proposal_type": proposal["proposal_type"],
            "reference_signature": proposal["reference_signature"], "incumbent_before": trace["incumbent_before"],
            "incumbent_after": trace["incumbent_after"], "incumbent_improved": expected_gain,
            "assignment": proposal["assignment"], "reference_edits": edits,
            "net_baseline_edits": {key: {"before": problem["baseline"][key], "after": value}
                                   for key, value in proposal["assignment"].items() if value != problem["baseline"][key]},
            "changed_actions": proposal["reference_changed_actions"], "source_domain_legal": True,
            "domain_classification": proposal["domain_classification"], "expanded_values_introduced": introduced,
            "relation_evidence": proposal.get("relation_evidence"),
            "compiler_relation_type": proposal.get("compiler_relation_type", []),
            "matching_template_signature": proposal.get("matching_template_signature"),
            "matching_template_actions": proposal.get("matching_template_actions"),
            "matching_magnitude_distance_log2": proposal.get("matching_magnitude_distance_log2"),
            "action_domain_tuple_counts": {label: len(c.tcad.action_tuples(problem["actions"][label]))
                                           for label in proposal["reference_changed_actions"]},
            "matching_template_domain_tuple_counts": {label: len(c.tcad.action_tuples(problem["actions"][label]))
                                                      for label in proposal.get("matching_template_actions", [])},
            "result": result, "physical": {key: value for key, value in measurement.items() if key != "result"}}
        queries.append(row)
        if proposal["proposal_type"] == "composition":
            a, z = (by_sig[sig] for sig in proposal["constituent_signatures"])
            require(a["reference_signature"] == z["reference_signature"] == proposal["reference_signature"], "Composition references differ")
            require(len(set(a["reference_changed_actions"] + z["reference_changed_actions"])) == 2, "Composition does not join distinct actions")
            joint = factorial(reference_row["measurement"], a["measurement"], z["measurement"], result)
            compositions.append({**identity, "step": trace["step"], "signature": proposal["signature"],
                "constituents": proposal["constituent_signatures"], "reference_signature": proposal["reference_signature"],
                "factorial": joint, "increment_pct_original_baseline":
                    100 * (before["measurement"]["ADP"] - result["ADP"]) / history[0]["measurement"]["ADP"] if joint and expected_gain else 0.0})
        if manifest["policy"] == "hybrid" and proposal.get("allocation") == "reserved_structural_slot":
            prefix = history[:trace["step"]-1]
            incumbent = ctl._best(prefix)
            alternative = p.choose("compose", problem, prefix, incumbent["assignment"] if incumbent else problem["baseline"], manifest["seed"])
            alternatives.append({"step": trace["step"], "structural_signature": proposal["signature"],
                "structural_improved": expected_gain, "compose_next_type": alternative["proposal_type"] if alternative else None,
                "compose_next_signature": alternative["signature"] if alternative else None,
                "note": "Selection-only counterfactual on this hybrid history; no unobserved QoR inferred"})
    usable = [row for row in queries if ctl._usable(row["result"])]
    winner = min(usable, key=lambda row: row["result"]["ADP"]) if usable else None
    cost = {"logical_evaluations": len(history) + int(pending is not None),
        "unique_assignments": len({row["signature"] for row in history}),
        "new_physical_syntheses": sum(row["new_physical_synthesis_runs"] for row in records),
        "authenticated_reuses": sum(row["authenticated_measurement_reuses"] for row in records),
        "HLS_runtime_seconds": sum(row["physical_synthesis_runtime_seconds"] for row in records),
        "controller_authentication_persistence_seconds": sum(row["controller_runtime_seconds"] for row in journal if row["event_type"] == "query" and row["status"] == "completed"),
        "selection_only_seconds": None, "authentication_persistence_only_seconds": None,
        "end_to_end_elapsed_seconds": None,
        "recorded_unique_candidate_runtime_proxy_seconds": sum(row["result"].get("hls_runtime_seconds", 0) for row in queries),
        "feasible": len(usable),
        "infeasible": sum(row["result"].get("synthesis_status") == "success" and row["result"].get("feasible") is False for row in queries),
        "failed": sum(row["result"]["synthesis_status"] not in ("success", "unavailable") for row in queries),
        "unavailable": sum(row["result"]["synthesis_status"] == "unavailable" for row in queries),
        "timeouts": sum(row["result"]["synthesis_status"] == "timeout" for row in queries)}
    return {**identity, "termination": terminal, "pending": pending is not None,
        "points": normalized_points(history, terminal), "cost": cost, "counts": dict(counts),
        "baseline_result": records[0]["result"] if records else None,
        "best_signature": winner["signature"] if winner else None,
        "best_result": winner["result"] if winner else None,
        "matching_distances": [row["matching_magnitude_distance_log2"] for row in queries if row["proposal_type"] == "matched_control"],
        "compositions": compositions, "hybrid_opportunities": alternatives}, queries


def authenticate(output, live_tool=None):
    output = Path(output); protocol = b.load_protocol(output)
    require(protocol["seeds"] == [123] and protocol["max_evaluations"] == 8,
            "This assessment is scoped to R11 seed 123 / budget 8")
    curves = ctl._read_sealed(output / "optimization_curve_data.json")
    environment = ctl._read_sealed(output / "execution_environment.json")
    require(curves["protocol_seal"] == environment["protocol_seal"] == protocol["seal"], "Export/environment protocol mismatch")
    artifacts = {c.display_path(output / name): c.sha256(output / name) for name in
                 ("protocol.json", "optimization_curve_data.json", "execution_environment.json", "search_traces.jsonl", "measurements.jsonl")}
    artifacts.update(protocol["core_sha256"])
    artifacts["experiments/refinement_search/benchmark.py"] = protocol["benchmark_sha256"]
    for filename, digest in curves["journal_sha256"].items():
        require(c.sha256(c.repo_path(filename)) == digest, "Export journal mismatch: " + filename)
        artifacts[filename] = digest
    contexts, queries, kernels = [], [], {}
    with b.runtime(protocol):
        for entry in protocol["runs"]:
            path = c.repo_path(entry["path"]); manifest, shared = ctl.load_run(path)
            state = b._checked_state(path, manifest, shared)
            context, paid = assess_context(entry, manifest, shared, state)
            contexts.append(context); queries += paid; kernels.update(shared["kernels"])
            artifacts[c.display_path(path / "manifest.json")] = c.sha256(path / "manifest.json")
            artifacts[manifest["shared_manifest"]] = manifest["shared_manifest_sha256"]
            artifacts.update(shared["input_sha256"])
            print(entry["kernel"] + "/" + entry["policy"] + ": exact replay and measurement authentication passed", flush=True)
    # Cross-policy equality requires source/signature plus actual baseline metrics.
    for name in protocol["kernels"]:
        selected = [row for row in contexts if row["kernel"] == name]
        require(len(selected) == 6 and {r["policy"] for r in selected} == set(POLICIES), "Incomplete policy coverage")
        baselines = [row["baseline_result"] for row in selected]
        for field in ("ADP", "latency_cycles", "resources", "area_proxy", "comparison_context_id", "canonical_signature", "tool_identity"):
            require(all(row[field] == baselines[0][field] for row in baselines), "Cross-policy baseline mismatch: " + field)
        actual = {(row["policy"], pt["budget"]): pt for row in selected for pt in row["points"]}
        for point in (x for x in curves["curves"] if x["kernel"] == name and x["budget"] in BUDGETS):
            reconstructed = actual[(point["policy"], point["budget"])]
            require(all(reconstructed[k] == point[k] for k in ("ADP", "ratio", "coverage")), "Sealed curve differs from authenticated replay")
    # Published combined journals must also equal the paid per-policy journals.
    combined = c.read_jsonl(output / "measurements.jsonl")
    lookup = {(r["kernel"], r["policy"], r["seed"], r["step"]): r for r in queries}
    require(len(combined) == len(lookup), "Combined measurement count differs")
    for row in combined:
        paid = lookup[(row["kernel"], row["policy"], row["seed"], row["step"])]
        require(row["result"] == paid["result"] and row["candidate_signature"] == paid["signature"], "Combined measurement differs from paid observation")
    combined_traces = [r for r in c.read_jsonl(output / "search_traces.jsonl") if r["event_type"] == "query"]
    require(len(combined_traces) == len(lookup), "Combined trace count differs")
    for row in combined_traces:
        paid = lookup[(row["kernel"], row["policy"], row["seed"], row["step"])]
        require(row["candidate_signature"] == paid["signature"] and row["proposal"]["assignment"] == paid["assignment"], "Combined candidate differs")
        require(row["measured_ADP"] == paid["result"].get("ADP") and row["measured_resources"] == paid["result"].get("resources") and row["measured_latency"] == paid["result"].get("latency_cycles"), "Combined trace QoR differs")
    functional = [verify_functional(row, protocol, kernels) for row in b._validation_records(output).values()]
    for row in functional:
        artifacts.update(row["evidence_sha256"])
    receipt_file = output / "functional/receipts.jsonl"
    if receipt_file.exists():
        artifacts[c.display_path(receipt_file)] = c.sha256(receipt_file)
    expected_tool = next(iter(protocol["contexts"].values()))["expected_tool_identity"]
    live_identity = hls.tool_identity(hls.tool_info(live_tool)) if live_tool else None
    require(live_identity is None or live_identity == expected_tool, "Live tool/header mismatch")
    physical = []
    for root in protocol["physical_cache_roots"]:
        ledger = c.repo_path(root) / "physical/attempts.jsonl"
        if ledger.exists():
            entries = c.read_jsonl(ledger)
            require(all(row["status"] != "reserved" for row in entries), "Unresolved physical reservation")
            artifacts[c.display_path(ledger)] = c.sha256(ledger)
            physical.append({"root": root, "attempts": len(entries), "statuses": dict(Counter(row["status"] for row in entries))})
    final = []
    for name in protocol["kernels"]:
        completed = [row for row in contexts if row["kernel"] == name and row["points"][-1]["ADP"] is not None]
        if not completed:
            continue
        winner = min(completed, key=lambda row: row["points"][-1]["ADP"])
        signature = winner["best_signature"]
        validation = next((r for r in functional if r["kernel"] == name and r["signature"] == signature), None)
        query = next(r for r in queries if r["kernel"] == name and r["signature"] == signature)
        final.append({"kernel": name, "policy": winner["policy"], "signature": signature,
            "policy_associations": [r["policy"] for r in completed if r["best_signature"] == signature],
            "assignment": query["assignment"], "net_baseline_edits": query["net_baseline_edits"],
            "baseline_result": winner["baseline_result"], "result": winner["best_result"],
            "improvement_pct": winner["points"][-1]["improvement_pct"], "functional": validation})
    complete = all(row["points"][-1]["ADP"] is not None and not row["pending"] for row in contexts)
    return ctl._sealed({"schema": "salt-refinement-assessment-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_seal": protocol["seal"], "protocol_git_commit": protocol["git_commit"],
        "protocol_sha256": c.sha256(output / "protocol.json"), "assessment_code_sha256": c.sha256(__file__),
        "execution_environment": environment, "assessment_environment": {"python": sys.version, "libc": platform.libc_ver()},
        "original_archive": c.display_path(output), "state": "COMPLETE_AUTHENTICATED" if complete and all(r["functional"] and r["functional"]["status"] == "csim_pass" for r in final) else "COMPLETE_WITH_FUNCTIONAL_VALIDATION_PENDING" if complete else "PARTIAL_RESUMABLE",
        "live_tool_identity": live_identity, "files_sha256": artifacts,
        "contexts": contexts, "queries": queries, "aggregate": aggregate(contexts, protocol["kernels"]),
        "finalists": final, "functional": functional, "physical_ledgers": physical,
        "unique_assignments_all_policies": len({(r["kernel"], r["signature"]) for r in queries}),
        "cost_note": "Physical execution depends on cache/order. Recorded unique-candidate runtime is a retrospective cold-cache workload proxy, not a fresh runtime comparison. Selection and authentication/persistence were not separately timed; end-to-end elapsed time unavailable."})


def verify_export(path):
    report = ctl._read_sealed(path)
    require(report["assessment_code_sha256"] == c.sha256(__file__), "Assessment implementation changed")
    for filename, digest in report["files_sha256"].items():
        require(c.sha256(c.repo_path(filename)) == digest, "Assessment input changed: " + filename)
    return report


def compiler_diagnostic(report):
    """Supplementary verifier sensitivity; never changes a frozen relation.

    Dependency evidence is a set of graph edges. Its iteration order can differ
    with Python hash randomization. Compare the full edge multisets here only;
    exact policy replay remains unchanged and is performed on Coroni.
    """
    import networkx as nx
    from LLM_branch.common import directive_domains as domains
    from experiments.refinement import relations
    from experiments.structural_audit import audit
    protocol = b.load_protocol(c.repo_path(report["original_archive"]))
    rows, artifacts = [], {c.display_path(Path(audit.__file__)): c.sha256(audit.__file__)}
    with b.runtime(protocol):
        for name in protocol["kernels"]:
            entry = next(r for r in protocol["runs"] if r["kernel"] == name)
            _, shared = ctl.load_run(c.repo_path(entry["path"]))
            kernel = shared["kernels"][name]
            graph_path = c.repo_path(kernel["relations"]["artifact"])
            artifacts[c.display_path(graph_path)] = c.sha256(graph_path)
            graph = nx.read_gexf(graph_path)
            raw = domains.load_source_actions(name, c.repo_path(kernel["source_provenance"]["snapshot_metadata"]).parent)
            actual, inactive = relations.audit_kernel(kernel, raw)
            frozen = shared["relations"][name]
            require(inactive == shared["inactive_actions"][name] and len(actual) == len(frozen), "Compiler action/relation coverage differs")
            lookup = {}
            for current, saved in zip(actual, frozen):
                compare_relation_facts(current, saved)
                strict = audit.strict_verify_pair(graph, saved["action_i"], saved["action_j"],
                    artifact_hash=kernel["relations"]["artifact_hash"]) if "relevant_memory_root" in saved else saved
                row = {"kernel": name, "actions": [saved["action_i"], saved["action_j"]],
                    "historical_class": saved["verified_relation_class"], "diagnostic_class": strict["verified_relation_class"],
                    "kind": "array_loop" if "relevant_memory_root" in saved else "loop_loop",
                    "diagnostic_reason": strict["reason"],
                    "historical_support_operations": [r["text"] for r in saved["supporting_mlir_operations"]],
                    "diagnostic_support_operations": [r["text"] for r in strict["supporting_mlir_operations"]]}
                rows.append(row); lookup[frozenset(row["actions"])] = row
            for query in (r for r in report["queries"] if r["kernel"] == name and r["proposal_type"] in ("structural", "matched_control")):
                relation = lookup.get(frozenset(query["changed_actions"]))
                require(relation is not None, "Queried structural/control pair not in compiler audit")
                relation.setdefault("queried_by", []).append({"policy": query["policy"], "step": query["step"],
                    "signature": query["signature"], "improved": query["incumbent_improved"]})
    return ctl._sealed({"schema": "salt-r11-compiler-sensitivity-v1", "assessment_seal": report["seal"],
        "protocol_seal": report["protocol_seal"], "verifier_policy": audit.VERIFIER_POLICY,
        "files_sha256": artifacts, "original_relation_facts_reproduced": True,
        "comparison_note": "Original facts and dependence-edge multisets reproduced. This supplementary stricter audit changes no historical policy, relation, candidate, or QoR.",
        "relations": rows})


def compare_relation_facts(current, saved):
    require({k:v for k,v in current.items() if k != "dependence_evidence"} ==
            {k:v for k,v in saved.items() if k != "dependence_evidence"}, "Compiler relation facts differ")
    require(sorted(map(c.object_hash, current.get("dependence_evidence", []))) ==
            sorted(map(c.object_hash, saved.get("dependence_evidence", []))), "Compiler dependence evidence differs")


def write_tables(report, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    def write(name, rows):
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n"); writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k,v in row.items()})
    contexts = []
    for row in report["contexts"]:
        item = {k:row[k] for k in ("kernel", "policy", "seed", "termination")}
        for point in row["points"]:
            item.update({f"{key}@{point['budget']}": point[key] for key in ("ADP", "ratio", "improvement_pct", "coverage")})
        contexts.append({**item, **row["cost"], **row["counts"]})
    write("contexts.csv", contexts)
    write("aggregate.csv", report["aggregate"])
    queries = []
    for row in report["queries"]:
        result = row["result"]
        item = {k: row[k] for k in ("kernel", "policy", "seed", "step", "signature", "proposal_type", "reference_signature",
            "reference_edits", "net_baseline_edits", "changed_actions", "incumbent_improved", "matching_template_actions",
            "matching_magnitude_distance_log2", "action_domain_tuple_counts", "matching_template_domain_tuple_counts", "expanded_values_introduced")}
        item.update({k: result.get(k) for k in ("ADP", "latency_cycles", "area_proxy", "resources", "feasible", "synthesis_status", "estimated_clock_ns")})
        item.update({k:row["physical"][k] for k in ("new_physical_synthesis_runs", "authenticated_measurement_reuses", "physical_synthesis_runtime_seconds")})
        relation = row["relation_evidence"] or {}
        item.update(relation_class=relation.get("verified_relation_class"),
                    relation_types=row["compiler_relation_type"], evidence_graph=relation.get("evidence_artifact_hash"))
        queries.append(item)
    write("queries.csv", queries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(b.DEFAULT_OUTPUT))
    parser.add_argument("--output", help="Write a new sealed assessment JSON; never modifies input evidence")
    parser.add_argument("--verify-export", help="Portable seal/input-hash verification, without exact trace replay")
    parser.add_argument("--live-tool", help="Verify executable/header identity; does not synthesize or simulate")
    parser.add_argument("--compiler-output", help="Separate compiler-verifier sensitivity JSON (CPU torch/networkx required)")
    parser.add_argument("--tables-dir", help="Export compact CSV tables; optional original sealed curve PNG/PDF")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_export:
        report = verify_export(args.verify_export)
    else:
        require(args.output is not None, "Specify a fresh --output")
        target = Path(args.output).resolve(); source = Path(args.input_dir).resolve()
        require(target != source and source not in target.parents, "Assessment output must be outside frozen archive")
        require(not target.exists(), "Assessment output exists; preserve it and select a fresh path")
        report = authenticate(source, args.live_tool)
        target.parent.mkdir(parents=True, exist_ok=True)
        c.write_json(target, report)
    if args.compiler_output:
        target = Path(args.compiler_output)
        require(not target.exists(), "Compiler diagnostic output exists; preserve it")
        c.write_json(target, compiler_diagnostic(report))
    if args.tables_dir:
        write_tables(report, args.tables_dir)
        if args.plot:
            curves = ctl._read_sealed(c.repo_path(report["original_archive"]) / "optimization_curve_data.json")
            protocol = b.load_protocol(c.repo_path(report["original_archive"]))
            b._plot(Path(args.tables_dir), protocol, curves["curves"])
    print(report["state"], len(report["queries"]), "logical observations,", len(report["functional"]), "executed functional checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
