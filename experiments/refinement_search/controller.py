"""Budgeted search with isolated logical histories and an authenticated HLS cache."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from LLM_branch.common import directive_domains as domains
from experiments.refinement import configurations as c, experiment as existing, synthesis as hls
from . import policies

ROOT = c.ROOT
DEFAULT_INPUT = ROOT / "experiments/refinement/results/local_refinement"
DEFAULT_OUTPUT = ROOT / "experiments/refinement/results/search/replay"
BUDGETS = (1, 2, 4, 8, 16)


def code_hashes():
    result = existing.implementation_hashes()
    result.update({c.display_path(p): c.sha256(p) for p in Path(__file__).parent.glob("*.py")})
    return result


def _sealed(payload):
    return {**payload, "seal": c.object_hash(payload)}


def _read_sealed(path):
    data = json.loads(Path(path).read_text())
    if data.get("seal") != c.object_hash({k: v for k, v in data.items() if k != "seal"}):
        raise ValueError("Manifest seal mismatch: " + str(path))
    return data


def _copy_kernel(name, original, output):
    kernel = copy.deepcopy(original)
    mapping = {}
    parent = Path(original["source_provenance"]["snapshot_source"]).parent
    for entry in kernel["source_provenance"]["files"]:
        source = c.repo_path(entry["snapshot"])
        target = output / "sources" / name / Path(entry["snapshot"]).relative_to(parent)
        if c.sha256(source) != entry["sha256"]:
            raise ValueError("Source snapshot hash mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        mapping[entry["snapshot"]] = entry["snapshot"] = c.display_path(target)
    for key in ("snapshot_source", "snapshot_metadata"):
        kernel["source_provenance"][key] = mapping[original["source_provenance"][key]]
    target = output / "sources" / name / "source_template.txt"
    target.write_bytes(c.repo_path(original["template_path"]).read_bytes())
    kernel["template_path"] = c.display_path(target)
    relation = kernel["relations"]
    if relation["status"] == "available":
        target = output / "graphs" / (name + ".gexf")
        target.parent.mkdir(parents=True, exist_ok=True)
        if c.sha256(c.repo_path(relation["artifact"])) != relation["artifact_hash"]:
            raise ValueError("Graph hash mismatch")
        target.write_bytes(c.repo_path(relation["artifact"]).read_bytes())
        relation["artifact"] = c.display_path(target)
        if relation.get("compiler_active_metadata_path"):
            target = output / "graphs" / (name + "_active") / "kernel_info.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(c.repo_path(relation["compiler_active_metadata_path"]).read_bytes())
            relation["compiler_active_metadata_path"] = c.display_path(target)
    return kernel


def prepare(args):
    """Freeze source facts and a hidden physical cache, never a winning design."""
    output, source = c.repo_path(args.output_dir).resolve(), c.repo_path(args.input_dir).resolve()
    archives = list(dict.fromkeys([str(source), *[str(c.repo_path(p).resolve()) for p in args.measurements]]))
    for path in archives:
        archive = Path(path)
        if output == archive or output in archive.parents or archive in output.parents:
            raise ValueError("Search output must be separate from historical inputs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty; resume it or prepare a fresh experiment directory")
    if not 1 <= args.max_evaluations <= 16 or args.execution not in ("replay", "synthesis"):
        raise ValueError("Supported logical limit is 1..16; execution is replay or synthesis")
    original, _ = existing.load_output(source)
    if original["context"] != c.CONTEXT or original["domain_policy"] != "tcad":
        raise ValueError("Search currently supports authenticated TCAD ADP/ZCU104/10ns/full-capacity inputs only")
    names = args.kernels
    if any(name not in original["kernels"] for name in names):
        raise ValueError("Kernel absent from the selected authenticated input")
    if any(name not in policies.POLICIES for name in args.policies):
        raise ValueError("Unknown policy")
    output.mkdir(parents=True, exist_ok=True)
    shared_dir = output / "shared"
    kernels, native, relations, inactive, baselines = {}, {}, {}, {}, {}
    from experiments.refinement import relations as compiler
    for name in names:
        kernel = _copy_kernel(name, original["kernels"][name], shared_dir)
        raw = domains.load_source_actions(name, c.repo_path(kernel["source_provenance"]["snapshot_metadata"]).parent)
        if c.from_paper(kernel["paper_baseline"], raw) != kernel["initial_assignment"]:
            raise ValueError("Initial assignment differs from the original complete SALT prediction")
        verified, missing = compiler.audit_kernel(kernel, raw)
        actions = c.compiler_actions(kernel, raw)
        baseline = kernel["initial_assignment"]
        c.policy_record(baseline, baseline, actions, original["native_domains"][name], "tcad")
        kernels[name], native[name], relations[name], inactive[name] = kernel, original["native_domains"][name], verified, missing
        baselines[name] = dict(baseline)
    # Cache authentication is not a policy observation. Selection receives none
    # of the cached signatures, metrics, winners or historical candidate pools.
    inventory, audit = {}, []
    for archive in archives:
        incoming, checks, _ = existing.read_measurements(archive)
        inventory.update(incoming); audit += checks
    cache, identities = {}, {}
    for (name, signature), proof in inventory.items():
        if name not in kernels or hls.source_context(proof["kernel"]) != hls.source_context(kernels[name]):
            continue
        identity = proof["verified"]["tool_identity"]
        identities[c.object_hash(identity)] = identity
        if not args.fresh_synthesis:
            cache[name + ":" + signature] = {
                **proof["result"], **proof["verified"], "canonical_signature": signature,
                "authentication": "authenticated_reuse", "reuse_proof": existing.copy_measurement(proof, shared_dir)}
    if len(identities) > 1:
        raise ValueError("Input caches use different tool/header builds; select a consistent context")
    files = [p for p in shared_dir.rglob("*") if p.is_file()]
    shared = _sealed({"context": c.CONTEXT, "kernels": kernels, "native_domains": native,
        "relations": relations, "inactive_actions": inactive, "baselines": baselines,
        "cache": cache, "cache_authentication": audit,
        "expected_tool_identity": next(iter(identities.values())) if identities else None,
        "input_sha256": {c.display_path(p): c.sha256(p) for p in files},
        "code_sha256": code_hashes(), "initial_input_manifest_sha256": c.sha256(source / "manifest.json")})
    c.write_json(shared_dir / "manifest.json", shared)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in names:
        for policy in args.policies:
            path = output / name / policy
            path.mkdir(parents=True)
            kernel = kernels[name]
            manifest = _sealed({"experiment_id": "budgeted-refinement-search", "git_commit": commit,
                "created_utc": datetime.now(timezone.utc).isoformat(), "policy": policy, "seed": args.seed,
                "execution": args.execution, "fresh_synthesis": args.fresh_synthesis,
                "max_evaluations": args.max_evaluations, "kernel": name,
                "request": {"kernel": name, "objective": "ADP", "device": c.CONTEXT["device"],
                    "part": c.PART, "clock_period_ns": c.CLOCK_NS, "resource_capacities": c.CAPACITIES,
                    "resource_budgets": c.CAPACITIES, "domain_policy": domains.tcad.SOURCE_DOMAIN_POLICY,
                    "source_sha256": kernel["source_provenance"]["source_sha256"],
                    "metadata_sha256": kernel["source_provenance"]["metadata_sha256"],
                    "compiler_graph_sha256": kernel["relations"].get("artifact_hash"),
                    "toolchain": hls.HLS_VERSION, "maximum_evaluations": args.max_evaluations},
                "baseline_signature": c.signature(baselines[name]),
                "shared_manifest": c.display_path(shared_dir / "manifest.json"),
                "shared_manifest_sha256": c.sha256(shared_dir / "manifest.json"),
                "policy_rules": {"coordinate": "immediate greedy reanchor", "composition_sibling_batch": policies.BATCH_SIZE,
                    "hybrid_structural_logical_slots": list(range(policies.STRUCTURAL_PERIOD, 17, policies.STRUCTURAL_PERIOD)),
                    "numerical_order": "nearest lower and upper, then disable; round-robin families", "random": "seeded signature hash order"}})
            c.write_json(path / "manifest.json", manifest)
            c.write_jsonl(path / "search_trace.jsonl", [])
            c.write_jsonl(path / "measurements.jsonl", [])
    summarize_all(output)
    print(f"Prepared {len(names) * len(args.policies)} isolated searches; {len(cache)} hidden authenticated cache entries. No synthesis launched.")


def load_run(path):
    path = Path(path)
    manifest = _read_sealed(path / "manifest.json")
    shared_path = c.repo_path(manifest["shared_manifest"])
    if c.sha256(shared_path) != manifest["shared_manifest_sha256"]:
        raise ValueError("Shared input manifest changed")
    shared = _read_sealed(shared_path)
    if shared["code_sha256"] != code_hashes() or shared["context"] != c.CONTEXT:
        raise ValueError("Frozen implementation or deployment changed; prepare a fresh search")
    for filename, digest in shared["input_sha256"].items():
        if c.sha256(c.repo_path(filename)) != digest:
            raise ValueError("Frozen source/cache evidence changed: " + filename)
    if not 1 <= manifest["max_evaluations"] <= 16 or manifest["policy"] not in policies.POLICIES:
        raise ValueError("Unsupported search policy or budget")
    name = manifest["kernel"]; request = manifest["request"]
    expected_request = {"kernel": name, "objective": "ADP", "device": c.CONTEXT["device"],
        "part": c.PART, "clock_period_ns": c.CLOCK_NS, "resource_capacities": c.CAPACITIES,
        "resource_budgets": c.CAPACITIES, "domain_policy": domains.tcad.SOURCE_DOMAIN_POLICY,
        "source_sha256": shared["kernels"][name]["source_provenance"]["source_sha256"],
        "metadata_sha256": shared["kernels"][name]["source_provenance"]["metadata_sha256"],
        "compiler_graph_sha256": shared["kernels"][name]["relations"].get("artifact_hash"),
        "toolchain": hls.HLS_VERSION, "maximum_evaluations": manifest["max_evaluations"]}
    if request != expected_request or manifest["execution"] not in ("replay", "synthesis"):
        raise ValueError("Unsupported request or inconsistent source/deployment context")
    if manifest["baseline_signature"] != c.signature(shared["baselines"][name]):
        raise ValueError("Initial SALT configuration changed")
    return manifest, shared


def problem_for(manifest, shared):
    """Only source facts, never the hidden cache or historical QoR, enter policy code."""
    name = manifest["kernel"]; kernel = shared["kernels"][name]
    raw = domains.load_source_actions(name, c.repo_path(kernel["source_provenance"]["snapshot_metadata"]).parent)
    return {"baseline": shared["baselines"][name], "actions": c.recorded_actions(kernel, raw),
            "native_domains": shared["native_domains"][name], "relations": shared["relations"][name],
            "inactive_actions": shared["inactive_actions"][name]}


def physical_candidate(manifest, shared, proposal):
    name = manifest["kernel"]; signature = proposal["signature"]
    kernel = shared["kernels"][name]
    raw = domains.load_source_actions(name, c.repo_path(kernel["source_provenance"]["snapshot_metadata"]).parent)
    return {"kernel": name, "candidate_id": name.lower() + "-" + signature[:16],
        "candidate_group": "baseline" if signature == manifest["baseline_signature"] else "refinement",
        "assignment": proposal["assignment"], "canonical_signature": signature,
        "target": c.serialize(proposal["assignment"], raw),
        "evaluation_context_hash": c.object_hash({"source_context": hls.source_context(kernel),
            "signature": signature, "search_code": shared["code_sha256"]})}


def _usable(result):
    return (result.get("synthesis_status") == "success" and result.get("rankable") is True
            and result.get("feasible") is True and result.get("ADP") is not None
            and math.isfinite(result["ADP"]))


def _history_entry(trace, record):
    # The policy can see precisely the outcomes it paid for, and nothing else.
    fields = ("ADP", "rankable", "feasible", "synthesis_status", "comparison_context_id")
    return {**trace["proposal"], "measurement": {key: record["result"].get(key) for key in fields}}


def _best(history):
    valid = [row for row in history if _usable(row["measurement"])]
    # Stable min retains the earlier incumbent on equal measured ADP.
    return min(valid, key=lambda r: r["measurement"]["ADP"]) if valid else None


def _check_context(history, result):
    if result.get("synthesis_status") != "success":
        return
    contexts = {r["measurement"]["comparison_context_id"] for r in history
                if r["measurement"]["synthesis_status"] == "success"}
    context = result.get("comparison_context_id")
    if context is None or contexts and contexts != {context}:
        raise ValueError("Search observations have mismatched evaluation contexts")


def _verify_result(record, candidate, kernel, expected_tool):
    if record.get("candidate_signature", candidate["canonical_signature"]) != candidate["canonical_signature"]:
        raise ValueError("Logical measurement signature mismatch")
    result = record["result"]
    if result["synthesis_status"] == "unavailable":
        if result.get("ADP") is not None or result.get("feasible") is not None or result.get("rankable"):
            raise ValueError("Unavailable replay queries cannot contain invented QoR")
        return
    if result["synthesis_status"] == "success":
        verified = existing.verify_measurement(result, candidate, kernel)
        if expected_tool and verified["tool_identity"] != expected_tool:
            raise ValueError("Measurement tool/header context differs")
    elif result.get("ADP") is not None or result.get("rankable"):
        raise ValueError("Failed synthesis cannot provide a numerical objective")


def read_state(path, manifest, shared):
    """Reconstruct and verify decisions from a policy's own journals."""
    traces = c.read_jsonl(Path(path) / "search_trace.jsonl")
    records = c.read_jsonl(Path(path) / "measurements.jsonl")
    by_step = {r["step"]: r for r in records}
    if len(by_step) != len(records):
        raise ValueError("Duplicate logical measurement steps")
    history, pending, terminal = [], None, None
    problem = problem_for(manifest, shared)
    for index, trace in enumerate(traces):
        if trace["event_type"] == "termination":
            if index != len(traces) - 1:
                raise ValueError("Termination must be the last trace entry")
            terminal = trace["reason"]; break
        step = len(history) + 1
        if trace["step"] != step or trace["logical_evaluation_cost"] != 1:
            raise ValueError("Nonsequential or unpaid logical query")
        best = _best(history)
        incumbent = best["assignment"] if best else problem["baseline"]
        expected = policies.choose(manifest["policy"], problem, history, incumbent, manifest["seed"])
        if trace["proposal"] != expected or trace["candidate_signature"] in {r["signature"] for r in history}:
            raise ValueError("Trace is not the deterministic policy's next eligible query")
        if trace["candidate_signature"] != expected["signature"] or trace["incumbent_before"] != (best["signature"] if best else None):
            raise ValueError("Trace incumbent/signature mismatch")
        actions, fields = c.changed(incumbent, expected["assignment"], problem["actions"])
        if (trace["changed_actions"] != actions or trace["changed_fields"] != fields
                or trace["proposal_type"] != expected["proposal_type"]
                or trace["selection_reason"] != expected["selection_reason"]
                or trace["relation_evidence"] != expected.get("relation_evidence")):
            raise ValueError("Trace explanation differs from the selected proposal")
        candidate = physical_candidate(manifest, shared, expected)
        c.policy_record(expected["assignment"], problem["baseline"], problem["actions"], problem["native_domains"], "tcad")
        if trace["status"] == "pending":
            if index != len(traces) - 1:
                raise ValueError("Only the last query may be pending")
            pending = trace; break
        record = by_step.get(step)
        if record is None or record["candidate_signature"] != expected["signature"]:
            raise ValueError("Completed trace lacks its exact measurement")
        _verify_result(record, candidate, shared["kernels"][manifest["kernel"]], shared["expected_tool_identity"])
        _check_context(history, record["result"])
        history.append(_history_entry(trace, record))
        after = _best(history)
        if trace["incumbent_after"] != (after["signature"] if after else None) or trace["best_ADP_so_far"] != (after["measurement"]["ADP"] if after else None):
            raise ValueError("Recorded incumbent update is not the best feasible measured design")
    queries = [r for r in traces if r["event_type"] == "query"]
    if len(queries) > manifest["max_evaluations"]:
        raise ValueError("Logical budget exceeded")
    if set(by_step) - {r["step"] for r in queries}:
        raise ValueError("Measurement without a paid query")
    if terminal == "budget_exhausted" and len(history) != manifest["max_evaluations"]:
        raise ValueError("False budget termination")
    if terminal == "neighborhood_exhausted":
        best = _best(history)
        if policies.choose(manifest["policy"], problem, history, best["assignment"] if best else problem["baseline"], manifest["seed"]) is not None:
            raise ValueError("False neighborhood exhaustion")
    if terminal not in (None, "budget_exhausted", "neighborhood_exhausted"):
        raise ValueError("Unknown termination reason")
    return traces, records, history, pending, terminal


def measure(manifest, shared, proposal, query_id, root, executable, timeout):
    """Called only after a durable logical reservation; this is the sole oracle."""
    candidate = physical_candidate(manifest, shared, proposal)
    name, signature = manifest["kernel"], proposal["signature"]
    kernel = shared["kernels"][name]
    key = name + ":" + signature
    if not manifest["fresh_synthesis"] and key in shared["cache"]:
        result = {**shared["cache"][key], "candidate_id": candidate["candidate_id"],
                  "prepared_context_hash": candidate["evaluation_context_hash"]}
        existing.verify_measurement(result, candidate, kernel)
        return result, {"physical_synthesis_status": "authenticated_reuse", "new_physical_synthesis_runs": 0,
                        "authenticated_measurement_reuses": 1, "physical_synthesis_runtime_seconds": 0.0}
    if manifest["execution"] == "replay":
        return {"synthesis_status": "unavailable", "ADP": None, "feasible": None, "rankable": False,
                "error": "Exact candidate absent from the frozen authenticated archive"}, {
                    "physical_synthesis_status": "unavailable", "new_physical_synthesis_runs": 0,
                    "authenticated_measurement_reuses": 0, "physical_synthesis_runtime_seconds": 0.0}
    physical = Path(root) / "physical"
    physical.mkdir(exist_ok=True)
    with (physical / "cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path, results_path = physical / "attempts.jsonl", physical / "measurements.jsonl"
        ledger = c.read_jsonl(ledger_path) if ledger_path.exists() else []
        results = c.read_jsonl(results_path) if results_path.exists() else []
        tool = hls.tool_info(executable); identity = hls.tool_identity(tool)
        expected = shared["expected_tool_identity"] or (ledger[0]["tool_identity"] if ledger else identity)
        if identity != expected:
            raise ValueError("Prospective HLS tool/header build differs from the frozen context")
        reusable = [r for r in results if r["key"] == key and
                    (r["query_id"] == query_id or not manifest["fresh_synthesis"] and r["result"]["synthesis_status"] == "success")]
        recovered = False
        if reusable:
            entry = reusable[-1]; result = entry["result"]
        else:
            owned = [r for r in ledger if r["query_id"] == query_id]
            if owned:
                work = physical / "projects" / candidate["candidate_id"] / ("attempt-" + str(owned[-1]["work_attempt"]).zfill(3))
                saved = work / "result.json"
                if not saved.exists():
                    raise RuntimeError("A physical job is reserved without a durable result. Inspect its project/log before resuming; it will not be launched twice.")
                result = json.loads(saved.read_text()); recovered = True
            else:
                work_root = physical / "projects" / candidate["candidate_id"]
                number = 1
                while (work_root / ("attempt-" + str(number).zfill(3))).exists():
                    number += 1
                ledger.append({"query_id": query_id, "key": key, "work_attempt": number,
                    "tool_identity": identity, "status": "reserved", "started_utc": datetime.now(timezone.utc).isoformat()})
                c.write_jsonl(ledger_path, ledger)
                result = hls.run_candidate(candidate, kernel, physical, tool, timeout)
            result["canonical_signature"] = signature
            if result["synthesis_status"] == "success":
                result.update(hls.authenticate_work(candidate, result, kernel, c.repo_path,
                    expected_evaluation_hash=candidate["evaluation_context_hash"]), authentication="authenticated_new")
            else:
                result["authentication"] = "unsuccessful_new"
            entry = {"query_id": query_id, "key": key, "result": result}
            results.append(entry); c.write_jsonl(results_path, results)
            owned = [r for r in ledger if r["query_id"] == query_id]
            owned[-1].update(status=result["synthesis_status"], hls_runtime_seconds=result["hls_runtime_seconds"])
            c.write_jsonl(ledger_path, ledger)
        _verify_result({"result": result}, candidate, kernel, identity)
        own = entry["query_id"] == query_id
        return result, {"physical_synthesis_status": "recovered_synthesis" if recovered else "new_synthesis" if own else "shared_reuse",
            "new_physical_synthesis_runs": int(own), "authenticated_measurement_reuses": int(not own),
            "physical_synthesis_runtime_seconds": result.get("hls_runtime_seconds", 0) if own else 0}


def search(path, *, executable="vitis_hls", timeout=21600, limit=None):
    path = Path(path)
    with (path / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, shared = load_run(path)
        traces, records, history, pending, terminal = read_state(path, manifest, shared)
        problem = problem_for(manifest, shared)
        completed = 0
        while not terminal:
            if not pending and limit is not None and completed >= limit:
                break
            if not pending and len(history) >= manifest["max_evaluations"]:
                terminal = "budget_exhausted"; break
            started = time.monotonic()
            best = _best(history); incumbent = best["assignment"] if best else problem["baseline"]
            if pending:
                proposal = pending["proposal"]; trace = pending
            else:
                proposal = policies.choose(manifest["policy"], problem, history, incumbent, manifest["seed"])
                if proposal is None:
                    terminal = "neighborhood_exhausted"; break
                c.policy_record(proposal["assignment"], problem["baseline"], problem["actions"], problem["native_domains"], "tcad")
                if proposal["signature"] != c.signature(proposal["assignment"]) or proposal["signature"] in {r["signature"] for r in history}:
                    raise ValueError("Invalid signature or duplicate policy proposal")
                actions, fields = c.changed(incumbent, proposal["assignment"], problem["actions"])
                step = len(history) + 1
                trace = {"event_type": "query", "step": step, "policy": manifest["policy"], "status": "pending",
                    "incumbent_before": best["signature"] if best else None,
                    "candidate_signature": proposal["signature"], "changed_actions": actions, "changed_fields": fields,
                    "proposal_type": proposal["proposal_type"], "compiler_relation_type": proposal.get("compiler_relation_type", []),
                    "relation_evidence": proposal.get("relation_evidence"), "selection_reason": proposal["selection_reason"],
                    "logical_evaluation_cost": 1, "proposal": proposal,
                    "query_id": c.object_hash({"search": manifest["seal"], "step": step, "candidate": proposal["signature"]})}
                traces.append(trace)
                c.write_jsonl(path / "search_trace.jsonl", traces)  # Pay BEFORE oracle access.
            saved = next((r for r in records if r["step"] == trace["step"]), None)
            if saved is None:
                result, physical = measure(manifest, shared, proposal, trace["query_id"], path.parents[1], executable, timeout)
                saved = {"step": trace["step"], "candidate_signature": proposal["signature"], "result": result, **physical}
                records.append(saved); c.write_jsonl(path / "measurements.jsonl", records)
            candidate = physical_candidate(manifest, shared, proposal)
            _verify_result(saved, candidate, shared["kernels"][manifest["kernel"]], shared["expected_tool_identity"])
            _check_context(history, saved["result"])
            history.append(_history_entry(trace, saved))
            after = _best(history)
            result = saved["result"]
            trace.update(status="completed", physical_synthesis_status=saved["physical_synthesis_status"],
                measured_latency=result.get("latency_cycles"), measured_resources=result.get("resources"),
                measured_ADP=result.get("ADP"), feasibility=result.get("feasible"),
                incumbent_after=after["signature"] if after else None,
                best_ADP_so_far=after["measurement"]["ADP"] if after else None,
                incumbent_improved=bool(best and after and after["measurement"]["ADP"] < best["measurement"]["ADP"]),
                controller_runtime_seconds=max(0.0, time.monotonic() - started - saved["physical_synthesis_runtime_seconds"]))
            c.write_jsonl(path / "search_trace.jsonl", traces)
            pending = None; completed += 1
        if terminal and (not traces or traces[-1]["event_type"] != "termination"):
            traces.append({"event_type": "termination", "reason": terminal, "logical_evaluation_cost": 0})
            c.write_jsonl(path / "search_trace.jsonl", traces)
        summarize(path)
        print(f"{manifest['kernel']}/{manifest['policy']}: {len(history)}/{manifest['max_evaluations']} logical queries; {terminal or 'paused_at_invocation_limit'}", flush=True)


def summarize(path):
    path = Path(path); manifest, shared = load_run(path)
    traces, records, history, pending, terminal = read_state(path, manifest, shared)
    by_step = {r["step"]: r for r in records}
    queries = [r for r in traces if r["event_type"] == "query" and r["status"] == "completed"]
    baseline = history[0]["measurement"] if history and _usable(history[0]["measurement"]) else None
    rows = []
    for budget in BUDGETS:
        reached = len(history) >= budget
        carried = terminal == "neighborhood_exhausted" and len(history) < budget
        prefix = history[:budget] if reached else history if carried else []
        coverage = bool(prefix) and all(r["measurement"]["synthesis_status"] != "unavailable" for r in prefix)
        best = _best(prefix)
        valid_point = (reached or carried) and coverage and best is not None
        used = len(prefix)
        physical = [by_step[index] for index in range(1, used + 1)]
        rows.append({"kernel": manifest["kernel"], "policy": manifest["policy"], "seed": manifest["seed"],
            "execution": manifest["execution"], "budget": budget,
            "status": "early_exhausted_carried" if carried and valid_point else "measured" if valid_point else "partial_replay" if (reached or carried) and not coverage else "not_reached_or_no_feasible_design",
            "logical_evaluations": used, "coverage_complete": coverage,
            "authenticated_observations": sum(r["measurement"]["synthesis_status"] != "unavailable" for r in prefix),
            "best_ADP": best["measurement"]["ADP"] if valid_point else None,
            "best_observed_ADP": best["measurement"]["ADP"] if best else None,
            "improvement_pct": 100 * (baseline["ADP"] - best["measurement"]["ADP"]) / baseline["ADP"] if valid_point and baseline and baseline["ADP"] > 0 else None,
            "feasible_evaluations": sum(_usable(r["measurement"]) for r in prefix),
            "failures": sum(r["measurement"]["synthesis_status"] not in ("success", "unavailable") for r in prefix),
            "unavailable": sum(r["measurement"]["synthesis_status"] == "unavailable" for r in prefix),
            "unique_configurations": len({r["signature"] for r in prefix}),
            "incumbent_improvement_pct": 100 * sum(r["incumbent_improved"] for r in queries[:used]) / max(1, used - 1),
            "new_physical_synthesis_runs": sum(r["new_physical_synthesis_runs"] for r in physical),
            "authenticated_measurement_reuses": sum(r["authenticated_measurement_reuses"] for r in physical),
            "total_synthesis_runtime": sum(r["physical_synthesis_runtime_seconds"] for r in physical),
            "controller_runtime": sum(r["controller_runtime_seconds"] for r in queries[:used]),
            "termination": terminal, "early_termination": carried,
            "structural_proposals": sum(r["proposal_type"] == "structural" for r in prefix),
            "composition_proposals": sum(r["proposal_type"] == "composition" for r in prefix),
            "structural_incumbent_improvements": sum(r["proposal_type"] == "structural" and r["incumbent_improved"] for r in queries[:used]),
            "composition_incumbent_improvements": sum(r["proposal_type"] == "composition" and r["incumbent_improved"] for r in queries[:used]),
            "single_action_incumbent_improvements": sum(r["proposal_type"] in ("coordinate", "random") and r["incumbent_improved"] for r in queries[:used]),
            "tcad_expanded_value_queries": sum(bool(r["domain_classification"]["changed_values_new_under_tcad"]) for r in prefix)})
    _write_csv(path / "summary.csv", rows)
    lines = ["# Budgeted refinement search", "", f"Kernel: {manifest['kernel']}; policy: {manifest['policy']}; execution: {manifest['execution']}; seed: {manifest['seed']}.",
        "Original SALT configuration is logical evaluation 1. Cache hits cost one logical query. Each policy has an isolated history.", "",
        "| Budget | Coverage | Best ADP | Improvement % | Logical queries | New HLS runs | Reuses |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['budget']} | {row['status']} | {row['best_ADP']} | {row['improvement_pct']} | {row['logical_evaluations']} | {row['new_physical_synthesis_runs']} | {row['authenticated_measurement_reuses']} |")
    lines += ["", "## Paid query trace", "", "| Step | Proposal | Changes from incumbent | Outcome | ADP | Best so far | Incumbent improved |", "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for row in queries:
        outcome = by_step[row["step"]]["result"]["synthesis_status"]
        edits = ", ".join(f"{field}={row['proposal']['assignment'][field]}" for field in row['changed_fields']) or "Original SALT configuration"
        lines.append(f"| {row['step']} | {row['proposal_type']} | {edits} | {outcome} | {row['measured_ADP']} | {row['best_ADP_so_far']} | {row['incumbent_improved']} |")
    if pending:
        lines += ["", "A paid query is pending; resume the same command. It must not be selected or charged again."]
    if any(r["measurement"]["synthesis_status"] == "unavailable" for r in history):
        lines += ["", "Replay coverage is incomplete. Unavailable means absent exact measurement, not synthesis failure. Best observed values are partial observations, not fair measured-budget performance estimates. Do not rank policies using this incomplete archive."]
    lines += ["", "Only feasible authenticated same-context results update the incumbent. HLS timing is estimated and separate from resource feasibility; no functional equivalence or post-route timing was established.",
        "Composition requires observed beneficial singles at one reference. Structural proposals need verified compiler evidence but do not require beneficial constituents. Joint improvements are not automatically compiler-guidance improvements.",
        "Physical cache costs depend on execution order; compare methods at equal logical budgets. The fixed target/objective, small samples, limited local neighborhoods and unreplicated synthesis limit generalization."]
    (path / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize_all(output):
    output = Path(output)
    rows = []
    for manifest in sorted(output.glob("*/*/manifest.json")):
        rows += summarize(manifest.parent)
    _write_csv(output / "summary.csv", rows)
    lines = ["# Matched logical-budget refinement comparison", "", "| Kernel | Policy | Budget | Coverage | Best ADP | Improvement % |", "| --- | --- | ---: | --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['kernel']} | {row['policy']} | {row['budget']} | {row['status']} | {row['best_ADP']} | {row['improvement_pct']} |")
    lines += ["", "Only rows marked measured or early_exhausted_carried are supported budget points. Partial replay is a controller validation exercise and cannot establish policy superiority. Larger-budget QoR must not be presented as better search efficiency."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "replay", "run", "summarize"), default="prepare")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--measurements", action="append", default=[])
    parser.add_argument("--kernels", nargs="+", default=["GAN", "covariance", "Kalman"])
    parser.add_argument("--policies", nargs="+", choices=policies.POLICIES, default=list(policies.POLICIES))
    parser.add_argument("--execution", choices=("replay", "synthesis"), default="replay")
    parser.add_argument("--max-evaluations", type=int, default=8); parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fresh-synthesis", action="store_true", help="Ignore cache for prospective measurements; prepare-time setting")
    parser.add_argument("--allow-batch", action="store_true", help="Explicitly allow multiple prospective kernel/policy searches")
    parser.add_argument("--limit", type=int, help="Maximum additional logical queries in this invocation")
    parser.add_argument("--vitis-hls", default="vitis_hls"); parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args(argv)
    try:
        args.kernels = list(dict.fromkeys("covariance" if k.lower() == "covariance" else k for k in args.kernels))
        args.policies = list(dict.fromkeys(args.policies))
        if args.limit is not None and not 1 <= args.limit <= 16 or args.timeout <= 0:
            raise ValueError("Invalid query limit or timeout")
        output = c.repo_path(args.output_dir)
        if args.mode == "prepare":
            prepare(args)
        elif args.mode == "summarize":
            summarize_all(output)
        else:
            if args.mode == "run" and len(args.kernels) * len(args.policies) > 1 and not args.allow_batch:
                raise ValueError("Prospective batches require --allow-batch; first select one kernel and policy for a smoke test")
            for name in args.kernels:
                for policy in args.policies:
                    path = output / name / policy
                    manifest, _ = load_run(path)
                    expected = "replay" if args.mode == "replay" else "synthesis"
                    if manifest["execution"] != expected:
                        raise ValueError("Execution mode differs from the prepared search")
                    search(path, executable=args.vitis_hls, timeout=args.timeout, limit=args.limit)
            summarize_all(output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("Refinement search: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
