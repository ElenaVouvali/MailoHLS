"""R0.1: authenticated factorial attribution, isolated relation audit, bounded HLS."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

from LLM_branch.common import directive_domains as dd
from experiments.refinement import r0_candidates as c, r0_synthesis as hls
from experiments.refinement.r0_common import ROOT, repo_path, display_path, object_hash, sha256, read_jsonl, write_json, write_jsonl
from experiments.refinement.r0_local_headroom import CONTEXT, SPECS, snapshot_source, code_hashes as r0_code_hashes
from experiments.refinement.r0_relations import graph_relations, compiler_tools, regenerate_graph
from .r01_analysis import domain_classification, factorial_candidates, factorial_analysis
from .r01_relation_audit import audit_all
from .r01_provenance import R0Archive, snapshot_kernel, baseline_comparison, source_context, comparison_id, tool_identity, authenticate_work

DEFAULT_OUTPUT = ROOT / "experiments/refinement/results/r01_attribution"
DEFAULT_R0 = ROOT / "experiments/refinement/results/r0_local_headroom"


def implementation_hashes():
    return {display_path(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _safe_output(output, r0):
    output, r0 = Path(output).resolve(), Path(r0).resolve()
    if output == r0 or r0 in output.parents or output in r0.parents:
        raise ValueError("R0.1 output must be separate from the original R0 archive")
    return output


def _proof_copy(old, archive, output):
    original = old["original_candidate"]
    result = {key: value for key, value in old.items() if key not in {"original_candidate", "evidence"}}
    old_work = Path(result["receipt_path"]).parent
    directory = Path(output) / "reused" / original["canonical_signature"]
    files = [archive.resolve(result[key]) for key in ("report_path", "receipt_path", "log_path")]
    work = archive.resolve(result["receipt_path"]).parent
    files += [work / "run_hls.tcl", work / Path(archive.kernel["source_provenance"]["source"]).name]
    old_source_parent = Path(archive.kernel["source_provenance"]["snapshot_source"]).parent
    files += [work / Path(entry["snapshot"]).relative_to(old_source_parent)
        for entry in archive.kernel["source_provenance"]["files"]
        if Path(entry["snapshot"]).suffix in {".h", ".hpp", ".hh", ".inc"}
        and Path(entry["snapshot"]).name not in hls.VENDOR_HEADERS]
    for path in files:
        target = directory / path.relative_to(work)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    proof = {"original_candidate": original, "original_result": result,
        "original_work": str(old_work), "copied_work": display_path(directory)}
    target = directory / "reuse_proof.json"
    write_json(target, proof)
    return display_path(target)


def _audit_inputs(args, output):
    path = repo_path(args.r0_output).resolve()
    archive = None
    try:
        archive = R0Archive(path)
    except (OSError, ValueError, KeyError) as exc:
        error = str(exc)
    if archive is not None:
        audits, reused = archive.authenticate()
        kernel = snapshot_kernel(archive.kernel, archive.resolve, output)
        actions, baseline, domains, native = archive.actions, archive.base, archive.domains, archive.native_domains
        comparison = baseline_comparison(archive, reused)
        evidence = archive.path / "historical_evidence/GAN"
        if evidence.is_dir():
            shutil.copytree(evidence, Path(output) / "baseline_evidence/GAN", dirs_exist_ok=True)
        recovery = {"status": "recovered", "r0_output": str(path),
            "files_sha256": {name: sha256(path / name) for name in
                ("manifest.json", "candidates.jsonl", "candidate_pool.jsonl", "synthesis_results.jsonl")}, "measurements": audits}
    else:
        # Reconstruction uses the current authoritative source/metadata and the
        # exact published assignment, never the numbers quoted in the task.
        kernel, actions, template = snapshot_source("GAN", SPECS["GAN"], output)
        paper = [row for row in read_jsonl(ROOT / "paper_plots/SALT_experiments.jsonl")
            if row.get("kernel") == "GAN" and row.get("objective") == "ADP" and row.get("stage") == "predicted-directives"]
        if len(paper) != 1:
            raise ValueError("Cannot reconstruct a unique complete GAN ADP baseline")
        baseline = c.from_paper(paper[0], actions)
        native = kernel["domains"]
        domains = c.published_factor_domains(baseline, actions, native)
        kernel["domains"] = domains
        graph = Path(output) / "graphs/GAN.gexf"
        try:
            regenerate_graph(repo_path(kernel["source_provenance"]["snapshot_source"]), graph, compiler_tools(), compiler_symbol=SPECS["GAN"]["compiler_symbol"])
            kernel["relations"] = graph_relations(graph, kernel["source_provenance"]["source_sha256"], kernel["source_provenance"]["metadata_sha256"], actions)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            kernel["relations"] = {"status": "blocked", "error": str(exc)}
        recovery = {"status": "reconstructed_no_measurements", "reason": error, "r0_output": str(path), "measurements": []}
        comparison = {"status": "unresolved_baseline_discrepancy", "paper_reference": paper[0], "reason": "No authenticated historical/new baseline available"}
        reused = {}
    if args.fresh_measurements:
        reused = {}
    relation = kernel.get("relations", {})
    relations = []
    if relation.get("status") == "available":
        import networkx as nx
        graph = repo_path(relation["artifact"])
        validated = graph_relations(graph, kernel["source_provenance"]["source_sha256"], kernel["source_provenance"]["metadata_sha256"], actions)
        relations = audit_all(nx.read_gexf(graph), actions, validated["pairs"], sha256(graph))
    return archive, kernel, actions, baseline, domains, native, reused, recovery, comparison, relations


def prepare(args, audit_only=False):
    output = _safe_output(repo_path(args.output_dir), repo_path(args.r0_output))
    if (output / "R01_MANIFEST.json").exists():
        raise ValueError("Immutable R0.1 manifest exists; resume run/summarize or choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    archive, kernel, actions, base, domains, native, reused, recovery, comparison, relations = _audit_inputs(args, output)
    write_json(output / "R01_R0_AUTHENTICATION.json", recovery)
    write_json(output / "R01_BASELINE_COMPARISON.json", comparison)
    write_jsonl(output / "R01_RELATIONS.jsonl", relations)
    if audit_only:
        text = ["# R0.1 audit", "", "R0 recovery: " + recovery["status"], "Baseline: " + comparison["status"], ""]
        text += [f"{r['action_i']}–{r['action_j']}: {r['verified_relation_class']}" for r in relations]
        (output / "R01_AUDIT.md").write_text("\n".join(text) + "\n")
        print(f"Audited {len(recovery['measurements'])} records and {len(relations)} loop-array pairs. No synthesis launched.")
        return
    requirements = {c.key("ARRAY_F", "L2"): "5", c.key("ARRAY_T", "L2"): "cyclic", c.key("ARRAY_D", "L2"): "1",
        c.key("PIPE", "L11"): "0", c.key("UNROLL", "L11"): "5", c.key("ARRAY_T", "L3"): "cyclic"}
    if any(base.get(k) != value for k, value in requirements.items()):
        raise ValueError("Recovered baseline does not match the preregistered GAN factorial configuration")
    a, b, cc = {c.key("ARRAY_F", "L2"): "6"}, {c.key("PIPE", "L11"): "1"}, {c.key("ARRAY_T", "L3"): "block"}
    by_pair = {(r["action_i"], r["action_j"]): r for r in relations}
    sets = {"AB": {"left": a, "right": b, "relation": by_pair.get(("L2", "L11")), "labels": ["d0", "dA", "dB", "dAB"]}}
    blockers = []
    if by_pair.get(("L3", "L11"), {}).get("verified_relation_class") == "direct_verified":
        sets["CB"] = {"left": cc, "right": b, "relation": by_pair[("L3", "L11")], "labels": ["d0", "dC", "dB", "dCB"]}
    else:
        blockers.append("L3–L11 lacks verified direct evidence; the direct-pair factorial is blocked")
    controls = [r for r in relations if r["action_j"] == "L11" and r["action_i"] != "L3"
        and r["verified_relation_class"] == "no_evidenced_relation"
        and base.get(c.key("ARRAY_T", r["action_i"])) == "cyclic"]
    if "CB" in sets and controls:
        control = sorted(controls, key=lambda r: int(r["action_i"][1:]))[0]
        sets["control"] = {"left": {c.key("ARRAY_T", control["action_i"]): "block"}, "right": b,
            "relation": control, "labels": ["d0", "dM", "dB", "dMB"],
            "matching": "Same loop B, cyclic-to-block type edit, two actions/two fields, same full configuration-domain class; arrays may differ in extent/type. No independence claim. Selection uses no QoR."}
    else:
        blockers.append("No suitable no-evidenced loop-array control with the same B modification")
    configs = {}
    for test in sets.values():
        factorial = factorial_candidates(base, actions, domains, test["left"], test["right"])
        test["cells"] = {}
        for label, values in zip(test["labels"], factorial.values()):
            digest = c.signature(values)
            if digest not in configs:
                labels, fields = c.changed(base, values, actions)
                configs[digest] = {"kernel": "GAN", "candidate_id": "r01-gan-" + digest[:16],
                    "candidate_group": "factorial", "canonical_signature": digest, "assignment": values,
                    "target": c.serialize(values, actions), "changed_action_ids": labels,
                    "changed_directive_fields": fields, "changed_action_count": len(labels), "changed_field_count": len(fields),
                    "domain": domain_classification(values, base, native), "aliases": []}
            if label not in configs[digest]["aliases"]:
                configs[digest]["aliases"].append(label)
            test["cells"][label] = configs[digest]["candidate_id"]
    # A matched native alternative already admitted by the original R0 policy.
    # This is an explicit domain comparator, not a new arbitrary search space.
    native_a = {**base, c.key("ARRAY_F", "L2"): "4"}
    if "4" in native[c.key("ARRAY_F", "L2")] and "4" in domains[c.key("ARRAY_F", "L2")]:
        c.validate(native_a, actions, domains)
        digest = c.signature(native_a)
        labels, fields = c.changed(base, native_a, actions)
        configs[digest] = {"kernel": "GAN", "candidate_id": "r01-gan-" + digest[:16], "candidate_group": "native_comparator",
            "canonical_signature": digest, "assignment": native_a, "target": c.serialize(native_a, actions),
            "changed_action_ids": labels, "changed_directive_fields": fields, "changed_action_count": 1, "changed_field_count": 1,
            "domain": domain_classification(native_a, base, native), "aliases": ["dA_native4"]}
    expected_tools = {object_hash(v["tool_identity"]): v["tool_identity"] for sig, v in reused.items() if sig in configs}
    if len(expected_tools) > 1:
        raise ValueError("Authenticated R0 cells use different tools; prepare fresh measurements")
    measurements = []
    for digest, candidate in configs.items():
        candidate["evaluation_context_hash"] = object_hash({"experiment": "R0.1", "source": source_context(kernel),
            "assignment": digest, "r01_code": implementation_hashes(), "r0_code": r0_code_hashes()})
        if digest in reused:
            old = reused[digest]
            proof = _proof_copy(old, archive, output)
            measurements.append({**{k: v for k, v in old.items() if k not in {"original_candidate", "evidence"}},
                "candidate_id": candidate["candidate_id"], "canonical_signature": digest,
                "authentication": "authenticated_reuse", "reuse_proof": proof,
                "r01_evaluation_context_hash": candidate["evaluation_context_hash"]})
    # Primary missing L2-only job first if the other three are authenticated.
    priorities = {"dA": 0, "d0": 1, "dC": 2, "dCB": 3, "dB": 4, "dAB": 5, "dM": 6, "dMB": 7, "dA_native4": 8}
    candidates = sorted(configs.values(), key=lambda r: min(priorities[a] for a in r["aliases"]))
    if not any("d0" in row["aliases"] and row["canonical_signature"] in reused for row in candidates):
        candidates.sort(key=lambda row: "d0" not in row["aliases"])
    write_jsonl(output / "R01_CANDIDATES.jsonl", candidates)
    immutable_files = [output / name for name in ("R01_CANDIDATES.jsonl", "R01_RELATIONS.jsonl", "R01_R0_AUTHENTICATION.json", "R01_BASELINE_COMPARISON.json")]
    for directory in ("sources", "graphs", "reused", "baseline_evidence"):
        immutable_files += [p for p in (output / directory).rglob("*") if p.is_file()]
    manifest = {"experiment_id": "R0.1-GAN-attribution", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "context": CONTEXT, "kernel": kernel, "baseline_assignment": base, "native_domains": native, "r0_local_domains": domains,
        "factorials": sets, "blockers": blockers, "max_new_syntheses": 12,
        "expected_tool_identity": next(iter(expected_tools.values())) if expected_tools else None,
        "r0_code_sha256": r0_code_hashes(), "r01_code_sha256": implementation_hashes(),
        "r0_recovery": recovery["status"], "input_sha256": {display_path(p): sha256(p) for p in immutable_files},
        "physical_configuration_count": len(candidates), "initial_reused_count": len(measurements),
        "candidate_policy": "preregistered GAN factorials, evidence-selected matched control, one native factor-4 comparator; no result-guided selection"}
    write_json(output / "R01_MANIFEST.json", manifest)
    (output / "R01_MANIFEST.sha256").write_text(sha256(output / "R01_MANIFEST.json") + "\n")
    write_jsonl(output / "R01_MEASUREMENTS.jsonl", measurements)
    write_jsonl(output / "R01_ATTEMPTS.jsonl", [])
    summarize(output)
    print(f"Prepared {len(candidates)} configurations; {len(measurements)} authenticated reuses; {len(candidates)-len(measurements)} missing. No synthesis launched.")


def load_output(output):
    output = Path(output)
    if sha256(output / "R01_MANIFEST.json") != (output / "R01_MANIFEST.sha256").read_text().strip():
        raise ValueError("Immutable R0.1 manifest changed")
    manifest = json.loads((output / "R01_MANIFEST.json").read_text())
    if manifest["context"] != CONTEXT or manifest["r0_code_sha256"] != r0_code_hashes() or manifest["r01_code_sha256"] != implementation_hashes():
        raise ValueError("Frozen implementation/deployment context changed")
    for path, digest in manifest["input_sha256"].items():
        if sha256(repo_path(path)) != digest:
            raise ValueError(f"Immutable prepared input changed: {path}")
    candidates = read_jsonl(output / "R01_CANDIDATES.jsonl")
    return manifest, candidates


def verify_measurement(row, candidate, kernel):
    if row.get("authentication") == "authenticated_reuse":
        proof = json.loads(repo_path(row["reuse_proof"]).read_text())
        if (proof["original_candidate"]["assignment"] != candidate["assignment"]
                or proof["original_candidate"]["canonical_signature"] != candidate["canonical_signature"]
                or row.get("r01_evaluation_context_hash") != candidate["evaluation_context_hash"]):
            raise ValueError("Reused measurement belongs to a different assignment/context")
        def resolve(value):
            try:
                return repo_path(proof["copied_work"]) / Path(value).relative_to(proof["original_work"])
            except ValueError:
                return repo_path(value)
        verified = authenticate_work(proof["original_candidate"], proof["original_result"], kernel, resolve,
            expected_evaluation_hash=proof["original_candidate"]["evaluation_context_hash"])
    elif row.get("authentication") == "authenticated_new":
        verified = authenticate_work(candidate, row, kernel, repo_path,
            expected_evaluation_hash=candidate["evaluation_context_hash"])
    else:
        return
    if row["canonical_signature"] != candidate["canonical_signature"] or row["comparison_context_id"] != verified["comparison_context_id"]:
        raise ValueError("Measurement signature/comparison context mismatch")
    for key in ("ADP", "latency_cycles", "resources", "feasible", "rankable"):
        if row[key] != verified[key]:
            raise ValueError("Measurement differs from authenticated report")


def run(output, executable, limit=None, retry_failed=False):
    output = Path(output)
    with (output / "R01_RUN.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, candidates = load_output(output)
        tool = hls.tool_info(executable)
        identity = tool_identity(tool)
        if manifest["expected_tool_identity"] and identity != manifest["expected_tool_identity"]:
            raise ValueError("HLS tool/header build differs from reused cells; prepare a fresh --fresh-measurements experiment")
        seal = output / "R01_TOOL.json"
        if seal.exists() and json.loads(seal.read_text()) != identity:
            raise ValueError("HLS execution tool changed within experiment")
        if not seal.exists():
            write_json(seal, identity)
        rows = read_jsonl(output / "R01_MEASUREMENTS.jsonl")
        done = {r["candidate_id"]: r for r in rows}
        attempts = read_jsonl(output / "R01_ATTEMPTS.jsonl")
        launched = 0
        for candidate in candidates:
            old = done.get(candidate["candidate_id"])
            if old and old.get("synthesis_status") == "success":
                verify_measurement(old, candidate, manifest["kernel"])
                continue
            reservations = [entry for entry in attempts if entry["candidate_id"] == candidate["candidate_id"]]
            if reservations and not retry_failed:
                # A durable result can be recovered after a reporting crash.
                saved = sorted((output / "projects" / candidate["candidate_id"]).glob("attempt-*/result.json"))
                if saved and candidate["candidate_id"] not in done:
                    result = json.loads(saved[-1].read_text())
                    verified = authenticate_work(candidate, result, manifest["kernel"], repo_path, expected_evaluation_hash=candidate["evaluation_context_hash"])
                    done[candidate["candidate_id"]] = {**result, **verified, "canonical_signature": candidate["canonical_signature"], "authentication": "authenticated_new"}
                continue
            if len(attempts) >= min(12, manifest["max_new_syntheses"]) or (limit is not None and launched >= limit):
                break
            attempts.append({"attempt_index": len(attempts) + 1, "candidate_id": candidate["candidate_id"], "status": "reserved"})
            write_jsonl(output / "R01_ATTEMPTS.jsonl", attempts)
            try:
                result = hls.run_candidate(candidate, manifest["kernel"], output, tool)
                result.update(canonical_signature=candidate["canonical_signature"])
                if result["synthesis_status"] == "success":
                    verified = authenticate_work(candidate, result, manifest["kernel"], repo_path, expected_evaluation_hash=candidate["evaluation_context_hash"])
                    result.update(verified, authentication="authenticated_new")
                else:
                    result["authentication"] = "unsuccessful_new"
                done[result["candidate_id"]] = result
                attempts[-1]["status"] = result["synthesis_status"]
            except BaseException as exc:
                attempts[-1].update(status="interrupted_or_preflight_failure", error=str(exc))
                write_jsonl(output / "R01_ATTEMPTS.jsonl", attempts)
                raise
            launched += 1
            write_jsonl(output / "R01_ATTEMPTS.jsonl", attempts)
            write_jsonl(output / "R01_MEASUREMENTS.jsonl", list(done.values()))
        write_jsonl(output / "R01_MEASUREMENTS.jsonl", list(done.values()))
        summarize(output)
        print(f"Launched {launched} new synthesis evaluations; total reservations {len(attempts)}/12.")


def summarize(output):
    output = Path(output)
    manifest, candidates = load_output(output)
    lookup = {c["candidate_id"]: c for c in candidates}
    rows = read_jsonl(output / "R01_MEASUREMENTS.jsonl")
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate measurement records")
    for row in rows:
        verify_measurement(row, lookup[row["candidate_id"]], manifest["kernel"])
    by_id = {r["candidate_id"]: r for r in rows}
    results, csv_rows = {}, []
    lines = ["# R0.1 — GAN refinement attribution", "", "Fixed context: Vitis HLS 2021.1, ZCU104, 10 ns, full capacities, ADP. No model inference or training.", "",
        "## Baseline reconciliation", "", json.loads((output / "R01_BASELINE_COMPARISON.json").read_text())["status"],
        "See R01_BASELINE_COMPARISON.json for source/header/Tcl/tool/clock/part checks. Paper data is retained; only authenticated same-context cells enter factorial calculations.", "",
        "## Domains", "", "Classification is for the complete assignment. The published baseline already contains values outside today's native proposal whitelist. Consequently all these configurations may be extended_domain even when their edits themselves use native values. New edit values are recorded separately; factor 6 at ARRAY_F_L2 is checked against the actual source domain, not assumed legal/native.", ""]
    native_l2 = manifest["native_domains"][c.key("ARRAY_F", "L2")]
    lines += ["Native ARRAY_F_L2 proposals: `" + json.dumps(native_l2) + "`; factor 6 is native: **" + str("6" in native_l2) + "**.",
        "Baseline values outside native proposals: `" + json.dumps(domain_classification(manifest["baseline_assignment"], manifest["baseline_assignment"], manifest["native_domains"])["outside_native_domain"], sort_keys=True) + "`.", ""]
    for name, test in manifest["factorials"].items():
        cell_rows = {slot: by_id.get(test["cells"][label])
                     for slot, label in zip(("d0", "dX", "dY", "dXY"), test["labels"])}
        result = factorial_analysis(cell_rows)
        results[name] = result
        relation_class = (test.get("relation") or {}).get("verified_relation_class", "conservative_unknown")
        lines += ["## " + name, "", f"Relation: {relation_class}. Left edit: `{test['left']}`; right edit: `{test['right']}`.", "",
            "| Cell | Authentication | Cycles | BRAM / DSP / FF / LUT | Area | ADP | Domain | Changed value needs extension |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for label in test["labels"]:
            candidate_id = test["cells"][label]
            row = by_id.get(candidate_id)
            candidate = lookup[candidate_id]
            csv_rows.append({"experiment": name, "cell": label, "candidate_id": candidate_id,
                "measurement_status": row.get("authentication") if row else "pending",
                "latency_cycles": row.get("latency_cycles") if row else None,
                "ADP": row.get("ADP") if row else None, "resource_feasible": row.get("feasible") if row else None,
                "configuration_domain": candidate["domain"]["configuration_domain"],
                "changed_value_requires_extension": candidate["domain"]["changed_value_requires_extension"],
                "factorial_status": result["status"], "interaction_ADP": result.get("interaction_ADP")})
            resources = "/".join(str(row["resources"][key]) for key in ("BRAM_18K", "DSP", "FF", "LUT")) if row and row.get("resources") else "unavailable"
            lines.append(f"| {label} | {row.get('authentication') if row else 'pending'} | {row.get('latency_cycles') if row else 'unavailable'} | {resources} | {row.get('area_proxy') if row else 'unavailable'} | {row.get('ADP') if row else 'unavailable'} | {candidate['domain']['configuration_domain']} | {candidate['domain']['changed_value_requires_extension']} |")
        if result["status"] == "measured":
            lines += ["", f"ADP interaction: **{result['interaction_ADP']:.12g}** ({result['interaction_class']}); {result['interaction_pct_of_baseline']:.8g}% of baseline ADP.", "",
                "| Modification | ADP improvement % | Delta cycles | Delta BRAM / DSP / FF / LUT | Delta ADP |",
                "| --- | --- | --- | --- | --- |"]
            for slot, label in zip(("dX", "dY", "dXY"), test["labels"][1:]):
                delta = "/".join(str(result["resource_deltas"][slot][key]) for key in ("BRAM_18K", "DSP", "FF", "LUT"))
                lines.append(f"| {label} | {result['improvement_pct'][slot]:.8g} | {result['latency_deltas'][slot]} | {delta} | {result['ADP_deltas'][slot]:.12g} |")
        else:
            lines += ["", "Interaction unavailable: " + result["reason"]]
        lines += [""]
        if name == "control":
            lines += [test["matching"], "No evidenced relation does not establish semantic independence.", ""]
    attempts = read_jsonl(output / "R01_ATTEMPTS.jsonl")
    reused = sum(row.get("authentication") == "authenticated_reuse" for row in rows)
    new = sum(row.get("authentication") == "authenticated_new" for row in rows)
    lines += ["## Cost and interpretation", "", f"Unique configurations: {len(candidates)}. Authenticated reused: {reused}. Authenticated new: {new}. New synthesis reservations: {len(attempts)}/12. Each factorial has four logical cells; shared cells and reuse reduce physical synthesis cost.", ""]
    for blocker in manifest["blockers"]:
        lines += ["Blocker: " + blocker]
    primary = results["AB"]
    if any(result["status"] != "measured" for result in results.values()) or manifest["blockers"]:
        decision = "Research decision pending: required factorial measurements or verified comparisons are incomplete."
    else:
        d0, da, db, dab = (by_id[manifest["factorials"]["AB"]["cells"][name]] for name in ("d0", "dA", "dB", "dAB"))
        if abs(da["ADP"] - dab["ADP"]) <= max(1e-9, 1e-10 * d0["ADP"]) and da["ADP"] < d0["ADP"]:
            decision = "Outcome 2 is supported locally: the isolated A edit accounts for the joint gain; B adds no measured ADP benefit. A uses a value outside the native proposal domain, also supporting further domain exploration (Outcome 3). This does not establish that expansion alone is sufficient or that effects generalize."
        elif primary["interaction_ADP"] < 0 and (manifest["factorials"]["AB"].get("relation") or {}).get("verified_relation_class") in {"direct_verified", "indirect_verified"}:
            decision = "Favorable interaction is measured on a verified relation; assess the direct and matched-control factorials before considering Outcome 1."
        else:
            decision = "The primary factorial does not establish strong verified structural synergy. Inspect individual effects, the native comparator, direct-pair and matched-control results before choosing an architectural direction."
    lines += [decision, "", "A native factor-4 single-edit comparator is included only because it belongs to both the production domain and the existing R0 local policy. It does not turn the entire historically extended baseline into a native configuration.", ""]
    comparator = next((r for r in candidates if "dA_native4" in r["aliases"]), None)
    if comparator:
        row = by_id.get(comparator["candidate_id"])
        if row and row.get("authentication") in {"authenticated_reuse", "authenticated_new"}:
            lines += [f"Native factor-4 comparator: cycles={row['latency_cycles']}; resources={row['resources']}; ADP={row['ADP']}; resource-feasible={row['feasible']}."]
            ab = manifest["factorials"]["AB"]["cells"]
            arow = by_id.get(ab["dA"])
            if arow and row["feasible"] and arow["feasible"] and row["comparison_context_id"] == arow["comparison_context_id"]:
                lines += [f"Factor 6 versus factor 4 ADP difference: {arow['ADP']-row['ADP']:.12g}. This checks one matched native edit; it does not exhaust the native domain or prove domain-expansion dominance."]
        else:
            lines += ["Native factor-4 comparator: pending; domain-attribution decision remains pending."]
        lines += [""]
    lines += ["Interaction sign is a finite-difference ADP result, not proof of a physical mechanism. Original relation bits remain unchanged. Shared-helper reachability is kept conservative unless exact call-site roots/bindings are present. No hardware-target comparison was run, so Outcome 4 is unsupported. HLS timing is an estimate, not post-route timing.", ""]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader(); writer.writerows(csv_rows)
    (output / "R01_FACTORIAL.csv").write_text(stream.getvalue())
    write_json(output / "R01_INTERACTIONS.json", results)
    (output / "R01_REPORT.md").write_text("\n".join(lines))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "prepare", "run", "summarize"], default="audit")
    parser.add_argument("--r0-output", default=str(DEFAULT_R0))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--vitis-hls", default="vitis_hls")
    parser.add_argument("--fresh-measurements", action="store_true", help="No measurement reuse; evaluate all cells in a new context")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and not 1 <= args.limit <= 12:
            raise ValueError("--limit must be between one and the hard cap of 12")
        output = _safe_output(repo_path(args.output_dir), repo_path(args.r0_output))
        if args.mode in {"prepare", "audit"}:
            prepare(args, audit_only=args.mode == "audit")
        elif args.mode == "run":
            run(output, args.vitis_hls, args.limit, args.retry_failed)
        else:
            summarize(output)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print("R0.1: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
