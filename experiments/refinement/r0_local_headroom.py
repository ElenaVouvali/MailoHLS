"""R0: deterministic preparation, explicit synthesis, measured-only summaries."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

from LLM_branch.common import directive_domains as dd
from . import r0_candidates as candidates
from .r0_common import ROOT, repo_path, display_path, sha256, object_hash, write_json, write_jsonl, read_jsonl
from .r0_metrics import PART, CAPACITIES, CLOCK_NS, evaluate_metrics, best_feasible, improvement_pct
from .r0_relations import compiler_tools, regenerate_graph, graph_relations
from .r0_synthesis import HLS_VERSION, SETTINGS, materialize, make_tcl, run_queue

SPECS = {
    "covariance": {"source": "evaluation_set/covariance/kernel.cpp", "metadata": "evaluation_set/covariance/kernel_info.txt", "top": "kernel_covariance"},
    "GAN": {"source": "evaluation_set/GAN/network.cpp", "metadata": "evaluation_set/GAN/code_to_memory_outputs/meta/kernel_info.txt", "top": "forward_propagation",
            "compiler_symbol": "_Z19forward_propagationPfS_"},
    "Kalman": {"source": "evaluation_set/serrano-kalman-filter/kernel.cpp", "metadata": "evaluation_set/serrano-kalman-filter/kernel_info.txt", "top": "krnl_KALMAN",
               "relations": "GNN_branch/MLIR_graphs/serrano-kalman-filter.gexf",
               "relation_metadata": "Data/ApplicationDataset/serrano-kalman-filter/kernel_info.txt",
               "relation_source": "Data/ApplicationDataset/serrano-kalman-filter/kernel.cpp"},
}
CONTEXT = {"device": "ZCU104", "part": PART, "requested_clock_ns": CLOCK_NS,
           "resource_budget": CAPACITIES, "device_capacities": CAPACITIES,
           "objective": "ADP", "area_definition": "mean of four device-capacity utilization percentages",
           "adp_definition": "latency_cycles * area_proxy", "hls_version": HLS_VERSION,
           "flow_target": "vivado", "synthesis_settings": SETTINGS}
POLICY = {"schema": "salt-r0-local-headroom-v2", "independent": "all legal complete one-action alternatives; closest domain-index distance; alternate loop/array sites",
          "structural": "genuine relation pairs only; Cartesian product of three nearest legal alternatives per action; prefer loop-array pairs; diverse pairs",
          "ties": "SHA256(seed:signature), never measured QoR", "per_arm": 4,
          "baseline_policy": "use exact published final directives; owner-authorized waiver of powers-of-two factor whitelist; never repair baseline",
          "factor_policy": "R0 only: per site {0, published factor, 2, factor-1, factor+1}; new nonzero factors >=2 and bounded by source trip count/max array extent; other domains unchanged",
          "inactive_actions": "when authenticated compiler-active metadata exists, exclude unrepresented sites from both arms; retain all fields in complete assignments",
          "comparison": "equal selected budgets; different action distance/cardinality is a confound"}
OUTPUT = ROOT / "experiments/refinement/results/r0_local_headroom"


def preserve_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"Existing snapshot differs: {target}. Choose a fresh --output-dir.")
    else:
        shutil.copy2(source, target)


def snapshot_source(name, spec, output):
    source, metadata = repo_path(spec["source"]), repo_path(spec["metadata"])
    if not source.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Missing source or metadata: {source}, {metadata}")
    top = metadata.read_text().splitlines()[0].strip()
    if top != spec["top"]:
        raise ValueError(f"Top-level function mismatch for {name}: {top}")
    directory = output / "sources" / name
    inputs = [source, metadata] + sorted(p for p in source.parent.rglob("*")
                                       if p.is_file() and p.suffix.lower() in {".h", ".hpp", ".hh", ".inc"}
                                       and "code_to_memory_outputs" not in p.parts)
    files = []
    for original in dict.fromkeys(inputs):
        relative = Path(source.name) if original == source else (Path("kernel_info.txt") if original == metadata else original.relative_to(source.parent))
        target = directory / relative
        preserve_copy(original, target)
        files.append({"original": display_path(original), "snapshot": display_path(target), "sha256": sha256(original)})
    raw = directory / source.name
    actions = dd.load_source_actions(name, directory)
    template = dd.prepare_source_template(name, raw, directory)
    domains = dd.source_site_domains(name, template, directory)
    template_path = directory / "source_template.txt"
    if template_path.exists() and template_path.read_text() != template:
        raise ValueError("Existing source template differs; choose a new output directory")
    template_path.write_text(template)
    provenance = {"source": display_path(source), "source_sha256": sha256(source),
                  "metadata": display_path(metadata), "metadata_sha256": sha256(metadata),
                  "snapshot_source": display_path(raw), "snapshot_metadata": display_path(directory / "kernel_info.txt"),
                  "files": files, "source_tree_hash": object_hash(files)}
    return {"top": top, "source_provenance": provenance, "template_path": display_path(template_path),
            "template_sha256": sha256(template_path), "domains": domains,
            "actions": {label: {"kind": a.kind, "trip_count": a.trip_count, "dimensions": a.dimensions} for label, a in actions.items()}}, actions, template


def syntax_check(text, source, log):
    compiler = shutil.which("g++")
    if not compiler:
        return {"status": "unavailable", "reason": "g++ not found; placeholder validation still performed"}
    command = [compiler, "-std=c++14", "-Wno-unknown-pragmas", "-fsyntax-only", "-x", "c++", "-I", str(source.parent), "-"]
    result = subprocess.run(command, input=text, capture_output=True, text=True, timeout=60)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr)
    return {"status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode,
            "argv": command, "log": display_path(log),
            "scope": "syntax only using repository analysis headers; not HLS or fixed-point validation"}


def code_hashes():
    files = sorted(Path(__file__).parent.glob("*.py")) + [ROOT / "LLM_branch/common/directive_domains.py",
        ROOT / "LLM_branch/common/mailohls_contract.py", ROOT / "GNN_branch/insert_placeholders.py",
        ROOT / "GNN_branch/build_structural_memory.py", ROOT / "GNN_branch/mlir_graph_gen.py"]
    return {display_path(path): sha256(path) for path in files}


def prepare(args):
    output = repo_path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = output / "synthesis_results.jsonl"
    if results.exists() and results.stat().st_size:
        raise ValueError("Measured results already exist. Resume with --mode run; prepare into a new --output-dir.")
    baseline_path = repo_path(args.baseline_jsonl)
    paper_rows = read_jsonl(baseline_path)
    specs = dict(SPECS)
    if args.config:
        override = json.loads(repo_path(args.config).read_text())
        if set(override) - set(SPECS):
            raise ValueError("R0 accepts exactly the three specified kernel identities")
        for name, values in override.items():
            specs[name] = {**specs[name], **values}
    try:
        tools = compiler_tools(args.polygeist_build, args.mlir_python)
        tools_status = {"status": "available", **tools}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        tools, tools_status = None, {"status": "unavailable", "error": str(exc)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"experiment_id": "R0-local-headroom", "git_commit": commit, "context": CONTEXT,
                "seed": args.seed, "candidate_generation_policy": POLICY,
                "baseline_jsonl": display_path(baseline_path), "baseline_jsonl_sha256": sha256(baseline_path),
                "compiler_tools": tools_status, "code_sha256": code_hashes(), "kernels": {},
                "target_total_configurations": len(args.kernels) * 9, "historical_measurements_reused": False}
    manifest["git_working_tree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip())
    requested_hls = getattr(args, "vitis_hls", "vitis_hls")
    manifest["synthesis_tool"] = {"requested_executable": requested_hls,
        "status": "executable_found_not_version_verified" if shutil.which(requested_hls) else "unavailable",
        "required_version": HLS_VERSION}
    queue, pool_records = [], []
    for name in args.kernels:
        kernel = {"status": "blocked", "blockers": [], "candidate_counts": {"baseline": 0, "independent": 0, "structural": 0}}
        manifest["kernels"][name] = kernel
        matching = [row for row in paper_rows if row.get("kernel") == name
                    and row.get("stage") == "predicted-directives" and row.get("objective") == "ADP"]
        if len(matching) != 1:
            kernel["blockers"].append(f"Expected one published ADP predicted-directives row; got {len(matching)}")
            continue
        row = matching[0]
        kernel["paper_baseline"] = row
        if (row.get("device", "ZCU104") not in {"ZCU104", PART}
                or row.get("part", PART) != PART
                or float(row.get("clock_period_ns", CLOCK_NS)) != CLOCK_NS
                or row.get("resource_budget", "full") != "full"):
            kernel["blockers"].append("Published baseline explicitly specifies a different deployment context")
            continue
        kernel["paper_context_verification"] = "Reference only: row lacks exact source/tool/part hashes; never used as a newly measured R0 baseline"
        kernel["paper_metrics"] = evaluate_metrics(row.get("latency"), {key: row.get("resources", {}).get(old)
            for key, old in {"BRAM_18K":"bram", "DSP":"dsp", "FF":"ff", "LUT":"lut"}.items()})
        try:
            source_info, actions, template = snapshot_source(name, specs[name], output)
            kernel.update(source_info)
            base = candidates.from_paper(row, actions)
            kernel["initial_assignment"] = base
            kernel["initial_target"] = candidates.serialize(base, actions)
            kernel["source_proposal_domains"] = kernel["domains"]
            kernel["published_factor_whitelist_exceptions"] = {
                lhs: {"published_value": rhs, "old_proposal_values": kernel["domains"][lhs]}
                for lhs, rhs in base.items() if rhs not in kernel["domains"][lhs]}
            kernel["domains"] = candidates.published_factor_domains(base, actions, kernel["domains"])
            kernel["baseline_domain_status"] = "valid_under_owner_authorized_published_factor_policy"
            relation_path = repo_path(specs[name]["relations"]) if specs[name].get("relations") else output / "graphs" / (name + ".gexf")
            try:
                if not relation_path.exists():
                    if not tools:
                        raise FileNotFoundError(tools_status["error"])
                    kernel["graph_generation"] = regenerate_graph(repo_path(kernel["source_provenance"]["snapshot_source"]), relation_path, tools, args.graph_timeout,
                        compiler_symbol=specs[name].get("compiler_symbol"))
                relation_metadata = repo_path(specs[name]["relation_metadata"]) if specs[name].get("relation_metadata") else None
                if specs[name].get("relation_source"):
                    related_source = repo_path(specs[name]["relation_source"])
                    if sha256(related_source) != kernel["source_provenance"]["source_sha256"]:
                        raise ValueError("Related graph was built from a different source snapshot")
                    for entry in kernel["source_provenance"]["files"]:
                        original = repo_path(entry["original"])
                        if original.suffix.lower() in {".h", ".hpp"}:
                            corresponding = related_source.parent / original.relative_to(repo_path(specs[name]["source"]).parent)
                            if not corresponding.is_file() or sha256(corresponding) != entry["sha256"]:
                                raise ValueError("Compiler graph and synthesis analysis headers differ")
                relation = graph_relations(relation_path, kernel["source_provenance"]["source_sha256"], kernel["source_provenance"]["metadata_sha256"], actions, relation_metadata)
                if relation["graph_metadata"].get("kernel") != kernel["top"]:
                    raise ValueError("MLIR graph top-level function mismatch")
                if specs[name].get("compiler_symbol") and relation["graph_metadata"].get("compiler_symbol") != specs[name]["compiler_symbol"]:
                    raise ValueError("MLIR graph compiler-symbol provenance mismatch")
                kernel["relations"] = relation
                if tools:
                    kernel["relations"]["compiler_build_hashes_match_existing_graph"] = all(
                        relation["graph_metadata"].get(key) == tools[key]
                        for key in ("cgeist_sha256", "analysis_extension_sha256"))
            except (OSError, RuntimeError, ValueError, ImportError, subprocess.TimeoutExpired) as exc:
                kernel["relations"] = {"status": "blocked", "error": str(exc), "preparation_command":
                    f"POLYGEIST_BUILD=/path/to/locked-build MLIR_PYTHON=/path/to/python3.11 {sys.executable} -m experiments.refinement.r0_local_headroom --mode prepare --output-dir {display_path(output)}"}
            kernel["source_validation"] = syntax_check(materialize(template, kernel["initial_target"]),
                repo_path(kernel["source_provenance"]["snapshot_source"]), output / "validation" / (name + "-baseline.log"))
            if kernel["source_validation"]["status"] == "failed":
                kernel["blockers"].append("Source syntax validation failed; see source_validation log")
            if kernel["blockers"]:
                continue
            independent, structural = candidates.generate_pools(name, base, actions, kernel["domains"], kernel["relations"].get("pairs", []), kernel["relations"].get("inactive_actions", []))
            chosen_independent = candidates.select_diverse(independent, actions, args.seed)
            chosen_structural = candidates.select_diverse(structural, actions, args.seed)
            selected = [candidates.candidate_record(name, "baseline", base, base, actions, kernel["domains"]), *chosen_independent, *chosen_structural]
            for candidate in selected:
                text = materialize(template, candidate["target"])
                check = syntax_check(text, repo_path(kernel["source_provenance"]["snapshot_source"]), output / "validation" / (candidate["candidate_id"] + ".log"))
                if check["status"] == "failed":
                    raise ValueError(f"Candidate source validation failed: {candidate['candidate_id']}")
                candidate["source_validation"] = check
                candidate["source_provenance"] = kernel["source_provenance"]
                candidate["evaluation_context_hash"] = object_hash({"context": CONTEXT, "source": kernel["source_provenance"],
                    "template_sha256": kernel["template_sha256"], "top": kernel["top"],
                    "assignment": candidate["canonical_signature"], "code": manifest["code_sha256"]})
                materialized = output / "materialized" / candidate["candidate_id"]
                materialized.mkdir(parents=True, exist_ok=True)
                (materialized / Path(specs[name]["source"]).name).write_text(text)
                (materialized / "run_hls.tcl").write_text(make_tcl(Path(specs[name]["source"]).name, kernel["top"]))
            queue.extend(selected)
            chosen_ids = {row["candidate_id"] for row in selected}
            pool_records.extend({**row, "selected": row["candidate_id"] in chosen_ids} for row in independent + structural)
            kernel["candidate_counts"] = {group: sum(c["candidate_group"] == group for c in selected) for group in ("baseline", "independent", "structural")}
            kernel["pool_counts"] = {"independent": len(independent), "structural": len(structural)}
            kernel["status"] = "prepared" if len(chosen_structural) == 4 else "partial"
            if len(chosen_structural) < 4:
                kernel["blockers"].append("Fewer than four evidenced structural neighbors; no fabricated pairs were added")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            kernel["blockers"].append(str(exc))
    for name in manifest["kernels"]:
        rows = [row for row in queue if row["kernel"] == name]
        if len({row["canonical_signature"] for row in rows}) != len(rows):
            raise ValueError("Duplicate selected configuration within kernel")
    manifest["actual_prepared_configurations"] = len(queue)
    manifest["candidates_sha256"] = object_hash(queue)
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "candidates.jsonl", queue)
    write_jsonl(output / "candidate_pool.jsonl", pool_records)
    if not results.exists():
        write_jsonl(results, [])
    summarize(output)
    for name, kernel in manifest["kernels"].items():
        print(name, kernel["status"], kernel["candidate_counts"], "; ".join(kernel["blockers"]))
    print(f"Prepared {len(queue)}/{manifest['target_total_configurations']} configurations; no synthesis launched. Output: {output}")
    return manifest


def load_prepared(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    rows = read_jsonl(output / "candidates.jsonl")
    if manifest["context"] != CONTEXT or object_hash(rows) != manifest["candidates_sha256"]:
        raise ValueError("Prepared queue/context was modified")
    for row in rows:
        kernel = manifest["kernels"][row["kernel"]]
        actions = dd.load_source_actions(row["kernel"], repo_path(kernel["source_provenance"]["snapshot_metadata"]).parent)
        candidates.validate(row["assignment"], actions, kernel["domains"])
        if candidates.parse_assignment(row["target"]) != row["assignment"] or candidates.signature(row["assignment"]) != row["canonical_signature"]:
            raise ValueError("Candidate target/signature mismatch")
    return manifest, rows


def summarize(output):
    output = Path(output)
    manifest, queue = load_prepared(output)
    result_rows = read_jsonl(output / "synthesis_results.jsonl")
    queued = {row["candidate_id"]: row for row in queue}
    if len({r["candidate_id"] for r in result_rows}) != len(result_rows):
        raise ValueError("Duplicate result rows")
    for row in result_rows:
        candidate = queued.get(row["candidate_id"])
        if not candidate or row["evaluation_context_hash"] != candidate["evaluation_context_hash"]:
            raise ValueError("Results from an unmatched synthesis context")
    table = []
    report = ["# R0 — Local Refinement Headroom", "", f"Git base: `{manifest['git_commit']}`. Seed: {manifest['seed']}.",
        "Context: ZCU104 / xczu7ev-ffvc1156-2-e, 10 ns, full device resources, Vitis HLS 2021.1, ADP.",
        "Area = mean(BRAM%, DSP%, FF%, LUT%) using device capacities; ADP = worst-case cycles × area.", "",
        f"Prepared {len(queue)}/{manifest['target_total_configurations']} configurations. Recorded synthesis results: {len(result_rows)}.",
        "Paper QoR is reference data only. No historical measurements were imported as R0 synthesis results.", "",
        "HLS readiness when prepared: **" + manifest.get("synthesis_tool", {}).get("status", "unknown") + "**.", "",
        "## Kernel readiness and baseline configurations", ""]
    for name, kernel in manifest["kernels"].items():
        rows = [r for r in result_rows if r["kernel"] == name]
        bases = [r for r in rows if r["candidate_group"] == "baseline" and r["synthesis_status"] == "success"]
        baseline = bases[0] if bases else None
        report += [f"### {name}", "", f"Status: **{kernel['status']}**. Selected counts: `{kernel['candidate_counts']}`.",
                   f"Relations: {kernel.get('relations', {}).get('status', 'unavailable')}; pairs: {len(kernel.get('relations', {}).get('pairs', []))}."]
        if "source_provenance" in kernel:
            report += [f"Source: `{kernel['source_provenance']['source']}`; metadata: `{kernel['source_provenance']['metadata']}`; top: `{kernel['top']}`.",
                       f"Source SHA256: `{kernel['source_provenance']['source_sha256']}`."]
        report += ["Materialized-source syntax check: " + kernel.get("source_validation", {}).get("status", "not performed") + "."]
        report += ["Factor policy: exact published values are preserved; adjacent integer factors replace the old powers-of-two proposal restriction for R0 only. All tuple, identity, completeness, and non-factor domain checks remain active."]
        if kernel.get("published_factor_whitelist_exceptions"):
            report += ["Published factors admitted by the explicit owner override: `" + json.dumps(kernel["published_factor_whitelist_exceptions"], sort_keys=True) + "`."]
        if kernel.get("relations", {}).get("inactive_actions"):
            report += ["Compiler-inactive sites excluded from edit selection in both arms: " + ", ".join(kernel["relations"]["inactive_actions"]) + ". All assignment fields remain present."]
        if kernel.get("paper_metrics"):
            paper = kernel["paper_metrics"]
            report += [f"Published reference: {paper['latency_cycles']} cycles; area proxy {paper['area_proxy']:.8g}; ADP {paper['ADP']:.8g}. Exact historical source/tool context is not authenticated."]
        if baseline:
            paper = kernel["paper_metrics"]
            discrepancy = {"latency_cycles": baseline["latency_cycles"] != paper["latency_cycles"], "resources": baseline["resources"] != paper["resources"]}
            report += [f"New R0 baseline: {baseline['latency_cycles']} cycles, area {baseline['area_proxy']:.8g}, ADP {baseline['ADP']:.8g}; discrepancy flags: `{discrepancy}`."]
            if any(discrepancy.values()):
                report += ["**Unexplained baseline discrepancy:** possible source/header, tool/build, or synthesis-setting differences. Paper values are preserved; only this newly synthesized baseline is used within R0."]
        else:
            report += ["New R0 baseline: **not measured successfully**. No measured improvement can be claimed."]
        report.extend("- Blocker: " + blocker for blocker in kernel["blockers"])
        if kernel.get("relations", {}).get("status") == "blocked":
            report += ["- Structural blocker: " + kernel["relations"]["error"],
                       "- Preparation: `" + kernel["relations"]["preparation_command"] + "`"]
        if "initial_target" in kernel:
            report += ["", "Initial published complete configuration:", "```text", kernel["initial_target"], "```"]
        winners = {}
        arm_attempts = {}
        for group in ("independent", "structural"):
            group_rows = [r for r in rows if r["candidate_group"] == group]
            best = best_feasible(group_rows)
            improvement = improvement_pct(baseline, best)
            winners[group] = best
            arm_attempts[group] = sum(r.get("attempt_count", 1) for r in group_rows)
            table.append({"kernel": name, "neighborhood": group,
                "baseline_latency": baseline.get("latency_cycles") if baseline else None,
                "baseline_area_proxy": baseline.get("area_proxy") if baseline else None,
                "baseline_ADP": baseline.get("ADP") if baseline else None,
                "selected_candidates": kernel["candidate_counts"][group],
                "valid_candidates": sum(r.get("rankable") is True and r["synthesis_status"] == "success" for r in group_rows),
                "attempted_syntheses": sum(r.get("attempt_count", 1) for r in group_rows),
                "successful_syntheses": sum(r["synthesis_status"] == "success" for r in group_rows),
                "best_candidate": best.get("candidate_id") if best else None,
                "best_latency": best.get("latency_cycles") if best else None,
                "best_area_proxy": best.get("area_proxy") if best else None,
                "best_ADP": best.get("ADP") if best else None, "ADP_improvement_pct": improvement})
            if best:
                finding = f"{best['candidate_id']}, ADP={best['ADP']:.8g}, improvement={improvement if improvement is not None else 'unknown'}%"
            else:
                finding = "**none measured successfully and feasibly**"
            report += ["", f"{group}: {len(group_rows)} attempted configurations; best feasible candidate: {finding}."]
        if winners.get("independent") and winners.get("structural") and baseline:
            a, b = winners["independent"]["ADP"], winners["structural"]["ADP"]
            comparison = "lower structural-arm ADP" if b < a else ("lower independent-arm ADP" if a < b else "equal best ADP")
            report += ["", f"Observed comparison: {comparison}. Actual synthesis attempts by arm: {arm_attempts}. This is exploratory, not causal evidence."]
            if len(set(arm_attempts.values())) != 1:
                report += ["**Unequal actual budgets:** best-of-arm comparison is not an equal-budget result."]
        else:
            report += ["", "Neighborhood comparison: **unavailable** until both arms have successful feasible measurements and a newly measured baseline."]
        report += [""]
    report += ["## Completed synthesis outcomes", ""]
    if not result_rows:
        report += ["None. The experiment is prepared, not yet synthesis-measured.", ""]
    else:
        report += ["| Candidate | Status | Cycles | BRAM/DSP/FF/LUT | Area | ADP | Resource feasible | Estimated clock (ns) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in result_rows:
            resources = row.get("resources")
            counts = "/".join(str(resources[k]) for k in ("BRAM_18K", "DSP", "FF", "LUT")) if resources else "unknown"
            report.append(f"| {row['candidate_id']} | {row['synthesis_status']} | {row.get('latency_cycles')} | {counts} | {row.get('area_proxy')} | {row.get('ADP')} | {row.get('feasible')} | {row.get('estimated_clock_ns')} |")
        report += [""]
    report += ["## Selected changes and compiler evidence", "", "| Kernel / arm | Candidate | Changed actions / fields | Relation evidence |", "| --- | --- | --- | --- |"]
    for row in queue:
        evidence = row["action_relation_evidence"]
        description = ", ".join(evidence["relation_type"]) + "; bits=" + str(evidence["relation_bits"]) + "; artifact=" + evidence["graph_or_memory_artifact"] if evidence else "none (baseline/single action)"
        report.append(f"| {row['kernel']} / {row['candidate_group']} | {row['candidate_id']} | {', '.join(row['changed_action_ids']) or 'none'} / {row['changed_field_count']} | {description} |")
    report += ["", "## Interpretation and limitations", "",
        "No conclusion about headroom or structural superiority is supported without completed, feasible synthesis measurements relative to a newly measured baseline.",
        "One-action and two-action arms have different edit distances, candidate pool cardinalities, and type mixes. Equal four-job budgets do not establish a causal benefit of compiler guidance. This small deterministic neighborhood is not exhaustive.",
        "Missing structural data blocks only the structural arm. No unsupported relationships, repaired baselines, or numerical failure penalties are substituted.",
        "The owner explicitly waived the historical factor whitelist: R0 preserves the final JSONL baseline and uses a bounded adjacent-integer factor neighborhood. This experiment-only policy change is recorded; production SALT domains, training, and inference remain unchanged. Synthesis acceptance and QoR still require measurement.",
        "Repository ap_fixed/ap_int headers are simplified compiler-analysis stubs. Syntax/MLIR validation is not HLS or bit-accurate validation. The HLS runner uses verified Vitis vendor headers; this can expose historical source/environment discrepancies.",
        "Timing estimates, unknown timing, resource feasibility, latency, and all four resource counts are recorded separately. No post-route timing claim is made.",
        "A learned refinement policy is premature until measured local headroom and repeatability are established. Next: run a three-job smoke test on an available kernel, then finish the fixed queue without candidate reselection; any remaining structural blockers are listed above.", ""]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(table[0]) if table else [])
    writer.writeheader()
    writer.writerows(table)
    (output / "summary.csv").write_text(stream.getvalue())
    (output / "R0_REPORT.md").write_text("\n".join(report))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "run", "summarize"], default="prepare")
    parser.add_argument("--kernels", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--baseline-jsonl", default="paper_plots/SALT_experiments.jsonl")
    parser.add_argument("--config", help="Optional JSON per-kernel source/metadata/top/relations path overrides")
    parser.add_argument("--polygeist-build")
    parser.add_argument("--mlir-python")
    parser.add_argument("--graph-timeout", type=int, default=600)
    parser.add_argument("--vitis-hls", default="vitis_hls")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", choices=list(SPECS), help="Only first baseline + independent + structural candidate of this kernel")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly spend additional jobs on failed attempts")
    args = parser.parse_args(argv)
    try:
        if args.graph_timeout <= 0 or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
            raise ValueError("Timeouts and synthesis limits must be positive")
        if len(args.kernels) != len(set(args.kernels)):
            raise ValueError("Duplicate kernel identities")
        output = repo_path(args.output_dir).resolve()
        if args.mode == "prepare":
            prepare(args)
        elif args.mode == "summarize":
            summarize(output)
            print(output / "R0_REPORT.md")
        else:
            manifest, queue = load_prepared(output)
            if manifest["code_sha256"] != code_hashes():
                raise ValueError("Implementation changed after preparation. Prepare a new output directory.")
            if not queue:
                raise ValueError("No valid prepared configurations; inspect R0_REPORT.md blockers before synthesis")
            count = run_queue(manifest, queue, output, args.vitis_hls, args.timeout, args.limit, args.smoke, args.retry_failed)
            summarize(output)
            print(f"Launched {count} synthesis job(s); existing completed results were not duplicated.")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"R0: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
