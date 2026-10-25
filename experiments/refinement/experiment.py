"""Prepare, synthesize and report deterministic local HLS refinement experiments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import csv
import fcntl
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from LLM_branch.common import directive_domains as legacy
from LLM_branch.common.directive_domains import tcad
from . import configurations as c, synthesis as hls
from .configurations import ROOT, CONTEXT, SPECS, sha256, object_hash, repo_path, display_path, read_jsonl, write_json, write_jsonl

DEFAULT_OUTPUT = ROOT / "experiments/refinement/results/local_refinement"
DEFAULT_GRAPHS = ROOT / "experiments/refinement/results/compiler_graphs"
MAX_EVALUATIONS = 48


def implementation_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / name for name in (
        "LLM_branch/common/directive_domains.py", "LLM_branch/common/directive_domain_workflow.py",
        "LLM_branch/common/mailohls_contract.py", "GNN_branch/insert_placeholders.py",
        "GNN_branch/build_structural_memory.py", "GNN_branch/mlir_graph_gen.py")]
    return {display_path(p): sha256(p) for p in sorted(paths)}


def _artifact(directory, names):
    """Read plain filenames and older prefixed artifact names without old code."""
    for name in names:
        path = Path(directory) / name
        if path.is_file():
            return path
    matches = [p for p in Path(directory).iterdir() if p.is_file()
               and any(p.name.lower().endswith("_" + name.lower()) for name in names)]
    if not matches:
        raise FileNotFoundError(f"Missing {names} artifact in {directory}")
    # A prefixed attempt ledger can coexist with its longer 'prior' ledger.
    shortest = min(len(p.name) for p in matches)
    matches = [p for p in matches if len(p.name) == shortest]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous {names} artifact in {directory}")
    return matches[0]


def _resolver(directory, kernel):
    directory = Path(directory).resolve()
    prefix = Path(kernel["template_path"]).parents[2]
    def resolve(value):
        path = Path(value)
        try:
            candidate = directory / path.relative_to(prefix)
            if candidate.exists():
                return candidate
        except ValueError:
            pass
        candidate = repo_path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing archived evidence: {value}")
    return resolve


def read_measurements(directory):
    """Authenticate physical measurements, independently of historical selectors."""
    directory = Path(directory).resolve()
    path = _artifact(directory, ["manifest.json"])
    manifest = json.loads(path.read_text())
    if manifest.get("seal"):
        payload = {k: v for k, v in manifest.items() if k != "seal"}
        if object_hash(payload) != manifest["seal"]:
            raise ValueError("Archived manifest seal mismatch")
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists() and sha256(path) != sidecar.read_text().strip():
        raise ValueError("Archived manifest hash mismatch")
    if manifest["context"] != CONTEXT:
        raise ValueError("Archived deployment context differs")
    candidates = read_jsonl(_artifact(directory, ["candidates.jsonl"]))
    if manifest.get("candidates_sha256") and object_hash(candidates) != manifest["candidates_sha256"]:
        raise ValueError("Archived candidate queue hash mismatch")
    by_id = {r["candidate_id"]: r for r in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate archived candidate IDs")
    kernels = manifest.get("kernels")
    if kernels is None:
        names = {r["kernel"] for r in candidates}
        if len(names) != 1:
            raise ValueError("Ambiguous single-kernel archive")
        kernels = {next(iter(names)): manifest["kernel"]}
    for name, digest in manifest.get("input_sha256", {}).items():
        matches = [_resolver(directory, k) for k in kernels.values()]
        resolved = None
        for resolve in matches:
            try:
                resolved = resolve(name)
                break
            except FileNotFoundError:
                pass
        if resolved is None or sha256(resolved) != digest:
            raise ValueError("Archived frozen input changed: " + name)
    rows = read_jsonl(_artifact(directory, ["measurements.jsonl", "synthesis_results.jsonl"]))
    inventory, audit = {}, []
    for row in rows:
        candidate = by_id[row["candidate_id"]]
        kernel = kernels[candidate["kernel"]]
        resolve = _resolver(directory, kernel)
        record = {"candidate_id": row["candidate_id"], "status": "historical_unverified"}
        try:
            if row.get("authentication") == "authenticated_reuse":
                proof = json.loads(resolve(row["reuse_proof"]).read_text())
                original = proof.get("candidate", proof.get("original_candidate"))
                result = proof.get("result", proof.get("original_result"))
                old_work = Path(proof.get("old_work", proof.get("original_work")))
                copied_work = resolve(proof["copied_work"])
                def physical(value, old_work=old_work, copied_work=copied_work, fallback=resolve):
                    try:
                        return copied_work / Path(value).relative_to(old_work)
                    except ValueError:
                        return fallback(value)
            else:
                original, result, physical = candidate, row, resolve
            if original["assignment"] != candidate["assignment"]:
                raise ValueError("Reuse proof assignment mismatch")
            verified = hls.authenticate_work(original, result, kernel, physical,
                expected_evaluation_hash=original["evaluation_context_hash"])
            inventory[(candidate["kernel"], candidate["canonical_signature"])] = {
                "candidate": original, "result": result, "kernel": kernel,
                "resolve": physical, "verified": verified, "archive": display_path(directory)}
            record["status"] = "authenticated"
        except (ValueError, KeyError, OSError) as exc:
            record["reason"] = str(exc)
        audit.append(record)
    try:
        attempts = read_jsonl(_artifact(directory, ["attempts.jsonl"]))
    except FileNotFoundError:
        attempts = []
    cost = manifest.get("prior_new_synthesis_cost", 0) + len(attempts)
    return inventory, audit, cost


def snapshot_source(name, spec, output):
    """Copy inputs once; never materialize directives over the original source."""
    source, metadata = repo_path(spec["source"]), repo_path(spec["metadata"])
    if metadata.read_text().splitlines()[0].strip() != spec["top"]:
        raise ValueError(f"Source top differs from metadata for {name}")
    directory = Path(output) / "sources" / name
    files = []
    inputs = [source, metadata] + sorted(p for p in source.parent.rglob("*")
        if p.is_file() and p.suffix.lower() in {".h", ".hpp", ".hh", ".inc"}
        and "code_to_memory_outputs" not in p.parts)
    for original in dict.fromkeys(inputs):
        relative = Path(source.name) if original == source else Path("kernel_info.txt") if original == metadata else original.relative_to(source.parent)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != sha256(original):
            raise ValueError("Existing source snapshot differs")
        if not target.exists():
            shutil.copy2(original, target)
        files.append(
            {
                "original": display_path(original),
                "snapshot": display_path(target),
                "sha256": sha256(original)
            }
        )
    raw = directory / source.name
    actions = legacy.load_source_actions(name, directory)
    template = legacy.prepare_source_template(name, raw, directory)
    template_path = directory / "source_template.txt"
    template_path.write_text(template)
    kernel = {"top": spec["top"], "source_provenance": {
        "source": display_path(source), "source_sha256": sha256(source),
        "metadata": display_path(metadata), "metadata_sha256": sha256(metadata),
        "snapshot_source": display_path(raw), "snapshot_metadata": display_path(directory / "kernel_info.txt"),
        "files": files}, "template_path": display_path(template_path), "template_sha256": sha256(template_path)}
    return kernel, actions, template


def prepare_kernel(name, spec, output, args):
    from . import relations
    kernel, actions, template = snapshot_source(name, spec, output)
    rows = [r for r in read_jsonl(repo_path(args.baselines)) if r.get("kernel") == name
            and r.get("stage") == "predicted-directives" and r.get("objective") == "ADP"]
    if len(rows) != 1:
        raise ValueError(f"Expected one complete measured ADP prediction for {name}")
    kernel.update(paper_baseline=rows[0], initial_assignment=c.from_paper(rows[0], actions))
    graph = Path(output) / "graphs" / (name + ".gexf")
    cached = repo_path(args.graphs_dir) / (name + ".gexf")
    tracked = repo_path(spec["relations"]) if spec.get("relations") else None
    try:
        graph.parent.mkdir(parents=True, exist_ok=True)
        if cached.is_file():
            shutil.copy2(cached, graph)
        elif tracked and tracked.is_file():
            shutil.copy2(tracked, graph)
        else:
            tools = relations.compiler_tools(args.polygeist_build, args.mlir_python)
            kernel["graph_generation"] = relations.regenerate_graph(
                repo_path(kernel["source_provenance"]["snapshot_source"]), graph, tools,
                timeout=args.graph_timeout, compiler_symbol=spec.get("compiler_symbol"))
        active = repo_path(spec["relation_metadata"]) if spec.get("relation_metadata") else None
        if spec.get("relation_source"):
            related = repo_path(spec["relation_source"])
            if sha256(related) != kernel["source_provenance"]["source_sha256"]:
                raise ValueError("Compiler-active source differs from the synthesized source")
            for entry in kernel["source_provenance"]["files"]:
                original = repo_path(entry["original"])
                if original.suffix.lower() in {".h", ".hpp", ".hh", ".inc"}:
                    corresponding = related.parent / original.relative_to(repo_path(spec["source"]).parent)
                    if not corresponding.is_file() or sha256(corresponding) != entry["sha256"]:
                        raise ValueError("Compiler and synthesis analysis headers differ")
        record = relations.graph_relations(graph, kernel["source_provenance"]["source_sha256"],
            kernel["source_provenance"]["metadata_sha256"], actions, active)
        if record["graph_metadata"].get("kernel") != kernel["top"]:
            raise ValueError("Graph top-function provenance mismatch")
        if spec.get("compiler_symbol") and record["graph_metadata"].get("compiler_symbol") != spec["compiler_symbol"]:
            raise ValueError("Graph C++ symbol provenance mismatch")
        if active:
            target = Path(output) / "graphs" / (name + "_active") / "kernel_info.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(active, target)
            record["compiler_active_metadata_path"] = display_path(target)
        kernel["relations"] = record
        verified, inactive = relations.audit_kernel(kernel, actions)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
        kernel["relations"] = {"status": "blocked", "reason": str(exc),
            "preparation": "Set POLYGEIST_BUILD and MLIR_PYTHON following toolchain/README.md, then prepare into a new output directory."}
        verified, inactive = [], []
    factor_actions = c.compiler_actions(kernel, actions) if args.domain_policy == "tcad" else actions
    return kernel, factor_actions, c.legacy_domains(actions, repo_path(kernel["source_provenance"]["snapshot_metadata"]).parent), verified, inactive


def copy_measurement(proof, output):
    candidate, result, kernel, resolve = (proof[k] for k in ("candidate", "result", "kernel", "resolve"))
    work = resolve(result["receipt_path"]).parent
    directory = Path(output) / "reused" / (candidate["kernel"] + "-" + candidate["canonical_signature"])
    files = [resolve(result[k]) for k in ("receipt_path", "report_path", "log_path")]
    files += [work / "run_hls.tcl", work / Path(kernel["source_provenance"]["source"]).name]
    parent = Path(kernel["source_provenance"]["snapshot_source"]).parent
    files += [work / Path(f["snapshot"]).relative_to(parent) for f in kernel["source_provenance"]["files"]
        if Path(f["snapshot"]).suffix.lower() in {".h", ".hpp", ".hh", ".inc"} and Path(f["snapshot"]).name not in hls.VENDOR_HEADERS]
    for path in files:
        target = directory / path.relative_to(work)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    path = directory / "proof.json"
    write_json(path, {"candidate": candidate, "result": result,
        "old_work": str(Path(result["receipt_path"]).parent), "copied_work": display_path(directory), "archive": proof["archive"]})
    return display_path(path)


def verify_measurement(row, candidate, kernel):
    if row.get("authentication") == "authenticated_reuse":
        if row.get("prepared_context_hash") != candidate["evaluation_context_hash"]:
            raise ValueError("Reuse belongs to a different prepared context")
        proof = json.loads(repo_path(row["reuse_proof"]).read_text())
        if proof["candidate"]["assignment"] != candidate["assignment"]:
            raise ValueError("Reuse belongs to a different complete assignment")
        def resolve(value):
            try:
                return repo_path(proof["copied_work"]) / Path(value).relative_to(proof["old_work"])
            except ValueError:
                return repo_path(value)
        verified = hls.authenticate_work(proof["candidate"], proof["result"], kernel, resolve,
            expected_evaluation_hash=proof["candidate"]["evaluation_context_hash"])
    elif row.get("authentication") == "authenticated_new":
        verified = hls.authenticate_work(
            candidate,
            row,
            kernel,
            repo_path,
            expected_evaluation_hash=candidate["evaluation_context_hash"]
        )
    elif row.get("synthesis_status") == "success":
        raise ValueError("Successful measurement lacks authenticated evidence")
    else:
        return None
    if row["canonical_signature"] != candidate["canonical_signature"] or row["comparison_context_id"] != verified["comparison_context_id"]:
        raise ValueError("Measurement signature or comparison context differs")
    for key in ("ADP", "latency_cycles", "resources", "area_proxy", "feasible", "rankable"):
        if row[key] != verified[key]:
            raise ValueError("Stored measurement differs from authenticated XML")
    return verified


def load_output(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "seal"}
    if object_hash(payload) != manifest.get("seal"):
        raise ValueError("Immutable manifest seal changed")
    if manifest["code_sha256"] != implementation_hashes() or manifest["context"] != CONTEXT:
        raise ValueError("Frozen implementation or deployment changed; prepare a new output directory")
    for path, digest in manifest["input_sha256"].items():
        if sha256(repo_path(path)) != digest:
            raise ValueError("Immutable prepared input changed: " + path)
    candidates = read_jsonl(output / "candidates.jsonl")
    if len({r["candidate_id"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate identifiers")
    for name, kernel in manifest["kernels"].items():
        actions = c.recorded_actions(
            kernel,
            legacy.load_source_actions(name, repo_path(kernel["source_provenance"]["snapshot_metadata"]).parent)
        )
        for row in (r for r in candidates if r["kernel"] == name):
            if (c.signature(row["assignment"]) != row["canonical_signature"]
                    or c.parse_assignment(row["target"]) != row["assignment"]
                    or row["policy"] != c.policy_record(row["assignment"], kernel["initial_assignment"], actions, manifest["native_domains"][name], manifest["domain_policy"])):
                raise ValueError("Candidate assignment or policy changed")
    return manifest, candidates


def prepare(args):
    output = repo_path(args.output_dir).resolve()
    for path in args.reuse:
        archive = repo_path(path).resolve()
        if output == archive or archive in output.parents or output in archive.parents:
            raise ValueError("Output must be separate from every measurement archive")
    if (output / "manifest.json").exists():
        manifest, candidates = load_output(output)
        if set(args.kernels) != set(manifest["kernels"]) or args.seed != manifest["seed"] or args.domain_policy != manifest["domain_policy"]:
            raise ValueError("Prepared selection is immutable; choose another output directory")
        print(f"Already prepared: {len(candidates)} configurations. No synthesis launched.")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty; preserve it and choose a fresh directory")
    output.mkdir(parents=True)
    specs = copy.deepcopy(SPECS)
    if args.config:
        overrides = json.loads(repo_path(args.config).read_text())
        for name, values in overrides.items():
            specs[name] = {**specs.get(name, {}), **values}
    kernels, native, candidates, factorials, considered, evidence, limitations = {}, {}, [], [], [], [], {}
    for name in args.kernels:
        if name not in specs:
            raise ValueError("Missing source specification for " + name)
        kernel, actions, domains, relations, inactive = prepare_kernel(name, specs[name], output, args)
        selected, sets, pool, blocked = c.select_design(name, kernel["initial_assignment"], actions, domains,
            relations, inactive, args.domain_policy, args.seed)
        kernels[name], native[name], limitations[name] = kernel, domains, blocked
        candidates += selected; factorials += sets
        considered += [{**row, "kernel": name} for row in pool]
        evidence += [{**row, "kernel": name} for row in relations]
    code = implementation_hashes()
    for row in candidates:
        row["evaluation_context_hash"] = object_hash({"source_context": hls.source_context(kernels[row["kernel"]]),
            "assignment": row["canonical_signature"], "code": code, "domain_policy": args.domain_policy})
        text = hls.materialize(repo_path(kernels[row["kernel"]]["template_path"]).read_text(), row["target"])
        row["materialized_source_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    # Archive QoR is deliberately consulted only after proposals are selected.
    inventory, audits, imported_cost = {}, [], 0
    for path in dict.fromkeys(args.reuse):
        incoming, audit, charged = read_measurements(repo_path(path))
        inventory.update(incoming)
        audits += audit
        imported_cost += charged
    prior = max(args.prior_evaluations, imported_cost)
    if prior > MAX_EVALUATIONS:
        raise ValueError("Imported evaluation cost exceeds this experiment's hard budget")
    measurements, tools = [], {}
    for row in candidates:
        proof = inventory.get((row["kernel"], row["canonical_signature"]))
        if not proof or hls.source_context(proof["kernel"]) != hls.source_context(kernels[row["kernel"]]):
            continue
        verified = proof["verified"]
        tools[object_hash(verified["tool_identity"])] = verified["tool_identity"]
        measurements.append({**proof["result"], **verified, "candidate_id": row["candidate_id"],
            "canonical_signature": row["canonical_signature"], "authentication": "authenticated_reuse",
            "reuse_proof": copy_measurement(proof, output), "prepared_context_hash": row["evaluation_context_hash"]})
    if len(tools) > 1:
        raise ValueError("Reused measurements have different tool/header builds; use compatible archives")
    # Evaluate complete factorial blocks before remaining independent-only cells.
    priority = [r["candidate_id"] for r in candidates if not r["changed_action_count"]]
    for index in range(2):
        for name in args.kernels:
            sets = [t for t in factorials if t["group"] == "structural" and t["id"].startswith(name + "/")]
            if index < len(sets):
                priority += list(sets[index]["cells"].values())
    for test in (t for t in factorials if t["group"] == "control"):
        priority += list(test["cells"].values())
    priority += [r["candidate_id"] for r in candidates]
    order = {value: index for index, value in reversed(list(enumerate(priority)))}
    candidates.sort(key=lambda r: order[r["candidate_id"]])
    write_jsonl(output / "candidates.jsonl", candidates)
    audit = json.loads(repo_path(args.domain_audit).read_text()) if args.domain_audit else None
    files = [output / "candidates.jsonl"]
    for directory in ("sources", "graphs", "reused"):
        files += [p for p in (output / directory).rglob("*") if p.is_file()]
    manifest = {"experiment_id": "local-refinement", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "context": CONTEXT,
        "seed": args.seed, "domain_policy": args.domain_policy,
        "domain_policy_id": tcad.SOURCE_DOMAIN_POLICY if args.domain_policy == "tcad" else legacy.SOURCE_DOMAIN_POLICY,
        "kernels": kernels, "native_domains": native, "factorials": factorials,
        "candidate_pool": considered, "relation_evidence": evidence, "limitations": limitations,
        "archive_authentication": audits, "domain_audit": audit,
        "configuration_count": len(candidates), "initial_reuse_count": len(measurements),
        "prior_new_synthesis_cost": prior, "max_new_syntheses": MAX_EVALUATIONS - prior,
        "expected_tool_identity": next(iter(tools.values())) if tools else None,
        "code_sha256": code, "input_sha256": {display_path(p): sha256(p) for p in files},
        "candidate_policy": "QoR-blind; four diverse singles plus factorial singles; two verified pairs; one matched unverified pair; nearest supported increasing factors, fallback decreases; seed controls SHA ties.",
        "functional_validation": "not_tested; synthesis is not functional equivalence"}
    manifest["seal"] = object_hash(manifest)
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "measurements.jsonl", measurements)
    write_jsonl(output / "attempts.jsonl", [])
    summarize(output)
    print(
        f"Prepared {len(candidates)} configurations; {len(measurements)} authenticated reuses. No synthesis launched."
    )
    return manifest


def reserve_attempt(attempts, candidate_id, cap=MAX_EVALUATIONS):
    if len(attempts) >= min(MAX_EVALUATIONS, cap):
        raise ValueError("Hard synthesis budget exhausted")
    attempts.append({"attempt_index": len(attempts) + 1, "candidate_id": candidate_id,
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "reserved"})


def run(output, executable, limit=None, retry_failed=False, timeout=21600):
    output = Path(output)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, candidates = load_output(output)
        tool = hls.tool_info(executable); identity = hls.tool_identity(tool)
        attempts = read_jsonl(output / "attempts.jsonl")
        expected = manifest["expected_tool_identity"] or next(
            (entry["tool_identity"] for entry in attempts if entry.get("tool_identity")), None)
        if expected and expected != identity:
            raise ValueError("Tool/header build differs from the experiment")
        rows = read_jsonl(output / "measurements.jsonl")
        if len({r["candidate_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate measurements")
        done = {r["candidate_id"]: r for r in rows}
        launched = 0
        for candidate in candidates:
            key = candidate["candidate_id"]; kernel = manifest["kernels"][candidate["kernel"]]
            old = done.get(key)
            if old and old["synthesis_status"] == "success":
                if verify_measurement(old, candidate, kernel)["tool_identity"] != identity:
                    raise ValueError("Cached measurement tool differs")
                continue
            if candidate["changed_action_count"]:
                baseline = next(
                    r for r in candidates if r["kernel"] == candidate["kernel"] and not r["changed_action_count"]
                )
                if done.get(baseline["candidate_id"], {}).get("synthesis_status") != "success":
                    continue
            reserved = [r for r in attempts if r["candidate_id"] == key]
            if reserved:
                saved = sorted((output / "projects" / key).glob("attempt-*/result.json"))
                if saved:
                    result = json.loads(saved[-1].read_text())
                    if result["synthesis_status"] == "success":
                        result.update(hls.authenticate_work(candidate, result, kernel, repo_path,
                            expected_evaluation_hash=candidate["evaluation_context_hash"]), authentication="authenticated_new")
                        if result["tool_identity"] != identity:
                            raise ValueError("Recovered measurement tool differs")
                        result["canonical_signature"] = candidate["canonical_signature"]
                        done[key] = result
                        reserved[-1].update(status="success_recovered", hls_runtime_seconds=result["hls_runtime_seconds"])
                        write_jsonl(output / "attempts.jsonl", attempts)
                        write_jsonl(output / "measurements.jsonl", list(done.values()))
                        continue
                    if not old:
                        result.update(authentication="unsuccessful_new", canonical_signature=candidate["canonical_signature"])
                        done[key] = result
                        write_jsonl(output / "measurements.jsonl", list(done.values()))
                if not retry_failed:
                    continue
            if len(attempts) >= manifest["max_new_syntheses"] or limit is not None and launched >= limit:
                break
            reserve_attempt(attempts, key, manifest["max_new_syntheses"])
            attempts[-1]["tool_identity"] = identity
            write_jsonl(output / "attempts.jsonl", attempts)
            try:
                result = hls.run_candidate(candidate, kernel, output, tool, timeout)
                result["canonical_signature"] = candidate["canonical_signature"]
                if result["synthesis_status"] == "success":
                    result.update(hls.authenticate_work(candidate, result, kernel, repo_path,
                        expected_evaluation_hash=candidate["evaluation_context_hash"]), authentication="authenticated_new")
                else:
                    result["authentication"] = "unsuccessful_new"
                done[key] = result
                attempts[-1].update(status=result["synthesis_status"], hls_runtime_seconds=result["hls_runtime_seconds"])
            except BaseException as exc:
                attempts[-1].update(status="interrupted_or_preflight_failure", error=str(exc))
                write_jsonl(output / "attempts.jsonl", attempts)
                raise
            launched += 1
            write_jsonl(output / "attempts.jsonl", attempts)
            write_jsonl(output / "measurements.jsonl", list(done.values()))
        summarize(output)
        print(
            f"Launched {launched}; total charged evaluations {manifest['prior_new_synthesis_cost'] + len(attempts)}/{MAX_EVALUATIONS}."
        )


def _csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def summarize(output):
    output = Path(output); manifest, candidates = load_output(output)
    lookup = {r["candidate_id"]: r for r in candidates}
    rows = read_jsonl(output / "measurements.jsonl")
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate measurements")
    attempts = read_jsonl(output / "attempts.jsonl")
    expected = manifest["expected_tool_identity"] or next(
        (entry["tool_identity"] for entry in attempts if entry.get("tool_identity")), None)
    if any(entry.get("tool_identity", expected) != expected for entry in attempts):
        raise ValueError("Synthesis attempt tool differs from the experiment")
    for row in rows:
        candidate = lookup[row["candidate_id"]]
        verified = verify_measurement(row, candidate, manifest["kernels"][candidate["kernel"]])
        if verified and expected and verified["tool_identity"] != expected:
            raise ValueError("Measurement tool differs from sealed tool")
    by_id = {r["candidate_id"]: r for r in rows}
    factorial_rows, results, summary = [], {}, []
    lines = ["# Local refinement", "", "ADP, ZCU104, 10 ns, full resource capacities, Vitis HLS 2021.1. All comparisons use authenticated same-context measurements.", "",
        "Candidates are selected deterministically before consulting synthesis QoR. No model inference or training occurs in refinement.", "",
        "## Source and domain validation", "", "TCAD factors are powers of two ≤ n plus divisors ≥2; factors use the selected array dimension. Proven MLIR static loop counts override approximate metadata; unresolved loops have no enabled unroll proposals.", ""]
    for name, kernel in manifest["kernels"].items():
        prov = kernel["source_provenance"]
        lines += [
            f"{name}: source `{prov['source']}` SHA256 `{prov['source_sha256']}`; metadata `{prov['metadata_sha256']}`; compiler artifact `{kernel['relations'].get('artifact_hash', 'unavailable')}`."
        ]
        if kernel["relations"]["status"] != "available":
            lines += ["Structural preparation blocked: " + kernel["relations"]["reason"]]
        for proof in kernel.get("loop_domain_evidence", []):
            if proof["status"] != "static_agrees":
                lines += ["Loop-count evidence: `" + json.dumps(proof, sort_keys=True) + "`."]
    lines += [
        "",
        "Historical paper measurements are references, never automatically interchangeable with new synthesis. In particular GAN's floating-point-analysis-stub reference is separate from genuine vendor-header synthesis.",
        ""
    ]
    if manifest.get("domain_audit"):
        for name, audit in manifest["domain_audit"].get("datasets", {}).items():
            if audit.get("rows"):
                lines.append(
                    f"Metadata-conditional {name} coverage: {audit['rows']} rows; "
                    f"legacy={audit.get('legacy_admissible', 'unavailable')}, "
                    f"TCAD={audit.get('tcad_admissible', 'unavailable')}. "
                    "This is not a compiler trip-count audit of the whole corpus.")
    for test in manifest["factorials"]:
        cells = {slot: by_id.get(cid) for slot, cid in test["cells"].items()}
        result = c.analyze(cells); results[test["id"]] = result
        lines += ["## " + test["id"], "", f"{test['relation']['verified_relation_class']}: `{test['left']['updates']}` + `{test['right']['updates']}`.",
            test["relation"]["reason"], "", "| Cell | Status | Cycles | BRAM / DSP / FF / LUT | ADP | HLS clock ns |", "| --- | --- | ---: | --- | ---: | ---: |"]
        for slot in ("d0", "dA", "dB", "dAB"):
            cid = test["cells"][slot]; row = cells[slot]
            resource = "/".join(str(row["resources"][k]) for k in c.RESOURCE_KEYS) if row and row.get("resources") else "unavailable"
            lines.append(
                f"| {slot} | {row.get('authentication') if row else 'pending'} | {row.get('latency_cycles') if row else 'unavailable'} | {resource} | {row.get('ADP') if row else 'unavailable'} | {row.get('estimated_clock_ns') if row else 'unavailable'} |"
            )
            factorial_rows.append({"factorial": test["id"], "group": test["group"], "cell": slot, "candidate_id": cid,
                "status": row.get("synthesis_status") if row else "pending", "latency_cycles": row.get("latency_cycles") if row else None,
                "ADP": row.get("ADP") if row else None, "resource_feasible": row.get("feasible") if row else None,
                "estimated_clock_ns": row.get("estimated_clock_ns") if row else None,
                "requested_clock_ns": row.get("requested_clock_ns") if row else None,
                "estimated_clock_le_requested": row.get("estimated_clock_le_requested") if row else None,
                **{
                    k: row["resources"][k] if row and row.get("resources") else None for k in c.RESOURCE_KEYS
                },
                "interaction_ADP": result.get("interaction_ADP"), "G_joint_pct": result.get("G_joint_pct"),
                "new_tcad_values": json.dumps(lookup[cid]["policy"]["changed_values_new_under_tcad"], sort_keys=True)})
        if result["status"] == "measured":
            lines += [
                "",
                f"I_ADP={result['interaction_ADP']:.12g}; G_joint={result['G_joint_pct']:.6g}%. Individual/joint ADP improvements: `{result['improvement_pct']}`. Resource deltas: `{result['resource_deltas']}`.",
                ""
            ]
        else:
            lines += ["", "Attribution unavailable: " + result["reason"], ""]
        if test["group"] == "control":
            lines += [test["matching_note"], f"Matching score: `{test['match_score']}`.", ""]
    for name in manifest["kernels"]:
        base = next(r for r in candidates if r["kernel"] == name and not r["changed_action_count"])
        baseline = by_id.get(base["candidate_id"])
        for group in ("independent", "structural", "control"):
            selected = [r for r in candidates if r["kernel"] == name and any(m["group"] == group and
                (group == "independent" or m.get("cell") == "dAB") for m in r["memberships"])]
            measured = [by_id[r["candidate_id"]] for r in selected if r["candidate_id"] in by_id]
            valid = [
                r for r in measured if r.get("rankable") and r.get("synthesis_status") == "success" and baseline and r["comparison_context_id"] == baseline["comparison_context_id"]
            ]
            best = min(valid, key=lambda r: (r["ADP"], r["candidate_id"])) if valid else None
            summary.append({"kernel": name, "group": group, "candidate_count": len(selected), "measured_count": len(measured),
                "feasible_count": len(valid), "baseline_ADP": baseline.get("ADP") if baseline else None,
                "best_candidate": best["candidate_id"] if best else None, "best_ADP": best["ADP"] if best else None,
                "best_latency": best["latency_cycles"] if best else None,
                "improvement_pct": 100 * (baseline["ADP"] - best["ADP"]) / baseline["ADP"] if best and baseline["rankable"] else None})
    charged = manifest["prior_new_synthesis_cost"] + len(attempts)
    lines += [
        "## Summary",
        "",
        "| Kernel | Neighborhood | Feasible / selected | Best ADP | Improvement % |",
        "| --- | --- | --- | ---: | ---: |"
    ]
    for row in summary:
        lines.append(
            f"| {row['kernel']} | {row['group']} | {row['feasible_count']} / {row['candidate_count']} | {row['best_ADP']} | {row['improvement_pct']} |"
        )
    lines += [
        "",
        f"Unique configurations: {len(candidates)}; authenticated measurements: {len(rows)}; charged evaluations: {charged}/{MAX_EVALUATIONS}. Pending: {len(candidates) - len(rows)}. Newly launched HLS runtime: {sum(r.get('hls_runtime_seconds', 0) for r in attempts):.3f} s.",
        ""
    ]
    for group in ("structural", "control"):
        complete = [
            results[t["id"]] for t in manifest["factorials"] if t["group"] == group and results[t["id"]]["status"] == "measured"
        ]
        lines += [
            f"{group}: {len(complete)} complete factorials, {sum(r['interaction_ADP'] < 0 for r in complete)} favorable interactions, {sum(r['G_joint'] > 0 for r in complete)} joints better than both singles and baseline."
        ]
    violations = [r for r in rows if r.get("estimated_clock_le_requested") is False]
    lines += [
        "",
        f"HLS timing estimates above the requested clock: {len(violations)}. Resource feasibility and timing estimates are separate."
    ]
    for row in rows:
        if row.get("rankable") is not True:
            lines.append(
                f"Unrankable {row['candidate_id']}: status={row.get('synthesis_status')}, "
                f"resources={row.get('resources')}, full limits={CONTEXT['resource_budget']}, "
                f"error={row.get('error')}.")
    lines += [
        "",
        "A favorable interaction alone does not imply improvement. No verified relation is not proof of independence. Controls may be imperfectly matched; this small fixed-target study does not establish general structural superiority. HLS timing is estimated, not post-route timing. No functional simulation/equivalence was performed.",
        ""
    ]
    for name, limitations in manifest["limitations"].items():
        for limitation in limitations:
            lines.append(f"{name}: {limitation}.")
    if not results or any(r["status"] != "measured" for r in results.values()):
        lines += ["Research conclusions remain pending for incomplete comparisons."]
    else:
        useful = [name for name, r in results.items() if r["G_joint"] > 0]
        lines += [
            f"Joints improving over both singles and baseline: {useful}. Assess these alongside the independent results before adding a learned structural policy."
        ]
    _csv(output / "factorials.csv", factorial_rows); _csv(output / "summary.csv", summary)
    (output / "report.md").write_text("\n".join(lines))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "audit-domains", "run", "summarize"], default="prepare")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--kernels", nargs="+", default=["GAN", "covariance", "Kalman"])
    parser.add_argument("--domain-policy", choices=["tcad", "legacy"], default="tcad")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--baselines", default=str(ROOT / "paper_plots/SALT_experiments.jsonl"))
    parser.add_argument("--graphs-dir", default=str(DEFAULT_GRAPHS))
    parser.add_argument("--config", help="JSON overrides for source, metadata, top or graph identities")
    parser.add_argument(
        "--reuse",
        action="append",
        default=[],
        help="Read-only measured experiment directory; may be repeated"
    )
    parser.add_argument("--prior-evaluations", type=int, default=0)
    parser.add_argument("--domain-audit"); parser.add_argument("--sft-dataset")
    parser.add_argument("--polygeist-build"); parser.add_argument("--mlir-python"); parser.add_argument("--graph-timeout", type=int, default=600)
    parser.add_argument("--vitis-hls", default="vitis_hls"); parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--limit", type=int); parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.kernels = list(
            dict.fromkeys("covariance" if name.lower() == "covariance" else name for name in args.kernels)
        )
        if not 0 <= args.prior_evaluations <= MAX_EVALUATIONS or args.timeout <= 0 or args.graph_timeout <= 0:
            raise ValueError("Invalid evaluation budget or timeout")
        if args.limit is not None and not 1 <= args.limit <= MAX_EVALUATIONS:
            raise ValueError("--limit must be between 1 and 48")
        output = repo_path(args.output_dir)
        if args.mode == "prepare":
            prepare(args)
        elif args.mode == "run":
            run(output, args.vitis_hls, args.limit, args.retry_failed, args.timeout)
        elif args.mode == "audit-domains":
            if (output / "manifest.json").exists():
                load_output(output)
                print("Prepared inputs are immutable; use a separate output directory for a new audit.")
            else:
                c.audit_domains(output / "domain_audit.json", args.sft_dataset)
        else:
            summarize(output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("Refinement: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
