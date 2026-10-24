"""Reuse the existing compiler action projector without importing its CLI/model."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .r0_common import ROOT, sha256, display_path

RELATIONS = {1: "self", 2: "parent_loop", 4: "child_loop", 8: "loop_array_access",
             16: "proven_dependence_forward", 32: "proven_dependence_reverse"}
SCHEMA = "mailohls-action-relations-v1"
BUILDER = ROOT / "GNN_branch/build_structural_memory.py"


def existing_projector():
    """Compile the unchanged helper definitions, as existing helper tests do.

    The exporter's top level parses required training/checkpoint CLI arguments.
    AST extraction avoids those side effects, torch_geometric, and GPU/model
    loading. The projection algorithm itself is not copied or approximated.
    """
    import torch
    names = {"_edge_records", "_has_edge", "_numeric_node_map", "_loop_action_scope_pseudo",
             "_loop_region_distances", "build_action_relation_mask"}
    tree = ast.parse(BUILDER.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Existing relation projector changed; review R0 adapter")
    constants = {"NODE_TYPE_OP", "NODE_TYPE_PSEUDO_BLOCK", "NODE_TYPE_PRAGMA", "NODE_TYPE_ARRAY_SCOPE",
                 "FLOW_CONTROL", "FLOW_CALL", "FLOW_PSEUDO_BLOCK", "FLOW_LOOP_HIERARCHY",
                 "FLOW_ARRAY_SCOPE", "FLOW_REGION", "FLOW_MEMORY_DEPENDENCE", "REL_SELF", "REL_PARENT",
                 "REL_CHILD", "REL_ARRAY", "REL_DEP_FORWARD", "REL_DEP_REVERSE"}
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id in constants]
    namespace = {"torch": torch}
    exec(compile(ast.Module(assignments + nodes, type_ignores=[]), str(BUILDER), "exec"), namespace)
    return namespace["build_action_relation_mask"]


def pairs_from_bits(bits, active, labels, actions, evidence, inactive_actions=()):
    """Absolute Lk index alignment; only real active, non-self relations qualify."""
    n = len(active)
    if len(labels) != n or len(bits) != n or any(len(row) != n for row in bits):
        raise ValueError("Inconsistent relation tensor shapes")
    for i in range(n):
        if bool(active[i]):
            if labels[i] != i + 1 or f"L{i+1}" not in actions:
                raise ValueError("Sparse absolute Lk slot alignment mismatch")
        for j in range(n):
            value = bits[i][j]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value & ~63:
                raise ValueError("Unknown/noninteger relation bits")
            if value and (not active[i] or not active[j]):
                raise ValueError("Inactive slots contain relation bits")
        if active[i] and not bits[i][i] & 1:
            raise ValueError("Active action lost its self relation")
    observed = {f"L{i+1}" for i in range(n) if active[i]}
    if set(inactive_actions) - set(actions) or observed != set(actions) - set(inactive_actions):
        raise ValueError("Relations do not cover exactly the source actions")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not active[i] or not active[j]:
                continue
            forward, reverse = bits[i][j] & ~1, bits[j][i] & ~1
            combined = forward | reverse
            if not combined:
                continue
            # Validate the direction/type semantics rather than treating the
            # matrix as a generic attention adjacency.
            if bool(forward & 8) != bool(reverse & 8) or bool(forward & 2) != bool(reverse & 4) or bool(forward & 4) != bool(reverse & 2):
                raise ValueError("Asymmetric compiler relation semantics")
            if bool(forward & 16) != bool(reverse & 32) or bool(forward & 32) != bool(reverse & 16):
                raise ValueError("Asymmetric proven dependence semantics")
            kinds = {actions[f"L{i+1}"].kind, actions[f"L{j+1}"].kind}
            if combined & 8 and kinds != {"loop", "array"}:
                raise ValueError("Loop-array relation has incompatible action kinds")
            if combined & (2 | 4 | 16 | 32) and kinds != {"loop"}:
                raise ValueError("Loop hierarchy/dependence relation requires two loops")
            pairs.append({"action_i": f"L{i+1}", "action_j": f"L{j+1}",
                          "relation_type": [name for bit, name in RELATIONS.items() if bit != 1 and combined & bit],
                          "relation_bits": {"i_to_j": forward, "j_to_i": reverse},
                          **evidence})
    return pairs


def graph_relations(path, source_hash, metadata_hash, actions, relation_metadata_path=None):
    import networkx as nx
    if Path(path).suffix != ".gexf":
        raise ValueError("R0 requires an authenticated MLIR GEXF, not a dense attention or unverified memory pack")
    try:
        graph = nx.read_gexf(path)
    except Exception as exc:
        if isinstance(exc, OSError):
            raise
        raise ValueError(f"Unreadable MLIR graph: {exc}") from exc
    name = graph.graph.get("name", "")
    if not name.startswith("mailohls-meta-v1:"):
        raise ValueError("Not a provenance-bearing MLIR graph (LLVM/HARP is not accepted)")
    metadata = json.loads(name.split(":", 1)[1])
    relation_actions = actions
    if relation_metadata_path:
        from LLM_branch.common import directive_domains as dd
        relation_metadata_path = Path(relation_metadata_path)
        relation_actions = dd.load_source_actions("relation-source", relation_metadata_path.parent)
        if any(label not in actions or action != actions[label] for label, action in relation_actions.items()):
            raise ValueError("Compiler-active metadata is not an exact subset of the synthesis action contract")
        metadata_hash = sha256(relation_metadata_path)
    if metadata.get("source_sha256") != source_hash or metadata.get("action_sha256") != metadata_hash:
        raise ValueError("MLIR graph source/metadata hashes do not match this R0 source snapshot")
    if metadata.get("compiler_analysis_schema") != "mailohls-native-analysis-v3" or metadata.get("mlir_level") != "affine+scf+memref+arith+func":
        raise ValueError("Native compiler analysis at the required MLIR level is missing")
    n = max(int(label[1:]) for label in actions)
    node_ids, active, labels = [-1] * n, [False] * n, [-1] * n
    declared = {str(a.get("action_id")) for _, a in graph.nodes(data=True) if int(a.get("type", -1)) == 100}
    if declared != set(relation_actions):
        raise ValueError(f"Graph action set mismatch: graph={sorted(declared)}, compiler-active contract={sorted(relation_actions)}")
    for node, attrs in graph.nodes(data=True):
        label = str(attrs.get("action_id", ""))
        if label not in relation_actions:
            continue
        loop = int(attrs.get("is_loop", 0)) == 1
        array = int(attrs.get("type", -1)) == 104
        if not loop and not array:
            continue
        if (actions[label].kind == "loop") != loop:
            raise ValueError(f"{label}: semantic anchor kind mismatch")
        index = int(label[1:]) - 1
        if active[index]:
            raise ValueError(f"Duplicate semantic anchor for {label}")
        node_ids[index], active[index], labels[index] = int(node), True, index + 1
    if {f"L{i+1}" for i, flag in enumerate(active) if flag} != set(relation_actions):
        raise ValueError("Source action has no semantic MLIR anchor")
    _, bits, stats = existing_projector()(graph, node_ids=node_ids, active_mask=active, max_slots=n)
    evidence = {"compiler_evidence_source": "GNN_branch/build_structural_memory.py:build_action_relation_mask",
                "relation_schema": SCHEMA, "projector_sha256": sha256(BUILDER),
                "graph_or_memory_artifact": display_path(path), "artifact_hash": sha256(path),
                "compiler_analysis_schema": metadata["compiler_analysis_schema"],
                "dependence_policy": "Only certainty=proven flow-12 edges; uncertain analyses excluded"}
    inactive = sorted(set(actions) - set(relation_actions), key=lambda label: int(label[1:]))
    evidence["compiler_active_metadata"] = display_path(relation_metadata_path) if relation_metadata_path else None
    evidence["compiler_active_metadata_sha256"] = metadata_hash
    return {"status": "available", "pairs": pairs_from_bits(bits.tolist(), active, labels, actions, evidence, inactive),
            "artifact": display_path(path), "artifact_hash": sha256(path),
            "action_relation_bits": bits.tolist(), "active_mask": active, "labels": labels,
            "graph_metadata": metadata, "stats": stats, "inactive_actions": inactive,
            "inactive_policy": "Unrepresented actions retain their complete directive assignment, but never become edit sites",
            "compiler_active_metadata_path": display_path(relation_metadata_path) if relation_metadata_path else None,
            "compiler_active_metadata_sha256": metadata_hash}


def compiler_tools(build=None, python=None):
    """Find the existing native pipeline; never install/replace a toolchain."""
    selected = build or os.environ.get("POLYGEIST_BUILD")
    found = shutil.which(os.environ.get("CGEIST", "cgeist"))
    builds = [Path(selected)] if selected else []
    if found:
        default = Path(found).resolve().parents[1]
        builds += [default, default.parent / "build-mailohls-assertions"]
    for directory in dict.fromkeys(builds):
        libs = directory / "tools/mlir/python_packages/mlir_core"
        extensions = list(libs.glob("mlir/_mlir_libs/_mailohls_analysis*.so"))
        cgeist = directory / "bin/cgeist"
        if not cgeist.is_file() or not extensions:
            continue
        choices = [python or os.environ.get("MLIR_PYTHON"), str(ROOT / ".toolchain-python/bin/python"),
                   str(Path.home() / ".mlir-python311/bin/python"), shutil.which("python3.11"), sys.executable]
        for interpreter in dict.fromkeys(p for p in choices if p and Path(p).is_file()):
            env = {**os.environ, "PYTHONPATH": str(libs), "PYTHONHASHSEED": "0"}
            probe = subprocess.run([interpreter, "-c", "from mlir import ir; from mlir._mlir_libs import _mailohls_analysis"], env=env, capture_output=True, text=True, timeout=30)
            if probe.returncode == 0:
                return {"build": str(directory), "python": interpreter, "pythonpath": str(libs),
                        "cgeist": str(cgeist), "cgeist_sha256": sha256(cgeist),
                        "analysis_extension_sha256": sha256(extensions[0]),
                        "lock_file_sha256": sha256(ROOT / "toolchain/lock.env")}
    raise FileNotFoundError("Native MLIR/Polygeist pipeline unavailable. Build toolchain/bootstrap.sh, then set POLYGEIST_BUILD and MLIR_PYTHON.")


def regenerate_graph(source, output, tools, timeout=600, compiler_symbol=None):
    command = [tools["python"], str(ROOT / "GNN_branch/mlir_graph_gen.py"), str(source),
               "--cgeist", tools["cgeist"], "--output", str(output),
               "--mlir-output", str(output.with_suffix(".mlir")),
               "--analysis-output", str(output.with_suffix(".analysis.json"))]
    if compiler_symbol is not None:
        command += ["--compiler-symbol", compiler_symbol]
    env = {**os.environ, "PYTHONPATH": tools["pythonpath"], "PYTHONHASHSEED": "0"}
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".generation.log")
    with log.open("w") as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"MLIR generation timed out; see {log}") from exc
    if result.returncode or not output.is_file():
        raise RuntimeError(f"MLIR generation failed (exit {result.returncode}); see {log}")
    return {"argv": command, "environment": {k: env[k] for k in ("PYTHONPATH", "PYTHONHASHSEED")}, "log": display_path(log)}
