"""Conservative call-site audit over existing graph evidence, with no schema changes."""
from __future__ import annotations

from collections import deque
import json

from experiments.refinement.r0_relations import existing_projector


def _edges(graph, node):
    return graph.out_edges(node, data=True)


def local_region(graph, loop):
    """Existing loop-region traversal restricted to the owning function; no calls."""
    owner = graph.nodes[loop].get("function")
    pending = deque(target for _, target, edge in _edges(graph, loop)
                    if int(edge.get("flow", -1)) == 8 and edge.get("role") == "region_entry")
    found = set()
    while pending:
        node = pending.popleft()
        if node in found or graph.nodes[node].get("function") != owner:
            continue
        found.add(node)
        for _, target, edge in _edges(graph, node):
            flow, role = int(edge.get("flow", -1)), edge.get("role")
            if (flow == 8 and role == "region_entry") or (flow == 0 and role in {"next_in_block", "block_successor"}):
                pending.append(target)
    return found


def _anchor(graph, label, kind):
    matches = [node for node, a in graph.nodes(data=True) if a.get("action_id") == label
               and ((kind == "loop" and int(a.get("is_loop", 0)) == 1)
                    or (kind == "array" and int(a.get("type", -1)) == 104))]
    if len(matches) != 1:
        raise ValueError(f"Missing/ambiguous semantic anchor for {label}")
    node = matches[0]
    if graph.nodes[node].get("action_resolution") != "source_location":
        raise ValueError(f"Non-exact source binding for {label}")
    return node


def _operation(graph, node):
    a = graph.nodes[node]
    return {"graph_node": str(node), **{key: a[key] for key in
        ("op_uid", "text", "full_text", "function", "block", "source_file", "source_line", "source_column") if key in a}}


def _root_access(graph, root, operation):
    """The compiler's concrete root -> access edge, never just SSA reachability."""
    return [dict(edge) for _, target, edge in _edges(graph, root) if target == operation
            and int(edge.get("flow", -1)) == 10 and edge.get("role") in {"read", "write", "readwrite"}]


def verify_pair(graph, array_label, loop_label, original=None, artifact_hash=None):
    array, loop = _anchor(graph, array_label, "array"), _anchor(graph, loop_label, "loop")
    roots = [target for _, target, edge in _edges(graph, array)
             if int(edge.get("flow", -1)) == 7 and edge.get("role") == "array_root"]
    accesses = {target for _, target, edge in _edges(graph, array)
                if int(edge.get("flow", -1)) == 7 and edge.get("role") in {"array_read", "array_write", "array_readwrite"}}
    owned = local_region(graph, loop)
    full = set(existing_projector().__globals__["_loop_region_distances"](graph, loop))
    supporting, callsites, direct, indirect = [], [], [], []
    valid_root = len(roots) == 1 and int(graph.nodes[roots[0]].get("is_memory_root", 0)) == 1
    if valid_root:
        root = roots[0]
        for operation in sorted(accesses & owned, key=str):
            bindings = _root_access(graph, root, operation)
            if bindings:
                direct.append(operation)
                supporting.append(_operation(graph, operation))
                callsites.append({"call_or_access_node": str(operation), "root_node": str(root),
                    "root_access_edges": bindings, "binding_rule": "compiler root-access edge in the loop's own function/region"})
        # Bind actual operands at this call to the called function's formals.
        # Shared formal nodes alone never establish a caller buffer identity.
        for call in sorted(owned, key=str):
            if graph.nodes[call].get("text") != "func.call":
                continue
            actuals = {n: int(edge["position"]) for n, _, edge in graph.in_edges(call, data=True)
                       if int(edge.get("flow", -1)) == 1 and edge.get("role") == "operand" and "position" in edge}
            callees = {graph.nodes[target].get("function") for _, target, edge in _edges(graph, call)
                       if int(edge.get("flow", -1)) == 2 and edge.get("role") == "calls"}
            for actual, position in actuals.items():
                if actual != root and not any(int(e.get("flow", -1)) == 9 and e.get("certainty") in {"proven", "must_alias"}
                    for _, target, e in _edges(graph, root) if target == actual):
                    continue
                for _, formal, binding in _edges(graph, actual):
                    if (int(binding.get("flow", -1)) != 2 or binding.get("role") != "actual_to_formal"
                            or graph.nodes[formal].get("function") not in callees):
                        continue
                    if "callsite" in binding:
                        matched = str(binding["callsite"]) == str(call)
                    else:
                        matched = int(binding.get("position", -1)) == 1 + position
                    if not matched:
                        continue
                    for operation in sorted(accesses & full, key=str):
                        if _root_access(graph, formal, operation):
                            indirect.append(operation)
                            supporting.append(_operation(graph, operation))
                            callsites.append({"call_node": str(call), "actual_node": str(actual), "actual_operand_position": position,
                                "formal_node": str(formal), "callee_function": graph.nodes[formal].get("function"), "binding_edge": dict(binding)})
    shared = sorted((accesses & full) - owned, key=str)
    if direct:
        classification, reason = "direct_verified", "Concrete compiler root-access edge and exact source action bindings in the loop's own region."
    elif indirect:
        classification, reason = "indirect_verified", "Call-site-specific actual/formal binding and callee memory access verified."
    elif accesses & full or (original and any(original.get("relation_bits", {}).values())) or not valid_root:
        classification, reason = "conservative_unknown", "Only shared helper reachability, unresolved root/access bindings, or legacy relation evidence; no context-specific proof."
    else:
        classification, reason = "no_evidenced_relation", "No qualifying access/binding established; this is not proof of independence."
    if not supporting:
        supporting = [_operation(graph, node) for node in shared]
    bindings = {label: {key: graph.nodes[node].get(key) for key in
        ("action_id", "action_resolution", "action_source_file", "action_source_line", "action_source_column", "array_var")}
        for label, node in [(array_label, array), (loop_label, loop)]}
    return {"action_i": array_label, "action_j": loop_label,
        "original_relation_bits": (original or {}).get("relation_bits", {"i_to_j": 0, "j_to_i": 0}),
        "original_relation_types": (original or {}).get("relation_type", []),
        "verified_relation_class": classification, "source_action_bindings": bindings,
        "relevant_memory_root": [_operation(graph, node) for node in roots],
        "supporting_mlir_operations": supporting, "supporting_graph_nodes": [op["graph_node"] for op in supporting],
        "shared_helper_nodes": shared, "callsite_provenance": callsites,
        "evidence_artifact_hash": artifact_hash, "reason": reason}


def audit_all(graph, actions, legacy_pairs, artifact_hash):
    by_pair = {frozenset((p["action_i"], p["action_j"])): p for p in legacy_pairs}
    return [verify_pair(graph, array, loop, by_pair.get(frozenset((array, loop))), artifact_hash)
        for array, a in actions.items() if a.kind == "array"
        for loop, b in actions.items() if b.kind == "loop"]
