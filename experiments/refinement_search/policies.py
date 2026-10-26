"""Pure candidate selection: source facts and this policy's revealed history only."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
import math

from experiments.refinement import configurations as c

POLICIES = ("random", "coordinate", "compose", "structural", "hybrid")
FAMILIES = ("UNROLL", "ARRAY_F", "PIPE", "ARRAY_T", "ARRAY_D", "tuple")
BATCH_SIZE = 4
STRUCTURAL_PERIOD = 4  # Hybrid reserves logical steps 4, 8, 12, 16.


def _tie(seed, value):
    return c.object_hash({"seed": seed, "value": value})


def _proposal(problem, reference, values, kind, reason, **metadata):
    if values == reference:
        return None
    try:
        policy = c.policy_record(values, problem["baseline"], problem["actions"],
                                 problem["native_domains"], "tcad")
    except (ValueError, KeyError):
        return None
    actions, fields = c.changed(reference, values, problem["actions"])
    return {"assignment": values, "signature": c.signature(values),
            "reference_signature": c.signature(reference),
            "reference_changed_actions": actions, "reference_changed_fields": fields,
            "proposal_type": kind, "selection_reason": reason,
            "domain_classification": policy, **metadata}


def single_proposals(problem, reference, seed=123):
    """Nearest smaller/larger factors, toggles and two legal whole-tuple edits/site.

    Round-robin action families; numerical families alternate lower and upper
    directions. Disable is a separate alternative. No direction predicts QoR.
    """
    groups = defaultdict(list)
    for label, action in problem["actions"].items():
        if label in problem.get("inactive_actions", ()):
            continue
        options = c._options(label, action, reference, problem["native_domains"], "tcad")
        bins = defaultdict(list)
        for option in options:
            family = option["family"]
            side = "categorical"
            if family in ("UNROLL", "ARRAY_F"):
                key, value = next(iter(option["updates"].items()))
                new, old = int(value), int(reference[key])
                side = "disable" if new == 0 else "lower" if new < old else "upper"
            bins[(family, side)].append(option)
        for (family, side), alternatives in bins.items():
            alternatives.sort(key=lambda row: (row["field_count"], row["distance"],
                              _tie(seed, row["updates"])))
            # Flat vocabularies never replace whole-action semantic validation.
            for option in alternatives[:2 if family == "tuple" else 1]:
                row = _proposal(problem, reference, {**reference, **option["updates"]},
                    "coordinate", "Balanced source-derived single-action neighborhood",
                    family=family, direction=side, distance=option["distance"])
                if row is not None:
                    groups[(family, side)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["distance"], _tie(seed, row["signature"])))
    family_lists = {}
    for family in FAMILIES:
        rows = []
        sides = ("lower", "upper", "disable") if family in ("UNROLL", "ARRAY_F") else ("categorical",)
        queues = [list(groups[(family, side)]) for side in sides]
        while any(queues):
            for queue in queues:
                if queue:
                    rows.append(queue.pop(0))
        family_lists[family] = rows
    result, seen = [], set()
    while any(family_lists.values()):
        for family in FAMILIES:
            if family_lists[family]:
                row = family_lists[family].pop(0)
                if row["signature"] not in seen:
                    result.append(row); seen.add(row["signature"])
    return result


def structural_proposals(problem, reference, seed=123):
    """Explore both endpoints without requiring beneficial constituent singles."""
    singles = single_proposals(problem, reference, seed)
    by_action = defaultdict(list)
    for row in singles:
        by_action[row["reference_changed_actions"][0]].append(row)
    pair_lists = []
    verified = [r for r in problem["relations"] if r["verified_relation_class"] in
                ("direct_verified", "indirect_verified") and r["action_i"] != r["action_j"]]
    verified.sort(key=lambda r: (r["verified_relation_class"] != "direct_verified",
        {problem["actions"][r["action_i"]].kind, problem["actions"][r["action_j"]].kind} != {"loop", "array"},
        _tie(seed, [r["action_i"], r["action_j"]])))
    for relation in verified:
        left, right = relation["action_i"], relation["action_j"]
        endpoints = []
        for label in (left, right):
            preferred = "UNROLL" if problem["actions"][label].kind == "loop" else "ARRAY_F"
            rows = by_action[label]
            preferred_rows = [r for r in rows if r["family"] == preferred]
            endpoints.append((preferred_rows or rows)[:2])
        rows = []
        for a, b in product(*endpoints):
            updates = {field: row["assignment"][field] for row in (a, b)
                       for field in row["reference_changed_fields"]}
            row = _proposal(problem, reference, {**reference, **updates}, "structural",
                "Verified compiler pair; constituent improvements are not required",
                relation_evidence=relation,
                compiler_relation_type=relation.get("verified_relation_types", relation.get("original_relation_types", [])))
            if row and len(row["reference_changed_actions"]) == 2:
                rows.append(row)
        pair_lists.append(rows)
    result, seen = [], set()
    while any(pair_lists):
        for rows in pair_lists:
            if rows:
                row = rows.pop(0)
                if row["signature"] not in seen:
                    result.append(row); seen.add(row["signature"])
    return result


def _rankable(row):
    result = row.get("measurement", {})
    return (result.get("synthesis_status") == "success" and result.get("rankable") is True
            and result.get("feasible") is True and result.get("ADP") is not None
            and math.isfinite(result["ADP"]) and result.get("comparison_context_id") is not None)


def composition_proposals(problem, history, incumbent, seed=123):
    """Only measured beneficial edits to distinct actions at the same reference.

    A composition is reference+A+B. Current endpoint tuples must still equal
    either the reference or their observed edit; unrelated changes cannot be
    rolled back. Stale combinations are rejected, never silently rebased.
    """
    observed = {row["signature"]: row for row in history}
    groups = defaultdict(list)
    for row in history:
        reference = observed.get(row.get("reference_signature"))
        if (row["proposal_type"] == "coordinate" and len(row["reference_changed_actions"]) == 1
                and reference and _rankable(reference) and _rankable(row)
                and row["measurement"]["comparison_context_id"] == reference["measurement"]["comparison_context_id"]
                and row["measurement"]["ADP"] < reference["measurement"]["ADP"]):
            groups[row["reference_signature"]].append(row)
    proposals, rejected = [], []
    for reference_sig, rows in groups.items():
        base = observed[reference_sig]["assignment"]
        for a, b in combinations(rows, 2):
            left, right = a["reference_changed_actions"][0], b["reference_changed_actions"][0]
            if left == right:
                continue
            labels = {left, right}
            compatible = True
            for label, action in problem["actions"].items():
                keys = [c.field_key(kind, label) for kind in c.KINDS[action.kind]]
                current = tuple(incumbent[key] for key in keys)
                allowed = {tuple(base[key] for key in keys)}
                if label in labels:
                    edit = a if label == left else b
                    allowed.add(tuple(edit["assignment"][key] for key in keys))
                if current not in allowed:
                    compatible = False; break
            if not compatible:
                rejected.append({"reference": reference_sig, "actions": sorted(labels), "reason": "stale_or_unrelated_incumbent_edits"})
                continue
            updates = {key: row["assignment"][key] for row in (a, b) for key in row["reference_changed_fields"]}
            candidate = {**base, **updates}
            if candidate == incumbent:
                continue
            gain = sum(observed[reference_sig]["measurement"]["ADP"] - row["measurement"]["ADP"] for row in (a, b))
            proposal = _proposal(problem, base, candidate, "composition",
                "Combine two observed beneficial singles from the same reference; rank by observed gains only",
                constituent_signatures=[a["signature"], b["signature"]], observed_gain_sum=gain)
            if proposal:
                proposals.append(proposal)
    proposals.sort(key=lambda r: (-r["observed_gain_sum"], _tie(seed, r["signature"])))
    return proposals, rejected


def _coordinate_batch(problem, history, incumbent, seed):
    """Four sibling probes let composition observe two gains at one reference."""
    coordinates = [row for row in history if row["proposal_type"] == "coordinate" and "batch_id" in row]
    last = coordinates[-1] if coordinates else None
    batch_id = last["batch_id"] if last else 0
    reference = incumbent
    reanchor = history[-1]["proposal_type"] == "composition" or (
        history[-1]["proposal_type"] == "structural" and history[-1]["signature"] == c.signature(incumbent))
    if last and not reanchor:
        same_batch = [r for r in coordinates if r["batch_id"] == batch_id]
        old_reference = next(r["assignment"] for r in history if r["signature"] == last["reference_signature"])
        if len(same_batch) < BATCH_SIZE:
            reference = old_reference
        else:
            batch_id += 1
    elif last:
        batch_id += 1
    seen = {row["signature"] for row in history}
    available = [r for r in single_proposals(problem, reference, seed) if r["signature"] not in seen]
    if not available and reference != incumbent:
        batch_id += 1
        available = [r for r in single_proposals(problem, incumbent, seed) if r["signature"] not in seen]
    if available:
        return {**available[0], "batch_id": batch_id,
                "selection_reason": "Composition reference sweep (at most four single-action probes); " + available[0]["selection_reason"]}
    return None


def choose(policy, problem, history, incumbent, seed=123):
    """One deterministic decision. Neither a cache nor other histories is passed."""
    if policy not in POLICIES:
        raise ValueError("Unknown search policy")
    seen = {row["signature"] for row in history}
    if not history:
        base = problem["baseline"]
        return {"assignment": dict(base), "signature": c.signature(base),
                "reference_signature": c.signature(base), "reference_changed_actions": [],
                "reference_changed_fields": [], "proposal_type": "baseline",
                "selection_reason": "Original complete SALT prediction; first logical evaluation",
                "domain_classification": c.policy_record(base, base, problem["actions"], problem["native_domains"], "tcad")}
    if policy == "hybrid" and (len(history) + 1) % STRUCTURAL_PERIOD == 0:
        structural = [r for r in structural_proposals(problem, incumbent, seed) if r["signature"] not in seen]
        if structural:
            return {**structural[0], "allocation": "reserved_structural_slot"}
    if policy in ("compose", "hybrid"):
        compositions, rejected = composition_proposals(problem, history, incumbent, seed)
        compositions = [r for r in compositions if r["signature"] not in seen]
        if compositions:
            return {**compositions[0], "rejected_compositions": rejected}
        candidate = _coordinate_batch(problem, history, incumbent, seed)
        if candidate:
            return {**candidate, "rejected_compositions": rejected,
                    "allocation": "single_or_structural_fallback" if policy == "hybrid" else "reference_sweep"}
        if policy == "compose":
            return None
    if policy in ("structural", "hybrid"):
        return next((r for r in structural_proposals(problem, incumbent, seed) if r["signature"] not in seen), None)
    singles = [r for r in single_proposals(problem, incumbent, seed) if r["signature"] not in seen]
    if policy == "random":
        singles.sort(key=lambda r: _tie(seed, [c.signature(incumbent), r["signature"]]))
        if singles:
            return {**singles[0], "proposal_type": "random", "selection_reason": "Seeded random order over the same legal single-action neighborhood"}
    return singles[0] if singles else None
