"""Deterministic feedback ranking over the existing legal proposal pools."""
from __future__ import annotations

from experiments.refinement import configurations as c
from experiments.refinement_search import policies as base
from .feedback import history_feedback

POLICIES = ("fgr_static", "fgr_feedback", "fgr_structural")


def choose(policy, problem, history, incumbent, seed=123):
    if policy not in POLICIES:
        raise ValueError("Unknown FGR policy")
    if not history:
        return {"assignment": dict(problem["baseline"]), "signature": c.signature(problem["baseline"]),
                "reference_signature": c.signature(problem["baseline"]), "reference_changed_actions": [],
                "reference_changed_fields": [], "proposal_type": "baseline",
                "selection_reason": "Original complete SALT prediction; first logical evaluation",
                "domain_classification": c.policy_record(problem["baseline"], problem["baseline"], problem["actions"], problem["native_domains"], "tcad")}
    feedback_states = history_feedback(history, {
        "device_capacities": problem.get("device_capacities", c.CONTEXT["device_capacities"]),
        "resource_budget": problem.get("resource_budget", c.CONTEXT["resource_budget"])})
    singles = base.single_proposals(problem, incumbent, seed)
    joints = base.structural_proposals(problem, incumbent, seed)
    pool = singles + joints
    seen = {row["signature"] for row in history}
    pool = [row for row in pool if row["signature"] not in seen]
    if not pool:
        return None
    if policy == "fgr_static":
        # Same legal pool construction; deterministic round-robin proposal order.
        return {**pool[0], "proposal_type": "fgr_single" if len(pool[0]["reference_changed_actions"]) == 1 else "fgr_joint",
                "selection_reason": "P0 static order over the common legal single/joint pool"}
    measured = {r["signature"]: r for r in history}
    family_failures, family_success, family_savings = {}, {}, {}
    for r in history:
        if len(r["reference_changed_actions"]) != 1:
            continue
        family = r.get("family", "tuple")
        m = r["measurement"]
        usable = m.get("synthesis_status") == "success" and m.get("feasible") is True and m.get("ADP") is not None
        if usable:
            ref = measured.get(r.get("reference_signature"), {}).get("measurement", {})
            same_context = m.get("comparison_context_id") == ref.get("comparison_context_id")
            if ref.get("ADP") is not None and same_context and m["ADP"] < ref["ADP"]:
                family_success[family] = family_success.get(family, 0) + 1
            else:
                family_failures[family] = family_failures.get(family, 0) + 1
            before, after = ref.get("resources") or {}, m.get("resources") or {}
            if same_context and any(before.get(k) is not None and after.get(k) is not None and after[k] < before[k]
                                    for k in problem.get("resource_keys", ())):
                family_savings[family] = family_savings.get(family, 0) + 1
        else:
            family_failures[family] = family_failures.get(family, 0) + 1
    def rank(row):
        actions = row["reference_changed_actions"]
        is_joint = len(actions) > 1
        fam = row.get("family", "tuple")
        # Structural priority is strictly the last criterion in P2.
        unproductive = family_failures.get(fam, 0) > family_success.get(fam, 0)
        novelty = sum(1 for r in history if r.get("family", "tuple") == fam)
        joint_evidence = row.get("relation_evidence", {})
        joint_support = joint_evidence.get("verified_relation_class") in ("direct_verified", "indirect_verified")
        budget = problem.get("resource_budget", {})
        current = next((r["measurement"] for r in reversed(history)
                        if r["measurement"].get("feasible") is True and r["measurement"].get("resources")), {})
        tight = any(budget.get(k, 0) > 0 and
                    100.0 * (budget[k] - (current.get("resources") or {}).get(k, budget[k])) / budget[k] < 10.0
                    for k in problem.get("resource_keys", ()))
        observed_saving_priority = not (tight and family_savings.get(fam, 0) > 0)
        observed_improvement_priority = family_success.get(fam, 0) == 0
        return (unproductive, observed_improvement_priority, observed_saving_priority, novelty,
                is_joint if policy == "fgr_feedback" else not joint_support,
                base._tie(seed, row["signature"]))
    pool.sort(key=rank)
    selected = pool[0]
    return {**selected, "proposal_type": "fgr_joint" if len(selected["reference_changed_actions"]) > 1 else "fgr_single",
            "ranking_features": {"family": selected.get("family", "tuple"), "family_observations": sum(1 for r in history if r.get("family", "tuple") == selected.get("family", "tuple")),
                "family_improvements": family_success.get(selected.get("family", "tuple"), 0),
                "family_nonimprovements_or_infeasible": family_failures.get(selected.get("family", "tuple"), 0),
                "low_resource_headroom": any(problem.get("resource_budget", {}).get(k, 0) > 0 and
                    100.0 * (problem["resource_budget"][k] - (history[-1]["measurement"].get("resources") or {}).get(k, problem["resource_budget"][k])) / problem["resource_budget"][k] < 10.0
                    for k in problem.get("resource_keys", ())),
                "family_observed_resource_savings": family_savings.get(selected.get("family", "tuple"), 0),
                "structured_feedback_observations": sum(bool(state["quantities"]["ADP"]["available"]) for state in feedback_states.values()),
                "compiler_relation_enabled": policy == "fgr_structural",
                "verified_relation": bool(selected.get("relation_evidence", {}).get("verified_relation_class") in ("direct_verified", "indirect_verified"))},
            "selection_reason": f"{policy}: lexicographic observed-family productivity/diversity ranking; compiler priority {'enabled' if policy == 'fgr_structural' else 'disabled'}"}
