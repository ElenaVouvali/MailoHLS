"""CPU-only FGR checks; all QoR values below are explicitly synthetic fixtures."""
from copy import deepcopy

import pytest

from experiments.refinement import configurations as c
from experiments.refinement_feedback.feedback import parse_feedback
from experiments.refinement_feedback import policies as fgr
from experiments.refinement_feedback import experiment as adapter
from experiments.refinement_search import policies as base
from experiments.refinement_search import controller as ctl

pytest_plugins = ["test_refinement_search"]

from argparse import Namespace
from pathlib import Path
from experiments.refinement_feedback import experiment as adapter
from experiments.refinement_search import controller as ctl


def test_feedback_units_headroom_zero_budget_and_unknowns():
    context = {"device_capacities": {"BRAM_18K": 100, "DSP": 10, "FF": 1000, "LUT": 2000},
               "resource_budget": {"BRAM_18K": 50, "DSP": 0, "FF": 500, "LUT": 1000}}
    measurement = {"synthesis_status": "success", "feasible": False, "latency_cycles": 40,
        "resources": {"BRAM_18K": 60, "DSP": 0, "FF": 250, "LUT": 400}, "normalized_utilization":
        {"BRAM_18K": 60, "DSP": 0, "FF": 25, "LUT": 20}, "area_proxy": 26.25, "ADP": 1050,
        "estimated_clock_ns": 8.4, "comparison_context_id": "ctx"}
    state = parse_feedback("candidate", measurement, context)
    assert state.quantities["latency_cycles"].unit == "cycles"
    assert state.quantities["BRAM_18K_utilization_pct"].value == 60
    assert state.resource_headroom["BRAM_18K"].value == -20
    assert state.resource_headroom["DSP"].value == 100
    assert state.quantities["estimated_clock_ns"].kind == "measured_estimate"
    assert state.diagnostics["initiation_interval"].available is False
    assert "post-route" in state.quantities["estimated_clock_ns"].provenance


def test_feedback_delta_and_invalid_numbers():
    ctx = {"device_capacities": {k: 100 for k in c.RESOURCE_KEYS},
           "resource_budget": {k: 100 for k in c.RESOURCE_KEYS}}
    old = {"latency_cycles": 50, "ADP": 500, "resources": {k: 10 for k in c.RESOURCE_KEYS}}
    new = {"latency_cycles": 40, "ADP": 480, "area_proxy": 12,
           "resources": {k: 12 for k in c.RESOURCE_KEYS}}
    state = parse_feedback("id", new, ctx, old)
    assert state.deltas["latency_cycles"].value == -10
    assert state.deltas["BRAM_18K"].value == 2
    with pytest.raises(ValueError):
        parse_feedback("id", {**new, "ADP": float("nan")}, ctx, old)


def test_policy_pool_is_common_deterministic_and_feedback_changes_choice(source):
    _, problem = source
    baseline = problem["baseline"]
    initial = fgr.choose("fgr_feedback", problem, [], baseline, 77)
    assert initial["proposal_type"] == "baseline"
    assert fgr.choose("fgr_feedback", problem, [], baseline, 77) == initial
    pool_a = {r["signature"] for r in base.single_proposals(problem, baseline, 77) + base.structural_proposals(problem, baseline, 77)}
    pool_b = {r["signature"] for r in base.single_proposals(problem, baseline, 77) + base.structural_proposals(problem, baseline, 77)}
    assert pool_a == pool_b
    history = [{**initial, "measurement": {"ADP": 100, "synthesis_status": "success", "feasible": True}}]
    choice1 = fgr.choose("fgr_feedback", problem, history, baseline, 77)
    first_family = choice1.get("family", "tuple")
    bad = {**choice1, "measurement": {"ADP": 150, "synthesis_status": "success", "feasible": True}}
    good = {**choice1, "measurement": {"ADP": 90, "synthesis_status": "success", "feasible": True}}
    bad_history, good_history = history + [bad], history + [good]
    choice2 = fgr.choose("fgr_feedback", problem, bad_history, baseline, 77)
    good_choice = fgr.choose("fgr_feedback", problem, good_history, baseline, 77)
    assert choice2["signature"] != choice1["signature"]
    assert choice2["ranking_features"]["family_nonimprovements_or_infeasible"] >= 0
    assert good_choice["signature"] != choice2["signature"]
    assert fgr.choose("fgr_feedback", problem, bad_history, baseline, 77) == choice2


def test_structural_priority_requires_enabled_verified_relation(source):
    _, problem = source
    base0 = problem["baseline"]
    history = [{**fgr.choose("fgr_feedback", problem, [], base0), "measurement":
                {"ADP": 100, "synthesis_status": "success", "feasible": True,
                 "comparison_context_id": "ctx", "resources": {k: 1 for k in c.RESOURCE_KEYS}}}]
    p1 = fgr.choose("fgr_feedback", problem, history, base0, 2)
    p2 = fgr.choose("fgr_structural", problem, history, base0, 2)
    assert p1["ranking_features"]["compiler_relation_enabled"] is False
    assert p2["ranking_features"]["compiler_relation_enabled"] is True
    unknown = deepcopy(problem); unknown["relations"] = [dict(r, verified_relation_class="conservative_unknown") for r in problem["relations"]]
    joints = base.structural_proposals(unknown, base0, 2)
    assert not joints


def test_unknown_relationship_is_not_independence_and_inactive_preserved(source):
    _, problem = source
    problem["relations"] = []
    singles = base.single_proposals(problem, problem["baseline"])
    assert singles
    assert all(len(row["reference_changed_actions"]) == 1 for row in singles)
    problem["inactive_actions"] = ["L9"]
    assert all("L9" not in row["reference_changed_actions"] for row in base.single_proposals(problem, problem["baseline"]))


def test_scoped_adapter_exposes_paid_measurements_and_restores_hooks():
    before = (ctl.code_hashes, base.choose, base.POLICIES, ctl._history_entry, ctl.physical_candidate,
              ctl.summarize_all, ctl.problem_for)
    with pytest.raises(RuntimeError):
        with adapter.runtime():
            proposal = {"signature": "synthetic", "assignment": {}, "reference_signature": "synthetic",
                        "reference_changed_actions": [], "reference_changed_fields": [], "proposal_type": "baseline",
                        "selection_reason": "fixture"}
            record = {"result": {"ADP": 9, "latency_cycles": 3, "resources": {"BRAM_18K": 1},
                                  "normalized_utilization": {"BRAM_18K": 1}, "area_proxy": 3,
                                  "estimated_clock_ns": 7, "feasible": True, "rankable": True,
                                  "synthesis_status": "success", "comparison_context_id": "ctx"}}
            row = ctl._history_entry({"proposal": proposal}, record)
            assert row["measurement"]["resources"]["BRAM_18K"] == 1
            assert row["measurement"]["estimated_clock_ns"] == 7
            raise RuntimeError("fixture restoration")
    after = (ctl.code_hashes, base.choose, base.POLICIES, ctl._history_entry, ctl.physical_candidate,
             ctl.summarize_all, ctl.problem_for)
    assert after == before


def test_synthetic_fgr_controller_baseline_feedback_and_best_incumbent(prepared, oracle, tmp_path):
    _, args = prepared
    args = Namespace(**vars(args)); args.output_dir = str(tmp_path / "fgr-controller")
    args.policies = ["fgr_feedback"]; args.max_evaluations = 3
    unavailable_details = {"resources": None, "normalized_utilization": None,
                           "area_proxy": None, "latency_cycles": None}
    oracle.values = [{"ADP": 100, **unavailable_details}, {"ADP": 120, **unavailable_details},
                     {"ADP": 90, **unavailable_details}]
    with adapter.runtime():
        ctl.prepare(args)
        path = Path(args.output_dir) / "toy" / "fgr_feedback"
        ctl.search(path, limit=3)
        ctl.search(path)
        manifest, shared = ctl.load_run(path)
        traces, records, history, pending, terminal = ctl.read_state(path, manifest, shared)
        assert len(history) == 3 and pending is None and terminal == "budget_exhausted"
        assert history[0]["proposal_type"] == "baseline"
        assert history[1]["measurement"]["ADP"] == 120
        assert history[2]["signature"] != history[1]["signature"]
        assert ctl._best(history)["measurement"]["ADP"] == 90
        alternative = [dict(row) for row in history[:2]]
        alternative[1] = {**alternative[1], "measurement": {**alternative[1]["measurement"], "ADP": 80}}
        problem = ctl.problem_for(manifest, shared)
        changed_feedback_choice = fgr.choose("fgr_feedback", problem, alternative,
            history[0]["assignment"], manifest["seed"])
        assert changed_feedback_choice["signature"] != history[2]["signature"]
