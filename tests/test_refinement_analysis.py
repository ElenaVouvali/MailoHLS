"""Independent scientific reporting contracts around the frozen search engine."""
import copy
import math

import pytest

from experiments import refinement_analysis as a
from experiments.refinement import configurations as c
from experiments.refinement_search import controller as ctl


def metric(adp, feasible=True, status="success", context="same"):
    return {"ADP": adp, "feasible": feasible, "rankable": feasible,
            "synthesis_status": status, "comparison_context_id": context}


def history(*results):
    return [{"signature": str(i), "measurement": row} for i, row in enumerate(results)]


def test_baseline_cost_and_failed_infeasible_queries_are_paid():
    rows = history(metric(100), metric(None, False, "timeout"), metric(1, False), metric(75))
    points = a.normalized_points(rows, None)
    assert [(r["budget"], r["ratio"]) for r in points] == [(1, 1), (2, 1), (4, .75), (8, None)]
    assert points[-1]["coverage"] == "incomplete"


def test_missing_replay_remains_undefined_and_no_future_extrapolation():
    rows = history(metric(100), metric(None, False, "unavailable"), metric(70))
    points = a.normalized_points(rows, None)
    assert points[0]["ratio"] == 1 and all(r["ratio"] is None for r in points[1:])
    assert all(r["ratio"] is None for r in a.normalized_points(rows, "neighborhood_exhausted")[1:])


def test_only_explicit_exhaustion_carries_valid_incumbent():
    rows = history(metric(100), metric(80))
    carried = a.normalized_points(rows, "neighborhood_exhausted")
    assert carried[-1]["ratio"] == .8 and carried[-1]["coverage"] == "carried_early_termination"
    assert a.normalized_points(rows, None)[-1]["ratio"] is None


@pytest.mark.parametrize("baseline", [metric(100, False), metric(None, False, "unavailable")])
def test_invalid_baseline_cannot_define_normalized_improvement(baseline):
    assert all(r["ratio"] is None for r in a.normalized_points(history(baseline, metric(30)), "neighborhood_exhausted"))


def test_joint_improvement_does_not_require_negative_interaction():
    value = a.factorial(*(metric(x) for x in (100, 80, 70, 60)))
    assert value["interaction_ADP"] == 10
    assert value["joint_gain_fraction_of_reference"] == .1


@pytest.mark.parametrize("replacement", [metric(60, False), metric(None, False, "timeout"), metric(60, context="other")])
def test_factorial_requires_all_feasible_cells_and_identical_context(replacement):
    assert a.factorial(metric(100), metric(80), metric(70), replacement) is None


def context(kernel, policy, value):
    return {"kernel": kernel, "policy": policy,
            "points": [{"budget": b, "ratio": value} for b in a.BUDGETS]}


def test_aggregate_uses_ratios_and_requires_complete_unique_kernel_coverage():
    rows = [context("small", "random", .5), context("large", "random", 1)]
    result = a.aggregate(rows, ["small", "large"], ["random"])[-1]
    assert result["arithmetic_mean"] == .75
    assert result["geometric_mean"] == pytest.approx(math.sqrt(.5))
    assert result["median"] == .75
    assert result["kernels_improved"] == result["kernels_unchanged"] == 1
    for invalid in (rows[:1], [rows[0], rows[0]], [rows[0], context("large", "random", None)]):
        assert a.aggregate(invalid, ["small", "large"], ["random"])[-1]["arithmetic_mean"] is None


def test_portable_export_rejects_tampered_input_code_and_seal(tmp_path):
    source = tmp_path / "report.xml"; source.write_text("original")
    report = tmp_path / "assessment.json"
    payload = {"assessment_code_sha256": c.sha256(a.__file__), "files_sha256": {str(source): c.sha256(source)}}
    c.write_json(report, ctl._sealed(payload))
    assert a.verify_export(report)["files_sha256"][str(source)] == c.sha256(source)
    source.write_text("tampered")
    with pytest.raises(ValueError, match="input changed"):
        a.verify_export(report)
    source.write_text("original")
    c.write_json(report, ctl._sealed({**payload, "assessment_code_sha256": "wrong"}))
    with pytest.raises(ValueError, match="implementation changed"):
        a.verify_export(report)
    data = ctl._sealed(payload); data["files_sha256"] = {}; c.write_json(report, data)
    with pytest.raises(ValueError, match="seal mismatch"):
        a.verify_export(report)


def test_functional_pass_requires_execution():
    with pytest.raises(ValueError, match="lacks execution"):
        a.verify_functional({"protocol_seal": "same", "executed": False}, {"seal": "same"}, {})


def test_supplementary_relation_reordering_preserves_all_edge_facts_and_multiplicity():
    first = {"actions": ["L1", "L2"], "dependence_evidence": [{"edge": {"certainty": "proven", "id": 1}}, {"edge": {"certainty": "proven", "id": 2}}]}
    other = copy.deepcopy(first); other["dependence_evidence"].reverse()
    a.compare_relation_facts(first, other)
    other["dependence_evidence"][0]["edge"]["certainty"] = "unknown"
    with pytest.raises(ValueError, match="dependence evidence differs"):
        a.compare_relation_facts(first, other)
    other = copy.deepcopy(first); other["actions"].reverse()
    with pytest.raises(ValueError, match="relation facts differ"):
        a.compare_relation_facts(first, other)
    other = copy.deepcopy(first); other["dependence_evidence"].append(other["dependence_evidence"][0])
    with pytest.raises(ValueError, match="dependence evidence differs"):
        a.compare_relation_facts(first, other)


def test_csv_export_preserves_missing_values_instead_of_inventing_future_points(tmp_path):
    row = {"kernel": "toy", "policy": "random", "seed": 123, "termination": None,
           "points": a.normalized_points(history(metric(100)), None), "cost": {}, "counts": {}}
    a.write_tables({"contexts": [row], "aggregate": [], "queries": []}, tmp_path)
    import csv
    exported = list(csv.DictReader((tmp_path / "contexts.csv").open()))[0]
    assert exported["ratio@1"] == "1.0" and exported["ratio@8"] == ""
    assert exported["coverage@8"] == "incomplete"
    assert b"\r\n" not in (tmp_path / "contexts.csv").read_bytes()
