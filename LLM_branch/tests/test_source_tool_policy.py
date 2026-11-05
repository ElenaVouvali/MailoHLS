import json
from pathlib import Path

import pytest

from LLM_branch.common import directive_domains
from LLM_branch.common.source_tool_policy import (
    filter_array_candidates, load_policy, validate_complete_assignment,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY = load_policy(ROOT / "experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy.json")
POLICY_V2 = load_policy(ROOT / "experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy_v2.json")


def test_gan_extent_50_rejects_factor_64_and_keeps_32_and_50():
    metadata = ROOT / "evaluation_set/GAN/code_to_memory_outputs/meta"
    actions = directive_domains.load_source_actions("forward_propagation", metadata)
    action = actions["L3"]
    assert action.array_name == "layer_2_out"
    original = ["0", "2", "4", "8", "16", "32", "64"]
    values, audit = filter_array_candidates(
        "AUTO{_ARRAY_F_L3}", original,
        {"AUTO{_ARRAY_T_L3}": "block", "AUTO{_ARRAY_D_L3}": "1"},
        action, POLICY,
    )
    assert "32" in values and "50" in values
    assert "64" not in values
    assert audit["rejected_candidates"]["64"] == ["factor_exceeds_selected_dimension_extent:64>50"]
    values, _ = filter_array_candidates(
        "AUTO{_ARRAY_F_L3}", ["3", "64"],
        {"AUTO{_ARRAY_T_L3}": "block", "AUTO{_ARRAY_D_L3}": "1"},
        action, POLICY,
    )
    assert "3" in values  # UG1399's count semantics permit non-divisor factors.
    assert "64" not in values


def test_kalman_large_complete_partition_is_rejected_by_tool_profile():
    actions = directive_domains.load_source_actions("serrano-kalman-filter")
    action = actions["L1"]
    assert action.array_name == "in_local"
    values, audit = filter_array_candidates(
        "AUTO{_ARRAY_T_L1}", ["block", "complete", "cyclic", "none"], {}, action, POLICY
    )
    assert "complete" not in values
    assert "block" in values and "cyclic" in values and "none" in values
    assert any("262144>1024" in reason for reason in audit["rejected_candidates"]["complete"])


def test_small_complete_partition_remains_and_disabled_sentinel_is_preserved():
    action = directive_domains.SourceAction(kind="array", dimensions=((1, 64),), array_name="small")
    values, _ = filter_array_candidates("AUTO{_ARRAY_T_L1}", ["block", "complete", "cyclic", "none"], {}, action, POLICY)
    assert set(values) == {"block", "complete", "cyclic", "none"}
    f_values, _ = filter_array_candidates("AUTO{_ARRAY_F_L1}", ["0", "64"],
        {"AUTO{_ARRAY_T_L1}": "none", "AUTO{_ARRAY_D_L1}": "0"}, action, POLICY)
    assert f_values == ["0"]


def test_multidimensional_selected_dimension_limit_is_applied_independently():
    action = directive_domains.SourceAction(
        kind="array", dimensions=((1, 4096), (2, 32)), array_name="matrix"
    )
    values, audit = filter_array_candidates("AUTO{_ARRAY_T_L1}", ["complete"],
        {"AUTO{_ARRAY_D_L1}": "1"}, action, POLICY)
    assert "complete" not in values
    assert "4096>1024" in audit["rejected_candidates"]["complete"][0]
    values, _ = filter_array_candidates("AUTO{_ARRAY_T_L1}", ["complete"],
        {"AUTO{_ARRAY_D_L1}": "2"}, action, POLICY)
    assert "complete" in values  # Partitioning only the 32-element second axis.


def test_multidimensional_dim_zero_uses_all_dimension_semantics():
    action = directive_domains.SourceAction(
        kind="array", dimensions=((1, 64), (2, 32)), array_name="matrix"
    )
    # A block/cyclic factor is applied to each dimension; it cannot exceed
    # the smaller extent. Complete dim=0 partitions the whole product.
    values, audit = filter_array_candidates(
        "AUTO{_ARRAY_F_L1}", ["16", "32", "33", "64"],
        {"AUTO{_ARRAY_T_L1}": "block", "AUTO{_ARRAY_D_L1}": "0"},
        action, POLICY_V2,
    )
    assert "32" in values
    assert "33" not in values and "64" not in values
    assert audit["rejected_candidates"]["64"]
    types, _ = filter_array_candidates(
        "AUTO{_ARRAY_T_L1}", ["complete"],
        {"AUTO{_ARRAY_D_L1}": "0"}, action, POLICY_V2,
    )
    # 64*32 exceeds the compiler profile's conservative complete limit.
    assert "complete" not in types
    small = directive_domains.SourceAction(kind="array", dimensions=((1, 8), (2, 4)))
    types, _ = filter_array_candidates(
        "AUTO{_ARRAY_T_L1}", ["complete"],
        {"AUTO{_ARRAY_D_L1}": "0"}, small, POLICY_V2,
    )
    assert "complete" in types
    from LLM_branch.common.mailohls_contract import validate_directive_assignments
    assignment = {
        "AUTO{_ARRAY_T_L1}": "complete",
        "AUTO{_ARRAY_F_L1}": "0",
        "AUTO{_ARRAY_D_L1}": "0",
    }
    with pytest.raises(ValueError):
        validate_directive_assignments(assignment)
    validate_directive_assignments(assignment, allow_dim_zero_all=True)
    validate_directive_assignments({
        "AUTO{_ARRAY_T_L1}": "cyclic",
        "AUTO{_ARRAY_F_L1}": "2",
        "AUTO{_ARRAY_D_L1}": "0",
    }, allow_dim_zero_all=True)


def test_dim_zero_candidate_policy_is_explicitly_versioned():
    assert POLICY.policy_id == "vitis_hls_2021_1_array_partition_guard_v1"
    assert POLICY.dim_zero_all_dimensions is False
    assert POLICY_V2.policy_id == "vitis_hls_2021_1_array_partition_guard_v2"
    assert POLICY_V2.dim_zero_all_dimensions is True
    from LLM_branch.common.mailohls_contract import filter_semantic_candidates
    domains = {"AUTO{_ARRAY_F_L1}": ["0"], "AUTO{_ARRAY_D_L1}": ["0"]}
    prefix = {"AUTO{_ARRAY_F_L1}": "0", "AUTO{_ARRAY_D_L1}": "0"}
    with pytest.raises(ValueError):
        filter_semantic_candidates("AUTO{_ARRAY_T_L1}", ["complete"], prefix, domains)
    assert filter_semantic_candidates("AUTO{_ARRAY_T_L1}", ["complete"], prefix, domains,
                                      allow_dim_zero_all=True) == ["complete"]


def test_unresolved_array_dimensions_fail_closed():
    action = directive_domains.SourceAction(kind="array", array_name="unknown")
    with pytest.raises(ValueError, match="dimension extents"):
        filter_array_candidates("AUTO{_ARRAY_T_L1}", ["complete", "none"], {}, action, POLICY)
    with pytest.raises(ValueError, match="dimension extents"):
        filter_array_candidates("AUTO{_ARRAY_T_L1}", ["complete", "none"],
                                {"AUTO{_ARRAY_D_L1}": "0"}, action, POLICY_V2)


def test_complete_assignment_is_checked_as_a_whole_tuple():
    actions = {"L1": directive_domains.SourceAction(kind="array", dimensions=((1, 50),))}
    good = {
        "AUTO{_ARRAY_T_L1}": "block", "AUTO{_ARRAY_F_L1}": "50", "AUTO{_ARRAY_D_L1}": "1"
    }
    validate_complete_assignment(good, actions, POLICY)
    validate_complete_assignment(dict(good, **{"AUTO{_ARRAY_F_L1}": "3"}), actions, POLICY)
    bad = dict(good, **{"AUTO{_ARRAY_F_L1}": "64"})
    with pytest.raises(ValueError, match="Source array tuple violation|Tool policy rejects"):
        validate_complete_assignment(bad, actions, POLICY)


def test_loop_tuple_remains_source_bounded():
    actions = {"L1": directive_domains.SourceAction(kind="loop", trip_count=8)}
    validate_complete_assignment({"AUTO{_PIPE_L1}": "1", "AUTO{_UNROLL_L1}": "8"}, actions, POLICY)
    with pytest.raises(ValueError, match="Source loop tuple violation"):
        validate_complete_assignment({"AUTO{_PIPE_L1}": "0", "AUTO{_UNROLL_L1}": "16"}, actions, POLICY)


def test_policy_is_versioned_and_reports_2021_1_profile():
    record = json.loads((ROOT / "experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy.json").read_text())
    assert record["policy_id"] == POLICY.policy_id
    assert POLICY.compiler == "AMD Vitis HLS"
    assert POLICY.version == "2021.1"
    assert POLICY.complete_partition_limit == 1024


def test_real_production_decoder_masks_tool_invalid_candidates_before_scoring(monkeypatch):
    torch = pytest.importorskip("torch")
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference

    source = """/* L1: */
auto{_ARRAY_T_L1}
auto{_ARRAY_F_L1}
auto{_ARRAY_D_L1}
"""
    registry = {"test_array": {
        "AUTO{_ARRAY_T_L1}": ["block", "complete", "cyclic", "none"],
        "AUTO{_ARRAY_F_L1}": ["0", "2", "4", "8", "16", "32", "64"],
        "AUTO{_ARRAY_D_L1}": ["0", "1"],
    }}
    scored_rhs = []

    def fake_score(*, candidate_text, **kwargs):
        rhs = candidate_text.strip()
        scored_rhs.append(rhs)
        # If the invalid factor reaches scoring it would win; candidate masking
        # must remove it first. Small-array complete partition remains legal.
        score = {"complete": -100.0, "64": 200.0, "50": 50.0}.get(rhs, 0.0)
        return {"mean_logprob": score, "sum_logprob": score, "token_count": 1}

    monkeypatch.setattr(inference, "score_rhs_candidate_suffix", fake_score)
    result = inference.constrained_decode_rhs_by_candidate_scoring(
        model=torch.nn.Linear(1, 1), tok=lambda text, **kwargs: {"input_ids": [1]},
        prompt_ids=[1], source_text=source, kernel_name="test-array",
        directive_domain_registry=registry, source_actions={
            "L1": directive_domains.SourceAction(kind="array", dimensions=((1, 50),), array_name="v")
        }, admissibility_policy=POLICY, debug_topk=1,
    )
    assert "64" not in scored_rhs
    assert "50" in scored_rhs
    assert result["admissibility_policy"]["final_assignment_validated"] is True
    assert "complete" in result["admissibility_policy"]["domain_audit"][0]["admissible_candidates"]
    assert result["admissibility_policy"]["domain_audit"][1]["rejected_candidates"]["64"]


def test_real_decoder_uses_v2_dim_zero_complete_tuple(monkeypatch):
    torch = pytest.importorskip("torch")
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference

    source = "/* L1: */\nauto{_ARRAY_T_L1}\nauto{_ARRAY_F_L1}\nauto{_ARRAY_D_L1}\n"
    registry = {"toy": {
        "AUTO{_ARRAY_T_L1}": ["block", "complete", "cyclic", "none"],
        "AUTO{_ARRAY_F_L1}": ["0", "2", "4"],
        "AUTO{_ARRAY_D_L1}": ["0", "1", "2"],
    }}

    def fake_score(*, candidate_text, **kwargs):
        rhs = candidate_text.strip()
        score = {"complete": 10.0, "0": 10.0}.get(rhs, 0.0)
        return {"mean_logprob": score, "sum_logprob": score, "token_count": 1}

    monkeypatch.setattr(inference, "score_rhs_candidate_suffix", fake_score)
    result = inference.constrained_decode_rhs_by_candidate_scoring(
        model=torch.nn.Linear(1, 1), tok=lambda text, **kwargs: {"input_ids": [1]},
        prompt_ids=[1], source_text=source, kernel_name="toy",
        directive_domain_registry=registry,
        source_actions={"L1": directive_domains.SourceAction(
            kind="array", dimensions=((1, 8), (2, 4)), array_name="matrix")},
        admissibility_policy=POLICY_V2, debug_topk=1,
    )
    from experiments.stage1_quality_sft.core import parse
    values = parse(result["canonical_prediction"])
    assert values["AUTO{_ARRAY_T_L1}"] == "complete"
    assert values["AUTO{_ARRAY_F_L1}"] == "0"
    assert values["AUTO{_ARRAY_D_L1}"] == "0"
    assert result["admissibility_policy"]["final_assignment_validated"] is True
