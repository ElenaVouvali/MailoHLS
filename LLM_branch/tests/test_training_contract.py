"""Contract, split, and prediction-metric regression tests."""

from __future__ import annotations

import copy
import json

import pytest

import LLM_branch.train.train_SFT_xattn_new as trainer
from LLM_branch.common import mailohls_contract
from LLM_branch.train.train_SFT_xattn_new import (
    STAGE1_COMPATIBILITY_FIELDS,
    assert_disjoint_nonempty_kernel_splits,
    build_selection_cases,
    build_clock_pack,
    build_deterministic_rhs_pack,
    compute_directive_loss_weights,
    evaluate_prediction,
    get_rhs_candidates_for_lhs,
    load_directive_domain_registry,
    require_compatible_stage1_contract,
    summarize_selection_rows,
)


class _CharacterTokenizer:
    eos_token = "<EOS>"

    def __init__(self):
        self.ids = {"<CLOCK>": 900, "<L1>": 901, "<EOS>": 902}

    def __call__(self, text, add_special_tokens=False):
        output = []
        index = 0
        while index < len(text):
            matched = next(
                (token for token in self.ids if text.startswith(token, index)),
                None,
            )
            if matched is not None:
                output.append(self.ids[matched])
                index += len(matched)
            else:
                output.append(ord(text[index]))
                index += 1
        return {"input_ids": output}


def _contract():
    contract = {
        "schema": "mailohls-training-contract-v1",
        "stage": "stage1",
    }
    contract.update({key: f"value-for-{key}" for key in STAGE1_COMPATIBILITY_FIELDS})
    return contract


def _platform_row():
    return {
        "kernel_name": "kernel-a",
        "device": "xczu7ev-ffvc1156-2-e",
        "clock_period": 5.0,
    }


@pytest.mark.parametrize(
    ("objective", "token"),
    [
        ("PARETO_LATENCY", "<OBJ=PARETO_LATENCY>"),
        ("PARETO_ADP", "<OBJ=PARETO_ADP>"),
        ("PARETO_AREA", "<OBJ=PARETO_AREA>"),
    ],
)
def test_canonical_prompt_supports_every_objective(objective, token):
    fields = mailohls_contract.target_prompt_fields(_platform_row())
    prompt = mailohls_contract.build_prompt("L1: for (;;) {}", objective, fields)
    assert token in prompt
    assert "<SRC_L1>" in prompt
    other_tokens = {
        spec["token"] for name, spec in mailohls_contract.GOALS.items()
        if name != objective
    }
    assert all(other not in prompt for other in other_tokens)


def test_all_expands_to_all_three_canonical_objectives():
    assert mailohls_contract.resolve_objectives("ALL") == (
        "PARETO_LATENCY", "PARETO_ADP", "PARETO_AREA"
    )
    for objective in mailohls_contract.resolve_objectives("ALL"):
        assert mailohls_contract.resolve_objectives(objective) == (objective,)
    with pytest.raises(ValueError, match="Unsupported objective"):
        mailohls_contract.resolve_objectives("PARETO_LATENCY_EXTREME")


def test_stage2_rejects_stage1_contract_mismatch(tmp_path):
    stage1 = _contract()
    (tmp_path / "training_contract.json").write_text(json.dumps(stage1))
    expected = copy.deepcopy(stage1)
    expected["dataset_sha256"] = "different-dataset"

    with pytest.raises(ValueError, match="dataset_sha256"):
        require_compatible_stage1_contract(tmp_path, expected)


def test_directive_domains_are_exact_per_kernel_and_site(tmp_path):
    path = tmp_path / "domains.json"
    path.write_text(json.dumps({
        "schema": "mailohls-directive-domain-registry-v1",
        "kernels": {
            "kernel-a": {
                "auto{_PIPE_L1}": ["0", "1"],
                "auto{_UNROLL_L1}": ["0", "2", "4"],
            },
            "kernel-b": {"auto{_UNROLL_L1}": ["0", "3"]},
        },
    }))
    registry = load_directive_domain_registry(str(path))
    assert get_rhs_candidates_for_lhs(
        "kernel-a", "auto{_UNROLL_L1}", registry
    ) == ["0", "2", "4"]
    assert get_rhs_candidates_for_lhs(
        "kernel-b", "auto{_UNROLL_L1}", registry
    ) == ["0", "3"]
    with pytest.raises(KeyError, match="legal RHS domain"):
        get_rhs_candidates_for_lhs("kernel-b", "auto{_PIPE_L1}", registry)


def test_clock_and_schema_tokens_are_never_supervised_by_default():
    tok = _CharacterTokenizer()
    clock = build_clock_pack({"selected_clock_period": 5.0}, tok)
    assert 900 in clock.input_ids
    assert 900 not in {label for label in clock.labels if label != -100}
    fixed_length = clock.input_ids.index(ord("5"))
    assert clock.labels[:fixed_length] == [-100] * fixed_length

    source = "<L1>\nauto{_PIPE_L1} = ?"
    target = "auto{_PIPE_L1} = 1"
    pack = build_deterministic_rhs_pack(source, target, tok)
    supervised = {label for label in pack.labels if label != -100}
    assert 901 not in supervised
    assert 902 not in pack.input_ids

    eos_pack = build_deterministic_rhs_pack(
        source, target, tok, supervise_eos=True
    )
    assert eos_pack.input_ids[-1] == 902
    assert eos_pack.labels[-1] == 902


def test_directive_loss_weighting_is_uniform_by_default_and_train_only():
    train_rows = [
        {
            "input": "<L1>\nauto{_PIPE_L1} = ?\nauto{_UNROLL_L1} = ?",
            "target": "auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 2",
        },
        {
            "input": "<L1>\nauto{_PIPE_L1} = ?",
            "target": "auto{_PIPE_L1} = 0",
        },
    ]
    assert compute_directive_loss_weights(train_rows, "uniform") == {}
    weights = compute_directive_loss_weights(
        train_rows, "inverse_sqrt_frequency"
    )
    assert weights["UNROLL"] > weights["PIPE"]

    # A held-out row cannot influence weights unless explicitly passed in.
    heldout = {
        "input": "<L1>\nauto{_ARRAY_D_L1} = ?",
        "target": "auto{_ARRAY_D_L1} = 1",
    }
    assert "ARRAY_D" not in weights
    assert "ARRAY_D" in compute_directive_loss_weights(
        train_rows + [heldout], "inverse_sqrt_frequency"
    )


def test_family_kernel_sets_must_be_disjoint_and_nonempty():
    assert_disjoint_nonempty_kernel_splits(
        [{"kernel_name": "train-kernel"}],
        [{"kernel_name": "val-kernel"}],
        [{"kernel_name": "test-kernel"}],
    )
    with pytest.raises(AssertionError, match="overlap"):
        assert_disjoint_nonempty_kernel_splits(
            [{"kernel_name": "shared"}],
            [{"kernel_name": "shared"}],
            [{"kernel_name": "test-kernel"}],
        )
    with pytest.raises(AssertionError, match="empty"):
        assert_disjoint_nonempty_kernel_splits(
            [{"kernel_name": "train-kernel"}], [], [{"kernel_name": "test-kernel"}]
        )


def test_prediction_metrics_distinguish_schema_keys_and_values():
    reference = "auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 0"
    exact = evaluate_prediction(reference, reference)
    assert exact["schema_compliant"]
    assert exact["expected_key_match"]
    assert exact["exact_design_match"]

    wrong_value = evaluate_prediction(
        reference, "auto{_PIPE_L1} = 2\nauto{_UNROLL_L1} = 0"
    )
    assert wrong_value["schema_compliant"]
    assert wrong_value["expected_key_match"]
    assert not wrong_value["exact_design_match"]

    missing_key = evaluate_prediction(reference, "auto{_PIPE_L1} = 1")
    assert not missing_key["schema_compliant"]
    assert not missing_key["expected_key_match"]
    assert not missing_key["exact_design_match"]


def test_selection_cases_are_limited_per_distinct_kernel(monkeypatch):
    monkeypatch.setattr(
        trainer,
        "target_bucket_key",
        lambda row: (row["kernel_name"], row["device"], row["bucket"]),
    )
    monkeypatch.setattr(
        trainer,
        "shared_budget_fraction",
        lambda row: float(row["bucket"]),
    )
    monkeypatch.setattr(
        trainer,
        "build_partial_deterministic_target_text",
        lambda source, target, minimum: (
            target,
            {"coverage": 1.0, "n_supervised": minimum},
        ),
    )
    rows = [
        {
            "kernel_name": kernel,
            "device": device,
            "bucket": bucket,
            "obj_mode": "PARETO_ADP",
            "frequency_mode": "specified",
            "input": f"source-{kernel}-{bucket}",
            "target": f"target-{kernel}-{bucket}",
            "_rank_within_kernel": 0,
            "_score": float(bucket),
        }
        for kernel in ("kernel-a", "kernel-b", "kernel-c")
        for device in ("device-a", "device-b")
        for bucket in range(5)
    ]
    selected = build_selection_cases(
        rows,
        "PARETO_ADP",
        max_kernels=2,
        cases_per_kernel_device=2,
    )
    assert {case.kernel_name for case in selected} == {"kernel-a", "kernel-b"}
    assert sum(case.kernel_name == "kernel-a" for case in selected) == 4
    assert sum(case.kernel_name == "kernel-b" for case in selected) == 4
    assert {
        (case.kernel_name, case.row["device"], case.row["bucket"])
        for case in selected
    } == {
        (kernel, device, bucket)
        for kernel in ("kernel-a", "kernel-b")
        for device in ("device-a", "device-b")
        for bucket in (0, 4)
    }


def test_selection_score_macro_averages_kernels():
    def row(kernel, accuracy):
        return {
            "kernel_name": kernel,
            "value_accuracy_over_expected": accuracy,
            "schema_compliant": True,
            "expected_key_match": True,
            "exact_design_match": accuracy == 1.0,
            "pragma_kind_counts": {
                "PIPE": {"correct": int(accuracy == 1.0), "expected": 1}
            },
        }

    summary = summarize_selection_rows(
        [row("kernel-a", 1.0), row("kernel-a", 1.0), row("kernel-b", 0.0)]
    )
    assert summary["mean_value_acc"] == pytest.approx(2.0 / 3.0)
    assert summary["selection_score"] == pytest.approx(0.5)
    assert summary["minimum_kernel_accuracy"] == 0.0
