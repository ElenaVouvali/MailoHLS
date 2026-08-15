"""Contract, split, and prediction-metric regression tests."""

from __future__ import annotations

import copy
import json

import pytest

import LLM_branch.train.train_SFT_xattn_new as trainer
from LLM_branch.train.train_SFT_xattn_new import (
    STAGE1_COMPATIBILITY_FIELDS,
    assert_disjoint_nonempty_kernel_splits,
    build_selection_cases,
    evaluate_prediction,
    require_compatible_stage1_contract,
    summarize_selection_rows,
)


def _contract():
    contract = {
        "schema": "mailohls-training-contract-v1",
        "stage": "stage1",
    }
    contract.update({key: f"value-for-{key}" for key in STAGE1_COMPATIBILITY_FIELDS})
    return contract


def test_stage2_rejects_stage1_contract_mismatch(tmp_path):
    stage1 = _contract()
    (tmp_path / "training_contract.json").write_text(json.dumps(stage1))
    expected = copy.deepcopy(stage1)
    expected["dataset_sha256"] = "different-dataset"

    with pytest.raises(ValueError, match="dataset_sha256"):
        require_compatible_stage1_contract(tmp_path, expected)


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
