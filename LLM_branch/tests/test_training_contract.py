"""Contract, split, and prediction-metric regression tests."""

from __future__ import annotations

import copy
import json

import pytest

from LLM_branch.train.train_SFT_xattn_new import (
    STAGE1_COMPATIBILITY_FIELDS,
    assert_disjoint_nonempty_kernel_splits,
    evaluate_prediction,
    require_compatible_stage1_contract,
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

