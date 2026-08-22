"""Focused regression tests for the locked MailoHLS Stage-1 architecture."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from LLM_branch.common import mailohls_contract
from LLM_branch.train import train_SFT_xattn_new as trainer
from Preprocessing.build_family_split import build_family_split
from Preprocessing.create_jsonl import validate_preprocessing_manifest
from Preprocessing.data_preprocess import assign_target_local_weights


class CharacterTokenizer:
    eos_token = "<EOS>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def _domains():
    return {
        "kernel_a": {
            "AUTO{_PIPE_L1}": ["0", "1"],
            "AUTO{_UNROLL_L1}": ["0", "2"],
            "AUTO{_ARRAY_T_L2}": ["none", "complete", "block"],
            "AUTO{_ARRAY_F_L2}": ["0", "2"],
            "AUTO{_ARRAY_D_L2}": ["0", "1"],
        }
    }


def test_specified_clock_is_context_but_automatic_clock_is_supervised():
    tok = CharacterTokenizer()
    specified = trainer.build_clock_pack({"clock_period": 5.0}, tok)
    automatic = trainer.build_clock_pack(
        {"clock_period": 5.0, "frequency_mode": "auto"}, tok
    )
    assert all(label == -100 for label in specified.labels)
    assert any(label != -100 for label in automatic.labels)
    assert mailohls_contract.selected_clock_response_prefix(
        {"clock_period": 5.0}
    ) == "<CLOCK>\nselected_clock_period_ns = 5\n"
    assert mailohls_contract.selected_clock_response_token_ids(
        {"clock_period": 5.0}, tok
    ) == specified.input_ids
    boundary_tokenizer = lambda text, add_special_tokens=False: {
        "input_ids": [len(text)]
    }
    prefix = mailohls_contract.selected_clock_response_prefix({"clock_period": 5.0})
    segmented = mailohls_contract.selected_clock_response_token_ids(
        {"clock_period": 5.0}, boundary_tokenizer
    )
    assert segmented == trainer.build_clock_pack(
        {"clock_period": 5.0}, boundary_tokenizer
    ).input_ids
    assert segmented != boundary_tokenizer(prefix)["input_ids"]


def test_single_choice_rhs_is_context_and_site_weights_are_normalized():
    source = "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?"
    target = "auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 12"
    domains = {"kernel_a": {
        "AUTO{_PIPE_L1}": ["1"],
        "AUTO{_UNROLL_L1}": ["0", "12"],
    }}
    pack = trainer.build_deterministic_rhs_pack(
        source, target, CharacterTokenizer(), value_w=2.0,
        directive_domain_registry=domains, kernel_name="kernel-a",
    )
    assert sum(pack.token_weights) == pytest.approx(2.0)
    assert pack.labels[pack.input_ids.index(ord("1"), pack.input_ids.index(ord("=")))] == -100
    assert sum(pack.xattn_target_mask) == 3  # RHS "12\n" only.


def test_semantic_domains_enforce_exclusive_loops_and_valid_arrays():
    domains = _domains()["kernel_a"]
    assert mailohls_contract.filter_semantic_candidates(
        "auto{_UNROLL_L1}", ["0", "2"], {"AUTO{_PIPE_L1}": "1"}, domains
    ) == ["0"]
    assert mailohls_contract.filter_semantic_candidates(
        "auto{_ARRAY_F_L2}", ["0", "2"], {"AUTO{_ARRAY_T_L2}": "complete"}, domains
    ) == ["0"]
    with pytest.raises(ValueError, match="Invalid directive action"):
        mailohls_contract.validate_directive_assignments({
            "AUTO{_PIPE_L1}": "1", "AUTO{_UNROLL_L1}": "2"
        })
    mailohls_contract.validate_directive_assignments({
        "AUTO{_ARRAY_T_L2}": "block",
        "AUTO{_ARRAY_F_L2}": "2",
        "AUTO{_ARRAY_D_L2}": "1",
    })


def test_per_clock_budget_compaction_never_drops_a_measured_clock(monkeypatch):
    device = "xczu7ev-ffvc1156-2-e"
    rows = []
    for clock in (3.33, 5.0, 10.0):
        for index in range(3):
            rows.append({
                "kernel_name": "kernel-a", "device": device,
                "clock_period": clock, "_jsonl_idx": len(rows),
                "input": "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?",
                "target": f"auto{{_PIPE_L1}} = 0\nauto{{_UNROLL_L1}} = {index}",
                "latency": index + 1.0, "area": index + 1.0,
                "bram_util_%": 0.0, "dsp_util_%": 0.0,
                "ff_util_%": 1.0, "lut_util_%": 1.0,
            })
    monkeypatch.setattr(trainer.TARGET_CFG, "candidate_pool_per_objective", 3)
    monkeypatch.setattr(
        trainer, "sample_shared_budgets",
        lambda *args: [trainer.SharedResourceBudget(1.0, 1.0, 1.0, 1.0)],
    )
    selected = trainer.augment_rows_with_random_resource_budgets(
        rows, num_budgets_per_case=1, seed=123, min_feasible_candidates=3
    )
    assert {row["clock_period"] for row in selected} == {3.33, 5.0, 10.0}


def test_disabled_automatic_clock_does_not_rank_unused_cases(monkeypatch):
    monkeypatch.setattr(trainer.TARGET_CFG, "auto_frequency_fraction", 0.0)
    modes = []

    def fake_rank(*args, **kwargs):
        modes.append(args[7])
        return [], {}

    monkeypatch.setattr(trainer, "_rank_and_select_case", fake_rank)
    rows = [{
        "kernel_name": "kernel-a", "device": "xczu7ev-ffvc1156-2-e",
        "clock_period": clock,
    } for clock in (3.33, 5.0)]
    trainer.select_goal_rows(rows, "PARETO_ADP", 1, 0.25, 0.12)
    assert modes == ["specified", "specified"]


def test_decision_sites_drive_checkpoint_selection():
    rows = [{
        "kernel_name": "kernel-a", "value_accuracy_over_expected": 0.9,
        "decision_site_accuracy": 0.5, "decision_site_count": 2,
        "forced_site_count": 8, "schema_compliant": True,
        "expected_key_match": True, "exact_design_match": False,
        "pragma_kind_counts": {"PIPE": {"correct": 1, "expected": 2}},
    }, {
        "kernel_name": "kernel-b", "value_accuracy_over_expected": 0.8,
        "decision_site_accuracy": 0.25, "decision_site_count": 4,
        "forced_site_count": 1, "schema_compliant": True,
        "expected_key_match": True, "exact_design_match": False,
        "pragma_kind_counts": {"PIPE": {"correct": 1, "expected": 4}},
    }]
    summary = trainer.summarize_selection_rows(rows)
    assert summary["selection_score"] == pytest.approx(0.375)
    assert summary["minimum_kernel_decision_accuracy"] == pytest.approx(0.25)
    assert summary["forced_site_count"] == 9


def test_duplicate_budget_targets_do_not_repeat_validation_work(monkeypatch):
    monkeypatch.setattr(
        trainer, "target_bucket_key",
        lambda row: (row["kernel_name"], row["clock_period"], row["bucket"]),
    )
    monkeypatch.setattr(
        trainer, "shared_budget_fraction", lambda row: float(row["bucket"])
    )
    monkeypatch.setattr(
        trainer, "build_partial_deterministic_target_text",
        lambda source, target, minimum: (target, {"coverage": 1.0}),
    )
    rows = [{
        "kernel_name": "kernel-a", "device": "device-a",
        "clock_period": clock, "bucket": budget,
        "obj_mode": "PARETO_ADP", "frequency_mode": "specified",
        "input": "source", "target": "identical-target",
        "_rank_within_kernel": 0, "_score": budget,
    } for clock in (3.33, 5.0) for budget in range(4)]
    selected = trainer.build_selection_cases(
        rows, "PARETO_ADP", cases_per_kernel_device=2
    )
    assert len(selected) == 2
    assert {case.row["clock_period"] for case in selected} == {3.33, 5.0}


def test_stage1_contract_rejects_unexpected_trainable_parameters():
    model = torch.nn.Module()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1)))
    model.register_parameter("trainable_tokens_delta", torch.nn.Parameter(torch.ones(1)))
    assert trainer.assert_stage1_trainable_contract(model)["schema"] == (
        "mailohls-stage1-trainables-v1"
    )
    model.register_parameter("base_weight", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(RuntimeError, match="Unexpected Stage-1"):
        trainer.assert_stage1_trainable_contract(model)


def test_reported_zero_utilization_is_preserved_in_area():
    frame = pd.DataFrame([{
        "Device": "device-a", "Clock_Period_nsec": 5.0,
        "Latency_msec": 1.0, "BRAM_Utilization_percentage": 0.0,
        "DSP_Utilization_percentage": 0.0,
        "FF_Utilization_percentage": 4.0,
        "LUT_Utilization_percentage": 8.0,
    }])
    measured = assign_target_local_weights(frame, minimum_weight=0.1, gamma=2.0)
    assert measured.loc[0, "BRAM_Utilization_percentage"] == 0.0
    assert measured.loc[0, "Area"] == pytest.approx(3.0)


def test_stage1_accepts_zero_area_qor(monkeypatch):
    device = next(iter(trainer.DEVICE_RESOURCES))
    monkeypatch.setattr(trainer.TARGET_CFG, "device_mode", "known")
    row = {
        "kernel_name": "kernel-a",
        "input": "source",
        "target": "auto{_PIPE_L1} = 0",
        "device": device,
        "clock_period": 5.0,
        "latency": 1.0,
        "area": 0.0,
    }
    for resource in trainer.RESOURCE_KEYS:
        row[trainer.UTIL_FIELD_BY_RESOURCE[resource]] = 0.0

    assert trainer.filter_rows_for_device_mode([row]) == [row]


def test_outdated_compact_dataset_is_rejected_before_training(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    dataset.with_suffix(".sources.json").write_text("{}\n")
    dataset.with_suffix(".manifest.json").write_text(json.dumps({
        "schema": "mailohls-sft-jsonl-manifest-v2-compact-source"
    }))
    with pytest.raises(ValueError, match="outdated resource policy"):
        trainer.load_rows(str(dataset))


def test_outdated_preprocessed_tables_cannot_be_repackaged(tmp_path):
    (tmp_path / "preprocessing_manifest.json").write_text(json.dumps({
        "schema": "mailohls-qor-preprocessing-v1", "mode": "llm",
    }))
    with pytest.raises(ValueError, match="outdated resource policy"):
        validate_preprocessing_manifest(tmp_path)


def test_family_split_is_disjoint_deterministic_and_provenance_complete(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rows = [{"kernel_name": f"rodinia_algo{family}_{variant}"}
            for family in range(12) for variant in range(2)]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = build_family_split(
        dataset, seed=123, val_family_count=3, test_family_count=3,
        val_kernel_target=6, test_kernel_target=6,
    )
    second = build_family_split(
        dataset, seed=123, val_family_count=3, test_family_count=3,
        val_kernel_target=6, test_kernel_target=6,
    )
    assert first == second
    assert set(first["train_families"]).isdisjoint(first["val_families"])
    assert set(first["train_families"]).isdisjoint(first["test_families"])
    assert len(first["val_kernels"]) == len(first["test_kernels"]) == 6
    assert len(first["dataset_sha256"]) == 64
