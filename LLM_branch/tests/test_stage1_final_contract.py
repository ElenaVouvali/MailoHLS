"""Focused regression tests for the locked MailoHLS Stage-1 architecture."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

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


class _BackwardOnlyAccelerator:
    scaler = None
    sync_gradients = True

    @staticmethod
    def backward(loss):
        loss.backward()


class _AccumulationHarness:
    training_step = trainer.LengthGroupedTrainer.training_step

    def __init__(self, gradient_accumulation_steps: int):
        self.args = SimpleNamespace(
            n_gpu=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            disable_structural_memory=True,
        )
        self.accelerator = _BackwardOnlyAccelerator()
        self.stage2_trainable_contract = False
        self.xattn_diagnostic_steps = 0
        self.state = SimpleNamespace(global_step=0)

    @staticmethod
    def _prepare_inputs(inputs):
        return inputs

    @staticmethod
    def compute_loss_context_manager():
        return nullcontext()

    @staticmethod
    def compute_loss(model, inputs, num_items_in_batch=None):
        del num_items_in_batch
        return (model(inputs["x"]) - inputs["target"]).square().mean()


class _FakeScaler:
    def __init__(self, scale):
        self.scale = scale

    def get_scale(self):
        return self.scale


class _AMPGuardHarness:
    _check_amp_optimizer_step = trainer.LengthGroupedTrainer._check_amp_optimizer_step

    def __init__(self, scale):
        self.state = SimpleNamespace(global_step=0)
        self.accelerator = SimpleNamespace(
            scaler=_FakeScaler(scale),
            optimizer_step_was_skipped=True,
        )
        self._consecutive_low_scale_amp_skips = 0
        self._last_amp_guard_step = -1


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


def test_public_auto_clock_menu_is_independent_of_budget():
    assert mailohls_contract.supported_clock_periods(
        "xczu7ev-ffvc1156-2-e"
    ) == (3.33, 5.0, 10.0)
    with pytest.raises(ValueError):
        mailohls_contract.supported_clock_periods("unknown-device")


def test_auto_clock_weights_use_only_auto_rows_and_normalize():
    rows = [
        {"frequency_mode": "auto", "selected_clock_period": 3.33},
        {"frequency_mode": "auto", "selected_clock_period": 5.0},
        {"frequency_mode": "specified", "selected_clock_period": 10.0},
    ]
    weights = trainer.compute_auto_clock_class_weights(
        rows, "inverse_sqrt_frequency"
    )
    assert set(weights) == {3.33, 5.0}
    assert all(0.5 <= value <= 4.0 for value in weights.values())


def test_auto_training_smoke_free_runs_clock(monkeypatch):
    def score(**kwargs):
        clock = kwargs["candidate_text"].strip()
        value = {"3.33": -1.0, "5": 2.0, "10": 0.0}[clock]
        return {"mean_logprob": value, "sum_logprob": value}
    monkeypatch.setattr(trainer, "score_rhs_candidate_suffix", score)
    chosen, scores = trainer.score_supported_clocks(
        torch.nn.Linear(1, 1), CharacterTokenizer(), [1, 2], [3.33, 5.0, 10.0]
    )
    assert chosen == 5.0
    assert [item["clock_period_ns"] for item in scores] == [5.0, 10.0, 3.33]


def test_auto_menu_does_not_depend_on_budget_feasibility():
    row = {
        "device": "xczu7ev-ffvc1156-2-e", "clock_period": 5.0,
        "frequency_mode": "auto", "available_clock_periods": [5.0],
        "avail_bram": 1, "avail_dsp": 1, "avail_ff": 1, "avail_lut": 1,
    }
    fields = mailohls_contract.target_prompt_fields(row)
    assert fields["supported_clock_periods"] == "3.33 ns, 5 ns, 10 ns"


def test_auto_validation_never_uses_gold_clock_prefix():
    row = {
        "device": "xczu7ev-ffvc1156-2-e", "clock_period": 5.0,
        "selected_clock_period": 10.0, "frequency_mode": "auto",
        "available_clock_periods": [3.33, 5.0, 10.0],
        "avail_bram": 1, "avail_dsp": 1, "avail_ff": 1, "avail_lut": 1,
    }
    fields = mailohls_contract.target_prompt_fields(row)
    assert fields["period_token"] == mailohls_contract.AUTO_PERIOD_TOKEN


def test_auto_clock_balanced_accuracy():
    rows = [
        {"frequency_mode": "auto", "reference_clock_period_ns": 3.33,
         "predicted_clock_period_ns": 3.33},
        {"frequency_mode": "auto", "reference_clock_period_ns": 3.33,
         "predicted_clock_period_ns": 5.0},
        {"frequency_mode": "auto", "reference_clock_period_ns": 5.0,
         "predicted_clock_period_ns": 5.0},
    ]
    assert trainer.summarize_auto_clock_metrics(rows)["balanced_clock_accuracy"] == pytest.approx(.75)


def test_auto_adp_regret():
    rows = [
        {"frequency_mode": "auto", "reference_clock_period_ns": 3.33,
         "predicted_clock_period_ns": 3.33, "adp_regret": .1},
        {"frequency_mode": "auto", "reference_clock_period_ns": 5.0,
         "predicted_clock_period_ns": 3.33, "adp_regret": .3},
    ]
    metrics = trainer.summarize_auto_clock_metrics(rows)
    assert metrics["mean_adp_regret"] == pytest.approx(.2)
    assert metrics["worst_adp_regret"] == pytest.approx(.3)


def test_specified_mode_is_unchanged():
    fields = mailohls_contract.target_prompt_fields({
        "device": "xczu7ev-ffvc1156-2-e", "clock_period": 5.0,
        "frequency_mode": "specified", "avail_bram": 1, "avail_dsp": 1,
        "avail_ff": 1, "avail_lut": 1,
    })
    assert fields["period_token"] == "<CLK=5NS>"
    assert fields["supported_clock_periods"] == "5 ns"


def test_single_choice_rhs_is_context_and_site_weights_are_normalized():
    source = "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?"
    target = "auto{_PIPE_L1} = 0\nauto{_UNROLL_L1} = 12"
    domains = {"kernel_a": {
        "AUTO{_PIPE_L1}": ["0"],
        "AUTO{_UNROLL_L1}": ["0", "12"],
    }}
    pack = trainer.build_deterministic_rhs_pack(
        source, target, CharacterTokenizer(), value_w=2.0,
        directive_domain_registry=domains, kernel_name="kernel-a",
    )
    assert sum(pack.token_weights) == pytest.approx(2.0)
    assert pack.labels[pack.input_ids.index(ord("0"), pack.input_ids.index(ord("=")))] == -100
    assert sum(pack.xattn_target_mask) == 3  # RHS "12\n" only.


def test_semantic_domains_allow_pipelined_unrolling_and_enforce_array_encoding():
    domains = _domains()["kernel_a"]
    assert mailohls_contract.filter_semantic_candidates(
        "auto{_UNROLL_L1}", ["0", "2"], {"AUTO{_PIPE_L1}": "1"}, domains
    ) == ["0", "2"]
    assert mailohls_contract.filter_semantic_candidates(
        "auto{_ARRAY_F_L2}", ["0", "2"], {"AUTO{_ARRAY_T_L2}": "complete"}, domains
    ) == ["0"]
    mailohls_contract.validate_directive_assignments({
        "AUTO{_PIPE_L1}": "1", "AUTO{_UNROLL_L1}": "2"
    })
    mailohls_contract.validate_directive_assignments({
        "AUTO{_ARRAY_T_L2}": "block",
        "AUTO{_ARRAY_F_L2}": "2",
        "AUTO{_ARRAY_D_L2}": "1",
    })
    with pytest.raises(ValueError, match="Invalid directive action"):
        mailohls_contract.validate_directive_assignments({
            "AUTO{_ARRAY_T_L2}": "complete",
            "AUTO{_ARRAY_F_L2}": "2",
            "AUTO{_ARRAY_D_L2}": "1",
        })


def test_free_running_decoder_scores_real_candidates_and_enforces_array_tuples(
    monkeypatch,
):
    source = (
        "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?\n"
        "L2: auto{_ARRAY_T_L2} = ? auto{_ARRAY_F_L2} = ? "
        "auto{_ARRAY_D_L2} = ?"
    )
    model = torch.nn.Linear(1, 1)

    def score_suffix(**kwargs):
        rhs = kwargs["candidate_text"].strip()
        scores = {"none": -5.0, "block": -2.0, "complete": 5.0,
                  "0": 0.0, "1": 2.0, "2": 4.0}
        score = scores[rhs]
        return {"mean_logprob": score, "sum_logprob": score}

    monkeypatch.setattr(trainer, "score_rhs_candidate_suffix", score_suffix)
    prediction, trace = trainer.constrained_decode_rhs_by_candidate_scoring(
        model=model,
        tok=CharacterTokenizer(),
        prompt_ids=[1],
        source_text=source,
        kernel_name="kernel-a",
        directive_domain_registry=_domains(),
        candidate_batch_size=1,
        return_score_trace=True,
    )

    assignments = trainer.parse_assignment_dict(prediction)
    assert assignments["AUTO{_PIPE_L1}"] == "1"
    assert assignments["AUTO{_UNROLL_L1}"] == "2"
    assert assignments["AUTO{_ARRAY_T_L2}"] == "complete"
    assert assignments["AUTO{_ARRAY_F_L2}"] == "0"
    assert assignments["AUTO{_ARRAY_D_L2}"] == "1"
    array_factor = next(row for row in trace if "ARRAY_F" in row["lhs"])
    assert array_factor["static_candidate_count"] == 2
    assert array_factor["forced_by_semantics"]


def test_related_static_fields_remain_independently_supervised():
    source = "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?"
    target = "auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 0"
    pack = trainer.build_deterministic_rhs_pack(
        source, target, CharacterTokenizer(),
        directive_domain_registry={"kernel_a": {
            "AUTO{_PIPE_L1}": ["0", "1"],
            "AUTO{_UNROLL_L1}": ["0", "2"],
        }},
        kernel_name="kernel-a",
    )
    # Both source-derived fields have >1 proposal and both receive supervision.
    assert sum(pack.xattn_target_mask) == 4
    assert sum(pack.token_weights) == pytest.approx(2.0)


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


def test_frontier_aware_budgets_keep_measured_breakpoints_and_three_witnesses(
    monkeypatch,
):
    monkeypatch.setattr(trainer.TARGET_CFG, "min_budget_frac", 0.05)
    monkeypatch.setattr(trainer.TARGET_CFG, "min_feasible_candidates", 3)
    monkeypatch.setattr(trainer.TARGET_CFG, "candidate_pool_per_objective", 6)
    rows = [{
        "kernel_name": "kernel-a",
        "device": "xczu7ev-ffvc1156-2-e",
        "clock_period": 10.0,
        "_jsonl_idx": index,
        "input": "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?",
        "target": (
            f"auto{{_PIPE_L1}} = 0\nauto{{_UNROLL_L1}} = {index}"
        ),
        "latency": float(10 - index),
        "area": float(index + 1),
        "bram_util_%": float(5 + index * 10),
        "dsp_util_%": float(8 + index * 4),
        "ff_util_%": float(10 + index * 2),
        "lut_util_%": float(12 + index * 3),
    } for index in range(6)]
    budgets = trainer.sample_shared_budgets(
        ("kernel-a", "xczu7ev-ffvc1156-2-e"), rows, 16, 123
    )
    breakpoints = trainer._frontier_breakpoint_budgets(rows)
    assert len(budgets) == 16
    assert trainer.SharedResourceBudget(1.0, 1.0, 1.0, 1.0) in budgets
    assert any(budget in breakpoints for budget in budgets)
    assert all(min(budget.as_dict().values()) >= 0.05 for budget in budgets)
    for budget in set(budgets).intersection(breakpoints):
        assert sum(trainer.design_fits_shared_budget(row, budget) for row in rows) >= 3


def test_budget_deduplication_preserves_targets_and_diverse_representatives():
    device = "xczu7ev-ffvc1156-2-e"
    capacities = mailohls_contract.DEVICE_RESOURCES[device]

    def make_row(index, fractions, target):
        row = {
            "kernel_name": "kernel-a", "device": device,
            "clock_period": 10.0, "obj_mode": "PARETO_ADP",
            "input": "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?",
            "target": f"auto{{_PIPE_L1}} = 0\nauto{{_UNROLL_L1}} = {target}",
            "resource_budget_id": f"budget-{index}",
            "_jsonl_idx": index,
        }
        for name, fraction in zip(mailohls_contract.RESOURCE_KEYS, fractions):
            row[mailohls_contract.AVAIL_FIELD_BY_RESOURCE[name]] = int(
                round(capacities[name] * fraction)
            )
        return row

    repeated = [
        make_row(0, (.1, .1, .1, .1), 0),
        make_row(1, (.12, .12, .12, .12), 0),
        make_row(2, (.15, .8, .15, .8), 0),
        make_row(3, (.8, .15, .8, .15), 0),
        make_row(4, (.9, .9, .9, .9), 0),
        make_row(5, (1.0, 1.0, 1.0, 1.0), 0),
        make_row(6, (.4, .4, .4, .4), 2),
    ]
    selected = trainer.compact_duplicate_budget_targets(repeated, 4)
    ids = {row["resource_budget_id"] for row in selected}
    assert len(selected) == 5
    assert {"budget-0", "budget-5", "budget-6"}.issubset(ids)
    assert "budget-2" in ids or "budget-3" in ids


def test_budget_deduplication_separates_auto_and_specified_prompts():
    device = "xczu7ev-ffvc1156-2-e"
    capacities = mailohls_contract.DEVICE_RESOURCES[device]
    base = {
        "kernel_name": "kernel-a",
        "device": device,
        "clock_period": 5.0,
        "selected_clock_period": 5.0,
        "obj_mode": "PARETO_ADP",
        "input": "L1: auto{_PIPE_L1} = ?",
        "target": "auto{_PIPE_L1} = 1",
        "avail_bram": capacities["BRAM_18K"],
        "avail_dsp": capacities["DSP"],
        "avail_ff": capacities["FF"],
        "avail_lut": capacities["LUT"],
    }
    rows = [
        dict(base, frequency_mode="specified", resource_budget_id="specified"),
        dict(
            base,
            frequency_mode="auto",
            available_clock_periods=[3.33, 5.0, 10.0],
            resource_budget_id="auto",
        ),
    ]

    selected = trainer.compact_duplicate_budget_targets(rows, 1)

    assert {row["frequency_mode"] for row in selected} == {"specified", "auto"}


def test_budget_transition_metric_requires_both_measured_optima_to_be_correct():
    base = {
        "kernel_name": "kernel-a", "device": "device-a",
        "selected_clock_period": 10.0, "obj_mode": "PARETO_ADP",
        "value_accuracy_over_expected": 1.0, "decision_site_accuracy": 1.0,
        "schema_compliant": True, "expected_key_match": True,
        "pragma_kind_counts": {"PIPE": {"correct": 1, "expected": 1}},
    }
    rows = [{
        **base,
        "reference_target": "target-tight",
        "prediction": "target-tight",
        "exact_design_match": True,
    }, {
        **base,
        "reference_target": "target-loose",
        "prediction": "wrong-but-different",
        "exact_design_match": False,
    }]
    summary = trainer.summarize_selection_rows(rows)
    assert summary["budget_sensitive_prediction_groups"] == 1
    assert summary["budget_counterfactual_transition_accuracy"] == 0.0
    assert summary["budget_counterfactual_exact_design_accuracy"] == 0.5


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


def test_repeated_examples_match_grad_accumulation_one_and_eight():
    example = {"x": torch.tensor([[2.0]]), "target": torch.tensor([[1.0]])}
    reference = torch.nn.Linear(1, 1, bias=False)
    accumulated = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reference.weight.fill_(0.25)
        accumulated.weight.copy_(reference.weight)

    loss_one = _AccumulationHarness(1).training_step(reference, example)
    harness_eight = _AccumulationHarness(8)
    losses_eight = [
        harness_eight.training_step(accumulated, example)
        for _ in range(8)
    ]

    assert loss_one.item() == pytest.approx(sum(x.item() for x in losses_eight))
    assert accumulated.weight.grad == pytest.approx(reference.weight.grad)


def test_amp_guard_allows_calibration_and_only_fails_after_low_scale_streak():
    high_scale = _AMPGuardHarness(scale=65536.0)
    for step in range(1, 4):
        high_scale.state.global_step = step
        high_scale._check_amp_optimizer_step()
    assert high_scale._consecutive_low_scale_amp_skips == 0

    low_scale = _AMPGuardHarness(scale=trainer.AMP_LOW_SCALE_THRESHOLD)
    for step in range(1, trainer.AMP_LOW_SCALE_MAX_CONSECUTIVE_SKIPS):
        low_scale.state.global_step = step
        low_scale._check_amp_optimizer_step()
    with pytest.raises(FloatingPointError, match="consecutive synchronized"):
        low_scale.state.global_step += 1
        low_scale._check_amp_optimizer_step()


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
    rows = [{"kernel_name": f"rodinia_algo{family}_{variant}", "area": family + 1.0}
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
    training_areas = [rows[index]["area"] for index in first["train_jsonl_idx"]]
    assert first["effective_area_floor"] == min(training_areas) / 2.0
def test_teacher_forced_local_scoring_uses_reference_prefix(monkeypatch):
    source = "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?"
    domains = {
        "kernel_a": {
            "AUTO{_PIPE_L1}": ["0", "1"],
            "AUTO{_UNROLL_L1}": ["0", "2"],
        }
    }

    def fake_score(**kwargs):
        rhs = kwargs["candidate_text"].strip()
        score = {"0": 2.0, "1": 1.0, "2": 3.0}[rhs]
        return {"mean_logprob": score, "sum_logprob": score}

    monkeypatch.setattr(
        trainer, "score_rhs_candidate_suffix", fake_score
    )
    model = torch.nn.Linear(1, 1, bias=False)

    trace = trainer.score_teacher_forced_rhs_candidates(
        model=model,
        tok=CharacterTokenizer(),
        prompt_ids=[1],
        source_text=source,
        kernel_name="kernel-a",
        reference_assignments={
            "AUTO{_PIPE_L1}": "1",
            "AUTO{_UNROLL_L1}": "0",
        },
        directive_domain_registry=domains,
        candidate_batch_size=1,
    )
    assert trace[0]["candidates"][0]["rhs"] == "0"
    # The reference PIPE value is appended to the teacher-forced prefix, but
    # it must NOT delete source-supported UNROLL proposals. The local scorer
    # therefore remains free to prefer UNROLL=2.
    assert trace[1]["forced_by_semantics"] is False
    assert trace[1]["static_candidate_count"] == 2
    assert trace[1]["candidates"][0]["rhs"] == "2"


def test_selection_summary_exposes_teacher_mrr_cascade_and_budget_accuracy():
    def make_row(budget, target, accuracy, cascade):
        return {
            "kernel_name": "kernel-a",
            "device": "device-a",
            "selected_clock_period": 5.0,
            "obj_mode": "PARETO_ADP",
            "resource_budget_id": budget,
            "reference_target": target,
            "prediction": f"prediction-{budget}",
            "value_accuracy_over_expected": accuracy,
            "decision_site_accuracy": accuracy,
            "decision_site_count": 1,
            "forced_site_count": 0,
            "schema_compliant": True,
            "expected_key_match": True,
            "exact_design_match": accuracy == 1.0,
            "pragma_kind_counts": {
                "PIPE": {
                    "correct": int(accuracy == 1.0),
                    "expected": 1,
                }
            },
            "candidate_margins": [{
                "kind": "PIPE",
                "margin": None if cascade else 0.25,
                "gold_rank": 2 if cascade else 1,
                "gold_excluded_by_prior_decision": cascade,
            }],
            "teacher_forced_candidates": [{
                "kind": "PIPE",
                "margin": 0.25 if accuracy == 1.0 else -0.25,
                "gold_rank": 1 if accuracy == 1.0 else 2,
            }],
        }

    summary = trainer.summarize_selection_rows([
        make_row("b1", "target-a", 1.0, False),
        make_row("b2", "target-b", 0.0, True),
    ])
    assert summary["candidate_cascade_excluded_count"] == 1
    assert summary["candidate_cascade_excluded_fraction"] == pytest.approx(0.5)
    assert summary["teacher_forced_top1_accuracy"] == pytest.approx(0.5)
    assert summary["teacher_forced_mrr"] == pytest.approx(0.75)
    assert summary["budget_counterfactual_groups"] == 1
    assert summary["budget_counterfactual_decision_accuracy"] == pytest.approx(0.5)


def test_scratch_stage1_refuses_stale_custom_best(tmp_path):
    best_dir = tmp_path / "best_custom_stage1"
    best_dir.mkdir()
    (best_dir / "best_selection_metrics.json").write_text(
        json.dumps({
            "step": 100,
            "checkpoint_key": [0.5, 0.4, 0.7, -1.0],
            "selection_contract_sha256": "old",
        })
    )
    with pytest.raises(RuntimeError, match="stale best"):
        trainer.StageValSelectionCallback(
            tokenizer=CharacterTokenizer(),
            selection_cases=[],
            directive_domain_registry={},
            output_dir=str(tmp_path),
            training_contract={"schema": "test"},
        )


def test_effective_domain_registry_hash_matches_saved_bytes(tmp_path):
    path, digest = trainer.write_effective_directive_domain_registry(
        str(tmp_path),
        {
            "kernel_a": {
                "AUTO{_PIPE_L1}": ["0", "1"],
            }
        },
        "source_policy",
    )
    assert trainer._file_sha256(trainer.Path(path)) == digest
