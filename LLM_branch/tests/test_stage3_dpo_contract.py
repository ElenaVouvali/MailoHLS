"""Stage-3 preference/lineage regression tests."""

from __future__ import annotations

import json

import pytest
import torch

from LLM_branch.train.train_DPO_harp_xattn import (
    GoalPreferencePairBuilder,
    load_stage2_contract,
)


class _PairMod:
    @staticmethod
    def build_partial_deterministic_target_text(
        source,
        target,
        min_supervised_sites=1,
    ):
        return target, {
            "coverage": 1.0,
            "n_supervised": 2,
        }

    @staticmethod
    def extract_ordered_lhs_plan(source):
        return [
            ("L1", "auto{_PIPE_L1}"),
            ("L1", "auto{_UNROLL_L1}"),
        ]

    @staticmethod
    def build_prompt(source, objective, row=None):
        return "|".join((
            source,
            objective,
            str(row["device"]),
            str(row["clock_period"]),
            str(row["avail_dsp"]),
        ))


def _row(
    *,
    device,
    target,
    score,
    rank,
    latency,
    area,
):
    return {
        "kernel_name": "kernel-a",
        "_family": "family-a",
        "input": "L1: for (;;) {}",
        "target": target,
        "device": device,
        "clock_period": 5.0,
        "selected_clock_period": 5.0,
        "frequency_mode": "specified",
        "avail_bram": 10,
        "avail_dsp": 20,
        "avail_ff": 30,
        "avail_lut": 40,
        "latency": latency,
        "area": area,
        "_score": score,
        "_rank_within_kernel": rank,
    }


def _target(pipe, unroll):
    return (
        f"auto{{_PIPE_L1}} = {pipe}\n"
        f"auto{{_UNROLL_L1}} = {unroll}"
    )


def test_preferences_never_cross_target_conditioning_contexts():
    rows = []
    for device, scale in (("device-a", 1.0), ("device-b", 2.0)):
        rows.extend([
            _row(
                device=device,
                target=_target(1, 8),
                score=0.0,
                rank=1,
                latency=10.0 * scale,
                area=10.0 * scale,
            ),
            _row(
                device=device,
                target=_target(0, 2),
                score=0.1,
                rank=2,
                latency=12.0 * scale,
                area=12.0 * scale,
            ),
        ])

    builder = GoalPreferencePairBuilder(
        _PairMod,
        "PARETO_ADP",
        chosen_top_k=1,
        hard_negatives_per_chosen=1,
        medium_negatives_per_chosen=0,
    )
    pairs = builder.build(rows)
    assert len(pairs) == 2
    assert {pair["platform_row"]["device"] for pair in pairs} == {
        "device-a", "device-b"
    }
    assert len({pair["context_id"] for pair in pairs}) == 2


def test_adp_preferences_require_actual_product_improvement():
    rows = [
        _row(
            device="device-a",
            target=_target(1, 8),
            score=0.0,
            rank=1,
            latency=9.0,
            area=15.0,
        ),
        _row(
            device="device-a",
            target=_target(0, 2),
            score=0.1,
            rank=2,
            latency=10.0,
            area=10.0,
        ),
    ]
    builder = GoalPreferencePairBuilder(
        _PairMod,
        "PARETO_ADP",
        chosen_top_k=1,
    )
    assert builder.build(rows) == []


def test_stage2_parent_artifact_must_be_self_contained(tmp_path):
    structural = {
        "mem_dim": 64,
        "max_slots": 64,
        "every_n_layers": 8,
        "xattn_heads": 4,
        "xattn_dim_head": 64,
        "xattn_ff_mult": 1,
        "xattn_enable_ff": False,
        "xattn_placement": "post_decoder_residual",
        "selection_eval_gate_scale": 1.0,
        "selection_eval_memory_value_scale": 1.0,
        "structural_routing": "exact_slot",
        "memory_manifest_sha256": "digest",
        "selected_xattn_layers_1based": [8, 16, 24, 32],
    }
    contract = {
        "schema": "mailohls-training-contract-v1",
        "stage": "stage2",
        "structural": structural,
    }
    (tmp_path / "training_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="structural_xattn"):
        load_stage2_contract(str(tmp_path))

    torch.save({}, tmp_path / "structural_xattn.pt")
    assert load_stage2_contract(str(tmp_path))["stage"] == "stage2"
