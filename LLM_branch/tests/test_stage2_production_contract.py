"""Focused production-contract regression tests for structural Stages 2 and 3."""

from __future__ import annotations

import types

import pytest
import torch

from LLM_branch.common import frozen_stage1
from LLM_branch.common.structural_memory import load_memory_bank, memory_bank_summary
from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
import LLM_branch.train.train_SFT_xattn_new as sft


class _Stage1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
        self.trainable_tokens_delta = torch.nn.Parameter(torch.ones(2, 2))
        self.lora_dropout = torch.nn.Dropout(0.5)


class _ProductionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = _Stage1()
        self.gated_cross_attn_layer = GatedCrossAttentionBlock(
            dim=8, dim_memory=4, heads=2, dim_head=4, mask_mode="segment"
        )
        self.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if "gated_cross_attn_layer.attn." in name or name.endswith(
                "gated_cross_attn_layer.attn_gate"
            ):
                parameter.requires_grad_(True)


def _memory_pack():
    return {
        "node_embs": torch.ones(2, 2),
        "node_embs_mask": torch.tensor([True, True]),
        "labels": torch.tensor([1, 2]),
        "slot_ids": torch.tensor([1, 2]),
        "gnn_dim": 2,
        "max_slots": 2,
    }


def test_lora_and_special_tokens_remain_frozen_and_hash_is_stable():
    model = _ProductionModel()
    contract = sft.assert_stage2_trainable_contract(model)
    before = frozen_stage1.frozen_stage1_hashes(model)
    frozen_stage1.assert_frozen_stage1_unchanged(model, before)
    assert contract["allowed_groups"] == [
        "structural_cross_attention", "structural_attention_gates"
    ]
    assert not model.stage1.lora_A.requires_grad
    assert not model.stage1.trainable_tokens_delta.requires_grad
    model.stage1.lora_A.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Stage-1 adapter tensors"):
        sft.assert_stage2_trainable_contract(model)


def test_frozen_lora_dropout_is_disabled_after_model_train():
    model = _ProductionModel()
    model.train()
    assert model.stage1.lora_dropout.training
    assert frozen_stage1.disable_frozen_lora_dropout(model) == 1
    assert not model.stage1.lora_dropout.training
    model.train()
    frozen_stage1.disable_frozen_lora_dropout(model)
    assert not model.stage1.lora_dropout.training


def test_displayed_gradient_is_unscaled_without_mutating_grad():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameter.grad = torch.full((2, 2), 8.0)
    original = parameter.grad.clone()
    stats = sft._projection_grad_stats(parameter, parameter.grad, amp_scale=4.0)
    assert stats["grad_rms"] == pytest.approx(2.0)
    assert torch.equal(parameter.grad, original)


def test_zero_gate_preserves_hidden_states_and_rhs_mask_limits_changes():
    model = _ProductionModel()
    block = model.gated_cross_attn_layer
    hidden = torch.randn(1, 4, 8)
    memory = torch.randn(1, 2, 4)
    slots = torch.tensor([[1, 1, 2, 2]])
    apply = torch.tensor([[False, True, False, True]])
    relation = torch.ones(1, 2, 2, dtype=torch.bool)
    output = block(
        hidden, memory, placeholder_slot_ids=slots,
        memory_mask=torch.ones(1, 2, dtype=torch.bool),
        action_relation_mask=relation, xattn_apply_mask=apply,
    )
    assert torch.equal(output, hidden)
    block.attn_gate.data.fill_(1.0)
    output = block(
        hidden, memory, placeholder_slot_ids=slots,
        memory_mask=torch.ones(1, 2, dtype=torch.bool),
        action_relation_mask=relation, xattn_apply_mask=apply,
    )
    assert torch.equal(output[:, ~apply[0]], hidden[:, ~apply[0]])
    assert not torch.equal(output[:, apply[0]], hidden[:, apply[0]])


def test_unique_memory_records_are_distinct_from_lookup_aliases(tmp_path):
    torch.save(_memory_pack(), tmp_path / "kernel-a.memory.pt")
    bank, _ = load_memory_bank(str(tmp_path))
    summary = memory_bank_summary(str(tmp_path), bank, ["kernel_a"])
    assert summary["memory_files"] == 1
    assert summary["unique_memory_records"] == 1
    assert summary["lookup_aliases"] == 2
    assert summary["required_split_kernels_covered"] == 1


def test_memory_normalization_collisions_are_rejected(tmp_path):
    torch.save(_memory_pack(), tmp_path / "kernel-a.memory.pt")
    torch.save(_memory_pack(), tmp_path / "kernel_a.memory.pt")
    with pytest.raises(ValueError, match="normalization collision"):
        load_memory_bank(str(tmp_path))


def test_no_hard_negative_bank_is_built_when_candidate_loss_is_disabled(monkeypatch):
    row = {"input": "source", "target": "target", "clock_period": 10.0}
    monkeypatch.setattr(sft, "rank_goal_candidates", lambda *args: [{"row": row}])
    monkeypatch.setattr(sft, "canonical_completion_key", lambda *args: "completion")
    monkeypatch.setattr(sft, "goal_sort_key", lambda *args: (1.0,))
    monkeypatch.setattr(sft, "score_gap_weight", lambda *args: 1.0)
    monkeypatch.setattr(
        sft,
        "build_local_hard_negative_bank",
        lambda *args, **kwargs: pytest.fail("hard-negative bank was unexpectedly built"),
    )
    selected, _ = sft._rank_and_select_case(
        [row], "PARETO_LATENCY", 1, 0.0, 0.0, 0.6, 1.0, "specified",
        build_training_hard_negatives=False,
    )
    assert "_local_hard_negatives" not in selected[0]
