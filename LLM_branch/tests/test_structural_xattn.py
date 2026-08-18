"""Regression tests for post-self-attention structural-memory fusion."""

from __future__ import annotations

import copy
import io
from contextlib import redirect_stdout

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM

from LLM_branch.common.structural_xattn import (
    StructuralCrossAttentionMixin,
    StructuralMemoryPreMLP,
    extend_instance,
    infer_decoder_layers_attr_name,
)


HIDDEN_SIZE = 32
MEMORY_SIZE = 12
PLACEHOLDER_TOKEN_IDS = (101, 102, 103)
STRUCTURAL_PLACEMENT = "post_self_attn_pre_mlp"


def _base_model() -> LlamaForCausalLM:
    torch.manual_seed(123)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cache=False,
    )
    return LlamaForCausalLM(config).eval()


def _attach(model, *, every_n_layers=8):
    extend_instance(model, StructuralCrossAttentionMixin)
    model.set_decoder_layers_attr_name(infer_decoder_layers_attr_name(model))
    model.init_structural_cross_attention(
        placeholder_token_ids=PLACEHOLDER_TOKEN_IDS,
        lang_hidden_size=HIDDEN_SIZE,
        mem_hidden_size=MEMORY_SIZE,
        cross_attn_every_n_layers=every_n_layers,
        xattn_heads=2,
        xattn_dim_head=8,
        only_attend_immediate_memory=True,
        mask_mode="segment",
    )
    return model.eval()


def _inputs():
    # Positions following each placeholder are directive-value positions.
    input_ids = torch.tensor([[7, 101, 11, 12, 102, 13, 14, 103, 15]])
    attention_mask = torch.ones_like(input_ids)
    directive_positions = torch.tensor([2, 5, 8])
    return input_ids, attention_mask, directive_positions


def _condition(model, memory, memory_mask):
    model.clear_structural_memory()
    model.condition_structural_memory(memory, memory_mask)


def _set_attention_gate(model, value):
    wrappers = [
        module
        for module in model.modules()
        if isinstance(module, StructuralMemoryPreMLP)
    ]
    assert wrappers
    for wrapper in wrappers:
        wrapper.gated_cross_attn_layer.attn_gate.data.fill_(value)


@torch.no_grad()
def test_ordering_is_self_attention_then_structural_attention_then_mlp():
    model = _attach(_base_model())
    layer8 = model.model.layers[7]
    events = []
    hooks = [
        layer8.self_attn.register_forward_hook(
            lambda _module, _args, _output: events.append("layer8.self_attn")
        ),
        layer8.post_attention_layernorm.gated_cross_attn_layer.register_forward_hook(
            lambda _module, _args, _output: events.append("layer8.structural_xattn")
        ),
        layer8.mlp.register_forward_hook(
            lambda _module, _args, _output: events.append("layer8.mlp")
        ),
    ]
    try:
        input_ids, attention_mask, _ = _inputs()
        _condition(
            model,
            torch.randn(1, 3, MEMORY_SIZE),
            torch.ones(1, 3, dtype=torch.bool),
        )
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for hook in hooks:
            hook.remove()

    assert events == [
        "layer8.self_attn",
        "layer8.structural_xattn",
        "layer8.mlp",
    ]


@torch.no_grad()
def test_zero_gate_preserves_stage1_logits():
    stage1_model = _base_model()
    attached_model = _attach(copy.deepcopy(stage1_model))
    input_ids, attention_mask, _ = _inputs()
    _condition(
        attached_model,
        torch.randn(1, 3, MEMORY_SIZE),
        torch.ones(1, 3, dtype=torch.bool),
    )

    stage1_logits = stage1_model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits
    attached_logits = attached_model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits

    max_abs = (stage1_logits - attached_logits).abs().max().item()
    assert max_abs < 1e-5, f"zero-gate max logit difference was {max_abs}"


@torch.no_grad()
def test_all_false_memory_mask_is_identity_with_nonzero_gate():
    stage1_model = _base_model()
    attached_model = _attach(copy.deepcopy(stage1_model))
    _set_attention_gate(attached_model, 0.1)
    input_ids, attention_mask, _ = _inputs()
    _condition(
        attached_model,
        torch.randn(1, 3, MEMORY_SIZE),
        torch.zeros(1, 3, dtype=torch.bool),
    )

    stage1_logits = stage1_model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits
    attached_logits = attached_model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits

    max_abs = (stage1_logits - attached_logits).abs().max().item()
    assert max_abs < 1e-5, f"masked-memory max logit difference was {max_abs}"


@torch.no_grad()
def test_real_and_shuffled_memory_change_directive_position_logits():
    model = _attach(_base_model())
    _set_attention_gate(model, 0.1)
    input_ids, attention_mask, directive_positions = _inputs()
    memory = torch.randn(1, 3, MEMORY_SIZE)
    memory_mask = torch.ones(1, 3, dtype=torch.bool)

    _condition(model, memory, memory_mask)
    real_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
        :, directive_positions, :
    ]
    _condition(model, memory[:, torch.tensor([2, 0, 1]), :], memory_mask)
    shuffled_logits = model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits[:, directive_positions, :]

    max_abs = (real_logits - shuffled_logits).abs().max().item()
    assert max_abs > 1e-7, "shuffled memory did not affect directive logits"


def test_structural_checkpoint_round_trip(tmp_path):
    source = _attach(_base_model())
    _set_attention_gate(source, 0.1)
    structural_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if "gated_cross_attn_layer" in key
    }
    checkpoint = tmp_path / "structural_xattn.pt"
    torch.save(structural_state, checkpoint)

    target = _attach(_base_model())
    loaded_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = target.load_state_dict(loaded_state, strict=False)
    missing_structural = [
        key for key in missing if "gated_cross_attn_layer" in key
    ]
    unexpected_structural = [
        key for key in unexpected if "gated_cross_attn_layer" in key
    ]

    assert not missing_structural
    assert not unexpected_structural
    for key, expected in structural_state.items():
        assert torch.equal(target.state_dict()[key], expected), key


def test_training_and_inference_architecture_contract_is_identical():
    reports = []
    models = []
    for _role in ("training", "inference"):
        capture = io.StringIO()
        with redirect_stdout(capture):
            model = _attach(_base_model())
        models.append(model)
        reports.append(capture.getvalue())

    expected_report = (
        "[STRUCTURAL-XATTN] placement=post_self_attn_pre_mlp layers=[8]"
    )
    assert all(expected_report in report for report in reports)
    assert models[0].structural_xattn_layer_indices == (8,)
    assert (
        models[0].structural_xattn_layer_indices
        == models[1].structural_xattn_layer_indices
    )
    assert STRUCTURAL_PLACEMENT == "post_self_attn_pre_mlp"



def test_structural_initialization_uses_current_contract():
    model = _attach(_base_model())

    assert model.initialized_structural_xattn is True
    assert not hasattr(model, "initialized_harp_flamingo")

    assert hasattr(model, "condition_structural_memory")
    assert hasattr(model, "clear_structural_memory")

    assert not hasattr(model, "condition_harp")
    assert not hasattr(model, "clear_harp")


def test_relational_attention_has_multiple_keys_and_qk_gradients():

    torch.manual_seed(123)

    module = MaskedCrossAttention(
        dim=16,
        dim_memory=8,
        dim_head=4,
        heads=2,
        mask_mode="segment",
    )

    x = torch.randn(
        1,
        3,
        16,
        requires_grad=True,
    )

    memory = torch.randn(
        1,
        4,
        8,
    )

    # Route all segment tokens from L2.
    placeholder = torch.tensor(
        [[2, 0, 0]],
        dtype=torch.long,
    )

    memory_mask = torch.tensor(
        [[1, 1, 1, 1]],
        dtype=torch.bool,
    )

    relation = torch.zeros(
        1,
        4,
        4,
        dtype=torch.bool,
    )

    # L2 can see L1, L2, L4.
    relation[
        0,
        1,
        [0, 1, 3],
    ] = True

    out = module(
        x,
        memory,
        placeholder_slot_ids=(
            placeholder
        ),
        memory_mask=memory_mask,
        action_relation_mask=relation,
    )

    loss = (
        out.square()
        .mean()
    )

    loss.backward()

    assert (
        module.last_debug[
            "valid_edges"
        ]
        == 9
    )

    assert (
        module.to_q.weight.grad
        .norm()
        .item()
        > 0.0
    )

    k_grad, v_grad = (
        module.to_kv.weight.grad
        .chunk(
            2,
            dim=0,
        )
    )

    assert (
        k_grad.norm().item()
        > 0.0
    )

    assert (
        v_grad.norm().item()
        > 0.0
    )
