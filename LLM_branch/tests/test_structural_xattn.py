"""Regression tests for structural-memory fusion placements."""

from __future__ import annotations

import copy
import io
from contextlib import redirect_stdout

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM

from LLM_branch.common.structural_xattn import (
    GatedCrossAttentionBlock,
    MaskedCrossAttention,
    StructuralCrossAttentionMixin,
    StructuralMemoryPreMLP,
    StructuralPostDecoderResidual,
    StructuralPostSelfAttentionResidual,
    extend_instance,
    infer_decoder_layers_attr_name,
    iter_structural_runtime_wrappers,
    make_structural_checkpoint_context_fn,
)

HIDDEN_SIZE = 32
MEMORY_SIZE = 12
PLACEHOLDER_TOKEN_IDS = (101, 102, 103)
LEGACY_PLACEMENT = "legacy_norm_wrapper"
DIRECT_PLACEMENT = "post_decoder_residual"
THESIS_PLACEMENT = "post_self_attention_residual"


def test_checkpoint_context_restores_per_forward_structural_state():
    block = GatedCrossAttentionBlock(
        dim=HIDDEN_SIZE,
        dim_memory=MEMORY_SIZE,
        dim_head=8,
        heads=2,
    )
    wrapper = StructuralPostSelfAttentionResidual(block)
    backbone = torch.nn.Sequential(wrapper)
    original_memory = torch.randn(1, 2, MEMORY_SIZE)
    original_mask = torch.ones(1, 2, dtype=torch.bool)
    original_slots = torch.ones(1, 3, dtype=torch.long)
    wrapper.structural_memory = original_memory
    wrapper.structural_memory_mask = original_mask
    wrapper.placeholder_slot_ids = original_slots
    context_fn = make_structural_checkpoint_context_fn(backbone)
    _forward_context, recompute_context = context_fn()

    later_memory = torch.randn(1, 2, MEMORY_SIZE)
    wrapper.structural_memory = later_memory
    wrapper.placeholder_slot_ids = torch.zeros_like(original_slots)
    wrapper._pending_structural_residual = torch.ones(1)
    with recompute_context:
        assert wrapper.structural_memory is original_memory
        assert wrapper.structural_memory_mask is original_mask
        assert wrapper.placeholder_slot_ids is original_slots
        assert wrapper._pending_structural_residual is None

    assert wrapper.structural_memory is later_memory
    assert torch.equal(wrapper.placeholder_slot_ids, torch.zeros_like(original_slots))
    assert wrapper._pending_structural_residual is None
    assert tuple(iter_structural_runtime_wrappers(backbone)) == (wrapper,)


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


def _attach(
    model,
    *,
    every_n_layers=8,
    attn_gate_scale=1.0,
    memory_value_scale=1.0,
    structural_fusion_placement=LEGACY_PLACEMENT,
):
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
        attn_gate_scale=attn_gate_scale,
        memory_value_scale=memory_value_scale,
        structural_fusion_placement=structural_fusion_placement,
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


class _ZeroMLP(torch.nn.Module):
    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


def _set_attention_gate(model, value):
    blocks = [
        module
        for module in model.modules()
        if isinstance(module, GatedCrossAttentionBlock)
    ]
    assert blocks
    for block in blocks:
        block.attn_gate.data.fill_(value)


@torch.no_grad()
def test_legacy_ordering_is_self_attention_then_structural_attention_then_mlp():
    model = _attach(
        _base_model(),
        structural_fusion_placement=LEGACY_PLACEMENT,
    )
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
def test_direct_ordering_is_native_decoder_then_structural_residual():
    model = _attach(
        _base_model(),
        structural_fusion_placement=DIRECT_PLACEMENT,
    )
    layer8 = model.model.layers[7]
    events = []
    hooks = [
        layer8.self_attn.register_forward_hook(
            lambda _module, _args, _output: events.append("layer8.self_attn")
        ),
        layer8.mlp.register_forward_hook(
            lambda _module, _args, _output: events.append("layer8.mlp")
        ),
        layer8.structural_post_decoder_residual.gated_cross_attn_layer.register_forward_hook(
            lambda _module, _args, _output: events.append("layer8.structural_xattn")
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
        "layer8.mlp",
        "layer8.structural_xattn",
    ]


@torch.no_grad()
def test_thesis_ordering_is_self_attention_then_structural_then_mlp():
    model = _attach(
        _base_model(),
        structural_fusion_placement=THESIS_PLACEMENT,
    )
    layer8 = model.model.layers[7]
    events = []
    hooks = [
        layer8.self_attn.register_forward_hook(
            lambda _module, _args, _output: events.append("layer8.self_attn")
        ),
        layer8.structural_post_self_attention_residual.gated_cross_attn_layer.register_forward_hook(
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
    assert (
        layer8.structural_post_self_attention_residual
        ._pending_structural_residual
        is None
    )


@pytest.mark.parametrize(
    "placement",
    [LEGACY_PLACEMENT, DIRECT_PLACEMENT, THESIS_PLACEMENT],
)
@torch.no_grad()
def test_zero_gate_preserves_stage1_logits(placement):
    stage1_model = _base_model()
    attached_model = _attach(
        copy.deepcopy(stage1_model),
        structural_fusion_placement=placement,
    )
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
def test_direct_residual_survives_when_native_mlp_is_zero():
    stage1_model = _base_model()
    stage1_model.model.layers[7].mlp = _ZeroMLP()

    legacy_model = _attach(
        copy.deepcopy(stage1_model),
        structural_fusion_placement=LEGACY_PLACEMENT,
    )
    direct_model = _attach(
        copy.deepcopy(stage1_model),
        structural_fusion_placement=DIRECT_PLACEMENT,
    )
    thesis_model = _attach(
        copy.deepcopy(stage1_model),
        structural_fusion_placement=THESIS_PLACEMENT,
    )
    _set_attention_gate(legacy_model, 0.5)
    _set_attention_gate(direct_model, 0.5)
    _set_attention_gate(thesis_model, 0.5)

    input_ids, attention_mask, directive_positions = _inputs()
    memory = torch.randn(1, 3, MEMORY_SIZE)
    memory_mask = torch.ones(1, 3, dtype=torch.bool)
    _condition(legacy_model, memory, memory_mask)
    _condition(direct_model, memory, memory_mask)
    _condition(thesis_model, memory, memory_mask)

    stage1_logits = stage1_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, directive_positions, :]
    legacy_logits = legacy_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, directive_positions, :]
    direct_logits = direct_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, directive_positions, :]
    thesis_logits = thesis_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, directive_positions, :]

    legacy_delta = (legacy_logits - stage1_logits).abs().max().item()
    direct_delta = (direct_logits - stage1_logits).abs().max().item()
    thesis_delta = (thesis_logits - stage1_logits).abs().max().item()

    assert legacy_delta < 1e-6
    assert direct_delta > 1e-6
    assert thesis_delta > 1e-6


@pytest.mark.parametrize(
    "placement",
    [LEGACY_PLACEMENT, DIRECT_PLACEMENT, THESIS_PLACEMENT],
)
@torch.no_grad()
def test_zero_memory_value_scale_preserves_stage1_logits_with_open_gate(
    placement,
):
    stage1_model = _base_model()
    attached_model = _attach(
        copy.deepcopy(stage1_model),
        memory_value_scale=0.0,
        structural_fusion_placement=placement,
    )
    _set_attention_gate(attached_model, 0.1)
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
    assert max_abs < 1e-5, f"zero-memory max logit difference was {max_abs}"


@torch.no_grad()
def test_gate_scale_is_runtime_only_and_scales_block_residual_linearly():
    torch.manual_seed(123)
    base = GatedCrossAttentionBlock(
        dim=16,
        dim_memory=8,
        dim_head=4,
        heads=2,
        attn_gate_scale=1.0,
    ).eval()
    scaled = copy.deepcopy(base)
    scaled.attn_gate_scale = 4.0
    base.attn_gate.data.fill_(0.1)
    scaled.attn_gate.data.fill_(0.1)

    x = torch.randn(1, 3, 16)
    memory = torch.randn(1, 3, 8)
    placeholder = torch.tensor([[1, 0, 0]], dtype=torch.long)
    memory_mask = torch.ones(1, 3, dtype=torch.bool)

    y1 = base(
        x,
        memory,
        placeholder_slot_ids=placeholder,
        memory_mask=memory_mask,
    )
    y4 = scaled(
        x,
        memory,
        placeholder_slot_ids=placeholder,
        memory_mask=memory_mask,
    )

    assert torch.allclose(y4 - x, 4.0 * (y1 - x), atol=1e-6, rtol=1e-5)
    assert not any("gate_scale" in key for key in base.state_dict())
    assert not any("memory_value_scale" in key for key in base.state_dict())


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


@pytest.mark.parametrize(
    "placement",
    [LEGACY_PLACEMENT, DIRECT_PLACEMENT, THESIS_PLACEMENT],
)
@torch.no_grad()
def test_real_and_shuffled_memory_change_directive_position_logits(
    placement,
):
    model = _attach(
        _base_model(),
        structural_fusion_placement=placement,
    )
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
    source = _attach(
        _base_model(),
        every_n_layers=2,
        structural_fusion_placement=DIRECT_PLACEMENT,
    )
    _set_attention_gate(source, 0.1)
    structural_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if "gated_cross_attn_layer" in key
    }
    checkpoint = tmp_path / "structural_xattn.pt"
    torch.save(structural_state, checkpoint)

    target = _attach(
        _base_model(),
        every_n_layers=2,
        structural_fusion_placement=DIRECT_PLACEMENT,
    )
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
    assert len(
        [
            module
            for module in target.modules()
            if isinstance(module, StructuralPostDecoderResidual)
        ]
    ) == 4
    for key, expected in structural_state.items():
        assert torch.equal(target.state_dict()[key], expected), key


@pytest.mark.parametrize(
    "placement",
    [DIRECT_PLACEMENT, THESIS_PLACEMENT],
)
@torch.no_grad()
def test_structural_residual_full_and_cached_logits_match(placement):
    full_model = _attach(
        _base_model(),
        structural_fusion_placement=placement,
    )
    cached_model = _attach(
        _base_model(),
        structural_fusion_placement=placement,
    )
    _set_attention_gate(full_model, 0.1)
    _set_attention_gate(cached_model, 0.1)

    memory = torch.randn(1, 3, MEMORY_SIZE)
    memory_mask = torch.ones(1, 3, dtype=torch.bool)
    _condition(full_model, memory, memory_mask)
    _condition(cached_model, memory, memory_mask)

    input_ids = torch.tensor([[7, 101, 11, 12]])
    full = full_model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
    )

    prefix_ids = input_ids[:, :3]
    prefix = cached_model(
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        use_cache=True,
    )
    step = cached_model(
        input_ids=input_ids[:, 3:],
        attention_mask=torch.ones_like(input_ids),
        past_key_values=prefix.past_key_values,
        use_cache=True,
    )

    torch.testing.assert_close(
        step.logits[:, -1, :],
        full.logits[:, -1, :],
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "placement",
    [LEGACY_PLACEMENT, DIRECT_PLACEMENT, THESIS_PLACEMENT],
)
def test_training_and_inference_architecture_contract_is_identical(
    placement,
):
    reports = []
    models = []
    for _role in ("training", "inference"):
        capture = io.StringIO()
        with redirect_stdout(capture):
            model = _attach(
                _base_model(),
                structural_fusion_placement=placement,
            )
        models.append(model)
        reports.append(capture.getvalue())

    expected_report = (
        f"[STRUCTURAL-XATTN] placement={placement} layers=[8]"
    )
    assert all(expected_report in report for report in reports)
    assert models[0].structural_xattn_layer_indices == (8,)
    assert (
        models[0].structural_xattn_layer_indices
        == models[1].structural_xattn_layer_indices
    )
    assert models[0].structural_fusion_placement == placement



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

    module.collect_diagnostics = True

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
        module.last_debug[
            "tokens_with_route"
        ]
        == 3
    )

    assert (
        module.last_debug[
            "keys_per_routed_token_mean"
        ]
        == pytest.approx(
            3.0
        )
    )

    assert (
        module.last_debug[
            "keys_per_routed_token_max"
        ]
        == 3
    )

    assert (
        module.last_debug[
            "multi_key_token_fraction"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        module.last_debug[
            "multi_key_attention_entropy_mean"
        ]
        > 0.0
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
