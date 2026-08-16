"""Shared structural-memory cross-attention for decoder-only language models."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum


def exists(value):
    return value is not None


def extend_instance(obj, mixin):
    """Apply ``mixin`` to an existing model instance."""
    base_cls = obj.__class__
    obj.__class__ = type(base_cls.__name__, (mixin, base_cls), {})


def getattr_recursive(obj, attribute):
    if not attribute:
        return obj
    head, separator, tail = attribute.partition(".")
    value = getattr(obj, head)
    return getattr_recursive(value, tail) if separator else value


def infer_decoder_layers_attr_name(model) -> str:
    candidates = (
        "base_model.model.model.layers",
        "base_model.model.decoder.layers",
        "base_model.model.transformer.h",
        "base_model.model.gpt_neox.layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
    )
    for attribute in candidates:
        try:
            layers = getattr_recursive(model, attribute)
        except (AttributeError, TypeError):
            continue
        if isinstance(layers, (nn.ModuleList, list)) and layers:
            return attribute
    raise ValueError(
        "Could not infer decoder layer path. Add the correct recursive path "
        "for this backbone."
    )


def FeedForward(dim: int, mult: int = 4) -> nn.Module:
    inner = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner, bias=False),
        nn.GELU(),
        nn.Linear(inner, dim, bias=False),
    )


def forward_fill_slot_ids(slot_ids):
    batch, sequence_length = slot_ids.shape
    positions = torch.arange(sequence_length, device=slot_ids.device).unsqueeze(0)
    positions = positions.expand(batch, sequence_length)
    seen_positions = torch.where(
        slot_ids.ne(0), positions, torch.full_like(positions, -1)
    )
    last_positions = torch.cummax(seen_positions, dim=1).values
    active = slot_ids.gather(1, last_positions.clamp(min=0))
    return torch.where(last_positions.ge(0), active, torch.zeros_like(active))


def last_seen_slot_id(slot_ids):
    batch, sequence_length = slot_ids.shape
    positions = torch.arange(sequence_length, device=slot_ids.device).unsqueeze(0)
    positions = positions.expand(batch, sequence_length)
    seen_positions = torch.where(
        slot_ids.ne(0), positions, torch.full_like(positions, -1)
    )
    last_positions = seen_positions.max(dim=1).values
    last_slot = slot_ids.gather(1, last_positions.clamp(min=0).unsqueeze(1))
    return torch.where(
        last_positions.ge(0).unsqueeze(1), last_slot, torch.zeros_like(last_slot)
    )


def build_placeholder_slot_ids(input_ids, placeholder_token_ids, routing_start_idx=None):
    """Map target placeholder tokens to one-based structural-memory slots."""
    slot_ids = torch.zeros_like(input_ids, dtype=torch.long)
    for slot_idx, token_id in enumerate(placeholder_token_ids, start=1):
        slot_ids[input_ids == token_id] = slot_idx

    if routing_start_idx is not None:
        if routing_start_idx.ndim == 0:
            routing_start_idx = routing_start_idx.unsqueeze(0)
        batch, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch, sequence_length)
        slot_ids = torch.where(
            positions >= routing_start_idx.unsqueeze(1),
            slot_ids,
            torch.zeros_like(slot_ids),
        )
    return slot_ids


class MaskedCrossAttention(nn.Module):
    """Attention from language hidden states to routed structural slots."""

    def __init__(
        self,
        *,
        dim,
        dim_memory,
        dim_head=64,
        heads=8,
        only_attend_immediate_memory=True,
        mask_mode="segment",
    ):
        super().__init__()
        if mask_mode not in {"segment", "token"}:
            raise ValueError(f"Unsupported structural routing mode: {mask_mode}")
        self.scale = dim_head**-0.5
        self.heads = heads
        self.mask_mode = mask_mode
        self.only_attend_immediate_memory = only_attend_immediate_memory
        self.last_debug = {}
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_memory, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x,
        memory,
        placeholder_slot_ids=None,
        memory_mask=None,
        use_cached_memory=False,
    ):
        batch, text_length, _ = x.shape
        _, slot_count, _ = memory.shape
        heads = self.heads
        if not use_cached_memory and placeholder_slot_ids is None:
            raise ValueError(
                "placeholder_slot_ids is required unless use_cached_memory=True"
            )

        x = self.norm(x)
        memory = memory.to(dtype=x.dtype)
        q = self.to_q(x)
        k, v = self.to_kv(memory).chunk(2, dim=-1)
        q, k, v = rearrange_many(
            (q, k, v), "b n (h d) -> b h n d", h=heads
        )
        similarity = einsum("b h i d, b h j d -> b h i j", q * self.scale, k)
        memory_slots = torch.arange(
            1, slot_count + 1, device=x.device, dtype=torch.long
        )

        if placeholder_slot_ids is not None:
            if use_cached_memory:
                active_slot_ids = last_seen_slot_id(placeholder_slot_ids).expand(
                    batch, text_length
                )
            elif self.mask_mode == "segment":
                active_slot_ids = forward_fill_slot_ids(placeholder_slot_ids)
            else:
                active_slot_ids = placeholder_slot_ids

            text_to_memory_mask = torch.eq(
                rearrange(active_slot_ids, "b t -> b 1 t 1"),
                rearrange(memory_slots, "s -> 1 1 1 s"),
            )
            if self.mask_mode == "token" and not use_cached_memory:
                text_to_memory_mask &= rearrange(
                    placeholder_slot_ids.ne(0), "b t -> b 1 t 1"
                )
            if self.mask_mode == "segment" and not self.only_attend_immediate_memory:
                text_to_memory_mask = torch.ge(
                    rearrange(active_slot_ids, "b t -> b 1 t 1"),
                    rearrange(memory_slots, "s -> 1 1 1 s"),
                )
            if memory_mask is not None:
                text_to_memory_mask &= rearrange(
                    memory_mask, "b s -> b 1 1 s"
                )
            similarity = similarity.masked_fill(
                ~text_to_memory_mask, -torch.finfo(similarity.dtype).max
            )

        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
        attention = similarity.softmax(dim=-1)
        if placeholder_slot_ids is not None:
            attention = attention.masked_fill(
                ~text_to_memory_mask.any(dim=-1, keepdim=True), 0.0
            )
        output = einsum("b h i j, b h j d -> b h i d", attention, v)
        output = rearrange(output, "b h n d -> b n (h d)")

        with torch.no_grad():
            debug = {
                "B": int(batch),
                "T_txt": int(text_length),
                "S": int(slot_count),
                "memory_mask_true": (
                    int(memory_mask.sum().item()) if memory_mask is not None else None
                ),
                "out_abs_mean": float(output.abs().mean().item()),
                "out_l2_mean": float(output.float().norm(dim=-1).mean().item()),
                "attn_mean": float(attention.mean().item()),
                "attn_max": float(attention.max().item()),
            }
            if placeholder_slot_ids is not None:
                debug.update(
                    placeholder_tokens=int(placeholder_slot_ids.ne(0).sum().item()),
                    active_tokens_after_fill=int(active_slot_ids.ne(0).sum().item()),
                    valid_edges=int(text_to_memory_mask.sum().item()),
                    tokens_with_route=int(
                        text_to_memory_mask.any(dim=-1).sum().item()
                    ),
                )
            self.last_debug = debug
        return self.to_out(output)


class GatedCrossAttentionBlock(nn.Module):
    """Gated structural cross-attention with an optional residual FF branch."""

    def __init__(
        self,
        *,
        dim,
        dim_memory,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_memory=True,
        mask_mode="segment",
        enable_ff=False,
        attn_gate_init=0.0,
        ff_gate_init=0.0,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_memory=dim_memory,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_memory=only_attend_immediate_memory,
            mask_mode=mask_mode,
        )
        self.attn_gate = nn.Parameter(torch.tensor([attn_gate_init]))
        self.enable_ff = enable_ff
        if enable_ff:
            self.ff = FeedForward(dim, mult=ff_mult)
            self.ff_gate = nn.Parameter(torch.tensor([ff_gate_init]))
        else:
            self.ff = None
            self.register_parameter("ff_gate", None)

    def forward(
        self,
        x,
        memory,
        placeholder_slot_ids=None,
        memory_mask=None,
        use_cached_memory=False,
        xattn_apply_mask=None,
    ):
        attention_output = self.attn(
            x,
            memory,
            placeholder_slot_ids=placeholder_slot_ids,
            memory_mask=memory_mask,
            use_cached_memory=use_cached_memory,
        )
        apply_mask = None
        if xattn_apply_mask is not None:
            apply_mask = xattn_apply_mask.to(
                device=attention_output.device, dtype=attention_output.dtype
            )
            if apply_mask.ndim == 2:
                apply_mask = apply_mask.unsqueeze(-1)
            attention_output = attention_output * apply_mask
        x = x + attention_output * self.attn_gate.tanh()
        if self.ff is not None:
            ff_output = self.ff(x)
            if apply_mask is not None:
                ff_output = ff_output * apply_mask
            x = x + ff_output * self.ff_gate.tanh()
        return x


class StructuralMemoryPreMLP(nn.Module):
    """Inject structural memory after self-attention and before the native MLP."""

    def __init__(self, original_norm, gated_cross_attn_layer):
        super().__init__()
        self.original_norm = original_norm
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.structural_memory = None
        self.structural_memory_mask = None
        self.placeholder_slot_ids = None
        self.use_cached_memory = False
        self.xattn_apply_mask = None

    def is_conditioned(self):
        return (
            self.structural_memory is not None
            and self.structural_memory_mask is not None
            and self.placeholder_slot_ids is not None
        )

    def clear_conditioning(self):
        self.structural_memory = None
        self.structural_memory_mask = None
        self.placeholder_slot_ids = None
        self.use_cached_memory = False
        self.xattn_apply_mask = None

    def forward(self, hidden_states):
        if not self.is_conditioned():
            raise ValueError(
                "Structural memory and placeholder routing must be conditioned "
                "before the decoder forward pass"
            )
        hidden_states = self.gated_cross_attn_layer(
            hidden_states,
            self.structural_memory,
            placeholder_slot_ids=self.placeholder_slot_ids,
            memory_mask=self.structural_memory_mask,
            use_cached_memory=self.use_cached_memory,
            xattn_apply_mask=self.xattn_apply_mask,
        )
        return self.original_norm(hidden_states)


class StructuralCrossAttentionMixin(nn.Module):
    """Condition and route structural memory into selected decoder layers."""

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _structural_wrappers(self):
        for decoder_layer in self._get_decoder_layers():
            wrapper = getattr(decoder_layer, "post_attention_layernorm", None)
            if isinstance(wrapper, StructuralMemoryPreMLP):
                yield wrapper

    def init_structural_cross_attention(
        self,
        placeholder_token_ids,
        lang_hidden_size,
        mem_hidden_size,
        cross_attn_every_n_layers,
        xattn_heads=4,
        xattn_dim_head=64,
        only_attend_immediate_memory=True,
        mask_mode="segment",
    ):
        if cross_attn_every_n_layers <= 0:
            raise ValueError("cross_attn_every_n_layers must be positive")
        selected_layers = []
        for layer_idx, decoder_layer in enumerate(self._get_decoder_layers()):
            if (layer_idx + 1) % cross_attn_every_n_layers != 0:
                continue
            if not hasattr(decoder_layer, "post_attention_layernorm"):
                raise TypeError(
                    f"Layer {layer_idx + 1} does not expose "
                    "post_attention_layernorm; the selected fusion placement "
                    "is unsupported for this backbone"
                )
            gated_xattn = GatedCrossAttentionBlock(
                dim=lang_hidden_size,
                dim_memory=mem_hidden_size,
                dim_head=xattn_dim_head,
                heads=xattn_heads,
                only_attend_immediate_memory=only_attend_immediate_memory,
                mask_mode=mask_mode,
                enable_ff=False,
                attn_gate_init=0.0,
                ff_gate_init=0.0,
            )
            decoder_layer.post_attention_layernorm = StructuralMemoryPreMLP(
                original_norm=decoder_layer.post_attention_layernorm,
                gated_cross_attn_layer=gated_xattn,
            )
            selected_layers.append(layer_idx + 1)

        if not selected_layers:
            raise ValueError("No structural cross-attention layers were inserted")
        self.placeholder_token_ids = tuple(map(int, placeholder_token_ids))
        self.structural_xattn_layer_indices = tuple(selected_layers)
        self.initialized_structural_xattn = True
        self._use_cached_structural_memory = False
        print(
            "[STRUCTURAL-XATTN] placement=post_self_attn_pre_mlp "
            f"layers={selected_layers}"
        )

    def condition_structural_memory(self, structural_memory, structural_memory_mask):
        for wrapper in self._structural_wrappers():
            wrapper.structural_memory = structural_memory
            wrapper.structural_memory_mask = structural_memory_mask
        self._use_cached_structural_memory = True

    def clear_structural_memory(self):
        for wrapper in self._structural_wrappers():
            wrapper.clear_conditioning()
        self._use_cached_structural_memory = False

    def is_conditioned(self):
        wrappers = tuple(self._structural_wrappers())
        return bool(wrappers) and all(wrapper.is_conditioned() for wrapper in wrappers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        routing_start_idx=None,
        xattn_apply_mask=None,
        **kwargs,
    ):
        if not getattr(self, "initialized_structural_xattn", False):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        if input_ids is None:
            raise ValueError("input_ids must be provided for structural-xattn forward")

        placeholder_slot_ids = build_placeholder_slot_ids(
            input_ids,
            self.placeholder_token_ids,
            routing_start_idx=routing_start_idx,
        )
        use_cached_memory = (
            self._use_cached_structural_memory
            and self.is_conditioned()
            and not placeholder_slot_ids.ne(0).any()
        )
        for wrapper in self._structural_wrappers():
            if not use_cached_memory:
                wrapper.placeholder_slot_ids = placeholder_slot_ids
            wrapper.use_cached_memory = use_cached_memory
            wrapper.xattn_apply_mask = xattn_apply_mask

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        if labels is not None:
            kwargs["labels"] = labels
        return super().forward(**kwargs)


