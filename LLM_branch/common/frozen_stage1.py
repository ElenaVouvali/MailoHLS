"""Shared invariants for an immutable Stage-1 language adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import torch


def _stage1_group(name: str) -> str | None:
    if "lora_" in name:
        return "lora"
    if "trainable_tokens_delta" in name:
        return "special_tokens"
    return None


def frozen_stage1_hashes(model: torch.nn.Module) -> Dict[str, Any]:
    """Hash names, shapes, dtypes, and bytes of all Stage-1 adapter tensors."""
    digests = {name: hashlib.sha256() for name in ("lora", "special_tokens", "combined")}
    counts = {"lora": 0, "special_tokens": 0}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        group = _stage1_group(name)
        if group is None:
            continue
        tensor = parameter.detach().cpu().contiguous()
        header = f"{name}\0{tuple(tensor.shape)}\0{tensor.dtype}\0".encode("utf-8")
        payload = tensor.view(torch.uint8).numpy().tobytes()
        for digest in (digests[group], digests["combined"]):
            digest.update(header)
            digest.update(payload)
        counts[group] += 1
    if not counts["lora"]:
        raise ValueError("Frozen Stage-1 state contains no LoRA tensors")
    if not counts["special_tokens"]:
        raise ValueError("Frozen Stage-1 state contains no special-token embedding deltas")
    return {
        "schema": "mailohls-frozen-stage1-v1",
        "lora_sha256": digests["lora"].hexdigest(),
        "special_token_sha256": digests["special_tokens"].hexdigest(),
        "combined_sha256": digests["combined"].hexdigest(),
        "tensor_counts": counts,
    }


def assert_stage1_frozen(model: torch.nn.Module) -> None:
    unexpected = [
        name for name, parameter in model.named_parameters()
        if _stage1_group(name) is not None and parameter.requires_grad
    ]
    if unexpected:
        raise RuntimeError("Stage-1 adapter tensors must remain frozen: " + ", ".join(unexpected[:12]))


def disable_frozen_lora_dropout(model: torch.nn.Module) -> int:
    """Undo recursive ``model.train()`` for frozen PEFT LoRA dropout modules."""
    assert_stage1_frozen(model)
    count = 0
    for name, module in model.named_modules():
        if "lora_dropout" in name and isinstance(module, torch.nn.Dropout):
            module.eval()
            count += 1
    return count


def assert_frozen_stage1_unchanged(model: torch.nn.Module, expected: Dict[str, Any]) -> None:
    assert_stage1_frozen(model)
    actual = frozen_stage1_hashes(model)
    if actual != expected:
        raise RuntimeError(
            "Frozen Stage-1 adapter changed during training: "
            f"expected={expected['combined_sha256']}, actual={actual['combined_sha256']}"
        )


def adapter_weights_path(adapter_dir: str | Path) -> Path:
    root = Path(adapter_dir)
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        path = root / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Stage-1 adapter has no adapter_model weights in {root}")
