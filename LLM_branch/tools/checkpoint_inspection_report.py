#!/usr/bin/env python3
"""Emit metadata-only reports for structural cross-attention checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_report(name: str, tensor: torch.Tensor, gate_scale: float) -> dict:
    floating = tensor.detach().float()
    report = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "finite": bool(torch.isfinite(floating).all()),
        "l2_norm": float(torch.linalg.vector_norm(floating)),
    }
    if name.endswith(".attn_gate"):
        raw = floating.reshape(-1)
        tanh = raw.tanh()
        report["gate"] = {
            "raw": raw.tolist(),
            "tanh": tanh.tolist(),
            "effective": (gate_scale * tanh).tolist(),
            "scale": gate_scale,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gate-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Invalid or empty state dict: {args.checkpoint}")

    payload = {
        "schema": "mailohls-checkpoint-inspection-v1",
        "checkpoint_filename": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "gate_scale": args.gate_scale,
        "tensor_count": len(state),
        "all_finite": True,
        "tensors": [],
    }
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor state entry: {name}")
        report = tensor_report(name, value, args.gate_scale)
        payload["all_finite"] &= report["finite"]
        payload["tensors"].append(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
