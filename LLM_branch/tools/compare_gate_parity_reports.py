#!/usr/bin/env python3
"""Compare effective gates in two metadata-only checkpoint reports."""

import argparse
import json
from pathlib import Path


def gates(report: dict) -> dict:
    return {
        tensor["name"]: tensor["gate"]["effective"]
        for tensor in report["tensors"]
        if "gate" in tensor
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("baked", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    original_report = json.loads(args.original.read_text(encoding="utf-8"))
    baked_report = json.loads(args.baked.read_text(encoding="utf-8"))
    original_gates = gates(original_report)
    baked_gates = gates(baked_report)
    if original_gates.keys() != baked_gates.keys():
        raise RuntimeError("Gate tensor names differ")

    comparisons = []
    max_abs_error = 0.0
    for name in sorted(original_gates):
        left = original_gates[name]
        right = baked_gates[name]
        if len(left) != len(right):
            raise RuntimeError(f"Gate shape differs: {name}")
        error = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
        max_abs_error = max(max_abs_error, error)
        comparisons.append({"name": name, "max_abs_error": error})

    payload = {
        "schema": "mailohls-gate-parity-audit-v1",
        "original_checkpoint_sha256": original_report["checkpoint_sha256"],
        "baked_checkpoint_sha256": baked_report["checkpoint_sha256"],
        "original_gate_scale": original_report["gate_scale"],
        "baked_gate_scale": baked_report["gate_scale"],
        "gate_tensor_count": len(comparisons),
        "max_abs_effective_gate_error": max_abs_error,
        "atol": args.atol,
        "parity": max_abs_error <= args.atol,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
