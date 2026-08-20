#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=float)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    state = torch.load(
        args.input,
        map_location="cpu",
        weights_only=False,
    )

    baked = {}
    found = 0
    max_error = 0.0

    for name, tensor in state.items():
        value = tensor.clone()

        if name.endswith(".attn_gate"):
            old = tensor.float()
            effective = args.scale * old.tanh()

            if effective.abs().max().item() >= 1.0:
                raise ValueError(
                    f"{name}: effective gate is outside tanh range: "
                    f"{effective.tolist()}"
                )

            value = torch.atanh(effective).to(tensor.dtype)

            reconstructed = value.float().tanh()
            max_error = max(
                max_error,
                float((reconstructed - effective).abs().max()),
            )
            found += 1

        baked[name] = value

    if found == 0:
        raise RuntimeError("No .attn_gate parameters found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(baked, args.output)

    print(
        f"[GATE-BAKE] gates={found} scale={args.scale:g} "
        f"max_effective_error={max_error:.3e} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()