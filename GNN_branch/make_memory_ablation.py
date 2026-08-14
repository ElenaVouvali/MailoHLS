#!/usr/bin/env python3
"""Create deterministic Stage-2 controls from an immutable static memory bank."""

import argparse
import hashlib
import os
from pathlib import Path

import torch


def kernel_seed(global_seed: int, kernel_name: str) -> int:
    material = f"{global_seed}:{kernel_name}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="little")


def memory_files(directory: Path):
    return sorted(directory.glob("*.memory.pt"))


def require_empty_output(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static_dir", required=True, type=Path)
    parser.add_argument("--zero_out", required=True, type=Path)
    parser.add_argument("--shuffled_out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    files = memory_files(args.static_dir)
    if not files:
        raise FileNotFoundError(f"No *.memory.pt files in {args.static_dir}")
    require_empty_output(args.zero_out)
    require_empty_output(args.shuffled_out)

    for source in files:
        static = torch.load(source, map_location="cpu", weights_only=False)
        vectors = static["node_embs"]
        mask = static["node_embs_mask"].bool()

        zero = dict(static)
        zero["node_embs"] = torch.zeros_like(vectors)
        zero["node_embs_mask"] = torch.zeros_like(mask, dtype=torch.bool)
        zero["memory_ablation"] = "zero_information"
        torch.save(zero, args.zero_out / source.name)

        shuffled = dict(static)
        shuffled_vectors = vectors.clone()
        active = torch.where(mask)[0]
        if active.numel() >= 2:
            original = shuffled_vectors[active].clone()
            generator = torch.Generator().manual_seed(
                kernel_seed(args.seed, source.name[:-len(".memory.pt")])
            )
            permutation = torch.randperm(active.numel(), generator=generator)
            if torch.equal(permutation, torch.arange(active.numel())) and len(
                set(row.numpy().tobytes() for row in original)
            ) >= 2:
                permutation = torch.roll(permutation, shifts=1)
            shuffled_vectors[active] = original[permutation]
        shuffled["node_embs"] = shuffled_vectors
        shuffled["memory_ablation"] = "within_kernel_shuffled_slots"
        shuffled["memory_ablation_seed"] = args.seed
        torch.save(shuffled, args.shuffled_out / source.name)

    print(f"Created {len(files)} zero-information and shuffled-slot controls")


if __name__ == "__main__":
    main()
