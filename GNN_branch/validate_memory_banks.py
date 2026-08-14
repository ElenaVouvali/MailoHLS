#!/usr/bin/env python3
"""Validate four matched memory banks and write reproducibility manifests."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_by_kernel(directory: Path):
    files = {p.name[:-len(".memory.pt")]: p for p in directory.glob("*.memory.pt")}
    if not files:
        raise ValueError(f"No memory packs in {directory}")
    return files


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def rows_as_multiset(tensor: torch.Tensor):
    rows = tensor.detach().cpu().contiguous()
    return sorted(row.numpy().tobytes() for row in rows)


def metadata_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    return left == right


def validate_pack(kernel, current, static, zero, shuffled):
    packs = (current, static, zero, shuffled)
    vectors = [p["node_embs"] for p in packs]
    masks = [p["node_embs_mask"].bool() for p in packs]
    shape = vectors[0].shape
    if len(shape) != 2 or any(v.shape != shape for v in vectors):
        raise ValueError(f"{kernel}: dimensions do not match")
    max_slots = int(packs[0].get("max_slots", shape[0]))
    if any(int(p.get("max_slots", p["node_embs"].shape[0])) != max_slots for p in packs):
        raise ValueError(f"{kernel}: max_slots do not match")
    if max_slots != shape[0]:
        raise ValueError(f"{kernel}: max_slots differs from tensor shape")
    if not all(torch.isfinite(v).all().item() for v in vectors):
        raise ValueError(f"{kernel}: non-finite memory tensor")
    for key in ("slot_ids", "slot_cats", "node_ids", "labels", "disable_pragma_injection"):
        values = [pack.get(key) for pack in packs]
        if values[0] is None or any(not metadata_equal(values[0], value) for value in values[1:]):
            raise ValueError(f"{kernel}: inconsistent alignment field {key}")
    if not torch.equal(masks[0], masks[1]) or not metadata_equal(
        current.get("labels"), static.get("labels")
    ):
        raise ValueError(f"{kernel}: static/current masks or labels differ")
    if masks[2].any().item():
        raise ValueError(f"{kernel}: zero bank contains active slots")
    if torch.count_nonzero(vectors[2]).item():
        raise ValueError(f"{kernel}: zero bank contains non-zero vectors")
    if not torch.equal(masks[3], masks[1]) or not metadata_equal(
        shuffled.get("labels"), static.get("labels")
    ):
        raise ValueError(f"{kernel}: shuffled mask or labels differ from static")
    active = torch.where(masks[1])[0]
    original = vectors[1][active]
    permuted = vectors[3][active]
    if rows_as_multiset(original) != rows_as_multiset(permuted):
        raise ValueError(f"{kernel}: shuffled active vectors changed their multiset")
    if not torch.equal(
        torch.sort(torch.linalg.vector_norm(original, dim=1)).values,
        torch.sort(torch.linalg.vector_norm(permuted, dim=1)).values,
    ):
        raise ValueError(f"{kernel}: shuffled active vector norms differ")
    provenance_keys = (
        "gnn_checkpoint_sha256", "gnn_config_sha256",
        "source_pt_manifest_sha256", "git_commit",
    )
    for key in provenance_keys:
        values = [pack.get(key) for pack in packs]
        if not values[0] or any(value != values[0] for value in values[1:]):
            raise ValueError(f"{kernel}: inconsistent checkpoint provenance field {key}")
    distinct = len(set(rows_as_multiset(original))) >= 2
    changed = distinct and not torch.equal(original, permuted)
    return max_slots, shape[1], distinct, changed


def provenance(pack):
    return {
        key: pack.get(key)
        for key in (
            "ckpt", "gnn_checkpoint_sha256", "gnn_config_sha256",
            "source_pt_manifest_sha256", "git_commit", "embedding_mode",
            "disable_pragma_injection",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_dir", required=True, type=Path)
    parser.add_argument("--static_dir", required=True, type=Path)
    parser.add_argument("--zero_dir", required=True, type=Path)
    parser.add_argument("--shuffled_dir", required=True, type=Path)
    args = parser.parse_args()
    directories = {
        "current": args.current_dir,
        "static": args.static_dir,
        "zero": args.zero_dir,
        "shuffled": args.shuffled_dir,
    }
    banks = {name: files_by_kernel(path) for name, path in directories.items()}
    kernel_sets = [set(bank) for bank in banks.values()]
    if any(kernels != kernel_sets[0] for kernels in kernel_sets[1:]):
        raise ValueError("All four banks must contain exactly the same kernels")

    expected_shape = None
    eligible = changed = 0
    for kernel in sorted(kernel_sets[0]):
        loaded = [load(banks[name][kernel]) for name in directories]
        max_slots, dim, distinct, did_change = validate_pack(kernel, *loaded)
        if expected_shape is None:
            expected_shape = (max_slots, dim)
        elif expected_shape != (max_slots, dim):
            raise ValueError(f"{kernel}: bank-wide dimension/max_slots mismatch")
        eligible += int(distinct)
        changed += int(did_change)
    if eligible and changed != eligible:
        raise ValueError(
            f"Shuffle failed to change {eligible - changed}/{eligible} kernels with distinct active vectors"
        )

    for name, directory in directories.items():
        file_records = {}
        for kernel, path in sorted(banks[name].items()):
            pack = load(path)
            file_records[path.name] = {"sha256": sha256(path), "provenance": provenance(pack)}
        manifest = {
            "schema": "mailohls-memory-bank-v1",
            "bank": name,
            "kernel_count": len(file_records),
            "max_slots": expected_shape[0],
            "mem_dim": expected_shape[1],
            "files": file_records,
        }
        (directory / "memory_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"Validated {len(kernel_sets[0])} kernels; shuffled {changed} eligible kernels")


if __name__ == "__main__":
    main()
