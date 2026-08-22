#!/usr/bin/env python3
"""Create a deterministic, auditable family-disjoint MailoHLS split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from LLM_branch.common.mailohls_contract import family_id_from_kernel_name


def build_family_split(
    dataset: Path,
    *,
    seed: int = 123,
    val_family_count: int = 4,
    test_family_count: int = 4,
    val_kernel_target: int = 7,
    test_kernel_target: int = 8,
    trials: int = 4096,
) -> dict:
    families = defaultdict(lambda: {"indices": [], "kernels": set()})
    with dataset.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            kernel = str(row["kernel_name"])
            family = family_id_from_kernel_name(kernel)
            families[family]["indices"].append(index)
            families[family]["kernels"].add(kernel)
    names = sorted(families)
    if min(val_family_count, test_family_count) < 1:
        raise ValueError("Validation and test must each contain at least one family")
    if len(names) <= val_family_count + test_family_count:
        raise ValueError("Too few families remain for a nonempty training split")

    total_rows = sum(len(record["indices"]) for record in families.values())
    rng = random.Random(seed)
    best = None
    for _ in range(max(1, trials)):
        shuffled = rng.sample(names, len(names))
        val = tuple(sorted(shuffled[:val_family_count]))
        test = tuple(sorted(shuffled[val_family_count:val_family_count + test_family_count]))
        val_kernels = sum(len(families[name]["kernels"]) for name in val)
        test_kernels = sum(len(families[name]["kernels"]) for name in test)
        val_fraction = sum(len(families[name]["indices"]) for name in val) / total_rows
        test_fraction = sum(len(families[name]["indices"]) for name in test) / total_rows
        score = (
            abs(val_kernels - val_kernel_target) + abs(test_kernels - test_kernel_target),
            abs(val_fraction - 0.15) + abs(test_fraction - 0.15),
            val,
            test,
        )
        if best is None or score < best[0]:
            best = (score, set(val), set(test))

    _, val_families, test_families = best
    split = {"train": [], "val": [], "test": []}
    split_kernels = {name: set() for name in split}
    for family, record in sorted(families.items()):
        name = "val" if family in val_families else "test" if family in test_families else "train"
        split[name].extend(record["indices"])
        split_kernels[name].update(record["kernels"])
    digest = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema": "mailohls-family-split-v2",
        "seed": seed,
        "dataset_sha256": digest.hexdigest(),
        "train_jsonl_idx": sorted(split["train"]),
        "val_jsonl_idx": sorted(split["val"]),
        "test_jsonl_idx": sorted(split["test"]),
        "train_families": sorted(set(names) - val_families - test_families),
        "val_families": sorted(val_families),
        "test_families": sorted(test_families),
        "train_kernels": sorted(split_kernels["train"]),
        "val_kernels": sorted(split_kernels["val"]),
        "test_kernels": sorted(split_kernels["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_jsonl", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val_family_count", type=int, default=4)
    parser.add_argument("--test_family_count", type=int, default=4)
    parser.add_argument("--val_kernel_target", type=int, default=7)
    parser.add_argument("--test_kernel_target", type=int, default=8)
    args = parser.parse_args()
    split = build_family_split(
        args.dataset_jsonl,
        seed=args.seed,
        val_family_count=args.val_family_count,
        test_family_count=args.test_family_count,
        val_kernel_target=args.val_kernel_target,
        test_kernel_target=args.test_kernel_target,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    for name in ("train", "val", "test"):
        print(f"[SPLIT] {name}: families={len(split[name + '_families'])} "
              f"kernels={len(split[name + '_kernels'])} rows={len(split[name + '_jsonl_idx'])}")
    print(f"[SPLIT] saved {args.output_json}")


if __name__ == "__main__":
    main()
