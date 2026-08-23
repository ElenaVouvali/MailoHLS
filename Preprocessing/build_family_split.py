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
    val_kernels: tuple[str, ...] = (),
    test_kernels: tuple[str, ...] = (),
) -> dict:
    families = defaultdict(lambda: {"indices": [], "kernels": set()})
    rows = []
    with dataset.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((index, row))
            kernel = str(row["kernel_name"])
            family = family_id_from_kernel_name(kernel)
            families[family]["indices"].append(index)
            families[family]["kernels"].add(kernel)
    names = sorted(families)
    requested_val = set(val_kernels)
    requested_test = set(test_kernels)
    if bool(requested_val) != bool(requested_test):
        raise ValueError("Explicit validation and test kernels must both be supplied")
    if requested_val:
        available = set().union(*(record["kernels"] for record in families.values()))
        unknown = (requested_val | requested_test) - available
        if unknown:
            raise ValueError(f"Requested holdout kernels do not exist: {sorted(unknown)}")
        if requested_val & requested_test:
            raise ValueError("Validation and test kernels overlap")
        val_families = {family_id_from_kernel_name(kernel) for kernel in requested_val}
        test_families = {family_id_from_kernel_name(kernel) for kernel in requested_test}
        if val_families & test_families:
            raise ValueError("Validation and test kernels share a family")
        for label, requested, selected in (
            ("validation", requested_val, val_families),
            ("test", requested_test, test_families),
        ):
            complete = set().union(*(families[name]["kernels"] for name in selected))
            if complete != requested:
                raise ValueError(
                    f"{label} holdout splits a kernel family; also include "
                    f"{sorted(complete - requested)}"
                )
        if not set(names) - val_families - test_families:
            raise ValueError("Explicit holdouts leave no training families")
    else:
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
            val_count = sum(len(families[name]["kernels"]) for name in val)
            test_count = sum(len(families[name]["kernels"]) for name in test)
            val_fraction = sum(len(families[name]["indices"]) for name in val) / total_rows
            test_fraction = sum(len(families[name]["indices"]) for name in test) / total_rows
            score = (
                abs(val_count - val_kernel_target) + abs(test_count - test_kernel_target),
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
    training_indices = set(split["train"])
    positive_training_areas = [
        float(row["area"])
        for index, row in rows
        if index in training_indices and float(row["area"]) > 0.0
    ]
    if not positive_training_areas:
        raise ValueError("Cannot fit the effective-area floor without positive training areas")
    minimum_positive_area = min(positive_training_areas)
    digest = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema": "mailohls-family-split-v2",
        "seed": seed,
        "dataset_sha256": digest.hexdigest(),
        "effective_area_policy": "half_minimum_positive_training_area",
        "minimum_positive_training_area": minimum_positive_area,
        "effective_area_floor": minimum_positive_area / 2.0,
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
    parser.add_argument("--val_kernels", default="", help="Complete comma-separated validation families")
    parser.add_argument("--test_kernels", default="", help="Complete comma-separated sealed test families")
    args = parser.parse_args()
    split = build_family_split(
        args.dataset_jsonl,
        seed=args.seed,
        val_family_count=args.val_family_count,
        test_family_count=args.test_family_count,
        val_kernel_target=args.val_kernel_target,
        test_kernel_target=args.test_kernel_target,
        val_kernels=tuple(value.strip() for value in args.val_kernels.split(",") if value.strip()),
        test_kernels=tuple(value.strip() for value in args.test_kernels.split(",") if value.strip()),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    for name in ("train", "val", "test"):
        print(f"[SPLIT] {name}: families={len(split[name + '_families'])} "
              f"kernels={len(split[name + '_kernels'])} rows={len(split[name + '_jsonl_idx'])}")
    print(f"[SPLIT] saved {args.output_json}")
    print(f"[SPLIT] training-only effective_area_floor={split['effective_area_floor']:g}")


if __name__ == "__main__":
    main()
