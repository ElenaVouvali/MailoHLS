#!/usr/bin/env python3
"""Freeze QoR-blind Kalman contexts for the ASPLOS memory ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


BUDGET_FRACTION_FIELDS = (
    "budget_frac_bram_18k",
    "budget_frac_dsp",
    "budget_frac_ff",
    "budget_frac_lut",
)

PRIVATE_EVALUATION_FIELDS = {
    "area",
    "bram_util_%",
    "dsp_util_%",
    "ff_util_%",
    "is_pareto",
    "latency",
    "lut_util_%",
    "preprocessed_row",
    "weight",
    "_jsonl_idx",
    "_family",
    "_score",
    "_rank_within_kernel",
    "_sample_weight",
    "_local_hard_negatives",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_number"] = line_number
            rows.append(row)
    return rows


def budget_tightness(row: dict) -> float:
    values = []
    for field in BUDGET_FRACTION_FIELDS:
        value = row.get(field)
        if value is None:
            raise ValueError(
                f"Context {row.get('resource_budget_id')!r} lacks {field}"
            )
        values.append(float(value))
    # This score is used only to stratify requests, never as an area/QoR metric.
    return sum(values) / len(values)


def quantile_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    if count == 1:
        return [(length - 1) // 2]
    return sorted(
        {
            round(position * (length - 1) / (count - 1))
            for position in range(count)
        }
    )


def public_case(row: dict, rank: int, group_size: int) -> dict:
    case = {
        key: value
        for key, value in row.items()
        if key not in PRIVATE_EVALUATION_FIELDS
        and key != "_source_line_number"
    }
    device = str(case["device"])
    clock = float(case.get("selected_clock_period", case["clock_period"]))
    budget_id = str(case["resource_budget_id"])
    case["context_id"] = (
        f"kalman::{device}::{clock:g}ns::{budget_id}"
    )
    case["asplos_budget_stratum"] = rank
    case["asplos_budget_group_size"] = group_size
    case["asplos_budget_tightness"] = budget_tightness(row)
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--manifest_json", required=True, type=Path)
    parser.add_argument("--kernel", default="serrano-kalman-filter")
    parser.add_argument("--objective", default="PARETO_ADP")
    parser.add_argument("--budgets_per_device_clock", type=int, default=3)
    args = parser.parse_args()

    if args.budgets_per_device_clock < 1:
        raise ValueError("--budgets_per_device_clock must be positive")

    eligible = []
    for row in load_jsonl(args.input_jsonl):
        if row.get("kernel_name") != args.kernel:
            continue
        if str(row.get("obj_mode", "")).upper() != args.objective.upper():
            continue
        if str(row.get("frequency_mode", "specified")).lower() != "specified":
            continue
        eligible.append(row)

    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in eligible:
        clock = float(row.get("selected_clock_period", row["clock_period"]))
        grouped[(str(row["device"]), clock)].append(row)

    if not grouped:
        raise ValueError("No eligible specified-clock contexts")

    selected = []
    group_audit = []
    for (device, clock), rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                budget_tightness(row),
                str(row.get("resource_budget_id", "")),
                int(row["_source_line_number"]),
            ),
        )
        indices = quantile_indices(
            len(ordered), args.budgets_per_device_clock
        )
        chosen = [
            public_case(ordered[index], rank, len(ordered))
            for rank, index in enumerate(indices)
        ]
        selected.extend(chosen)
        group_audit.append(
            {
                "device": device,
                "clock_period_ns": clock,
                "eligible_contexts": len(ordered),
                "selected_indices": indices,
                "selected_context_ids": [row["context_id"] for row in chosen],
                "selected_budget_tightness": [
                    row["asplos_budget_tightness"] for row in chosen
                ],
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "schema": "mailohls-asplos-memory-ablation-cases-v1",
        "selection_policy": (
            "specified-clock only; group by device/clock; select evenly "
            "spaced quantiles of the arithmetic mean of four requested "
            "budget fractions; no QoR field participates in selection"
        ),
        "kernel": args.kernel,
        "objective": args.objective.upper(),
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": sha256_file(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_sha256": sha256_file(args.output_jsonl),
        "eligible_context_count": len(eligible),
        "selected_context_count": len(selected),
        "groups": group_audit,
        "selection_excluded_fields": sorted(PRIVATE_EVALUATION_FIELDS),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[DONE] selected {len(selected)} of {len(eligible)} contexts "
        f"across {len(grouped)} device/clock groups"
    )
    print(f"[DONE] cases -> {args.output_jsonl}")
    print(f"[DONE] manifest -> {args.manifest_json}")


if __name__ == "__main__":
    main()
