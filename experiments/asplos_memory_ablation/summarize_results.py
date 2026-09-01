#!/usr/bin/env python3
"""Summarize fixed-checkpoint memory-ablation inference outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def greedy_candidate(row: dict) -> dict:
    candidates = row.get("candidates", [])
    greedy = [candidate for candidate in candidates if candidate.get("sample_id") == 0]
    if len(greedy) != 1:
        raise ValueError(
            f"{row.get('context_id')}: expected exactly one sample_id=0"
        )
    return greedy[0]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="NAME=JSONL",
    )
    parser.add_argument("--summary_csv", required=True, type=Path)
    parser.add_argument("--synthesis_queue_jsonl", required=True, type=Path)
    args = parser.parse_args()

    variants: dict[str, Path] = {}
    for item in args.result:
        if "=" not in item:
            raise ValueError(f"Expected NAME=JSONL, got {item!r}")
        name, raw_path = item.split("=", 1)
        if name in variants:
            raise ValueError(f"Duplicate variant: {name}")
        variants[name] = Path(raw_path)

    summary = []
    queue = []
    context_sets = {}
    for name, path in variants.items():
        rows = load_jsonl(path)
        context_sets[name] = {row.get("context_id") for row in rows}
        candidates = [greedy_candidate(row) for row in rows]
        summary.append(
            {
                "variant": name,
                "contexts": len(rows),
                "schema_valid_rate": mean(
                    [float(bool(candidate.get("schema_compliant"))) for candidate in candidates]
                ),
                "exact_design_match_rate": mean(
                    [float(bool(candidate.get("exact_design_match"))) for candidate in candidates]
                ),
                "directive_value_accuracy": mean(
                    [float(candidate["value_accuracy_over_expected"]) for candidate in candidates]
                ),
                "mean_active_memory_slots": mean(
                    [float(row.get("memory_active_slots", 0)) for row in rows]
                ),
            }
        )
        for row, candidate in zip(rows, candidates):
            queue.append(
                {
                    "schema": "mailohls-asplos-synthesis-queue-v1",
                    "variant": name,
                    "context_id": row.get("context_id"),
                    "kernel_name": row.get("kernel_name"),
                    "objective": row.get("obj_mode"),
                    "device": row.get("device"),
                    "clock_period_ns": row.get("clock_period_ns"),
                    "resource_budget_id": row.get("resource_budget_id"),
                    "resource_budget": row.get("resource_budget"),
                    "canonical_prediction": candidate["canonical_prediction"],
                }
            )

    expected_contexts = None
    for name, contexts in context_sets.items():
        if expected_contexts is None:
            expected_contexts = contexts
        elif contexts != expected_contexts:
            raise ValueError(
                f"Context mismatch for {name}: "
                f"missing={sorted(expected_contexts - contexts)[:5]}, "
                f"extra={sorted(contexts - expected_contexts)[:5]}"
            )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0])
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    args.synthesis_queue_jsonl.parent.mkdir(
        parents=True, exist_ok=True
    )

    unique = {}

    for row in queue:
        key = (
            row["context_id"],
            row["canonical_prediction"],
        )

        if key not in unique:
            rec = dict(row)
            rec["variants"] = [rec.pop("variant")]
            unique[key] = rec
        else:
            variant = row["variant"]
            if variant not in unique[key]["variants"]:
                unique[key]["variants"].append(variant)

    with args.synthesis_queue_jsonl.open(
        "w", encoding="utf-8"
    ) as handle:
        for row in unique.values():
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"[DONE] summary -> {args.summary_csv}")
    print(
        f"[DONE] deduplicated synthesis queue "
        f"({len(unique)} designs) -> "
        f"{args.synthesis_queue_jsonl}"
    )


if __name__ == "__main__":
    main()
