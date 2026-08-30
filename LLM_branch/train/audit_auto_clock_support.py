#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict


def get_qor_by_clock(row):
    values = row.get("adp_by_clock") or row.get("qor_by_clock")
    if not isinstance(values, dict) or len(values) < 2:
        return None
    return {float(k): float(v) for k, v in values.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    rows = [
        json.loads(line)
        for line in open(args.input, encoding="utf-8")
        if line.strip()
    ]

    best_counts = Counter()
    group_stats = defaultdict(lambda: {
        "n": 0,
        "nonfast": 0,
        "fastest_regret_sum": 0.0,
    })

    usable = 0
    nonfast = 0

    for row in rows:
        qor = get_qor_by_clock(row)
        if qor is None:
            continue

        usable += 1

        clocks = sorted(qor)
        fastest = clocks[0]
        best = min(clocks, key=lambda c: qor[c])

        family = (
            row.get("family")
            or row.get("kernel_family")
            or row.get("kernel_name")
            or row.get("kernel")
        )
        device = row.get("device", "UNKNOWN")
        key = (family, device)

        best_counts[best] += 1
        group_stats[key]["n"] += 1

        best_qor = qor[best]
        fastest_regret = (
            qor[fastest] / best_qor - 1.0
            if best_qor > 0 else 0.0
        )
        group_stats[key]["fastest_regret_sum"] += fastest_regret

        if best != fastest:
            nonfast += 1
            group_stats[key]["nonfast"] += 1

    print("usable_cases =", usable)
    print("best_clock_counts =", dict(sorted(best_counts.items())))
    print("nonfast_cases =", nonfast)
    print("nonfast_fraction =", nonfast / max(1, usable))

    informative_groups = []

    for key, stats in sorted(group_stats.items()):
        if stats["nonfast"]:
            informative_groups.append(key)
            print(
                "NONFAST_GROUP",
                key,
                "cases=", stats["n"],
                "nonfast=", stats["nonfast"],
                "fraction=", stats["nonfast"] / stats["n"],
                "mean_fastest_regret=",
                stats["fastest_regret_sum"] / stats["n"],
            )

    print("groups_with_nonfast_optimum =", len(informative_groups))
    print("total_groups =", len(group_stats))


if __name__ == "__main__":
    main()