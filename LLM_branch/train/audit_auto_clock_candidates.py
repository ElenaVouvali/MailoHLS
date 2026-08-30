from __future__ import annotations

import argparse
import json
import math
from collections import Counter


def get_qor(case):
    if isinstance(case.get("qor_by_clock"), dict):
        return case["qor_by_clock"]

    # AUTO case banks use the compact scalar representation.
    if isinstance(case.get("adp_by_clock"), dict):
        return case["adp_by_clock"]

    qor = case.get("qor")
    return qor if isinstance(qor, dict) else None


def valid_rows(case):
    qor = get_qor(case)
    if not qor:
        return []

    rows = []

    for clk_s, info in qor.items():
        try:
            clk = round(float(clk_s), 2)
            adp = float(info["adp"]) if isinstance(info, dict) else float(info)
        except (TypeError, ValueError, KeyError):
            continue

        if math.isfinite(adp):
            rows.append((clk, adp))

    return sorted(rows, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    candidate_sets = Counter()
    winner_counts = Counter()
    strict_override_winner_counts = Counter()

    usable = 0

    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            case = json.loads(line)
            rows = valid_rows(case)

            if len(rows) < 2:
                continue

            usable += 1

            clocks = tuple(clk for clk, _ in rows)
            candidate_sets[clocks] += 1

            fastest_clk, _ = rows[0]
            best_clk, _ = min(rows, key=lambda x: x[1])

            winner_counts[best_clk] += 1

            if best_clk != fastest_clk:
                strict_override_winner_counts[best_clk] += 1

    print("usable_cases =", usable)

    print("candidate_sets =")
    for clocks, count in candidate_sets.most_common():
        print(" ", clocks, ":", count)

    print(
        "winner_counts =",
        dict(sorted(winner_counts.items())),
    )

    print(
        "strict_override_winner_counts =",
        dict(sorted(strict_override_winner_counts.items())),
    )


if __name__ == "__main__":
    main()
