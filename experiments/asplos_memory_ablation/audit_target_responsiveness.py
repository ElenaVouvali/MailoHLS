#!/usr/bin/env python3

import argparse
import json
import re

from collections import defaultdict
from itertools import combinations


ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9_]+_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assignments(text):
    out = {}

    for raw in str(text or "").splitlines():
        line = raw.strip()
        m = ASSIGN_RE.match(line)

        if m:
            out[m.group(1).upper()] = m.group(2).strip()

    return out


def greedy_prediction(row):
    candidates = [
        c
        for c in row.get("candidates", [])
        if c.get("sample_id") == 0
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"{row.get('context_id')}: "
            f"expected one greedy candidate, got {len(candidates)}"
        )

    return candidates[0]["canonical_prediction"]


def clock_of(row):
    value = row.get("selected_clock_period")

    if value is None:
        value = row.get("clock_period")

    if value is None:
        value = row.get("Clock_Period_nsec")

    return round(float(value), 2)


def evaluate(name, result_path, cases_by_id):
    results = load_jsonl(result_path)

    pred_by_id = {
        row["context_id"]: greedy_prediction(row)
        for row in results
    }

    groups = defaultdict(list)

    for context_id, case in cases_by_id.items():
        if context_id not in pred_by_id:
            raise RuntimeError(
                f"{name}: missing prediction for {context_id}"
            )

        key = (
            case["device"],
            clock_of(case),
        )

        groups[key].append(
            (
                case,
                pred_by_id[context_id],
            )
        )

    informative_groups = 0
    prediction_sensitive_groups = 0

    informative_pairs = 0
    responding_pairs = 0

    changed_sites = 0
    responding_changed_sites = 0
    correctly_changed_sites = 0

    print()
    print("=" * 88)
    print(name)
    print("=" * 88)

    for key, rows in sorted(groups.items()):
        rows.sort(
            key=lambda item:
            float(item[0].get("asplos_budget_tightness", 0.0))
        )

        unique_predictions = len(
            {pred for _, pred in rows}
        )

        group_is_informative = False

        for (case_a, pred_text_a), (case_b, pred_text_b) in combinations(
            rows, 2
        ):
            ref_a = assignments(
                case_a.get("reference_target")
                or case_a.get("target")
            )
            ref_b = assignments(
                case_b.get("reference_target")
                or case_b.get("target")
            )

            pred_a = assignments(pred_text_a)
            pred_b = assignments(pred_text_b)

            changed = [
                lhs
                for lhs in sorted(set(ref_a) & set(ref_b))
                if ref_a[lhs] != ref_b[lhs]
            ]

            if not changed:
                continue

            group_is_informative = True
            informative_pairs += 1

            pair_responded = any(
                pred_a.get(lhs) != pred_b.get(lhs)
                for lhs in changed
            )

            responding_pairs += int(pair_responded)

            for lhs in changed:
                changed_sites += 1

                if pred_a.get(lhs) != pred_b.get(lhs):
                    responding_changed_sites += 1

                if (
                    pred_a.get(lhs) == ref_a.get(lhs)
                    and
                    pred_b.get(lhs) == ref_b.get(lhs)
                ):
                    correctly_changed_sites += 1

        if group_is_informative:
            informative_groups += 1

            if unique_predictions > 1:
                prediction_sensitive_groups += 1

        print(
            f"{key}: "
            f"contexts={len(rows)} "
            f"unique_predictions={unique_predictions} "
            f"informative={group_is_informative}"
        )

    def frac(a, b):
        return a / b if b else float("nan")

    print()
    print(f"groups                         = {len(groups)}")
    print(f"informative_groups             = {informative_groups}")
    print(
        "prediction_sensitive_groups   = "
        f"{prediction_sensitive_groups}/{informative_groups}"
    )
    print(f"informative_pairs              = {informative_pairs}")
    print(
        "pair_transition_response_rate = "
        f"{frac(responding_pairs, informative_pairs):.6f}"
    )
    print(
        "changed_site_response_rate    = "
        f"{frac(responding_changed_sites, changed_sites):.6f}"
    )
    print(
        "changed_site_correct_rate     = "
        f"{frac(correctly_changed_sites, changed_sites):.6f}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cases",
        required=True,
    )

    ap.add_argument(
        "--result",
        action="append",
        required=True,
        help="NAME=JSONL",
    )

    args = ap.parse_args()

    cases = load_jsonl(args.cases)

    cases_by_id = {
        row["context_id"]: row
        for row in cases
    }

    for spec in args.result:
        name, path = spec.split("=", 1)
        evaluate(
            name,
            path,
            cases_by_id,
        )


if __name__ == "__main__":
    main()
