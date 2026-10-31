"""Training-only audit of measured, feasible near-optimal directive sets.

This is descriptive evidence. Epsilon is an engineering tolerance, not a
measurement-noise estimate. No selected target or training loss is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.stage1_recovery.decisions import parse
from experiments.stage1_recovery.prepare import prepare, sha

UTILIZATION = ("bram_util_%", "dsp_util_%", "ff_util_%", "lut_util_%")
BUDGET = ("budget_frac_bram_18k", "budget_frac_dsp", "budget_frac_ff", "budget_frac_lut")
TOLERANCES = (0.0, 0.005, 0.01, 0.02)


def assignment(text):
    return tuple(sorted(parse(text).items()))


def physical_context(row):
    return (row["kernel_name"], row.get("source_key"), row.get("source_file"),
            str(row["device"]).lower(), round(float(row["clock_period"]), 2))


def load_training_rows(dataset, split, expected_sha):
    """Read only train IDs; reject changed bytes and missing split members."""
    if sha(dataset) != expected_sha:
        raise ValueError("Authenticated GNΩSIS dataset SHA256 changed")
    membership = json.loads(Path(split).read_text())
    train = set(map(int, membership["train_jsonl_idx"]))
    val = set(map(int, membership["val_jsonl_idx"]))
    test = set(map(int, membership["test_jsonl_idx"]))
    if train & val or train & test or val & test:
        raise ValueError("Family split overlaps")
    rows = []
    with Path(dataset).open() as handle:
        for index, line in enumerate(handle):
            if index in train:
                row = json.loads(line)
                row["_jsonl_idx"] = index
                rows.append(row)
    if {r["_jsonl_idx"] for r in rows} != train:
        raise ValueError("Training measurements do not cover the authenticated split")
    return rows


def candidate_banks(rows):
    """Collapse only physically identical repeated rows, never conflicting QoR."""
    banks = defaultdict(dict)
    repeats = 0
    for row in rows:
        context, design = physical_context(row), assignment(row["target"])
        qos = tuple(float(row[k]) for k in ("latency", *UTILIZATION))
        key = (context, design)
        old = banks[context].get(design)
        if old is not None:
            if old["qos"] != qos:
                raise ValueError(f"Conflicting measured QoR for repeated design: {key}")
            repeats += 1
            continue
        banks[context][design] = dict(qos=qos, row_id=int(row["_jsonl_idx"]))
    return banks, repeats


def feasible_designs(request, bank):
    limits = tuple(float(request[k]) for k in BUDGET)
    return {design: receipt for design, receipt in bank.items()
            if all(receipt["qos"][i + 1] / 100.0 <= limits[i] + 1e-9
                   for i in range(4))}


def near_optimal(request, bank, epsilon):
    feasible = feasible_designs(request, bank)
    if not feasible:
        raise ValueError("Authenticated selected request has no feasible measured design")
    optimum = min(item["qos"][0] for item in feasible.values())
    if optimum <= 0:
        raise ValueError("Expected positive measured latency")
    accepted = {design: item for design, item in feasible.items()
                if item["qos"][0] <= (1 + epsilon) * optimum + 1e-12}
    return feasible, accepted, optimum


def audit(selected, raw_rows, tolerances=TOLERANCES):
    banks, duplicate_rows = candidate_banks(raw_rows)
    counters = {str(e): Counter() for e in tolerances}
    family = defaultdict(Counter)
    selected_rows = set()
    selected_assignments = set()
    reuse = Counter()
    missing_selected = 0
    gaps = []
    for request in selected:
        context = physical_context(request)
        if context not in banks:
            raise ValueError(f"No authenticated training measurement bank: {context}")
        chosen = assignment(request["target"])
        if chosen not in banks[context]:
            missing_selected += 1
            continue
        row_id = int(request["_jsonl_idx"])
        selected_rows.add(row_id)
        selected_assignments.add((context[0], chosen))
        reuse[row_id] += 1
        feasible, _, optimum = near_optimal(request, banks[context], 0)
        if chosen not in feasible:
            raise ValueError("Selected measured target is infeasible in its own request")
        other = sorted(item["qos"][0] for design, item in feasible.items() if design != chosen)
        gaps.append((other[0] - optimum) / optimum if other else None)
        for epsilon in tolerances:
            _, accepted, _ = near_optimal(request, banks[context], epsilon)
            stat = counters[str(epsilon)]
            stat["contexts"] += 1
            stat["feasible_designs_total"] += len(feasible)
            stat["accepted_designs_total"] += len(accepted)
            stat["multiple_accepted"] += len(accepted) > 1
            stat["selected_not_accepted"] += chosen not in accepted
            differences = {field for design in accepted for field, value in design
                           if dict(chosen).get(field) != value}
            stat["different_accepted_assignment"] += bool(differences)
            for kind, prefix in (("pipeline", "PIPE"), ("unroll", "UNROLL"),
                                 ("array", "ARRAY")):
                stat[f"disagreement_{kind}"] += any(prefix in field for field in differences)
            family[(str(epsilon), request.get("_family", "unknown"))]["contexts"] += 1
            family[(str(epsilon), request.get("_family", "unknown"))]["multiple_accepted"] += len(accepted) > 1
    if missing_selected:
        raise ValueError(f"{missing_selected} selected designs lack exact measured assignment identity")
    return dict(schema="stage1-training-ambiguity-audit-v1", training_only=True,
                tolerance_interpretation="engineering sensitivity; no statistical noise assertion",
                raw_training_rows=len(raw_rows), duplicate_identical_measurement_rows=duplicate_rows,
                selected_contexts=len(selected), distinct_selected_measured_rows=len(selected_rows),
                distinct_selected_program_assignments=len(selected_assignments),
                repeated_selected_contexts=sum(v - 1 for v in reuse.values()),
                maximum_context_reuse=max(reuse.values(), default=0),
                second_best_relative_gap=dict(available=sum(v is not None for v in gaps),
                    zero_or_better=sum(v is not None and v <= 1e-12 for v in gaps),
                    within_0p5_percent=sum(v is not None and v <= .005 + 1e-12 for v in gaps),
                    within_1_percent=sum(v is not None and v <= .01 + 1e-12 for v in gaps),
                    within_2_percent=sum(v is not None and v <= .02 + 1e-12 for v in gaps)),
                tolerances={key: dict(value) for key, value in counters.items()},
                family={f"{epsilon}:{name}": dict(stat) for (epsilon, name), stat in sorted(family.items())})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--dataset", default="artifacts/llm/mailohls_sft.jsonl")
    ap.add_argument("--split", default="mailohls_runs/mailohls_final_family_split_s123.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest = prepare(Path("mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1"), args.prepared)
    selected = [json.loads(line) for line in (Path(args.prepared) / "train_reconstructed_selected.jsonl").read_text().splitlines()]
    raw = load_training_rows(args.dataset, args.split, manifest["dataset_sha256"])
    result = audit(selected, raw)
    result.update(dataset_sha256=manifest["dataset_sha256"], split_sha256=manifest["split_sha256"],
                  prepared_manifest_sha256=sha(Path(args.prepared) / "manifest.json"))
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("selected_contexts", "distinct_selected_measured_rows", "tolerances")}, indent=2))


if __name__ == "__main__":
    main()
