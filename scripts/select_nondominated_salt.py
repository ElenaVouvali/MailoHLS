#!/usr/bin/env python3
"""Arbitrate already synthesized SALT candidates without changing raw outputs."""
import argparse, json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "paper_plots"

METRICS = ("latency", "bram", "dsp", "ff", "lut")

def dominates(a, b):
    av = [a[m] for m in METRICS]
    bv = [b[m] for m in METRICS]
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, default=RESULTS_DIR / "SALT_experiments.jsonl", nargs="?")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "SALT_deployment_candidates.jsonl")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    groups = {}
    for row in rows:
        if row["stage"] == "baseline" or row["stage"] == "stage-3" or row["stage"] == "predicted-directives":
            if row.get("latency") is not None and row.get("resources"):
                groups.setdefault(row["kernel"], []).append(row)
    selected = []
    for kernel, candidates in groups.items():
        for candidate in candidates:
            point = {**candidate, **candidate["resources"]}
            dominated_by = [other.get("stage") + ":" + str(other.get("objective"))
                            for other in candidates if other is not candidate and dominates(
                                {**other, **other["resources"]}, point)]
            selected.append({"kernel": kernel, "stage": candidate["stage"],
                             "objective": candidate.get("objective"),
                             "latency": candidate["latency"],
                             "resources": candidate["resources"],
                             "dominated": bool(dominated_by), "dominated_by": dominated_by,
                             "directives": candidate.get("directives")})
    args.output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected))

if __name__ == "__main__":
    main()
