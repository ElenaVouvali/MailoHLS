#!/usr/bin/env python3
"""Aggregate final physical-unit test metrics across independent seeds."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


METRICS = ("mape", "rmse", "mse", "mae", "max_err", "tau")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-glob",
        required=True,
        help="Glob matching run1/test_physical_metrics.csv files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(arguments.runs_glob))]
    if len(paths) < 3:
        raise SystemExit("At least three completed seed runs are required.")

    frames = []
    expected_keys = None
    for path in paths:
        frame = pd.read_csv(path)
        required = {"target", "aggregation", *METRICS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise SystemExit(f"{path} is missing columns {missing}")
        keys = list(zip(frame["target"], frame["aggregation"]))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise SystemExit(f"Metric rows differ in {path}")
        frame = frame.copy()
        frame["run"] = path.parent.parent.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby(
        ["target", "aggregation"], sort=False
    )[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(arguments.output, index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Wrote {len(paths)}-seed summary to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
