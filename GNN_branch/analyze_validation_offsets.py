#!/usr/bin/env python3
"""Quantify unseen-kernel offset error from validation predictions only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    required = {"target", "kernel", "actual_log2", "predicted_log2"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing prediction columns: {sorted(missing)}")
    if not np.isfinite(
        frame[["actual_log2", "predicted_log2"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Predictions contain NaN or infinity")

    frame["signed_error_log2"] = (
        frame["actual_log2"] - frame["predicted_log2"]
    )
    offsets = (
        frame.groupby(["target", "kernel"], sort=True)["signed_error_log2"]
        .mean()
        .rename("oracle_kernel_offset_log2")
        .reset_index()
    )
    audited = frame.merge(offsets, on=["target", "kernel"], validate="many_to_one")
    audited["oracle_recentered_log2"] = (
        audited["predicted_log2"] + audited["oracle_kernel_offset_log2"]
    )
    audited["oracle_recentered_physical"] = np.maximum(
        0.0, np.exp2(audited["oracle_recentered_log2"]) - args.epsilon
    )

    summaries = []
    for target, group in audited.groupby("target", sort=True):
        raw_error = group["predicted_log2"] - group["actual_log2"]
        recentered_error = (
            group["oracle_recentered_log2"] - group["actual_log2"]
        )
        raw_mse = float(np.mean(np.square(raw_error)))
        recentered_mse = float(np.mean(np.square(recentered_error)))
        summaries.append(
            {
                "target": target,
                "points": len(group),
                "kernels": group["kernel"].nunique(),
                "raw_log2_mse": raw_mse,
                "oracle_recentered_log2_mse": recentered_mse,
                "fraction_error_removed_by_kernel_offset": (
                    1.0 - recentered_mse / raw_mse if raw_mse > 0 else 0.0
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audited.to_csv(args.output_dir / "oracle_recentered_predictions.csv", index=False)
    offsets.to_csv(args.output_dir / "kernel_offsets.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "offset_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(
        "Diagnostic only: oracle offsets use validation labels and must never "
        "be fed back into training or test inference."
    )


if __name__ == "__main__":
    main()
