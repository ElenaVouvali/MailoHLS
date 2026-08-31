#!/usr/bin/env python3
"""Summarize ranking and downstream selection quality from GNN prediction CSVs.

Expected MailoHLS columns are the validation CSVs produced by train_GNN.py:
  target,kernel,target_group,point_key,actual_log2,predicted_log2,
  actual_physical,predicted_physical,...

Selection metrics minimize the predicted physical target (latency or area).
Regret is (best_actual_in_predicted_top_k / oracle_actual) - 1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def parse_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/predictions.csv")
    name, path = text.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Use NAME=/path/to/predictions.csv")
    return name.strip(), Path(path).expanduser().resolve()


def finite_pairs(group: pd.DataFrame):
    actual = pd.to_numeric(group["actual_physical"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(group["predicted_physical"], errors="coerce").to_numpy(float)
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0.0) & (predicted > 0.0)
    return actual[mask], predicted[mask], group.loc[mask].copy()


def safe_corr(fn, a, b) -> float:
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    value = fn(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def group_metrics(group: pd.DataFrame, ks: list[int]) -> dict:
    actual, predicted, valid = finite_pairs(group)
    if len(actual) == 0:
        return {"count": 0}
    out = {
        "count": int(len(actual)),
        "kendall_tau_b": safe_corr(kendalltau, actual, predicted),
        "spearman_rho": safe_corr(spearmanr, actual, predicted),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
    }
    oracle = float(np.min(actual))
    oracle_mask = np.isclose(actual, oracle, rtol=1e-9, atol=1e-12)
    order = np.argsort(predicted, kind="stable")
    for k in ks:
        kk = min(int(k), len(order))
        selected = order[:kk]
        best_selected = float(np.min(actual[selected]))
        out[f"top{k}_best_actual"] = best_selected
        out[f"top{k}_regret"] = float(best_selected / oracle - 1.0)
        out[f"top{k}_oracle_recall"] = float(bool(np.any(oracle_mask[selected])))
    return out


def nanmean(values):
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def summarize_one(name: str, path: Path, ks: list[int]):
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"target", "kernel", "target_group", "actual_physical", "predicted_physical"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    rows = []
    for (target, target_group), group in df.groupby(["target", "target_group"], sort=True):
        metrics = group_metrics(group, ks)
        rows.append({
            "model": name,
            "target": str(target),
            "target_group": str(target_group),
            "kernel": str(group["kernel"].iloc[0]),
            **metrics,
        })
    per_group = pd.DataFrame(rows)

    aggregate = []
    for target, table in per_group.groupby("target", sort=True):
        record = {
            "model": name,
            "target": target,
            "groups": int(len(table)),
            "points": int(table["count"].sum()),
            "macro_kendall_tau_b": nanmean(table["kendall_tau_b"]),
            "macro_spearman_rho": nanmean(table["spearman_rho"]),
            "macro_mae": nanmean(table["mae"]),
            "macro_rmse": nanmean(table["rmse"]),
        }
        for k in ks:
            record[f"macro_top{k}_regret"] = nanmean(table[f"top{k}_regret"])
            record[f"macro_top{k}_oracle_recall"] = nanmean(table[f"top{k}_oracle_recall"])
        aggregate.append(record)
    return per_group, pd.DataFrame(aggregate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append", required=True, type=parse_spec,
                    help="Repeat as --predictions NAME=file.csv")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    ks = sorted(set(k for k in args.k if k > 0))
    if not ks:
        raise ValueError("--k must contain positive integers")

    all_groups, all_summary = [], []
    for name, path in args.predictions:
        per_group, summary = summarize_one(name, path, ks)
        all_groups.append(per_group)
        all_summary.append(summary)

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups = pd.concat(all_groups, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    groups.to_csv(out / "per_target_group_metrics.csv", index=False)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    payload = {
        "schema": "mailohls-surrogate-selection-summary-v1",
        "k": ks,
        "regret_definition": "best actual among predicted top-k / oracle actual - 1",
        "selection_direction": "minimize",
        "models": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    (out / "summary_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"[DONE] wrote {out}")


if __name__ == "__main__":
    main()
