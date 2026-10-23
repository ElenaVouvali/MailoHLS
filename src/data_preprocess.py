#!/usr/bin/env python3
"""
data_preprocess.py

Multi-target preprocessing for MailoHLS / GNOSIS-style HLS CSVs.

Main behavior:
  1) Keep ALL devices and ALL clock periods from the input CSV.
  2) Preserve Device and Clock_Period_nsec in the preprocessed output.
  3) Remove invalid designs from the preprocessed CSV.
  4) Compute Area = mean(BRAM%, DSP%, FF%, LUT%).
  5) Compute Pareto flags and weights separately inside each (Device, Clock_Period_nsec) bucket.
  6) Drop columns that are entirely NDIR, except mandatory metadata/QoR columns.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Pareto helper
# -----------------------------------------------------------------------------

def pareto_front_2d(x, y):
    """
    Return a boolean mask indicating Pareto-front points assuming MINIMIZATION
    of both x and y.

    A point i is dominated if there exists j such that:
      x_j <= x_i and y_j <= y_i, and at least one inequality is strict.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                is_pareto[i] = False
                break

    return is_pareto


# -----------------------------------------------------------------------------
# Weight computation
# -----------------------------------------------------------------------------

def compute_pareto_weights(perf, area, w_min_valid=0.1, gamma=2.0):
    """
    Compute Pareto-aware weights for one target bucket.

    Inputs:
      perf: latency values for VALID designs in one (Device, Clock) bucket.
      area: area proxy values for the same designs.

    Output:
      weights: float array in [w_min_valid, 1.0]
      pareto_mask: bool array, True for Pareto-front points.
    """
    perf = np.asarray(perf, dtype=float)
    area = np.asarray(area, dtype=float)
    n = len(perf)

    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    if n == 1:
        return np.array([1.0], dtype=float), np.array([True], dtype=bool)

    eps = 1e-8

    # Normalize inside the target bucket only.
    perf_min, perf_max = perf.min(), perf.max()
    area_min, area_max = area.min(), area.max()

    perf_n = (perf - perf_min) / (perf_max - perf_min + eps)
    area_n = (area - area_min) / (area_max - area_min + eps)

    pareto_mask = pareto_front_2d(perf_n, area_n)

    x_front = perf_n[pareto_mask]
    y_front = area_n[pareto_mask]

    # Sort frontier by latency for segment-distance computation.
    order = np.argsort(x_front)
    x_front = x_front[order]
    y_front = y_front[order]

    # Best frontier region: closest to origin in normalized latency/area space.
    best_idx = np.argmin(x_front + y_front)
    x_best = x_front[best_idx]
    y_best = y_front[best_idx]

    d_perp = np.zeros(n, dtype=float)
    d_best = np.zeros(n, dtype=float)

    for i in range(n):
        px, py = perf_n[i], area_n[i]

        # Distance to the best frontier point.
        d_best[i] = np.sqrt((px - x_best) ** 2 + (py - y_best) ** 2)

        # Distance to the Pareto-front polyline.
        if len(x_front) == 1:
            min_seg_dist = np.sqrt((px - x_front[0]) ** 2 + (py - y_front[0]) ** 2)
        else:
            min_seg_dist = np.inf
            for k in range(len(x_front) - 1):
                x1, y1 = x_front[k], y_front[k]
                x2, y2 = x_front[k + 1], y_front[k + 1]

                vx, vy = x2 - x1, y2 - y1
                wx, wy = px - x1, py - y1
                seg_len2 = vx * vx + vy * vy + eps

                t = (vx * wx + vy * wy) / seg_len2
                t = np.clip(t, 0.0, 1.0)

                proj_x = x1 + t * vx
                proj_y = y1 + t * vy

                dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
                min_seg_dist = min(min_seg_dist, dist)

        d_perp[i] = min_seg_dist

    # Combine distance-to-front and distance-to-best-region.
    alpha = 1.0
    beta = 1.0
    d_total = alpha * d_perp + beta * d_best

    d_min = d_total.min()
    d_max = d_total.max()

    if abs(d_max - d_min) < eps:
        weights = np.ones(n, dtype=float)
    else:
        d_norm = (d_total - d_min) / (d_max - d_min + eps)
        weights = w_min_valid + (1.0 - w_min_valid) * (1.0 - d_norm) ** gamma

    # Pareto points should receive maximum supervision weight.
    weights[pareto_mask] = 1.0

    return weights, pareto_mask


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

def print_weight_stats(kernel_csv_name, w):
    w = np.asarray(w, dtype=float)
    if len(w) == 0:
        print(f"\n[Weight stats] {kernel_csv_name}: no valid weights")
        return

    print(f"\n[Weight stats] {kernel_csv_name}")
    print(f"  n = {len(w)}")
    print(f"  min / max : {w.min():.3f} / {w.max():.3f}")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        print(f"  p{int(q * 100):02d}: {np.quantile(w, q):.3f}")


def _normalize_clock_for_grouping(series):
    """
    Convert Clock_Period_nsec to numeric when possible, while preserving rows.
    This makes 10 and 10.00 group together.
    """
    return pd.to_numeric(series, errors="coerce")


# -----------------------------------------------------------------------------
# Main preprocessing
# -----------------------------------------------------------------------------

def preprocess_csv(
    csv_file,
    out_dir="/home/ubuntu/Data4LLMPrompting/preprocessed_CSVS",
    w_min_valid=0.1,
    gamma=2.0,
    invalid_latency_threshold=100000,
):
    """
    Preprocess one raw HLS design-space CSV.

    Differences from the old single-target version:
      - DOES NOT filter Device == xczu7ev-ffvc1156-2-e.
      - DOES NOT filter Clock_Period_nsec == 10.00.
      - DOES NOT drop Device / Clock_Period_nsec.
      - STILL removes invalid designs from the output.
      - Computes Pareto weights per (Device, Clock_Period_nsec).
    """
    df_raw = pd.read_csv(csv_file)
    input_rows = len(df_raw)

    required_cols = [
        "Version",
        "Device",
        "Clock_Period_nsec",
        "Latency_msec",
        "Synthesis_Time_sec",
        "BRAM_Utilization_percentage",
        "DSP_Utilization_percentage",
        "FF_Utilization_percentage",
        "LUT_Utilization_percentage",
    ]

    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise RuntimeError(f"{csv_file} is missing required columns: {missing}")

    # Work on a full copy. Do not target-filter.
    df = df_raw.copy(deep=True)

    # Drop columns that are entirely NDIR, but never drop metadata/QoR columns.
    always_keep = set(required_cols)
    nonempty_cols = []
    for col in df.columns:
        col_as_str = df[col].astype(str).str.strip()
        all_ndir = col_as_str.eq("NDIR").all()
        if col in always_keep or not all_ndir:
            nonempty_cols.append(col)

    df = df.loc[:, nonempty_cols].copy(deep=True)

    utilization_cols = [
        "BRAM_Utilization_percentage",
        "DSP_Utilization_percentage",
        "FF_Utilization_percentage",
        "LUT_Utilization_percentage",
    ]

    # Numeric cleanup for QoR columns.
    for col in ["Clock_Period_nsec", "Latency_msec", "Synthesis_Time_sec"] + utilization_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mark invalid rows using the same intent as the old script.
    overutil_mask = (df[utilization_cols] >= 100).any(axis=1)
    too_slow_mask = df["Latency_msec"] > invalid_latency_threshold
    nonpositive_or_nan_latency_mask = df["Latency_msec"].isna() | (df["Latency_msec"] <= 0)
    missing_target_mask = df["Device"].isna() | df["Clock_Period_nsec"].isna()

    invalid_mask = (
        overutil_mask
        | too_slow_mask
        | nonpositive_or_nan_latency_mask
        | missing_target_mask
    )

    # Keep only valid designs in the preprocessed CSV.
    df_valid = df.loc[~invalid_mask].copy(deep=True)

    if df_valid.empty:
        print(f"[preprocess_csv] No valid designs in {csv_file}; no output written.")
        return None

    # Convert 0% -> 1% only for valid designs, preserving old behavior.
    df_valid.loc[:, utilization_cols] = df_valid[utilization_cols].replace(0, 1)

    # Area proxy = average resource utilization percentage.
    df_valid["Area"] = df_valid[utilization_cols].sum(axis=1) / 4.0

    # Initialize labels.
    df_valid["Weight"] = float(w_min_valid)
    df_valid["is_pareto"] = False

    # Normalize 10 and 10.00 into the same group.
    df_valid["_Clock_Group_Key"] = _normalize_clock_for_grouping(df_valid["Clock_Period_nsec"])

    group_cols = ["Device", "_Clock_Group_Key"]

    for (device, clock), group in df_valid.groupby(group_cols, dropna=False, sort=True):
        idx = group.index
        perf = group["Latency_msec"].to_numpy(dtype=float)
        area = group["Area"].to_numpy(dtype=float)

        weights, pareto_mask = compute_pareto_weights(
            perf=perf,
            area=area,
            w_min_valid=w_min_valid,
            gamma=gamma,
        )

        df_valid.loc[idx, "Weight"] = weights
        df_valid.loc[idx, "is_pareto"] = pareto_mask

        print(
            f"  [bucket] Device={device}, Clock_Period_nsec={clock}: "
            f"valid={len(idx)}, pareto={int(np.sum(pareto_mask))}"
        )

    # Remove internal grouping helper from final CSV.
    df_valid.drop(columns=["_Clock_Group_Key"], inplace=True)

    # Keep original-ish column order: original kept columns + new computed columns at end.
    computed_cols = ["Area", "Weight", "is_pareto"]
    ordered_cols = [c for c in nonempty_cols if c in df_valid.columns]
    ordered_cols += [c for c in computed_cols if c in df_valid.columns]
    df_valid = df_valid.loc[:, ordered_cols]

    filename = os.path.basename(csv_file)
    out_path = os.path.join(out_dir, f"preprocessed-{filename}")
    os.makedirs(out_dir, exist_ok=True)
    df_valid.to_csv(out_path, index=False)

    print(f"[preprocess_csv] Saved preprocessed CSV to: {out_path}")
    print(f"  input rows        : {input_rows}")
    print(f"  output valid rows : {len(df_valid)}")
    print(f"  removed invalid   : {input_rows - len(df_valid)}")
    print(f"  kept targets      : {df_valid[['Device', 'Clock_Period_nsec']].drop_duplicates().shape[0]}")

    print_weight_stats(os.path.basename(csv_file), df_valid["Weight"].to_numpy())

    return out_path


# -----------------------------------------------------------------------------
# Optional plotting helpers
# -----------------------------------------------------------------------------

def plot_pareto_for_kernel(
    csv_file,
    output_dir="/home/elvouvali/Data4LLMPrompting/pareto_per_kernel",
):
    """
    Plot Pareto points for a preprocessed CSV.

    If the CSV contains multiple devices/clocks, this function creates one plot
    per (Device, Clock_Period_nsec) bucket to avoid mixing target domains.
    """
    df = pd.read_csv(csv_file)

    required = {"Device", "Clock_Period_nsec", "Latency_msec", "Area", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{csv_file} is missing required plot columns: {sorted(missing)}")

    if df.empty:
        print(f"No points in {csv_file}.")
        return

    base = os.path.splitext(os.path.basename(csv_file))[0]
    if base.startswith("preprocessed-"):
        base = base[len("preprocessed-"):]

    for (device, clock), g in df.groupby(["Device", "Clock_Period_nsec"], dropna=False, sort=True):
        if g.empty:
            continue

        perf = g["Latency_msec"].to_numpy(dtype=float)
        area = g["Area"].to_numpy(dtype=float)
        weights = g["Weight"].to_numpy(dtype=float)

        pareto_mask = pareto_front_2d(perf, area)
        perf_p = perf[pareto_mask]
        area_p = area[pareto_mask]

        order = np.argsort(perf_p)
        perf_p = perf_p[order]
        area_p = area_p[order]

        plt.figure(figsize=(8, 6))

        sc = plt.scatter(
            perf,
            area,
            c=weights,
            cmap="viridis",
            s=20,
            alpha=0.7,
            label="Design points (colored by Weight)",
        )

        plt.scatter(
            perf_p,
            area_p,
            color="red",
            edgecolors="black",
            s=40,
            label="Pareto frontier points",
            zorder=3,
        )
        plt.plot(perf_p, area_p, color="red", linewidth=2, alpha=0.8, zorder=2)

        cbar = plt.colorbar(sc)
        cbar.set_label("Weight (higher = closer to Pareto)", rotation=90)

        plt.xlabel("Latency_msec")
        plt.ylabel("Area = (BRAM% + DSP% + FF% + LUT%) / 4")
        plt.title(f"{base}\nDevice={device}, Clock={clock} ns")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()

        safe_device = str(device).replace("/", "_").replace(" ", "_")
        safe_clock = str(clock).replace(".", "p")
        out_path = os.path.join(
            output_dir,
            f"pareto-{base}__{safe_device}__clk{safe_clock}ns.png",
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved Pareto plot to: {out_path}")


# -----------------------------------------------------------------------------
# CLI / batch entry point
# -----------------------------------------------------------------------------

def main():
    in_dir = "/home/elvouvali/Data4LLMPrompting/CSVS"
    out_dir = "/home/elvouvali/Data4LLMPrompting/preprocessed_CSVS"

    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    print(f"Found {len(csv_files)} CSV files in {in_dir}")

    ok = 0
    failed = 0

    for i, csv_file in enumerate(csv_files, start=1):
        name = os.path.basename(csv_file)
        print(f"\n[{i}/{len(csv_files)}] Processing {name}")
        try:
            out_path = preprocess_csv(csv_file, out_dir=out_dir)
            if out_path is not None:
                ok += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")

    print("\nDone.")
    print(f"Processed successfully: {ok}")
    print(f"Failed/skipped: {failed}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
