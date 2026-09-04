#!/usr/bin/env python3
"""Plot objective-conditioned ASPLOS results directly from the JSONL contract."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

KERNELS = ["Kalman", "GAN", "trmm-opt", "covariance", "jacobi-2d", "syr2k",
           "chstone-aes", "chstone-jpeg", "rosetta-3d-rendering"]
OBJECTIVES = ["LATENCY", "AREA", "ADP"]
COLORS = {"LATENCY": "#2878B5", "AREA": "#59A14F", "ADP": "#E07B39"}
RESOURCES = ["bram", "dsp", "ff", "lut"]
BUDGETS = {"bram": 624, "dsp": 1728, "ff": 460800, "lut": 230400}

def arguments():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    # The corrected contract is the canonical no-argument input.  The old
    # ASPLOS_experiments.jsonl remains available explicitly via --jsonl when
    # reproducing the pre-correction figures.
    p.add_argument("--jsonl", type=Path, default=repo / "SALT_experiments.jsonl")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "final")
    return p.parse_args()

def load(path):
    with path.open() as src:
        rows = [json.loads(line) for line in src if line.strip()]
    baseline, result = {}, {}
    for row in rows:
        kernel = row["kernel"]
        if kernel not in KERNELS:
            continue
        if row["stage"] == "baseline":
            baseline[kernel] = row
        elif row["stage"] == "stage-3" and row["objective"] == "LATENCY":
            result[kernel, "LATENCY"] = row
        elif row["stage"] == "predicted-directives" and row["objective"] in ("AREA", "ADP"):
            result[kernel, row["objective"]] = row
    expected = {(k, o) for k in KERNELS for o in OBJECTIVES}
    if set(KERNELS) - set(baseline) or expected - set(result):
        raise ValueError("JSONL lacks one or more baseline/objective rows")
    return baseline, result

def metrics(baseline, result):
    speedup = np.zeros((len(KERNELS), 3))
    utilization = np.zeros((len(KERNELS), 3, 4))
    for ki, kernel in enumerate(KERNELS):
        base_cycles = baseline[kernel]["latency"]
        for oi, objective in enumerate(OBJECTIVES):
            row = result[kernel, objective]
            speedup[ki, oi] = base_cycles / row["latency"]
            for ri, resource in enumerate(RESOURCES):
                utilization[ki, oi, ri] = 100 * row["resources"][resource] / BUDGETS[resource]
    return speedup, utilization

def overview(speedup, utilization, out):
    infeasible = utilization.max(axis=2) > 100
    fig = plt.figure(figsize=(14.2, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 1.3))
    ax = fig.add_subplot(grid[0, :]); x = np.arange(len(KERNELS)); width = .25
    for oi, objective in enumerate(OBJECTIVES):
        bars = ax.bar(x + (oi-1)*width, speedup[:, oi], width, color=COLORS[objective], label=objective)
        for bar, value in zip(bars, speedup[:, oi]):
            ax.annotate(f"{value:.2g}×", (bar.get_x()+bar.get_width()/2, value), xytext=(0,3),
                        textcoords="offset points", ha="center", fontsize=7, rotation=90)
        for ki, bar in enumerate(bars):
            if infeasible[ki, oi]:
                # Keep the infeasibility marker below the bar so it does not
                # obscure the speedup annotation on the logarithmic axis.
                ax.scatter(bar.get_x()+bar.get_width()/2, 0.55, marker="x",
                           s=85, color="#d62728", linewidth=2.2, zorder=5)
    ax.axhline(1, color="#333", ls="--", lw=1); ax.set_yscale("log")
    ax.set_ylabel("Speedup over baseline (×, log scale)")
    ax.set_xticks(x, KERNELS, rotation=25, ha="right"); ax.grid(axis="y", alpha=.25, which="both")
    ax.legend(ncol=3, frameon=False); ax.set_title("Objective-conditioned speedup and FPGA utilization")
    cmap = ListedColormap(["#f7fbff","#c6dbef","#6baed6","#2171b5","#fdae6b","#de2d26"])
    norm = BoundaryNorm([0,1,5,10,25,50,101], cmap.N); image = None
    for oi, objective in enumerate(OBJECTIVES):
        heat = utilization[:, oi, :]; hax = fig.add_subplot(grid[1, oi])
        image = hax.imshow(heat, aspect="auto", cmap=cmap, norm=norm)
        hax.set_title(objective, color=COLORS[objective], fontweight="bold")
        hax.set_xticks(range(4), [r.upper() for r in RESOURCES])
        hax.set_yticks(range(len(KERNELS)), KERNELS if oi == 0 else [])
        for row in range(heat.shape[0]):
            for col in range(4):
                v = heat[row,col]; hax.text(col,row,f"{v:.1f}",ha="center",va="center",fontsize=7,
                                             color="white" if v >= 25 else "#111")
                if v > 100:
                    hax.scatter(col, row, marker="x", s=120, color="#d62728",
                                linewidth=2.4, zorder=5)
    cb = fig.colorbar(image, ax=fig.axes[1:], location="bottom", shrink=.6, pad=.08)
    cb.set_label("Per-resource utilization (% of device budget)")
    for suffix in ("pdf","png"):
        fig.savefig(out/f"objective_speedup_resource_overview.{suffix}", bbox_inches="tight",
                    **({"dpi":300} if suffix=="png" else {}))
    plt.close(fig)

def tradeoff(speedup, utilization, out):
    fig, ax = plt.subplots(figsize=(9.2,6.5), constrained_layout=True)
    bottleneck = utilization.max(axis=2); infeasible = bottleneck > 100
    markers = list("os^DPXv<>")
    for ki in range(len(KERNELS)):
        ax.plot(bottleneck[ki,:], speedup[ki,:], color="#999", alpha=.45, lw=.8, zorder=1)
    for oi, objective in enumerate(OBJECTIVES):
        for ki, kernel in enumerate(KERNELS):
            ax.scatter(bottleneck[ki,oi], speedup[ki,oi], s=70, marker=markers[ki],
                       color=COLORS[objective], edgecolor="white", linewidth=.7, zorder=2)
        ax.scatter([],[],s=70,color=COLORS[objective],label=objective)
    for ki, oi in np.argwhere(infeasible):
        ax.scatter(bottleneck[ki,oi], speedup[ki,oi], s=150, marker="x",
                   color="#d62728", linewidth=2.5, zorder=5)
    ax.scatter([], [], s=100, marker="x", color="#d62728", linewidth=2.2,
               label="Infeasible (>100% any resource)")
    for ki, kernel in enumerate(KERNELS):
        label_x = bottleneck[ki,:].mean()
        label_y = np.exp(np.log(speedup[ki,:]).mean())
        ax.annotate(kernel, (label_x,label_y), xytext=(4,3), textcoords="offset points", fontsize=7)
    ax.axhline(1,color="#333",ls="--",lw=1); ax.set_yscale("log")
    ax.set_xlabel("Bottleneck utilization (max of BRAM/DSP/FF/LUT, %)")
    ax.set_ylabel("Speedup over baseline (×, log scale)"); ax.grid(alpha=.25,which="both")
    ax.legend(frameon=False,ncol=3); ax.set_title("Performance–resource trade-off")
    for suffix in ("pdf","png"):
        fig.savefig(out/f"objective_speedup_resource_tradeoff.{suffix}", bbox_inches="tight",
                    **({"dpi":300} if suffix=="png" else {}))
    plt.close(fig)

def main():
    args=arguments(); args.output_dir.mkdir(parents=True,exist_ok=True)
    speedup,utilization=metrics(*load(args.jsonl))
    overview(speedup,utilization,args.output_dir); tradeoff(speedup,utilization,args.output_dir)
    print(f"Generated plots in {args.output_dir}")

if __name__ == "__main__": main()
