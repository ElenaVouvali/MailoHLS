#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Fixed ASPLOS evaluation setup
# ============================================================

KERNELS = [
    "covariance",
    "jacobi-2d",
    "syr2k",
    "trmm-opt",
    "GAN",
    "Kalman",
]

OBJECTIVES = [
    "Latency request",
    "ADP request",
    "Area request",
]

x = np.arange(len(KERNELS))


# ============================================================
# REPLACE THESE MOCK VALUES WITH YOUR FINAL MEASUREMENTS
# ============================================================

# ------------------------------------------------------------
# Figure 1: Speedup
#
# Recommended definition:
#   speedup = baseline_latency / SALT-HLS_latency
#
# > 1 means SALT-HLS is faster than the baseline.
# ------------------------------------------------------------

speedup_latency = np.array([
    6.8,
    4.9,
    8.1,
    5.7,
    9.4,
    7.2,
])

speedup_adp = np.array([
    5.6,
    4.3,
    6.9,
    5.2,
    7.8,
    6.4,
])

speedup_area = np.array([
    3.9,
    3.5,
    4.7,
    4.1,
    5.0,
    4.4,
])


# ------------------------------------------------------------
# Figure 2: Resource utilization
#
# Put here the resource metric you decide to report.
#
# For example:
#   arithmetic mean of BRAM/DSP/FF/LUT utilization (%)
#
# OR replace this with effective-area utilization if that is
# the metric used consistently in the paper.
# ------------------------------------------------------------

resource_latency = np.array([
    63.0,
    71.0,
    68.0,
    74.0,
    79.0,
    66.0,
])

resource_adp = np.array([
    52.0,
    60.0,
    57.0,
    61.0,
    66.0,
    55.0,
])

resource_area = np.array([
    38.0,
    46.0,
    41.0,
    45.0,
    49.0,
    42.0,
])


# ============================================================
# Shared plotting function
# ============================================================

def make_objective_plot(
    latency_values,
    adp_values,
    area_values,
    ylabel,
    title,
    output_stem,
    reference_value=None,
    reference_label=None,
    ylim=None,
):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.plot(
        x,
        latency_values,
        marker="o",
        linewidth=2.0,
        markersize=7,
        label="Latency request",
    )

    ax.plot(
        x,
        adp_values,
        marker="s",
        linewidth=2.0,
        markersize=7,
        label="ADP request",
    )

    ax.plot(
        x,
        area_values,
        marker="^",
        linewidth=2.0,
        markersize=8,
        label="Area request",
    )

    if reference_value is not None:
        ax.axhline(
            reference_value,
            linestyle="--",
            linewidth=1.6,
            label=reference_label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        KERNELS,
        rotation=20,
        ha="right",
    )

    ax.set_xlabel("Evaluation kernel")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.6,
        alpha=0.30,
    )

    ax.legend(
        loc="best",
        frameon=True,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        f"{output_stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Figure 7.1a — synthesized speedup
# ============================================================

make_objective_plot(
    latency_values=speedup_latency,
    adp_values=speedup_adp,
    area_values=speedup_area,
    ylabel="Synthesized speedup (×)",
    title="End-to-end synthesized performance across objective requests",
    output_stem="fig_7_1_speedup",
    reference_value=1.0,
    reference_label="Baseline",
    ylim=(0.0, None),
)


# ============================================================
# Figure 7.1b — synthesized resource utilization
# ============================================================

make_objective_plot(
    latency_values=resource_latency,
    adp_values=resource_adp,
    area_values=resource_area,
    ylabel="Resource utilization (%)",
    title="End-to-end synthesized resource utilization across objective requests",
    output_stem="fig_7_1_resource_utilization",
    reference_value=None,
    reference_label=None,
    ylim=(0.0, 100.0),
)


print("Generated:")
print("  fig_7_1_speedup.pdf")
print("  fig_7_1_resource_utilization.pdf")
