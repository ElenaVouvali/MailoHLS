#!/usr/bin/env python3
"""
Diagnose Stage-2 structural-memory geometry.

Measures, per kernel:
  1) pairwise cosine similarity among active STATIC slots,
  2) raw / centered effective rank,
  3) centered-energy fraction (how much action-specific variation exists),
  4) stable rank and 90%-energy rank,
  5) aligned STATIC-vs-SHUFFLE cosine / L2 displacement,
  6) fraction of active slots actually changed by the shuffle.

This script does not load the LLM or use a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch


EPS = 1e-12


def load_pack(path: Path) -> dict:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def scalar_stats(values: torch.Tensor) -> Dict[str, Optional[float]]:
    values = values.detach().double().flatten()

    if values.numel() == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "std": float(
            values.std(unbiased=False).item()
        ),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def pairwise_cosine_stats(x: torch.Tensor) -> dict:
    """
    Cosine similarity between every distinct pair of active action vectors.
    """
    x = x.double()
    norms = torch.linalg.vector_norm(x, dim=1)
    valid = norms > EPS

    zero_norm_count = int((~valid).sum().item())
    x = x[valid]

    if x.shape[0] < 2:
        return {
            "pair_count": 0,
            "zero_norm_vectors": zero_norm_count,
            **{
                f"cosine_{k}": v
                for k, v in scalar_stats(
                    torch.empty(0)
                ).items()
            },
        }

    x = x / torch.linalg.vector_norm(
        x, dim=1, keepdim=True
    ).clamp_min(EPS)

    similarity = x @ x.T

    row, col = torch.triu_indices(
        similarity.shape[0],
        similarity.shape[1],
        offset=1,
    )
    values = similarity[row, col]

    stats = scalar_stats(values)

    return {
        "pair_count": int(values.numel()),
        "zero_norm_vectors": zero_norm_count,
        **{
            f"cosine_{key}": value
            for key, value in stats.items()
        },
    }


def spectral_stats(
    x: torch.Tensor,
    *,
    center: bool,
) -> dict:
    """
    Roy-Vetterli effective rank:
        erank(X) = exp(H(p))
        p_i = sigma_i / sum_j sigma_j

    We report it on both raw X and centered X.

    Centered X is especially important here because it measures the
    dimensionality of ACTION-TO-ACTION differences after removing the
    common kernel-level embedding direction.
    """
    x = x.double()

    if center:
        x = x - x.mean(dim=0, keepdim=True)
        theoretical_max_rank = min(
            max(x.shape[0] - 1, 0),
            x.shape[1],
        )
    else:
        theoretical_max_rank = min(
            x.shape[0],
            x.shape[1],
        )

    if theoretical_max_rank <= 0:
        return {
            "effective_rank": None,
            "normalized_effective_rank": None,
            "stable_rank": None,
            "rank90": None,
            "numerical_rank": 0,
            "max_rank": int(theoretical_max_rank),
        }

    singular_values = torch.linalg.svdvals(x)

    if (
        singular_values.numel() == 0
        or singular_values[0].item() <= EPS
    ):
        return {
            "effective_rank": 0.0,
            "normalized_effective_rank": 0.0,
            "stable_rank": 0.0,
            "rank90": 0,
            "numerical_rank": 0,
            "max_rank": int(theoretical_max_rank),
        }

    tolerance = (
        torch.finfo(singular_values.dtype).eps
        * max(x.shape)
        * singular_values[0]
    )

    singular_values = singular_values[
        singular_values > tolerance
    ]

    if singular_values.numel() == 0:
        return {
            "effective_rank": 0.0,
            "normalized_effective_rank": 0.0,
            "stable_rank": 0.0,
            "rank90": 0,
            "numerical_rank": 0,
            "max_rank": int(theoretical_max_rank),
        }

    probabilities = (
        singular_values / singular_values.sum()
    )

    entropy = -(
        probabilities
        * probabilities.clamp_min(EPS).log()
    ).sum()

    effective_rank = float(
        torch.exp(entropy).item()
    )

    stable_rank = float(
        (
            singular_values.square().sum()
            / singular_values[0].square()
        ).item()
    )

    energy = singular_values.square()
    cumulative = (
        torch.cumsum(energy, dim=0)
        / energy.sum().clamp_min(EPS)
    )
    rank90 = int(
        (cumulative < 0.90).sum().item() + 1
    )

    return {
        "effective_rank": effective_rank,
        "normalized_effective_rank": (
            effective_rank
            / max(float(theoretical_max_rank), 1.0)
        ),
        "stable_rank": stable_rank,
        "rank90": rank90,
        "numerical_rank": int(
            singular_values.numel()
        ),
        "max_rank": int(theoretical_max_rank),
    }


def analyze_kernel(
    kernel: str,
    static_path: Path,
    shuffle_path: Path,
) -> dict:
    static = load_pack(static_path)
    shuffled = load_pack(shuffle_path)

    static_x = static["node_embs"].float()
    shuffled_x = shuffled["node_embs"].float()

    static_mask = (
        static["node_embs_mask"]
        .bool()
        .view(-1)
    )
    shuffle_mask = (
        shuffled["node_embs_mask"]
        .bool()
        .view(-1)
    )

    if static_x.shape != shuffled_x.shape:
        raise ValueError(
            f"{kernel}: STATIC/SHUFFLE shape mismatch"
        )

    if not torch.equal(
        static_mask,
        shuffle_mask,
    ):
        raise ValueError(
            f"{kernel}: STATIC/SHUFFLE mask mismatch"
        )

    active_idx = torch.where(
        static_mask
    )[0]

    x = static_x[active_idx]
    y = shuffled_x[active_idx]

    n_active = int(x.shape[0])

    pairwise = pairwise_cosine_stats(x)

    raw_spectral = spectral_stats(
        x,
        center=False,
    )
    centered_spectral = spectral_stats(
        x,
        center=True,
    )

    if n_active:
        centered = (
            x.double()
            - x.double().mean(
                dim=0,
                keepdim=True,
            )
        )

        raw_energy = float(
            x.double().square().sum().item()
        )
        centered_energy = float(
            centered.square().sum().item()
        )

        centered_energy_fraction = (
            centered_energy
            / max(raw_energy, EPS)
        )

        feature_variance_mean = float(
            x.double()
            .var(dim=0, unbiased=False)
            .mean()
            .item()
        )

        norms = torch.linalg.vector_norm(
            x.double(),
            dim=1,
        )

        norm_mean = float(
            norms.mean().item()
        )
        norm_std = float(
            norms.std(
                unbiased=False
            ).item()
        )

        norm_cv = (
            norm_std
            / max(norm_mean, EPS)
        )

        static_norms = torch.linalg.vector_norm(
            x.double(),
            dim=1,
        )
        shuffle_norms = torch.linalg.vector_norm(
            y.double(),
            dim=1,
        )

        valid = (
            (static_norms > EPS)
            & (shuffle_norms > EPS)
        )

        if valid.any():
            aligned_cos = torch.nn.functional.cosine_similarity(
                x[valid].double(),
                y[valid].double(),
                dim=1,
            )
        else:
            aligned_cos = torch.empty(
                0,
                dtype=torch.double,
            )

        displacement = torch.linalg.vector_norm(
            x.double() - y.double(),
            dim=1,
        )

        relative_displacement = (
            displacement
            / static_norms.clamp_min(EPS)
        )

        unchanged = (
            x == y
        ).all(dim=1)

        changed_fraction = float(
            (~unchanged)
            .double()
            .mean()
            .item()
        )
    else:
        centered_energy_fraction = None
        feature_variance_mean = None
        norm_mean = None
        norm_std = None
        norm_cv = None
        aligned_cos = torch.empty(0)
        displacement = torch.empty(0)
        relative_displacement = torch.empty(0)
        changed_fraction = None

    category_counts = {}

    slot_cats = static.get(
        "slot_cats",
        None,
    )

    if slot_cats is not None:
        slot_cats = torch.as_tensor(
            slot_cats
        ).view(-1)

        category_counts = {
            str(int(cat)): int(count)
            for cat, count in Counter(
                int(v)
                for v in slot_cats[
                    active_idx
                ].tolist()
            ).items()
        }

    return {
        "kernel": kernel,
        "active_slots": n_active,
        "active_indices_zero_based": (
            active_idx.tolist()
        ),
        "category_counts": category_counts,

        "pairwise_static": pairwise,

        "raw_spectral": raw_spectral,
        "centered_spectral": (
            centered_spectral
        ),

        "centered_energy_fraction": (
            centered_energy_fraction
        ),
        "feature_variance_mean": (
            feature_variance_mean
        ),

        "vector_norm_mean": norm_mean,
        "vector_norm_std": norm_std,
        "vector_norm_cv": norm_cv,

        "static_vs_shuffle": {
            **{
                f"aligned_cosine_{key}": value
                for key, value in scalar_stats(
                    aligned_cos
                ).items()
            },
            **{
                f"l2_displacement_{key}": value
                for key, value in scalar_stats(
                    displacement
                ).items()
            },
            **{
                f"relative_l2_{key}": value
                for key, value in scalar_stats(
                    relative_displacement
                ).items()
            },
            "changed_fraction": (
                changed_fraction
            ),
        },
    }


def mean_defined(values: List[Optional[float]]):
    values = [
        float(v)
        for v in values
        if v is not None
        and math.isfinite(float(v))
    ]
    return (
        sum(values) / len(values)
        if values
        else None
    )


def median_defined(values: List[Optional[float]]):
    values = sorted(
        float(v)
        for v in values
        if v is not None
        and math.isfinite(float(v))
    )

    if not values:
        return None

    middle = len(values) // 2

    if len(values) % 2:
        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


def build_summary(records: List[dict]) -> dict:
    pair_cos = [
        r["pairwise_static"][
            "cosine_mean"
        ]
        for r in records
    ]

    centered_fraction = [
        r["centered_energy_fraction"]
        for r in records
    ]

    normalized_erank = [
        r["centered_spectral"][
            "normalized_effective_rank"
        ]
        for r in records
    ]

    shuffle_cos = [
        r["static_vs_shuffle"][
            "aligned_cosine_mean"
        ]
        for r in records
    ]

    shuffle_rel_l2 = [
        r["static_vs_shuffle"][
            "relative_l2_mean"
        ]
        for r in records
    ]

    return {
        "kernel_count": len(records),
        "active_slots_total": sum(
            r["active_slots"]
            for r in records
        ),

        "pairwise_cosine_mean_of_kernels":
            mean_defined(pair_cos),

        "pairwise_cosine_median_of_kernels":
            median_defined(pair_cos),

        "fraction_kernels_pair_cos_ge_0_90":
            (
                sum(
                    v is not None
                    and v >= 0.90
                    for v in pair_cos
                )
                / max(
                    sum(
                        v is not None
                        for v in pair_cos
                    ),
                    1,
                )
            ),

        "centered_energy_fraction_mean":
            mean_defined(
                centered_fraction
            ),

        "centered_energy_fraction_median":
            median_defined(
                centered_fraction
            ),

        "centered_normalized_effective_rank_mean":
            mean_defined(
                normalized_erank
            ),

        "centered_normalized_effective_rank_median":
            median_defined(
                normalized_erank
            ),

        "static_shuffle_aligned_cosine_mean":
            mean_defined(
                shuffle_cos
            ),

        "static_shuffle_relative_l2_mean":
            mean_defined(
                shuffle_rel_l2
            ),
    }


def fmt(value):
    if value is None:
        return "NA"
    return f"{float(value):.5f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--static_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--shuffle_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out_json",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    static_files = {
        p.name: p
        for p in args.static_dir.glob(
            "*.memory.pt"
        )
    }

    shuffle_files = {
        p.name: p
        for p in args.shuffle_dir.glob(
            "*.memory.pt"
        )
    }

    if not static_files:
        raise ValueError(
            f"No STATIC memory files in "
            f"{args.static_dir}"
        )

    if set(static_files) != set(
        shuffle_files
    ):
        raise ValueError(
            "STATIC and SHUFFLE banks "
            "contain different kernels"
        )

    records = []

    for filename in sorted(
        static_files
    ):
        kernel = filename[
            :-len(".memory.pt")
        ]

        record = analyze_kernel(
            kernel,
            static_files[filename],
            shuffle_files[filename],
        )
        records.append(record)

        print(
            "[MEM-DIAG] "
            f"{kernel} "
            f"n={record['active_slots']} "
            f"pair_cos="
            f"{fmt(record['pairwise_static']['cosine_mean'])} "
            f"center_energy="
            f"{fmt(record['centered_energy_fraction'])} "
            f"center_erank="
            f"{fmt(record['centered_spectral']['effective_rank'])}"
            f"/{record['centered_spectral']['max_rank']} "
            f"center_erank_norm="
            f"{fmt(record['centered_spectral']['normalized_effective_rank'])} "
            f"shuffle_cos="
            f"{fmt(record['static_vs_shuffle']['aligned_cosine_mean'])} "
            f"shuffle_rel_l2="
            f"{fmt(record['static_vs_shuffle']['relative_l2_mean'])} "
            f"changed="
            f"{fmt(record['static_vs_shuffle']['changed_fraction'])}"
        )

    summary = build_summary(records)

    payload = {
        "schema":
            "mailohls-structural-memory-diagnostic-v1",
        "static_dir":
            str(args.static_dir.resolve()),
        "shuffle_dir":
            str(args.shuffle_dir.resolve()),
        "summary": summary,
        "kernels": records,
    }

    args.out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n[MEM-SUMMARY]")
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"\n[MEM-SUMMARY] wrote "
        f"{args.out_json}"
    )


if __name__ == "__main__":
    main()