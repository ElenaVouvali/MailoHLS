#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from diagnose_structural_memory import (
    EPS,
    pairwise_cosine_stats,
    spectral_stats,
    mean_defined,
    median_defined,
)


LAYERS = (
    "conv_1",
    "conv_2",
    "conv_3",
    "conv_4",
    "jkn",
)


def load_active_vectors(
    path: Path,
):
    pack = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    x = (
        pack["node_embs"]
        .float()
    )

    mask = (
        pack["node_embs_mask"]
        .bool()
        .view(-1)
    )

    if x.ndim != 2:
        raise ValueError(
            f"{path}: node_embs must be [S,D]"
        )

    if mask.numel() != x.shape[0]:
        raise ValueError(
            f"{path}: mask/embedding mismatch"
        )

    return (
        x[mask],
        mask,
        pack,
    )


def analyze_layer(
    layer_dir: Path,
):
    files = sorted(
        layer_dir.glob(
            "*.memory.pt"
        )
    )

    if not files:
        raise ValueError(
            f"No memory files in {layer_dir}"
        )

    records = []

    for path in files:

        kernel = path.name[
            :-len(".memory.pt")
        ]

        x, _, _ = (
            load_active_vectors(path)
        )

        pair = (
            pairwise_cosine_stats(x)
        )

        raw = spectral_stats(
            x,
            center=False,
        )

        centered = spectral_stats(
            x,
            center=True,
        )

        if x.shape[0] > 0:
            xd = x.double()

            xc = (
                xd
                - xd.mean(
                    dim=0,
                    keepdim=True,
                )
            )

            raw_energy = float(
                xd.square()
                .sum()
                .item()
            )

            centered_energy = float(
                xc.square()
                .sum()
                .item()
            )

            centered_fraction = (
                centered_energy
                / max(
                    raw_energy,
                    EPS,
                )
            )
        else:
            centered_fraction = None

        records.append(
            {
                "kernel": kernel,
                "active_slots":
                    int(x.shape[0]),

                "pairwise_cosine_mean":
                    pair[
                        "cosine_mean"
                    ],

                "centered_energy_fraction":
                    centered_fraction,

                "raw_effective_rank":
                    raw[
                        "effective_rank"
                    ],

                "raw_rank90":
                    raw[
                        "rank90"
                    ],

                "centered_effective_rank":
                    centered[
                        "effective_rank"
                    ],

                "centered_normalized_effective_rank":
                    centered[
                        "normalized_effective_rank"
                    ],

                "centered_rank90":
                    centered[
                        "rank90"
                    ],

                "centered_max_rank":
                    centered[
                        "max_rank"
                    ],
            }
        )

    def values(key):
        return [
            r[key]
            for r in records
            if r[key] is not None
        ]

    summary = {
        "kernel_count":
            len(records),

        "active_slots_total":
            sum(
                r["active_slots"]
                for r in records
            ),

        "pairwise_cosine_mean":
            mean_defined(
                values(
                    "pairwise_cosine_mean"
                )
            ),

        "pairwise_cosine_median":
            median_defined(
                values(
                    "pairwise_cosine_mean"
                )
            ),

        "centered_energy_mean":
            mean_defined(
                values(
                    "centered_energy_fraction"
                )
            ),

        "centered_energy_median":
            median_defined(
                values(
                    "centered_energy_fraction"
                )
            ),

        "centered_erank_mean":
            mean_defined(
                values(
                    "centered_effective_rank"
                )
            ),

        "centered_erank_norm_mean":
            mean_defined(
                values(
                    "centered_normalized_effective_rank"
                )
            ),

        "centered_rank90_mean":
            mean_defined(
                values(
                    "centered_rank90"
                )
            ),

        "raw_rank90_mean":
            mean_defined(
                values(
                    "raw_rank90"
                )
            ),

        "fraction_pair_cos_ge_090":
            (
                sum(
                    value >= 0.90
                    for value
                    in values(
                        "pairwise_cosine_mean"
                    )
                )
                /
                max(
                    len(
                        values(
                            "pairwise_cosine_mean"
                        )
                    ),
                    1,
                )
            ),
    }

    return {
        "summary": summary,
        "kernels": records,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layerwise_root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out_json",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    payload = {
        "schema":
            "mailohls-gnn-layerwise-geometry-v1",
        "layerwise_root":
            str(
                args.layerwise_root.resolve()
            ),
        "layers": {},
    }

    print()
    print(
        "LAYER\tCOSINE\tCENTER_ENERGY\t"
        "CENTER_ERANK_NORM\tCENTER_RANK90"
    )

    for layer in LAYERS:

        result = analyze_layer(
            args.layerwise_root
            / layer
        )

        payload[
            "layers"
        ][layer] = result

        s = result["summary"]

        print(
            f"{layer}\t"
            f"{s['pairwise_cosine_mean']:.6f}\t"
            f"{s['centered_energy_mean']:.6f}\t"
            f"{s['centered_erank_norm_mean']:.6f}\t"
            f"{s['centered_rank90_mean']:.3f}"
        )

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

    print(
        f"\n[DONE] wrote "
        f"{args.out_json}"
    )


if __name__ == "__main__":
    main()