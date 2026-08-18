#!/usr/bin/env python3
"""
Create deterministic Stage-2 structural-memory controls from
an immutable STATIC memory bank.

Outputs:
  ZERO
      No structural information; mask is disabled.

  SHUFFLED
      Original STATIC embeddings, but all active action slots are
      deranged within each kernel.

  GLOBAL
      Every active Lk receives the kernel-level mean embedding.

  LOCAL
      Every active Lk receives its centered action residual
          z_k - mean(z)
      RMS-matched to the original STATIC embeddings.

  LOCAL_SHUFFLED
      Same LOCAL residual vectors, but deterministically deranged
      across active Lk slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


EPS = 1e-8


def kernel_seed(
    global_seed: int,
    kernel_name: str,
) -> int:
    material = (
        f"{global_seed}:{kernel_name}"
        .encode()
    )

    return int.from_bytes(
        hashlib.sha256(material).digest()[:8],
        byteorder="little",
    )


def memory_files(directory: Path):
    return sorted(
        directory.glob("*.memory.pt")
    )


def require_empty_output(
    directory: Path,
) -> None:
    if (
        directory.exists()
        and any(directory.iterdir())
    ):
        raise FileExistsError(
            "Refusing to overwrite non-empty "
            f"output directory: {directory}"
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def decompose_action_memory(
    vectors: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Decompose

        z_k = mu_kernel + delta_k

    where:
        mu_kernel = mean active embedding
        delta_k   = z_k - mu_kernel

    LOCAL is RMS-matched to the original active embeddings
    so that the diagnostic tests information content rather
    than merely vector magnitude.
    """

    vectors = vectors.float()
    mask = mask.bool().view(-1)

    active_idx = torch.where(mask)[0]

    global_vectors = torch.zeros_like(
        vectors
    )
    local_vectors = torch.zeros_like(
        vectors
    )

    if active_idx.numel() == 0:
        return (
            global_vectors,
            local_vectors,
        )

    active = vectors[active_idx]

    mu = active.mean(
        dim=0,
        keepdim=True,
    )

    local = active - mu

    raw_rms = (
        active.square()
        .mean()
        .sqrt()
    )

    local_rms = (
        local.square()
        .mean()
        .sqrt()
    )

    if float(local_rms.item()) > EPS:
        scale = (
            raw_rms
            / local_rms
        )

        # Keep the diagnostic numerically controlled.
        scale = torch.clamp(
            scale,
            max=4.0,
        )

        local = local * scale

    global_vectors[
        active_idx
    ] = mu.expand_as(active)

    local_vectors[
        active_idx
    ] = local

    return (
        global_vectors,
        local_vectors,
    )


def derange_active_vectors(
    vectors: torch.Tensor,
    mask: torch.Tensor,
    *,
    seed: int,
    kernel_name: str,
) -> torch.Tensor:
    """
    Deterministically move every active vector to a
    DIFFERENT active slot whenever N >= 2.

    This is stronger than a plain torch.randperm(),
    because randperm may leave fixed points.
    """

    result = vectors.clone()

    mask = mask.bool().view(-1)
    active = torch.where(mask)[0]

    n = int(active.numel())

    if n < 2:
        return result

    original = result[
        active
    ].clone()

    generator = (
        torch.Generator()
        .manual_seed(
            kernel_seed(
                seed,
                kernel_name,
            )
        )
    )

    # Random ordering of action positions.
    order = torch.randperm(
        n,
        generator=generator,
    )

    # Rotate the random ordering.
    #
    # Target:
    #   active[order[i]]
    #
    # receives:
    #   original[order[i-1]]
    #
    # Therefore no target receives its own original
    # embedding.
    source_order = torch.roll(
        order,
        shifts=1,
    )

    result[
        active[order]
    ] = original[
        source_order
    ]

    # Strong invariant:
    # no active slot should retain its original vector.
    unchanged = (
        result[active]
        == original
    ).all(dim=1)

    if unchanged.any():
        raise RuntimeError(
            f"{kernel_name}: derangement "
            "unexpectedly left one or more "
            "active slots unchanged"
        )

    return result


def copy_manifest_if_present(
    static_dir: Path,
    output_dir: Path,
    ablation_name: str,
) -> None:
    """
    Keep provenance with each generated bank.

    If the original manifest is JSON, copy it and annotate
    the derived memory-bank type.
    """

    source_manifest = (
        static_dir
        / "memory_manifest.json"
    )

    if not source_manifest.exists():
        return

    try:
        manifest = json.loads(
            source_manifest.read_text(
                encoding="utf-8"
            )
        )

        manifest[
            "memory_ablation"
        ] = ablation_name

        manifest[
            "derived_from"
        ] = str(
            static_dir.resolve()
        )

        (
            output_dir
            / "memory_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    except Exception:
        # Better to preserve the original manifest than
        # silently omit provenance if its schema changes.
        shutil.copy2(
            source_manifest,
            output_dir
            / "memory_manifest.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--static_dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--zero_out",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--shuffled_out",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--global_out",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--local_out",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--local_shuffled_out",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    args = parser.parse_args()

    files = memory_files(
        args.static_dir
    )

    if not files:
        raise FileNotFoundError(
            "No *.memory.pt files in "
            f"{args.static_dir}"
        )

    outputs = [
        args.zero_out,
        args.shuffled_out,
        args.global_out,
        args.local_out,
        args.local_shuffled_out,
    ]

    # Avoid accidentally passing the same output twice.
    resolved = [
        str(path.resolve())
        for path in outputs
    ]

    if len(set(resolved)) != len(
        resolved
    ):
        raise ValueError(
            "All output directories must "
            "be distinct"
        )

    for directory in outputs:
        require_empty_output(
            directory
        )

    for source in files:
        static = torch.load(
            source,
            map_location="cpu",
            weights_only=False,
        )

        vectors = (
            static["node_embs"]
            .float()
        )

        mask = (
            static[
                "node_embs_mask"
            ]
            .bool()
            .view(-1)
        )

        if vectors.shape[0] != mask.shape[0]:
            raise ValueError(
                f"{source.name}: "
                "node_embs/node_embs_mask "
                "slot mismatch: "
                f"{tuple(vectors.shape)} vs "
                f"{tuple(mask.shape)}"
            )

        kernel_name = source.name[
            :-len(".memory.pt")
        ]

        # ----------------------------------------------
        # GLOBAL + LOCAL decomposition
        # ----------------------------------------------

        (
            global_vectors,
            local_vectors,
        ) = decompose_action_memory(
            vectors,
            mask,
        )

        # ----------------------------------------------
        # ZERO
        # ----------------------------------------------

        zero = dict(static)

        if (
            "action_relation_mask"
            in zero
        ):
            zero[
                "action_relation_mask"
            ] = torch.zeros_like(
                zero[
                    "action_relation_mask"
                ],
                dtype=torch.bool,
            )

        if (
            "action_relation_bits"
            in zero
        ):
            zero[
                "action_relation_bits"
            ] = torch.zeros_like(
                zero[
                    "action_relation_bits"
                ]
            )

        zero[
            "node_embs"
        ] = torch.zeros_like(
            vectors
        )

        zero[
            "node_embs_mask"
        ] = torch.zeros_like(
            mask,
            dtype=torch.bool,
        )

        zero[
            "memory_ablation"
        ] = "zero_information"

        torch.save(
            zero,
            args.zero_out
            / source.name,
        )

        # ----------------------------------------------
        # STATIC SHUFFLE
        # ----------------------------------------------

        shuffled = dict(static)

        shuffled[
            "node_embs"
        ] = derange_active_vectors(
            vectors,
            mask,
            seed=args.seed,
            kernel_name=kernel_name,
        )

        shuffled[
            "memory_ablation"
        ] = (
            "within_kernel_deranged_slots"
        )

        shuffled[
            "memory_ablation_seed"
        ] = args.seed

        torch.save(
            shuffled,
            args.shuffled_out
            / source.name,
        )

        # ----------------------------------------------
        # GLOBAL ONLY
        # ----------------------------------------------

        global_pack = dict(static)

        global_pack[
            "node_embs"
        ] = global_vectors

        global_pack[
            "memory_ablation"
        ] = "kernel_global_only"

        torch.save(
            global_pack,
            args.global_out
            / source.name,
        )

        # ----------------------------------------------
        # LOCAL ALIGNED
        # ----------------------------------------------

        local_pack = dict(static)

        local_pack[
            "node_embs"
        ] = local_vectors

        local_pack[
            "memory_ablation"
        ] = (
            "action_local_centered_"
            "rms_matched"
        )

        torch.save(
            local_pack,
            args.local_out
            / source.name,
        )

        # ----------------------------------------------
        # LOCAL SHUFFLE
        # ----------------------------------------------

        local_shuffled_pack = dict(
            static
        )

        local_shuffled_pack[
            "node_embs"
        ] = derange_active_vectors(
            local_vectors,
            mask,
            seed=args.seed,
            kernel_name=kernel_name,
        )

        local_shuffled_pack[
            "memory_ablation"
        ] = (
            "action_local_centered_"
            "rms_matched_deranged"
        )

        local_shuffled_pack[
            "memory_ablation_seed"
        ] = args.seed

        torch.save(
            local_shuffled_pack,
            args.local_shuffled_out
            / source.name,
        )

    # Preserve/annotate provenance manifests.
    copy_manifest_if_present(
        args.static_dir,
        args.zero_out,
        "zero_information",
    )

    copy_manifest_if_present(
        args.static_dir,
        args.shuffled_out,
        "within_kernel_deranged_slots",
    )

    copy_manifest_if_present(
        args.static_dir,
        args.global_out,
        "kernel_global_only",
    )

    copy_manifest_if_present(
        args.static_dir,
        args.local_out,
        "action_local_centered_rms_matched",
    )

    copy_manifest_if_present(
        args.static_dir,
        args.local_shuffled_out,
        "action_local_centered_rms_matched_deranged",
    )

    print(
        f"[DONE] Processed {len(files)} kernels"
    )

    print(
        f"[DONE] ZERO          -> "
        f"{args.zero_out}"
    )

    print(
        f"[DONE] SHUFFLED      -> "
        f"{args.shuffled_out}"
    )

    print(
        f"[DONE] GLOBAL        -> "
        f"{args.global_out}"
    )

    print(
        f"[DONE] LOCAL         -> "
        f"{args.local_out}"
    )

    print(
        f"[DONE] LOCAL_SHUFFLE -> "
        f"{args.local_shuffled_out}"
    )


if __name__ == "__main__":
    main()