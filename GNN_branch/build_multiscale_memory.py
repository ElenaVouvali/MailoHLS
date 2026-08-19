#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(a, b, label: str) -> None:
    if a is None and b is None:
        return
    if a is None or b is None:
        raise ValueError(f"{label}: present in only one pack")

    a = torch.as_tensor(a)
    b = torch.as_tensor(b)

    if not torch.equal(a, b):
        raise ValueError(f"{label}: JKN/conv1 packs disagree")


def kernel_seed(seed: int, kernel: str) -> int:
    material = f"{seed}:{kernel}".encode()
    return int.from_bytes(
        hashlib.sha256(material).digest()[:8],
        "little",
    )


def derange_local(
    local: torch.Tensor,
    mask: torch.Tensor,
    *,
    seed: int,
    kernel: str,
) -> torch.Tensor:
    result = local.clone()
    active = torch.where(mask)[0]
    n = int(active.numel())

    if n < 2:
        return result

    generator = torch.Generator().manual_seed(
        kernel_seed(seed, kernel)
    )
    order = torch.randperm(n, generator=generator)
    source = torch.roll(order, shifts=1)

    original = result[active].clone()
    result[active[order]] = original[source]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jkn_dir", required=True, type=Path)
    parser.add_argument("--conv1_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--local_scale", type=float, default=1.0)
    parser.add_argument(
        "--local_mode",
        choices=("aligned", "deranged"),
        default="aligned",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty directory: {args.out}"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    jkn_files = sorted(args.jkn_dir.glob("*.memory.pt"))
    conv_files = sorted(args.conv1_dir.glob("*.memory.pt"))

    if not jkn_files:
        raise FileNotFoundError("No JKN memory files")
    if {p.name for p in jkn_files} != {p.name for p in conv_files}:
        raise ValueError("JKN and conv1 kernel-file sets differ")

    records = []
    jkn_sumsq = 0.0
    jkn_count = 0
    local_sumsq = 0.0
    local_count = 0

    invariant_keys = (
        "node_embs_mask",
        "labels",
        "slot_ids",
        "slot_cats",
        "action_relation_mask",
        "action_relation_bits",
    )

    for jkn_path in jkn_files:
        conv_path = args.conv1_dir / jkn_path.name

        jpack = torch.load(
            jkn_path,
            map_location="cpu",
            weights_only=False,
        )
        cpack = torch.load(
            conv_path,
            map_location="cpu",
            weights_only=False,
        )

        for key in invariant_keys:
            require_equal(
                jpack.get(key),
                cpack.get(key),
                f"{jkn_path.name}/{key}",
            )

        mask = torch.as_tensor(
            jpack["node_embs_mask"],
            dtype=torch.bool,
        ).view(-1)

        jkn = torch.as_tensor(
            jpack["node_embs"],
            dtype=torch.float32,
        )
        conv1 = torch.as_tensor(
            cpack["node_embs"],
            dtype=torch.float32,
        )

        if jkn.shape != conv1.shape:
            raise ValueError(
                f"{jkn_path.name}: JKN {tuple(jkn.shape)} "
                f"!= conv1 {tuple(conv1.shape)}"
            )

        active = torch.where(mask)[0]
        local = torch.zeros_like(conv1)

        if active.numel():
            active_conv1 = conv1.index_select(0, active)
            centred = active_conv1 - active_conv1.mean(
                dim=0,
                keepdim=True,
            )
            local[active] = centred

            active_jkn = jkn.index_select(0, active)
            jkn_sumsq += float(active_jkn.square().sum())
            jkn_count += active_jkn.numel()

            local_sumsq += float(centred.square().sum())
            local_count += centred.numel()

        records.append((jkn_path, jpack, mask, jkn, local))

    jkn_rms = math.sqrt(jkn_sumsq / max(jkn_count, 1))
    local_rms = math.sqrt(local_sumsq / max(local_count, 1))

    if jkn_rms <= 0 or local_rms <= 0:
        raise RuntimeError(
            f"Invalid component RMS: jkn={jkn_rms}, local={local_rms}"
        )

    for jkn_path, pack, mask, jkn, local in records:
        active = torch.where(mask)[0]

        memory = torch.zeros(
            (jkn.size(0), jkn.size(1) * 2),
            dtype=torch.float32,
        )


        if active.numel():
            jkn_part = jkn.index_select(0, active) / jkn_rms
            local_used = local
            if args.local_mode == "deranged":
                suffix = ".memory.pt"
                filename = jkn_path.name
                if not filename.endswith(suffix):
                    raise ValueError(
                        f"Unexpected memory filename: {filename}"
                    )
                kernel = filename[:-len(suffix)]
                local_used = derange_local(
                    local,
                    mask,
                    seed=args.seed,
                    kernel=kernel,
                )

            local_part = (
                local_used.index_select(0, active)
                / local_rms
                * args.local_scale
            )

            combined = torch.cat(
                (jkn_part, local_part),
                dim=-1,
            ) / math.sqrt(2.0)

            # Match the existing exporter safety bound.
            norms = combined.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)

            combined = combined * (
                20.0 / norms
            ).clamp(max=1.0)

            memory[active] = combined

        output_pack = dict(pack)
        output_pack["node_embs"] = memory
        output_pack["gnn_dim"] = int(memory.size(-1))
        output_pack["embedding_mode"] = (
            "multiscale::jkn+centered_conv1"
        )
        output_pack["multiscale"] = {
            "jkn_rms": jkn_rms,
            "centered_conv1_rms": local_rms,
            "local_scale": args.local_scale,
            "local_mode": args.local_mode,
            "combination": "concat_div_sqrt2",
        }

        output_pack["multiscale"]["local_seed"] = args.seed if args.local_mode == "deranged" else None

        torch.save(
            output_pack,
            args.out / jkn_path.name,
        )

    manifest_path = args.jkn_dir / "memory_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["embedding_mode"] = (
        "multiscale::jkn+centered_conv1"
    )
    manifest["gnn_dim"] = 128
    manifest["multiscale"] = {
        "jkn_manifest_sha256": sha256_file(manifest_path),
        "conv1_dir": str(args.conv1_dir.resolve()),
        "jkn_rms": jkn_rms,
        "centered_conv1_rms": local_rms,
        "local_scale": args.local_scale,
        "local_mode": args.local_mode,
        "combination": "concat_div_sqrt2",
    }

    manifest["multiscale"]["local_seed"] = args.seed if args.local_mode == "deranged" else None

    (args.out / "memory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"[DONE] kernels={len(records)} dim=128 "
        f"jkn_rms={jkn_rms:.6g} "
        f"local_rms={local_rms:.6g}"
    )


if __name__ == "__main__":
    main()