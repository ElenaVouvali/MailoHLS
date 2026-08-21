#!/usr/bin/env python3
"""Build matched multiscale memories using training-kernel-only RMS statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import torch


NORMALIZATION_SCHEMA = "mailohls-multiscale-normalization-v1"
PROVENANCE_FIELDS = (
    "gnn_contract_sha256", "gnn_checkpoint_sha256", "feature_schema_sha256",
    "source_pt_manifest_sha256", "source_gexf_manifest_sha256",
    "exporter_git_commit", "action_relation_schema", "action_slot_schema",
    "kernel_count", "ordered_kernel_list_sha256",
)
INVARIANT_KEYS = (
    "node_embs_mask", "labels", "slot_ids", "slot_cats",
    "action_relation_mask", "action_relation_bits",
)
PACK_PROVENANCE_FIELDS = (
    "gnn_contract_sha256", "gnn_checkpoint_sha256", "feature_schema_sha256",
    "source_pt_manifest_sha256", "source_gexf_manifest_sha256",
    "source_gexf_sha256", "action_relation_schema", "max_slots",
    "disable_pragma_injection",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_kernel_name(name: str) -> str:
    return re.sub(r"[-\s]+", "_", str(name).strip().lower())


def ordered_names_sha256(names) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def require_equal(a, b, label: str) -> None:
    if a is None and b is None:
        return
    if a is None or b is None:
        raise ValueError(f"{label}: present in only one pack")
    if not torch.equal(torch.as_tensor(a), torch.as_tensor(b)):
        raise ValueError(f"{label}: JKN/conv1 packs disagree")


def validate_layer_manifests(jkn_dir: Path, conv1_dir: Path) -> tuple[dict, dict]:
    manifests = []
    for directory, expected_layer in ((jkn_dir, "jkn"), (conv1_dir, "conv_1")):
        path = directory / "memory_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Layerwise memory manifest is missing: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "mailohls-memory-bank-manifest-v2":
            raise ValueError(f"Unsupported layerwise manifest: {path}")
        if manifest.get("layer_name") != expected_layer:
            raise ValueError(f"{path}: expected layer_name={expected_layer!r}")
        missing = [field for field in PROVENANCE_FIELDS if manifest.get(field) is None]
        if missing:
            raise ValueError(f"{path}: incomplete layerwise provenance: {missing}")
        manifests.append(manifest)
    jkn_manifest, conv1_manifest = manifests
    mismatches = {
        field: {"jkn": jkn_manifest[field], "conv1": conv1_manifest[field]}
        for field in PROVENANCE_FIELDS
        if jkn_manifest[field] != conv1_manifest[field]
    }
    if mismatches:
        raise ValueError("JKN/conv1 provenance mismatch: " + json.dumps(mismatches, sort_keys=True))
    return jkn_manifest, conv1_manifest


def load_training_kernel_names(dataset_jsonl: Path, split_json: Path) -> set[str]:
    split = json.loads(split_json.read_text(encoding="utf-8"))
    if not isinstance(split.get("train_jsonl_idx"), list):
        raise ValueError("Split must contain train_jsonl_idx")
    wanted = {int(index) for index in split["train_jsonl_idx"]}
    if not wanted:
        raise ValueError("Training split is empty")
    found_indices = set()
    kernels = set()
    with dataset_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in wanted:
                continue
            if not line.strip():
                raise ValueError(f"Training JSONL index {index} points to a blank line")
            row = json.loads(line)
            kernel = str(row.get("kernel_name", "")).strip()
            if not kernel:
                raise ValueError(f"Training JSONL index {index} has no kernel_name")
            kernels.add(normalize_kernel_name(kernel))
            found_indices.add(index)
    missing = sorted(wanted - found_indices)
    if missing:
        raise ValueError(f"Training split references missing JSONL indices: {missing[:20]}")
    return kernels


def kernel_seed(seed: int, kernel: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{kernel}".encode()).digest()[:8], "little")


def derange_local(local: torch.Tensor, mask: torch.Tensor, *, seed: int, kernel: str) -> torch.Tensor:
    result = local.clone()
    active = torch.where(mask)[0]
    if active.numel() < 2:
        return result
    generator = torch.Generator().manual_seed(kernel_seed(seed, kernel))
    order = torch.randperm(active.numel(), generator=generator)
    source = torch.roll(order, shifts=1)
    original = result[active].clone()
    result[active[order]] = original[source]
    return result


def validate_normalization_stats(
    stats: dict, *, dataset_jsonl: Path, split_json: Path,
    training_kernels: set[str], jkn_dim: int, local_dim: int,
) -> None:
    expected = {
        "schema": NORMALIZATION_SCHEMA,
        "fit_policy": "training-kernels-only",
        "dataset_sha256": sha256_file(dataset_jsonl),
        "split_sha256": sha256_file(split_json),
        "training_kernel_count": len(training_kernels),
        "training_kernel_names_sha256": ordered_names_sha256(training_kernels),
        "jkn_dim": jkn_dim,
        "local_dim": local_dim,
    }
    mismatches = {
        key: {"expected": value, "actual": stats.get(key)}
        for key, value in expected.items() if stats.get(key) != value
    }
    if mismatches:
        raise ValueError("Normalization artifact does not match this training split: " + json.dumps(mismatches))
    for field in ("jkn_rms", "centered_conv1_rms"):
        value = float(stats.get(field, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid normalization statistic {field}={value}")


def build_memory(args: argparse.Namespace) -> dict:
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.out}")
    if args.local_mode != "aligned" and args.normalization_stats_in is None:
        raise ValueError("Deranged and zero controls require --normalization_stats_in from the aligned run")
    if args.normalization_stats_in is None and args.normalization_stats_out is None:
        raise ValueError("Fitting normalization requires --normalization_stats_out")
    if not math.isfinite(args.local_scale) or args.local_scale < 0.0:
        raise ValueError("--local_scale must be finite and non-negative")

    jkn_manifest, conv1_manifest = validate_layer_manifests(args.jkn_dir, args.conv1_dir)
    jkn_by_name = {path.name: path for path in args.jkn_dir.glob("*.memory.pt")}
    conv_by_name = {path.name: path for path in args.conv1_dir.glob("*.memory.pt")}
    if not jkn_by_name:
        raise FileNotFoundError("No JKN memory files")
    missing_from_jkn = sorted(set(conv_by_name) - set(jkn_by_name))
    missing_from_conv1 = sorted(set(jkn_by_name) - set(conv_by_name))
    if missing_from_jkn or missing_from_conv1:
        raise ValueError(
            "JKN/conv1 kernel-file sets differ: "
            f"missing_from_jkn={missing_from_jkn}, missing_from_conv1={missing_from_conv1}"
        )
    kernel_names = [name[:-len(".memory.pt")] for name in sorted(jkn_by_name)]
    if len(kernel_names) != int(jkn_manifest["kernel_count"]):
        raise ValueError("Layerwise file count disagrees with the manifest kernel_count")
    if ordered_names_sha256(kernel_names) != jkn_manifest["ordered_kernel_list_sha256"]:
        raise ValueError("Layerwise files disagree with the manifest ordered kernel list")

    training_kernels = load_training_kernel_names(args.dataset_jsonl, args.split_json)
    normalized_memory_names = {}
    for kernel in kernel_names:
        normalized = normalize_kernel_name(kernel)
        existing = normalized_memory_names.get(normalized)
        if existing is not None and existing != kernel:
            raise ValueError(f"Kernel normalization collision: {existing!r} and {kernel!r}")
        normalized_memory_names[normalized] = kernel
    missing_training = sorted(training_kernels - set(normalized_memory_names))
    if missing_training:
        raise ValueError("Training kernels have no matched structural memory: " + ", ".join(missing_training))

    records = []
    jkn_sumsq = local_sumsq = 0.0
    jkn_count = local_count = 0
    jkn_dim = local_dim = None
    for filename in sorted(jkn_by_name):
        jpack = torch.load(jkn_by_name[filename], map_location="cpu", weights_only=False)
        cpack = torch.load(conv_by_name[filename], map_location="cpu", weights_only=False)
        for key in INVARIANT_KEYS:
            if jpack.get(key) is None or cpack.get(key) is None:
                raise ValueError(f"{filename}: required action-slot invariant is missing: {key}")
            require_equal(jpack.get(key), cpack.get(key), f"{filename}/{key}")
        for key in PACK_PROVENANCE_FIELDS:
            if jpack.get(key) is None or cpack.get(key) is None:
                raise ValueError(f"{filename}: required source provenance is missing: {key}")
            if jpack.get(key) != cpack.get(key):
                raise ValueError(f"{filename}: JKN/conv1 pack provenance mismatch for {key}")
        mask = torch.as_tensor(jpack["node_embs_mask"], dtype=torch.bool).view(-1)
        jkn = torch.as_tensor(jpack["node_embs"], dtype=torch.float32)
        conv1 = torch.as_tensor(cpack["node_embs"], dtype=torch.float32)
        if jkn.ndim != 2 or conv1.ndim != 2 or jkn.size(0) != conv1.size(0) or jkn.size(0) != mask.numel():
            raise ValueError(f"{filename}: incompatible JKN/conv1 shapes or active mask")
        if jkn_dim is None:
            jkn_dim, local_dim = int(jkn.size(1)), int(conv1.size(1))
        if (jkn.size(1), conv1.size(1)) != (jkn_dim, local_dim):
            raise ValueError(f"{filename}: component dimensions differ across kernels")
        active = torch.where(mask)[0]
        local = torch.zeros_like(conv1)
        if active.numel():
            active_conv1 = conv1.index_select(0, active)
            centered = active_conv1 - active_conv1.mean(dim=0, keepdim=True)
            local[active] = centered
            kernel = normalize_kernel_name(filename[:-len(".memory.pt")])
            if kernel in training_kernels:
                active_jkn = jkn.index_select(0, active)
                jkn_sumsq += float(active_jkn.square().sum())
                jkn_count += active_jkn.numel()
                local_sumsq += float(centered.square().sum())
                local_count += centered.numel()
        records.append((filename, jpack, mask, jkn, local))

    if args.normalization_stats_in is not None:
        stats_source = args.normalization_stats_in
        stats = json.loads(stats_source.read_text(encoding="utf-8"))
    else:
        stats = {
            "schema": NORMALIZATION_SCHEMA,
            "fit_policy": "training-kernels-only",
            "dataset_sha256": sha256_file(args.dataset_jsonl),
            "split_sha256": sha256_file(args.split_json),
            "training_kernel_count": len(training_kernels),
            "training_kernel_names_sha256": ordered_names_sha256(training_kernels),
            "jkn_rms": math.sqrt(jkn_sumsq / max(jkn_count, 1)),
            "centered_conv1_rms": math.sqrt(local_sumsq / max(local_count, 1)),
            "jkn_dim": jkn_dim,
            "local_dim": local_dim,
        }
        stats_source = args.normalization_stats_out
    validate_normalization_stats(
        stats, dataset_jsonl=args.dataset_jsonl, split_json=args.split_json,
        training_kernels=training_kernels, jkn_dim=jkn_dim, local_dim=local_dim,
    )

    if args.normalization_stats_in is None:
        stats_source.parent.mkdir(parents=True, exist_ok=True)
        stats_source.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.normalization_stats_out is not None and args.normalization_stats_out != stats_source:
        args.normalization_stats_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stats_source, args.normalization_stats_out)

    args.out.mkdir(parents=True, exist_ok=True)
    output_stats = args.out / "normalization_stats.json"
    if stats_source.resolve() != output_stats.resolve():
        shutil.copyfile(stats_source, output_stats)
    normalization_sha256 = sha256_file(output_stats)
    jkn_rms = float(stats["jkn_rms"])
    local_rms = float(stats["centered_conv1_rms"])

    for filename, pack, mask, jkn, local in records:
        active = torch.where(mask)[0]
        memory = torch.zeros((jkn.size(0), jkn_dim + local_dim), dtype=torch.float32)
        if active.numel():
            if args.local_mode == "deranged":
                local = derange_local(local, mask, seed=args.seed, kernel=filename[:-len(".memory.pt")])
            elif args.local_mode == "zero":
                local = torch.zeros_like(local)
            combined = torch.cat(
                (
                    jkn.index_select(0, active) / jkn_rms,
                    local.index_select(0, active) / local_rms * args.local_scale,
                ), dim=-1,
            ) / math.sqrt(2.0)
            norms = combined.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            memory[active] = combined * (20.0 / norms).clamp(max=1.0)
        output_pack = dict(pack)
        output_pack.update({
            "node_embs": memory,
            "gnn_dim": int(memory.size(-1)),
            "embedding_mode": "multiscale::jkn+centered_conv1",
            "multiscale": {
                "jkn_rms": jkn_rms,
                "centered_conv1_rms": local_rms,
                "local_scale": args.local_scale,
                "local_mode": args.local_mode,
                "local_seed": args.seed if args.local_mode == "deranged" else None,
                "normalization_stats_sha256": normalization_sha256,
                "combination": "concat_div_sqrt2",
            },
        })
        torch.save(output_pack, args.out / filename)

    manifest = dict(jkn_manifest)
    manifest.pop("layer_name", None)
    manifest.update({
        "embedding_mode": "multiscale::jkn+centered_conv1",
        "gnn_dim": jkn_dim + local_dim,
        "multiscale": {
            "jkn_manifest_sha256": sha256_file(args.jkn_dir / "memory_manifest.json"),
            "conv1_manifest_sha256": sha256_file(args.conv1_dir / "memory_manifest.json"),
            "normalization_stats_sha256": normalization_sha256,
            "normalization_stats_file": "normalization_stats.json",
            "jkn_rms": jkn_rms,
            "centered_conv1_rms": local_rms,
            "local_scale": args.local_scale,
            "local_mode": args.local_mode,
            "local_seed": args.seed if args.local_mode == "deranged" else None,
            "combination": "concat_div_sqrt2",
        },
    })
    (args.out / "memory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[DONE] kernels={len(records)} training_kernels={len(training_kernels)} "
        f"dim={jkn_dim + local_dim} local_mode={args.local_mode} "
        f"jkn_rms={jkn_rms:.6g} local_rms={local_rms:.6g} "
        f"normalization_sha256={normalization_sha256}"
    )
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jkn_dir", required=True, type=Path)
    parser.add_argument("--conv1_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset_jsonl", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--normalization_stats_out", type=Path)
    parser.add_argument("--normalization_stats_in", type=Path)
    parser.add_argument("--local_scale", type=float, default=1.0)
    parser.add_argument("--local_mode", choices=("aligned", "deranged", "zero"), default="aligned")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args(argv)


def main() -> None:
    build_memory(parse_args())


if __name__ == "__main__":
    main()
