#!/usr/bin/env python3
"""Export AUTHENTIC V4b action embeddings and heatmaps from local frozen checkpoints.

This does not pretend to convert production V1 GEMV GEXF to V4b RC3: the
requested kernel must exist in the supplied authenticated V4b prepared cache.
Example:
 python inspect_v4b_action_embeddings.py \
    --cache /path/to/prepared_cache.pt \
    --run-dir /home/elvouvali/MailoHLS-GNN-runs/v4b-v2-20261102-8f3593b/corrected_v3_full37 \
    --kernel machsuite-gemm-ncubed --clock 10 --out-dir /tmp/v4b-embedding-figures
Use the authenticated V4b training Python environment (torch + torch_geometric).
Never interpret a plot as proof of downstream HLS/LLM optimization benefit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.v22_gnn_training.v4b.models import V4bGraphModel, TargetFilmV4bModel


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def graph_key(cache, kernel, profile, clock):
    keys = [key for key in cache if key[0] == kernel and key[1] == profile
            and abs(float(key[2]) - clock) < 1e-6]
    if len(keys) != 1:
        choices = sorted((str(k) for k in cache if k[0] == kernel))[:40]
        raise SystemExit(f"No unique graph for ({kernel}, {profile}, {clock}). "
                         f"Available contexts for kernel: {choices}")
    return keys[0]


def heatmap(matrix, title, destination, labels):
    fig = plt.figure(figsize=(13.5, max(3.0, min(11.0, 0.35 * len(labels) + 1.3))))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Embedding coordinate value")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Learned structural feature dimension (0–127)")
    plt.ylabel("Matched action point")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path,
                        help="Authenticated PyTorch V4b preparation cache containing 'prepared'")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Corrected_v3_full37 run with compiler/hardware/target_film_selected.pt")
    parser.add_argument("--kernel", default="machsuite-gemm-ncubed")
    parser.add_argument("--clock", type=float, default=10.0)
    parser.add_argument("--profiles", nargs="+", default=["zcu104", "u200"])
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # torch.load(weights_only=False) is for locally trusted, authenticated research files ONLY.
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    prepared = cached["prepared"] if isinstance(cached, dict) and "prepared" in cached else cached
    if not isinstance(prepared, dict) or "graph_cache" not in prepared or "profile_features" not in prepared:
        raise SystemExit("This is not a V4b prepared cache. Expected graph_cache and profile_features.")

    dim = int(prepared["profile_feature_dim"])
    models = {
        "compiler": V4bGraphModel(hardware_messages=False),
        "hardware": V4bGraphModel(hardware_messages=True),
        "target_film": TargetFilmV4bModel(profile_feature_dim=dim),
    }
    # Verify exact tracked checkpoint bytes where possible.
    manifest_path = (REPO /
                     "experiments/v22_gnn_training/v4b/receipts_v3/artifact_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    selected = manifest.get("runs", {}).get("full37", {}) if manifest else {}
    trusted = selected.get("selected_weights_sha256", {})
    weights = {}
    for arm, model in models.items():
        path = args.run_dir / f"{arm}_selected.pt"
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint: {path}")
        checksum = digest(path)
        if arm + "_selected.pt" in trusted and checksum != trusted[arm + "_selected.pt"]:
            raise SystemExit(f"Checkpoint SHA256 mismatch: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state"], strict=True)
        model.eval()
        weights[arm] = {"file": str(path), "sha256": checksum,
                        "selected_step": state.get("selected_step")}

    data = {}
    summaries = {}
    with torch.inference_mode():
        for profile in args.profiles:
            key = graph_key(prepared["graph_cache"], args.kernel, profile, args.clock)
            graph = prepared["graph_cache"][key].clone().cpu()
            active = (graph.registered_mask & graph.matched_mask)[0].nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                raise SystemExit(f"No compiler-matched action slots: {key}")
            labels = [f"L{int(index)+1}" for index in active]
            if (profile, args.clock) not in prepared["profile_features"]:
                raise SystemExit(f"No authenticated profile features for {(profile,args.clock)}")
            feat = prepared["profile_features"][(profile, args.clock)]
            for arm, model in models.items():
                if arm == "target_film":
                    memory, pooled = model.encode(graph, feat)
                else:
                    memory, pooled = model.encode(graph)
                # The raw 64x128 embeddings and masks are retained, not only a 2D projection.
                array = memory[0].detach().cpu().numpy().copy()
                data[(arm, profile)] = array
                np.savez_compressed(args.out_dir / f"{arm}_{profile}_raw.npz",
                                    memory=array, pooled=pooled[0].cpu().numpy(),
                                    registered_mask=graph.registered_mask[0].numpy(),
                                    matched_mask=graph.matched_mask[0].numpy(),
                                    active_slots=active.numpy(), labels=np.asarray(labels))
                np.savetxt(args.out_dir / f"{arm}_{profile}_active.csv",
                           np.column_stack((active.numpy(), array[active.numpy()])),
                           delimiter=",", comments="",
                           header="slot_0based," + ",".join(f"feature_{i}" for i in range(128)))
                heatmap(array[active.numpy()], f"{args.kernel}: {arm} / {profile}, {args.clock}ns",
                        args.out_dir / f"{arm}_{profile}_actions.png", labels)
                summaries[f"{arm}/{profile}"] = {
                    "matched_action_ids": labels,
                    "action_norms": {label: float(np.linalg.norm(array[int(slot)]))
                                     for label, slot in zip(labels, active)},
                    "first_8_dimensions": {label: array[int(slot), :8].tolist()
                                           for label, slot in zip(labels, active)},
                }
                if arm == "target_film":
                    baseline = data[("compiler", profile)]
                    delta = array - baseline
                    heatmap(delta[active.numpy()], f"FiLM minus compiler / {profile}",
                            args.out_dir / f"film_effect_{profile}.png", labels)
                    summaries[f"{arm}/{profile}"]["film_effect_frobenius"] = float(
                        np.linalg.norm(delta[active.numpy()]))

        if len(args.profiles) == 2:
            left, right = args.profiles
            for arm in models:
                A, B = data[(arm, left)], data[(arm, right)]
                key = graph_key(prepared["graph_cache"], args.kernel, left, args.clock)
                mask = prepared["graph_cache"][key].matched_mask[0].numpy().copy()
                other_key = graph_key(prepared["graph_cache"], args.kernel, right, args.clock)
                mask &= prepared["graph_cache"][other_key].matched_mask[0].numpy()
                slots = np.flatnonzero(mask)
                labels = [f"L{int(s)+1}" for s in slots]
                delta = B - A
                if len(slots):
                    heatmap(delta[slots], f"{arm}: {right} minus {left} memory",
                            args.out_dir / f"{arm}_target_difference.png", labels)
                summaries[f"{arm}/target_delta"] = {
                    "compared_action_ids": labels,
                    "frobenius_norm": float(np.linalg.norm(delta[slots]))
                }

    report = {
        "kernel": args.kernel, "clock_ns": args.clock, "profiles": args.profiles,
        "input_provenance": cached.get("provenance") if isinstance(cached, dict) else None,
        "prepared_cache": str(args.cache), "prepared_cache_sha256": digest(args.cache),
        "checkpoints": weights, "results": summaries,
        "note": "Actual trained embeddings. No assumption of GEMV unless authenticated GEMV RC3 tensors exist."
    }
    (args.out_dir / "embedding_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2))
    print(f"Wrote authentic matrices and heatmaps to {args.out_dir}")

if __name__ == "__main__":
    main()
