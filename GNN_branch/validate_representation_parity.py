#!/usr/bin/env python3
"""Verify that HARP-Rep and Structured-MLIR consume identical design points.

The static graph tensors are intentionally different.  The experiment is valid
only when index membership, pragma vectors, measured QoR/resource labels and
device/clock conditioning are exactly equal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

POINT_FIELDS = (
    "keys",
    "target_devices",
    "target_clock_period_ns",
    "target_groups",
    "target_condition",
    "directive_indices",
    "pragmas",
    "perf",
    "actual_perf",
    "kernel_speedup",
    "area",
    "actual_area",
    "actual_effective_area",
    "resource_util",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return False
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir-cache", required=True)
    ap.add_argument("--harp-cache", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    mlir = Path(args.mlir_cache).expanduser().resolve()
    harp = Path(args.harp_cache).expanduser().resolve()
    split = Path(args.split_json).expanduser().resolve()
    for path in (mlir / "index.pt", harp / "index.pt", split):
        if not path.is_file():
            raise FileNotFoundError(path)

    mlir_index = torch.load(mlir / "index.pt", weights_only=False)
    harp_index = torch.load(harp / "index.pt", weights_only=False)
    failures: list[str] = []

    if not equal(mlir_index, harp_index):
        failures.append("index.pt records differ")

    graph_names = sorted({str(row["graph_name"]) for row in mlir_index})
    field_checks = 0
    point_file_checks = 0
    per_graph = []
    for graph_name in graph_names:
        mp = mlir / "points" / f"{graph_name}.pt"
        hp = harp / "points" / f"{graph_name}.pt"
        if not mp.is_file() or not hp.is_file():
            failures.append(f"missing point file for {graph_name}")
            continue
        m = torch.load(mp, weights_only=False)
        h = torch.load(hp, weights_only=False)
        bad = []
        for field in POINT_FIELDS:
            field_checks += 1
            if field not in m or field not in h:
                bad.append(f"{field}:missing")
                continue
            if not equal(m[field], h[field]):
                bad.append(field)
        # X_pragma_per_node MUST differ in node dimension; it is excluded from
        # equality, but both arms must still have a finite matrix for every
        # unique directive vector.
        for label, payload in (("mlir", m), ("harp", h)):
            xpn = payload.get("X_pragma_per_node")
            if not torch.is_tensor(xpn) or xpn.ndim != 3 or not torch.isfinite(xpn).all():
                bad.append(f"{label}:invalid_X_pragma_per_node")
            elif xpn.shape[0] != payload["pragmas"].shape[0]:
                bad.append(f"{label}:directive_count_mismatch")
        if bad:
            failures.append(f"{graph_name}: " + ",".join(bad))
        point_file_checks += 1
        per_graph.append({"graph_name": graph_name, "status": "PASS" if not bad else "FAIL", "mismatches": bad})

    report = {
        "schema": "mailohls-paired-representation-parity-v1",
        "status": "PASS" if not failures else "FAIL",
        "mlir_cache": str(mlir),
        "harp_cache": str(harp),
        "split_json": str(split),
        "split_sha256": sha(split),
        "index_records": len(mlir_index),
        "graph_count": len(graph_names),
        "point_files_checked": point_file_checks,
        "point_fields_checked": field_checks,
        "equal_fields": list(POINT_FIELDS),
        "intentionally_representation_specific": [
            "graphs/*.pt:x", "graphs/*.pt:edge_index", "graphs/*.pt:edge_attr",
            "points/*.pt:X_pragma_per_node",
        ],
        "failures": failures,
        "graphs": per_graph,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "index_records", "graph_count", "failures")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
