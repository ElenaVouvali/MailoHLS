"""Measured neutral-reference targets for cross-kernel QoR learning.

The manifest is deliberately external to the cached GEXF/PT representation:
the graph describes the kernel, while this file records the Vitis/device/clock
measurement used to anchor absolute QoR.  Training predicts only the pragma
response relative to that anchor.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {
    "kernel",
    "status",
    "source_sha256",
    "toolchain_id",
    "device",
    "clock_period_ns",
    "baseline_latency_ms",
    "baseline_area_score",
}


def _split_names(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def _source_tree_sha256(app_dir):
    digest = hashlib.sha256()
    files = sorted(
        path for path in app_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".h", ".hpp"}
    )
    if not files:
        raise RuntimeError(f"No source files found in {app_dir}")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def load_reference_baselines(
    manifest_path,
    *,
    required_kernels,
    forbidden_kernels=(),
    expected_device,
    expected_clock_period_ns,
    expected_toolchain_version,
    epsilon,
    source_root=None,
    verify_source_hashes=True,
):
    """Load and strictly validate one successful neutral point per kernel."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Neutral-baseline manifest not found: {path}")

    required = _split_names(required_kernels)
    forbidden = _split_names(forbidden_kernels)
    if verify_source_hashes:
        source_root = (
            Path(source_root).expanduser().resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[1]
            / "Data" / "ApplicationDataset"
        )
    references = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise RuntimeError(
                f"{path} is missing columns: {sorted(missing_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            kernel = row["kernel"].strip()
            if not kernel:
                raise RuntimeError(f"Empty kernel at {path}:{line_number}")
            if kernel in references:
                raise RuntimeError(f"Duplicate neutral baseline for {kernel}")
            if row["status"].strip().lower() != "success":
                # Failed rows are useful provenance, but cannot anchor a target.
                continue
            if kernel in forbidden:
                raise RuntimeError(
                    f"Held-out test kernel {kernel!r} is present in {path}. "
                    "Remove it until --evaluate_test is explicitly requested."
                )
            if row["device"].strip() != str(expected_device):
                raise RuntimeError(
                    f"Device mismatch for {kernel}: {row['device']!r} != "
                    f"{expected_device!r}"
                )
            clock = float(row["clock_period_ns"])
            if not math.isclose(
                clock, float(expected_clock_period_ns), rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError(
                    f"Clock mismatch for {kernel}: {clock} != "
                    f"{expected_clock_period_ns} ns"
                )
            if len(row["source_sha256"].strip()) != 64:
                raise RuntimeError(f"Invalid source SHA-256 for {kernel}")
            if verify_source_hashes:
                app_dir = source_root / kernel
                actual_source_hash = _source_tree_sha256(app_dir)
                if actual_source_hash != row["source_sha256"].strip().lower():
                    raise RuntimeError(
                        f"Neutral baseline for {kernel} is stale: source tree "
                        f"hash {actual_source_hash} != manifest "
                        f"{row['source_sha256'].strip().lower()}"
                    )
            if not row["toolchain_id"].strip():
                raise RuntimeError(f"Missing toolchain_id for {kernel}")
            if str(expected_toolchain_version) not in row["toolchain_id"]:
                raise RuntimeError(
                    f"Toolchain mismatch for {kernel}: expected version "
                    f"{expected_toolchain_version!r} in {row['toolchain_id']!r}"
                )

            latency = float(row["baseline_latency_ms"])
            area = float(row["baseline_area_score"])
            if not math.isfinite(latency) or latency <= 0.0:
                raise RuntimeError(f"Invalid neutral latency for {kernel}: {latency}")
            if not math.isfinite(area) or area <= 0.0:
                raise RuntimeError(f"Invalid neutral area for {kernel}: {area}")
            references[kernel] = {
                "perf": math.log2(latency + float(epsilon)),
                "area": math.log2(area + float(epsilon)),
                "latency_ms": latency,
                "area_score": area,
            }

    missing = required - set(references)
    if missing:
        raise RuntimeError(
            "Neutral baselines are missing for required kernels: "
            + ", ".join(sorted(missing))
        )
    return references


class ReferenceDeltaDataset(Dataset):
    """Attach reference log2 values and measured-minus-reference deltas."""

    def __init__(self, dataset, references, targets):
        self.dataset = dataset
        self.references = references
        self.targets = tuple(targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        kernel = sample.kernel
        if isinstance(kernel, (list, tuple)):
            if len(kernel) != 1:
                raise RuntimeError(f"Expected one kernel, got {kernel!r}")
            kernel = kernel[0]
        if kernel not in self.references:
            raise RuntimeError(f"No neutral baseline loaded for {kernel}")
        for target in self.targets:
            if target not in ("perf", "area"):
                raise RuntimeError(
                    f"reference_delta does not define target {target!r}"
                )
            absolute = getattr(sample, target).reshape(-1)
            if absolute.numel() != 1:
                raise RuntimeError(f"Expected scalar {target} for {kernel}")
            baseline = torch.tensor(
                [self.references[kernel][target]], dtype=torch.float32
            )
            setattr(sample, f"{target}_baseline", baseline)
            setattr(sample, f"{target}_delta", absolute - baseline)
        return sample
