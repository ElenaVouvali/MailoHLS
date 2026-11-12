"""Materialize measured ZCU104/10 ns designs for the archived LLVM trainer.

No MLIR model, target-conditioning code, or invented QoR labels are used.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def materialize(compact, sources, memory, output):
    templates = json.loads(sources.read_text())["templates"]
    allowed = {p.name.removesuffix(".memory.pt") for p in memory.glob("*.memory.pt")}
    counts = collections.Counter()
    source_hashes = {}
    dropped = collections.Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with compact.open() as f, output.open("w") as dst:
        for idx, line in enumerate(f):
            r = json.loads(line)
            if r["device"] != "xczu7ev-ffvc1156-2-e" or float(r["clock_period"]) != 10:
                continue
            kernel = r["kernel_name"]
            if kernel not in allowed:
                raise ValueError(f"No original LLVM memory for {kernel}")
            if not all(math.isfinite(float(r[k])) and float(r[k]) > 0 for k in ("latency", "area")):
                dropped["nonpositive_or_nonfinite_measurement"] += 1
                continue
            source = templates[r["source_key"]]
            expected = set(re.findall(r"auto\{([^}]+)\}", source))
            actual = set(re.findall(r"auto\{([^}]+)\}", r["target"]))
            if not expected or expected != actual:
                raise ValueError(f"Source/target directive schema mismatch: {kernel}")
            # Keep measured numbers and their row identity; the historical trainer
            # consumes input/target/kernel_name/latency/area and no device tokens.
            row = dict(r, input=source, reconstruction_measurement_row=idx)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[kernel] += 1
            source_hashes[kernel] = hashlib.sha256(source.encode()).hexdigest()
    if not counts:
        raise ValueError("No compatible measured examples")
    return {
        "status": "reconstructed_dataset_not_original_archived_jsonl",
        "device": "xczu7ev-ffvc1156-2-e", "clock_period_ns": 10,
        "examples": sum(counts.values()), "kernels": len(counts),
        "counts": dict(sorted(counts.items())), "dropped": dict(dropped),
        "compact_sha256": sha256(compact), "sources_sha256": sha256(sources),
        "dataset_sha256": sha256(output), "source_sha256": source_hashes,
        "memory_sha256": {p.name: sha256(p) for p in sorted(memory.glob("*.memory.pt"))},
        "excluded_evaluation_kernels": ["trmm-opt", "jacobi-2d", "covariance", "syr2k"],
        "limitation": "New reconstruction from local measured tables; not byte-identical to the unavailable 1.1 GB archived training JSONL.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--memory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    report = materialize(a.compact, a.sources, a.memory, a.output)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "examples", "kernels", "dropped", "dataset_sha256")}))


if __name__ == "__main__":
    main()
