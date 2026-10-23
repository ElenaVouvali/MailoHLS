#!/usr/bin/env python3
"""Write the reproducibility manifest for the canonical SALT JSONL."""
import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = [root / "SALT_experiments.jsonl", root / "plots/objective_response/objective_response.py"]
manifest = {
    "canonical_jsonl": "SALT_experiments.jsonl",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "tool": "Vitis HLS 2021.1",
    "device": "ZCU104 / xczu7ev-ffvc1156-2-e",
    "clock_period_ns": 10.0,
    "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in files if path.exists()},
}
(root / "SALT_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
