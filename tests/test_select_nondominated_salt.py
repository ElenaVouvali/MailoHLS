import json
import subprocess
from pathlib import Path


def test_known_dominated_points_are_flagged(tmp_path):
    out = tmp_path / "candidates.jsonl"
    subprocess.run(["python3", "scripts/select_nondominated_salt.py", "SALT_experiments.jsonl",
                    "--output", str(out)], check=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    dominated = {(r["kernel"], r["objective"]) for r in rows if r["dominated"]}
    assert dominated == {
        ("jacobi-2d", "LATENCY"),
        ("chstone-aes", "AREA"),
        ("chstone-aes", "ADP"),
        ("chstone-jpeg", "ADP"),
    }
