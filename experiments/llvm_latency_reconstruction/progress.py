"""Read training status without importing a model or allocating GPU memory."""
import argparse
import ast
import json
from pathlib import Path
import re


def progress(root):
    result = {}
    for name in ("stage1", "stage2", "stage3"):
        marker = root / (name + ".complete.json")
        log = root / "logs" / (name + ".log")
        if marker.exists():
            r = json.loads(marker.read_text())
            result[name] = {k: r[k] for k in ("status", "global_step", "epoch")}
        elif log.exists():
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 65536))
                tail = f.read().decode(errors="replace")
            records = []
            for match in re.finditer(r"\{[^\r\n]*'(?:loss|eval_loss)'[^\r\n]*\}", tail):
                try:
                    records.append(ast.literal_eval(match.group()))
                except (ValueError, SyntaxError):
                    pass
            steps = re.findall(r"(\d+)/(160|60)\s*\[([^\]]+)\]", tail)
            result[name] = {"status": "log_present_not_completed",
                            "last_progress": steps[-1] if steps else None,
                            "last_metrics": records[-2:],
                            "traceback": "Traceback (most recent call last)" in tail,
                            "log_bytes": log.stat().st_size}
        else:
            result[name] = {"status": "not_started"}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(progress(a.runtime), indent=2))
