"""Hash a completed native trainer output; never manufacture a completion marker."""
import argparse
import json
import math
from pathlib import Path
from prepare import sha256


def record(checkpoint):
    weights = list(checkpoint.glob("adapter_model.*"))
    states = sorted(checkpoint.glob("checkpoint-*/trainer_state.json"))
    if (checkpoint / "trainer_state.json").exists():
        states.append(checkpoint / "trainer_state.json")
    if not weights or not states:
        raise ValueError("Missing trained adapter or Trainer state")
    state = max((json.loads(p.read_text()) for p in states), key=lambda s: s["global_step"])
    if state["global_step"] <= 0:
        raise ValueError("Checkpoint has no optimizer updates")
    for row in state.get("log_history", []):
        for key in ("loss", "eval_loss", "grad_norm"):
            if key in row and not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite training diagnostic: {key}")
    files = [p for p in checkpoint.iterdir() if p.is_file()]
    for directory in checkpoint.glob("best_custom_stage*"):
        files.extend(p for p in directory.iterdir() if p.is_file())
    return {
        "status": "completed_new_reconstruction_stage",
        "checkpoint": str(checkpoint.resolve()),
        "global_step": state["global_step"], "epoch": state.get("epoch"),
        "sha256": {str(p.relative_to(checkpoint)): sha256(p) for p in sorted(files)},
        "log_history": state.get("log_history", []),
        "not_recovered_original_paper_weights": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = record(a.checkpoint)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "global_step", "epoch")}))
