"""Verify the completed framework bundle and optionally run the four predictions."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--wait-pid", type=int)
    p.add_argument("--inference", action="store_true")
    p.add_argument("--timeout-hours", type=float, default=24)
    a = p.parse_args()
    root = a.runtime.resolve()
    repo = Path(__file__).resolve().parents[2]
    exp = Path(__file__).resolve().parent
    deadline = time.monotonic() + a.timeout_hours * 3600
    while not all((root / f"stage{i}.complete.json").exists() for i in (1, 2, 3)):
        if not a.wait_pid:
            raise ValueError("Three completed training stages are required")
        os.kill(a.wait_pid, 0)  # Fail if the training process exited prematurely.
        if time.monotonic() >= deadline:
            raise TimeoutError("Training remains incomplete; no framework bundle was published")
        time.sleep(15)
    stages = {f"stage{i}": json.loads((root / f"stage{i}.complete.json").read_text()) for i in (1, 2, 3)}
    selected = {
        "stage1_adapter": root / "stage1/best_custom_stage1/adapter_model.safetensors",
        "stage1_tokenizer": root / "stage1/best_custom_stage1/tokenizer.json",
        "stage2_fusion": root / "stage2/best_custom_stage2/harp_xattn.pt",
        "stage3_fusion": root / "stage3/best_custom_stage3/harp_xattn.pt",
        "gnn": root / "assets/gnn_val.pth",
        "encoders": root / "assets/encoders.klepto",
    }
    identities = {k: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                  for k, path in selected.items()}
    for key, stage, relative in (
        ("stage1_adapter", "stage1", "best_custom_stage1/adapter_model.safetensors"),
        ("stage1_tokenizer", "stage1", "best_custom_stage1/tokenizer.json"),
        ("stage2_fusion", "stage2", "best_custom_stage2/harp_xattn.pt"),
        ("stage3_fusion", "stage3", "best_custom_stage3/harp_xattn.pt"),
    ):
        if identities[key]["sha256"] != stages[stage]["sha256"][relative]:
            raise ValueError(f"Trained checkpoint identity changed: {key}")
    bank = json.loads((root / "memory_rebuild_manifest.json").read_text())
    external = json.loads((root / "external_memory_manifest.json").read_text())
    if bank["checkpoint_sha256"] != identities["gnn"]["sha256"] or external["checkpoint_sha256"] != bank["checkpoint_sha256"]:
        raise ValueError("Training and external inference GNNs differ")
    for name, info in bank["kernels"].items():
        if sha256(root / "memory" / (name + ".memory.pt")) != info["memory_sha256"]:
            raise ValueError("Training memory bank changed")
    for name, info in external["kernels"].items():
        if sha256(root / "external_memory" / (name + ".memory.pt")) != info["memory_sha256"]:
            raise ValueError("External memory identity changed")
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0, str(repo / "LLM_branch/train"))
    import train_SFT_xattn as s
    tok = AutoTokenizer.from_pretrained(os.environ["LLVM_BASE_MODEL"], local_files_only=True)
    tok.add_special_tokens({"additional_special_tokens": [g["token"] for g in s.GOALS.values()] + s.SOURCE_PLACEHOLDER_TOKENS + s.TARGET_PLACEHOLDER_TOKENS})
    saved = AutoTokenizer.from_pretrained(selected["stage1_tokenizer"].parent, local_files_only=True)
    if tok.get_vocab() != saved.get_vocab():
        raise ValueError("Saved parent tokenizer differs from inference token IDs")
    bundle = {
        "status": "completed_new_LLVM_latency_framework_checkpoint",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "original_weights_recovered": False, "components": identities,
        "base_model_snapshot": os.environ["LLVM_BASE_MODEL"],
        "objective": "PARETO_LATENCY_EXTREME", "target": "ZCU104", "clock_period_ns": 10,
        "memory_shape": [64, 64], "tokenizer_vocab_size": len(saved),
        "stage_updates": {k: v["global_step"] for k, v in stages.items()},
        "selected_validation": {k: json.loads((root / k / ("best_custom_" + k) / "best_selection_metrics.json").read_text()) for k in stages},
        "dataset_sha256": sha256(root / "dataset.jsonl"),
        "split_sha256": sha256(root / "split.json"),
        "source_sha256": {str(path.relative_to(repo)): sha256(path) for path in
            [repo / "LLM_branch/train/train_SFT_xattn.py", repo / "LLM_branch/train/train_DPO_harp_xattn.py", repo / "LLM_branch/inference/eval_stage1_stage2_stage3.py", repo / "GNN_branch/model.py"]},
        "physical_qor_improvement": "NOT EVALUATED",
    }
    dst = root / "framework_checkpoint.json"
    dst.write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"framework_checkpoint": str(dst), "sha256": sha256(dst), "stage_updates": bundle["stage_updates"]}), flush=True)
    if not a.inference:
        return
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    # The training process may still be returning from its final hash operation.
    def occupied():
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
        return any(parts[1].strip() == gpu and int(parts[0].strip()) != os.getpid()
                   for line in apps.splitlines() if len(parts := line.split(",")) == 2)
    while occupied():
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU occupied; inference not allocated")
        time.sleep(15)
    subprocess.run(["bash", str(exp / "run_inference.sh")], cwd=repo, check=True)
    cases = {r["kernel_name"]: r for r in map(json.loads, (exp / "cases.jsonl").read_text().splitlines())}
    rows = [json.loads(line) for line in (root / "predictions/llvm_latency_four_kernels.jsonl").read_text().splitlines()]
    if {r["kernel_name"] for r in rows} != set(cases) or len(rows) != 4:
        raise ValueError("Incomplete external inference")
    compact = []
    for row in rows:
        case = cases[row["kernel_name"]]
        candidate = next(c for c in row["candidates"] if c["is_unique"])
        pairs = re.findall(r"^(auto\{[^}]+\})\s*=\s*(.+)$", candidate["canonical_prediction"], re.MULTILINE)
        expected = set(re.findall(r"auto\{[^}]+\}", case["code"]))
        if len(pairs) != len(expected) or {k for k, _ in pairs} != expected:
            raise ValueError("Predicted assignment does not match source directive schema")
        compact.append({"kernel_name": row["kernel_name"], "objective": bundle["objective"],
            "framework_checkpoint_sha256": sha256(dst), "source_sha256": hashlib.sha256(case["code"].encode()).hexdigest(),
            "directives": dict(pairs), "source_schema_complete": True,
            "physical_qor_improvement": "NOT EVALUATED"})
    output = root / "predictions/predicted_directives.jsonl"
    output.write_text("".join(json.dumps(r) + "\n" for r in compact))
    print(json.dumps({"predictions": str(output), "kernels": 4, "fields": sum(len(r["directives"]) for r in compact)}), flush=True)


if __name__ == "__main__":
    main()
