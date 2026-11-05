"""Write immutable model/input/decoder identities for a paired inference run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.stage1_quality_sft.support import sha

ROOT = Path(__file__).resolve().parents[2]


def freeze(output_dir, v4_root, production, gpu_uuid):
    output_dir, v4_root, production = map(Path, (output_dir, v4_root, production))
    destination = output_dir / "inference_execution_manifest.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace inference execution identity: {destination}")
    output = output_dir
    model_paths = {
        "single_reference": v4_root / "single_reference/run",
        "multi_reference": v4_root / "multi_reference/run",
        "production_latency": production,
    }
    models = {}
    for name, path in model_paths.items():
        registry_path = path / "directive_domain_registry.json"
        if not registry_path.is_file():
            registry_path = path.parent / "directive_domain_registry.json"
        required = ("adapter_model.safetensors", "training_contract.json", "tokenizer.json")
        missing = [item for item in required if not (path / item).is_file()]
        if not registry_path.is_file():
            missing.append("directive_domain_registry.json")
        if missing:
            raise FileNotFoundError(f"{name} checkpoint is missing contract files: {missing}")
        models[name] = {
            "directory": str(path.resolve()),
            "adapter_sha256": sha(path / "adapter_model.safetensors"),
            "training_contract_sha256": sha(path / "training_contract.json"),
            "tokenizer_sha256": sha(path / "tokenizer.json"),
            "registry_path": str(registry_path.resolve()),
            "registry_sha256": sha(registry_path),
        }
    manifest = {
        "schema": "stage1-v4-paired-inference-execution-v1",
        "validation_manifest_sha256": sha(output / "manifest.json"),
        "frozen_request_sha256": sha(output / "frozen_validation_requests.jsonl"),
        "candidate_pool_sha256": sha(output / "candidate_pools.jsonl.gz"),
        "metric_request_sha256": sha(output / "metric_requests.jsonl.gz"),
        "source_tool_policy_sha256": sha(ROOT / "experiments/stage1_quality_sft/policies/vitis_hls_2021_1_partition_policy.json"),
        "git_head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu_uuid": gpu_uuid,
        "generation": {"mode": "greedy", "num_samples": 1, "base_revision": "ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912",
                       "same_qor_free_ordered_requests_for_all_models": True,
                       "policies": ["legacy", "source_tool_v1"]},
        "models": models,
        "kalman_labels_loaded": False,
    }
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/stage1_quality_sft/v4_paired_20261105")
    ap.add_argument("--v4-root", type=Path, default=Path.home() / "mailohls_runs/stage1_quality_sft_latency_v4")
    ap.add_argument("--production", type=Path, default=ROOT / "mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1")
    ap.add_argument("--gpu-uuid", required=True)
    args = ap.parse_args()
    print(freeze(args.output_dir, args.v4_root, args.production, args.gpu_uuid))


if __name__ == "__main__":
    main()
