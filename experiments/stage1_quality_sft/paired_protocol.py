"""Authenticate the frozen V4 54-request cohort and export exact QoR pools.

Only raw rows assigned to the authenticated seed-123 train/validation indices
are read.  The held-out Kalman test partition is never decoded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from experiments.stage1_quality_sft import core
from experiments.stage1_quality_sft.build import _budget_row
from experiments.stage1_quality_sft.observability import audit as audit_observability
from experiments.stage1_quality_sft.support import (
    constrained_rhs_candidates, family_id_from_kernel_name, load_registry,
    normalize_kname, sha,
)
from LLM_branch.common import mailohls_contract


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION = Path.home() / "mailohls_runs/stage1_smoke_inference_check_v4/validation54.jsonl"
DEFAULT_RUNS = Path.home() / "mailohls_runs/stage1_quality_sft_latency_v4"
DEFAULT_RESULT = ROOT / "artifacts/stage1_quality_sft/v4_paired_20261105"
PREDECLARED_SYNTHESIS_STRATA = (
    ("machsuite-viterbi", "xcu200-fsgd2104-2-e", 3.33, "uniform-100"),
    ("machsuite-viterbi", "xczu7ev-ffvc1156-2-e", 10.0, "uniform-025"),
    ("rodinia_pathfinder_0_baseline_0", "xcu200-fsgd2104-2-e", 10.0, "uniform-025"),
    ("rodinia_pathfinder_0_baseline_0", "xczu7ev-ffvc1156-2-e", 5.0, "uniform-100"),
    ("rodinia_pathfinder_4_doublebuffer_0", "xcu200-fsgd2104-2-e", 5.0, "uniform-050"),
    ("rodinia_pathfinder_4_doublebuffer_0", "xczu7ev-ffvc1156-2-e", 3.33, "uniform-025"),
)
DATASET = ROOT / "artifacts/llm/mailohls_sft.jsonl"
DATASET_MANIFEST = ROOT / "artifacts/llm/mailohls_sft.manifest.json"
SOURCE_SIDECAR = ROOT / "artifacts/llm/mailohls_sft.sources.json"
FAMILY_SPLIT = ROOT / "mailohls_runs/mailohls_final_family_split_s123.json"
REGISTRY = ROOT / "mailohls_runs/stage1_final_final_latency_s123/directive_domain_registry.json"
PREP_MANIFEST = ROOT / "LLM_branch/Data/preprocessed_CSVS/preprocessing_manifest.json"
PROCESSED_DIR = Path.home() / "MailoHLS/LLM_branch/Data/preprocessed_CSVS"


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_gzip_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _family_split_identity(run_dir):
    split = json.loads((run_dir / "split.json").read_text())
    manifest = json.loads((run_dir / "train.manifest.json").read_text())
    if manifest["output_sha256"] != sha(run_dir / "train.jsonl"):
        raise ValueError(f"V4 {run_dir.name} validation source dataset hash mismatch")
    if manifest["parent_dataset_sha256"] != sha(DATASET):
        raise ValueError("V4 arm parent dataset does not match authenticated GNΩSIS dataset")
    if split["dataset_sha256"] != manifest["output_sha256"]:
        raise ValueError("V4 split and prepared dataset identities differ")
    return split, manifest


def build_synthesis_subset(requests):
    subset = []
    for kernel, device, clock, budget_id in PREDECLARED_SYNTHESIS_STRATA:
        matching = [r for r in requests if r["kernel_name"] == kernel and r["device"] == device
                    and float(r["clock_period"]) == clock and r["resource_budget_id"] == budget_id]
        if len(matching) != 1:
            raise ValueError(f"Predeclared synthesis stratum must identify one request: {(kernel, device, clock, budget_id)}")
        row = matching[0]
        subset.append({k: row[k] for k in ("request_id", "kernel_name", "source_key", "source_file",
                       "device", "clock_period", "resource_budget_id", "objective")})
    payload = {
        "schema": "stage1-v4-paired-synthesis-subset-v1",
        "selection_seed": 20261105,
        "selection_rule": "Six request strata declared before inference outcomes; source/kernel, device, clock and budget are fixed; any stratum collision is resolved by SHA256(seed:request_id). No predictions or QoR rankings are used.",
        "strata": [list(x) for x in PREDECLARED_SYNTHESIS_STRATA],
        "requests": subset,
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def freeze_synthesis_subset(output_dir):
    output_dir = Path(output_dir)
    request_file = output_dir / "frozen_validation_requests.jsonl"
    manifest_file = output_dir / "manifest.json"
    payload = build_synthesis_subset(_jsonl(request_file))
    subset_path = output_dir / "synthesis_subset.json"
    subset_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = json.loads(manifest_file.read_text())
    manifest["synthesis_subset_sha256"] = sha(subset_path)
    manifest["synthesis_subset_request_ids"] = [r["request_id"] for r in payload["requests"]]
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return payload


def _audit_arms(run_root, request_rows, split, raw_split):
    request_ids = [r["request_id"] for r in request_rows]
    all_arm_rows = {}
    measurement_ids = set()
    for arm in ("single_reference", "multi_reference"):
        run = run_root / arm
        s, manifest = _family_split_identity(run)
        for key in ("train_families", "val_families", "train_jsonl_idx", "val_jsonl_idx", "test_jsonl_idx"):
            if s[key] != split[key]:
                raise ValueError(f"V4 arm {arm} split mismatch in {key}")
        if s["test_jsonl_idx"] or manifest.get("parent_dataset_sha256") != split.get("parent_dataset_sha256", manifest.get("parent_dataset_sha256")):
            if s["test_jsonl_idx"]:
                raise ValueError("Unexpected test rows in the V4 training/evaluation arm")
        index = set(s["train_jsonl_idx"]) | set(s["val_jsonl_idx"])
        selected = []
        with (run / "train.jsonl").open(encoding="utf-8") as stream:
            for i, line in enumerate(stream):
                if i in index:
                    row = json.loads(line)
                    selected.append(row)
        if len(selected) != len(index):
            raise ValueError(f"Missing prepared V4 rows for {arm}")
        val = [r for r in selected if r.get("_quality_split") == "validation"]
        if len(val) != 54 or len({r["_jsonl_idx"] for r in val}) != 54:
            raise ValueError(f"V4 {arm} does not have the frozen 54 validation rows")
        arm_by_id = {}
        for row in val:
            rid = core.request_id(row)
            if rid in arm_by_id:
                raise ValueError(f"Duplicate V4 validation request identity in {arm}: {rid}")
            arm_by_id[rid] = row
        if list(arm_by_id) != request_ids:
            raise ValueError(f"Ordered V4 validation requests differ in {arm}")
        all_arm_rows[arm] = selected
        measurement_ids.update(int(r["_source_measurement_jsonl_idx"]) for r in selected)

    # Confirm selected targets point to exact parent-measurement rows and that
    # each arm uses only its authenticated source train/validation partition.
    parent_rows = {}
    train_ids, val_ids, test_ids = (set(raw_split[f"{s}_jsonl_idx"]) for s in ("train", "val", "test"))
    for arm, rows in all_arm_rows.items():
        for row in rows:
            source_idx = int(row["_source_measurement_jsonl_idx"])
            expected_partition = train_ids if row["_quality_split"] == "train" else val_ids
            if source_idx not in expected_partition or source_idx in test_ids:
                raise ValueError(f"Train/validation/test leakage in {arm} at parent row {source_idx}")
    with DATASET.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i in measurement_ids:
                parent_rows[i] = json.loads(line)
    if set(parent_rows) != measurement_ids:
        raise ValueError("A V4 target references a missing authenticated parent measurement")
    for arm, rows in all_arm_rows.items():
        for row in rows:
            parent = parent_rows[int(row["_source_measurement_jsonl_idx"])]
            if parent["target"] != row["target"]:
                raise ValueError(f"{arm} target is not the exact referenced measured assignment")

    templates = json.loads((run_root / "single_reference/train.sources.json").read_text())["templates"]
    parent_manifest = json.loads(DATASET_MANIFEST.read_text())
    action_proof, eligible = audit_observability(
        list({int(idx): parent_rows[int(idx)] for idx in measurement_ids}.values()),
        parent_manifest, PREP_MANIFEST, templates, ROOT, PROCESSED_DIR,
    )
    if not eligible or any(not action_proof[r["kernel_name"]].get("eligible") for r in parent_rows.values()):
        raise ValueError("Selected V4 label/source observability audit did not fully pass")
    return all_arm_rows, parent_rows, action_proof


def prepare(validation_path=DEFAULT_VALIDATION, run_root=DEFAULT_RUNS,
            output_dir=DEFAULT_RESULT):
    validation_path, run_root, output_dir = map(Path, (validation_path, run_root, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen paired evaluation directory: {output_dir}")
    requests = _jsonl(validation_path)
    if len(requests) != 54:
        raise ValueError(f"Expected exactly 54 frozen requests, received {len(requests)}")
    if len({r.get("request_id") for r in requests}) != 54:
        raise ValueError("Validation requests contain missing or duplicate request IDs")
    if any("target" in r or "reference_target" in r for r in requests):
        raise ValueError("QoR/reference labels are forbidden in inference requests")

    dataset_manifest = json.loads(DATASET_MANIFEST.read_text())
    if dataset_manifest.get("output_sha256") != sha(DATASET):
        raise ValueError("Raw GNΩSIS dataset hash does not match its authenticated manifest")
    raw_split = json.loads(FAMILY_SPLIT.read_text())
    if raw_split.get("dataset_sha256") != sha(DATASET):
        raise ValueError("Family split does not match the authenticated raw dataset")
    val_ids = set(raw_split["val_jsonl_idx"])
    test_ids = set(raw_split["test_jsonl_idx"])
    if val_ids & test_ids:
        raise ValueError("Family split overlaps validation and test")

    run_root = run_root.resolve()
    single, multi = (run_root / arm for arm in ("single_reference", "multi_reference"))
    single_split, single_manifest = _family_split_identity(single)
    multi_split, multi_manifest = _family_split_identity(multi)
    split_identity_fields = ("train_families", "val_families", "train_jsonl_idx", "val_jsonl_idx", "test_jsonl_idx")
    if any(single_split[k] != multi_split[k] for k in split_identity_fields):
        raise ValueError("Single/multi V4 family/request partition identities differ")
    if single_split.get("test_jsonl_idx"):
        raise ValueError("V4 training split unexpectedly includes test rows")
    if single_split["val_families"] != multi_split["val_families"]:
        raise ValueError("Single/multi V4 validation family lists differ")
    matched = json.loads((run_root / "matched_arms.json").read_text())
    if (not matched.get("independent_fresh_adapters")
            or matched.get("historical_adapter_initialization")
            or matched.get("backbone_revision") != "ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912"):
        raise ValueError("V4 arm provenance is not the declared independent fresh-adapter protocol")
    contracts = [json.loads((single / "run/training_contract.json").read_text()),
                 json.loads((multi / "run/training_contract.json").read_text())]
    common = ("stage", "model", "model_revision", "objective", "prompt_schema_version",
              "tokenizer_size", "special_token_to_id", "directive_domain_registry_sha256")
    if any(any(contract.get(k) != contracts[0].get(k) for k in common) for contract in contracts[1:]):
        raise ValueError("V4 checkpoint tokenizer, prompt, source-domain, or objective contracts differ")
    if contracts[0].get("stage") != "stage1" or contracts[0].get("objective") != "PARETO_LATENCY":
        raise ValueError("V4 adapters are not latency-specialized Stage-1 checkpoints")
    if contracts[0].get("directive_domain_registry_sha256") != sha(single / "run/directive_domain_registry.json"):
        raise ValueError("Single-reference saved domain registry hash mismatch")
    if sha(single / "run/directive_domain_registry.json") != sha(multi / "run/directive_domain_registry.json"):
        raise ValueError("V4 saved directive domain registries differ")

    arm_rows, parent_rows, action_proof = _audit_arms(run_root, requests, single_split, raw_split)
    source_templates = json.loads((single / "train.sources.json").read_text())["templates"]
    families = set(single_split["val_families"])
    request_kernels = set()
    for row in requests:
        if row.get("objective") != "PARETO_LATENCY" or row.get("frequency_mode") != "specified":
            raise ValueError("Frozen V4 request objective/clock mode mismatch")
        if row["input"] != source_templates.get(row["source_key"]):
            raise ValueError(f"Request source does not match V4 source sidecar: {row['request_id']}")
        if family_id_from_kernel_name(row["kernel_name"]) not in families:
            raise ValueError(f"Validation request lies outside the V4 validation families: {row['kernel_name']}")
        if family_id_from_kernel_name(row["kernel_name"]) in raw_split["test_families"]:
            raise ValueError("Held-out test family appears in validation requests")
        if row["device"] not in mailohls_contract.DEVICE_RESOURCES:
            raise ValueError(f"Unsupported device in validation request: {row['device']}")
        request_kernels.add(row["kernel_name"])

    # Read only raw family-validation rows for kernels in the frozen request set.
    raw_validation_rows = []
    with DATASET.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i not in val_ids:
                continue
            row = json.loads(line)
            if row["kernel_name"] in request_kernels:
                row["_jsonl_idx"] = i
                row["_family"] = family_id_from_kernel_name(row["kernel_name"])
                raw_validation_rows.append(row)
    if any(r["kernel_name"] not in request_kernels for r in raw_validation_rows):
        raise AssertionError("Validation candidate-pool source filter changed")
    registry = load_registry(REGISTRY)
    validated_assignment_ids = set()
    for row in raw_validation_rows:
        signature = (row["source_key"], tuple(sorted(core.parse(row["target"]).items())))
        if signature in validated_assignment_ids:
            continue
        core.validate_complete_assignment(row, registry, constrained_rhs_candidates,
                                          normalize_kname, core.parse)
        validated_assignment_ids.add(signature)
    banks, duplicate_rows = core.build_banks(raw_validation_rows)

    metric_requests, unsupported = [], []
    for prompt in requests:
        base = {
            "kernel_name": prompt["kernel_name"], "source_key": prompt["source_key"],
            "source_file": prompt["source_file"], "device": prompt["device"],
            "clock_period": float(prompt["clock_period"]), "objective": "PARETO_LATENCY",
            "_family": family_id_from_kernel_name(prompt["kernel_name"]),
            **{key: float(prompt["quality_budget_fractions"][key]) for key in core.BUDGET},
        }
        pool = banks.get(core.physical_context(base), {})
        if not pool:
            raise ValueError(f"No raw validation measurements match request: {prompt['request_id']}")
        req = core.build_compact_request(base, banks)
        if req["request_id"] != prompt["request_id"]:
            raise ValueError(f"Candidate request identity changed: {prompt['request_id']}")
        req["prompt_request_id"] = prompt["request_id"]
        metric_requests.append(req)
    subset = build_synthesis_subset(requests)["requests"]
    pool_records = [core.candidate_pool_record(context, bank)
                    for context, bank in sorted(banks.items(), key=lambda x: repr(x[0]))]

    output_dir.mkdir(parents=True, exist_ok=True)
    target_requests = output_dir / "frozen_validation_requests.jsonl"
    target_requests.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in requests))
    metric_path = output_dir / "metric_requests.jsonl.gz"
    pools_path = output_dir / "candidate_pools.jsonl.gz"
    subset_path = output_dir / "synthesis_subset.json"
    _write_gzip_jsonl(metric_path, metric_requests)
    _write_gzip_jsonl(pools_path, pool_records)
    subset_manifest = build_synthesis_subset(requests)
    subset_path.write_text(json.dumps(subset_manifest, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "stage1-v4-paired-validation-v1", "authenticated": True,
        "request_count": len(requests), "ordered_request_ids": [r["request_id"] for r in requests],
        "request_file_sha256": sha(target_requests), "original_prompt_file_sha256": sha(validation_path),
        "dataset_sha256": sha(DATASET), "dataset_manifest_sha256": sha(DATASET_MANIFEST),
        "family_split_sha256": sha(FAMILY_SPLIT), "v4_single_dataset_sha256": single_manifest["output_sha256"],
        "v4_multi_dataset_sha256": multi_manifest["output_sha256"],
        "v4_single_split_sha256": sha(single / "split.json"), "v4_multi_split_sha256": sha(multi / "split.json"),
        "v4_validation_families": single_split["val_families"],
        "v4_training_families": single_split["train_families"], "v4_test_indices": [],
        "test_labels_decoded": False, "held_out_test_family": raw_split["test_families"],
        "v4_matched_arms_protocol_sha256": matched["training_protocol_sha256"],
        "matched_arms_declared_validation_request_sha256": matched["validation_request_sha256"],
        "adapters": {arm: {
            "adapter_sha256": sha(run_root / arm / "run/adapter_model.safetensors"),
            "training_contract_sha256": sha(run_root / arm / "run/training_contract.json"),
            "tokenizer_sha256": sha(run_root / arm / "run/tokenizer.json"),
            "registry_sha256": sha(run_root / arm / "run/directive_domain_registry.json"),
        } for arm in ("single_reference", "multi_reference")},
        "tokenizer_and_domain_contract_equal": True,
        "action_observability": action_proof,
        "selected_v4_rows": len(arm_rows["single_reference"]),
        "selected_parent_measurement_rows": len(parent_rows),
        "raw_validation_candidate_rows": len(raw_validation_rows),
        "distinct_validated_assignments": len(validated_assignment_ids),
        "candidate_pool_count": len(pool_records), "identical_duplicate_rows": duplicate_rows,
        "unsupported_requests": unsupported,
        "candidate_pool_file_sha256": sha(pools_path), "metric_request_file_sha256": sha(metric_path),
        "synthesis_subset_sha256": sha(subset_path),
        "synthesis_subset_request_ids": [r["request_id"] for r in subset],
        "families": sorted({family_id_from_kernel_name(r["kernel_name"]) for r in requests}),
        "kernels": sorted(request_kernels),
        "device_clock_budget_counts": {},
        "measurement_tool_version": "not recorded in the authenticated GNΩSIS raw-row schema; QoR is dataset-measured with compiler version unknown",
    }
    from collections import Counter
    count = Counter((r["device"], float(r["clock_period"]), r["resource_budget_id"]) for r in requests)
    manifest["device_clock_budget_counts"] = {f"{d}@{c:g}ns/{b}": n for (d, c, b), n in sorted(count.items())}
    manifest["eligibility_summary"] = {
        "all_selected_train_and_validation_targets_match_parent_measurements": True,
        "all_selected_source_actions_directly_measured": all(v.get("eligible") for v in action_proof.values()),
        "eligible_kernel_count": len(action_proof),
        "unknown_action_count": sum(len(v.get("unknown_action_ids", [])) for v in action_proof.values()),
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--v4-runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--freeze-synthesis-only", action="store_true",
                        help="Freeze the predeclared synthesis subset into an already prepared result directory")
    args = parser.parse_args()
    if args.freeze_synthesis_only:
        result = freeze_synthesis_subset(args.output_dir)
        print(json.dumps({"subset_sha256": result["sha256"], "request_ids": [r["request_id"] for r in result["requests"]]}, indent=2))
        return
    result = prepare(args.validation, args.v4_runs, args.output_dir)
    print(json.dumps({k: result[k] for k in ("request_count", "request_file_sha256", "manifest_sha256",
        "v4_validation_families", "kernels", "raw_validation_candidate_rows", "candidate_pool_count",
        "eligibility_summary")}, indent=2))


if __name__ == "__main__":
    main()
