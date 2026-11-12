"""Count candidate-scoring work on CPU; do not load an LLM or alter a run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LLM_branch.train import train_SFT_xattn_new as trainer


def analyze(run):
    data = run / "prepared/stage2_matched.jsonl"
    split_path = run / "prepared/stage2_matched_split.json"
    registry_path = run / "seed_123/A/directive_domain_registry.json"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    split = json.loads(split_path.read_text())
    validation = [rows[i] for i in split["val_jsonl_idx"]]
    registry = trainer.load_directive_domain_registry(str(registry_path))
    cases = trainer.build_selection_cases(
        validation, "PARETO_LATENCY", max_kernels=0, cases_per_kernel_device=4
    )
    records = []
    for case in cases:
        gold = trainer.parse_assignment_dict(case.reference_target)
        chosen = {}
        forwards = 0
        plan = trainer.extract_ordered_lhs_plan(case.source_text)
        for _, lhs in plan:
            _, candidates = trainer.constrained_rhs_candidates(
                case.kernel_name, lhs, registry, chosen
            )
            # The maintained decoder skips forced singleton candidates.
            forwards += len(candidates) if len(candidates) > 1 else 0
            chosen[lhs.strip().upper()] = gold[lhs.strip().upper()]
        records.append(dict(
            kernel=case.kernel_name, jsonl_idx=case.row.get("_jsonl_idx"),
            directive_fields=len(plan), teacher_forced_model_forwards=forwards,
        ))
    return dict(
        schema="historical-gmt-validation-workload-v1",
        method="CPU reconstruction from prepared validation rows; not a live profiler or timing benchmark",
        input_sha256={str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (data, split_path, registry_path)},
        validation_loss_examples=len(validation), reconstructed_selection_cases=len(cases),
        teacher_forced_model_forwards=sum(r["teacher_forced_model_forwards"] for r in records),
        free_running_additional_forwards="Not counted: depends on model-selected earlier directives",
        structural_candidate_batch_size=1, prefix_kv_cache_reused=False,
        smoke_selection_evaluations_per_arm=3,
        smoke_selection_evaluations_both_arms=6,
        evaluation_schedule_note="Configured start, step 1 and step 2; counts assume successful completion",
        per_case=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(analyze(args.run), indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(result)
    else:
        print(result, end="")
