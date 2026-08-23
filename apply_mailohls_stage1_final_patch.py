
# coding: utf-8
# Apply the final MailoHLS Stage-1 validation/reproducibility patch.
#
# Inspected base:
#   ElenaVouvali/MailoHLS, stage2-analysis-refactor
#   LLM_branch/train/train_SFT_xattn_new.py
#     blob fa459c40a67497acc596c7ac2484eea9b909bb13
#   LLM_branch/tests/test_stage1_final_contract.py
#     blob b34be8bec8fed2b54ab1f7c60c8292dae101d05e
#
# Intentionally NOT changed:
#   - Stage-1 CE objective
#   - candidate-ranking training loss (stays off)
#   - LoRA architecture/hyperparameters
#   - family sampling
#   - counterfactual-budget training weights
#   - SFT dataset/preprocessing
#
# Run from the MailoHLS repository root:
#   python apply_mailohls_stage1_final_patch.py

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

ROOT = Path.cwd().resolve()
TRAIN = ROOT / "LLM_branch/train/train_SFT_xattn_new.py"
TEST = ROOT / "LLM_branch/tests/test_stage1_final_contract.py"

EXPECTED = {
    TRAIN: "fa459c40a67497acc596c7ac2484eea9b909bb13",
    TEST: "b34be8bec8fed2b54ab1f7c60c8292dae101d05e",
}


def blob_sha(path):
    return subprocess.check_output(
        ["git", "hash-object", str(path)],
        cwd=ROOT,
        text=True,
    ).strip()


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one source block, found {count}. "
            "No files were changed."
        )
    return text.replace(old, new, 1)


def replace_function(source, name, replacement):
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one top-level function {name}, found {len(matches)}"
        )
    node = matches[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [
        replacement.rstrip() + "\n"
    ]
    return "".join(lines)


def replace_method(source, class_name, method_name, replacement):
    tree = ast.parse(source)
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise RuntimeError(
            f"Expected one class {class_name}, found {len(classes)}"
        )
    methods = [
        node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise RuntimeError(
            f"Expected one {class_name}.{method_name}, found {len(methods)}"
        )
    node = methods[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [
        replacement.rstrip() + "\n"
    ]
    return "".join(lines)


MARGIN_FUNCTIONS = r'''
def summarize_candidate_margins(
    rows: List[dict],
) -> Dict[str, object]:
    # Free-running local scores.  Cascade exclusions are deliberately kept in
    # the denominator instead of being silently dropped.
    all_records = [
        record
        for row in rows
        for record in row.get("candidate_margins", [])
    ]
    valid_records = [
        record
        for record in all_records
        if record.get("margin") is not None
    ]
    cascade_records = [
        record
        for record in all_records
        if bool(record.get("gold_excluded_by_prior_decision", False))
    ]

    def aggregate(group):
        if not group:
            return None
        margins = [float(record["margin"]) for record in group]
        ranks = [float(record["gold_rank"]) for record in group]
        return {
            "count": len(group),
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "positive_fraction": float(
                np.mean([margin > 0.0 for margin in margins])
            ),
            "mean_gold_rank": float(np.mean(ranks)),
        }

    by_kind = defaultdict(list)
    by_kernel = defaultdict(list)
    for row in rows:
        for record in row.get("candidate_margins", []):
            if record.get("margin") is None:
                continue
            by_kind[record["kind"]].append(record)
            by_kernel[row["kernel_name"]].append(record)

    overall = aggregate(valid_records)
    return {
        "candidate_margin_count": len(valid_records),
        "candidate_margin_mean": (
            overall["mean_margin"] if overall else None
        ),
        "candidate_margin_median": (
            overall["median_margin"] if overall else None
        ),
        "candidate_margin_positive_fraction": (
            overall["positive_fraction"] if overall else None
        ),
        "candidate_gold_mean_rank": (
            overall["mean_gold_rank"] if overall else None
        ),
        "candidate_margin_per_kind": {
            kind: aggregate(group)
            for kind, group in sorted(by_kind.items())
        },
        "candidate_margin_per_kernel": {
            kernel: aggregate(group)
            for kernel, group in sorted(by_kernel.items())
        },
        "candidate_cascade_site_count": len(all_records),
        "candidate_cascade_excluded_count": len(cascade_records),
        "candidate_cascade_excluded_fraction": (
            float(len(cascade_records) / len(all_records))
            if all_records else None
        ),
    }


def summarize_teacher_forced_candidates(
    rows: List[dict],
) -> Dict[str, object]:
    # Evaluation only: local RHS ranking when earlier supervised RHS values are
    # fixed to their reference values.
    records = [
        record
        for row in rows
        for record in row.get("teacher_forced_candidates", [])
    ]
    if not records:
        return {
            "teacher_forced_candidate_count": 0,
            "teacher_forced_top1_accuracy": None,
            "teacher_forced_mrr": None,
            "teacher_forced_mean_margin": None,
            "teacher_forced_per_kind": {},
        }

    def aggregate(group):
        ranks = [float(record["gold_rank"]) for record in group]
        margins = [
            float(record["margin"])
            for record in group
            if record.get("margin") is not None
        ]
        return {
            "count": len(group),
            "top1_accuracy": float(
                np.mean([rank == 1.0 for rank in ranks])
            ),
            "mrr": float(np.mean([1.0 / rank for rank in ranks])),
            "mean_margin": (
                float(np.mean(margins)) if margins else None
            ),
        }

    by_kind = defaultdict(list)
    for record in records:
        by_kind[record["kind"]].append(record)

    overall = aggregate(records)
    return {
        "teacher_forced_candidate_count": len(records),
        "teacher_forced_top1_accuracy": overall["top1_accuracy"],
        "teacher_forced_mrr": overall["mrr"],
        "teacher_forced_mean_margin": overall["mean_margin"],
        "teacher_forced_per_kind": {
            kind: aggregate(group)
            for kind, group in sorted(by_kind.items())
        },
    }
'''


RUN_CASE = r'''
    def _run_case(self, model, case: SelectionCase) -> dict:
        header, kernel, suffix, _ = build_prompt_sections(
            case.source_text,
            case.obj_mode,
            row=case.row,
            device_token_dropout=0.0,
        )
        base_prompt_ids = [
            token_id
            for section in (header, kernel, suffix)
            for token_id in self.tok(
                section, add_special_tokens=False
            )["input_ids"]
        ]
        prompt_ids = (
            base_prompt_ids
            + mailohls_contract.selected_clock_response_token_ids(
                case.row, self.tok
            )
        )
        if len(prompt_ids) > self.max_prompt_tokens:
            raise ValueError(
                "Validation prompt and selected-clock prefix exceed "
                "--max_length"
            )

        device = next(model.parameters()).device
        routing_start_idx = torch.tensor(
            [len(base_prompt_ids)],
            dtype=torch.long,
            device=device,
        )

        structural_memory = None
        structural_memory_mask = None
        structural_relation_mask = None
        if (
            hasattr(model, "initialized_structural_xattn")
            and model.initialized_structural_xattn
        ):
            (
                structural_memory,
                structural_memory_mask,
                structural_relation_mask,
            ) = (
                structural_memory_utils
                .get_structural_memory_pack_for_kernel(
                    self.mem_bank,
                    case.kernel_name,
                    self.max_slots,
                    self.mem_dim,
                    structural_routing=self.structural_routing,
                )
            )

        # Primary metric: normal free-running constrained decoding.
        pred, score_trace = constrained_decode_rhs_by_candidate_scoring(
            model=model,
            tok=self.tok,
            prompt_ids=prompt_ids,
            source_text=case.source_text,
            kernel_name=case.kernel_name,
            directive_domain_registry=self.directive_domain_registry,
            score_reduction=self.candidate_score_reduction,
            structural_memory=structural_memory,
            structural_memory_mask=structural_memory_mask,
            structural_relation_mask=structural_relation_mask,
            routing_start_idx=routing_start_idx,
            candidate_batch_size=self.candidate_batch_size,
            return_score_trace=True,
        )

        ref_assign = parse_assignment_dict(case.reference_target)

        # Diagnostic only: candidate scores under the correct earlier RHS
        # prefix.  This does not participate in Stage-1 training loss.
        teacher_trace = score_teacher_forced_rhs_candidates(
            model=model,
            tok=self.tok,
            prompt_ids=prompt_ids,
            source_text=case.source_text,
            kernel_name=case.kernel_name,
            reference_assignments=ref_assign,
            directive_domain_registry=self.directive_domain_registry,
            score_reduction=self.candidate_score_reduction,
            structural_memory=structural_memory,
            structural_memory_mask=structural_memory_mask,
            structural_relation_mask=structural_relation_mask,
            routing_start_idx=routing_start_idx,
            candidate_batch_size=self.candidate_batch_size,
        )

        case_id = "::".join([
            str(case.kernel_name),
            str(case.obj_mode),
            str(_norm_device(case.row.get("device", ""))),
            str(case.row.get("resource_budget_id")),
            str(case.row.get("selected_clock_period")),
            str(case.row.get("_jsonl_idx")),
        ])

        # This now mirrors Stage-1 supervision: a reference site is a real
        # decision only if >1 RHS is legal under the correct prior decisions.
        decision_keys = {
            site["lhs"].strip().upper()
            for site in teacher_trace
            if not bool(site["forced_by_semantics"])
        }
        if not decision_keys:
            raise ValueError(
                f"Validation case has no real directive decisions: "
                f"{case.kernel_name}"
            )

        def make_record(site, require_gold):
            lhs_key = site["lhs"].strip().upper()
            gold_rhs = ref_assign.get(lhs_key)
            if gold_rhs is None:
                return None
            gold_rhs = gold_rhs.strip()
            ranked = site["candidates"]

            gold_records = [
                record
                for record in ranked
                if record["rhs"].strip() == gold_rhs
            ]
            if len(gold_records) > 1:
                raise RuntimeError(
                    f"Duplicate gold candidates for "
                    f"{case.kernel_name}/{lhs_key}"
                )
            gold = gold_records[0] if gold_records else None
            if require_gold and gold is None:
                raise RuntimeError(
                    f"Teacher-forced gold candidate disappeared for "
                    f"{case.kernel_name}/{lhs_key}={gold_rhs}"
                )

            wrong = [
                record
                for record in ranked
                if record["rhs"].strip() != gold_rhs
            ]
            gold_rank = next(
                (
                    index + 1
                    for index, record in enumerate(ranked)
                    if record["rhs"].strip() == gold_rhs
                ),
                len(ranked) + 1,
            )
            best_wrong = (
                max(wrong, key=lambda record: record["score"])
                if wrong else None
            )
            margin = (
                float(gold["score"] - best_wrong["score"])
                if gold is not None and best_wrong is not None
                else None
            )
            return {
                "case_id": case_id,
                "label": site["label"],
                "lhs": site["lhs"],
                "kind": lhs_kind(site["lhs"]),
                "gold_rhs": gold_rhs,
                "predicted_rhs": ranked[0]["rhs"],
                "gold_score": (
                    float(gold["score"]) if gold is not None else None
                ),
                "best_wrong_rhs": (
                    best_wrong["rhs"] if best_wrong is not None else None
                ),
                "best_wrong_score": (
                    float(best_wrong["score"])
                    if best_wrong is not None else None
                ),
                "margin": margin,
                "gold_rank": int(gold_rank),
                "candidate_count": int(len(ranked)),
                "static_candidate_count": int(
                    site["static_candidate_count"]
                ),
                "forced_by_semantics": bool(
                    site["forced_by_semantics"]
                ),
                "gold_excluded_by_prior_decision": gold is None,
                "gold_top1": bool(gold_rank == 1),
                "site_id": (
                    f"{case.row.get('_jsonl_idx')}::"
                    f"{site['label']}::{lhs_key}"
                ),
                "paired_site_id": (
                    f"{case_id}::{site['label']}::{lhs_key}"
                ),
            }

        candidate_margins = []
        for site in score_trace:
            lhs_key = site["lhs"].strip().upper()
            if lhs_key not in decision_keys:
                continue
            record = make_record(site, require_gold=False)
            if record is not None:
                candidate_margins.append(record)

        teacher_forced_candidates = []
        for site in teacher_trace:
            lhs_key = site["lhs"].strip().upper()
            if lhs_key not in decision_keys:
                continue
            record = make_record(site, require_gold=True)
            if record is not None:
                teacher_forced_candidates.append(record)

        metrics = evaluate_prediction(case.reference_target, pred)
        pred_assign = parse_assignment_dict(pred)
        decision_correct = sum(
            pred_assign.get(key) == ref_assign.get(key)
            for key in decision_keys
        )

        return {
            "case_id": case_id,
            "kernel_name": case.kernel_name,
            "obj_mode": case.obj_mode,
            "reference_target": case.reference_target,
            "prediction": metrics["canonical_prediction"],
            "value_accuracy_over_expected": float(
                metrics["value_accuracy_over_expected"]
            ),
            "schema_compliant": bool(metrics["schema_compliant"]),
            "expected_key_match": bool(metrics["expected_key_match"]),
            "exact_design_match": bool(metrics["exact_design_match"]),
            "pragma_kind_counts": metrics["pragma_kind_counts"],
            "candidate_margins": candidate_margins,
            "teacher_forced_candidates": teacher_forced_candidates,
            "decision_site_count": len(decision_keys),
            "decision_site_correct_count": decision_correct,
            "decision_site_accuracy": (
                decision_correct / len(decision_keys)
            ),
            "forced_site_count": (
                len(ref_assign) - len(decision_keys)
            ),
            "jsonl_idx": case.row.get("_jsonl_idx"),
            "device": case.row.get("device"),
            "resource_budget_id": case.row.get("resource_budget_id"),
            "selected_clock_period": case.row.get(
                "selected_clock_period"
            ),
        }
'''


TEACHER_SCORER = r'''
@torch.no_grad()
def score_teacher_forced_rhs_candidates(
    *,
    model,
    tok,
    prompt_ids: List[int],
    source_text: str,
    kernel_name: str,
    reference_assignments: Mapping[str, str],
    directive_domain_registry: Dict[str, Dict[str, List[str]]],
    score_reduction: str = "mean",
    structural_memory: Optional[torch.Tensor] = None,
    structural_memory_mask: Optional[torch.Tensor] = None,
    structural_relation_mask: Optional[torch.Tensor] = None,
    routing_start_idx: Optional[torch.Tensor] = None,
    candidate_batch_size: int = 1,
) -> List[dict]:
    # Evaluation-only counterpart of constrained decoding. Candidate scores are
    # computed normally, but the reference RHS is appended after every site so
    # a later site's score is not contaminated by an earlier prediction error.
    assert score_reduction in {"mean", "sum"}
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be >= 1")

    teacher = {
        str(lhs).strip().upper(): str(rhs).strip()
        for lhs, rhs in reference_assignments.items()
    }
    plan = [
        (label, lhs)
        for label, lhs in extract_ordered_lhs_plan(source_text)
        if lhs.strip().upper() in teacher
    ]
    if not plan:
        raise ValueError(
            f"Teacher-forced validation has no reference sites: "
            f"{kernel_name}"
        )

    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    if routing_start_idx is None:
        routing_start_idx = torch.tensor(
            [len(prompt_ids)], dtype=torch.long, device=device
        )

    chosen_assignments = {}
    site_domains = directive_domain_registry[
        normalize_kname(kernel_name)
    ]
    trace = []
    current_label = None

    structural_enabled = (
        hasattr(model, "condition_structural_memory")
        and getattr(model, "initialized_structural_xattn", False)
    )
    use_structural_memory = (
        structural_enabled
        and structural_memory is not None
        and structural_memory_mask is not None
    )
    if use_structural_memory:
        model.condition_structural_memory(
            structural_memory.to(device),
            structural_memory_mask.to(device),
            action_relation_mask=(
                structural_relation_mask.to(device)
                if structural_relation_mask is not None
                else None
            ),
        )

    try:
        for label, lhs in plan:
            if label != current_label:
                anchor_ids = tok(
                    f"{target_placeholder_token(label)}\n",
                    add_special_tokens=False,
                )["input_ids"]
                input_ids, attention_mask = append_token_ids(
                    input_ids, attention_mask, anchor_ids
                )
                current_label = label

            prefix_ids = tok(
                f"{lhs} = ", add_special_tokens=False
            )["input_ids"]
            input_ids, attention_mask = append_token_ids(
                input_ids, attention_mask, prefix_ids
            )

            original_candidates = get_rhs_candidates_for_lhs(
                kernel_name, lhs, directive_domain_registry
            )
            candidates = mailohls_contract.filter_semantic_candidates(
                lhs,
                original_candidates,
                chosen_assignments,
                site_domains,
            )

            scored = []
            effective_batch_size = (
                1 if use_structural_memory else candidate_batch_size
            )
            if len(candidates) == 1:
                scored.append({
                    "rhs": candidates[0],
                    "score": 0.0,
                    "mean_logprob": 0.0,
                    "sum_logprob": 0.0,
                })

            for start in range(
                0 if len(candidates) > 1 else len(candidates),
                len(candidates),
                effective_batch_size,
            ):
                rhs_batch = candidates[
                    start:start + effective_batch_size
                ]
                if effective_batch_size == 1:
                    batch_stats = [
                        score_rhs_candidate_suffix(
                            model=model,
                            tok=tok,
                            base_input_ids=input_ids,
                            base_attention_mask=attention_mask,
                            candidate_text=rhs_batch[0] + "\n",
                            routing_start_idx=routing_start_idx,
                            use_structural_memory=use_structural_memory,
                        )
                    ]
                else:
                    batch_stats = score_rhs_candidate_batch(
                        model=model,
                        tok=tok,
                        base_input_ids=input_ids,
                        base_attention_mask=attention_mask,
                        candidate_texts=[
                            rhs + "\n" for rhs in rhs_batch
                        ],
                    )
                for rhs, stats in zip(rhs_batch, batch_stats):
                    scored.append({
                        "rhs": rhs,
                        "score": (
                            stats["mean_logprob"]
                            if score_reduction == "mean"
                            else stats["sum_logprob"]
                        ),
                        "mean_logprob": stats["mean_logprob"],
                        "sum_logprob": stats["sum_logprob"],
                    })

            scored.sort(
                key=lambda record: (
                    record["score"],
                    record["sum_logprob"],
                ),
                reverse=True,
            )
            trace.append({
                "label": label,
                "lhs": lhs,
                "static_candidate_count": len(original_candidates),
                "forced_by_semantics": len(candidates) == 1,
                "candidates": [dict(record) for record in scored],
            })

            lhs_key = lhs.strip().upper()
            gold_rhs = teacher[lhs_key]
            if gold_rhs not in candidates:
                raise RuntimeError(
                    "Reference RHS becomes illegal under its own "
                    f"teacher-forced prefix: "
                    f"{kernel_name}/{lhs_key}={gold_rhs}; "
                    f"legal={candidates}"
                )
            chosen_assignments[lhs_key] = gold_rhs
            gold_ids = tok(
                gold_rhs + "\n", add_special_tokens=False
            )["input_ids"]
            input_ids, attention_mask = append_token_ids(
                input_ids, attention_mask, gold_ids
            )

        return trace
    finally:
        if hasattr(model, "clear_structural_memory"):
            model.clear_structural_memory()
'''


HELPERS = r'''
def selection_contract_sha256(contract: Mapping[str, Any]) -> str:
    # Stable resume identity. Runtime argv/host state is intentionally excluded.
    ignored = {
        "runtime",
        "stage1_trainable_parameter_contract",
    }
    payload = {
        key: value
        for key, value in contract.items()
        if key not in ignored
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_effective_directive_domain_registry(
    output_dir: str,
    registry: Mapping[str, Mapping[str, Sequence[str]]],
    generation_policy: Any,
) -> Tuple[str, str]:
    # Save the exact normalized legal domains actually used by this run.
    payload = {
        "schema": DIRECTIVE_DOMAIN_REGISTRY_SCHEMA,
        "generation_policy": generation_policy,
        "kernels": {
            str(kernel): {
                str(lhs): [str(value) for value in values]
                for lhs, values in sorted(sites.items())
            }
            for kernel, sites in sorted(registry.items())
        },
    }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    path = Path(output_dir) / "directive_domain_registry.json"
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}"
    )
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return str(path), hashlib.sha256(encoded).hexdigest()
'''


TESTS = r'''

def test_teacher_forced_local_scoring_uses_reference_prefix(monkeypatch):
    source = "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?"
    domains = {
        "kernel_a": {
            "AUTO{_PIPE_L1}": ["0", "1"],
            "AUTO{_UNROLL_L1}": ["0", "2"],
        }
    }

    def fake_score(**kwargs):
        rhs = kwargs["candidate_text"].strip()
        score = {"0": 2.0, "1": 1.0, "2": 3.0}[rhs]
        return {"mean_logprob": score, "sum_logprob": score}

    monkeypatch.setattr(
        trainer, "score_rhs_candidate_suffix", fake_score
    )
    model = torch.nn.Linear(1, 1, bias=False)

    trace = trainer.score_teacher_forced_rhs_candidates(
        model=model,
        tok=CharacterTokenizer(),
        prompt_ids=[1],
        source_text=source,
        kernel_name="kernel-a",
        reference_assignments={
            "AUTO{_PIPE_L1}": "1",
            "AUTO{_UNROLL_L1}": "0",
        },
        directive_domain_registry=domains,
        candidate_batch_size=1,
    )
    assert trace[0]["candidates"][0]["rhs"] == "0"
    # PIPE is teacher-forced to 1 for the prefix, therefore UNROLL becomes
    # semantically forced to 0 even though the model preferred PIPE=0.
    assert trace[1]["forced_by_semantics"] is True
    assert trace[1]["candidates"][0]["rhs"] == "0"


def test_selection_summary_exposes_teacher_mrr_cascade_and_budget_accuracy():
    def make_row(budget, target, accuracy, cascade):
        return {
            "kernel_name": "kernel-a",
            "device": "device-a",
            "selected_clock_period": 5.0,
            "obj_mode": "PARETO_ADP",
            "resource_budget_id": budget,
            "reference_target": target,
            "prediction": f"prediction-{budget}",
            "value_accuracy_over_expected": accuracy,
            "decision_site_accuracy": accuracy,
            "decision_site_count": 1,
            "forced_site_count": 0,
            "schema_compliant": True,
            "expected_key_match": True,
            "exact_design_match": accuracy == 1.0,
            "pragma_kind_counts": {
                "PIPE": {
                    "correct": int(accuracy == 1.0),
                    "expected": 1,
                }
            },
            "candidate_margins": [{
                "kind": "PIPE",
                "margin": None if cascade else 0.25,
                "gold_rank": 2 if cascade else 1,
                "gold_excluded_by_prior_decision": cascade,
            }],
            "teacher_forced_candidates": [{
                "kind": "PIPE",
                "margin": 0.25 if accuracy == 1.0 else -0.25,
                "gold_rank": 1 if accuracy == 1.0 else 2,
            }],
        }

    summary = trainer.summarize_selection_rows([
        make_row("b1", "target-a", 1.0, False),
        make_row("b2", "target-b", 0.0, True),
    ])
    assert summary["candidate_cascade_excluded_count"] == 1
    assert summary["candidate_cascade_excluded_fraction"] == pytest.approx(0.5)
    assert summary["teacher_forced_top1_accuracy"] == pytest.approx(0.5)
    assert summary["teacher_forced_mrr"] == pytest.approx(0.75)
    assert summary["budget_counterfactual_groups"] == 1
    assert summary["budget_counterfactual_decision_accuracy"] == pytest.approx(0.5)


def test_scratch_stage1_refuses_stale_custom_best(tmp_path):
    best_dir = tmp_path / "best_custom_stage1"
    best_dir.mkdir()
    (best_dir / "best_selection_metrics.json").write_text(
        json.dumps({
            "step": 100,
            "checkpoint_key": [0.5, 0.4, 0.7, -1.0],
            "selection_contract_sha256": "old",
        })
    )
    with pytest.raises(RuntimeError, match="stale best"):
        trainer.StageValSelectionCallback(
            tokenizer=CharacterTokenizer(),
            selection_cases=[],
            directive_domain_registry={},
            output_dir=str(tmp_path),
            training_contract={"schema": "test"},
        )


def test_effective_domain_registry_hash_matches_saved_bytes(tmp_path):
    path, digest = trainer.write_effective_directive_domain_registry(
        str(tmp_path),
        {
            "kernel_a": {
                "AUTO{_PIPE_L1}": ["0", "1"],
            }
        },
        "source_policy",
    )
    assert trainer._file_sha256(trainer.Path(path)) == digest
'''


def patch_train(text):
    # 1. Free-running margin summary + teacher-forced summary.
    text = replace_function(
        text,
        "summarize_candidate_margins",
        MARGIN_FUNCTIONS,
    )

    # 2. Add teacher-forced scoring helper immediately before normal decoder.
    marker = (
        "@torch.no_grad()\n"
        "def constrained_decode_rhs_by_candidate_scoring(\n"
    )
    text = replace_once(
        text,
        marker,
        TEACHER_SCORER.strip() + "\n\n" + marker,
        "insert teacher-forced scoring helper",
    )

    # 3. Make the per-case evaluator use the new diagnostics and the same
    # semantic decision-site definition as Stage-1 supervision.
    text = replace_method(
        text,
        "StageValSelectionCallback",
        "_run_case",
        RUN_CASE,
    )

    # 4. Insert stable contract/registry helpers next to _file_sha256.
    helper_anchor = '''def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(jsonl_path: str) -> List[dict]:
'''
    helper_replacement = '''def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


''' + HELPERS.strip() + '''


def load_rows(jsonl_path: str) -> List[dict]:
'''
    text = replace_once(
        text,
        helper_anchor,
        helper_replacement,
        "insert stable contract/domain helpers",
    )

    # 5. Persist/hash the effective source-derived domain registry.
    text = replace_once(
        text,
        '''    print(f"[DOMAINS] registry_policy={registry_policy!r}")
    if registry_policy and "pre-split" in str(registry_policy).lower():
''',
        '''    print(f"[DOMAINS] registry_policy={registry_policy!r}")
    (
        effective_registry_path,
        effective_registry_sha256,
    ) = write_effective_directive_domain_registry(
        args.output_dir,
        directive_domain_registry,
        registry_policy,
    )
    print(
        "[DOMAINS] effective_registry_sha256="
        f"{effective_registry_sha256} "
        f"artifact={effective_registry_path}"
    )
    if registry_policy and "pre-split" in str(registry_policy).lower():
''',
        "save effective directive registry",
    )

    text = replace_once(
        text,
        '''        "directive_domain_registry_sha256": (
            _file_sha256(Path(args.directive_domain_registry_json))
            if args.directive_domain_registry_json else None
        ),
        "directive_loss_weighting": args.directive_loss_weighting,
''',
        '''        "directive_domain_registry_sha256": (
            effective_registry_sha256
        ),
        "directive_domain_registry_artifact": (
            os.path.basename(effective_registry_path)
        ),
        "directive_loss_weighting": args.directive_loss_weighting,
''',
        "contract effective directive registry",
    )

    # 6. Broaden validation coverage by selecting up to four DISTINCT correct
    # target configurations per kernel/device/clock instead of repeated targets.
    text = replace_once(
        text,
        "    cases_per_kernel_device: int = 2,\n",
        "    cases_per_kernel_device: int = 4,\n",
        "build_selection_cases default",
    )
    old_choice = '''        group = candidates_by_kernel_device_clock[group_key]
        distinct_target_count = len({case.reference_target for case in group})
        chosen = evenly_spaced_cases(
            group,
            min(cases_per_kernel_device, distinct_target_count),
        )
        selected_targets = {case.reference_target for case in chosen}
        if len(chosen) > 1 and len(selected_targets) == 1:
            alternative = next(
                (case for case in sorted(group, key=lambda item: shared_budget_fraction(item.row))
                 if case.reference_target not in selected_targets),
                None,
            )
            if alternative is not None:
                chosen[-1] = alternative
        selected.extend(chosen)
'''
    new_choice = '''        group = sorted(
            candidates_by_kernel_device_clock[group_key],
            key=lambda case: (
                shared_budget_fraction(case.row),
                int(case.row.get("_jsonl_idx", -1)),
            ),
        )
        by_target = defaultdict(list)
        for case in group:
            by_target[case.reference_target].append(case)

        # One median-budget representative per distinct correct target.
        # This maximizes target-changing budget information without repeatedly
        # evaluating budgets whose correct whole design is identical.
        representatives = [
            target_cases[len(target_cases) // 2]
            for _, target_cases in sorted(by_target.items())
        ]
        representatives.sort(
            key=lambda case: (
                shared_budget_fraction(case.row),
                int(case.row.get("_jsonl_idx", -1)),
            )
        )
        chosen = evenly_spaced_cases(
            representatives,
            min(cases_per_kernel_device, len(representatives)),
        )
        selected.extend(chosen)
'''
    text = replace_once(
        text,
        old_choice,
        new_choice,
        "select distinct validation targets",
    )

    # CLI default follows the same policy.
    text = replace_once(
        text,
        '''        dest="selection_cases_per_kernel_device",
        type=int,
        default=2,
    )
''',
        '''        dest="selection_cases_per_kernel_device",
        type=int,
        default=4,
    )
''',
        "selection coverage CLI default",
    )

    # 7. Expand selection summary with actual budget-sensitive accuracy and
    # teacher-forced local metrics.
    old_budget = '''    summary["budget_counterfactual_groups"] = len(informative)
    summary["budget_sensitive_prediction_groups"] = sum(
        len({row["prediction"] for row in group}) > 1 for group in informative
    )

    summary.update(
        summarize_candidate_margins(
            rows
        )
    )

    return summary
'''
    new_budget = '''    summary["budget_counterfactual_groups"] = len(informative)
    summary["budget_counterfactual_case_count"] = sum(
        len(group) for group in informative
    )
    summary["budget_sensitive_prediction_groups"] = sum(
        len({row["prediction"] for row in group}) > 1 for group in informative
    )
    group_decision_accuracies = [
        float(np.mean([
            row.get(
                "decision_site_accuracy",
                row["value_accuracy_over_expected"],
            )
            for row in group
        ]))
        for group in informative
    ]
    summary["budget_counterfactual_decision_accuracy"] = (
        float(np.mean(group_decision_accuracies))
        if group_decision_accuracies else None
    )

    summary.update(summarize_candidate_margins(rows))
    summary.update(summarize_teacher_forced_candidates(rows))

    return summary
'''
    text = replace_once(
        text,
        old_budget,
        new_budget,
        "budget/teacher selection summaries",
    )

    # 8. Stale-best protection. Scratch runs never inherit output state;
    # resumed runs require a matching stable training/selection contract.
    text = replace_once(
        text,
        '''        structural_routing: str = "exact_slot",
        early_stopping_patience: int = 0,
    ):
''',
        '''        structural_routing: str = "exact_slot",
        early_stopping_patience: int = 0,
        early_stopping_min_step: int = 0,
        resume_from_checkpoint: str = "",
    ):
''',
        "callback signature",
    )
    text = replace_once(
        text,
        '''        self.early_stopping_patience = int(early_stopping_patience)
        self.evaluations_without_improvement = 0
        best_path = Path(output_dir) / best_dir_name / "best_selection_metrics.json"
''',
        '''        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_step = max(0, int(early_stopping_min_step))
        self.resume_from_checkpoint = str(resume_from_checkpoint or "")
        self.evaluations_without_improvement = 0
        self.selection_contract_sha256 = selection_contract_sha256(
            self.training_contract
        )
        best_path = Path(output_dir) / best_dir_name / "best_selection_metrics.json"
''',
        "callback state",
    )
    old_best = '''        if best_path.is_file():
            previous = json.loads(best_path.read_text(encoding="utf-8"))
            self.best_key = tuple(previous["checkpoint_key"])
            self.best_step = int(previous["step"])
            print(
                f"[VAL-SELECTION] Restored previous best: "
                f"step={self.best_step}, key={self.best_key}"
            )
        else:
            self.best_key = (float("-inf"),) * 4
            self.best_step = -1
        self.last_selection_step = None
'''
    new_best = '''        if self.resume_from_checkpoint:
            if best_path.is_file():
                previous = json.loads(
                    best_path.read_text(encoding="utf-8")
                )
                if previous.get("selection_contract_sha256") != (
                    self.selection_contract_sha256
                ):
                    raise RuntimeError(
                        "Existing custom-best metrics are incompatible with "
                        "the current Stage-1 contract. Resume only the exact "
                        "same experiment, or use a new --output_dir."
                    )
                self.best_key = tuple(previous["checkpoint_key"])
                self.best_step = int(previous["step"])
                print(
                    "[VAL-SELECTION] Restored compatible previous best: "
                    f"step={self.best_step}, key={self.best_key}"
                )
            else:
                self.best_key = (float("-inf"),) * 4
                self.best_step = -1
        else:
            if best_path.is_file():
                raise RuntimeError(
                    "Scratch training output already contains a stale best "
                    f"checkpoint: {best_path}. Use a new --output_dir or "
                    "explicitly --resume_from_checkpoint."
                )
            self.best_key = (float("-inf"),) * 4
            self.best_step = -1
        self.last_selection_step = None
'''
    text = replace_once(
        text,
        old_best,
        new_best,
        "stale custom-best protection",
    )

    # 9. Keep kernel-macro free-running decision accuracy primary; replace the
    # near-binary exact-design tie-breaker by teacher-forced local MRR.
    old_key = '''            checkpoint_key = (
                selection_score,
                minimum_kernel_accuracy,
                exact_design_accuracy,
                -eval_loss,
            )
'''
    new_key = '''            teacher_mrr = summary["teacher_forced_mrr"]
            if teacher_mrr is None:
                raise RuntimeError(
                    "Teacher-forced validation produced no decision-site "
                    "ranking records."
                )
            checkpoint_key = (
                selection_score,
                minimum_kernel_accuracy,
                float(teacher_mrr),
                -eval_loss,
            )
'''
    text = replace_once(
        text,
        old_key,
        new_key,
        "checkpoint key uses teacher MRR",
    )

    # Add concise live diagnostics before the existing free-running margin log.
    text = replace_once(
        text,
        '''            print("=" * 100)

            print(
                "[VAL-MARGIN] "
''',
        '''            print("=" * 100)

            print(
                "[VAL-LOCAL] "
                f"count={summary['teacher_forced_candidate_count']} "
                f"top1={summary['teacher_forced_top1_accuracy']} "
                f"mrr={summary['teacher_forced_mrr']} "
                f"mean_margin={summary['teacher_forced_mean_margin']} "
                f"cascade={summary['candidate_cascade_excluded_count']}/"
                f"{summary['candidate_cascade_site_count']} "
                f"cascade_fraction="
                f"{summary['candidate_cascade_excluded_fraction']}"
            )
            print(
                "[VAL-LOCAL] per_kind="
                f"{summary['teacher_forced_per_kind']}"
            )
            print(
                "[VAL-BUDGET] "
                f"groups={summary['budget_counterfactual_groups']} "
                f"cases={summary['budget_counterfactual_case_count']} "
                f"decision_accuracy="
                f"{summary['budget_counterfactual_decision_accuracy']} "
                f"prediction_sensitive_groups="
                f"{summary['budget_sensitive_prediction_groups']}"
            )

            print(
                "[VAL-MARGIN] "
''',
        "live local/budget diagnostics",
    )

    text = replace_once(
        text,
        '''            metrics_obj = {
                "step": int(state.global_step),
                "eval_loss": eval_loss,
                "checkpoint_key": list(checkpoint_key),
                **summary,
                "rows": rows,
            }
''',
        '''            metrics_obj = {
                "step": int(state.global_step),
                "eval_loss": eval_loss,
                "checkpoint_key": list(checkpoint_key),
                "selection_contract_sha256": (
                    self.selection_contract_sha256
                ),
                **summary,
                "rows": rows,
            }
''',
        "persist selection contract hash",
    )

    # 10. One-effective-epoch floor before early-stop misses start counting.
    old_stop = '''            elif self.early_stopping_patience > 0 and state.global_step > 0:
                self.evaluations_without_improvement += 1
                print(f"[EARLY-STOP] no improvement for "
                      f"{self.evaluations_without_improvement}/"
                      f"{self.early_stopping_patience} validation evaluations")
                if self.evaluations_without_improvement >= self.early_stopping_patience:
                    control.should_training_stop = True
                    print(f"[EARLY-STOP] retaining best checkpoint from step {self.best_step}")
'''
    new_stop = '''            elif self.early_stopping_patience > 0 and state.global_step > 0:
                if state.global_step < self.early_stopping_min_step:
                    print(
                        "[EARLY-STOP] not counting non-improvement before "
                        f"minimum step {self.early_stopping_min_step}; "
                        f"current={state.global_step}"
                    )
                else:
                    self.evaluations_without_improvement += 1
                    print(f"[EARLY-STOP] no improvement for "
                          f"{self.evaluations_without_improvement}/"
                          f"{self.early_stopping_patience} validation evaluations")
                    if self.evaluations_without_improvement >= self.early_stopping_patience:
                        control.should_training_stop = True
                        print(f"[EARLY-STOP] retaining best checkpoint from step {self.best_step}")
'''
    text = replace_once(
        text,
        old_stop,
        new_stop,
        "one-epoch early-stop floor",
    )

    # 11. Compute the optimizer-update epoch after the real SFTDataset exists.
    text = replace_once(
        text,
        '''    warmup_steps = int(
        args.warmup_ratio * effective_total_steps
    )

    bf16_ok = native_bf16
''',
        '''    warmup_steps = int(
        args.warmup_ratio * effective_total_steps
    )
    effective_updates_per_epoch = (
        math.ceil(steps_per_epoch / max(1, args.grad_accum))
        if not args.selection_eval_only
        else 0
    )
    stage1_early_stopping_min_step = (
        effective_updates_per_epoch
        if args.disable_structural_memory
        else 0
    )
    training_contract["effective_updates_per_epoch"] = (
        effective_updates_per_epoch
    )
    training_contract["early_stopping_min_step"] = (
        stage1_early_stopping_min_step
    )
    dump_json(
        os.path.join(args.output_dir, "training_contract.json"),
        training_contract,
    )
    print(
        "[EARLY-STOP] effective_updates_per_epoch="
        f"{effective_updates_per_epoch} "
        f"minimum_counted_step={stage1_early_stopping_min_step}"
    )

    bf16_ok = native_bf16
''',
        "compute effective epoch floor",
    )

    text = replace_once(
        text,
        '''                early_stopping_patience=(
                    args.early_stopping_patience if args.disable_structural_memory else 0
                ),
            )
''',
        '''                early_stopping_patience=(
                    args.early_stopping_patience
                    if args.disable_structural_memory else 0
                ),
                early_stopping_min_step=(
                    stage1_early_stopping_min_step
                ),
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
''',
        "pass callback floor/resume state",
    )

    # 12. Optional clean tracked-tree guard for the final publication run.
    text = replace_once(
        text,
        '''    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default="")
    ap.add_argument("--init_adapter_dir", type=str, default="")
''',
        '''    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default="")
    ap.add_argument(
        "--require_clean_git",
        action="store_true",
        help=(
            "Final-run guard: reject tracked working-tree changes. "
            "Untracked output/helper files are ignored."
        ),
    )
    ap.add_argument("--init_adapter_dir", type=str, default="")
''',
        "require_clean_git CLI",
    )

    text = replace_once(
        text,
        '''        "objective": args.objective,
        "seed": args.seed,
        "tokenizer_size": len(tok),
''',
        '''        "objective": args.objective,
        "seed": args.seed,
        "require_clean_git": bool(args.require_clean_git),
        "tokenizer_size": len(tok),
''',
        "record clean-git policy",
    )

    text = replace_once(
        text,
        '''    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    )

    training_contract["runtime"] = {
''',
        '''    git_status_tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()
    git_dirty = bool(git_status_tracked)
    if args.require_clean_git and git_dirty:
        raise RuntimeError(
            "Final Stage-1 requires a clean tracked git tree. "
            "Commit or restore tracked changes before training. "
            f"Dirty entries: {git_status_tracked.splitlines()[:12]}"
        )

    training_contract["runtime"] = {
''',
        "clean tracked git guard",
    )
    text = replace_once(
        text,
        '''        "git_dirty": git_dirty,
        "cuda_visible_devices": os.environ.get(
''',
        '''        "git_dirty": git_dirty,
        "git_dirty_policy": "tracked_files_only",
        "cuda_visible_devices": os.environ.get(
''',
        "git dirty policy in runtime contract",
    )

    return text


def main():
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Run this script from the MailoHLS repo root: missing {path}"
            )
        actual = blob_sha(path)
        if actual != expected:
            raise RuntimeError(
                f"{path}: blob {actual} != inspected base {expected}. "
                "No files were changed. Send `git diff -- <path>` before "
                "trying to force this patch."
            )

    originals = {
        TRAIN: TRAIN.read_text(encoding="utf-8"),
        TEST: TEST.read_text(encoding="utf-8"),
    }

    modified_train = patch_train(originals[TRAIN])
    modified_test = (
        originals[TEST].rstrip()
        + "\n"
        + TESTS.strip()
        + "\n"
    )

    # Full preflight before touching the working tree.
    ast.parse(modified_train)
    ast.parse(modified_test)

    modified = {
        TRAIN: modified_train,
        TEST: modified_test,
    }
    temps = {}
    try:
        for path, content in modified.items():
            temp = path.with_name(
                f"{path.name}.stage1-patch-{os.getpid()}.tmp"
            )
            temp.write_text(content, encoding="utf-8")
            ast.parse(temp.read_text(encoding="utf-8"))
            temps[path] = temp

        for path, temp in temps.items():
            os.replace(temp, path)

        for path in modified:
            subprocess.run(
                ["python", "-m", "py_compile", str(path)],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            ["git", "diff", "--check"],
            cwd=ROOT,
            check=True,
        )
    except Exception:
        # Transactional rollback on any post-write validation failure.
        for path, content in originals.items():
            path.write_text(content, encoding="utf-8")
        for temp in temps.values():
            if temp.exists():
                temp.unlink()
        raise

    print("\n[DONE] Stage-1 patch applied transactionally.")
    print("No Stage-1 training loss, LoRA, family sampler, or dataset was changed.")
    print("\nTargeted tests:")
    print(
        "  python -m pytest -q "
        "LLM_branch/tests/test_stage1_final_contract.py "
        "LLM_branch/tests/test_training_contract.py"
    )
    print("Then:")
    print("  python -m pytest -q LLM_branch/tests")
    print("  git diff --check")
    print(
        "  git diff -- "
        "LLM_branch/train/train_SFT_xattn_new.py "
        "LLM_branch/tests/test_stage1_final_contract.py"
    )


if __name__ == "__main__":
    main()
