#!/usr/bin/env python3
"""
Apply the final MailoHLS Stage-1/GNN simplification agreed for the final experiments.

Audited repository state:
  ElenaVouvali/MailoHLS
  branch: stage2-analysis-refactor
  HEAD: a6e38321f92fb0401737e74b98b8d1ae091999b6 ("GNN final")

The patch is transactional: it verifies the audited Git blobs, performs all
source transformations in memory, syntax-checks them, and only then writes the
files atomically.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path.cwd() / "MailoHLS", Path.cwd() / "MailoHLS-next"]
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate.resolve()
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate.resolve()
    raise SystemExit("Could not locate the MailoHLS git repository. Run from its root.")


ROOT = find_repo_root()
EXPECTED_HEAD = "a6e38321f92fb0401737e74b98b8d1ae091999b6"
EXPECTED_BLOBS = {
    "LLM_branch/common/directive_domains.py": "347abdf95663ba79820028fae9776be7e48c3459",
    "LLM_branch/train/train_SFT_xattn_new.py": "59a336c13abe6294e95c3f86cf9263416cf7e435",
    "LLM_branch/tests/test_stage1_final_contract.py": "8942255976d20e9c5570fa4f566996ff065cf3af",
    "GNN_branch/config.py": "1abc8004257014c1d4aac678d71d74d961cf9cb4",
    "GNN_branch/model.py": "8800875226d469fcddb49015710ac7a1719ef7fc",
    "GNN_branch/train_GNN.py": "077b2d3bd897d4c7c0c1c262d777a3c967c040b0",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_audited_state() -> None:
    head = git("rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise SystemExit(
            f"Refusing to patch HEAD={head}; expected audited HEAD={EXPECTED_HEAD}."
        )
    for rel, expected in EXPECTED_BLOBS.items():
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"Missing required source file: {rel}")
        actual = git("hash-object", rel)
        if actual != expected:
            raise SystemExit(
                f"Refusing to patch modified/stale source {rel}: "
                f"blob={actual}, expected={expected}"
            )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_n(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


def regex_replace(text: str, pattern: str, repl, label: str, expected=None, flags=0) -> str:
    new_text, count = re.subn(pattern, repl, text, flags=flags)
    if expected is None:
        if count == 0:
            raise RuntimeError(f"{label}: expected at least one match")
    elif count != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, found {count}")
    return new_text


def patch_directive_domains(text: str) -> str:
    text = replace_once(
        text,
        '"""Derive legal directive choices from public source/action metadata only."""',
        '"""Derive finite directive proposal domains from source/action metadata only."""',
        "directive-domain module description",
    )
    text = replace_once(
        text,
        'SOURCE_DOMAIN_POLICY = "source_action_metadata_and_compiler_legal_values_v1"',
        'SOURCE_DOMAIN_POLICY = "source_action_metadata_proposal_domains_v2"',
        "directive-domain policy",
    )
    return text


def patch_stage1(text: str) -> str:
    old = '''        if site_domains is None:\n            supervise = True\n        else:\n            candidates = get_rhs_candidates_for_lhs(\n                kernel_name, lhs, directive_domain_registry\n            )\n            legal = mailohls_contract.filter_semantic_candidates(\n                lhs, candidates, chosen_assignments, site_domains\n            )\n            if rhs not in legal:\n                raise ValueError(f"Gold directive is semantically illegal: {kernel_name}/{lhs}={rhs}")\n            supervise = len(legal) > 1\n'''
    new = '''        if site_domains is None:\n            supervise = True\n        else:\n            candidates = get_rhs_candidates_for_lhs(\n                kernel_name, lhs, directive_domain_registry\n            )\n            if rhs not in candidates:\n                raise ValueError(\n                    "Gold directive is outside the source-derived proposal "\n                    f"domain: {kernel_name}/{lhs}={rhs}; candidates={candidates}"\n                )\n            supervise = len(candidates) > 1\n'''
    text = replace_once(text, old, new, "Stage-1 static-domain supervision")

    # Replace all candidate filtering blocks that first load original_candidates.
    pattern = r'''(?P<i>[ \t]*)original_candidates = get_rhs_candidates_for_lhs\(\n(?P=i)    kernel_name, lhs, directive_domain_registry\n(?P=i)\)\n(?P=i)candidates = mailohls_contract\.filter_semantic_candidates\(\n(?:(?P=i)    .*\n)+?(?P=i)\)\n'''

    def static_candidates(match: re.Match[str]) -> str:
        i = match.group("i")
        return (
            f"{i}candidates = get_rhs_candidates_for_lhs(\n"
            f"{i}    kernel_name, lhs, directive_domain_registry\n"
            f"{i})\n"
        )

    text = regex_replace(
        text,
        pattern,
        static_candidates,
        "runtime static-domain candidate replacement",
        expected=None,
        flags=re.MULTILINE,
    )

    text = text.replace(
        '        mailohls_contract.validate_directive_assignments(chosen_assignments)\n',
        '        # Cross-field directive compatibility is learned, not hard-filtered here.\n',
    )

    old = '''        # This now mirrors Stage-1 supervision: a reference site is a real\n        # decision only if >1 RHS is legal under the correct prior decisions.\n        decision_keys = {\n            site["lhs"].strip().upper()\n            for site in teacher_trace\n            if not bool(site["forced_by_semantics"])\n        }\n'''
    new = '''        # Production Stage-1 evaluates every source-derived field whose\n        # proposal domain contains more than one RHS. Related decisions never\n        # disappear because an earlier field took a particular value.\n        decision_keys = {\n            lhs.strip().upper()\n            for _, lhs in extract_ordered_lhs_plan(case.source_text)\n            if len(get_rhs_candidates_for_lhs(\n                case.kernel_name, lhs, self.directive_domain_registry\n            )) > 1\n        }\n'''
    text = replace_once(text, old, new, "static validation decision keys")

    old = '''        decision_correct = sum(\n            pred_assign.get(key) == ref_assign.get(key)\n            for key in decision_keys\n        )\n\n        return {\n'''
    new = '''        decision_correct = sum(\n            pred_assign.get(key) == ref_assign.get(key)\n            for key in decision_keys\n        )\n\n        action_decision_keys = defaultdict(list)\n        for key in sorted(decision_keys):\n            match = re.search(r"_(L[1-9][0-9]*)\\}$", key)\n            if match is None:\n                raise RuntimeError(f"Could not recover action label from {key!r}")\n            action_decision_keys[match.group(1)].append(key)\n        joint_action_count = len(action_decision_keys)\n        joint_action_correct = sum(\n            all(pred_assign.get(key) == ref_assign.get(key) for key in keys)\n            for keys in action_decision_keys.values()\n        )\n\n        return {\n'''
    text = replace_once(text, old, new, "joint-action computation")

    old = '''            "decision_site_accuracy": (\n                decision_correct / len(decision_keys)\n            ),\n            "forced_site_count": (\n'''
    new = '''            "decision_site_accuracy": (\n                decision_correct / len(decision_keys)\n            ),\n            "joint_action_count": joint_action_count,\n            "joint_action_correct_count": joint_action_correct,\n            "joint_action_accuracy": (\n                joint_action_correct / joint_action_count\n                if joint_action_count else 0.0\n            ),\n            "forced_site_count": (\n'''
    text = replace_once(text, old, new, "joint-action row fields")

    anchor = '    pragma_kind_totals = defaultdict(lambda: {"correct": 0, "expected": 0})\n'
    insertion = '''    kernel_joint_action_acc = {\n        kernel: sum(\n            row.get(\n                "joint_action_accuracy",\n                row.get("decision_site_accuracy", row["value_accuracy_over_expected"]),\n            )\n            for row in kernel_rows\n        ) / len(kernel_rows)\n        for kernel, kernel_rows in sorted(rows_by_kernel.items())\n    }\n'''
    text = replace_once(text, anchor, insertion + anchor, "kernel joint-action aggregation")

    old = '''        "per_kernel_decision_accuracy": kernel_decision_acc,\n        "minimum_kernel_decision_accuracy": min(kernel_decision_acc.values()),\n        "mean_decision_site_accuracy": float(\n'''
    new = '''        "per_kernel_decision_accuracy": kernel_decision_acc,\n        "minimum_kernel_decision_accuracy": min(kernel_decision_acc.values()),\n        "per_kernel_joint_action_accuracy": kernel_joint_action_acc,\n        "minimum_kernel_joint_action_accuracy": min(kernel_joint_action_acc.values()),\n        "joint_action_score": float(\n            sum(kernel_joint_action_acc.values()) / len(kernel_joint_action_acc)\n        ),\n        "mean_decision_site_accuracy": float(\n'''
    text = replace_once(text, old, new, "joint-action summary fields")

    old = '''            teacher_mrr = summary["teacher_forced_mrr"]\n            if teacher_mrr is None:\n                raise RuntimeError(\n                    "Teacher-forced validation produced no decision-site "\n                    "ranking records."\n                )\n            checkpoint_key = (\n                selection_score,\n                minimum_kernel_accuracy,\n                float(teacher_mrr),\n                -eval_loss,\n            )\n'''
    new = '''            joint_action_score = summary["joint_action_score"]\n            checkpoint_key = (\n                selection_score,\n                joint_action_score,\n                -eval_loss,\n            )\n'''
    text = replace_once(text, old, new, "Stage-1 checkpoint key")

    text = replace_n(
        text,
        'self.best_key = (float("-inf"),) * 4',
        'self.best_key = (float("-inf"),) * 3',
        2,
        "Stage-1 best-key arity",
    )

    old = '''            print(f"[VAL-SELECTION] selection_score={selection_score:.6f}")\n            print(f"[VAL-SELECTION] checkpoint_key={checkpoint_key}")\n'''
    new = '''            print(f"[VAL-SELECTION] selection_score={selection_score:.6f}")\n            print(\n                f"[VAL-SELECTION] joint_action_score="\n                f"{summary['joint_action_score']:.6f}"\n            )\n            print(f"[VAL-SELECTION] checkpoint_key={checkpoint_key}")\n'''
    text = replace_once(text, old, new, "joint-action validation print")

    pattern = r'''    stage1_early_stopping_min_step = \(\n        effective_updates_per_epoch\n        if args\.disable_structural_memory\n        else 0\n    \)\n'''
    text = regex_replace(
        text,
        pattern,
        '    stage1_early_stopping_min_step = 0\n',
        "Stage-1 early-stop floor",
        expected=1,
    )

    helper_anchor = '\ndef summarize_budget_counterfactuals(rows: Sequence[Mapping[str, Any]]) -> dict:\n'
    helper = r'''
def compact_duplicate_budget_targets(
    rows: Sequence[Mapping[str, Any]],
    max_duplicates: int,
) -> List[dict]:
    """Compact duplicate training budgets only after objective selection.

    Repeated targets are grouped by kernel/device/clock/objective/canonical
    completion. The tightest budget is kept first. If a full-device budget is
    present it is the second representative; otherwise the loosest budget is.
    max_duplicates=0 disables compaction. Validation is never compacted.
    """
    if max_duplicates <= 0:
        return [dict(row) for row in rows]

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["kernel_name"],
            _norm_device(row.get("device", "")),
            _clock_of(row),
            row.get("obj_mode"),
            canonical_completion_key(row["input"], row["target"]),
        )
        groups[key].append(dict(row))

    def budget_vector(row):
        device = _norm_device(row.get("device", ""))
        capacities = DEVICE_RESOURCES[device]
        available = _available_resources(row)
        return tuple(
            available[name] / capacities[name]
            for name in RESOURCE_KEYS
        )

    def tightness_key(row):
        vector = budget_vector(row)
        return (
            min(vector),
            sum(vector) / len(vector),
            max(vector),
            str(row.get("resource_budget_id", "")),
            int(row.get("_jsonl_idx", -1)),
        )

    compacted = []
    repeated_groups = 0
    for _, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        ordered = sorted(group, key=tightness_key)
        if len(ordered) > max_duplicates:
            repeated_groups += 1
        chosen = [ordered[0]]
        if max_duplicates > 1 and len(ordered) > 1:
            full = [
                row for row in ordered
                if all(abs(value - 1.0) <= 1e-9 for value in budget_vector(row))
            ]
            second = full[-1] if full else ordered[-1]
            if second is not chosen[0]:
                chosen.append(second)
        if len(chosen) < min(max_duplicates, len(ordered)):
            chosen_ids = {id(row) for row in chosen}
            for row in ordered:
                if id(row) not in chosen_ids:
                    chosen.append(row)
                    chosen_ids.add(id(row))
                if len(chosen) == max_duplicates:
                    break
        compacted.extend(chosen[:max_duplicates])

    print(
        "[BUDGET-COMPACT] "
        f"input={len(rows)} output={len(compacted)} "
        f"target_groups={len(groups)} repeated_groups={repeated_groups} "
        f"max_duplicates={max_duplicates}"
    )
    return compacted

'''
    text = replace_once(
        text,
        helper_anchor,
        '\n' + helper + helper_anchor.lstrip('\n'),
        "budget compaction helper",
    )

    old = '''        train_rows, train_goal_info = (\n            select_objectives(\n                raw_train_rows\n            )\n        )\n\n        val_rows, val_goal_info = (\n'''
    new = '''        train_rows, train_goal_info = (\n            select_objectives(\n                raw_train_rows\n            )\n        )\n        train_rows = compact_duplicate_budget_targets(\n            train_rows,\n            args.budget_target_max_duplicates,\n        )\n\n        val_rows, val_goal_info = (\n'''
    text = replace_once(text, old, new, "training-only budget compaction")

    old = '''    ap.add_argument(\n        "--candidate_pool_per_objective",\n        type=int,\n        default=24,\n    )\n'''
    new = '''    ap.add_argument(\n        "--candidate_pool_per_objective",\n        type=int,\n        default=24,\n    )\n    ap.add_argument(\n        "--budget_target_max_duplicates",\n        type=int,\n        default=0,\n        help=(\n            "After objective selection, retain at most this many training "\n            "budget contexts for an identical canonical target. 0 disables "\n            "compaction; final MailoHLS uses 2 (tightest + full/loosest)."\n        ),\n    )\n'''
    text = replace_once(text, old, new, "budget compaction CLI")

    old = '''        "candidate_pool_per_objective": args.candidate_pool_per_objective,\n        "candidate_compaction_policy": "per_measured_clock",\n'''
    new = '''        "candidate_pool_per_objective": args.candidate_pool_per_objective,\n        "budget_target_max_duplicates": args.budget_target_max_duplicates,\n        "budget_target_compaction_policy": (\n            "post_objective_target_tightest_plus_full_or_loosest"\n        ),\n        "candidate_compaction_policy": "per_measured_clock",\n'''
    text = replace_once(text, old, new, "budget compaction contract")

    old = '''        "--family_sampling_power", type=float, default=0.5,\n        help="Sample families with probability proportional to count^(1-power).",\n'''
    new = '''        "--family_sampling_power", type=float, default=0.0,\n        help=(\n            "Optional family-reweighted sampling ablation. Production Stage-1 "\n            "uses 0.0, i.e. the ordinary seeded example order."\n        ),\n'''
    text = replace_once(text, old, new, "family-sampling default")

    text = replace_once(
        text,
        '"selection_metric": "kernel_macro_decision_site_accuracy",',
        '"selection_metric": "kernel_macro_static_field_accuracy_with_joint_action_tiebreak",',
        "selection metric contract",
    )
    return text


def patch_stage1_test(text: str) -> str:
    old = '''def test_conditionally_forced_directive_is_not_supervised():\n    source = "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?"\n    target = "auto{_PIPE_L1} = 1\\nauto{_UNROLL_L1} = 0"\n    pack = trainer.build_deterministic_rhs_pack(\n        source, target, CharacterTokenizer(),\n        directive_domain_registry={"kernel_a": {\n            "AUTO{_PIPE_L1}": ["0", "1"],\n            "AUTO{_UNROLL_L1}": ["0", "2"],\n        }},\n        kernel_name="kernel-a",\n    )\n    assert sum(pack.xattn_target_mask) == 2  # Only the real PIPE decision: "1\\n".\n    assert sum(pack.token_weights) == pytest.approx(1.0)\n'''
    new = '''def test_related_static_fields_remain_independently_supervised():\n    source = "L1: auto{_PIPE_L1} = ? auto{_UNROLL_L1} = ?"\n    target = "auto{_PIPE_L1} = 1\\nauto{_UNROLL_L1} = 0"\n    pack = trainer.build_deterministic_rhs_pack(\n        source, target, CharacterTokenizer(),\n        directive_domain_registry={"kernel_a": {\n            "AUTO{_PIPE_L1}": ["0", "1"],\n            "AUTO{_UNROLL_L1}": ["0", "2"],\n        }},\n        kernel_name="kernel-a",\n    )\n    # Both source-derived fields have >1 proposal and both receive supervision.\n    assert sum(pack.xattn_target_mask) == 4\n    assert sum(pack.token_weights) == pytest.approx(2.0)\n'''
    return replace_once(text, old, new, "static-domain Stage-1 regression test")


def patch_gnn_config(text: str) -> str:
    text = replace_once(
        text,
        '    choices=("absolute", "qualified_rank", "embedding_rank", "structural_rank"),\n',
        '''    choices=(\n        "absolute",\n        "qualified_rank",\n        "embedding_rank",\n        "structural_rank",\n        "hardware_regression",\n    ),\n''',
        "GNN checkpoint-objective choices",
    )
    old = '''        "structural_rank:"\n        "maximize worst-target kernel-macro Kendall tau-b for the"\n        "Stage-2 structural encoder. Absolute QoR ratios are reported"\n        "as diagnostics but are not qualification gates."\n'''
    new = '''        "structural_rank: "\n        "maximize worst-target kernel-macro Kendall tau-b for the "\n        "Stage-2 structural encoder. Absolute QoR ratios are reported "\n        "as diagnostics but are not qualification gates. "\n        "hardware_regression: minimize the complete validation training "\n        "objective (QoR regression plus enabled physical-resource heads); "\n        "this is the final MailoHLS GNN setting."\n'''
    return replace_once(text, old, new, "GNN objective help")


def patch_gnn_model(text: str) -> str:
    return replace_once(
        text,
        '''                if self.training:\n                    kernels = list(\n''',
        '''                if self.training and float(FLAGS.rank_aux_weight) > 0.0:\n                    kernels = list(\n''',
        "skip inactive GNN rank-loss work",
    )


def patch_gnn_train(text: str) -> str:
    text = replace_once(
        text,
        '''    best_structural_rank_kernel_ratios = None\n\n    if FLAGS.resume_training and exists(ckpt_path):\n''',
        '''    best_structural_rank_kernel_ratios = None\n    best_hardware_regression_loss = float("inf")\n    best_hardware_regression_epoch = None\n\n    if FLAGS.resume_training and exists(ckpt_path):\n''',
        "hardware-regression state init",
    )
    text = replace_once(
        text,
        '''        best_embedding_rank_ratios = st.get(\n            "best_embedding_rank_ratios"\n        )\n        stored_baseline = st.get("baseline_breakdown")\n''',
        '''        best_embedding_rank_ratios = st.get(\n            "best_embedding_rank_ratios"\n        )\n        best_hardware_regression_loss = st.get(\n            "best_hardware_regression_loss", float("inf")\n        )\n        best_hardware_regression_epoch = st.get(\n            "best_hardware_regression_epoch"\n        )\n        stored_baseline = st.get("baseline_breakdown")\n''',
        "hardware-regression resume state",
    )

    anchor = '            ranking_score = compute_macro_ranking_score(val_metrics)\n'
    insertion = '''            if (\n                FLAGS.checkpoint_objective == "hardware_regression"\n                and val\n                < best_hardware_regression_loss\n                - float(FLAGS.early_stopping_min_delta)\n            ):\n                best_hardware_regression_loss = float(val)\n                best_hardware_regression_epoch = epoch\n                if FLAGS.save_model:\n                    save_checkpoint_with_sidecar(\n                        model.state_dict(),\n                        join(\n                            saver.model_logdir,\n                            "val_hardware_regression_model_state_dict.pth",\n                        ),\n                        "val_hardware_regression",\n                        epoch,\n                    )\n                saver.log_info(\n                    "Saved hardware-regression model at epoch "\n                    f"{epoch}; complete validation objective={val:.6f}"\n                )\n\n'''
    text = replace_once(text, anchor, insertion + anchor, "hardware-regression checkpoint save")

    old = '''                scheduler_score = (\n                    embedding_rank_control_score(\n                        current_selection_score,\n                        target_ratios,\n                        ranking_score,\n                    )\n                    if FLAGS.checkpoint_objective == 'embedding_rank'\n                    else current_selection_score\n                )\n'''
    new = '''                scheduler_score = (\n                    val\n                    if FLAGS.checkpoint_objective == "hardware_regression"\n                    else (\n                        embedding_rank_control_score(\n                            current_selection_score,\n                            target_ratios,\n                            ranking_score,\n                        )\n                        if FLAGS.checkpoint_objective == "embedding_rank"\n                        else current_selection_score\n                    )\n                )\n'''
    text = replace_once(text, old, new, "hardware-regression plateau scheduler")

    text = replace_once(
        text,
        '''                "best_embedding_rank_ratios": best_embedding_rank_ratios,\n                "initial_selection": initial_selection,\n''',
        '''                "best_embedding_rank_ratios": best_embedding_rank_ratios,\n                "best_hardware_regression_loss": best_hardware_regression_loss,\n                "best_hardware_regression_epoch": best_hardware_regression_epoch,\n                "initial_selection": initial_selection,\n''',
        "hardware-regression checkpoint state",
    )

    old = '''        selection_loss = (\n            (\n                embedding_rank_control_score(\n                    current_selection_score,\n                    target_ratios,\n                    ranking_score,\n                )\n                if FLAGS.checkpoint_objective == 'embedding_rank'\n                else current_selection_score\n            )\n            if len(val_loader) > 0 else loss\n        )\n'''
    new = '''        selection_loss = (\n            (\n                val\n                if FLAGS.checkpoint_objective == "hardware_regression"\n                else (\n                    embedding_rank_control_score(\n                        current_selection_score,\n                        target_ratios,\n                        ranking_score,\n                    )\n                    if FLAGS.checkpoint_objective == "embedding_rank"\n                    else current_selection_score\n                )\n            )\n            if len(val_loader) > 0 else loss\n        )\n'''
    text = replace_once(text, old, new, "hardware-regression early stopping")

    text = replace_once(
        text,
        '''        if absolute_score >= 1.0 or not all(\n            float(ratio) < 1.0 for ratio in absolute_ratios.values()\n        ):\n''',
        '''        if (\n            FLAGS.checkpoint_objective != "hardware_regression"\n            and (\n                absolute_score >= 1.0\n                or not all(\n                    float(ratio) < 1.0 for ratio in absolute_ratios.values()\n                )\n            )\n        ):\n''',
        "bypass obsolete absolute gate",
    )

    text = replace_once(
        text,
        '''        if FLAGS.checkpoint_objective == 'qualified_rank':\n''',
        '''        if FLAGS.checkpoint_objective == "hardware_regression":\n            if best_hardware_regression_epoch is None:\n                raise RuntimeError(\n                    "No hardware-regression validation checkpoint was produced."\n                )\n            selection_tag = "val_hardware_regression"\n            selection_epoch = best_hardware_regression_epoch\n            selection_path = join(\n                saver.model_logdir,\n                "val_hardware_regression_model_state_dict.pth",\n            )\n            saver.log_info(\n                "Final hardware-regression checkpoint: epoch "\n                f"{selection_epoch}; complete validation objective="\n                f"{best_hardware_regression_loss:.6f}"\n            )\n        elif FLAGS.checkpoint_objective == 'qualified_rank':\n''',
        "hardware-regression final restore branch",
    )
    return text


def validate_python(rel: str, text: str) -> None:
    try:
        ast.parse(text, filename=rel)
    except SyntaxError as exc:
        raise RuntimeError(f"Patched {rel} is not valid Python: {exc}") from exc


def atomic_write(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> int:
    require_audited_state()
    patchers = {
        "LLM_branch/common/directive_domains.py": patch_directive_domains,
        "LLM_branch/train/train_SFT_xattn_new.py": patch_stage1,
        "LLM_branch/tests/test_stage1_final_contract.py": patch_stage1_test,
        "GNN_branch/config.py": patch_gnn_config,
        "GNN_branch/model.py": patch_gnn_model,
        "GNN_branch/train_GNN.py": patch_gnn_train,
    }

    originals = {}
    patched = {}
    for rel, transform in patchers.items():
        original = (ROOT / rel).read_text(encoding="utf-8")
        originals[rel] = original
        patched[rel] = transform(original)
        validate_python(rel, patched[rel])

    try:
        for rel, source in patched.items():
            atomic_write(ROOT / rel, source)
        subprocess.run(
            ["git", "diff", "--check", "--", *patchers.keys()], cwd=ROOT, check=True
        )
        subprocess.run(
            [sys.executable, "-m", "py_compile", *patchers.keys()], cwd=ROOT, check=True
        )
    except Exception:
        for rel, source in originals.items():
            atomic_write(ROOT / rel, source)
        raise

    print("Applied final MailoHLS simplification patch.")
    print("Run next:")
    print("  python -m pytest -q LLM_branch/tests/test_stage1_final_contract.py")
    print("  python -m pytest -q GNN_branch/tests")
    print("  git diff --check -- LLM_branch GNN_branch")
    print("  git diff -- LLM_branch GNN_branch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
