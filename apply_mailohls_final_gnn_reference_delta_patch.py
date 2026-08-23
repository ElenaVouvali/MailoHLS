#!/usr/bin/env python3
# coding: utf-8
"""Apply the final MailoHLS reference-delta GNN alignment patch.

Base inspected: ElenaVouvali/MailoHLS, branch stage2-analysis-refactor.

This patch intentionally does NOT change the MLIR dataset, graph construction,
GNN architecture, regression equations, split, or pragma-conditioning path.
It only aligns checkpoint control/provenance with the final Stage-2 role:
  * reference-delta regression must first beat the zero-delta baseline;
  * among qualified checkpoints, worst-target kernel-macro Kendall tau controls
    LR/early stopping and the existing embedding-rank checkpoint remains active;
  * the neutral-reference manifest is frozen into checkpoint provenance;
  * static_pre_npt Stage-2 export is explicitly recorded as baseline-free.

Run from the MailoHLS repository root:
    python apply_mailohls_final_gnn_reference_delta_patch.py
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

ROOT = Path.cwd().resolve()
TRAIN = ROOT / "GNN_branch/train_GNN.py"
BUILD = ROOT / "GNN_branch/build_structural_memory.py"
TEST = ROOT / "GNN_branch/tests/test_stage_c_helpers.py"

EXPECTED = {
    TRAIN: "7a327915bf490a7eee73cdeecb62fe8b8b0410a9",
    BUILD: "323c49dec38323a62aa584f03daf7d5aaa95e033",
    TEST: "bac41290a484cecba44db701b36830e2a7b5f0f5",
}


def blob_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)], cwd=ROOT, text=True
    ).strip()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"{label}: expected exactly one matching block, found {n}. "
            "No files were changed."
        )
    return text.replace(old, new, 1)


CONTROL_HELPER = r'''
def embedding_rank_control_score(
    selection_score,
    target_ratios,
    ranking_score,
):
    """Qualification-aware control score for the Stage-2 GNN encoder.

    reference_delta has a meaningful no-learning predictor: zero delta, i.e.
    leave the measured neutral QoR unchanged.  Before BOTH public QoR targets
    beat that baseline, LR scheduling/early stopping should continue to favor
    target-balanced regression.  Once both targets beat it, the control score
    switches to negative worst-target kernel-macro Kendall tau so lower remains
    better for the existing scheduler/early-stopping code.

    Checkpoint qualification itself is intentionally unchanged: the existing
    embedding-rank checkpoint is still saved only when all aggregate target
    ratios are < 1 and --min_rank_tau is met.
    """
    ratios = [float(value) for value in target_ratios.values()]
    if not ratios:
        raise ValueError("embedding-rank control requires target ratios")
    if not all(np.isfinite(value) and value >= 0.0 for value in ratios):
        raise ValueError(f"invalid target ratios: {target_ratios}")

    selection_score = float(selection_score)
    ranking_score = float(ranking_score)
    if not np.isfinite(selection_score) or not np.isfinite(ranking_score):
        raise ValueError("embedding-rank control received a non-finite score")

    if all(value < 1.0 for value in ratios):
        return -ranking_score
    return selection_score
'''


def patch_train(text: str) -> str:
    # 1. Add one small, testable helper for scheduler/early-stop alignment.
    text = replace_once(
        text,
        "\ndef _git_commit():\n",
        "\n" + CONTROL_HELPER.strip() + "\n\n\ndef _git_commit():\n",
        "train: insert embedding_rank_control_score",
    )

    # 2. Freeze the neutral-reference manifest used to construct delta labels.
    old = '''    }
    if FLAGS.split_json:
        sources['experiment_split.json'] = Path(FLAGS.split_json).resolve()
'''
    new = '''    }
    if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta':
        if not FLAGS.baseline_manifest:
            raise RuntimeError(
                'reference_delta provenance requires --baseline_manifest.'
            )
        sources['neutral_baseline_manifest.csv'] = (
            Path(FLAGS.baseline_manifest).expanduser().resolve()
        )
    if FLAGS.split_json:
        sources['experiment_split.json'] = Path(FLAGS.split_json).resolve()
'''
    text = replace_once(text, old, new, "train: snapshot baseline manifest")

    # 3. Make the final encoder deployment contract explicit in the checkpoint.
    old = '''        'target_conditioning_policy': (
            'public_device_capacities_and_clock_in_qor_heads_only'
            if FLAGS.multi_target_qor else 'single_reference_target'
        ),
        'shared_initialization_sha256': shared_initialization_sha256,
'''
    new = '''        'target_conditioning_policy': (
            'public_device_capacities_and_clock_in_qor_heads_only'
            if FLAGS.multi_target_qor else 'single_reference_target'
        ),
        'reference_baseline_manifest': (
            {
                'path': str(
                    Path(FLAGS.baseline_manifest).expanduser().resolve()
                ),
                'sha256': _sha256_file(
                    Path(FLAGS.baseline_manifest).expanduser().resolve()
                ),
            }
            if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta'
            else None
        ),
        'mailohls_structural_encoder': {
            'training_target_mode': getattr(
                FLAGS, 'target_mode', 'absolute'
            ),
            'training_supervision': (
                'reference_delta_qor+within_kernel_rank+resource_aux'
                if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta'
                else 'qor'
            ),
            'stage2_embedding_mode': 'static_pre_npt',
            'reference_baseline_role': (
                'training_target_and_optional_absolute_qor_calibration'
            ),
            'reference_baseline_required_for_stage2_memory': False,
            'checkpoint_selection': (
                'worst_target_kernel_macro_tau_b_after_'
                'aggregate_zero_delta_qualification'
            ),
        },
        'shared_initialization_sha256': shared_initialization_sha256,
'''
    text = replace_once(text, old, new, "train: encoder provenance contract")

    # 4. Resume reconstruction must use the same two-phase control metric.
    old = '''        if val_selection_scores:
            if FLAGS.checkpoint_objective == 'embedding_rank':
                best_rank = max(val_ranking_scores)
                best_stopping_loss = -best_rank
                best_index = val_ranking_scores.index(best_rank)
            else:
                best_stopping_loss = min(val_selection_scores)
                best_index = val_selection_scores.index(best_stopping_loss)
'''
    new = '''        if val_selection_scores:
            if FLAGS.checkpoint_objective == 'embedding_rank':
                control_scores = [
                    embedding_rank_control_score(
                        selection_score,
                        target_ratios,
                        ranking_score,
                    )
                    for selection_score, target_ratios, ranking_score
                    in zip(
                        val_selection_scores,
                        val_selection_ratios,
                        val_ranking_scores,
                    )
                ]
                best_stopping_loss = min(control_scores)
                best_index = control_scores.index(best_stopping_loss)
            else:
                best_stopping_loss = min(val_selection_scores)
                best_index = val_selection_scores.index(best_stopping_loss)
'''
    text = replace_once(text, old, new, "train: resume control score")

    # 5. Report whether the run is still qualifying regression or optimizing rank.
    old = '''            saver.log_info(
                f'Validation worst-target kernel-macro tau-b: '
                f'{ranking_score:.6f}; worst kernel/target baseline ratio='
                f'{worst_kernel_ratio:.6f}; qualified_rank={qualified}; '
                f'embedding_rank_qualified={embedding_rank_qualified}'
            )
'''
    new = '''            embedding_control_score = embedding_rank_control_score(
                current_selection_score,
                target_ratios,
                ranking_score,
            )
            embedding_control_phase = (
                'rank'
                if all(
                    float(ratio) < 1.0
                    for ratio in target_ratios.values()
                )
                else 'zero_delta_qualification'
            )
            saver.log_info(
                f'Validation worst-target kernel-macro tau-b: '
                f'{ranking_score:.6f}; worst kernel/target baseline ratio='
                f'{worst_kernel_ratio:.6f}; qualified_rank={qualified}; '
                f'embedding_rank_qualified={embedding_rank_qualified}; '
                f'embedding_control_phase={embedding_control_phase}; '
                f'embedding_control_score={embedding_control_score:.6f}'
            )
'''
    text = replace_once(text, old, new, "train: control-phase diagnostics")

    # 6. Plateau scheduler: regression until qualified, then ranking.
    old = '''                scheduler_score = (
                    -ranking_score
                    if FLAGS.checkpoint_objective == 'embedding_rank'
                    else current_selection_score
                )
                lr_scheduler.step(scheduler_score)
'''
    new = '''                scheduler_score = (
                    embedding_rank_control_score(
                        current_selection_score,
                        target_ratios,
                        ranking_score,
                    )
                    if FLAGS.checkpoint_objective == 'embedding_rank'
                    else current_selection_score
                )
                lr_scheduler.step(scheduler_score)
'''
    text = replace_once(text, old, new, "train: scheduler control score")

    # 7. Early stopping: use exactly the same criterion as the scheduler.
    old = '''        selection_loss = (
            (
                -ranking_score
                if FLAGS.checkpoint_objective == 'embedding_rank'
                else current_selection_score
            )
            if len(val_loader) > 0 else loss
        )
'''
    new = '''        selection_loss = (
            (
                embedding_rank_control_score(
                    current_selection_score,
                    target_ratios,
                    ranking_score,
                )
                if FLAGS.checkpoint_objective == 'embedding_rank'
                else current_selection_score
            )
            if len(val_loader) > 0 else loss
        )
'''
    text = replace_once(text, old, new, "train: early-stop control score")

    return text


def patch_build(text: str) -> str:
    # The final memory builder never consumes the measured neutral baseline.
    old = '''    if sidecar.get('provenance_status') != contract['provenance_status']:
        raise RuntimeError('Checkpoint and contract provenance status differ.')
    _verify_hash(
'''
    new = '''    if sidecar.get('provenance_status') != contract['provenance_status']:
        raise RuntimeError('Checkpoint and contract provenance status differ.')

    encoder_contract = contract.get('mailohls_structural_encoder', {})
    if encoder_contract:
        required_mode = encoder_contract.get(
            'stage2_embedding_mode', 'static_pre_npt'
        )
        if args.embedding_mode != required_mode:
            raise ValueError(
                'MailoHLS structural memory from this checkpoint requires '
                f'--embedding_mode {required_mode}; got {args.embedding_mode}.'
            )
        if encoder_contract.get(
            'reference_baseline_required_for_stage2_memory', True
        ):
            raise RuntimeError(
                'Checkpoint contract unexpectedly requires a reference '
                'baseline during structural-memory export.'
            )
        print(
            '[STRUCTURAL-EXPORT] static MLIR graph only; neutral HLS '
            'reference is not consumed by Stage-2 memory export.'
        )

    _verify_hash(
'''
    text = replace_once(text, old, new, "build: enforce static_pre_npt")

    old = '''                "embedding_mode": args.embedding_mode,
                "disable_pragma_injection": True,

                "gnn_checkpoint_sha256":
'''
    new = '''                "embedding_mode": args.embedding_mode,
                "disable_pragma_injection": True,
                "gnn_training_target_mode": (
                    contract.get('model_init_flags', {}).get('target_mode')
                ),
                "reference_baseline_used_for_structural_export": False,

                "gnn_checkpoint_sha256":
'''
    text = replace_once(text, old, new, "build: pack baseline-free provenance")

    old = '''        'embedding_mode': args.embedding_mode,
        'exporter_git_commit': git_commit,
'''
    new = '''        'embedding_mode': args.embedding_mode,
        'gnn_training_target_mode': (
            contract.get('model_init_flags', {}).get('target_mode')
        ),
        'reference_baseline_used_for_structural_export': False,
        'reference_baseline_required_for_stage2_memory': False,
        'exporter_git_commit': git_commit,
'''
    text = replace_once(text, old, new, "build: manifest baseline-free provenance")
    return text


TESTS = r'''

    def test_embedding_rank_control_qualifies_regression_before_ranking(self):
        control = load_named_definition(
            ROOT / "train_GNN.py",
            "embedding_rank_control_score",
            {"np": np},
        )
        self.assertAlmostEqual(
            control(1.20, {"perf": 0.70, "area": 1.20}, 0.45),
            1.20,
        )
        self.assertAlmostEqual(
            control(0.80, {"perf": 0.70, "area": 0.80}, 0.45),
            -0.45,
        )

    def test_structural_export_explicitly_does_not_use_reference_baseline(self):
        source = (ROOT / "build_structural_memory.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"reference_baseline_used_for_structural_export": False',
            source,
        )
        self.assertIn(
            "'reference_baseline_required_for_stage2_memory': False",
            source,
        )
'''


def patch_test(text: str) -> str:
    anchor = '\n\nif __name__ == "__main__":\n    unittest.main()\n'
    return replace_once(
        text,
        anchor,
        TESTS + '\n\nif __name__ == "__main__":\n    unittest.main()\n',
        "tests: final GNN alignment tests",
    )


def main() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Run from the MailoHLS repository root; missing {path}"
            )
        actual = blob_sha(path)
        if actual != expected:
            raise RuntimeError(
                f"{path}: local blob {actual} != inspected branch blob "
                f"{expected}. No files were changed. Inspect `git diff -- "
                f"{path.relative_to(ROOT)}` instead of forcing the patch."
            )

    originals = {path: path.read_text(encoding="utf-8") for path in EXPECTED}
    modified = {
        TRAIN: patch_train(originals[TRAIN]),
        BUILD: patch_build(originals[BUILD]),
        TEST: patch_test(originals[TEST]),
    }

    # Preflight every generated file before touching the repository.
    for path, content in modified.items():
        ast.parse(content, filename=str(path))

    temps = {}
    try:
        for path, content in modified.items():
            temp = path.with_name(
                f"{path.name}.mailohls-final-gnn-{os.getpid()}.tmp"
            )
            temp.write_text(content, encoding="utf-8")
            ast.parse(temp.read_text(encoding="utf-8"), filename=str(temp))
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
            [
                "git", "diff", "--check", "--",
                *[str(path.relative_to(ROOT)) for path in modified],
            ],
            cwd=ROOT,
            check=True,
        )
    except Exception:
        for path, content in originals.items():
            path.write_text(content, encoding="utf-8")
        for temp in temps.values():
            if temp.exists():
                temp.unlink()
        raise

    print("[DONE] Final reference-delta GNN patch applied transactionally.")
    print("[UNCHANGED] MLIR dataset, graph schema, GNN architecture, losses, split.")
    print("Run:")
    print("  python -m pytest -q GNN_branch/tests/test_stage_c_helpers.py")
    print("  python -m pytest -q GNN_branch/tests")
    print("  git diff --check -- GNN_branch/train_GNN.py GNN_branch/build_structural_memory.py GNN_branch/tests/test_stage_c_helpers.py")


if __name__ == "__main__":
    main()
