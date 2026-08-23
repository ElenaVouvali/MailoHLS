#!/usr/bin/env python3
"""
Resume the MailoHLS GNN-alignment patch after the original patcher stopped at
train_GNN.py because a too-short anchor matched two dictionaries.

This script is intentionally designed for the PARTIALLY PATCHED state produced by
apply_mailohls_gnn_alignment_patch.py:
  - config.py: already patched
  - model.py: already patched
  - train_GNN.py: provenance helpers + training_artifacts assignment already patched
  - the rest: not yet patched

It is safe to re-run: every remaining edit is idempotent (already-applied blocks
are skipped), and ambiguous replacements use larger contextual anchors.

Run from the MailoHLS repository root:
    python resume_mailohls_gnn_alignment_patch.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path.cwd().resolve()


def require_marker(path: Path, marker: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        raise RuntimeError(
            f"{path}: expected already-applied marker missing for {label!r}. "
            "Do not continue; inspect git diff first."
        )


def ensure_replace(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[SKIP] {path}: {label} already applied")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: {label}: expected exactly one old block, found {count}. "
            "The local source differs from the inspected partial-patch state."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[PATCHED] {path}: {label}")


def main() -> None:
    config = ROOT / "GNN_branch/config.py"
    model = ROOT / "GNN_branch/model.py"
    train = ROOT / "GNN_branch/train_GNN.py"
    memory = ROOT / "GNN_branch/build_structural_memory.py"
    tests = ROOT / "GNN_branch/tests/test_stage_b_helpers.py"

    for path in (config, model, train, memory, tests):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run this script from the MailoHLS repo root."
            )

    # Confirm the first patch stopped exactly where the user log says it stopped.
    require_marker(config, '"embedding_rank"', "config checkpoint objective")
    require_marker(config, "--rank_tie_relative", "relative rank tie policy")
    require_marker(model, "self.decompose_targets or self.reference_delta", "kernel-center response decomposition")
    require_marker(model, "rank_tie_relative", "physical ranking tie policy")
    require_marker(train, "def _hash_tensor_tree(directory):", "provenance tree hashing")
    require_marker(train, "training_artifacts = snapshot_gnn_training_artifacts()", "provenance snapshot call")

    # 1) Complete checkpoint contract provenance.
    ensure_replace(
        train,
        """        'target_conditioning_policy': (
            'public_device_capacities_and_clock_in_qor_heads_only'
            if FLAGS.multi_target_qor else 'single_reference_target'
        ),
        'shared_initialization_sha256': shared_initialization_sha256,
    }
    path = Path(saver.model_logdir) / 'gnn_checkpoint_contract.json'
""",
        """        'target_conditioning_policy': (
            'public_device_capacities_and_clock_in_qor_heads_only'
            if FLAGS.multi_target_qor else 'single_reference_target'
        ),
        'shared_initialization_sha256': shared_initialization_sha256,
        'training_artifacts': training_artifacts,
    }
    path = Path(saver.model_logdir) / 'gnn_checkpoint_contract.json'
""",
        "store immutable training-artifact provenance in contract",
    )

    # 2) Add Stage-2-oriented embedding-rank selector.
    ensure_replace(
        train,
        """    improved = ranking_score > float(best_score) + float(min_delta)
    return qualified and improved


def compute_per_kernel_target_baseline_ratios(points_dict, target_stats):
""",
        """    improved = ranking_score > float(best_score) + float(min_delta)
    return qualified and improved


def should_update_embedding_rank(
    target_ratios,
    ranking_score,
    best_score,
    min_delta,
    min_rank_tau,
):
    \"\"\"Select a Stage-2-oriented rank checkpoint with aggregate QoR guardrails.\"\"\"
    ratios = [float(value) for value in target_ratios.values()]
    ranking_score = float(ranking_score)
    if not ratios or not all(np.isfinite(value) for value in ratios):
        raise RuntimeError(f'Invalid target ratios: {target_ratios}')
    if not np.isfinite(ranking_score):
        raise RuntimeError(f'Invalid ranking score: {ranking_score}')
    qualified = (
        all(value < 1.0 for value in ratios)
        and ranking_score >= float(min_rank_tau)
    )
    improved = ranking_score > float(best_score) + float(min_delta)
    return qualified and improved


def compute_per_kernel_target_baseline_ratios(points_dict, target_stats):
""",
        "add embedding-rank checkpoint qualification",
    )

    ensure_replace(
        train,
        """    best_qualified_rank_score = float('-inf')
    best_qualified_rank_epoch = None
    best_qualified_rank_ratios = None
    best_qualified_rank_kernel_ratios = None

    if FLAGS.resume_training and exists(ckpt_path):
""",
        """    best_qualified_rank_score = float('-inf')
    best_qualified_rank_epoch = None
    best_qualified_rank_ratios = None
    best_qualified_rank_kernel_ratios = None
    best_embedding_rank_score = float('-inf')
    best_embedding_rank_epoch = None
    best_embedding_rank_ratios = None

    if FLAGS.resume_training and exists(ckpt_path):
""",
        "initialize embedding-rank tracking",
    )

    ensure_replace(
        train,
        """        best_qualified_rank_kernel_ratios = st.get(
            "best_qualified_rank_kernel_ratios"
        )
        stored_baseline = st.get("baseline_breakdown")
""",
        """        best_qualified_rank_kernel_ratios = st.get(
            "best_qualified_rank_kernel_ratios"
        )
        best_embedding_rank_score = st.get(
            "best_embedding_rank_score", float('-inf')
        )
        best_embedding_rank_epoch = st.get("best_embedding_rank_epoch")
        best_embedding_rank_ratios = st.get(
            "best_embedding_rank_ratios"
        )
        stored_baseline = st.get("baseline_breakdown")
""",
        "restore embedding-rank state on resume",
    )

    ensure_replace(
        train,
        """        if val_selection_scores:
            best_stopping_loss = min(val_selection_scores)
            epochs_without_improvement = max(
                0,
                len(val_selection_scores)
                - 1
                - val_selection_scores.index(best_stopping_loss),
            )
""",
        """        if val_selection_scores:
            if FLAGS.checkpoint_objective == 'embedding_rank':
                best_rank = max(val_ranking_scores)
                best_stopping_loss = -best_rank
                best_index = val_ranking_scores.index(best_rank)
            else:
                best_stopping_loss = min(val_selection_scores)
                best_index = val_selection_scores.index(best_stopping_loss)
            epochs_without_improvement = max(
                0,
                len(val_selection_scores)
                - 1
                - best_index,
            )
""",
        "resume early-stopping state under embedding-rank objective",
    )

    ensure_replace(
        train,
        """            qualified = (
                all(float(ratio) < 1.0 for ratio in target_ratios.values())
                and ranking_score >= FLAGS.min_rank_tau
                and worst_kernel_ratio
                <= FLAGS.max_kernel_zero_baseline_ratio
            )
            saver.log_info(
                f'Validation worst-target kernel-macro tau-b: '
                f'{ranking_score:.6f}; worst kernel/target baseline ratio='
                f'{worst_kernel_ratio:.6f}; qualified={qualified}'
            )
""",
        """            qualified = (
                all(float(ratio) < 1.0 for ratio in target_ratios.values())
                and ranking_score >= FLAGS.min_rank_tau
                and worst_kernel_ratio
                <= FLAGS.max_kernel_zero_baseline_ratio
            )
            embedding_rank_qualified = (
                all(float(ratio) < 1.0 for ratio in target_ratios.values())
                and ranking_score >= FLAGS.min_rank_tau
            )
            saver.log_info(
                f'Validation worst-target kernel-macro tau-b: '
                f'{ranking_score:.6f}; worst kernel/target baseline ratio='
                f'{worst_kernel_ratio:.6f}; qualified_rank={qualified}; '
                f'embedding_rank_qualified={embedding_rank_qualified}'
            )
""",
        "report both strict and embedding-rank qualification",
    )

    ensure_replace(
        train,
        """                saver.log_info(
                    f'Saved qualified rank model at epoch {epoch}; '
                    f'worst-target kernel-macro tau-b={ranking_score:.6f}; '
                    f'worst kernel/target baseline ratio='
                    f'{worst_kernel_ratio:.6f}'
                )
            saver.writer.add_scalar('val/total_objective', val, epoch)
""",
        """                saver.log_info(
                    f'Saved qualified rank model at epoch {epoch}; '
                    f'worst-target kernel-macro tau-b={ranking_score:.6f}; '
                    f'worst kernel/target baseline ratio='
                    f'{worst_kernel_ratio:.6f}'
                )
            if should_update_embedding_rank(
                target_ratios,
                ranking_score,
                best_embedding_rank_score,
                FLAGS.early_stopping_min_delta,
                FLAGS.min_rank_tau,
            ):
                best_embedding_rank_score = ranking_score
                best_embedding_rank_epoch = epoch
                best_embedding_rank_ratios = dict(target_ratios)
                if FLAGS.save_model:
                    save_checkpoint_with_sidecar(
                        model.state_dict(),
                        join(
                            saver.model_logdir,
                            'val_embedding_rank_model_state_dict.pth',
                        ),
                        'val_embedding_rank',
                        epoch,
                    )
                saver.log_info(
                    f'Saved embedding-rank model at epoch {epoch}; '
                    f'worst-target kernel-macro tau-b={ranking_score:.6f}; '
                    f'aggregate target ratios={target_ratios}'
                )
            saver.writer.add_scalar('val/total_objective', val, epoch)
""",
        "save best embedding-rank checkpoint",
    )

    ensure_replace(
        train,
        """            if (
                FLAGS.scheduler == 'plateau'
                and epoch + 1 >= int(FLAGS.warmup_epochs)
            ):
                lr_scheduler.step(current_selection_score)
""",
        """            if (
                FLAGS.scheduler == 'plateau'
                and epoch + 1 >= int(FLAGS.warmup_epochs)
            ):
                scheduler_score = (
                    -ranking_score
                    if FLAGS.checkpoint_objective == 'embedding_rank'
                    else current_selection_score
                )
                lr_scheduler.step(scheduler_score)
""",
        "make plateau scheduler follow active checkpoint objective",
    )

    ensure_replace(
        train,
        """                "best_qualified_rank_kernel_ratios": (
                    best_qualified_rank_kernel_ratios
                ),
                "initial_selection": initial_selection,
""",
        """                "best_qualified_rank_kernel_ratios": (
                    best_qualified_rank_kernel_ratios
                ),
                "best_embedding_rank_score": best_embedding_rank_score,
                "best_embedding_rank_epoch": best_embedding_rank_epoch,
                "best_embedding_rank_ratios": best_embedding_rank_ratios,
                "initial_selection": initial_selection,
""",
        "persist embedding-rank state",
    )

    ensure_replace(
        train,
        """        selection_loss = (
            current_selection_score if len(val_loader) > 0 else loss
        )
""",
        """        selection_loss = (
            (
                -ranking_score
                if FLAGS.checkpoint_objective == 'embedding_rank'
                else current_selection_score
            )
            if len(val_loader) > 0 else loss
        )
""",
        "make early stopping follow active checkpoint objective",
    )

    ensure_replace(
        train,
        """        if FLAGS.checkpoint_objective == 'qualified_rank':
""",
        """        embedding_rank_path = join(
            saver.model_logdir, 'val_embedding_rank_model_state_dict.pth'
        )
        if best_embedding_rank_epoch is not None:
            if best_embedding_rank_ratios is None or not all(
                float(ratio) < 1.0
                for ratio in best_embedding_rank_ratios.values()
            ):
                raise RuntimeError(
                    'Stored embedding-rank checkpoint is not '
                    'aggregate-baseline-qualified.'
                )
            if best_embedding_rank_score < FLAGS.min_rank_tau:
                raise RuntimeError(
                    'Stored embedding-rank checkpoint is below --min_rank_tau.'
                )
            restore_checkpoint(
                embedding_rank_path, 'embedding-rank-selected'
            )
            saver.log_info(
                'Final embedding-rank validation report for epoch '
                f'{best_embedding_rank_epoch}; worst-target kernel-macro '
                f'tau-b={best_embedding_rank_score:.6f}; aggregate target '
                f'ratios={best_embedding_rank_ratios}'
            )
            test(
                val_loader,
                'best_embedding_rank_val',
                model,
                best_embedding_rank_epoch,
                test_losses=[],
            )

        if FLAGS.checkpoint_objective == 'qualified_rank':
""",
        "final embedding-rank validation report",
    )

    ensure_replace(
        train,
        """            selection_tag = 'val_rank'
            selection_epoch = best_qualified_rank_epoch
            selection_path = rank_path
        else:
            selection_tag = 'val'
""",
        """            selection_tag = 'val_rank'
            selection_epoch = best_qualified_rank_epoch
            selection_path = rank_path
        elif FLAGS.checkpoint_objective == 'embedding_rank':
            if best_embedding_rank_epoch is None:
                raise RuntimeError(
                    'No aggregate-baseline-qualified embedding-rank checkpoint '
                    'was produced. The held-out test set remains locked.'
                )
            selection_tag = 'val_embedding_rank'
            selection_epoch = best_embedding_rank_epoch
            selection_path = embedding_rank_path
        else:
            selection_tag = 'val'
""",
        "activate embedding-rank checkpoint",
    )

    ensure_replace(
        train,
        """        if best_qualified_rank_epoch is not None:
            saver.log_info(
                'qualified-rank checkpoint at epoch: '
                f'{best_qualified_rank_epoch}'
            )
        saver.log_info(
""",
        """        if best_qualified_rank_epoch is not None:
            saver.log_info(
                'qualified-rank checkpoint at epoch: '
                f'{best_qualified_rank_epoch}'
            )
        if best_embedding_rank_epoch is not None:
            saver.log_info(
                'embedding-rank checkpoint at epoch: '
                f'{best_embedding_rank_epoch}'
            )
        saver.log_info(
""",
        "log embedding-rank selection epoch",
    )

    # 3) Enforce exact static-tensor provenance at Stage-2 export.
    ensure_replace(
        memory,
        """    model.eval()
    checkpoint_sha256 = _sha256(args.ckpt)
    source_pt_manifest_sha256 = _source_pt_manifest_sha256(pt_files)
    source_gexf_files = [
""",
        """    model.eval()
    checkpoint_sha256 = _sha256(args.ckpt)
    source_pt_manifest_sha256 = _source_pt_manifest_sha256(pt_files)

    # Stage-2 export must use the exact static MLIR tensor tree that trained
    # this checkpoint. Matching tensor dimensions alone is not sufficient.
    expected_static_tree = (
        contract.get('training_artifacts', {})
        .get('static_graph_tensor_tree', {})
    )
    expected_static_sha256 = expected_static_tree.get('sha256')
    if (
        expected_static_sha256 is not None
        and source_pt_manifest_sha256 != expected_static_sha256
    ):
        raise RuntimeError(
            'Static MLIR tensor tree differs from the exact tensor tree used '
            'to train this GNN checkpoint.'
        )
    expected_static_count = expected_static_tree.get('count')
    if (
        expected_static_count is not None
        and len(pt_files) != int(expected_static_count)
    ):
        raise RuntimeError(
            'Static MLIR tensor count differs from GNN training provenance: '
            f'{len(pt_files)} != {expected_static_count}'
        )

    source_gexf_files = [
""",
        "verify exact static tensor tree before Stage-2 export",
    )

    # 4) Add a small unit test for the new selector.
    ensure_replace(
        tests,
        """    def test_macro_rank_is_equal_kernel_and_worst_target(self):
""",
        """    def test_embedding_rank_uses_aggregate_guardrails_and_tau(self):
        update = _load_function(
            "should_update_embedding_rank", {"np": np}
        )
        self.assertTrue(update(
            {"perf": 0.8, "area": 0.9},
            0.30, 0.20, 1e-4, 0.20,
        ))
        self.assertFalse(update(
            {"perf": 1.01, "area": 0.5},
            0.90, 0.20, 1e-4, 0.20,
        ))
        self.assertFalse(update(
            {"perf": 0.8, "area": 0.9},
            0.19, 0.10, 1e-4, 0.20,
        ))

    def test_macro_rank_is_equal_kernel_and_worst_target(self):
""",
        "test embedding-rank qualification",
    )

    # Static validation.
    for path in (config, model, train, memory, tests):
        subprocess.run(["python", "-m", "py_compile", str(path)], check=True)

    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)

    print("\n[DONE] Partial GNN patch completed successfully.")
    print("Next run:")
    print("  python -m pytest -q GNN_branch/tests")
    print("Then inspect:")
    print("  git diff -- GNN_branch")


if __name__ == "__main__":
    main()
