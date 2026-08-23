#!/usr/bin/env python3
# Apply the MailoHLS GNN-alignment patch.
#
# Target inspected:
#   ElenaVouvali/MailoHLS, stage2-analysis-refactor
#   commit 9902342127653313199876e3d1732489098714bc
#
# Run from the repository root:
#   python /path/to/apply_mailohls_gnn_alignment_patch.py

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path.cwd().resolve()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one matching patch block, found {count}. "
            "The source may have changed; do not apply this patch blindly."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[PATCHED] {path}")


def main() -> None:
    required = [
        ROOT / "GNN_branch/config.py",
        ROOT / "GNN_branch/model.py",
        ROOT / "GNN_branch/train_GNN.py",
        ROOT / "GNN_branch/build_structural_memory.py",
        ROOT / "GNN_branch/tests/test_stage_b_helpers.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run this script from the MailoHLS repository root. Missing: "
            + ", ".join(str(path) for path in missing)
        )

    # ------------------------------------------------------------------
    # config.py
    # ------------------------------------------------------------------
    path = ROOT / "GNN_branch/config.py"

    replace_once(
        path,
        '''parser.add_argument(
    "--checkpoint_objective",
    choices=("absolute", "qualified_rank"),
    default="absolute",
    help=(
        "Select either the lowest absolute validation-error checkpoint or "
        "the best within-kernel ranking checkpoint that also beats the "
        "constant baseline on every target."
    ),
)
''',
        '''parser.add_argument(
    "--checkpoint_objective",
    choices=("absolute", "qualified_rank", "embedding_rank"),
    default="absolute",
    help=(
        "Select either the lowest absolute validation-error checkpoint or "
        "a within-kernel ranking checkpoint. qualified_rank retains the "
        "strict per-kernel absolute-error gate used by earlier experiments; "
        "embedding_rank maximizes worst-target kernel-macro Kendall tau while "
        "requiring every aggregate target loss to beat its no-learning baseline."
    ),
)
''',
    )

    replace_once(
        path,
        '''parser.add_argument('--rank_aux_weight', type=float, default=0.0)
parser.add_argument('--rank_temperature', type=float, default=1.0)
parser.add_argument('--rank_tie_epsilon', type=float, default=0.05)
parser.add_argument('--resource_aux_weight', type=float, default=0.0)
''',
        '''parser.add_argument('--rank_aux_weight', type=float, default=0.0)
parser.add_argument('--rank_temperature', type=float, default=1.0)
rank_tie_group = parser.add_mutually_exclusive_group()
rank_tie_group.add_argument(
    '--rank_tie_relative',
    type=float,
    default=None,
    help=(
        'Ignore within-kernel ranking pairs whose measured QoR differs by no '
        'more than this relative fraction. With --norm_method log2, 0.05 means '
        'a 5% QoR tie band. This is the recommended MailoHLS ranking policy.'
    ),
)
rank_tie_group.add_argument(
    '--rank_tie_epsilon',
    type=float,
    default=None,
    help=(
        'Legacy compatibility: tie threshold in the model loss-target space. '
        'If neither tie option is supplied, the historical value 0.05 is used.'
    ),
)
parser.add_argument('--resource_aux_weight', type=float, default=0.0)
''',
    )

    replace_once(
        path,
        '''parser.add_argument('--resource_boundary_tolerance', type=float, default=0.02)
parser.add_argument('--kernel_grouped_sampling', action='store_true')
parser.add_argument('--kernels_per_batch', type=int, default=16)
parser.add_argument('--points_per_kernel', type=int, default=4)
''',
        '''parser.add_argument('--resource_boundary_tolerance', type=float, default=0.02)
parser.add_argument('--kernel_grouped_sampling', action='store_true')
parser.add_argument('--kernels_per_batch', type=int, default=16)
parser.add_argument(
    '--points_per_kernel',
    type=int,
    default=4,
    help=(
        'Design points drawn from each kernel in one GNN microbatch. Four '
        'points expose up to six within-kernel pairs to the ranking loss.'
    ),
)
''',
    )

    replace_once(
        path,
        '''if FLAGS.rank_aux_weight < 0:
    parser.error('--rank_aux_weight must be non-negative.')
if FLAGS.rank_temperature <= 0:
    parser.error('--rank_temperature must be positive.')
if FLAGS.rank_tie_epsilon < 0:
    parser.error('--rank_tie_epsilon must be non-negative.')
if FLAGS.resource_aux_weight < 0:
    parser.error('--resource_aux_weight must be non-negative.')
''',
        '''if FLAGS.rank_aux_weight < 0:
    parser.error('--rank_aux_weight must be non-negative.')
if FLAGS.rank_temperature <= 0:
    parser.error('--rank_temperature must be positive.')
if FLAGS.rank_tie_relative is None and FLAGS.rank_tie_epsilon is None:
    # Preserve historical behavior unless a new run explicitly opts into the
    # physically interpretable relative tie band.
    FLAGS.rank_tie_epsilon = 0.05
if FLAGS.rank_tie_relative is not None:
    if not 0.0 <= FLAGS.rank_tie_relative < 1.0:
        parser.error('--rank_tie_relative must be in [0, 1).')
    if FLAGS.norm_method != 'log2':
        parser.error('--rank_tie_relative requires --norm_method log2.')
if FLAGS.rank_tie_epsilon is not None and FLAGS.rank_tie_epsilon < 0:
    parser.error('--rank_tie_epsilon must be non-negative.')
if FLAGS.rank_aux_weight > 0:
    if not FLAGS.kernel_grouped_sampling:
        parser.error(
            '--rank_aux_weight > 0 requires --kernel_grouped_sampling so '
            'same-kernel design points coexist in each microbatch.'
        )
    if FLAGS.points_per_kernel < 2:
        parser.error(
            '--rank_aux_weight > 0 requires --points_per_kernel >= 2.'
        )
if FLAGS.resource_aux_weight < 0:
    parser.error('--resource_aux_weight must be non-negative.')
''',
    )

    # ------------------------------------------------------------------
    # model.py
    # ------------------------------------------------------------------
    path = ROOT / "GNN_branch/model.py"

    replace_once(
        path,
        '''import torch
import torch.nn.functional as F
''',
        '''import math
import torch
import torch.nn.functional as F
''',
    )

    replace_once(
        path,
        '''          response_in_D = (
              (in_D * 2 if self.reference_delta else in_D)
              + self.target_condition_dim
          )
''',
        '''          # Delta-oriented modes expose both the static kernel state
          # and the change induced by the pragma configuration. kernel_center
          # therefore learns its baseline instead of requiring a measured
          # neutral HLS design at inference.
          response_in_D = (
              (in_D * 2 if (self.decompose_targets or self.reference_delta)
               else in_D)
              + self.target_condition_dim
          )
''',
    )

    replace_once(
        path,
        '''        out_embed = (
            torch.cat((static_embed, out - static_embed), dim=1)
            if self.reference_delta else out
        )
''',
        '''        out_embed = (
            torch.cat((static_embed, out - static_embed), dim=1)
            if (self.decompose_targets or self.reference_delta)
            else out
        )
''',
    )

    replace_once(
        path,
        '''                    rank_loss = within_kernel_rank_loss(
                        out,
                        loss_target,
                        kernels,
                        temperature=float(FLAGS.rank_temperature),
                        tie_epsilon=float(FLAGS.rank_tie_epsilon),
                    )
''',
        '''                    # Ranking direction is learned from measured QoR
                    # in original log2 space. Regression may remain standardized;
                    # only the tie decision is decoupled from standardization.
                    if getattr(FLAGS, 'rank_tie_relative', None) is not None:
                        rank_target = target
                        rank_tie_epsilon = math.log2(
                            1.0 + float(FLAGS.rank_tie_relative)
                        )
                    else:
                        # Exact legacy behavior for old commands/checkpoints.
                        rank_target = loss_target
                        rank_tie_epsilon = float(FLAGS.rank_tie_epsilon)
                    rank_loss = within_kernel_rank_loss(
                        out,
                        rank_target,
                        kernels,
                        temperature=float(FLAGS.rank_temperature),
                        tie_epsilon=rank_tie_epsilon,
                    )
''',
    )

    # ------------------------------------------------------------------
    # train_GNN.py
    # ------------------------------------------------------------------
    path = ROOT / "GNN_branch/train_GNN.py"

    replace_once(
        path,
        '''def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
''',
        '''def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash_tensor_tree(directory):
    "Hash the exact cached tensor files consumed by GNN training."
    directory = Path(directory).resolve()
    paths = sorted(directory.glob('*.pt'), key=lambda path: path.name)
    if not paths:
        raise RuntimeError(f'No .pt files found for provenance: {directory}')
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\\0')
        digest.update(_sha256_file(path).encode('ascii'))
        digest.update(b'\\n')
    return {
        'path': str(directory),
        'count': len(paths),
        'sha256': digest.hexdigest(),
    }


def snapshot_gnn_training_artifacts():
    "Freeze small provenance files and hash the large tensor trees."
    snapshot_dir = Path(saver.model_logdir) / 'provenance'
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        'feature_schema.json': Path(data.SCHEMA_PATH).resolve(),
        'index.pt': Path(data.INDEX_PATH).resolve(),
        'encoders.pkl': Path(data.ENCODER_PATH).resolve(),
        'pragma_dim.pt': Path(data.PRAGMA_DIM_PATH).resolve(),
        'config.py': Path(__file__).with_name('config.py').resolve(),
        'model.py': Path(__file__).with_name('model.py').resolve(),
        'train_GNN.py': Path(__file__).resolve(),
        'mlir_data.py': Path(__file__).with_name('mlir_data.py').resolve(),
        'mlir_graph_gen.py': Path(__file__).with_name('mlir_graph_gen.py').resolve(),
        'utils.py': Path(__file__).with_name('utils.py').resolve(),
        'nn_att.py': Path(__file__).with_name('nn_att.py').resolve(),
        'saver.py': Path(__file__).with_name('saver.py').resolve(),
        'reference_delta.py': Path(__file__).with_name('reference_delta.py').resolve(),
    }
    if FLAGS.split_json:
        sources['experiment_split.json'] = Path(FLAGS.split_json).resolve()

    files = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f'Missing GNN provenance source: {source}')
        destination = snapshot_dir / name
        shutil.copy2(source, destination)
        files[name] = {
            'source_path': str(source),
            'snapshot_path': str(destination),
            'sha256': _sha256_file(destination),
        }

    manifest = {
        'schema': 'mailohls-gnn-training-artifacts-v1',
        'git_commit': _git_commit(),
        'files': files,
        'static_graph_tensor_tree': _hash_tensor_tree(data.GRAPH_DIR),
        'design_point_tensor_tree': _hash_tensor_tree(data.POINT_DIR),
    }
    manifest_path = snapshot_dir / 'provenance_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\\n',
        encoding='utf-8',
    )
    return {
        **manifest,
        'manifest_path': str(manifest_path),
        'manifest_sha256': _sha256_file(manifest_path),
    }


def _jsonable(value):
''',
    )

    replace_once(
        path,
        '''    model_init_flags = {
        name: resolved_flags[name]
        for name in model_init_flag_names if name in resolved_flags
    }
    contract = {
''',
        '''    model_init_flags = {
        name: resolved_flags[name]
        for name in model_init_flag_names if name in resolved_flags
    }
    training_artifacts = snapshot_gnn_training_artifacts()
    contract = {
''',
    )

    replace_once(
        path,
        '''        'shared_initialization_sha256': shared_initialization_sha256,
    }
''',
        '''        'shared_initialization_sha256': shared_initialization_sha256,
        'training_artifacts': training_artifacts,
    }
''',
    )

    replace_once(
        path,
        '''    improved = ranking_score > float(best_score) + float(min_delta)
    return qualified and improved


def compute_per_kernel_target_baseline_ratios(points_dict, target_stats):
''',
        '''    improved = ranking_score > float(best_score) + float(min_delta)
    return qualified and improved


def should_update_embedding_rank(
    target_ratios,
    ranking_score,
    best_score,
    min_delta,
    min_rank_tau,
):
    "Select a Stage-2-oriented rank checkpoint with aggregate QoR guardrails."
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
''',
    )

    replace_once(
        path,
        '''    best_qualified_rank_score = float('-inf')
    best_qualified_rank_epoch = None
    best_qualified_rank_ratios = None
    best_qualified_rank_kernel_ratios = None

    if FLAGS.resume_training and exists(ckpt_path):
''',
        '''    best_qualified_rank_score = float('-inf')
    best_qualified_rank_epoch = None
    best_qualified_rank_ratios = None
    best_qualified_rank_kernel_ratios = None
    best_embedding_rank_score = float('-inf')
    best_embedding_rank_epoch = None
    best_embedding_rank_ratios = None

    if FLAGS.resume_training and exists(ckpt_path):
''',
    )

    replace_once(
        path,
        '''        best_qualified_rank_kernel_ratios = st.get(
            "best_qualified_rank_kernel_ratios"
        )
        stored_baseline = st.get("baseline_breakdown")
''',
        '''        best_qualified_rank_kernel_ratios = st.get(
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
''',
    )

    replace_once(
        path,
        '''        if val_selection_scores:
            best_stopping_loss = min(val_selection_scores)
            epochs_without_improvement = max(
                0,
                len(val_selection_scores)
                - 1
                - val_selection_scores.index(best_stopping_loss),
            )
''',
        '''        if val_selection_scores:
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
''',
    )

    replace_once(
        path,
        '''            qualified = (
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
''',
        '''            qualified = (
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
''',
    )

    replace_once(
        path,
        '''                saver.log_info(
                    f'Saved qualified rank model at epoch {epoch}; '
                    f'worst-target kernel-macro tau-b={ranking_score:.6f}; '
                    f'worst kernel/target baseline ratio='
                    f'{worst_kernel_ratio:.6f}'
                )
            saver.writer.add_scalar('val/total_objective', val, epoch)
''',
        '''                saver.log_info(
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
''',
    )

    replace_once(
        path,
        '''            if (
                FLAGS.scheduler == 'plateau'
                and epoch + 1 >= int(FLAGS.warmup_epochs)
            ):
                lr_scheduler.step(current_selection_score)
''',
        '''            if (
                FLAGS.scheduler == 'plateau'
                and epoch + 1 >= int(FLAGS.warmup_epochs)
            ):
                scheduler_score = (
                    -ranking_score
                    if FLAGS.checkpoint_objective == 'embedding_rank'
                    else current_selection_score
                )
                lr_scheduler.step(scheduler_score)
''',
    )

    replace_once(
        path,
        '''                "best_qualified_rank_kernel_ratios": (
                    best_qualified_rank_kernel_ratios
                ),
                "initial_selection": initial_selection,
''',
        '''                "best_qualified_rank_kernel_ratios": (
                    best_qualified_rank_kernel_ratios
                ),
                "best_embedding_rank_score": best_embedding_rank_score,
                "best_embedding_rank_epoch": best_embedding_rank_epoch,
                "best_embedding_rank_ratios": best_embedding_rank_ratios,
                "initial_selection": initial_selection,
''',
    )

    replace_once(
        path,
        '''        selection_loss = (
            current_selection_score if len(val_loader) > 0 else loss
        )
''',
        '''        selection_loss = (
            (
                -ranking_score
                if FLAGS.checkpoint_objective == 'embedding_rank'
                else current_selection_score
            )
            if len(val_loader) > 0 else loss
        )
''',
    )

    replace_once(
        path,
        '''        if FLAGS.checkpoint_objective == 'qualified_rank':
''',
        '''        embedding_rank_path = join(
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
''',
    )

    replace_once(
        path,
        '''            selection_tag = 'val_rank'
            selection_epoch = best_qualified_rank_epoch
            selection_path = rank_path
        else:
            selection_tag = 'val'
''',
        '''            selection_tag = 'val_rank'
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
''',
    )

    replace_once(
        path,
        '''        if best_qualified_rank_epoch is not None:
            saver.log_info(
                'qualified-rank checkpoint at epoch: '
                f'{best_qualified_rank_epoch}'
            )
        saver.log_info(
''',
        '''        if best_qualified_rank_epoch is not None:
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
''',
    )

    # ------------------------------------------------------------------
    # build_structural_memory.py
    # ------------------------------------------------------------------
    path = ROOT / "GNN_branch/build_structural_memory.py"
    replace_once(
        path,
        '''    model.eval()
    checkpoint_sha256 = _sha256(args.ckpt)
    source_pt_manifest_sha256 = _source_pt_manifest_sha256(pt_files)
    source_gexf_files = [
''',
        '''    model.eval()
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
''',
    )

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    path = ROOT / "GNN_branch/tests/test_stage_b_helpers.py"
    replace_once(
        path,
        '''    def test_macro_rank_is_equal_kernel_and_worst_target(self):
''',
        '''    def test_embedding_rank_uses_aggregate_guardrails_and_tau(self):
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
''',
    )

    for candidate in required:
        if candidate.suffix == ".py":
            subprocess.run(
                ["python", "-m", "py_compile", str(candidate)],
                check=True,
            )

    print("\nPatch applied and Python syntax checks passed.")
    print("Next: python -m pytest -q GNN_branch/tests")
    print("Then inspect: git diff --check && git diff -- GNN_branch")


if __name__ == "__main__":
    main()
