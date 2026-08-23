#-----------------------------------------------------------
#                       train_GNN.py
#-----------------------------------------------------------

from config import FLAGS
from saver import saver
from utils import (
    MLP, OurTimer, MLP_multi_objective, plot_loss_trend,
    _get_y_with_target, create_dir_if_not_exists, plot_lr_trend,
    hash_state_dict, set_reproducible_seed,
)
# from data import MyOwnDataset, get_kernel_samples, split_dataset, split_dataset_resample, split_train_test_kernel
# import data
from mlir_data import (
    MyOwnDataset,
    get_kernel_samples,
    split_dataset,
    split_dataset_resample,
    split_train_val_test_kernel,
)
import mlir_data as data
SAVE_DIR = data.SAVE_DIR

from model import Net
from reference_delta import ReferenceDeltaDataset, load_reference_baselines

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix

import torch
import pytorch_warmup as warmup
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, Sampler, Subset, WeightedRandomSampler
import torch.nn as nn
import shutil
import numpy as np
from scipy.stats import kendalltau

from tqdm import tqdm
from os.path import join, exists, basename

from collections import Counter, OrderedDict, defaultdict

import pandas as pd
import hashlib
import json
import os
from pathlib import Path
import subprocess

RESOURCE_NAMES = ("bram", "dsp", "ff", "lut")
GNN_CHECKPOINT_CONTRACT_PATH = None


def _sha256_file(path):
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
        digest.update(b'\0')
        digest.update(_sha256_file(path).encode('ascii'))
        digest.update(b'\n')
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
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return {
        **manifest,
        'manifest_path': str(manifest_path),
        'manifest_sha256': _sha256_file(manifest_path),
    }


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _canonical_json_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


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


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _dataset_records(dataset):
    if hasattr(dataset, 'records'):
        return list(dataset.records)
    if isinstance(dataset, Subset):
        records = _dataset_records(dataset.dataset)
        return [records[index] for index in dataset.indices]
    if hasattr(dataset, 'dataset'):
        return _dataset_records(dataset.dataset)
    raise TypeError(f'Cannot extract split records from {type(dataset)!r}.')


def _split_sha256(datasets):
    payload = {
        name: _jsonable(_dataset_records(dataset))
        for name, dataset in zip(('train', 'val', 'test'), datasets)
    }
    return _canonical_json_sha256(payload)


def write_gnn_checkpoint_contract(
    *, num_features, edge_dim, target_stats, resource_stats,
    resource_diagnostics, split_sha256, shared_initialization_sha256
):
    global GNN_CHECKPOINT_CONTRACT_PATH
    feature_schema_path = Path(data.SCHEMA_PATH).resolve()
    dataset_manifest_path = Path(data.INDEX_PATH).resolve()
    feature_schema = json.loads(feature_schema_path.read_text(encoding='utf-8'))
    resolved_flags = {
        key: _jsonable(value)
        for key, value in sorted(vars(FLAGS).items())
    }
    model_init_flag_names = (
        'task', 'num_layers', 'D', 'target', 'gnn_type', 'encode_edge',
        'encode_edge_position', 'dropout', 'jkn_mode', 'jkn_enable',
        'gnn_layer_after_MLP', 'node_attention',
        'node_attention_MLP', 'separate_T', 'separate_P', 'separate_pseudo',
        'separate_icmp', 'P_use_all_nodes', 'gae_T', 'gae_P', 'input_encode',
        'decoder_type',
        'decompose_targets', 'target_mode', 'standardize_targets',
        'MLP_common_lyr', 'pragma_as_MLP', 'pragma_as_MLP_list',
        'pragma_scope', 'keep_pragma_attribute', 'pragma_order',
        'pragma_MLP_hidden_channels',
        'merge_MLP_hidden_channels', 'activation', 'resource_aux_weight',
        'multi_target_qor', 'graph_attention_heads', 'graph_residual_beta',
        'graph_layer_norm',
    )
    model_init_flags = {
        name: resolved_flags[name]
        for name in model_init_flag_names if name in resolved_flags
    }
    training_artifacts = snapshot_gnn_training_artifacts()
    contract = {
        'format': 'mailohls-gnn-checkpoint-v1',
        'provenance_status': 'captured_at_training',
        'git_commit': _git_commit(),
        'resolved_flags': resolved_flags,
        'model_init_flags': model_init_flags,
        'model_init': {
            'in_channels': int(num_features),
            'edge_dim': int(edge_dim),
            'targets': _jsonable(FLAGS.target),
            'resource_aux_heads': resource_stats is not None,
            'target_condition_dim': data.TARGET_CONDITION_DIM if FLAGS.multi_target_qor else 0,
        },
        'feature_schema': {
            'path': str(feature_schema_path),
            'sha256': _sha256_file(feature_schema_path),
            'version': feature_schema.get('feature_schema_version'),
            'node_feature_dim': feature_schema.get('node_feature_dim'),
            'edge_feature_dim': feature_schema.get('edge_feature_dim'),
        },
        'dataset_manifest': {
            'path': str(dataset_manifest_path),
            'sha256': _sha256_file(dataset_manifest_path),
        },
        'dataset_manifest_sha256': _sha256_file(dataset_manifest_path),
        'target_stats': _jsonable(target_stats),
        'resource_stats': _jsonable(resource_stats),
        'resource_diagnostics': _jsonable(resource_diagnostics),
        'split_sha256': split_sha256,
        'experiment_split': {
            'path': FLAGS.split_json,
            'sha256': FLAGS.experiment_split_sha256,
        },
        'effective_area_floor': float(FLAGS.effective_area_floor),
        'target_conditioning_policy': (
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
        'training_artifacts': training_artifacts,
    }
    path = Path(saver.model_logdir) / 'gnn_checkpoint_contract.json'
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    GNN_CHECKPOINT_CONTRACT_PATH = str(path)
    saver.log_info(f'Wrote captured GNN checkpoint contract -> {path}')
    return contract


def save_checkpoint_with_sidecar(payload, path, checkpoint_tag, epoch):
    torch.save(payload, path)
    if GNN_CHECKPOINT_CONTRACT_PATH is None:
        raise RuntimeError('GNN checkpoint contract was not initialized.')
    sidecar = {
        'format': 'mailohls-gnn-checkpoint-sidecar-v1',
        'provenance_status': 'captured_at_training',
        'checkpoint_sha256': _sha256_file(path),
        'contract_sha256': _sha256_file(GNN_CHECKPOINT_CONTRACT_PATH),
        'checkpoint_tag': checkpoint_tag,
        'checkpoint_epoch': int(epoch),
    }
    Path(f'{path}.json').write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


def assert_resource_contract_matches_state_dict(contract, state_dict):
    expected = bool(contract['model_init']['resource_aux_heads'])
    present = any(name.startswith('resource_heads.') for name in state_dict)
    if present != expected:
        raise RuntimeError(
            'GNN resource-head contract/state mismatch: '
            f'contract={expected}, state_dict={present}'
        )


def _as_int_seed(seed):
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    if isinstance(seed, str):
        return int(seed.strip())
    raise TypeError(f"random seed must be int-like, got {type(seed)}: {seed}")


def _name_set(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(',') if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def _sample_kernel(sample):
    """Read the kernel name from one unbatched PyG sample."""
    kernel = getattr(sample, 'kernel', None)
    if isinstance(kernel, (list, tuple)):
        if len(kernel) != 1:
            raise RuntimeError(f'Expected one kernel per sample, got {kernel!r}')
        kernel = kernel[0]
    if not isinstance(kernel, str) or not kernel:
        raise RuntimeError(f'Missing kernel identity on dataset sample: {kernel!r}')
    return kernel


def _sample_target_group(sample):
    group = getattr(sample, 'target_group', None)
    if isinstance(group, (list, tuple)):
        if len(group) != 1:
            raise RuntimeError(f'Expected one target group per sample, got {group!r}')
        group = group[0]
    return str(group) if group else _sample_kernel(sample)


def _model_target_name(target):
    """Tensor optimized by the regressor for one public QoR target."""
    if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta':
        return f'{target}_delta'
    return 'actual_perf' if FLAGS.encode_log and target == 'perf' else target


def _model_target(sample, target):
    return _get_y_with_target(sample, _model_target_name(target))


def _dataset_kernel_names(dataset):
    """Read identities only; callers must not pass the locked test split."""
    return {_sample_kernel(dataset[index]) for index in range(len(dataset))}


def fit_target_statistics(dataset):
    """Fit target moments using the same kernel weighting as the loss."""
    model_targets = FLAGS.target if isinstance(FLAGS.target, list) else [FLAGS.target]
    values = {target: [] for target in model_targets}
    kernels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        kernels.append(_sample_kernel(sample))
        for model_target in model_targets:
            data_target = _model_target_name(model_target)
            value = _model_target(sample, model_target).reshape(-1)
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise RuntimeError(
                    f'Invalid {data_target} target at training index '
                    f'{index}: {value}'
                )
            values[model_target].append(float(value.item()))

    counts = Counter(kernels)
    if bool(getattr(FLAGS, 'kernel_balanced_loss', False)):
        weights = np.asarray(
            [1.0 / counts[kernel] for kernel in kernels],
            dtype=np.float64,
        )
        weighting = 'kernel_balanced'
    else:
        weights = np.ones(len(kernels), dtype=np.float64)
        weighting = 'point_micro'
    weights /= weights.sum()

    stats = {}
    for target, target_values in values.items():
        array = np.asarray(target_values, dtype=np.float64)
        if array.size == 0:
            raise RuntimeError(f'Cannot fit target statistics for empty {target}.')
        mean = float(np.sum(weights * array))
        variance = float(np.sum(weights * np.square(array - mean)))
        stats[target] = {
            'mean': mean,
            'std': max(float(np.sqrt(variance)), 1e-8),
            'count': int(array.size),
            'weighting': weighting,
        }
    saver.log_info(f'Training-only log2 target statistics: {stats}')
    return stats


def fit_resource_statistics(dataset):
    """Fit log1p resource moments on training kernels only."""
    rows = []
    kernels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not hasattr(sample, 'resource_util'):
            raise RuntimeError(
                'Dataset lacks resource_util; regenerate with --force_regen.'
            )
        values = sample.resource_util.reshape(-1).detach().cpu().numpy()
        if values.shape != (len(RESOURCE_NAMES),):
            raise RuntimeError(
                f'Expected four resource targets, got shape {values.shape}.'
            )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise RuntimeError('Invalid resource utilization target.')
        rows.append(np.log1p(values.astype(np.float64)))
        kernels.append(_sample_kernel(sample))
    if not rows:
        raise RuntimeError('Cannot fit resource statistics on an empty dataset.')
    counts = Counter(kernels)
    weights = np.asarray(
        [
            1.0 / counts[kernel]
            if bool(getattr(FLAGS, 'kernel_balanced_loss', False))
            else 1.0
            for kernel in kernels
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    matrix = np.stack(rows, axis=0)
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(
        np.square(matrix - mean) * weights[:, None], axis=0
    )
    stats = {
        'names': list(RESOURCE_NAMES),
        'transform': 'log1p',
        'mean': mean.tolist(),
        'std': np.maximum(np.sqrt(variance), 1e-8).tolist(),
        'count': len(rows),
        'weighting': (
            'kernel_balanced'
            if bool(getattr(FLAGS, 'kernel_balanced_loss', False))
            else 'point_micro'
        ),
    }
    saver.log_info(f'Training-only resource statistics: {stats}')
    return stats


def maybe_fit_resource_statistics(dataset, resource_aux_weight):
    if float(resource_aux_weight) <= 0.0:
        return None
    return fit_resource_statistics(dataset)


def resource_training_diagnostics(dataset):
    """Describe raw resource supervision using training points only."""
    rows = []
    kernels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not hasattr(sample, 'resource_util'):
            raise RuntimeError(
                'Dataset lacks resource_util; regenerate with --force_regen.'
            )
        values = sample.resource_util.reshape(-1).detach().cpu().numpy()
        if values.shape != (len(RESOURCE_NAMES),):
            raise RuntimeError(f'Expected four resources, got {values.shape}.')
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise RuntimeError('Invalid resource utilization target.')
        rows.append(values.astype(np.float64))
        kernels.append(_sample_kernel(sample))
    if not rows:
        raise RuntimeError('Cannot diagnose resources on an empty training split.')

    matrix = np.stack(rows)
    diagnostics = {}
    for column, resource in enumerate(RESOURCE_NAMES):
        values = matrix[:, column]
        by_kernel = defaultdict(list)
        for kernel, value in zip(kernels, values):
            by_kernel[kernel].append(float(value))
        kernel_variances = {
            kernel: float(np.var(kernel_values))
            for kernel, kernel_values in by_kernel.items()
        }
        distinct_kernel_count = sum(
            len(set(kernel_values)) >= 2
            for kernel_values in by_kernel.values()
        )
        train_mean = float(np.mean(values))
        baseline_prediction = np.full_like(values, train_mean)
        # A deterministic constant predictor has no ordering information.
        baseline_tau = 0.0
        diagnostics[resource] = {
            'fraction_zero': float(np.mean(values == 0.0)),
            'median_utilization': float(np.median(values)),
            'p95_utilization': float(np.percentile(values, 95)),
            'within_kernel_variance_mean': float(
                np.mean(list(kernel_variances.values()))
            ),
            'within_kernel_variance_median': float(
                np.median(list(kernel_variances.values()))
            ),
            'kernels_with_two_or_more_distinct_values': int(
                distinct_kernel_count
            ),
            'kernel_count': int(len(by_kernel)),
            'train_mean_baseline': train_mean,
            'train_mean_baseline_mae': float(
                np.mean(np.abs(baseline_prediction - values))
            ),
            'train_mean_baseline_macro_kendall_tau': baseline_tau,
        }
    saver.log_info(
        'Training-only raw resource diagnostics: '
        + json.dumps(diagnostics, sort_keys=True)
    )
    return diagnostics


def require_paired_comparison_contract(
    comparison_contract_path,
    *, dataset_manifest_sha256, split_sha256,
    shared_initialization_sha256,
):
    """Reject an R1 comparison unless its R0 causal invariants match."""
    if not comparison_contract_path:
        return
    with open(comparison_contract_path, 'r', encoding='utf-8') as handle:
        control = json.load(handle)
    control_flags = control.get('resolved_flags', {})
    if float(control_flags.get('resource_aux_weight', -1.0)) != 0.0:
        raise RuntimeError('Paired control is not an R0 resource-disabled run.')
    if float(control_flags.get('rank_aux_weight', -1.0)) != 0.0:
        raise RuntimeError('Paired control did not keep rank auxiliary loss off.')
    expected = {
        'dataset_manifest_sha256': dataset_manifest_sha256,
        'split_sha256': split_sha256,
        'shared_initialization_sha256': shared_initialization_sha256,
    }
    mismatches = {
        name: {'control': control.get(name), 'current': value}
        for name, value in expected.items()
        if control.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            'Invalid paired R0/R1 comparison: ' + json.dumps(
                mismatches, sort_keys=True
            )
        )


def _point_loss_from_error(error):
    error = np.asarray(error, dtype=np.float64)
    mode = str(FLAGS.loss).lower()
    if mode == 'mse':
        return np.square(error)
    if mode == 'rmse':
        # The square root is applied after the weighted mean below.
        return np.square(error)
    if mode == 'smooth_l1':
        beta = float(FLAGS.smooth_l1_beta)
        absolute = np.abs(error)
        return np.where(
            absolute < beta,
            0.5 * np.square(error) / beta,
            absolute - 0.5 * beta,
        )
    raise RuntimeError(f'Unknown loss {FLAGS.loss!r}')


def deterministic_validation_baseline(dataset, target_stats):
    """Evaluate the no-learning predictor under the configured loss policy.

    Absolute/Stage-B training uses the training mean. Reference-delta training
    uses zero delta, i.e. the measured neutral synthesis point for that kernel.
    """
    kernels = [_sample_kernel(dataset[index]) for index in range(len(dataset))]
    counts = Counter(kernels)
    scale = len(kernels) / len(counts) if counts else 1.0
    breakdown = {}
    for model_target, stats in target_stats.items():
        squared_errors = []
        weights = []
        for index, kernel in enumerate(kernels):
            sample = dataset[index]
            actual = float(_model_target(sample, model_target).reshape(-1)[0])
            prediction = (
                0.0
                if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta'
                else stats['mean']
            )
            error = actual - prediction
            if bool(getattr(FLAGS, 'standardize_targets', False)):
                error /= stats['std']
            squared_errors.append(float(_point_loss_from_error(error)))
            weights.append(
                scale / counts[kernel]
                if bool(getattr(FLAGS, 'kernel_balanced_loss', False))
                else 1.0
            )
        value = float(np.mean(np.asarray(squared_errors) * np.asarray(weights)))
        breakdown[model_target] = (
            np.sqrt(value) if str(FLAGS.loss).lower() == 'rmse' else value
        )
    return float(sum(breakdown.values())), breakdown


def compute_validation_selection_score(loss_breakdown, baseline_breakdown):
    """Return a conservative, target-balanced validation score.

    Each target loss is divided by its deterministic no-learning baseline on
    the same validation kernels (training mean for legacy modes; measured
    neutral reference for reference-delta mode). The maximum ratio is
    minimized, so one target cannot hide regression in another.

    A score below one means every target beats its declared no-learning baseline.
    """
    if not isinstance(loss_breakdown, dict) or not isinstance(
        baseline_breakdown, dict
    ):
        raise TypeError(
            "Validation and baseline breakdowns must be dictionaries."
        )

    expected_targets = sorted(baseline_breakdown)
    if sorted(loss_breakdown) != expected_targets or not expected_targets:
        raise RuntimeError(
            "Validation/baseline target mismatch: "
            f"validation={sorted(loss_breakdown)}, "
            f"baseline={expected_targets}"
        )

    ratios = {}
    for target in expected_targets:
        loss = float(loss_breakdown[target])
        baseline = float(baseline_breakdown[target])
        if not np.isfinite(loss) or loss < 0.0:
            raise RuntimeError(
                f"Invalid validation loss for {target}: {loss!r}"
            )
        if not np.isfinite(baseline) or baseline <= 1e-12:
            raise RuntimeError(
                f"Invalid no-learning baseline loss for {target}: {baseline!r}"
            )
        ratios[target] = loss / baseline

    return max(ratios.values()), ratios


def should_update_qualified_rank(
    target_ratios,
    per_kernel_target_ratios,
    ranking_score,
    best_score,
    min_delta,
    min_rank_tau,
    max_kernel_zero_baseline_ratio,
):
    """Gate rank selection behind target- and kernel-level qualification."""
    ratios = [float(value) for value in target_ratios.values()]
    kernel_ratios = [
        float(value) for value in per_kernel_target_ratios.values()
    ]
    ranking_score = float(ranking_score)
    if not ratios or not all(np.isfinite(value) for value in ratios):
        raise RuntimeError(f'Invalid target ratios: {target_ratios}')
    if not kernel_ratios or not all(
        np.isfinite(value) for value in kernel_ratios
    ):
        raise RuntimeError(
            f'Invalid per-kernel target ratios: {per_kernel_target_ratios}'
        )
    if not np.isfinite(ranking_score):
        raise RuntimeError(f'Invalid ranking score: {ranking_score}')
    qualified = (
        all(value < 1.0 for value in ratios)
        and ranking_score >= float(min_rank_tau)
        and max(kernel_ratios) <= float(max_kernel_zero_baseline_ratio)
    )
    improved = ranking_score > float(best_score) + float(min_delta)
    return qualified and improved


def should_update_embedding_rank(
    target_ratios,
    ranking_score,
    best_score,
    min_delta,
    min_rank_tau,
):
    """Select a Stage-2-oriented rank checkpoint with aggregate QoR guardrails."""
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
    """Compare each validation kernel/target loss with its no-learning loss."""
    ratios = {}
    for target, values in points_dict.items():
        if target not in target_stats:
            raise RuntimeError(f'Missing target statistics for {target!r}.')
        kernels = values['kernel']
        predictions = values['pred']
        if not kernels or len(kernels) != len(predictions):
            raise RuntimeError(f'Incomplete validation points for {target!r}.')

        baseline_prediction = (
            0.0
            if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta'
            else float(target_stats[target]['mean'])
        )
        standardizer = (
            float(target_stats[target]['std'])
            if bool(getattr(FLAGS, 'standardize_targets', False))
            else 1.0
        )
        for kernel in sorted(set(kernels)):
            indices = [
                index for index, value in enumerate(kernels)
                if value == kernel
            ]
            actual = np.asarray(
                [predictions[index][0] for index in indices], dtype=np.float64
            )
            predicted = np.asarray(
                [predictions[index][1] for index in indices], dtype=np.float64
            )
            model_loss = float(np.mean(_point_loss_from_error(
                (actual - predicted) / standardizer
            )))
            baseline_loss = float(np.mean(_point_loss_from_error(
                (actual - baseline_prediction) / standardizer
            )))
            if str(FLAGS.loss).lower() == 'rmse':
                model_loss = float(np.sqrt(model_loss))
                baseline_loss = float(np.sqrt(baseline_loss))
            if not np.isfinite(baseline_loss) or baseline_loss <= 1e-12:
                raise RuntimeError(
                    'Invalid per-kernel no-learning baseline for '
                    f'{kernel}/{target}: {baseline_loss!r}'
                )
            ratio = model_loss / baseline_loss
            if not np.isfinite(ratio) or ratio < 0.0:
                raise RuntimeError(
                    f'Invalid per-kernel target ratio for {kernel}/{target}: '
                    f'{ratio!r}'
                )
            ratios[f'{kernel}/{target}'] = ratio
    return ratios


class KernelCenteredDataset(Dataset):
    """Attach training-kernel centers used only by auxiliary losses."""

    def __init__(self, dataset, targets):
        self.dataset = dataset
        values = {
            target: defaultdict(list)
            for target in targets
        }

        for index in range(len(dataset)):
            sample = dataset[index]
            kernel = _sample_kernel(sample)
            for target in targets:
                data_target = (
                    'actual_perf'
                    if FLAGS.encode_log and target == 'perf'
                    else target
                )
                value = float(
                    _get_y_with_target(sample, data_target).reshape(-1)[0]
                )
                if not np.isfinite(value):
                    raise RuntimeError(
                        f'Non-finite {data_target} target for {kernel}.'
                    )
                values[target][kernel].append(value)

        self.centers = {
            target: {
                kernel: float(np.mean(kernel_values))
                for kernel, kernel_values in target_values.items()
            }
            for target, target_values in values.items()
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # MyOwnDataset constructs a fresh Data wrapper for every access while
        # reusing the large static tensors, so adding scalar metadata is safe
        # and avoids cloning an entire graph for every design point.
        sample = self.dataset[index]
        kernel = _sample_kernel(sample)
        for target, centers in self.centers.items():
            setattr(
                sample,
                f"{target}_center",
                torch.tensor([centers[kernel]], dtype=torch.float32),
            )
        return sample


class KernelBalancedDataset(Dataset):
    """Attach weights whose total is equal for every kernel in one split."""

    def __init__(self, dataset, *, unit_loss_weights=False):
        self.dataset = dataset
        self.kernels = [
            _sample_kernel(dataset[index]) for index in range(len(dataset))
        ]
        counts = Counter(self.kernels)
        self.kernel_count = len(counts)
        if not counts:
            self.weights = []
            self.sampling_weights = []
            return
        scale = len(self.kernels) / len(counts)
        self.sampling_weights = [
            1.0 / counts[kernel] for kernel in self.kernels
        ]
        self.weights = (
            [1.0] * len(self.kernels)
            if unit_loss_weights
            else [scale / counts[kernel] for kernel in self.kernels]
        )
        saver.log_info(
            'Kernel-balanced split: '
            f'{len(counts)} kernels, {len(self.kernels)} points, '
            f'points/kernel={min(counts.values())}..{max(counts.values())}'
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample.kernel_loss_weight = torch.tensor(
            [self.weights[index]], dtype=torch.float32
        )
        return sample


class KernelGroupedBatchSampler(Sampler):
    """Yield balanced batches of kernels with multiple points per kernel."""

    def __init__(
        self,
        dataset,
        *,
        kernels_per_batch,
        points_per_kernel,
        samples_per_kernel_per_epoch,
        seed,
    ):
        self.dataset = dataset
        self.kernels_per_batch = int(kernels_per_batch)
        self.points_per_kernel = int(points_per_kernel)
        self.seed = _as_int_seed(seed)
        self.epoch = 0
        by_kernel = defaultdict(list)
        by_kernel_target = defaultdict(lambda: defaultdict(list))
        for index in range(len(dataset)):
            sample = dataset[index]
            kernel = _sample_kernel(sample)
            by_kernel[kernel].append(index)
            by_kernel_target[kernel][_sample_target_group(sample)].append(index)
        self.indices_by_kernel = {
            kernel: tuple(indices) for kernel, indices in sorted(by_kernel.items())
        }
        self.indices_by_kernel_target = {
            kernel: {
                target: tuple(indices)
                for target, indices in sorted(targets.items())
            }
            for kernel, targets in sorted(by_kernel_target.items())
        }
        self.kernels = tuple(self.indices_by_kernel)
        if len(self.kernels) < self.kernels_per_batch:
            raise ValueError(
                'Kernel-grouped sampling requires at least '
                f'{self.kernels_per_batch} kernels, found {len(self.kernels)}.'
            )
        if samples_per_kernel_per_epoch is None:
            samples_per_kernel_per_epoch = max(
                1, int(np.ceil(len(dataset) / len(self.kernels)))
            )
        self.samples_per_kernel_per_epoch = int(samples_per_kernel_per_epoch)
        if self.samples_per_kernel_per_epoch <= 0:
            raise ValueError('samples_per_kernel_per_epoch must be positive.')
        draws_per_batch = self.kernels_per_batch * self.points_per_kernel
        target_draws = len(self.kernels) * self.samples_per_kernel_per_epoch
        self.num_batches = max(1, int(np.ceil(target_draws / draws_per_batch)))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        kernel_order = list(rng.permutation(self.kernels))
        cursor = 0
        for _ in range(self.num_batches):
            selected_kernels = [
                kernel_order[(cursor + offset) % len(kernel_order)]
                for offset in range(self.kernels_per_batch)
            ]
            cursor = (cursor + self.kernels_per_batch) % len(kernel_order)
            batch = []
            for kernel in selected_kernels:
                groups = self.indices_by_kernel_target[str(kernel)]
                group_names = tuple(groups)
                target = group_names[int(rng.integers(len(group_names)))]
                indices = groups[target]
                chosen = rng.choice(
                    indices,
                    size=self.points_per_kernel,
                    replace=len(indices) < self.points_per_kernel,
                )
                batch.extend(int(index) for index in chosen)
            yield batch


def gen_dataset(li):
    if FLAGS.tiny_overfit:
        train_workers = FLAGS.tiny_overfit_workers
        eval_workers = FLAGS.tiny_overfit_workers
    else:
        train_workers = FLAGS.num_workers
        eval_workers = FLAGS.eval_num_workers

    def make_loader(ds, shuffle, workers, *, training=False):
        sampler = None
        batch_sampler = None
        if training and bool(getattr(FLAGS, 'kernel_grouped_sampling', False)):
            batch_sampler = KernelGroupedBatchSampler(
                ds,
                kernels_per_batch=FLAGS.kernels_per_batch,
                points_per_kernel=FLAGS.points_per_kernel,
                samples_per_kernel_per_epoch=FLAGS.samples_per_kernel_per_epoch,
                seed=FLAGS.random_seed,
            )
            saver.log_info(
                'Kernel-grouped sampling: '
                f'{FLAGS.kernels_per_batch} kernels x '
                f'{FLAGS.points_per_kernel} points, '
                f'{len(batch_sampler)} batches/epoch.'
            )
        if training and bool(
            getattr(FLAGS, 'kernel_uniform_sampling', False)
        ):
            if not isinstance(ds, KernelBalancedDataset):
                raise RuntimeError(
                    'Uniform kernel sampling requires KernelBalancedDataset.'
                )
            sampler = WeightedRandomSampler(
                torch.as_tensor(ds.sampling_weights, dtype=torch.double),
                num_samples=(
                    len(ds)
                    if FLAGS.samples_per_kernel_per_epoch is None
                    else ds.kernel_count * FLAGS.samples_per_kernel_per_epoch
                ),
                replacement=True,
                generator=torch.Generator().manual_seed(
                    _as_int_seed(FLAGS.random_seed)
                ),
            )
        kwargs = dict(num_workers=workers, pin_memory=False)
        if batch_sampler is not None:
            kwargs['batch_sampler'] = batch_sampler
        else:
            kwargs.update(
                batch_size=FLAGS.batch_size,
                shuffle=shuffle and sampler is None,
                sampler=sampler,
            )
        if shuffle and sampler is None and batch_sampler is None:
            kwargs["generator"] = torch.Generator().manual_seed(
                _as_int_seed(FLAGS.random_seed)
            )
        if workers > 0:
            kwargs["prefetch_factor"] = FLAGS.prefetch_factor
            kwargs["persistent_workers"] = FLAGS.persistent_workers
        return DataLoader(ds, **kwargs)

    train_loader = make_loader(
        li[0], shuffle=True, workers=train_workers, training=True
    )
    val_loader   = make_loader(li[1], shuffle=False, workers=eval_workers)
    test_loader  = make_loader(li[2], shuffle=False, workers=eval_workers)

    num_features = train_loader.dataset[0].num_features
    saver.info(f'num features for training: {num_features}')
    edge_dim = train_loader.dataset[0].edge_attr.shape[1]
    saver.info(f'size of the edge attribute is {edge_dim}')
    if FLAGS.num_features is not None and int(FLAGS.num_features) != int(num_features):
        raise ValueError(
            f'--num_features={FLAGS.num_features} does not match runtime {num_features}.'
        )
    if FLAGS.edge_dim is not None and int(FLAGS.edge_dim) != int(edge_dim):
        raise ValueError(
            f'--edge_dim={FLAGS.edge_dim} does not match runtime {edge_dim}.'
        )
    saver.save_resolved_runtime_dimensions(num_features, edge_dim)

    return train_loader, val_loader, test_loader, num_features, edge_dim


def _take_random_fraction(dataset, frac, seed, *, as_subset=False):
    n = len(dataset)
    k = max(1, int(round(n * frac)))
    seed = _as_int_seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False).tolist()

    if as_subset:
        from torch.utils.data import Subset
        return Subset(dataset, indices)

    # Build the sampled file list from the dataset
    if hasattr(dataset, 'data_files'):
        files_all = dataset.data_files
    else:
        files_all = dataset.processed_file_names  # uses glob on SAVE_DIR

    sampled_files = [files_all[i] for i in indices]

    # Recreate a MyOwnDataset that only sees the sampled files
    return MyOwnDataset(
        transform=getattr(dataset, 'transform', None),
        pre_transform=getattr(dataset, 'pre_transform', None),
        data_files=sampled_files,
    )


def _take_random_n(dataset, n, seed):
    n_total = len(dataset)
    k = min(int(n), n_total)
    seed = _as_int_seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=k, replace=False).tolist()
    return Subset(dataset, indices)


def _filter_dataset_by_kernel(dataset, kernel_name):
    """
    Robust tiny-overfit filtering for the new compact dataset.

    Works even if dataset.processed_file_names / records are not in the
    exact format we expect, because it filters by actual dataset samples.
    """
    selected_indices = []

    # Fast path: if compact records are present as dicts, use them directly
    records = getattr(dataset, 'records', None)
    if records is not None and len(records) > 0 and isinstance(records[0], dict):
        for i, rec in enumerate(records):
            gname = rec['graph_name']
            if gname.startswith(kernel_name):
                selected_indices.append(i)
    else:
        # Fallback path: inspect actual samples
        for i in range(len(dataset)):
            d = dataset[i]
            k = d.kernel[0] if isinstance(d.kernel, (list, tuple)) else d.kernel
            if k == kernel_name:
                selected_indices.append(i)

    if len(selected_indices) == 0:
        raise RuntimeError(f'No records found for kernel={kernel_name}')

    return Subset(dataset, selected_indices)


def _exclude_development_kernels(dataset, value):
    """Remove kernels that lack authenticated Stage-C reference measurements."""
    excluded = _name_set(value)
    if not excluded:
        return dataset

    overlap = excluded & (
        _name_set(FLAGS.val_kernels) | _name_set(FLAGS.test_kernels)
    )
    if overlap:
        raise RuntimeError(
            'Development exclusions overlap validation/test kernels: '
            + ', '.join(sorted(overlap))
        )

    records = getattr(dataset, 'records', None)
    if not isinstance(records, list) or (
        records and not isinstance(records[0], dict)
    ):
        raise RuntimeError(
            '--development_exclude_kernels requires the compact MLIR dataset.'
        )
    available = {
        record.get('kernel_name')
        for record in records
        if record.get('kernel_name')
    }
    unknown = excluded - available
    if unknown:
        raise RuntimeError(
            'Unknown development-excluded kernels: '
            + ', '.join(sorted(unknown))
        )
    retained = [
        record for record in records
        if record.get('kernel_name') not in excluded
    ]
    if not retained:
        raise RuntimeError('Development exclusions removed the entire dataset.')
    saver.log_info(
        'Development kernels excluded before splitting: '
        + ', '.join(sorted(excluded))
    )
    return MyOwnDataset(data_files=retained)


def process_split_data(dataset):
    dataset_dict = defaultdict(list)

    if FLAGS.tiny_overfit:
        kernel_ds = _filter_dataset_by_kernel(dataset, FLAGS.tiny_overfit_kernel)
        tiny_ds = _take_random_n(kernel_ds, FLAGS.tiny_overfit_num_samples, FLAGS.random_seed)

        saver.log_info(
            f"[tiny_overfit] kernel={FLAGS.tiny_overfit_kernel} "
            f"num_samples={len(tiny_ds)}"
        )

        dataset_dict['train'] = tiny_ds
        dataset_dict['val'] = tiny_ds
        dataset_dict['test'] = tiny_ds
        return dataset_dict

    dataset = _exclude_development_kernels(
        dataset,
        getattr(FLAGS, 'development_exclude_kernels', None),
    )

    # Full training: use the whole compact dataset directly
    dataset_dict['train'] = dataset
    dataset_dict['val'] = None
    dataset_dict['test'] = None

    if not FLAGS.all_kernels:
        dataset_dict['train'] = get_kernel_samples(dataset)
    elif FLAGS.test_kernels is not None or FLAGS.val_kernels is not None:
        dataset_dict = split_train_val_test_kernel(dataset)

    return dataset_dict


def get_train_val_count(num_graphs, val_ratio, test_ratio):
    if FLAGS.test_kernels is not None:
        r1 = int(num_graphs * (1.0 - val_ratio))
        r2 = int(num_graphs * (val_ratio))
    else:
        r1 = int(num_graphs * (1.0 - val_ratio - test_ratio))
        r2 = int(num_graphs * (val_ratio))

    return r1, r2


def check_feature_extract(model, key_word, gnn_layer=None):
    '''"
        checks that all parameters except for the ones that have "key_word" are fixed
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                assert param.requires_grad == False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer-1)]):
                    assert param.requires_grad == False


def model_update(model, losses_list, loss, epoch, plot_test, tag):
    '''
        Tracks the current loss, saves the best model so far and optionally triggers plotting
    '''
    saver.writer.add_scalar(f'{tag}/{tag}', loss, epoch)
    if not losses_list or loss < min(losses_list):
        if FLAGS.save_model:
            saver.log_info((f'Saved {tag} model at epoch {epoch}'))
            save_checkpoint_with_sidecar(
                model.state_dict(),
                join(saver.model_logdir, f"{tag}_model_state_dict.pth"),
                tag,
                epoch,
            )
        plot_test = True
    losses_list.append(loss)

    return plot_test


def log_loss(loss_dict, gae_loss, tag):
    saver.log_info((f'{tag} GAE loss: {gae_loss}'))
    saver.log_info((f'{tag} loss breakdown {loss_dict}'))


def set_target_list():
    '''
        Creates loss_dict per target (zero initialized)
    '''
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task =='regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]

    loss_dict = {}
    for t in target_list:
        loss_dict[t] = 0.0

    return target_list, loss_dict


def update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct):
    '''
        Accumulates training metrics per batch and updates the loss_dict for each target
    '''
    if FLAGS.task == 'regression':
        # MSELoss is a batch mean. Weight by the number of graphs so the final
        # epoch/validation loss is independent of the smaller last batch.
        batch_weight = int(data.num_graphs)
        total_loss += loss.item() * batch_weight
        for t in target_list:
            model_key = 'perf' if FLAGS.encode_log and t == 'actual_perf' else t
            loss_dict[t] += loss_dict_[model_key].item() * batch_weight
        for key, value in loss_dict_.items():
            if key.startswith('rank_aux/'):
                loss_dict.setdefault(key, 0.0)
                loss_dict[key] += value.item() * batch_weight
        return loss_dict, total_loss
    else:
        loss_, pred = torch.max(out_dict[FLAGS.target[0]], 1)
        labels = _get_y_with_target(data, FLAGS.target[0])
        correct += (pred == labels).sum().item()
        total_loss += labels.size(0)
        return pred, correct, total_loss


def inference_loss_function(pred, true):
    return (pred - true) ** 2


def _inverse_log2_target(value):
    """Convert a model output back to the positive physical QoR domain."""
    if str(FLAGS.norm_method) != 'log2':
        raise RuntimeError(
            "Physical-unit evaluation currently requires --norm_method log2."
        )
    return max(0.0, float(np.exp2(value) - float(FLAGS.epsilon)))


def _kernel_at(data, index):
    kernels = getattr(data, 'kernel', None)
    if isinstance(kernels, (list, tuple)):
        return str(kernels[index])
    if isinstance(kernels, str) and int(data.num_graphs) == 1:
        return kernels
    raise RuntimeError("Could not align a prediction with its kernel name.")


def _string_attribute_at(data, name, index):
    values = getattr(data, name, None)
    if isinstance(values, (list, tuple)):
        return str(values[index])
    if isinstance(values, str) and int(data.num_graphs) == 1:
        return values
    raise RuntimeError(
        f"Could not align string attribute {name!r} with prediction {index}."
    )


def _parse_resource_eval_budgets(spec):
    budgets = []
    for entry in str(spec).split(';'):
        if not entry.strip():
            continue
        values = [float(value) for value in entry.split(',')]
        if len(values) == 1:
            values *= len(RESOURCE_NAMES)
        if len(values) != len(RESOURCE_NAMES) or any(
            value < 0.0 for value in values
        ):
            raise ValueError(
                'Resource budgets must be non-negative scalar or '
                'BRAM,DSP,FF,LUT vectors.'
            )
        budgets.append(np.asarray(values, dtype=np.float64))
    if not budgets:
        raise ValueError('--resource_eval_budgets produced no budgets.')
    return budgets


def resource_feasibility_metrics(actual, predicted, budget, tolerance):
    """Evaluate joint feasibility and its signed max-constraint boundary."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(1, -1)
    actual_feasible = np.all(actual <= budget, axis=1)
    predicted_feasible = np.all(predicted <= budget, axis=1)
    false_feasible = predicted_feasible & ~actual_feasible
    joint_margin = np.max(actual - budget, axis=1)
    boundary = np.abs(joint_margin) <= float(tolerance)
    return {
        'accuracy': float(np.mean(predicted_feasible == actual_feasible)),
        'false_feasible_fdr': float(
            false_feasible.sum() / max(int(predicted_feasible.sum()), 1)
        ),
        'false_feasible_rate': float(false_feasible.mean()),
        'boundary_accuracy': (
            float(np.mean(
                predicted_feasible[boundary] == actual_feasible[boundary]
            )) if boundary.any() else float('nan')
        ),
        'boundary_samples': int(boundary.sum()),
        'joint_margin': joint_margin,
        'boundary_mask': boundary,
    }


def report_resource_metrics(resource_rows, label):
    """Report stable physical and feasibility metrics for resource heads."""
    if not resource_rows:
        return None
    actual = np.stack([row['actual'] for row in resource_rows])
    predicted = np.stack([row['predicted'] for row in resource_rows])
    kernels = [row['kernel'] for row in resource_rows]
    rows = []
    for index, resource in enumerate(RESOURCE_NAMES):
        kernel_taus = []
        for kernel in sorted(set(kernels)):
            selected = [i for i, value in enumerate(kernels) if value == kernel]
            tau = kendalltau(
                actual[selected, index], predicted[selected, index]
            )[0]
            kernel_taus.append(0.0 if not np.isfinite(tau) else float(tau))
        rows.append({
            'resource': resource,
            'mae_percentage_points': float(
                np.mean(np.abs(predicted[:, index] - actual[:, index])) * 100.0
            ),
            'bias_percentage_points': float(
                np.mean(predicted[:, index] - actual[:, index]) * 100.0
            ),
            'kernel_macro_tau': float(np.mean(kernel_taus)),
        })

    feasibility = []
    tolerance = float(FLAGS.resource_boundary_tolerance)
    for budget in _parse_resource_eval_budgets(FLAGS.resource_eval_budgets):
        metrics = resource_feasibility_metrics(
            actual, predicted, budget, tolerance
        )
        feasibility.append({
            'budget': budget.tolist(),
            **{
                key: value for key, value in metrics.items()
                if key not in {'joint_margin', 'boundary_mask'}
            },
        })
    report = {'resources': rows, 'feasibility': feasibility}
    saver.log_info(f'{label} resource metrics: {report}')
    return report


def update_csv_dict(csv_dict, data, i, target_name, target_value, out_value):
    '''
        Collects per graph "actual VS predicted" rows into a dict we can later write into a CSV
    '''
    if csv_dict is not None:
        gname = _get_y_with_target(data, 'gname')[i]
        pragma = _get_y_with_target(data, 'pragmas')[i][0].item()
        pragma = '-'.join([str(j.item()) for j in _get_y_with_target(data, 'pragmas')[i]])
        if True or 'blocked' in gname:
            if f'{gname}-{pragma}' not in csv_dict:
                csv_dict[f'{gname}-{pragma}'] = {'gname': gname, 'pragma': pragma}
            csv_dict[f'{gname}-{pragma}'][f'acutal-{target_name}'] = target_value
            csv_dict[f'{gname}-{pragma}'][f'predicted-{target_name}'] = out_value
            l = csv_dict['header']
            if f'acutal-{target_name}' not in l:
                l.extend([f'acutal-{target_name}', f'predicted-{target_name}'])
                csv_dict['header'] = l


def _print_sanity_rows(csv_dict, n=10):
    if csv_dict is None or n <= 0:
        return

    rows = []
    for k, v in csv_dict.items():
        if k == 'header':
            continue

        rows.append({
            'gname': v.get('gname'),
            'pragma': v.get('pragma'),
            'actual_perf': v.get('acutal-perf'),
            'pred_perf': v.get('predicted-perf'),
            'actual_area': v.get('acutal-area'),
            'pred_area': v.get('predicted-area'),
        })

    if not rows:
        saver.log_info('[sanity] no rows collected')
        return

    df = pd.DataFrame(rows).head(n).copy()

    saver.log_info('[sanity] first predictions vs targets:')
    saver.log_info(df.round(4).to_string(index=False))

    pred_cols = ['pred_perf', 'pred_area']
    has_nan = df[pred_cols].isna().any().any()
    saver.log_info(f'[sanity] NaNs present? {has_nan}')
    saver.log_info(
        f"[sanity] unique pred_perf={df['pred_perf'].nunique()} | "
        f"unique pred_area={df['pred_area'].nunique()}"
    )


def train_main(dataset, pragma_dim = None, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio, resample=-1):
    saver.info(f'Reading dataset from {SAVE_DIR}')

    dataset_dict = process_split_data(dataset)
    if bool(getattr(FLAGS, 'final_refit', False)):
        development_records = list(dataset_dict['train'].records)
        if dataset_dict.get('val') is not None:
            development_records.extend(dataset_dict['val'].records)
        dataset_dict['train'] = MyOwnDataset(data_files=development_records)
        dataset_dict['val'] = None
        saver.log_info(
            'Final refit: merged train and validation kernels; '
            f'{len(development_records)} development points. '
            'Configured test kernels remain excluded.'
        )
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)

    if FLAGS.tiny_overfit:
        tiny_ds = dataset_dict['train']
        li = [tiny_ds, tiny_ds, tiny_ds]
        saver.log_info(
            f"[tiny_overfit] using identical subset for train/val/test "
            f"(n={len(tiny_ds)})"
        )
    elif bool(getattr(FLAGS, 'final_refit', False)):
        li = [
            dataset_dict['train'],
            MyOwnDataset(data_files=[]),
            dataset_dict['test'] or MyOwnDataset(data_files=[]),
        ]
    else:
        if dataset_dict.get('val') is not None:
            li = [
                dataset_dict['train'],
                dataset_dict['val'],
                dataset_dict['test'] or MyOwnDataset(data_files=[]),
            ]
        elif resample == -1:
            li = split_dataset(dataset_dict['train'], r1, r2, dataset_test=dataset_dict['test'])
        else:
            li = split_dataset_resample(
                dataset_dict['train'],
                1.0 - val_ratio - test_ratio,
                val_ratio,
                test_ratio,
                test_id=resample
            )

    split_sha256 = _split_sha256(li)

    model_targets = (
        FLAGS.target if isinstance(FLAGS.target, list) else [FLAGS.target]
    )
    if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta':
        # Only development identities are inspected here. A successful row for
        # a configured test kernel is rejected until --evaluate_test is given.
        required_kernels = _dataset_kernel_names(li[0])
        required_kernels.update(_dataset_kernel_names(li[1]))
        evaluate_test = bool(getattr(FLAGS, 'evaluate_test', False))
        forbidden_kernels = (
            () if evaluate_test else (FLAGS.test_kernels or '')
        )
        if evaluate_test:
            required_kernels.update(_dataset_kernel_names(li[2]))
        references = load_reference_baselines(
            FLAGS.baseline_manifest,
            required_kernels=required_kernels,
            forbidden_kernels=forbidden_kernels,
            expected_device=FLAGS.target_device,
            expected_clock_period_ns=FLAGS.clock_period_ns,
            expected_toolchain_version=FLAGS.vitis_hls_version,
            epsilon=FLAGS.epsilon,
        )
        li[0] = ReferenceDeltaDataset(li[0], references, model_targets)
        li[1] = ReferenceDeltaDataset(li[1], references, model_targets)
        if evaluate_test:
            li[2] = ReferenceDeltaDataset(li[2], references, model_targets)
        saver.log_info(
            f'Reference-delta mode: loaded {len(references)} neutral '
            'development baselines; held-out test rows are '
            + ('unlocked.' if evaluate_test else 'forbidden.')
        )

    target_stats = fit_target_statistics(li[0])
    resource_diagnostics = resource_training_diagnostics(li[0])
    resource_stats = maybe_fit_resource_statistics(
        li[0], FLAGS.resource_aux_weight
    )
    if FLAGS.decompose_targets:
        li[0] = KernelCenteredDataset(li[0], model_targets)
    baseline_total = None
    baseline_breakdown = None
    if len(li[1]) > 0:
        baseline_total, baseline_breakdown = deterministic_validation_baseline(
            li[1], target_stats
        )
        saver.log_info(
            'Deterministic validation baseline under the configured '
            f'objective: total={baseline_total:.4f}, '
            f'breakdown={baseline_breakdown}'
        )
    if bool(getattr(FLAGS, 'kernel_balanced_loss', False)):
        uniform_sampling = bool(
            getattr(FLAGS, 'kernel_uniform_sampling', False)
            or getattr(FLAGS, 'kernel_grouped_sampling', False)
        )
        li[0] = KernelBalancedDataset(
            li[0], unit_loss_weights=uniform_sampling
        )
        li[1] = KernelBalancedDataset(li[1])
        # Constructing the wrapper enumerates its samples. Do not even do that
        # for the declared test split before the one-shot evaluation is armed.
        if bool(getattr(FLAGS, 'evaluate_test', False)):
            li[2] = KernelBalancedDataset(li[2])

    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    set_reproducible_seed(
        FLAGS.random_seed, FLAGS.allow_nondeterministic
    )
    model = Net(
        num_features,
        edge_dim=edge_dim,
        init_pragma_dict=pragma_dim,
        target_stats=target_stats,
        resource_stats=resource_stats,
    ).to(FLAGS.device)
    shared_initialization_sha256 = hash_state_dict(
        model.state_dict(),
        exclude_prefixes=('resource_heads.',),
        exclude_names=('resource_mean', 'resource_std'),
    )
    # Optional heads may consume initialization RNG; realign later dropout.
    set_reproducible_seed(
        FLAGS.random_seed, FLAGS.allow_nondeterministic
    )
    dataset_manifest_sha256 = _sha256_file(Path(data.INDEX_PATH).resolve())
    require_paired_comparison_contract(
        FLAGS.paired_control_contract,
        dataset_manifest_sha256=dataset_manifest_sha256,
        split_sha256=split_sha256,
        shared_initialization_sha256=shared_initialization_sha256,
    )
    checkpoint_contract = write_gnn_checkpoint_contract(
        num_features=num_features,
        edge_dim=edge_dim,
        target_stats=target_stats,
        resource_stats=resource_stats,
        resource_diagnostics=resource_diagnostics,
        split_sha256=split_sha256,
        shared_initialization_sha256=shared_initialization_sha256,
    )
    assert_resource_contract_matches_state_dict(
        checkpoint_contract, model.state_dict()
    )
    saver.log_info(f"Model first param device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("CPU training")

    if FLAGS.load_pretrained and FLAGS.model_path is not None:
        model_path = FLAGS.model_path[0] if isinstance(FLAGS.model_path, list) else FLAGS.model_path
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        )

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)

    saver.log_model_architecture(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    updates_per_epoch = (len(train_loader) + FLAGS.grad_accum_steps - 1) // FLAGS.grad_accum_steps
    num_steps = updates_per_epoch * FLAGS.epoch_num
    warmup_steps = min(
        max(0, int(FLAGS.warmup_epochs)) * updates_per_epoch,
        max(0, num_steps - 1),
    )
    saver.log_info(
        f'Optimization schedule: steps={num_steps}, '
        f'updates_per_epoch={updates_per_epoch}, '
        f'grad_accum_steps={FLAGS.grad_accum_steps}, '
        f'warmup_steps={warmup_steps} '
        f'({FLAGS.warmup_epochs} fixed epochs)'
    )

    if FLAGS.scheduler is None:
        lr_scheduler = None
    elif FLAGS.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[FLAGS.epoch_num // 3], gamma=0.1)
    elif FLAGS.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    elif FLAGS.scheduler == "plateau":
        if len(val_loader) == 0:
            raise RuntimeError(
                'The plateau scheduler requires a validation split.'
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(FLAGS.plateau_factor),
            patience=int(FLAGS.plateau_patience),
            threshold=float(FLAGS.early_stopping_min_delta),
            min_lr=1e-5,
        )
    else:
        raise ValueError(f'Unknown scheduler: {FLAGS.scheduler!r}')

    if FLAGS.scheduler == 'plateau':
        # Plateau scheduling is epoch/validation driven. Its short linear
        # warmup is applied explicitly at the start of each warmup epoch.
        warmup_scheduler = None
    elif FLAGS.warmup is None:
        warmup_scheduler = None
    elif FLAGS.warmup == 'linear':
        warmup_scheduler = warmup.LinearWarmup(
            optimizer, max(1, warmup_steps)
        )
    elif FLAGS.warmup == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif FLAGS.warmup == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        raise ValueError(f'Unknown warmup: {FLAGS.warmup!r}')

    ckpt_path = join(saver.model_logdir, "last_ckpt.pt")
    start_epoch = 0

    train_losses, val_losses, test_losses, total_lrs = [], [], [], []
    # Keep the summed validation objective for diagnostics, but select
    # checkpoints using a separate target-balanced score.
    val_selection_scores = []
    val_selection_ratios = []
    val_ranking_scores = []
    gae_train_losses, gae_val_losses, gae_test_losses = [], [], []
    plot_test = False
    best_stopping_loss = float('inf')
    epochs_without_improvement = 0
    initial_selection = None
    initial_ratios = None
    best_qualified_rank_score = float('-inf')
    best_qualified_rank_epoch = None
    best_qualified_rank_ratios = None
    best_qualified_rank_kernel_ratios = None
    best_embedding_rank_score = float('-inf')
    best_embedding_rank_epoch = None
    best_embedding_rank_ratios = None

    if FLAGS.resume_training and exists(ckpt_path):
        st = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(st["model"])
        model.to(FLAGS.device)
        optimizer.load_state_dict(st["optim"])
        if FLAGS.scheduler is not None and st.get("scheduler") is not None:
            lr_scheduler.load_state_dict(st["scheduler"])
        for s in optimizer.state.values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(FLAGS.device)
        start_epoch = st.get("epoch", -1) + 1
        train_losses = st.get("train_losses", train_losses)
        val_losses   = st.get("val_losses", val_losses)
        test_losses  = st.get("test_losses", test_losses)
        val_selection_scores = st.get("val_selection_scores", [])
        val_selection_ratios = st.get("val_selection_ratios", [])
        val_ranking_scores = st.get("val_ranking_scores", [])
        initial_selection = st.get("initial_selection")
        initial_ratios = st.get("initial_ratios")
        best_qualified_rank_score = st.get(
            "best_qualified_rank_score", float('-inf')
        )
        best_qualified_rank_epoch = st.get("best_qualified_rank_epoch")
        best_qualified_rank_ratios = st.get(
            "best_qualified_rank_ratios"
        )
        best_qualified_rank_kernel_ratios = st.get(
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
        if val_losses and (
            len(val_selection_scores) != len(val_losses)
            or len(val_selection_ratios) != len(val_losses)
            or len(val_ranking_scores) != len(val_losses)
        ):
            raise RuntimeError(
                "Checkpoint predates qualified-rank validation selection or "
                "contains incomplete selection history. Restart this run "
                "without --resume_training."
            )
        if (
            baseline_breakdown is not None
            and stored_baseline != baseline_breakdown
        ):
            raise RuntimeError(
                "Validation baseline changed since the checkpoint was written. "
                "Restart without --resume_training."
            )
        if val_selection_scores:
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
            epochs_without_improvement = max(
                0,
                len(val_selection_scores)
                - 1
                - best_index,
            )
        if len(val_loader) > 0 and (
            initial_selection is None or initial_ratios is None
        ):
            raise RuntimeError(
                'Checkpoint lacks the initialized-model validation baseline. '
                'Restart without --resume_training.'
            )
        saver.log_info(f"Resuming from checkpoint at epoch {start_epoch}")

    if start_epoch == 0 and len(val_loader) > 0:
        initial_val, initial_breakdown, initial_gae, _ = test(
            val_loader, 'initial_val', model, -1, test_losses=[]
        )
        initial_selection, initial_ratios = compute_validation_selection_score(
            initial_breakdown, baseline_breakdown
        )
        saver.log_info(
            f'Untrained validation loss: {initial_val:.4f}; '
            f'breakdown={initial_breakdown}; '
            f'target-balanced selection={initial_selection:.4f}; '
            f'relative target losses={initial_ratios}'
        )
        log_loss(initial_breakdown, initial_gae, 'Initial validation')

    for epoch in range(start_epoch, FLAGS.epoch_num):
        plot_test = False
        timer = OurTimer()
        if (
            FLAGS.scheduler == 'plateau'
            and epoch < int(FLAGS.warmup_epochs)
        ):
            warmup_scale = (epoch + 1) / max(1, int(FLAGS.warmup_epochs))
            for group in optimizer.param_groups:
                group['lr'] = float(FLAGS.lr) * warmup_scale
        if FLAGS.feature_extract:
            check_feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        saver.log_info(f'Test batch ID (resample): {resample} - Epoch {epoch} train')
        loss, loss_dict_train, gae_loss_train, lrs = train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler)
        plot_test = model_update(model, train_losses, loss, epoch, plot_test, 'train')
        total_lrs.extend(lrs)
        if len(val_loader) > 0:
            saver.log_info(f'Epoch {epoch} val')
            val, loss_dict_val, gae_loss_val, _, val_metrics = test(
                val_loader,
                'val',
                model,
                epoch,
                return_metrics=True,
                qualification_target_stats=target_stats,
            )
            val_losses.append(val)
            current_selection_score, target_ratios = (
                compute_validation_selection_score(
                    loss_dict_val, baseline_breakdown
                )
            )
            val_selection_ratios.append(target_ratios)
            saver.log_info(
                f'Validation target-balanced selection score: '
                f'{current_selection_score:.6f}; '
                f'relative target losses={target_ratios}'
            )
            ranking_score = compute_macro_ranking_score(val_metrics)
            val_ranking_scores.append(ranking_score)
            per_kernel_target_ratios = val_metrics.attrs.get(
                'per_kernel_target_baseline_ratios'
            )
            if not per_kernel_target_ratios:
                raise RuntimeError(
                    'Validation metrics lack per-kernel target baseline ratios.'
                )
            worst_kernel_ratio = max(per_kernel_target_ratios.values())
            qualified = (
                all(float(ratio) < 1.0 for ratio in target_ratios.values())
                and ranking_score >= FLAGS.min_rank_tau
                and worst_kernel_ratio
                <= FLAGS.max_kernel_zero_baseline_ratio
            )
            embedding_rank_qualified = (
                all(float(ratio) < 1.0 for ratio in target_ratios.values())
                and ranking_score >= FLAGS.min_rank_tau
            )
            embedding_control_score = embedding_rank_control_score(
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
            saver.writer.add_scalar(
                'val/worst_target_kernel_macro_tau', ranking_score, epoch
            )
            if should_update_qualified_rank(
                target_ratios,
                per_kernel_target_ratios,
                ranking_score,
                best_qualified_rank_score,
                FLAGS.early_stopping_min_delta,
                FLAGS.min_rank_tau,
                FLAGS.max_kernel_zero_baseline_ratio,
            ):
                best_qualified_rank_score = ranking_score
                best_qualified_rank_epoch = epoch
                best_qualified_rank_ratios = dict(target_ratios)
                best_qualified_rank_kernel_ratios = dict(
                    per_kernel_target_ratios
                )
                if FLAGS.save_model:
                    save_checkpoint_with_sidecar(
                        model.state_dict(),
                        join(
                            saver.model_logdir,
                            'val_rank_model_state_dict.pth',
                        ),
                        'val_rank',
                        epoch,
                    )
                saver.log_info(
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
            plot_test = model_update(
                model,
                val_selection_scores,
                current_selection_score,
                epoch,
                plot_test,
                'val',
            )
            if (
                FLAGS.scheduler == 'plateau'
                and epoch + 1 >= int(FLAGS.warmup_epochs)
            ):
                scheduler_score = (
                    embedding_rank_control_score(
                        current_selection_score,
                        target_ratios,
                        ranking_score,
                    )
                    if FLAGS.checkpoint_objective == 'embedding_rank'
                    else current_selection_score
                )
                lr_scheduler.step(scheduler_score)

        log_loss(loss_dict_train, gae_loss_train, "Train")
        if len(val_loader) > 0:
            log_loss(loss_dict_val, gae_loss_val, "Val")
            saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, '
                            'Val loss: {:.4f}, Time: {}'.format(
                            epoch, loss, val, timer.time_and_clear())))
        #     gae_val_losses.append(gae_loss_val)
        else:
            saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                            'Time: {}'.format(
                epoch, loss, timer.time_and_clear())))
        gae_train_losses.append(gae_loss_train)

        if (epoch + 1) % 2 == 0 or (epoch + 1) == FLAGS.epoch_num:
            checkpoint_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": (lr_scheduler.state_dict() if FLAGS.scheduler is not None else None),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_selection_scores": val_selection_scores,
                "val_selection_ratios": val_selection_ratios,
                "val_ranking_scores": val_ranking_scores,
                "best_qualified_rank_score": best_qualified_rank_score,
                "best_qualified_rank_epoch": best_qualified_rank_epoch,
                "best_qualified_rank_ratios": best_qualified_rank_ratios,
                "best_qualified_rank_kernel_ratios": (
                    best_qualified_rank_kernel_ratios
                ),
                "best_embedding_rank_score": best_embedding_rank_score,
                "best_embedding_rank_epoch": best_embedding_rank_epoch,
                "best_embedding_rank_ratios": best_embedding_rank_ratios,
                "initial_selection": initial_selection,
                "initial_ratios": initial_ratios,
                "test_losses": test_losses,
                "target_stats": target_stats,
                "resource_stats": resource_stats,
                "baseline_breakdown": baseline_breakdown,
            }
            save_checkpoint_with_sidecar(
                checkpoint_state, ckpt_path, 'last', epoch
            )
            saver.log_info(f"Checkpoint saved at epoch {epoch} -> {ckpt_path}")
            if epoch + 1 == 10:
                epoch_10_path = join(saver.model_logdir, 'epoch_10_ckpt.pt')
                epoch_10_model_path = join(
                    saver.model_logdir, 'epoch_10_model_state_dict.pth'
                )
                save_checkpoint_with_sidecar(
                    checkpoint_state, epoch_10_path, 'epoch_10', epoch
                )
                save_checkpoint_with_sidecar(
                    model.state_dict(),
                    epoch_10_model_path,
                    'epoch_10_model',
                    epoch,
                )
                saver.log_info(
                    'Archived epoch 10 secondary checkpoints -> '
                    f'{epoch_10_path}, {epoch_10_model_path}'
                )

        selection_loss = (
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
        if selection_loss < (
            best_stopping_loss - float(FLAGS.early_stopping_min_delta)
        ):
            best_stopping_loss = selection_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            not bool(getattr(FLAGS, 'final_refit', False))
            and epochs_without_improvement
            >= int(FLAGS.early_stopping_patience)
        ):
            saver.log_info(
                'Early stopping after '
                f'{epochs_without_improvement} epochs without a '
                f'{FLAGS.early_stopping_min_delta:g} improvement.'
            )
            break

    epochs = range(len(train_losses))
    plot_loss_trend(epochs, train_losses, val_losses, [], saver.get_log_dir(), file_name='losses.png')
    if FLAGS.gae_T or FLAGS.gae_P:
        plot_loss_trend(epochs, gae_train_losses, gae_val_losses, [], saver.get_log_dir(), file_name='gae_losses.png')

    def restore_checkpoint(path, tag):
        if not exists(path):
            raise RuntimeError(f'Missing {tag} checkpoint: {path}')
        model.load_state_dict(torch.load(
            path,
            map_location=torch.device('cpu'),
            weights_only=False,
        ))
        model.to(FLAGS.device)

    # A final refit has no validation signal and keeps the model after the
    # exact epoch count selected by grouped validation.
    if bool(getattr(FLAGS, 'final_refit', False)):
        selection_tag = 'final_refit'
        selection_epoch = len(train_losses) - 1
        selection_path = join(
            saver.model_logdir,
            f'{selection_tag}_model_state_dict.pth',
        )
        if FLAGS.save_model:
            save_checkpoint_with_sidecar(
                model.state_dict(),
                selection_path,
                selection_tag,
                selection_epoch,
            )
    elif len(val_loader) > 0:
        absolute_epoch = val_selection_scores.index(
            min(val_selection_scores)
        )
        absolute_score = val_selection_scores[absolute_epoch]
        absolute_ratios = val_selection_ratios[absolute_epoch]
        absolute_path = join(
            saver.model_logdir, 'val_model_state_dict.pth'
        )
        restore_checkpoint(absolute_path, 'absolute-validation-selected')
        saver.log_info(
            f'Final absolute validation report for epoch {absolute_epoch}; '
            f'target-balanced score={absolute_score:.6f}; '
            f'relative target losses={absolute_ratios}'
        )
        test(
            val_loader,
            'best_val',
            model,
            absolute_epoch,
            test_losses=[],
        )
        if absolute_score >= 1.0 or not all(
            float(ratio) < 1.0 for ratio in absolute_ratios.values()
        ):
            raise RuntimeError(
                'No validation checkpoint beat the deterministic no-learning '
                'baseline on every target. The held-out test set will not be '
                'opened; revise the training configuration using validation '
                'data only.'
            )

        rank_path = join(
            saver.model_logdir, 'val_rank_model_state_dict.pth'
        )
        if best_qualified_rank_epoch is not None:
            if best_qualified_rank_ratios is None or not all(
                float(ratio) < 1.0
                for ratio in best_qualified_rank_ratios.values()
            ):
                raise RuntimeError(
                    'Stored rank checkpoint is not baseline-qualified.'
                )
            if (
                not best_qualified_rank_kernel_ratios
                or max(best_qualified_rank_kernel_ratios.values())
                > FLAGS.max_kernel_zero_baseline_ratio
                or best_qualified_rank_score < FLAGS.min_rank_tau
            ):
                raise RuntimeError(
                    'Stored rank checkpoint is not kernel/rank-qualified.'
                )
            restore_checkpoint(rank_path, 'qualified-rank-selected')
            saver.log_info(
                'Final qualified-rank validation report for epoch '
                f'{best_qualified_rank_epoch}; worst-target kernel-macro '
                f'tau-b={best_qualified_rank_score:.6f}; relative target '
                f'losses={best_qualified_rank_ratios}; worst kernel/target '
                f'baseline ratio='
                f'{max(best_qualified_rank_kernel_ratios.values()):.6f}'
            )
            test(
                val_loader,
                'best_rank_val',
                model,
                best_qualified_rank_epoch,
                test_losses=[],
            )

        embedding_rank_path = join(
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
            if best_qualified_rank_epoch is None:
                raise RuntimeError(
                    'No baseline-qualified rank checkpoint was produced. '
                    'The held-out test set remains locked.'
                )
            selection_tag = 'val_rank'
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
            selection_epoch = absolute_epoch
            selection_path = absolute_path

        # The optional held-out evaluation below must use exactly the declared
        # checkpoint objective, independent of report ordering above.
        restore_checkpoint(selection_path, f'{selection_tag}-selected')

        if initial_ratios is None:
            raise RuntimeError(
                'Missing initialized-model validation ratios.'
            )
        saver.log_info(
            "Initialized-model ratios are diagnostic only; "
            "release qualification uses the deterministic no-learning baseline. "
            f"initialized={initial_ratios}"
        )
    else:
        selection_tag = 'train'
        selection_epoch = train_losses.index(min(train_losses))
        selection_path = join(
            saver.model_logdir, 'train_model_state_dict.pth'
        )
        if len(test_loader) > 0:
            restore_checkpoint(selection_path, 'train-selected')

    evaluate_test = bool(getattr(FLAGS, 'evaluate_test', False))
    if len(test_loader) > 0 and evaluate_test:
        saver.log_info(
            f'Final test using the best {selection_tag} checkpoint from '
            f'epoch {selection_epoch}'
        )
        test_loss, loss_dict_test, gae_loss_test, _ = test(
            test_loader,
            'test',
            model,
            selection_epoch,
            plot_test=True,
            test_losses=[],
        )
        test_losses.append(test_loss)
        saver.writer.add_scalar('test/final', test_loss, selection_epoch)
        log_loss(loss_dict_test, gae_loss_test, 'Final test')
        saver.log_info(f'Final held-out test loss: {test_loss:.4f}')

    if len(test_loader) > 0 and evaluate_test:
        saver.log_info('The test set was evaluated exactly once.')
    elif len(test_loader) > 0:
        saver.log_info(
            'The held-out test set remains locked. Pass --evaluate_test only '
            'after the model, hyperparameters and epoch count are frozen.'
        )
    if len(val_loader) > 0:
        saver.log_info(
            'minimum summed validation objective at epoch: '
            f'{val_losses.index(min(val_losses))}'
        )
        saver.log_info(
            'target-balanced absolute checkpoint at epoch: '
            f'{absolute_epoch}'
        )
        if best_qualified_rank_epoch is not None:
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
            f'active checkpoint objective={FLAGS.checkpoint_objective}; '
            f'tag={selection_tag}; epoch={selection_epoch}'
        )
    if FLAGS.scheduler is not None:
        plot_lr_trend(total_lrs, FLAGS.epoch_num + 1, saver.get_log_dir())
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')


def train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler):
    model.train()
    lrs = []
    total_loss, correct, i, example_count = 0, 0, 0, 0
    target_list, loss_dict = set_target_list()
    accumulation_steps = int(FLAGS.grad_accum_steps)
    batch_count = len(train_loader)
    optimizer.zero_grad(set_to_none=True)
    for data in tqdm(train_loader):
        if FLAGS.scheduler is not None:
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            if i == 0:
                saver.log_info(f"epoch = {epoch}, learning rate = {lr}")
        data = data.to(FLAGS.device)
        example_count += int(data.num_graphs)
        out_dict, loss, loss_dict_, gae_loss = model(data)
        window_start = (i // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, batch_count - window_start)
        (loss / window_size).backward()
        update_optimizer = (i + 1) % accumulation_steps == 0 or i + 1 == batch_count
        if update_optimizer:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        #if FLAGS.scheduler is not None:
            # lr_scheduler.step()
            # lr_scheduler.step(lr_scheduler.last_epoch+1)
        if update_optimizer and lr_scheduler is not None and FLAGS.scheduler != 'plateau':
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    lr_scheduler.step()
            else:
                lr_scheduler.step()
        # lr = optimizer.param_groups[0]['lr']

        total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct)
        if FLAGS.task == 'regression': loss_dict, total_loss = total_loss_dict
        else: pred, correct, total_loss = total_loss_dict

        saver.writer.add_scalar('loss/loss', loss, epoch * len(train_loader) + i)
        for key, value in loss_dict_.items():
            if key.startswith('rank_aux/'):
                saver.writer.add_scalar(
                    key, value.item(), epoch * len(train_loader) + i
                )
        i += 1

    if FLAGS.scheduler is not None and epoch < 2:
        create_dir_if_not_exists(join(saver.get_log_dir(), 'lrs'))
    if FLAGS.task == 'regression':
        return (
            total_loss / example_count,
            {key: v / example_count for key, v in loss_dict.items()},
            gae_loss,
            lrs,
        )
    else:
        return 1 - correct / total_loss, {key: v / len(train_loader) for key, v in loss_dict.items()}, gae_loss, lrs


def inference(dataset, init_pragma_dict=None, model_path=FLAGS.model_path,
              val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio,
              resample=-1, model_id=0, is_train_set=False, is_val_set=False):

    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)

    if FLAGS.tiny_overfit:
        tiny_ds = dataset_dict['train']
        li = [tiny_ds, tiny_ds, tiny_ds]
        saver.log_info(
            f"[tiny_overfit][inference] using identical subset for train/val/test "
            f"(n={len(tiny_ds)})"
        )
    else:
        if dataset_dict.get('val') is not None:
            li = [
                dataset_dict['train'],
                dataset_dict['val'],
                dataset_dict['test'] or MyOwnDataset(data_files=[]),
            ]
        elif resample == -1:
            li = split_dataset(
                dataset_dict['train'],
                r1,
                r2,
                dataset_test=dataset_dict['test']
            )
        else:
            li = split_dataset_resample(
                dataset_dict['train'],
                1.0 - val_ratio - test_ratio,
                val_ratio,
                test_ratio,
                test_id=resample
            )

    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    test_set = test_loader
    if is_train_set:
        test_set = train_loader
        saver.info('running inference on train set')
    elif is_val_set:
        test_set = val_loader
        saver.info('running inference on val set')

    if init_pragma_dict is None:
        init_pragma_dict = {'all': [1, 21]}
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=init_pragma_dict).to(FLAGS.device)

    if model_path is not None:
        saver.info(f'loading model from {model_path}')
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        model.load_state_dict(state)
        shutil.copy(model_path, join(saver.logdir, f"{(basename(model_path)).split('.')[0]}-{model_id}.pth"))
    else:
        saver.error('model path should be set during inference')
        raise RuntimeError()

    if model_id == 0:
        saver.log_model_architecture(model)

    data_list = []

    if FLAGS.task == 'regression':
        csv_dict = {'header': ['gname', 'pragma']}
        test_loss, loss_dict, gae_loss, MSE_loss = test(
            test_set,
            'test',
            model,
            0,
            plot_test=True,
            csv_dict=csv_dict,
            data_list=data_list,
            is_train_set=is_train_set,
            is_val_set=is_val_set
        )

        loss_dict = {k: round(v, 4) for k, v in loss_dict.items()}
        saver.log_info(f'{loss_dict}')
        saver.log_info('Test loss: {:.7f}, MSE loss: {:.7f}'.format(test_loss, MSE_loss))

        if FLAGS.sanity_print_n > 0:
            _print_sanity_rows(csv_dict, FLAGS.sanity_print_n)

        saver.log_dict_of_dicts_to_csv(f'actual-prediction-{model_id}', csv_dict, csv_dict['header'])
        print(len(data_list), 'out of', len(test_loader))
    else:
        test_loss, loss_dict_test = test(test_loader, 'test', model, 0)
        saver.log_info(('Test loss: {:.3f}'.format(test_loss)))


def test(loader, tvt, model, epoch, plot_test=False, test_losses=None,
         csv_dict=None, data_list=None, is_train_set=False,
         is_val_set=False, return_metrics=False,
         qualification_target_stats=None):
    if test_losses is None:
        test_losses = [-1]
    if data_list is None:
        data_list = []
    model.eval()
    my_softplus = nn.Softplus()
    inference_loss, correct, total, count_data = 0, 0, 0, 1
    points_dict = OrderedDict()
    resource_rows = []
    target_list, loss_dict = set_target_list()
    for target_name in target_list:
        points_dict[target_name] = {
            'true': [],
            'pred': [],
            'physical_true': [],
            'physical_pred': [],
            'baseline_log2': [],
            'actual_delta_log2': [],
            'predicted_delta_log2': [],
            'kernel': [],
            'target_group': [],
            'point_key': [],
            'sigma_mu': [],
            'sigma+mu': [],
            'sigma': [],
            'error': [],
        }
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(FLAGS.device)
            out_dict, loss, loss_dict_, gae_loss = model(data)
            total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total, correct)
            if FLAGS.task == 'regression': loss_dict, total = total_loss_dict
            else: pred, correct, total = total_loss_dict

            if 'resource_log1p' in out_dict:
                predicted_resource = torch.expm1(
                    out_dict['resource_log1p']
                ).clamp_min(0.0)
                actual_resource = data.resource_util.reshape(
                    -1, len(RESOURCE_NAMES)
                )
                for index in range(predicted_resource.shape[0]):
                    resource_rows.append({
                        'kernel': _kernel_at(data, index),
                        'actual': actual_resource[index].detach().cpu().numpy(),
                        'predicted': (
                            predicted_resource[index].detach().cpu().numpy()
                        ),
                    })

            for target_name in target_list:
                if 'inf' in FLAGS.subtask:
                    saver.info(f'{target_name}')
                if FLAGS.task == 'class': out = pred
                elif FLAGS.encode_log and 'perf' in target_name: out = out_dict['perf']
                else: out = out_dict[target_name]

                for i in range(len(out)):
                    out_value = out[i].item()
                    target_value = _get_y_with_target(data, target_name)[i].item()
                    if FLAGS.encode_log and target_name == 'actual_perf':
                        out_value = 2**(out_value) * (1 / FLAGS.normalizer)
                    if 'inf' in FLAGS.subtask:
                        inference_loss += inference_loss_function(out_value, target_value)
                        count_data += 1
                        update_csv_dict(csv_dict, data, i, target_name, target_value, out_value)

                        if out_value != target_value: # and sigma[i].item() > 0.57:
                            saver.info(f"{target_name} data {i} {_get_y_with_target(data, 'gname')[i]} pramga {_get_y_with_target(data, 'pragmas')[i][0].item()} actual value: {target_value:.2f}, predicted value: {out_value:.2f}") #, sigma: {sigma[i].item()}, log_var: {out_[i, 1].item()}')")

                    points_dict[target_name]['pred'].append((target_value, out_value))
                    points_dict[target_name]['true'].append((target_value, target_value))
                    points_dict[target_name]['error'].append((target_value, abs(target_value - out_value)))

                    physical_attr = (
                        'actual_perf'
                        if target_name in {'perf', 'actual_perf'}
                        else 'actual_effective_area'
                    )
                    physical_true = _get_y_with_target(
                        data, physical_attr
                    )[i].item()
                    points_dict[target_name]['physical_true'].append(
                        physical_true
                    )
                    points_dict[target_name]['physical_pred'].append(
                        _inverse_log2_target(out_value)
                    )
                    if getattr(FLAGS, 'target_mode', 'absolute') == 'reference_delta':
                        public_target = (
                            'perf'
                            if target_name == 'actual_perf' else target_name
                        )
                        baseline_value = out_dict[
                            f'{public_target}_baseline'
                        ][i].item()
                        predicted_delta = out_dict[
                            f'{public_target}_delta'
                        ][i].item()
                        points_dict[target_name]['baseline_log2'].append(
                            baseline_value
                        )
                        points_dict[target_name]['actual_delta_log2'].append(
                            target_value - baseline_value
                        )
                        points_dict[target_name]['predicted_delta_log2'].append(
                            predicted_delta
                        )
                    points_dict[target_name]['kernel'].append(
                        _kernel_at(data, i)
                    )
                    points_dict[target_name]['target_group'].append(
                        _string_attribute_at(data, 'target_group', i)
                    )
                    points_dict[target_name]['point_key'].append(
                        _string_attribute_at(data, 'key', i)
                    )


    if FLAGS.task != 'class' and FLAGS.plot_pred_points and tvt == 'test' and (
        plot_test or (
            test_losses
            and (total / len(loader.dataset)) < min(test_losses)
        )
    ):
        from utils import plot_points_with_subplot, plot_points_with_subplot_sigma
        saver.log_info(f'@@@ plot_pred_points')
        assert(isinstance(FLAGS.target, list))
        use_sigma = False
        label = f'epoch_{epoch+1}_{tvt}_train' if is_train_set else f'epoch_{epoch+1}_{tvt}_test'
        if is_val_set: label = f'epoch_{epoch+1}_{tvt}_val'
        if 'inf' in FLAGS.subtask:
            plot_points_with_subplot(points_dict, label, saver.plotdir, target_list, use_sigma=use_sigma)

    if FLAGS.task == 'regression':
        df = _report_rmse_etc(points_dict, f'[{tvt}] epoch {epoch}', print_result=True)
        report_resource_metrics(resource_rows, f'[{tvt}] epoch {epoch}')
        if qualification_target_stats is not None:
            df.attrs['per_kernel_target_baseline_ratios'] = (
                compute_per_kernel_target_baseline_ratios(
                    points_dict, qualification_target_stats
                )
            )
        for _, row in df[df['aggregation'] == 'point_micro'].iterrows():
            saver.log_info(
                f"[{tvt}] {row['target']} physical RMSE: "
                f"{row['rmse']:.4f} | MAE: {row['mae']:.4f} | "
                f"MAPE: {row['mape']*100:.2f}% | tau-b: "
                f"{row['tau']:.4f}"
            )
        if tvt in {'best_val', 'best_rank_val', 'test'}:
            metrics_path = join(
                saver.model_logdir, f'{tvt}_physical_metrics.csv'
            )
            df.to_csv(metrics_path, index=False)
            saver.log_info(f'Wrote physical-unit metrics to {metrics_path}')
            prediction_rows = []
            for target_name, values in points_dict.items():
                for index, kernel in enumerate(values['kernel']):
                    actual_log2, predicted_log2 = values['pred'][index]
                    prediction_rows.append({
                        'target': target_name,
                        'kernel': kernel,
                        'target_group': values['target_group'][index],
                        'point_key': values['point_key'][index],
                        'actual_log2': actual_log2,
                        'predicted_log2': predicted_log2,
                        'actual_physical': values['physical_true'][index],
                        'predicted_physical': values['physical_pred'][index],
                        'baseline_log2': (
                            values['baseline_log2'][index]
                            if values['baseline_log2'] else np.nan
                        ),
                        'actual_delta_log2': (
                            values['actual_delta_log2'][index]
                            if values['actual_delta_log2'] else np.nan
                        ),
                        'predicted_delta_log2': (
                            values['predicted_delta_log2'][index]
                            if values['predicted_delta_log2'] else np.nan
                        ),
                    })
            predictions_path = join(
                saver.model_logdir, f'{tvt}_predictions.csv'
            )
            pd.DataFrame(prediction_rows).to_csv(
                predictions_path, index=False
            )
            saver.log_info(
                f'Wrote point predictions to {predictions_path}'
            )

    if FLAGS.task == 'regression':
        if 'inf' in FLAGS.subtask:
            _report_rmse_etc(points_dict, f'epoch {epoch}:', True)
        example_count = len(loader.dataset)
        result = (
            total / example_count,
            {key: v / example_count for key, v in loss_dict.items()},
            gae_loss,
            inference_loss / max(1, count_data) * len(target_list),
        )
        return (*result, df) if return_metrics else result
    else:
        if 'inf' in FLAGS.subtask: report_class_loss(points_dict)
        result = (
            1 - correct / total,
            {key: v / len(loader) for key, v in loss_dict.items()},
            gae_loss,
            0,
        )
        return (*result, None) if return_metrics else result


def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = [data for data,_ in d['pred']]
    pred = [data for _,data in d['pred']]
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')


def _report_rmse_etc(points_dict, label, print_result=True):
    """Report physical-unit point and per-kernel macro regression metrics."""
    if print_result:
        saver.log_info(label)

    def metrics(true_values, predicted_values):
        true_values = np.asarray(true_values, dtype=float)
        predicted_values = np.asarray(predicted_values, dtype=float)
        mse = float(mean_squared_error(true_values, predicted_values))
        tau = float(kendalltau(true_values, predicted_values)[0])
        return {
            'mape': float(mean_absolute_percentage_error(
                true_values, predicted_values
            )),
            'rmse': float(np.sqrt(mse)),
            'mse': mse,
            'mae': float(mean_absolute_error(
                true_values, predicted_values
            )),
            'max_err': float(max_error(true_values, predicted_values)),
            'tau': tau,
        }

    rows = []
    for target_name, values in points_dict.items():
        physical_target = (
            'latency_ms'
            if target_name in {'perf', 'actual_perf'}
            else 'area_score'
        )
        true_values = values['physical_true']
        predicted_values = values['physical_pred']
        kernels = values['kernel']
        target_groups = values.get('target_group', kernels)
        if not true_values or not (
            len(true_values) == len(predicted_values) == len(kernels)
        ):
            raise RuntimeError(
                f'Incomplete physical metrics for {target_name}.'
            )

        micro = metrics(true_values, predicted_values)
        rows.append({
            'target': physical_target,
            'aggregation': 'point_micro',
            'samples': len(true_values),
            'kernels': len(set(kernels)),
            **micro,
        })

        per_kernel = []
        for kernel in sorted(set(kernels)):
            indices = [
                index for index, value in enumerate(kernels)
                if value == kernel
            ]
            kernel_metrics = metrics(
                [true_values[index] for index in indices],
                [predicted_values[index] for index in indices],
            )
            # Ranking is meaningful only between designs for the same FPGA
            # and clock; never reward cross-target QoR scale differences.
            target_taus = []
            for target_group in sorted({target_groups[index] for index in indices}):
                target_indices = [
                    index for index in indices if target_groups[index] == target_group
                ]
                if len(target_indices) < 2:
                    continue
                tau = kendalltau(
                    [true_values[index] for index in target_indices],
                    [predicted_values[index] for index in target_indices],
                )[0]
                if np.isfinite(tau):
                    target_taus.append(float(tau))
            kernel_metrics['tau'] = (
                float(np.mean(target_taus)) if target_taus else float('nan')
            )
            per_kernel.append(kernel_metrics)
            rows.append({
                'target': physical_target,
                'aggregation': 'kernel',
                'kernel': kernel,
                'samples': len(indices),
                'kernels': 1,
                **kernel_metrics,
            })

        macro = {}
        for name in ('mape', 'rmse', 'mse', 'mae', 'max_err', 'tau'):
            finite_values = [
                item[name] for item in per_kernel
                if np.isfinite(item[name])
            ]
            macro[name] = (
                float(np.mean(finite_values))
                if finite_values else float('nan')
            )

        rows.append({
            'target': physical_target,
            'aggregation': 'kernel_macro',
            'kernel': '',
            'samples': len(true_values),
            'kernels': len(per_kernel),
            **macro,
        })
    
    # Latency and area have different units, so there is deliberately no
    # combined "tot/avg" error row. Report both targets independently.
    df = pd.DataFrame(rows)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(df.round(4))
    return df


def compute_macro_ranking_score(metrics_df):
    """Return the worst target's equal-kernel mean Kendall tau-b."""
    rows = metrics_df[metrics_df['aggregation'] == 'kernel'].copy()
    if rows.empty:
        raise RuntimeError('Missing per-kernel rows in metrics table.')
    rows['tau'] = np.nan_to_num(
        rows['tau'].to_numpy(dtype=float),
        nan=0.0,
    )
    per_target = rows.groupby('target', sort=True)['tau'].mean()
    expected_targets = len(set_target_list()[0])
    if len(per_target) != expected_targets:
        raise RuntimeError(
            'Expected one kernel-macro ranking score per model target, got '
            f'{len(per_target)} for {expected_targets} targets.'
        )
    return float(per_target.min())
