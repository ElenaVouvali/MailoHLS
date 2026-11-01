"""Small, cached, family-disjoint V4 QoR learning probe.

The probe reuses the authenticated V4 graphs, tensorizer, candidate-demand
features, model, loss and evaluator. It caches only their deterministic input
tensors; measurements are never included in static graph memory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import numpy as np
import torch

from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch import target_graph_v22 as overlay
from experiments.v22_gnn_training import checkpointing
from experiments.v22_gnn_training import train

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'salt-hls-v4-small-family-disjoint-probe-v1'
CACHE_SCHEMA = 'salt-hls-v4-small-probe-feature-cache-v1'
SEED = 123
TRAIN_KERNELS = (
    'machsuite-gemm-ncubed',       # arithmetic-heavy, small compiler graph
    'machsuite-spmv-ellpack',      # sparse, memory-access-heavy
    'machsuite-stencil2d',         # regular loop and memory accesses
    'machsuite-md-knn',            # compact irregular-access computation
    'rodinia_dilate_0_baseline_0', # small graph with an array action
)
VALIDATION_KERNELS = (
    'machsuite-sort-radix',
    'rodinia_pathfinder_0_baseline_0',
)
PROFILES = ('zcu104', 'u200')
CLOCK_NS = 10.0
CANDIDATES_PER_CONTEXT = 8
TRAIN_UPDATES = 20
PREPARATION_LIMIT_SECONDS = 600
GPU_PHASE_LIMIT_SECONDS = 1200
FEATURE_FILES = (
    'experiments/v22_gnn_training/train.py',
    'experiments/v22_gnn_training/audit.py',
    'experiments/v22_gnn_training/fast_probe.py',
    'experiments/target_graph_readiness/experiment.py',
    'GNN_branch/target_graph_v22.py',
    'GNN_branch/target_graph_tensor_v22.py',
    'GNN_branch/action_demand_v22.py',
    'GNN_branch/semantic_tensor_features_v22.py',
    'GNN_branch/action_point_relations.py',
    'LLM_branch/common/directive_domains.py',
)


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _source_hashes():
    return {name: overlay.file_hash(ROOT / name) for name in FEATURE_FILES}


def _alarm(seconds):
    """Hard wall-clock guard for the bounded Linux experiment phases."""
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError(f'phase exceeded its {seconds}s wall-clock budget')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    class Alarm:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    return Alarm()


def _graph_sizes(release_dir, audit):
    sizes = {}
    roster = set(TRAIN_KERNELS + VALIDATION_KERNELS)
    for row in audit['kernels']:
        member = Path(row['graph_member']).name
        kernel = next((name for name in roster if member.startswith(name + '-')), None)
        if kernel is None:
            continue
        payload = json.loads((Path(release_dir) / row['graph_member']).read_text())
        sizes[kernel] = {
            'nodes': len(payload['nodes']), 'edges': len(payload['edges']),
            'registered_actions': row['registered_actions'],
            'matched_actions': row['matched_actions'],
            'array_actions': row['array_actions'],
            'source_sha256': row['source_sha256'],
            'metadata_sha256': row['metadata_sha256'],
            'graph_identity': row['graph_identity'],
            'graph_member_sha256': row['graph_member_sha256'],
            'apl_sha256': row['apl_sha256'],
        }
    if set(sizes) != set(TRAIN_KERNELS + VALIDATION_KERNELS):
        raise ValueError('Pilot roster does not resolve to authenticated graph-audit entries')
    return sizes


def _assignment_distance(left, right):
    keys = sorted(set(left) | set(right))
    return sum(left.get(key) != right.get(key) for key in keys)


def _select_training_candidates(rows, count):
    """Greedy directive-value coverage; it does not inspect QoR labels."""
    remaining = sorted(rows, key=lambda row: row['config_hash'])
    if len(remaining) <= count:
        return remaining
    chosen = [remaining.pop(0)]
    while remaining and len(chosen) < count:
        next_row = min(
            remaining,
            key=lambda row: (-min(_assignment_distance(row['assignment'], item['assignment'])
                                  for item in chosen), row['config_hash']))
        chosen.append(next_row)
        remaining.remove(next_row)
    return chosen


def _record_key(row, graph_identity):
    return _canonical_hash({
        'kernel': row['kernel'], 'profile': row['profile'], 'clock': row['clock'],
        'graph_identity': graph_identity, 'config_hash': row['config_hash'],
        'source_action_mapping_identity': row['source_action_mapping_sha256'],
        'feature_implementation_identity': row['feature_impl_sha256'],
        'candidate_demand_schema': train.action_demand.CANDIDATE_SCHEMA,
    })


def _graph_key(row, graph_identity, budget):
    return _canonical_hash({
        'graph_identity': graph_identity, 'tensor_schema': tensors.SCHEMA,
        'overlay_schema': overlay.SCHEMA, 'profile': row['profile'],
        'clock': row['clock'], 'budget_fraction': float(budget),
        'feature_implementation_identity': row['feature_impl_sha256'],
    })


def _tensor_fields_equal(left, right):
    left_fields, right_fields = dict(left), dict(right)
    if left_fields.keys() != right_fields.keys():
        return False
    return all(torch.equal(left_fields[key], right_fields[key])
               if torch.is_tensor(left_fields[key]) else left_fields[key] == right_fields[key]
               for key in left_fields)


def prepare(release_dir: Path, output: Path, timeout_seconds=PREPARATION_LIMIT_SECONDS):
    """Authenticate and freeze a small feature cache before any training."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty probe output: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timings = {}
    audit_path = output / 'data_audit.json'

    with _alarm(timeout_seconds):
        phase = time.monotonic()
        dataset = train.PreparedData(release_dir, audit_path,
                                     selected_kernels=TRAIN_KERNELS + VALIDATION_KERNELS)
        timings['authenticated_data_and_graph_loading_seconds'] = time.monotonic() - phase
        if set(dataset.graphs) != set(TRAIN_KERNELS + VALIDATION_KERNELS):
            raise ValueError('Prepared graph set differs from the frozen pilot roster')

        family_by_kernel = {
            'machsuite-gemm-ncubed': 'machsuite_gemm',
            'machsuite-spmv-ellpack': 'machsuite_spmv_ellpack',
            'machsuite-stencil2d': 'machsuite_stencil2d',
            'machsuite-md-knn': 'machsuite_md_knn',
            'rodinia_dilate_0_baseline_0': 'rodinia_dilate',
            'machsuite-sort-radix': 'machsuite_sort_radix',
            'rodinia_pathfinder_0_baseline_0': 'rodinia_pathfinder',
        }
        if any(kernel not in dataset.kernel_split for kernel in family_by_kernel):
            raise ValueError('Pilot kernel is absent from the authenticated family split')
        declared_families = set(dataset.split['train_families'] + dataset.split['val_families'])
        if not set(family_by_kernel.values()) <= declared_families:
            raise ValueError('Pilot family labels do not match the authenticated family list')
        if any(dataset.kernel_split[kernel] != ('train' if kernel in TRAIN_KERNELS else 'val')
               for kernel in family_by_kernel):
            raise ValueError('Pilot kernels are assigned to an unexpected split')
        if set(family_by_kernel[k] for k in TRAIN_KERNELS) & set(
                family_by_kernel[k] for k in VALIDATION_KERNELS):
            raise ValueError('Training and validation families are not disjoint')
        if family_by_kernel['machsuite-sort-radix'] == family_by_kernel['rodinia_pathfinder_0_baseline_0']:
            raise ValueError('Validation kernels must represent different held-out families')

        selected_groups = {}
        selection_report = {}
        for kernel in TRAIN_KERNELS + VALIDATION_KERNELS:
            split = 'train' if kernel in TRAIN_KERNELS else 'val'
            for profile in PROFILES:
                key = (kernel, profile, CLOCK_NS)
                rows = [dict(row) for row in dataset.groups.get(key, ())]
                rows = [row for row in rows if row['split'] == split]
                if not rows:
                    raise ValueError(f'No authenticated {split} measurements for {key}')
                if split == 'train':
                    chosen = _select_training_candidates(rows, CANDIDATES_PER_CONTEXT)
                    selection_method = 'greedy_maximum_complete-assignment_hamming_distance; hash tie-break'
                else:
                    chosen = sorted(rows, key=lambda row: row['config_hash'])[:CANDIDATES_PER_CONTEXT]
                    selection_method = 'first configuration hashes; no validation QoR used for selection'
                if len(chosen) < 2:
                    raise ValueError(f'Fewer than two measured candidates in {key}')
                group_id = _canonical_hash({'kernel': kernel, 'profile': profile, 'clock': CLOCK_NS})
                selected_groups[group_id] = chosen
                latency_outcomes = len({float(row['physical'][0]) for row in chosen})
                selection_report[group_id] = {
                    'kernel': kernel, 'split': split, 'family': family_by_kernel[kernel],
                    'profile': profile, 'clock_ns': CLOCK_NS,
                    'available_measured_configurations': len(rows),
                    'selected_measured_configurations': len(chosen),
                    'selection_method': selection_method,
                    'candidate_hashes': [row['config_hash'] for row in chosen],
                    'distinct_measured_latency_outcomes': latency_outcomes,
                    'measured_feasible_at_half_capacity': sum(
                        all(value <= capacity * 0.5 for value, capacity
                            in zip(row['physical'][1:], row['capacities'])) for row in chosen),
                }
                if split == 'val' and latency_outcomes < 2:
                    raise ValueError(f'Hash-selected validation pool lacks two latency outcomes: {key}')

        train_records = [row for group_id, rows in selected_groups.items()
                         if selection_report[group_id]['split'] == 'train' for row in rows]
        mean, std = train._fit_normalization(train_records)
        rank_scales = train.fit_ranking_scales(train_records)
        graph_sizes = _graph_sizes(release_dir, dataset.audit)
        audit_by_kernel = {
            kernel: next(item for item in dataset.audit['kernels']
                         if Path(item['graph_member']).name.startswith(kernel + '-'))
            for kernel in TRAIN_KERNELS + VALIDATION_KERNELS
        }
        for rows in selected_groups.values():
            for row in rows:
                info = audit_by_kernel[row['kernel']]
                row['source_action_mapping_sha256'] = _canonical_hash({
                    'source_sha256': info['source_sha256'],
                    'metadata_sha256': info['metadata_sha256'],
                    'apl_sha256': info['apl_sha256'],
                    'action_columns': info['action_columns'],
                })
                row['feature_impl_sha256'] = _canonical_hash(_source_hashes())
        graph_cache = {}
        graph_seconds = 0.0
        candidate_cache = {}
        candidate_seconds = 0.0
        equivalence_checks = {'graphs': 0, 'candidate_vectors': 0, 'candidate_demands': 0}
        for group_id, rows in selected_groups.items():
            representative = rows[0]
            info = audit_by_kernel[representative['kernel']]
            for budget in (1.0, 0.5):
                key = _graph_key(representative, info['graph_identity'], budget)
                phase = time.monotonic()
                base_graph, auth = dataset.graphs[representative['kernel']]
                target_profile = dataset.registry['profiles'][representative['profile']]
                budget_values = {resource: int(value * budget)
                                 for resource, value in zip(RESOURCE_ORDER,
                                                            representative['capacities'])}
                enriched = overlay.enrich(base_graph, representative['profile'],
                                          representative['clock'], budgets=budget_values,
                                          authentication=auth)
                standard_hardware = tensors.tensorize(enriched).cpu()
                standard_numerical = tensors.without_hardware_messages(standard_hardware)
                graph_seconds += time.monotonic() - phase
                # Rebuild once through the same public graph path as a direct
                # equality oracle. Both arms share this exact enriched tensor.
                if not _tensor_fields_equal(standard_hardware, dataset.graph_tensor(
                        representative, budget, 'hardware').cpu()):
                    raise ValueError('Precomputed hardware graph differs from the V4 public tensor path')
                graph_cache[key] = {'hardware': standard_hardware,
                                    'numerical': standard_numerical}
                equivalence_checks['graphs'] += 1
            for row in rows:
                identity = _record_key(row, info['graph_identity'])
                phase = time.monotonic()
                original_candidate, original_demand = dataset.candidate_inputs(row)
                candidate_seconds += time.monotonic() - phase
                candidate_cache[identity] = (original_candidate.cpu(), original_demand.cpu())
                # Re-evaluate once without the PreparedData LRU to verify the
                # frozen values are exactly the established V4 interfaces.
                direct_candidate = train.candidate_vector(row['assignment'], dataset.actions[row['kernel']])
                if not torch.equal(candidate_cache[identity][0], direct_candidate):
                    raise ValueError('Cached candidate vector differs from V4 candidate encoding')
                equivalence_checks['candidate_vectors'] += 1
                base, auth = dataset.graphs[row['kernel']]
                joint = overlay.enrich(base, row['profile'], row['clock'], authentication=auth)
                direct_demand, _ = train.candidate_demand_tensor(
                    joint, row['assignment'], dataset.actions[row['kernel']])
                if not torch.equal(candidate_cache[identity][1], direct_demand):
                    raise ValueError('Cached candidate-demand tensor differs from V4 descriptor path')
                equivalence_checks['candidate_demands'] += 1

        identity = {
            'graph_release_archive_sha256': dataset.audit['graph_release']['archive_sha256'],
            'dataset_sha256': dataset.audit['dataset']['dataset_sha256'],
            'split_sha256': dataset.audit['dataset']['split_sha256'],
            'graph_schema': train.program.SCHEMA_V22,
            'overlay_schema': overlay.SCHEMA,
            'tensor_schema': tensors.SCHEMA,
            'candidate_demand_schema': train.action_demand.CANDIDATE_SCHEMA,
            'source_hashes': _source_hashes(),
            'graphs': graph_sizes,
        }
        manifest = {
            'schema': SCHEMA, 'seed': SEED, 'identity': identity,
            'identity_sha256': _canonical_hash(identity),
            'training_kernels': list(TRAIN_KERNELS),
            'validation_kernels': list(VALIDATION_KERNELS),
            'validation_families': [family_by_kernel[k] for k in VALIDATION_KERNELS],
            'profiles': list(PROFILES), 'clock_ns': CLOCK_NS,
            'candidates_per_context_target': CANDIDATES_PER_CONTEXT,
            'train_updates_per_arm': TRAIN_UPDATES,
            'train_selection_uses_qor': False,
            'validation_candidate_selection_uses_qor': False,
            'validation_and_test_measurements_used_for_fitting': False,
            'group_selection': selection_report,
            'graph_sizes': graph_sizes,
            'feature_cache': 'features.pt',
            'feature_cache_keys': {
                'graph': ['graph_identity', 'tensor_schema', 'overlay_schema', 'profile', 'clock_ns', 'budget_fraction', 'feature implementation identity'],
                'candidate': ['graph_identity', 'source/action mapping identity', 'feature implementation identity', 'profile', 'clock_ns',
                              'complete assignment hash', 'candidate-demand schema'],
            },
            'equivalence_checks': equivalence_checks,
            'timings_seconds': {
                **timings, 'graph_tensorization': graph_seconds,
                'candidate_feature_preparation': candidate_seconds,
            },
        }
        payload = {
            'schema': CACHE_SCHEMA, 'identity_sha256': manifest['identity_sha256'],
            'records': selected_groups, 'graphs': graph_cache, 'candidates': candidate_cache,
            'normalization_mean': mean, 'normalization_std': std,
            'ranking_scales': rank_scales,
        }
        phase = time.monotonic()
        checkpointing.atomic_torch_save(output / 'features.pt', payload)
        timings['cache_serialization_seconds'] = time.monotonic() - phase
        manifest['timings_seconds'] = {
            **timings, 'graph_tensorization': graph_seconds,
            'candidate_feature_preparation': candidate_seconds,
            'total_preparation': time.monotonic() - started,
        }
        manifest['feature_cache_sha256'] = overlay.file_hash(output / 'features.pt')
        manifest['manifest_sha256'] = _canonical_hash(manifest)
        checkpointing.atomic_json(output / 'cohort_manifest.json', manifest)
        checkpointing.atomic_json(output / 'prepare_summary.json', {
            'status': 'prepared', 'timings_seconds': manifest['timings_seconds'],
            'selected_record_count': sum(len(rows) for rows in selected_groups.values()),
            'selected_contexts': len(selected_groups), 'equivalence_checks': equivalence_checks,
            'manifest_sha256': manifest['manifest_sha256'],
        })
    return manifest


class ProbeData:
    """Indexed, immutable tensors prepared once for both arms and validation."""
    def __init__(self, payload, manifest):
        self.groups = defaultdict(list)
        self.graphs = payload['graphs']
        self.candidates = payload['candidates']
        self.graph_identity = {kernel: item['graph_identity']
                               for kernel, item in manifest['graph_sizes'].items()}
        for group_id, rows in payload['records'].items():
            for row in rows:
                self.groups[(row['kernel'], row['profile'], float(row['clock']))].append(row)
        for rows in self.groups.values():
            rows.sort(key=lambda item: item['config_hash'])
        self.ablation = False

    def graph_tensor(self, record, budget_fraction, arm):
        key = _graph_key(record, self.graph_identity[record['kernel']], budget_fraction)
        if key not in self.graphs:
            raise KeyError('No authenticated precomputed graph cache entry: ' + key)
        selected_arm = 'numerical' if self.ablation else arm
        return self.graphs[key][selected_arm].clone()

    def candidate_inputs(self, record):
        key = _record_key(record, self.graph_identity[record['kernel']])
        if key not in self.candidates:
            raise KeyError('No authenticated precomputed candidate cache entry: ' + key)
        return self.candidates[key]


def _load_cache(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'cohort_manifest.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported small-probe cohort schema')
    stored_hash = manifest.pop('manifest_sha256', None)
    if stored_hash != _canonical_hash(manifest):
        raise ValueError('Cohort manifest identity/checksum mismatch')
    manifest['manifest_sha256'] = stored_hash
    if manifest['identity'].get('source_hashes') != _source_hashes():
        raise ValueError('Feature source changed after the cohort cache was prepared')
    cache_path = output / manifest['feature_cache']
    if overlay.file_hash(cache_path) != manifest['feature_cache_sha256']:
        raise ValueError('Precomputed feature-cache SHA256 mismatch')
    payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    if payload.get('schema') != CACHE_SCHEMA or payload.get('identity_sha256') != manifest['identity_sha256']:
        raise ValueError('Feature cache does not match the authenticated cohort manifest')
    return manifest, payload


def _verify_idle_gpu(expected_name, expected_uuid):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the requested physical GPU UUID')
    rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    matches = []
    for line in rows:
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 5 and parts[0] == expected_uuid:
            matches.append({'uuid': parts[0], 'name': parts[1], 'total': int(parts[2]),
                            'free': int(parts[3]), 'utilization': int(parts[4])})
    if len(matches) != 1 or expected_name.casefold() not in matches[0]['name'].casefold():
        raise RuntimeError(f'Requested physical GPU was not uniquely found: {matches}')
    gpu = matches[0]
    process_text = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True).strip()
    occupied = []
    for line in process_text.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 4 and parts[0] == expected_uuid:
            occupied.append(parts)
    own = os.getpid()
    others = [row for row in occupied if int(row[1]) != own]
    # During the second arm the probe process itself may still be visible to
    # NVML. It is the only allowed process; any unrelated process blocks use.
    if others or gpu['free'] < 6000 or (not occupied and gpu['utilization'] > 15):
        raise RuntimeError(f'Refusing to share or use busy GPU {gpu}; processes={occupied}')
    return gpu


def _hardware_relation_gradient_norm(model):
    relations = [name for name in tensors.RELATIONS
                 if name in overlay.RELATIONS and ('hardware:' in name or name.startswith('target:'))]
    indices = [tensors.RELATIONS.index(name) for name in relations]
    gradient = model.graph.relation.weight.grad
    if gradient is None or not indices:
        return 0.0
    selected = gradient[indices]
    if not torch.isfinite(selected).all():
        raise ValueError('Nonfinite hardware-relation gradient')
    return float(selected.norm().detach().cpu())


def _predict_action_memory(model, data, record, device, budget=1.0):
    graph = data.graph_tensor(record, budget, 'hardware').to(device)
    memory, pooled = model.graph.memories(graph)
    return memory.detach().cpu(), pooled.detach().cpu()


def _ablation_summary(model, data, mean, std, device, timeout_at):
    before = time.monotonic()
    data.ablation = False
    hardware_metrics = train.evaluate(model, data,
        {key: rows for key, rows in data.groups.items() if rows[0]['split'] == 'val'},
        mean, std, 'hardware', device, max_candidates=None)
    hw_contexts = {row['kernel'] + '|' + row['profile']: row
                   for row in hardware_metrics['contexts']}
    data.ablation = True
    numerical_metrics = train.evaluate(model, data,
        {key: rows for key, rows in data.groups.items() if rows[0]['split'] == 'val'},
        mean, std, 'hardware', device, max_candidates=None)
    num_contexts = {row['kernel'] + '|' + row['profile']: row
                    for row in numerical_metrics['contexts']}
    changed = 0
    prediction_deltas = []
    for key, rows in data.groups.items():
        if rows[0]['split'] != 'val':
            continue
        hw = hw_contexts[f'{key[0]}|{key[1]}']
        no_hw = num_contexts[f'{key[0]}|{key[1]}']
        hw_sel = hw['primary_full_budget']['feasibility']['PARETO_LATENCY']['selection'].get('selected_config_hash')
        num_sel = no_hw['primary_full_budget']['feasibility']['PARETO_LATENCY']['selection'].get('selected_config_hash')
        changed += hw_sel != num_sel
        candidates = torch.stack([data.candidate_inputs(row)[0] for row in rows]).to(device)
        demands = torch.stack([data.candidate_inputs(row)[1] for row in rows]).to(device)
        output_pair = []
        for flag in (False, True):
            data.ablation = flag
            graph = data.graph_tensor(rows[0], 1.0, 'hardware').to(device)
            memory, pooled = model.graph.memories(graph)
            z = model.from_memory(graph, memory, pooled, candidates, demands, 'hardware')
            output_pair.append(torch.expm1(z * std.to(device) + mean.to(device)).clamp_min(0))
        prediction_deltas.append((output_pair[0] - output_pair[1]).abs().mean(0).detach().cpu().tolist())

    data.ablation = False
    shifts = []
    model.eval()
    with torch.no_grad():
        for key, rows in data.groups.items():
            if rows[0]['split'] != 'val':
                continue
            hardware_memory, hardware_pool = _predict_action_memory(model, data, rows[0], device)
            data.ablation = True
            numerical_memory, numerical_pool = _predict_action_memory(model, data, rows[0], device)
            data.ablation = False
            active = sorted({int(match.group(1)) - 1
                             for name in rows[0]['assignment']
                             if (match := re.search(r'_L(\d+)\}$', name))})
            if not active:
                raise ValueError('No action IDs could be aligned to absolute memory slots')
            left = hardware_memory[:, active]
            right = numerical_memory[:, active]
            denominator = float(left.norm().clamp_min(1e-12))
            shifts.append({'kernel': key[0], 'profile': key[1],
                           'relative_action_embedding_l2_shift': float((left-right).norm()/denominator),
                           'pooled_embedding_l2_shift': float((hardware_pool-numerical_pool).norm())})
    return {
        'hardware_message_enabled_metrics': hardware_metrics,
        'hardware_message_disabled_metrics': numerical_metrics,
        'selected_latency_candidate_changed_contexts': int(changed),
        'validation_contexts': len(hw_contexts),
        'mean_absolute_prediction_delta_by_target': np.mean(prediction_deltas, axis=0).tolist(),
        'embedding_shifts': shifts,
        'elapsed_seconds': time.monotonic() - before,
    }


def run(output: Path, run_output: Path, gpu_name: str, gpu_uuid: str,
        updates=TRAIN_UPDATES, candidates_per_update=CANDIDATES_PER_CONTEXT,
        runtime_limit_seconds=GPU_PHASE_LIMIT_SECONDS):
    if updates < 1 or updates > 100:
        raise ValueError('Pilot update budget must be between 1 and 100')
    if runtime_limit_seconds < 1 or runtime_limit_seconds > GPU_PHASE_LIMIT_SECONDS:
        raise ValueError('GPU runtime limit may not exceed the 20-minute pilot budget')
    run_output = Path(run_output).resolve()
    if run_output.exists() and any(run_output.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty pilot training output: ' + str(run_output))
    run_output.mkdir(parents=True, exist_ok=True)
    _verify_idle_gpu(gpu_name, gpu_uuid)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Pilot requires exactly one isolated CUDA device')
    device = torch.device('cuda:0')
    if gpu_name.casefold() not in torch.cuda.get_device_name(device).casefold():
        raise RuntimeError('CUDA-visible device does not match the selected physical GPU')

    load_started = time.monotonic()
    manifest, payload = _load_cache(output)
    load_seconds = time.monotonic() - load_started
    data = ProbeData(payload, manifest)
    train_groups = {key: rows for key, rows in data.groups.items() if rows[0]['split'] == 'train'}
    val_groups = {key: rows for key, rows in data.groups.items() if rows[0]['split'] == 'val'}
    train_records = [row for rows in train_groups.values() for row in rows]
    mean, std = payload['normalization_mean'], payload['normalization_std']
    rank_scales = payload['ranking_scales']
    expected_contexts = {(kernel, profile, CLOCK_NS)
                         for kernel in TRAIN_KERNELS + VALIDATION_KERNELS for profile in PROFILES}
    if set(train_groups) | set(val_groups) != expected_contexts or set(train_groups) & set(val_groups):
        raise ValueError('Small probe train/validation context split is incomplete or overlapping')
    if any(row['split'] == 'test' for rows in data.groups.values() for row in rows):
        raise ValueError('Test-split records entered the frozen probe cache')
    if any(record['family'] == 'test' for record in train_records):
        raise ValueError('Test-family measurements cannot be used to fit the model')

    training_started = time.monotonic()
    protocol = {
        'schema': SCHEMA, 'cohort_manifest_sha256': manifest['manifest_sha256'],
        'feature_cache_sha256': manifest['feature_cache_sha256'],
        'seed': SEED, 'updates_per_arm': updates,
        'candidates_per_update': candidates_per_update,
        'optimizer': 'AdamW', 'learning_rate': 3e-4, 'weight_decay': 1e-4,
        'rank_weight': 0.1, 'budget_fractions': list(train.BUDGET_FRACTIONS),
        'objectives': list(train.OBJECTIVES), 'gpu_name': torch.cuda.get_device_name(device),
        'gpu_uuid': gpu_uuid, 'tensor_schema': tensors.SCHEMA,
        'graph_archive_sha256': manifest['identity']['graph_release_archive_sha256'],
    }
    protocol_hash = _canonical_hash(protocol)
    report = {'protocol': protocol, 'protocol_sha256': protocol_hash,
              'cache_load_seconds': load_seconds, 'arms': {}, 'status': 'running'}
    checkpointing.atomic_json(run_output / 'protocol.json', report)
    deadline = time.monotonic() + runtime_limit_seconds
    initial_hashes = {}
    traces = {}
    try:
        with _alarm(runtime_limit_seconds):
            for arm in ('numerical', 'hardware'):
                if time.monotonic() >= deadline:
                    raise TimeoutError('20-minute GPU phase budget reached before the next arm')
                _verify_idle_gpu(gpu_name, gpu_uuid)
                random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
                model = train.CandidateQoRModel().to(device)
                initial_hash = train.model_state_sha256(model)
                initial_hashes[arm] = initial_hash
                if len(set(initial_hashes.values())) != 1:
                    raise ValueError('Matched arms do not share identical initial parameter state')
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                trace = []
                updates_log = []
                hardware_grad_norms = []
                arm_started = time.monotonic()
                model.train()
                for step in range(1, updates + 1):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('20-minute GPU phase budget reached during training')
                    keys = sorted(train_groups)
                    group_key = random.choice(keys)
                    group = train_groups[group_key]
                    count = min(candidates_per_update, len(group))
                    sampled = sorted(random.sample(range(len(group)), count))
                    objective = random.choice(train.OBJECTIVES)
                    budget = random.choice(train.BUDGET_FRACTIONS)
                    identity = train._sample_identity(group_key, group, sampled, objective, budget)
                    trace.append(identity)
                    loss, regression, rank, details = train._train_step(
                        model, optimizer, data, group_key, sampled, objective, budget,
                        mean, std, arm, device, rank_weight=0.1,
                        rank_scales=rank_scales, return_details=True)
                    grad_norm = _hardware_relation_gradient_norm(model)
                    hardware_grad_norms.append(grad_norm)
                    if not all(np.isfinite(value) for value in (loss, regression, rank, grad_norm)):
                        raise ValueError('Nonfinite loss or relation gradient in fast probe')
                    updates_log.append({'step': step, 'loss': loss, 'qor_loss': regression,
                                        'ranking_loss': rank, 'details': details,
                                        'hardware_relation_gradient_norm': grad_norm,
                                        'sample_identity': identity})
                traces[arm] = trace
                if time.monotonic() >= deadline:
                    raise TimeoutError('20-minute GPU phase budget reached before validation')
                validation_started = time.monotonic()
                validation = train.evaluate(model, data, val_groups, mean, std, arm, device,
                                            max_candidates=None)
                validation_seconds = time.monotonic() - validation_started
                selected_path = run_output / f'{arm}_probe.pt'
                checkpointing.atomic_torch_save(selected_path, {
                    'schema': train.HEAD_SCHEMA, 'probe_schema': SCHEMA, 'arm': arm,
                    'protocol_sha256': protocol_hash,
                    'cohort_manifest_sha256': manifest['manifest_sha256'],
                    'initial_model_sha256': initial_hash,
                    'model': {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()},
                    'normalization_mean': mean, 'normalization_std': std,
                    'validation_metrics': validation,
                })
                report['arms'][arm] = {
                    'initial_model_sha256': initial_hash,
                    'final_model_sha256': train.model_state_sha256(model),
                    'parameter_count': sum(p.numel() for p in model.parameters()),
                    'updates': updates_log, 'sample_identity_trace': trace,
                    'sample_identity_trace_sha256': train._sample_trace_sha256(trace),
                    'training_loss_mean': float(np.mean([item['loss'] for item in updates_log])),
                    'training_loss_last': updates_log[-1]['loss'],
                    'validation_metrics': validation,
                    'validation_selection_score': train.validation_selection_score(validation),
                    'validation_seconds': validation_seconds,
                    'training_seconds': time.monotonic() - arm_started,
                    'hardware_relation_gradient_norm_min': min(hardware_grad_norms),
                    'hardware_relation_gradient_norm_max': max(hardware_grad_norms),
                    'checkpoint': selected_path.name,
                    'checkpoint_sha256': overlay.file_hash(selected_path),
                }
                checkpointing.atomic_json(run_output / 'protocol.json', report)
                del optimizer, model
                torch.cuda.empty_cache()
            if traces['numerical'] != traces['hardware']:
                raise ValueError('Matched arms sampled different training identities')
            if initial_hashes['numerical'] != initial_hashes['hardware']:
                raise ValueError('Matched arms had different initial parameters')

            hardware_payload = torch.load(run_output / 'hardware_probe.pt', map_location=device,
                                          weights_only=False)
            hardware_model = train.CandidateQoRModel().to(device)
            hardware_model.load_state_dict(hardware_payload['model'])
            ablation = _ablation_summary(hardware_model, data, mean, std, device, deadline)
            report['hardware_message_ablation'] = ablation
            report['elapsed_gpu_phase_seconds'] = time.monotonic() - training_started
            report['status'] = 'completed'
            checkpointing.atomic_json(run_output / 'protocol.json', report)
            checkpointing.atomic_json(run_output / 'status.json', {
                'status': 'completed', 'protocol_sha256': protocol_hash,
                'elapsed_gpu_phase_seconds': report['elapsed_gpu_phase_seconds'],
                'gpu_name': torch.cuda.get_device_name(device), 'gpu_uuid': gpu_uuid,
            })
    except TimeoutError as exc:
        report['status'] = 'resource_limit'; report['stop_reason'] = str(exc)
        report['elapsed_gpu_phase_seconds'] = time.monotonic() - training_started
        checkpointing.atomic_json(run_output / 'protocol.json', report)
        checkpointing.atomic_json(run_output / 'status.json', {'status':'resource_limit',
                                   'reason':str(exc), 'protocol_sha256':protocol_hash})
    except BaseException as exc:
        report['status'] = 'failed'; report['exception'] = type(exc).__name__
        report['message'] = str(exc)
        report['traceback'] = __import__('traceback').format_exc()
        report['elapsed_gpu_phase_seconds'] = time.monotonic() - training_started
        checkpointing.atomic_json(run_output / 'protocol.json', report)
        checkpointing.atomic_json(run_output / 'status.json', {'status':'failed',
                                   'exception':type(exc).__name__, 'message':str(exc),
                                   'protocol_sha256':protocol_hash})
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Authenticate and freeze the small cohort and input cache')
    prep.add_argument('--release-dir', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--preparation-limit-seconds', type=int, default=PREPARATION_LIMIT_SECONDS)
    gpu = sub.add_parser('run', help='Run two matched short training arms on one idle GPU')
    gpu.add_argument('--output', type=Path, required=True, help='Prepared cohort/cache directory')
    gpu.add_argument('--run-output', type=Path, required=True, help='New empty training output directory')
    gpu.add_argument('--gpu-name', required=True)
    gpu.add_argument('--gpu-uuid', required=True)
    gpu.add_argument('--updates', type=int, default=TRAIN_UPDATES)
    gpu.add_argument('--candidates-per-update', type=int, default=CANDIDATES_PER_CONTEXT)
    gpu.add_argument('--gpu-limit-seconds', type=int, default=GPU_PHASE_LIMIT_SECONDS)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare(args.release_dir, args.output, args.preparation_limit_seconds)
        print(json.dumps({'status':'prepared','manifest_sha256':manifest['manifest_sha256'],
                          'timings_seconds':manifest['timings_seconds']}, indent=2))
    else:
        report = run(args.output, args.run_output, args.gpu_name, args.gpu_uuid,
                     args.updates, args.candidates_per_update, args.gpu_limit_seconds)
        print(json.dumps({'status':report['status'], 'protocol_sha256':report['protocol_sha256'],
                          'elapsed_gpu_phase_seconds':report.get('elapsed_gpu_phase_seconds')}, indent=2))


if __name__ == '__main__':
    main()
