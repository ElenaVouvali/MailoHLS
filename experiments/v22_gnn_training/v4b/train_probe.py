"""Bounded V4b full-cohort supervised preference trainer.

The entry point is deliberately CUDA-only. It consumes the frozen QoR-blind
cohort manifest and writes to a new output directory. No test kernels are
loaded. This is experimental research code, not a production checkpoint path.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.v22_gnn_training import train, checkpointing
from experiments.v22_gnn_training.v4b.models import (
    ActionDifferenceComparator, NoGraphControl, TargetFilmV4bModel,
    V4bGraphModel, hybrid_loss,
)
from experiments.v22_gnn_training.v4b.preferences import preference_pairs


ARMS = ('no_graph', 'compiler', 'target_film', 'hardware', 'compiler_difference')
BUDGETS = (1.0, 0.75, 0.5, 0.25, 0.1)
PAIR_WEIGHT = 0.1
ASSIGNED_GPU_UUIDS = {
    'A2': 'GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35',
    # Explicitly selectable for a user-authorized run when the V100S is idle.
    # The default remains the A2; there is no automatic fallback.
    'V100S': 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb',
}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_isolated_gpu(gpu_name, gpu_uuid, minimum_free_mib):
    """Refuse unrecognized, occupied, or insufficiently free selected GPUs."""
    from experiments.v22_gnn_training.launch import compute_processes, gpu_inventory
    if gpu_name not in ASSIGNED_GPU_UUIDS:
        raise RuntimeError(f'GPU name must be one of {sorted(ASSIGNED_GPU_UUIDS)}')
    devices = [gpu for gpu in gpu_inventory() if gpu_name.casefold() in gpu['name'].casefold()]
    if len(devices) != 1:
        raise RuntimeError(f'Expected exactly one {gpu_name}; found {devices}')
    gpu = devices[0]
    if gpu['uuid'] != gpu_uuid or gpu_uuid != ASSIGNED_GPU_UUIDS[gpu_name]:
        raise RuntimeError(f'Refusing unapproved GPU UUID for {gpu_name}: requested={gpu_uuid}, found={gpu["uuid"]}')
    active = [row for row in compute_processes() if row['uuid'] == gpu_uuid]
    if active:
        raise RuntimeError(f'Refusing to share occupied {gpu_name}: {active}')
    if gpu['free_mib'] < minimum_free_mib or gpu['utilization_percent'] > 15:
        raise RuntimeError(f'{gpu_name} does not meet free-memory/utilization guard: {gpu}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the verified GPU UUID before Python starts')
    if os.environ.get('SALT_HLS_GPU_UUID') != gpu_uuid:
        raise RuntimeError('SALT_HLS_GPU_UUID must identify the verified GPU')
    return gpu


def _families(split):
    result = {}
    for family in split['train_families'] + split['val_families']:
        normalized = ''.join(ch for ch in family.lower() if ch.isalnum())
        for kernel in split['train_kernels'] + split['val_kernels']:
            if ''.join(ch for ch in kernel.lower() if ch.isalnum()).startswith(normalized):
                result[kernel] = family
    return result


def _cached_candidate_inputs(data, row, enriched_graph_cache, candidate_cache,
                            analysis_cache_by_kernel):
    """Reuse authenticated target overlays while preparing candidate features.

    The shared PreparedData.candidate_inputs path rebuilds the enriched graph
    for every candidate. Candidate-demand tensorization also returns only
    source/candidate features; its physical-target metadata is discarded by
    the tensor API. Therefore V4b reuses the authenticated program overlay
    once per kernel and the candidate vector/demand once per complete
    assignment, across physical targets. This preserves the model inputs while
    avoiding repeated compiler analysis.
    """
    candidate_key = (row['kernel'], row['config_hash'])
    if candidate_key in candidate_cache:
        return candidate_cache[candidate_key]
    kernel = row['kernel']
    if kernel not in enriched_graph_cache:
        base, authentication = data.graphs[row['kernel']]
        enriched_graph_cache[kernel] = train.overlay.enrich(
            base, row['profile'], row['clock'], authentication=authentication)
    enriched = enriched_graph_cache[kernel]
    actions = data.actions[kernel]
    analysis_cache = analysis_cache_by_kernel.setdefault(kernel, {})
    demand, _ = train.candidate_demand_tensor(
        enriched, row['assignment'], actions, analysis_cache=analysis_cache)
    candidate = train.candidate_vector(row['assignment'], actions)
    candidate_cache[candidate_key] = (candidate, demand)
    return candidate_cache[candidate_key]


def prepare_small_cohort(data, manifest_path, device):
    """Materialize only selected <=16 candidates/context; reuse each tensor."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('schema') != 'salt-hls-v4b-full-cohort-action-aligned-preference-v1-cohort-manifest':
        raise ValueError('Unsupported V4b cohort manifest')
    manifest_payload = dict(manifest)
    manifest_hash = manifest_payload.pop('manifest_sha256', None)
    if manifest_hash != _json_sha(manifest_payload):
        raise ValueError('V4b cohort manifest self-hash does not match')
    eligibility_path = Path(manifest_path).with_name('eligibility.json')
    eligibility = json.loads(eligibility_path.read_text())
    if _json_sha(eligibility) != manifest.get('eligibility_manifest_sha256'):
        raise ValueError('V4b eligibility manifest hash mismatch')
    if manifest.get('test_kernels_loaded') != [] or manifest.get('family_disjoint') is not True:
        raise ValueError('Cohort manifest violates held-out test/family contract')
    if (manifest.get('dataset_sha256') != data.audit['dataset']['dataset_sha256'] or
            manifest.get('family_split_sha256') != data.audit['dataset']['split_sha256'] or
            manifest.get('graph_archive_sha256') != data.audit['graph_release']['archive_sha256']):
        raise ValueError('V4b manifest dataset/split/graph identities do not match authenticated inputs')
    demand_hash = hashlib.sha256(
        (Path(__file__).resolve().parents[3] / 'GNN_branch/action_demand_v22.py').read_bytes()
    ).hexdigest()
    if manifest.get('action_demand_implementation_sha256') != demand_hash:
        raise ValueError('V4b manifest candidate-demand implementation hash does not match')
    if set(manifest.get('training_kernels', ())) & set(manifest.get('validation_kernels', ())):
        raise ValueError('V4b training and validation kernel lists overlap')
    indexed = defaultdict(dict)
    for row in data.records:
        indexed[(row['kernel'], row['profile'], float(row['clock']))][row['config_hash']] = row
    groups = []
    graph_cache, inputs, enriched_graph_cache, candidate_cache = {}, {}, {}, {}
    analysis_cache_by_kernel = {}
    from experiments.v22_gnn_training.directive_preference_probe import capability_vector
    profile_features = {}
    split = data.kernel_split
    if any(split.get(kernel) != 'train' for kernel in manifest['training_kernels']):
        raise ValueError('V4b training kernel inventory crosses authenticated split')
    if any(split.get(kernel) != 'val' for kernel in manifest['validation_kernels']):
        raise ValueError('V4b validation kernel inventory crosses authenticated split')
    families = _families(data.split)
    start = time.perf_counter()
    for context_id, context in sorted(manifest['contexts'].items()):
        kernel, profile, clock = context['kernel'], context['device'], float(context['clock_ns'])
        if split.get(kernel) not in ('train', 'val'):
            raise ValueError(f'Unexpected test/non-split kernel in cohort: {kernel}')
        available = indexed.get((kernel, profile, clock), {})
        records = [available[h] for h in context['selected_assignment_hashes'] if h in available]
        if len(records) != context['selected_candidate_count']:
            raise ValueError(f'Manifest assignment mismatch for {context_id}')
        if len({r['config_hash'] for r in records}) != len(records):
            raise ValueError(f'Duplicate assignment in context {context_id}')
        if not records:
            continue
        key = (kernel, profile, clock)
        profile_features[(profile, clock)] = capability_vector(data.registry, profile, clock)
        if key not in graph_cache:
            graph_cache[key] = data.graph_tensor(records[0], 1.0, 'hardware').cpu()
        for row in records:
            identity = (kernel, profile, clock, row['config_hash'])
            if identity not in inputs:
                candidate, demand = _cached_candidate_inputs(
                    data, row, enriched_graph_cache, candidate_cache,
                    analysis_cache_by_kernel)
                inputs[identity] = (candidate.cpu(), demand.cpu())
        print(f'V4b prepare: {len(groups)+1}/{len(manifest["contexts"])} {context_id}', flush=True)
        groups.append({'key': context_id, 'kernel': kernel, 'profile': profile,
                       'clock': clock, 'family': families.get(kernel, kernel),
                       'split': split[kernel], 'rows': records,
                       'identities': [(kernel, profile, clock, r['config_hash']) for r in records]})
    for group in groups:
        group['budget_pairs'] = {}
        for fraction in BUDGETS:
            pairs, counts = preference_pairs(group['rows'], fraction)
            group['budget_pairs'][fraction] = pairs
            group.setdefault('pair_audit', {})[str(fraction)] = counts
    train_rows = [r for g in groups if g['split'] == 'train' for r in g['rows']]
    mean, std = train._fit_normalization(train_rows)
    if (std <= 0).any() or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError('Training-only physical normalization is invalid')
    return {'groups': groups, 'graph_cache': graph_cache, 'inputs': inputs,
            'profile_features': profile_features,
            'profile_feature_dim': int(next(iter(profile_features.values())).numel()),
            'mean': mean, 'std': std, 'manifest': manifest,
            'preparation_seconds': time.perf_counter() - start,
            'selected_measurements': sum(len(g['rows']) for g in groups)}


def _pair_schedule(groups, updates, seed):
    """One family-balanced sampled pair trace shared by all model arms."""
    by_family = defaultdict(list)
    for gi, group in enumerate(groups):
        if group['split'] != 'train':
            continue
        for fraction in BUDGETS:
            pairs = group['budget_pairs'].get(fraction, [])
            if pairs:
                by_family[group['family']].append((gi, fraction, pairs))
    if not by_family:
        raise ValueError('No eligible training preference pairs in frozen cohort')
    rng = random.Random(seed)
    families = sorted(by_family)
    schedule = []
    for _ in range(updates):
        family = rng.choice(families)
        gi, fraction, pairs = rng.choice(by_family[family])
        pair_index = rng.randrange(len(pairs))
        schedule.append((gi, fraction, pair_index))
    return schedule


def sample_coverage(groups, schedule):
    from collections import Counter
    candidates, pairs, contexts = set(), set(), set()
    families, reasons = Counter(), Counter()
    for gi, fraction, pi in schedule:
        group = groups[gi]
        a, b, reason = group['budget_pairs'][fraction][pi]
        ah, bh = [group['rows'][i]['config_hash'] for i in (a, b)]
        candidates.update(((group['kernel'], ah), (group['kernel'], bh)))
        pairs.add((group['key'], fraction, ah, bh))
        contexts.add(group['key']); families[group['family']] += 1; reasons[reason] += 1
    return {'distinct_kernel_target_contexts': len(contexts), 'distinct_complete_assignments': len(candidates),
            'distinct_budget_pair_identities': len(pairs), 'per_family_updates': dict(families),
            'pair_reason_updates': dict(reasons), 'updates': len(schedule)}


def _initial_models(seed, profile_feature_dim):
    torch.manual_seed(seed)
    compiler = V4bGraphModel(hardware_messages=False)
    initial = {key: value.detach().clone() for key, value in compiler.state_dict().items()}
    hardware = V4bGraphModel(hardware_messages=True)
    hardware.load_state_dict(initial)
    torch.manual_seed(seed + 1)
    target_film = TargetFilmV4bModel(profile_feature_dim)
    target_film.load_state_dict(initial, strict=False)
    torch.manual_seed(seed + 1)
    difference_base = V4bGraphModel(hardware_messages=False)
    difference_base.load_state_dict(initial)
    difference = ActionDifferenceComparator()
    torch.manual_seed(seed + 2)
    no_graph = NoGraphControl()
    return {'no_graph': no_graph, 'compiler': compiler, 'target_film': target_film,
            'hardware': hardware,
            'compiler_difference': {'base': difference_base, 'compare': difference}}


def _all_parameters(model):
    if isinstance(model, dict):
        return list(model['base'].parameters()) + list(model['compare'].parameters())
    return list(model.parameters())


def _forward_pair(arm, model, graph, candidates, demands, fraction, mean, std,
                  profile_features=None):
    if arm == 'no_graph':
        output = model(graph, candidates, demands, fraction)
        logits = output['preference_score'][1:2] - output['preference_score'][0:1]
    elif arm in ('compiler', 'hardware'):
        memory, pooled = model.encode(graph)
        output = model.from_memory(graph, memory, pooled, candidates, demands, fraction)
        logits = output['preference_score'][1:2] - output['preference_score'][0:1]
    elif arm == 'target_film':
        memory, pooled = model.encode(graph, profile_features)
        output = model.from_memory(graph, memory, pooled, candidates, demands, fraction)
        logits = output['preference_score'][1:2] - output['preference_score'][0:1]
    else:
        base = model['base']
        memory, pooled = base.encode(graph)
        output = base.from_memory(graph, memory, pooled, candidates, demands, fraction)
        logits, _ = model['compare'](
            graph, memory, candidates[0:1], demands[0:1],
            candidates[1:2], demands[1:2], fraction)
    return output['qor_z'], logits


def _model_state(model):
    if isinstance(model, dict):
        return {name: {key: value.detach().cpu().clone()
                       for key, value in module.state_dict().items()}
                for name, module in model.items()}
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_model_state(model, state):
    if isinstance(model, dict):
        for name, module in model.items():
            module.load_state_dict(state[name])
    else:
        model.load_state_dict(state)


def evaluate_arm(arm, model, prepared, device, mode='correct'):
    from .evaluation import evaluate
    return evaluate(arm, model, prepared, device, mode)


def _physical(raw_z, mean, std):
    return torch.expm1(raw_z * std.to(raw_z.device) + mean.to(raw_z.device)).clamp_min(0)


def train_arm(arm, model, prepared, schedule, device, validation_steps, lr=3e-4,
              weight_decay=1e-4, pair_weight=PAIR_WEIGHT, state_path=None,
              fingerprint=None, stop_requested=lambda: False):
    groups, inputs = prepared['groups'], prepared['inputs']
    mean, std = prepared['mean'].to(device), prepared['std'].to(device)
    optimizer = torch.optim.AdamW(_all_parameters(model), lr=lr, weight_decay=weight_decay)
    history, checkpoints = [], {}
    start = time.perf_counter()
    start_step = 0
    if state_path is not None and Path(state_path).exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        checkpointing.verify_fingerprint(state, fingerprint)
        _load_model_state(model, state['model'])
        optimizer.load_state_dict(state['optimizer'])
        checkpointing.restore_rng_state(state['rng'])
        start_step, history, checkpoints = state['step'], state['history'], state['checkpoints']
    def save(step):
        if state_path is not None:
            checkpointing.atomic_torch_save(state_path, {
                'protocol_fingerprint': fingerprint, 'model': _model_state(model),
                'optimizer': optimizer.state_dict(), 'rng': checkpointing.capture_rng_state(),
                'step': step, 'history': history, 'checkpoints': checkpoints})
    for step, (gi, fraction, pair_index) in enumerate(schedule, 1):
        if step <= start_step:
            continue
        if stop_requested():
            save(step - 1)
            if state_path is not None:
                checkpointing.atomic_json(Path(state_path).parent / 'status.json', {'status': 'interrupted', 'arm': arm, 'completed_step': step - 1})
            raise InterruptedError('V4b interrupted at atomic update boundary')
        for module in (model.values() if isinstance(model, dict) else [model]):
            module.train()
        group = groups[gi]
        preferred, worse, _reason = group['budget_pairs'][fraction][pair_index]
        rows = (group['rows'][preferred], group['rows'][worse])
        graph = prepared['graph_cache'][(group['kernel'], group['profile'], group['clock'])].clone().to(device)
        features = [inputs[identity] for identity in (
            group['identities'][preferred], group['identities'][worse])]
        candidates = torch.stack([x[0] for x in features]).to(device)
        demands = torch.stack([x[1] for x in features]).to(device)
        target = torch.tensor([r['transformed'] for r in rows], dtype=torch.float32, device=device)
        target = (target - mean) / std
        qor, logits = _forward_pair(arm, model, graph, candidates, demands,
                                   fraction, mean, std,
                                   prepared['profile_features'][(group['profile'], group['clock'])].clone().to(device))
        loss, parts = hybrid_loss(qor, target, logits, [True], pair_weight)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite {arm} loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(_all_parameters(model), 1.0)
        if not torch.isfinite(grad):
            raise ValueError(f'Nonfinite {arm} gradient at step {step}')
        optimizer.step()
        history.append({'step': step, 'loss': float(loss.detach()),
                        'physical_qor_loss': float(parts['physical_qor']),
                        'pairwise_loss': float(parts['pairwise']),
                        'gradient_norm_preclip': float(grad),
                        'gradient_clipped': bool(float(grad) > 1.0),
                        'context': group['key'], 'budget_fraction': fraction,
                        'preferred_hash': rows[0]['config_hash'],
                        'worse_hash': rows[1]['config_hash']})
        if step in validation_steps:
            validation = evaluate_arm(arm, model, prepared, device)
            checkpoints[step] = {'state': _model_state(model), 'metrics': validation}
            history[-1]['validation'] = validation
        if step % 25 == 0 or step in validation_steps or stop_requested():
            save(step)
            print(f'V4b {arm}: update {step}/{len(schedule)}, loss={float(loss.detach()):.5f}', flush=True)
    save(len(schedule))
    viable = [(step, row) for step, row in checkpoints.items()
              if row['metrics']['selection_score'] is not None]
    if not viable:
        raise ValueError(f'No validation checkpoint had a defined feasible-selection score for {arm}')
    selected_step, selected = min(viable, key=lambda item: (
        item[1]['metrics']['selection_score'],
        -(item[1]['metrics']['validation_preference_accuracy'] or 0.0), item[0]))
    _load_model_state(model, selected['state'])
    return {'history': history, 'checkpoints': checkpoints,
            'selected_step': selected_step, 'selected_validation': selected['metrics'],
            'elapsed_seconds': time.perf_counter() - start,
            'parameter_count': sum(p.numel() for p in _all_parameters(model))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--authenticated-audit', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true', help='CPU feature preparation only; no training')
    parser.add_argument('--prepared-cache', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--updates', type=int, default=1500)
    parser.add_argument('--validate-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpu-name', choices=tuple(ASSIGNED_GPU_UUIDS), default='A2')
    parser.add_argument('--gpu-uuid', default=ASSIGNED_GPU_UUIDS['A2'])
    parser.add_argument('--minimum-free-mib', type=int, default=6000)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite V4b run directory: {output}')
    if args.resume and not (output / 'protocol.json').is_file():
        raise ValueError('Resume requires an existing authenticated V4b run')
    if args.resume and (output / 'status.json').exists() and json.loads((output / 'status.json').read_text()).get('status') == 'complete':
        raise ValueError('Refusing to resume or overwrite a completed V4b run')
    if not args.prepare_only:
        require_isolated_gpu(args.gpu_name, args.gpu_uuid, args.minimum_free_mib)
    if not args.prepare_only and not torch.cuda.is_available():
        raise RuntimeError('V4b training requires the independently available A2; CUDA is unavailable')
    if args.updates < 1 or args.validate_every < 1 or args.updates % args.validate_every:
        raise ValueError('updates must be positive and divisible by validate-every')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from experiments.v22_gnn_training.train import PreparedData
    manifest = json.loads(Path(args.manifest).read_text())
    kernels = sorted({row['kernel'] for row in manifest['contexts'].values()})
    data = PreparedData(args.release_dir, output / 'unused-audit.json',
                        selected_kernels=kernels,
                        authenticated_audit_path=args.authenticated_audit)
    print('V4b authenticated data loaded; preparing frozen candidate pools', flush=True)
    from experiments.v22_gnn_training.fast_probe import _source_hashes
    cache_identity = {'features': _source_hashes(), 'manifest': _json_sha(manifest),
                      'preparation': hashlib.sha256((__import__('inspect').getsource(prepare_small_cohort) + __import__('inspect').getsource(_cached_candidate_inputs)).encode()).hexdigest(),
                      'torch': torch.__version__,
                      'preferences': hashlib.sha256(Path(__file__).with_name('preferences.py').read_bytes()).hexdigest()}
    if args.prepared_cache and args.prepared_cache.exists():
        cached = torch.load(args.prepared_cache, map_location='cpu', weights_only=False)
        if cached['identity'] != cache_identity:
            raise ValueError('Prepared feature cache provenance changed')
        prepared = cached['prepared']
    else:
        prepared = prepare_small_cohort(data, args.manifest, torch.device('cpu'))
        if args.prepared_cache:
            checkpointing.atomic_torch_save(args.prepared_cache, {'identity': cache_identity, 'prepared': prepared})
    if args.prepare_only:
        print('CPU-only preparation completed; no training or GPU initialization', flush=True)
        return
    print(f'V4b preparation complete in {prepared["preparation_seconds"]:.1f}s', flush=True)
    schedule = _pair_schedule(prepared['groups'], args.updates, args.seed)
    schedule_hash = _json_sha([
        [prepared['groups'][gi]['key'], fraction, pair_index]
        for gi, fraction, pair_index in schedule])
    require_isolated_gpu(args.gpu_name, args.gpu_uuid, args.minimum_free_mib)
    output.mkdir(parents=True, exist_ok=args.resume)
    checkpointing.atomic_json(output / 'status.json', {'status': 'running'})
    source_files = sorted(p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_'))
    source_files += [Path(__file__).resolve().parents[3] / name for name in
                     __import__('experiments.v22_gnn_training.fast_probe', fromlist=['FEATURE_FILES']).FEATURE_FILES]
    protocol = {'version': 'v4b-evaluation-resume-v2', 'seed': args.seed,
                'updates': args.updates, 'validate_every': args.validate_every,
                'pair_weight': PAIR_WEIGHT, 'lr': 3e-4, 'weight_decay': 1e-4,
                'manifest': _json_sha(prepared['manifest']), 'schedule': schedule_hash,
                'source_sha256': {str(path.relative_to(Path(__file__).resolve().parents[3])):
                                 hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
                'torch': torch.__version__, 'gpu_uuid': args.gpu_uuid,
                'checkpoint_criterion': 'family macro selection failure + feasible top1 regret; pair accuracy tie break'}
    fingerprint = _json_sha(protocol)
    protocol_path = output / 'protocol.json'
    if args.resume:
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Resume protocol changed')
    else:
        checkpointing.atomic_json(protocol_path, protocol)
    stopped = []
    def on_signal(signum, frame):
        stopped.append(signum)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    initial = _initial_models(args.seed, prepared['profile_feature_dim'])
    initial_hashes = {}
    training = {}
    validation_steps = tuple(range(args.validate_every, args.updates + 1, args.validate_every))
    for arm in ARMS:
        model = initial[arm]
        if args.resume and (output / f'{arm}_selected.pt').exists():
            training[arm] = json.loads((output / f'{arm}_report.json').read_text())
            initial_hashes[arm] = train.model_state_sha256(model['base'] if isinstance(model, dict) else model)
            continue
        for module in (model.values() if isinstance(model, dict) else [model]):
            module.to('cuda')
        torch.cuda.reset_peak_memory_stats()
        initial_hashes[arm] = (
            train.model_state_sha256(model['base']) if isinstance(model, dict)
            else train.model_state_sha256(model))
        # All arms consume the same fixed family-balanced sampled pair trace.
        training[arm] = train_arm(arm, model, prepared, schedule,
                                  torch.device('cuda'), validation_steps,
                                  state_path=output / f'{arm}_training_state.pt',
                                  fingerprint=fingerprint, stop_requested=lambda: bool(stopped))
        training[arm]['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        from .evaluation import controlled_action_probe
        training[arm]['controlled_action_probe'] = controlled_action_probe(model, prepared, torch.device('cuda'))
        training[arm]['ablations'] = {}
        if arm != 'no_graph':
            for mode in ('zero', 'shuffled', 'global_mean', 'target_mismatched'):
                training[arm]['ablations'][mode] = evaluate_arm(arm, model, prepared, torch.device('cuda'), mode)
        checkpointing.atomic_json(output / f'{arm}_report.json', {k: v for k, v in training[arm].items() if k != 'checkpoints'})
        checkpointing.atomic_torch_save(output / f'{arm}_selected.pt', {'state': _model_state(model), 'updates': args.updates,
                    'schema': 'salt-hls-v4b-full-cohort-action-aligned-preference-v2',
                    'protocol_fingerprint': fingerprint, 'selected_step': training[arm]['selected_step'],
                    'arm': arm, 'seed': args.seed, 'manifest_sha256': _json_sha(prepared['manifest']),
                    'sample_trace_sha256': schedule_hash,
                    'selected_validation': training[arm]['selected_validation']},
                   )
        for module in (model.values() if isinstance(model, dict) else [model]):
            module.cpu()
        torch.cuda.empty_cache()
    receipt = {'schema': 'salt-hls-v4b-full-cohort-action-aligned-preference-v2-run',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'manifest_sha256': _json_sha(prepared['manifest']),
               'sample_trace_sha256': schedule_hash, 'seed': args.seed,
               'updates_per_arm': args.updates, 'validation_steps': validation_steps,
               'arms': list(ARMS), 'initial_encoder_hashes': initial_hashes,
               'training': {name: {key: value for key, value in report.items()
                                   if key != 'checkpoints'} for name, report in training.items()},
               'preparation_seconds': prepared['preparation_seconds'],
               'selected_measurements': prepared['selected_measurements'],
               'test_kernels_loaded': [], 'sample_coverage': sample_coverage(prepared['groups'], schedule), 'protocol_fingerprint': fingerprint, 'checkpoint_selection': protocol['checkpoint_criterion']}
    checkpointing.atomic_json(output / 'status.json', {'status': 'complete', 'protocol_fingerprint': fingerprint})
    (output / 'run_receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    (output / 'training_history.json').write_text(json.dumps(
        {name: report['history'] for name, report in training.items()},
        sort_keys=True, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
