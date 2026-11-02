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
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.v22_gnn_training import train
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
        raise RuntimeError(f'A2 does not meet free-memory/utilization guard: {gpu}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the verified A2 UUID before Python starts')
    if os.environ.get('SALT_HLS_GPU_UUID') != gpu_uuid:
        raise RuntimeError('SALT_HLS_GPU_UUID must identify the verified A2')
    return gpu


def _families(split):
    result = {}
    for family in split['train_families'] + split['val_families']:
        normalized = ''.join(ch for ch in family.lower() if ch.isalnum())
        for kernel in split['train_kernels'] + split['val_kernels']:
            if ''.join(ch for ch in kernel.lower() if ch.isalnum()).startswith(normalized):
                result[kernel] = family
    return result


def _cached_candidate_inputs(data, row, enriched_graph_cache):
    """Reuse authenticated target overlays while preparing candidate features.

    The shared PreparedData.candidate_inputs path rebuilds the enriched graph
    for every candidate. V4b only changes directive assignment within a
    kernel/device/clock context, so the authenticated overlay is identical
    across those candidates and can be safely reused. Candidate demand itself
    is still computed once per complete assignment and target context.
    """
    context = (row['kernel'], row['profile'], float(row['clock']))
    if context not in enriched_graph_cache:
        base, authentication = data.graphs[row['kernel']]
        enriched_graph_cache[context] = train.overlay.enrich(
            base, row['profile'], row['clock'], authentication=authentication)
    enriched = enriched_graph_cache[context]
    actions = data.actions[row['kernel']]
    demand, _ = train.candidate_demand_tensor(enriched, row['assignment'], actions)
    candidate = train.candidate_vector(row['assignment'], actions)
    return candidate, demand


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
    if set(manifest.get('training_kernels', ())) & set(manifest.get('validation_kernels', ())):
        raise ValueError('V4b training and validation kernel lists overlap')
    indexed = defaultdict(dict)
    for row in data.records:
        indexed[(row['kernel'], row['profile'], float(row['clock']))][row['config_hash']] = row
    groups = []
    graph_cache, inputs, enriched_graph_cache = {}, {}, {}
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
                    data, row, enriched_graph_cache)
                inputs[identity] = (candidate.cpu(), demand.cpu())
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


def evaluate_arm(arm, model, prepared, device):
    """Validation-only pair accuracy and feasibility-aware ADP selection."""
    model.eval()
    groups, inputs = prepared['groups'], prepared['inputs']
    mean, std = prepared['mean'].to(device), prepared['std'].to(device)
    pair_correct = pair_total = 0
    family_selection = defaultdict(list)
    selected_feasible = false_feasible = no_prediction = no_reference = 0
    regrets, top3_regrets = [], []
    top3_success = top3_failure = 0
    with torch.no_grad():
        for group in groups:
            if group['split'] != 'val':
                continue
            graph = prepared['graph_cache'][(group['kernel'], group['profile'], group['clock'])].to(device)
            data = [inputs[key] for key in group['identities']]
            candidates = torch.stack([row[0] for row in data]).to(device)
            demands = torch.stack([row[1] for row in data]).to(device)
            if arm == 'no_graph':
                output = model(graph, candidates, demands, 1.0)
            else:
                if isinstance(model, dict):
                    memory, pooled = model['base'].encode(graph)
                elif arm == 'target_film':
                    memory, pooled = model.encode(
                        graph, prepared['profile_features'][(group['profile'], group['clock'])].to(device))
                else:
                    memory, pooled = model.encode(graph)
                base = model['base'] if isinstance(model, dict) else model
                output = base.from_memory(graph, memory, pooled, candidates, demands, 1.0)
            resources = _physical(output['qor_z'], mean, std)[:, 1:]
            capacities = torch.tensor(group['rows'][0]['capacities'], device=device)
            measured_resources = np.asarray([row['physical'][1:] for row in group['rows']])
            adp = np.asarray([row['physical'][0] * _area(row) for row in group['rows']])
            for fraction in BUDGETS:
                if arm == 'no_graph':
                    budget_output = model(graph, candidates, demands, fraction)
                elif isinstance(model, dict):
                    budget_output = output
                else:
                    budget_output = base.from_memory(graph, memory, pooled, candidates, demands, fraction)
                pairs = group['budget_pairs'][fraction]
                if not pairs:
                    continue
                pair_logits = []
                for preferred, worse, _ in pairs:
                    if isinstance(model, dict):
                        logit, _ = model['compare'](
                            graph, memory, candidates[preferred:preferred+1], demands[preferred:preferred+1],
                            candidates[worse:worse+1], demands[worse:worse+1], fraction)
                        pair_logits.append(float(logit.item()))
                    else:
                        scores = budget_output['preference_score']
                        pair_logits.append(float((scores[worse] - scores[preferred]).item()))
                pair_correct += sum(value > 0 for value in pair_logits)
                pair_total += len(pair_logits)
                predicted_feasible = (resources <= capacities * fraction).all(-1).cpu().numpy()
                measured_feasible = np.all(measured_resources <= np.asarray(group['rows'][0]['capacities'])[None, :] * fraction, axis=1)
                candidates_feasible = np.flatnonzero(predicted_feasible)
                reference = np.flatnonzero(measured_feasible)
                if not len(reference):
                    no_reference += 1
                    continue
                if not len(candidates_feasible):
                    no_prediction += 1
                    top3_failure += 1
                    family_selection[group['family']].append((False, None))
                    continue
                if isinstance(model, dict):
                    scores = torch.zeros((len(group['rows']), len(group['rows'])), device=device)
                    for i in candidates_feasible:
                        for j in candidates_feasible:
                            if i < j:
                                value, _ = model['compare'](
                                    graph, memory, candidates[i:i+1], demands[i:i+1],
                                    candidates[j:j+1], demands[j:j+1], fraction)
                                scores[i, j] = value[0]
                                scores[j, i] = -value[0]
                    win_count = (scores[candidates_feasible][:, candidates_feasible] > 0).sum(1)
                    order_local = sorted(range(len(candidates_feasible)),
                                         key=lambda n: (-int(win_count[n]),
                                             group['rows'][candidates_feasible[n]]['config_hash']))
                    ranked = [int(candidates_feasible[n]) for n in order_local]
                else:
                    point_scores = budget_output['preference_score'][candidates_feasible]
                    ranked = [int(candidates_feasible[n]) for n in
                              torch.argsort(point_scores, stable=True).cpu().tolist()]
                chosen = ranked[0]
                shortlist = ranked[:3]
                shortlist_feasible = [index for index in shortlist if measured_feasible[index]]
                if shortlist_feasible:
                    top3_success += 1
                    top3_regrets.append(float((min(adp[index] for index in shortlist_feasible) -
                                               adp[reference].min()) /
                                              max(abs(adp[reference].min()), 1e-12)))
                else:
                    top3_failure += 1
                is_feasible = bool(measured_feasible[chosen])
                if is_feasible:
                    selected_feasible += 1
                    regret = float((adp[chosen] - adp[reference].min()) / max(abs(adp[reference].min()), 1e-12))
                    regrets.append(regret)
                    family_selection[group['family']].append((True, regret))
                else:
                    false_feasible += 1
                    family_selection[group['family']].append((False, None))
    success_contexts = selected_feasible + false_feasible + no_prediction
    failure_rate = ((false_feasible + no_prediction) / success_contexts
                    if success_contexts else None)
    selection_score = (failure_rate + (float(np.mean(regrets)) if regrets else 1.0)
                       if failure_rate is not None else None)
    return {'validation_preference_accuracy': pair_correct / pair_total if pair_total else None,
            'meaningful_pair_count': pair_total,
            'selected_measured_feasible': selected_feasible,
            'false_feasible_selections': false_feasible,
            'no_predicted_feasible_candidate': no_prediction,
            'no_measured_feasible_reference': no_reference,
            'feasible_top1_regret_macro': float(np.mean(regrets)) if regrets else None,
            'feasible_top3_success_count': top3_success,
            'feasible_top3_failure_count': top3_failure,
            'feasible_top3_regret_macro': float(np.mean(top3_regrets)) if top3_regrets else None,
            'selection_failure_rate': failure_rate,
            'selection_score': selection_score,
            'family_contexts': {family: len(values) for family, values in family_selection.items()}}


def _physical(raw_z, mean, std):
    return torch.expm1(raw_z * std.to(raw_z.device) + mean.to(raw_z.device)).clamp_min(0)


def train_arm(arm, model, prepared, schedule, device, validation_steps, lr=3e-4,
              weight_decay=1e-4, pair_weight=PAIR_WEIGHT):
    groups, inputs = prepared['groups'], prepared['inputs']
    mean, std = prepared['mean'].to(device), prepared['std'].to(device)
    optimizer = torch.optim.AdamW(_all_parameters(model), lr=lr, weight_decay=weight_decay)
    history, checkpoints = [], {}
    start = time.perf_counter()
    for step, (gi, fraction, pair_index) in enumerate(schedule, 1):
        group = groups[gi]
        preferred, worse, _reason = group['budget_pairs'][fraction][pair_index]
        rows = (group['rows'][preferred], group['rows'][worse])
        graph = prepared['graph_cache'][(group['kernel'], group['profile'], group['clock'])].to(device)
        features = [inputs[identity] for identity in (
            group['identities'][preferred], group['identities'][worse])]
        candidates = torch.stack([x[0] for x in features]).to(device)
        demands = torch.stack([x[1] for x in features]).to(device)
        target = torch.tensor([r['transformed'] for r in rows], dtype=torch.float32, device=device)
        target = (target - mean) / std
        qor, logits = _forward_pair(arm, model, graph, candidates, demands,
                                   fraction, mean, std,
                                   prepared['profile_features'][(group['profile'], group['clock'])].to(device))
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
    parser.add_argument('--updates', type=int, default=1500)
    parser.add_argument('--validate-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpu-name', choices=tuple(ASSIGNED_GPU_UUIDS), default='A2')
    parser.add_argument('--gpu-uuid', default=ASSIGNED_GPU_UUIDS['A2'])
    parser.add_argument('--minimum-free-mib', type=int, default=6000)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite V4b run directory: {output}')
    require_isolated_gpu(args.gpu_name, args.gpu_uuid, args.minimum_free_mib)
    if not torch.cuda.is_available():
        raise RuntimeError('V4b training requires the independently available A2; CUDA is unavailable')
    if args.updates < 1 or args.validate_every < 1 or args.updates % args.validate_every:
        raise ValueError('updates must be positive and divisible by validate-every')
    from experiments.v22_gnn_training.train import PreparedData
    manifest = json.loads(Path(args.manifest).read_text())
    kernels = sorted({row['kernel'] for row in manifest['contexts'].values()})
    data = PreparedData(args.release_dir, output / 'unused-audit.json',
                        selected_kernels=kernels,
                        authenticated_audit_path=args.authenticated_audit)
    prepared = prepare_small_cohort(data, args.manifest, torch.device('cuda'))
    schedule = _pair_schedule(prepared['groups'], args.updates, args.seed)
    schedule_hash = _json_sha([
        [prepared['groups'][gi]['key'], fraction, pair_index]
        for gi, fraction, pair_index in schedule])
    output.mkdir(parents=True)
    initial = _initial_models(args.seed, prepared['profile_feature_dim'])
    initial_hashes = {}
    training = {}
    validation_steps = tuple(range(args.validate_every, args.updates + 1, args.validate_every))
    for arm in ARMS:
        model = initial[arm]
        initial_hashes[arm] = (
            train.model_state_sha256(model['base']) if isinstance(model, dict)
            else train.model_state_sha256(model))
        # All arms consume the same fixed family-balanced sampled pair trace.
        training[arm] = train_arm(arm, model, prepared, schedule,
                                  torch.device('cuda'), validation_steps)
        torch.save({'state': _model_state(model), 'updates': args.updates,
                    'schema': 'salt-hls-v4b-full-cohort-action-aligned-preference-v1',
                    'arm': arm, 'seed': args.seed, 'manifest_sha256': _json_sha(prepared['manifest']),
                    'sample_trace_sha256': schedule_hash,
                    'selected_validation': training[arm]['selected_validation']},
                   output / f'{arm}_selected.pt')
    receipt = {'schema': 'salt-hls-v4b-full-cohort-action-aligned-preference-v1-run',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'manifest_sha256': _json_sha(prepared['manifest']),
               'sample_trace_sha256': schedule_hash, 'seed': args.seed,
               'updates_per_arm': args.updates, 'validation_steps': validation_steps,
               'arms': list(ARMS), 'initial_encoder_hashes': initial_hashes,
               'training': {name: {key: value for key, value in report.items()
                                   if key != 'checkpoints'} for name, report in training.items()},
               'preparation_seconds': prepared['preparation_seconds'],
               'selected_measurements': prepared['selected_measurements'],
               'test_kernels_loaded': [], 'checkpoint_selection': 'not selected: validation evaluator is required before publishing'}
    (output / 'run_receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    (output / 'training_history.json').write_text(json.dumps(
        {name: report['history'] for name, report in training.items()},
        sort_keys=True, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
