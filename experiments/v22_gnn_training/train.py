"""Train the experimental MLIR V2.2 target-aware QoR GNN.

All graph tensors are pragma-free. Candidate directives and candidate-demand
features enter only the prediction head. This module deliberately refuses to
train without CUDA: a CPU smoke update is not a substitute for the requested
measured-data experiment.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import kendalltau, spearmanr

from GNN_branch import action_demand_v22 as action_demand
from GNN_branch import mlir_graph_v2 as program
from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch import target_graph_v22 as overlay
from LLM_branch.common import directive_domains as domains
from Preprocessing.create_jsonl import build_examples, file_sha256, validate_preprocessing_manifest
from experiments.target_graph_readiness.experiment import candidate_demand_tensor, CANDIDATE_DEMAND_DIM, CANDIDATE_DIM
from experiments.v22_gnn_training.audit import _candidate_values, _family_split, audit_dataset

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED = ROOT / 'LLM_branch/Data/preprocessed_CSVS'
SPLIT = ROOT / 'mailohls_runs/mailohls_final_family_split_s123.json'
DEFAULT_RELEASE = ROOT / 'experiments/target_graph_readiness/results/release'
RESOURCE_LABELS = ('bram_util_%', 'dsp_util_%', 'ff_util_%', 'lut_util_%')
RESOURCE_ORDER = ('BRAM_18K', 'DSP', 'FF', 'LUT')
OBJECTIVES = ('PARETO_LATENCY', 'PARETO_AREA', 'PARETO_ADP')
BUDGET_FRACTIONS = (1.0, 0.5)
SEED = 123
HEAD_SCHEMA = 'salt-hls-v22-candidate-qor-head-v1-experimental'


def sha256_json(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def candidate_vector(values, actions, slots=64):
    """Encode the complete legal assignment with the readiness-pilot layout."""
    result = torch.zeros(slots, CANDIDATE_DIM, dtype=torch.float32)
    for label, action in actions.items():
        slot = int(label[1:]) - 1
        if slot >= slots:
            raise ValueError('Action slot exceeds the established absolute-slot interface')
        get = lambda kind: values['AUTO{_' + kind + '_' + label + '}']
        if action.kind == 'loop':
            result[slot, 0] = float(get('PIPE'))
            result[slot, 1] = math.log2(1 + int(get('UNROLL'))) / 12
        else:
            result[slot, 2 + ('none', 'complete', 'block', 'cyclic').index(get('ARRAY_T'))] = 1
            result[slot, 6] = math.log2(1 + int(get('ARRAY_F'))) / 12
            result[slot, 7] = int(get('ARRAY_D')) / 8
    return result


def physical_targets(row, profile):
    """Return log-transform inputs in [ms, physical BRAM/DSP/FF/LUT units]."""
    latency_ms = float(row['latency'])
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError('Latency measurement must be finite and positive (milliseconds)')
    resources = []
    for column, resource in zip(RESOURCE_LABELS, RESOURCE_ORDER):
        fraction = float(row[column]) / 100.0
        capacity = float(profile['resources'][resource]['value'])
        if not math.isfinite(fraction) or fraction < 0:
            raise ValueError('Resource utilization must be finite and nonnegative')
        resources.append(fraction * capacity)
    return [math.log1p(latency_ms)] + [math.log1p(value) for value in resources], [latency_ms] + resources


def measured_objective(record, objective):
    latency = record['physical'][0]
    utilization = [count / cap for count, cap in zip(record['physical'][1:], record['capacities'])]
    area = sum(utilization) / 4.0
    if objective == 'PARETO_LATENCY':
        return latency
    if objective == 'PARETO_AREA':
        return area
    if objective == 'PARETO_ADP':
        # Within one clock context, latency_cycles*area has the same ordering
        # as latency_ms*area. No cross-clock comparison is made.
        return latency * area
    raise ValueError('Unsupported MailoHLS objective: ' + objective)


def ranking_loss(prediction, records, objective, budget, target_mean=None, target_std=None):
    """Pairwise logistic loss among measured-feasible, non-tied candidates."""
    predicted = prediction.detach().new_tensor(0.)
    if target_mean is None:
        target_mean = prediction.new_zeros((5,))
    if target_std is None:
        target_std = prediction.new_ones((5,))
    physical_log_prediction = prediction * target_std.to(prediction.device) + target_mean.to(prediction.device)
    pred_latency = torch.expm1(physical_log_prediction[:, 0]).clamp_min(0)
    pred_counts = torch.expm1(physical_log_prediction[:, 1:]).clamp_min(0)
    capacities = prediction.new_tensor(records[0]['capacities'])
    pred_area = (pred_counts / capacities).mean(-1)
    if objective == 'PARETO_LATENCY':
        score = pred_latency
    elif objective == 'PARETO_AREA':
        score = pred_area
    elif objective == 'PARETO_ADP':
        score = pred_latency * pred_area
    else:
        raise ValueError('Unsupported ranking objective')
    feasible = []
    for i, row in enumerate(records):
        if all(value <= limit for value, limit in zip(row['physical'][1:], budget)):
            feasible.append(i)
    losses = []
    for p, i in enumerate(feasible):
        measured_i = measured_objective(records[i], objective)
        for j in feasible[p + 1:]:
            measured_j = measured_objective(records[j], objective)
            scale = max(abs(measured_i), abs(measured_j), 1e-12)
            if abs(measured_i - measured_j) / scale <= 0.05:
                continue
            # Lower score is better. Softplus(score_better - score_worse).
            better, worse = (i, j) if measured_i < measured_j else (j, i)
            losses.append(F.softplus(score[better] - score[worse]))
    return torch.stack(losses).mean() if losses else predicted


class CandidateQoRModel(nn.Module):
    """V3 encoder plus a small candidate-conditioned multi-output head."""
    def __init__(self):
        super().__init__()
        self.graph = tensors.SmallGraphEncoder(hidden=64)
        self.action_kind = nn.Embedding(3, 4)
        self.local = nn.Sequential(
            nn.Linear(128 + CANDIDATE_DIM + CANDIDATE_DEMAND_DIM + 5, 64),
            nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(128 + 32 + tensors.CONTEXT_DIM, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, graph, candidates, candidate_demands, arm='hardware'):
        memory, pooled = self.graph.memories(graph)
        if memory.shape[0] != 1:
            raise ValueError('One target-context graph is required per optimizer update')
        return self.from_memory(graph, memory, pooled, candidates, candidate_demands, arm)

    def from_memory(self, graph, memory, pooled, candidates, candidate_demands, arm='hardware'):
        """Score many candidates after one shared graph-encoder evaluation."""
        if memory.shape[0] != 1 or pooled.shape[0] != 1:
            raise ValueError('One target-context graph is required per optimizer update')
        n = candidates.shape[0]
        memory = memory.expand(n, -1, -1)
        kind_ids = graph.action_kind_slot.expand(n, -1)
        kind = self.action_kind(kind_ids)
        matched = graph.matched_mask.expand(n, -1).float().unsqueeze(-1)
        local = self.local(torch.cat([memory, candidates, candidate_demands, kind, matched], -1))
        registered = graph.registered_mask.expand(n, -1).float().unsqueeze(-1)
        local = (local * registered).sum(1) / registered.sum(1).clamp_min(1)
        context = graph.hardware_context.expand(n, -1)
        if arm not in ('numerical', 'hardware'):
            raise ValueError('arm must be numerical or hardware')
        # Both arms receive identical numeric target conditioning. The
        # numerical arm differs only because PreparedData removed hardware
        # message edges before encoding; zeroing this context would turn it
        # into an unconditioned baseline and invalidate the comparison.
        return self.head(torch.cat([pooled.expand(n, -1), local, context], -1))


class PreparedData:
    """Authenticated rows/graphs; expensive graph tensors are cached on CPU."""
    def __init__(self, release_dir, audit_path):
        self.release_dir = Path(release_dir)
        self.audit = audit_dataset(self.release_dir, audit_path)
        if self.audit['gate'] not in ('pass', 'pass_with_explicit_exclusions'):
            raise ValueError('Authenticated data audit did not pass; see ' + str(audit_path))
        self.split = json.loads(SPLIT.read_text())
        self.examples, self.kernel_manifest, _ = build_examples(PREPROCESSED)
        self.index_split, self.kernel_split = _family_split(self.split, len(self.examples))
        self.registry = overlay.load_registry()
        # The exact per-kernel action inventory is joined to the audited graph
        # member using its full kernel-name prefix, then rechecked by identity.
        self.graphs = {}
        self.actions = {}
        self.profile_names = {}
        self.preprocessed_action_cache = {}
        from LLM_branch.common.directive_domains import tcad
        for kernel in sorted({row['kernel_name'] for row in self.examples}):
            item = next((x for x in self.audit['kernels']
                         if Path(x['graph_member']).name.startswith(kernel + '-')), None)
            if item is None:
                raise ValueError('No exact authenticated graph mapping for ' + kernel)
            path = self.release_dir / item['graph_member']
            graph, auth = overlay.authenticated_input(path)
            if graph.graph['graph_identity'] != item['graph_identity']:
                raise ValueError('Graph identity changed after preflight for ' + kernel)
            self.graphs[kernel] = (graph, auth)
            self.actions[kernel] = tcad.load_source_actions(kernel, ROOT / 'Data/ApplicationDataset')
            self.profile_names[kernel] = item
        self.records = []
        excluded = defaultdict(int)
        device_map = {'xczu7ev-ffvc1156-2-e': 'zcu104', 'xcu200-fsgd2104-2-e': 'u200'}
        for row_index, row in enumerate(self.examples):
            kernel = row['kernel_name']
            if self.index_split[row_index] != self.kernel_split[kernel]:
                raise ValueError('Family split index/kernel mismatch')
            profile_name = device_map.get(row['device'].lower())
            if profile_name is None or float(row['clock_period']) not in (3.33, 5.0, 10.0):
                excluded['reserved_nontraining_target_context'] += 1
                continue
            actions = self.actions[kernel]
            missing = [action.action_id for action in self._preprocess_actions(kernel)
                       if not action.csv_column]
            if missing:
                # These rows contain preprocessing defaults for actions that
                # have no measured CSV/APL field. They are not measured labels.
                excluded['kernel_has_unmeasured_apl_actions'] += 1
                continue
            try:
                assignment = _candidate_values(row['target'])
                from LLM_branch.common.directive_domains import tcad
                tcad.validate_assignments(assignment, actions)
            except (TypeError, ValueError):
                excluded['invalid_complete_source_tuple'] += 1
                continue
            target_profile = self.registry['profiles'][profile_name]
            transformed, physical = physical_targets(row, target_profile)
            target = row['device'].lower()
            signature = sha256_json(assignment)
            self.records.append({
                'row_index': row_index, 'kernel': kernel,
                'family': self.kernel_split[kernel], 'split': self.index_split[row_index],
                'profile': profile_name, 'device': target,
                'clock': float(row['clock_period']), 'assignment': assignment,
                'config_hash': signature, 'transformed': transformed,
                'physical': physical,
                'capacities': [float(target_profile['resources'][r]['value']) for r in RESOURCE_ORDER],
            })
        self.exclusions = dict(sorted(excluded.items()))
        self.groups = defaultdict(list)
        for record in self.records:
            self.groups[(record['kernel'], record['profile'], record['clock'])].append(record)
        for values in self.groups.values():
            values.sort(key=lambda row: row['config_hash'])
        self.graph_tensor_cache = OrderedDict()
        self.demand_cache = OrderedDict()

    def _preprocess_actions(self, kernel):
        if kernel not in self.preprocessed_action_cache:
            from Preprocessing.create_jsonl import load_actions, application_sources
            source = application_sources({kernel})[kernel].read_text(encoding='utf-8')
            self.preprocessed_action_cache[kernel] = load_actions(kernel, source)
        return self.preprocessed_action_cache[kernel]

    def graph_tensor(self, record, budget_fraction, arm):
        key = (record['kernel'], record['profile'], record['clock'], budget_fraction, arm)
        if key not in self.graph_tensor_cache:
            base, auth = self.graphs[record['kernel']]
            profile = self.registry['profiles'][record['profile']]
            budget = {resource: int(value * budget_fraction)
                      for resource, value in zip(RESOURCE_ORDER, record['capacities'])}
            enriched = overlay.enrich(base, record['profile'], record['clock'], budgets=budget, authentication=auth)
            data = tensors.tensorize(enriched)
            if arm == 'numerical':
                data = tensors.without_hardware_messages(data)
            self.graph_tensor_cache[key] = data
            while len(self.graph_tensor_cache) > 8:
                self.graph_tensor_cache.popitem(last=False)
        else:
            self.graph_tensor_cache.move_to_end(key)
        return self.graph_tensor_cache[key]

    def candidate_inputs(self, record):
        key = (record['kernel'], record['config_hash'])
        if key not in self.demand_cache:
            base, auth = self.graphs[record['kernel']]
            enriched = overlay.enrich(base, record['profile'], record['clock'], authentication=auth)
            demand, _ = candidate_demand_tensor(enriched, record['assignment'], self.actions[record['kernel']])
            self.demand_cache[key] = demand
            while len(self.demand_cache) > 8192:
                self.demand_cache.popitem(last=False)
        else:
            self.demand_cache.move_to_end(key)
        return candidate_vector(record['assignment'], self.actions[record['kernel']]), self.demand_cache[key]


def _fit_normalization(records):
    rows = torch.tensor([r['transformed'] for r in records if r['split'] == 'train'], dtype=torch.float64)
    if not len(rows):
        raise ValueError('No training records remain after authenticated exclusions')
    mean, std = rows.mean(0), rows.std(0, unbiased=False).clamp_min(1e-6)
    return mean.float(), std.float()


def _train_step(model, optimizer, dataset, group_key, sampled, objective, fraction,
                mean, std, arm, device, rank_weight=0.1):
    records = [dataset.groups[group_key][i] for i in sampled]
    graph = dataset.graph_tensor(records[0], fraction, arm).to(device)
    candidate_pairs = [dataset.candidate_inputs(record) for record in records]
    candidates = torch.stack([pair[0] for pair in candidate_pairs]).to(device)
    demands = torch.stack([pair[1] for pair in candidate_pairs]).to(device)
    raw = torch.tensor([r['transformed'] for r in records], dtype=torch.float32, device=device)
    target = (raw - mean.to(device)) / std.to(device)
    prediction = model(graph, candidates, demands, arm)
    regression = F.smooth_l1_loss(prediction, target)
    budget = [capacity * fraction for capacity in records[0]['capacities']]
    rank = ranking_loss(prediction, records, objective, budget, mean, std)
    loss = regression + rank_weight * rank
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite supervised loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), float(regression.detach()), float(rank.detach())


def evaluate(model, dataset, groups, mean, std, arm, device, max_candidates=None):
    """Evaluate QoR and measured-pool ranking without test-based selection."""
    model.eval()
    per_group = []
    with torch.no_grad():
        for key in sorted(groups):
            all_records = groups[key]
            records = all_records if max_candidates is None else all_records[:max_candidates]
            if len(records) < 1:
                continue
            true = np.asarray([row['physical'] for row in records], dtype=np.float64)
            capacities = np.asarray(records[0]['capacities'], dtype=np.float64)
            measured_area = np.mean(true[:, 1:] / capacities[None, :], axis=1)
            budgets = {}
            for fraction in BUDGET_FRACTIONS:
                graph_cpu = dataset.graph_tensor(records[0], fraction, arm)
                graph = graph_cpu.to(device)
                memory, pooled = model.graph.memories(graph)
                predictions = []
                for start in range(0, len(records), 16):
                    chunk = records[start:start + 16]
                    pairs = [dataset.candidate_inputs(record) for record in chunk]
                    candidate = torch.stack([pair[0] for pair in pairs]).to(device)
                    demand = torch.stack([pair[1] for pair in pairs]).to(device)
                    z = model.from_memory(graph, memory, pooled, candidate, demand, arm)
                    raw = torch.expm1(z * std.to(device) + mean.to(device)).clamp_min(0)
                    predictions.extend(raw.cpu().tolist())
                pred = np.asarray(predictions, dtype=np.float64)
                lat_mape = np.mean(np.abs(pred[:, 0] - true[:, 0]) / np.maximum(true[:, 0], 1e-12))
                resource_mae_fraction = np.mean(np.abs(pred[:, 1:] - true[:, 1:]) / capacities[None, :])
                predicted_area = np.mean(pred[:, 1:] / capacities[None, :], axis=1)
                objectives = {}
                score_vectors = (
                    ('PARETO_LATENCY', true[:, 0], pred[:, 0]),
                    ('PARETO_AREA', measured_area, predicted_area),
                    ('PARETO_ADP', true[:, 0] * measured_area, pred[:, 0] * predicted_area),
                )
                for name, measured_score, predicted_score in score_vectors:
                    chosen = int(np.argmin(predicted_score))
                    best = float(np.min(measured_score))
                    objectives[name] = {
                        'spearman':float(spearmanr(measured_score, predicted_score)[0])
                            if len(set(measured_score.tolist())) > 1 and len(set(predicted_score.tolist())) > 1 else None,
                        'kendall':float(kendalltau(measured_score, predicted_score)[0])
                            if len(set(measured_score.tolist())) > 1 and len(set(predicted_score.tolist())) > 1 else None,
                        'measured_pool_regret_fraction':float((measured_score[chosen] - best) / max(abs(best),1e-12)),
                        'top1_measured_best':bool(measured_score[chosen] <= best * 1.000001),
                    }
                limits = capacities * fraction
                measured_feasible = np.all(true[:, 1:] <= limits[None, :], axis=1)
                predicted_feasible = np.all(pred[:, 1:] <= limits[None, :], axis=1)
                fp = int(np.sum(predicted_feasible & ~measured_feasible))
                predicted_count = int(predicted_feasible.sum())
                feasible_result = {
                    'fraction_of_physical_capacity':fraction,
                    'measured_feasible_candidates':int(measured_feasible.sum()),
                    'predicted_feasible_candidates':predicted_count,
                    'feasibility_accuracy':float(np.mean(predicted_feasible == measured_feasible)),
                    'false_feasible_rate_among_predicted_feasible':fp / max(1, predicted_count),
                    'no_measured_feasible_candidate':not bool(measured_feasible.any()),
                }
                for name, measured_score, predicted_score in score_vectors:
                    selected = np.flatnonzero(predicted_feasible)
                    if len(selected) and measured_feasible.any():
                        chosen = int(selected[np.argmin(predicted_score[selected])])
                        best_feasible = float(np.min(measured_score[measured_feasible]))
                        selected_is_feasible = bool(measured_feasible[chosen])
                        feasible_result[name] = {
                            'selected_measured_feasible':selected_is_feasible,
                            'feasible_pool_regret_fraction':float(
                                (measured_score[chosen] - best_feasible) / max(abs(best_feasible),1e-12))
                                if selected_is_feasible else None,
                        }
                    else:
                        feasible_result[name] = {'selected_measured_feasible':False,
                                                  'feasible_pool_regret_fraction':None}
                scenario = {'fraction':fraction,'latency_mape':float(lat_mape),
                            'resource_mae_fraction':float(resource_mae_fraction),
                            'latency_mae_ms':float(np.mean(np.abs(pred[:,0] - true[:,0]))),
                            'resource_mae_physical_units':np.mean(
                                np.abs(pred[:,1:] - true[:,1:]),axis=0).tolist(),
                            'resource_mae_utilization_percentage_points':(
                                100.0*np.mean(np.abs(pred[:,1:] - true[:,1:]),axis=0)/capacities).tolist(),
                            'objectives':objectives,'feasibility':feasible_result}
                if fraction == 1.0:
                    primary = scenario
                budgets[str(fraction)] = scenario
            group_metrics = {'kernel':key[0], 'profile':key[1], 'clock_ns':key[2],
                             'pool_size':len(all_records), 'evaluated_candidates':len(records),
                             'primary_full_budget':primary,'budget_scenarios':budgets}
            per_group.append(group_metrics)
    if not per_group:
        raise ValueError('No measured contexts available for evaluation')
    return {
        'contexts':per_group,
        'macro_latency_mape':float(np.mean([row['primary_full_budget']['latency_mape'] for row in per_group])),
        'macro_resource_mae_fraction':float(np.mean([row['primary_full_budget']['resource_mae_fraction'] for row in per_group])),
        'macro_latency_regret_fraction':float(np.mean([
            row['primary_full_budget']['objectives']['PARETO_LATENCY']['measured_pool_regret_fraction'] for row in per_group])),
        'evaluated_contexts':len(per_group),
        'evaluated_candidates':sum(row['evaluated_candidates'] for row in per_group),
        'macro_feasibility_accuracy':float(np.mean([
            scenario['feasibility']['feasibility_accuracy']
            for row in per_group for scenario in row['budget_scenarios'].values()])),
        'macro_rank_spearman':float(np.mean([
            metric['spearman'] for row in per_group
            for metric in row['primary_full_budget']['objectives'].values()
            if metric['spearman'] is not None])) if any(
                metric['spearman'] is not None for row in per_group
                for metric in row['primary_full_budget']['objectives'].values()) else 0.0,
        'macro_measured_pool_regret_fraction':float(np.mean([
            metric['measured_pool_regret_fraction'] for row in per_group
            for metric in row['primary_full_budget']['objectives'].values()])),
    }


def validation_selection_score(metrics):
    """Predeclared bounded score emphasizing feasibility then ranking/regret/QoR."""
    feasibility_error = 1.0 - metrics['macro_feasibility_accuracy']
    rank_error = (1.0 - metrics['macro_rank_spearman']) / 2.0
    regret = max(0.0, metrics['macro_measured_pool_regret_fraction'])
    regret_error = regret / (1.0 + regret)
    qor = (metrics['macro_latency_mape'] + metrics['macro_resource_mae_fraction']) / 2.0
    qor_error = qor / (1.0 + qor)
    return 0.4 * feasibility_error + 0.2 * rank_error + 0.2 * regret_error + 0.2 * qor_error


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Training blocked: CUDA is unavailable to this process. No GPU was detected and no checkpoint was produced.')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite experiment output: ' + str(args.output))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    audit_path = args.output / 'data_audit.json'
    dataset = PreparedData(args.release_dir, audit_path)
    train_groups = {k: v for k, v in dataset.groups.items() if v and v[0]['split'] == 'train'}
    val_groups = {k: v for k, v in dataset.groups.items() if v and v[0]['split'] == 'val'}
    test_groups = {k: v for k, v in dataset.groups.items() if v and v[0]['split'] == 'test'}
    train_rows = [r for values in train_groups.values() for r in values]
    mean, std = _fit_normalization(train_rows)
    device = torch.device('cuda')
    git_commit = __import__('subprocess').check_output(
        ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    git_status = __import__('subprocess').check_output(
        ['git','status','--porcelain'], cwd=ROOT, text=True)
    report = {
        'schema': 'salt-hls-v22-target-aware-training-protocol-v1',
        'git_commit': git_commit,
        'working_tree_dirty_at_start':bool(git_status.strip()),
        'training_source_sha256':{
            name:overlay.file_hash(ROOT/name) for name in (
                'experiments/v22_gnn_training/train.py',
                'experiments/v22_gnn_training/audit.py',
                'GNN_branch/target_graph_tensor_v22.py',
                'GNN_branch/action_demand_v22.py',
                'GNN_branch/target_graph_v22.py',
            )
        },
        'dataset_sha256': dataset.audit['dataset']['dataset_sha256'],
        'split_sha256': dataset.audit['dataset']['split_sha256'],
        'graph_archive_sha256': dataset.audit['graph_release']['archive_sha256'],
        'graph_schema': program.SCHEMA_V22, 'overlay_schema': overlay.SCHEMA,
        'tensor_schema': tensors.SCHEMA, 'candidate_demand_schema': action_demand.CANDIDATE_SCHEMA,
        'candidate_head_schema': HEAD_SCHEMA, 'seed': SEED,
        'rows_by_split': {k: sum(len(v) for v in groups.values()) for k, groups in
                          [('train', train_groups), ('val', val_groups), ('test', test_groups)]},
        'excluded_rows': dataset.exclusions,
        'models': {}, 'normalization_log1p_targets_mean': mean.tolist(),
        'normalization_log1p_targets_std': std.tolist(),
        'training': {'optimizer':'AdamW','lr':args.lr,'weight_decay':args.weight_decay,
                     'steps':args.steps,'candidates_per_update':args.candidates,
                     'loss':'SmoothL1(z(log1p([latency_ms, physical resources]))) + 0.1 * pairwise logistic; 5% relative tie exclusion',
                     'budget_contexts':'deterministic uniform physical-capacity fractions 1.0/0.5; measured QoR labels invariant',
                     'objectives':OBJECTIVES},
        'environment': {'torch':torch.__version__, 'pyg':__import__('torch_geometric').__version__,
                        'cuda':torch.version.cuda, 'device':torch.cuda.get_device_name(0)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'protocol.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for arm in ('numerical', 'hardware'):
        # Same initialization and per-step sampling sequence for the paired arms.
        random.seed(SEED); np.random.seed(SEED)
        torch.manual_seed(SEED)
        model = CandidateQoRModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        arm_report = {'parameters':sum(p.numel() for p in model.parameters()), 'train_curve':[],
                      'validation_history':[]}
        started = time.time()
        model.train()
        best_validation = float('inf')
        best_validation_metrics = None
        best_validation_step = None
        best_state = None
        stale_checks = 0
        for step in range(1, args.steps + 1):
            group_key = random.choice(sorted(train_groups))
            group = train_groups[group_key]
            count = min(args.candidates, len(group))
            sampled = sorted(random.sample(range(len(group)), count))
            objective = random.choice(OBJECTIVES)
            fraction = random.choice(BUDGET_FRACTIONS)
            loss, reg, rank = _train_step(model, optimizer, dataset, group_key, sampled,
                                          objective, fraction, mean, std, arm, device)
            if step % args.log_every == 0:
                arm_report['train_curve'].append({'step':step,'loss':loss,'qor':reg,'rank':rank})
            if step % args.val_every == 0 or step == args.steps:
                validation = evaluate(model, dataset, val_groups, mean, std, arm, device,
                                      max_candidates=args.validation_candidates)
                score = validation_selection_score(validation)
                arm_report['validation_history'].append({'step':step,'selection_score':score,
                                                         'metrics':validation})
                if score < best_validation:
                    best_validation = score
                    best_validation_metrics = validation
                    best_validation_step = step
                    best_state = {name:value.detach().cpu().clone()
                                  for name,value in model.state_dict().items()}
                    stale_checks = 0
                else:
                    stale_checks += 1
                model.train()
                if stale_checks >= args.patience:
                    break
        arm_report['seconds'] = time.time() - started
        if best_state is None:
            raise ValueError('No validation checkpoint was selected')
        model.load_state_dict(best_state)
        arm_report['best_validation_selection_score'] = best_validation
        arm_report['validation_selected_at_step'] = best_validation_step
        arm_report['test'] = evaluate(model, dataset, test_groups, mean, std, arm, device)
        arm_report['status'] = 'trained; checkpoint selected on validation; test evaluated once'
        torch.save({'schema':HEAD_SCHEMA,'arm':arm,'model':best_state,
                    'normalization_mean':mean,'normalization_std':std,
                    'graph_schema':program.SCHEMA_V22,'overlay_schema':overlay.SCHEMA,
                    'tensor_schema':tensors.SCHEMA,'candidate_demand_schema':action_demand.CANDIDATE_SCHEMA,
                    'protocol':report,'validation':{'step':best_validation_step,
                                                       'selection_score':best_validation,
                                                       'metrics':best_validation_metrics},
                    'test_metrics':arm_report['test']}, args.output / (arm + '_selected.pt'))
        report['models'][arm] = arm_report
        model.to(device)
        (args.output / 'protocol.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--output', type=Path, default=ROOT/'mailohls_runs/v22_target_aware_gnn_s123')
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--validation-candidates', type=int, default=256,
                        help='deterministic prefix of each validation context for model selection')
    parser.add_argument('--patience', type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.candidates < 2 or args.log_every < 1 or args.val_every < 1 or args.patience < 1:
        parser.error('steps/log interval must be positive and at least two candidates are required')
    result = run(args)
    print(json.dumps({'output':str(args.output),'rows_by_split':result['rows_by_split'],
                      'models':list(result['models'])}, sort_keys=True))


if __name__ == '__main__':
    main()
