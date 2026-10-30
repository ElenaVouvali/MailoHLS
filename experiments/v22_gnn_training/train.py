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
import os
from pathlib import Path
import random
import signal
import sys
import traceback
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
from experiments.v22_gnn_training import checkpointing

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


def fit_ranking_scales(training_records):
    """Median within-context log-score spread, fitted on training rows only.

    Pairwise comparisons are always made inside one kernel/device/clock group,
    so a global cross-context spread can be unrelated to the scale of those
    comparisons. The robust median of per-context standard deviations gives
    the logistic loss a representative within-pool margin scale.
    """
    if not training_records:
        raise ValueError('Cannot fit ranking scales without training records')
    grouped = defaultdict(list)
    for row in training_records:
        grouped[(row['kernel'],row['profile'],row['clock'])].append(row)
    scales = {}
    for objective in OBJECTIVES:
        context_spreads = []
        for rows in grouped.values():
            values = np.log1p(np.asarray(
                [measured_objective(row, objective) for row in rows], dtype=np.float64))
            if len(values) >= 2 and len(np.unique(values)) >= 2:
                spread = float(np.std(values))
                if math.isfinite(spread) and spread > 1e-12:
                    context_spreads.append(spread)
        scale = float(np.median(context_spreads)) if context_spreads else 0.0
        if not math.isfinite(scale) or scale <= 1e-12:
            raise ValueError(f'Training contexts have no finite objective variation: {objective}')
        scales[objective] = scale
    return scales


def ranking_loss(prediction, records, objective, budget, target_mean=None, target_std=None,
                 return_pair_count=False, objective_scales=None):
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
        score = torch.log1p(pred_latency)
    elif objective == 'PARETO_AREA':
        score = torch.log1p(pred_area)
    elif objective == 'PARETO_ADP':
        score = torch.log1p(pred_latency * pred_area)
    else:
        raise ValueError('Unsupported ranking objective')
    if objective_scales is not None:
        scale = float(objective_scales[objective])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Ranking objective scale must be finite and positive')
        # This monotonic log score matches the regression target scale while
        # preserving ranking order; train-only std prevents physical units
        # (especially utilization fractions and latency*area) from suppressing
        # the auxiliary gradient.
        score = score / scale
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
    result = torch.stack(losses).mean() if losses else predicted
    return (result, len(losses)) if return_pair_count else result


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
        # Candidate demand is currently derived from program/candidate facts,
        # but the descriptor also records the target context. Include it in the
        # cache key so later context-dependent analytical features cannot be
        # silently reused across devices or clocks.
        key = (record['kernel'], record['profile'], record['clock'], record['config_hash'])
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
                mean, std, arm, device, rank_weight=0.1, rank_scales=None,
                return_details=False):
    step_started = time.perf_counter()
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
    rank, informative_pairs = ranking_loss(
        prediction, records, objective, budget, mean, std, return_pair_count=True,
        objective_scales=rank_scales)
    loss = regression + rank_weight * rank
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite supervised loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    clipped = bool(gradient_norm.detach().item() > 1.0)
    optimizer.step()
    result = (float(loss.detach()), float(regression.detach()), float(rank.detach()))
    if return_details:
        return result + ({
            'informative_pair_count': informative_pairs,
            'rank_weight': rank_weight,
            'weighted_rank_loss': float((rank_weight * rank).detach()),
            'gradient_norm_before_clip': float(gradient_norm.detach()),
            'gradient_clipped': clipped,
            'step_seconds': time.perf_counter() - step_started,
        },)
    return result


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
                        # Validation may use a deterministic prefix. Record
                        # regret against the evaluated set, not as though the
                        # complete measured pool had been searched.
                        'evaluated_pool_regret_fraction':float((measured_score[chosen] - best) / max(abs(best),1e-12)),
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
                             'regret_reference_pool_complete':len(records) == len(all_records),
                             'regret_reference_pool':('complete_measured_pool' if len(records) == len(all_records)
                                                      else 'deterministic_hash_prefix_subset'),
                             'primary_full_budget':primary,'budget_scenarios':budgets}
            per_group.append(group_metrics)
    if not per_group:
        raise ValueError('No measured contexts available for evaluation')
    return {
        'contexts':per_group,
        'macro_latency_mape':float(np.mean([row['primary_full_budget']['latency_mape'] for row in per_group])),
        'macro_resource_mae_fraction':float(np.mean([row['primary_full_budget']['resource_mae_fraction'] for row in per_group])),
        'macro_latency_regret_fraction':float(np.mean([
            row['primary_full_budget']['objectives']['PARETO_LATENCY']['evaluated_pool_regret_fraction'] for row in per_group])),
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
        'macro_evaluated_pool_regret_fraction':float(np.mean([
            metric['evaluated_pool_regret_fraction'] for row in per_group
            for metric in row['primary_full_budget']['objectives'].values()])),
    }


def validation_selection_score(metrics):
    """Predeclared bounded score emphasizing feasibility then ranking/regret/QoR."""
    feasibility_error = 1.0 - metrics['macro_feasibility_accuracy']
    rank_error = (1.0 - metrics['macro_rank_spearman']) / 2.0
    regret = max(0.0, metrics['macro_evaluated_pool_regret_fraction'])
    regret_error = regret / (1.0 + regret)
    qor = (metrics['macro_latency_mape'] + metrics['macro_resource_mae_fraction']) / 2.0
    qor_error = qor / (1.0 + qor)
    return 0.4 * feasibility_error + 0.2 * rank_error + 0.2 * regret_error + 0.2 * qor_error


_STOP_SIGNAL = None


def _request_stop(signum, _frame):
    global _STOP_SIGNAL
    _STOP_SIGNAL = int(signum)


def _copy_state_to_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _gpu_memory(device):
    if not torch.cuda.is_available():
        return {}
    return {
        'allocated_bytes': int(torch.cuda.memory_allocated(device)),
        'reserved_bytes': int(torch.cuda.memory_reserved(device)),
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
    }


def _write_status(output, status, **fields):
    value = {'schema':'salt-hls-v22-training-status-v1', 'status':status,
             'updated_at':checkpointing.utc_now(), 'pid':os.getpid(), **fields}
    checkpointing.atomic_json(output / 'status.json', value)


def _save_state(output, state):
    state = dict(state)
    state['saved_at'] = checkpointing.utc_now()
    checkpointing.atomic_torch_save(output / 'training_state.pt', state)


def _verify_resume_lock(output):
    lock_path = output / 'launcher.lock'
    if not lock_path.exists():
        return
    try:
        lock = json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise RuntimeError('Resume refused: launcher.lock is unreadable; inspect the run directory')
    pid = int(lock.get('pid', -1))
    if checkpointing.active_pid(pid):
        # A detached launcher intentionally owns the lock while its child
        # trainer resumes. Permit only that direct parent; concurrent owners
        # and unrelated live PIDs remain a hard refusal.
        if pid == os.getppid():
            return
        raise RuntimeError(f'Resume refused: launcher process {pid} is still active')


def _source_hashes():
    names = (
        'experiments/v22_gnn_training/train.py',
        'experiments/v22_gnn_training/audit.py',
        'experiments/v22_gnn_training/checkpointing.py',
        'experiments/target_graph_readiness/experiment.py',
        'Preprocessing/create_jsonl.py',
        'GNN_branch/targets/fpga_profiles.json',
        'GNN_branch/target_graph_tensor_v22.py',
        'GNN_branch/semantic_tensor_features_v22.py',
        'GNN_branch/action_demand_v22.py',
        'GNN_branch/target_graph_v22.py',
        'GNN_branch/action_point_relations.py',
        'LLM_branch/common/directive_domains.py',
    )
    return {name:overlay.file_hash(ROOT / name) for name in names}


def _append_progress(output, event, **fields):
    checkpointing.append_jsonl(output / 'progress.jsonl',
                               {'time':checkpointing.utc_now(), 'event':event, **fields})


def _atomic_selected_checkpoint(path, payload, protocol_fingerprint):
    if path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=False)
        if (existing.get('protocol_fingerprint') != protocol_fingerprint
                or existing.get('schema') != HEAD_SCHEMA):
            raise ValueError('Refusing to overwrite an incompatible selected checkpoint: ' + str(path))
        return
    checkpointing.atomic_torch_save(path, payload)


def run(args):
    global _STOP_SIGNAL
    if not torch.cuda.is_available():
        raise RuntimeError('Training blocked: CUDA is unavailable; no checkpoint was produced')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Training requires exactly one explicitly selected CUDA device')
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(device)
    if args.expected_gpu.casefold() not in gpu_name.casefold():
        raise RuntimeError(f'Expected GPU {args.expected_gpu!r}, visible device is {gpu_name!r}')

    output = args.output
    if args.resume:
        if not output.is_dir() or not (output / 'training_state.pt').is_file():
            raise FileNotFoundError('--resume requires an existing output with training_state.pt')
        _verify_resume_lock(output)
        try:
            status = json.loads((output / 'status.json').read_text())
        except (OSError, json.JSONDecodeError):
            status = {}
        if status.get('status') == 'completed':
            raise RuntimeError('Resume refused: experiment is already marked completed')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Refusing to overwrite experiment output: ' + str(output))
        output.mkdir(parents=True, exist_ok=True)

    old_handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    run_started = time.time()
    try:
        torch.set_num_threads(1)
        audit_path = output / 'data_audit.json'
        dataset = PreparedData(args.release_dir, audit_path)
        train_groups = {k:v for k,v in dataset.groups.items() if v and v[0]['split']=='train'}
        val_groups = {k:v for k,v in dataset.groups.items() if v and v[0]['split']=='val'}
        test_groups = {k:v for k,v in dataset.groups.items() if v and v[0]['split']=='test'}
        train_rows = [row for rows in train_groups.values() for row in rows]
        mean, std = _fit_normalization(train_rows)
        rank_scales = fit_ranking_scales(train_rows)
        git_commit = __import__('subprocess').check_output(
            ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        git_status = __import__('subprocess').check_output(
            ['git','status','--porcelain'], cwd=ROOT, text=True)
        source_hashes = _source_hashes()
        environment = {
            'python':sys.version.split()[0], 'torch':torch.__version__,
            'pyg':__import__('torch_geometric').__version__, 'cuda_runtime':torch.version.cuda,
            'device':gpu_name, 'visible_cuda_devices':torch.cuda.device_count(),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        }
        training = {
            'optimizer':'AdamW','lr':args.lr,'weight_decay':args.weight_decay,
            'steps':args.steps,'candidates_per_update':args.candidates,
            'log_every':args.log_every,'validation_every':args.val_every,
            'validation_candidates':args.validation_candidates,'patience':args.patience,
            'checkpoint_every':args.checkpoint_every,'progress_every':args.progress_every,
            'rank_weight':args.rank_weight,'gradient_clip_norm':1.0,
            'test_evaluation_enabled':not args.skip_test,
            'loss':'SmoothL1(z(log1p([latency_ms, physical resources]))) + rank_weight * pairwise logistic; relative 5% tie exclusion',
            'budget_contexts':'uniform physical-capacity fractions 1.0/0.5; labels invariant across budgets',
            'objectives':OBJECTIVES,'seed':SEED,
        }
        fingerprint_payload = {
            'schema':'salt-hls-v22-target-aware-training-protocol-v2-resumable',
            'git_commit':git_commit,'working_tree_dirty_at_start':bool(git_status.strip()),
            'source_sha256':source_hashes,
            'dataset_sha256':dataset.audit['dataset']['dataset_sha256'],
            'split_sha256':dataset.audit['dataset']['split_sha256'],
            'selected_manifest_sha256':dataset.audit['dataset']['selected_manifest_sha256'],
            'graph_archive_sha256':dataset.audit['graph_release']['archive_sha256'],
            'graph_schema':program.SCHEMA_V22,'overlay_schema':overlay.SCHEMA,
            'tensor_schema':tensors.SCHEMA,'candidate_demand_schema':action_demand.CANDIDATE_SCHEMA,
            'candidate_head_schema':HEAD_SCHEMA,'environment':environment,'training':training,
            'normalization_mean':mean.tolist(),'normalization_std':std.tolist(),
            'ranking_objective_scales_training_only':rank_scales,
        }
        fingerprint = checkpointing.canonical_sha256(fingerprint_payload)
        report_path = output / 'protocol.json'
        if args.resume:
            if not report_path.is_file():
                raise FileNotFoundError('Resume refused: protocol.json is missing')
            prior_report = json.loads(report_path.read_text())
            if prior_report.get('protocol_fingerprint') != fingerprint:
                raise ValueError('Resume refused: current data/code/environment/config does not match protocol.json')
            report = prior_report
            state = torch.load(output / 'training_state.pt', map_location='cpu', weights_only=False)
            checkpointing.verify_fingerprint(state, fingerprint)
            if state.get('schema') != 'salt-hls-v22-training-state-v1':
                raise ValueError('Resume refused: unsupported training-state schema')
            if state.get('phase') == 'completed':
                raise RuntimeError('Resume refused: training state says completed')
        else:
            if report_path.exists():
                raise FileExistsError('Refusing to replace existing protocol.json')
            report = {
                **fingerprint_payload,
                'schema':'salt-hls-v22-target-aware-training-protocol-v2-resumable',
                'protocol_fingerprint':fingerprint,
                'rows_by_split':{name:sum(len(rows) for rows in groups.values()) for name,groups in
                                 [('train',train_groups),('val',val_groups),('test',test_groups)]},
                'excluded_rows':dataset.exclusions,'models':{},
            }
            checkpointing.atomic_json(report_path, report)
            state = {
                'schema':'salt-hls-v22-training-state-v1','protocol_fingerprint':fingerprint,
                'phase':'training','arm_index':0,'step':0,'arm_report':None,
                'model_state':None,'optimizer_state':None,'best_model_state':None,
                'best_validation':float('inf'),'best_validation_metrics':None,
                'best_validation_step':None,'stale_checks':0,'rng_state':None,
                'completed_arms':[],'test_results':{},'run_started_unix':run_started,
            }
            checkpointing.atomic_json(output / 'protocol_identity.json', fingerprint_payload)
            _save_state(output, state)
            _write_status(output, 'prepared', protocol_fingerprint=fingerprint,
                          rows_by_split=report['rows_by_split'], device=gpu_name)
            _append_progress(output, 'prepared', protocol_fingerprint=fingerprint,
                             rows_by_split=report['rows_by_split'], device=gpu_name)
        if _STOP_SIGNAL is not None:
            _write_status(output,'interrupted',signal=_STOP_SIGNAL,phase=state.get('phase'),
                          arm_index=state.get('arm_index'),step=state.get('step'))
            return {'interrupted':True}

        arm_names = ('numerical','hardware')
        start_arm = int(state['arm_index'])
        for arm_index in range(start_arm, len(arm_names)):
            arm = arm_names[arm_index]
            torch.cuda.reset_peak_memory_stats(device)
            if args.resume and arm_index == start_arm and state.get('model_state') is not None:
                model = CandidateQoRModel().to(device)
                model.load_state_dict(state['model_state'])
                optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                optimizer.load_state_dict(state['optimizer_state'])
                checkpointing.restore_rng_state(state['rng_state'])
                arm_report = state['arm_report']
                current_step = int(state['step'])
                best_validation = float(state['best_validation'])
                best_validation_metrics = state['best_validation_metrics']
                best_validation_step = state['best_validation_step']
                best_state = state['best_model_state']
                stale_checks = int(state['stale_checks'])
                arm_started = time.time()
            else:
                # Reset before each arm to match the initial weights and the
                # kernel/context/candidate/objective/budget sampling sequence.
                random.seed(SEED); np.random.seed(SEED)
                torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
                model = CandidateQoRModel().to(device)
                optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                arm_report = {'parameters':sum(p.numel() for p in model.parameters()),
                              'train_curve':[],'validation_history':[],'status':'training'}
                current_step = 0
                best_validation = float('inf')
                best_validation_metrics = None
                best_validation_step = None
                best_state = None
                stale_checks = 0
                arm_started = time.time()
            model.train()
            while current_step < args.steps:
                step = current_step + 1
                group_keys = sorted(train_groups)
                group_key = random.choice(group_keys)
                group = train_groups[group_key]
                count = min(args.candidates,len(group))
                sampled = sorted(random.sample(range(len(group)),count))
                objective = random.choice(OBJECTIVES)
                fraction = random.choice(BUDGET_FRACTIONS)
                loss, reg, rank, details = _train_step(
                    model,optimizer,dataset,group_key,sampled,objective,fraction,
                    mean,std,arm,device,rank_weight=args.rank_weight,
                    rank_scales=rank_scales,return_details=True)
                current_step = step
                if step % args.log_every == 0 or step == 1:
                    curve = {'step':step,'objective':objective,'budget_fraction':fraction,
                             'loss':loss,'qor':reg,'rank':rank,**details,**_gpu_memory(device)}
                    arm_report['train_curve'].append(curve)
                if step % args.progress_every == 0 or step == 1:
                    _append_progress(output,'optimizer_update',arm=arm,step=step,
                                     objective=objective,budget_fraction=fraction,
                                     loss=loss,qor_loss=reg,ranking_loss=rank,**details,
                                     **_gpu_memory(device))
                    _write_status(output,'training',protocol_fingerprint=fingerprint,
                                  arm=arm,arm_index=arm_index,step=step,maximum_steps=args.steps,
                                  elapsed_seconds=time.time()-run_started,device=gpu_name,
                                  **_gpu_memory(device))

                validated = False
                if step % args.val_every == 0 or step == args.steps:
                    validation_started = time.perf_counter()
                    validation = evaluate(model,dataset,val_groups,mean,std,arm,device,
                                          max_candidates=args.validation_candidates)
                    validation_seconds = time.perf_counter()-validation_started
                    score = validation_selection_score(validation)
                    arm_report['validation_history'].append({'step':step,'selection_score':score,
                        'elapsed_seconds':validation_seconds,'gpu_memory':_gpu_memory(device),
                        'metrics':validation})
                    if score < best_validation:
                        best_validation=score
                        best_validation_metrics=validation
                        best_validation_step=step
                        best_state=_copy_state_to_cpu(model)
                        stale_checks=0
                    else:
                        stale_checks+=1
                    validated=True
                    report['models'][arm]=arm_report
                    checkpointing.atomic_json(report_path,report)
                    _append_progress(output,'validation',arm=arm,step=step,
                                     selection_score=score,validation_seconds=validation_seconds,
                                     best_validation_step=best_validation_step,
                                     stale_checks=stale_checks,metrics=validation,
                                     **_gpu_memory(device))
                    _write_status(output,'validation_complete',protocol_fingerprint=fingerprint,
                                  arm=arm,arm_index=arm_index,step=step,
                                  best_validation_step=best_validation_step,
                                  validation_seconds=validation_seconds,device=gpu_name,
                                  **_gpu_memory(device))
                    model.train()

                should_checkpoint=(step % args.checkpoint_every == 0 or validated
                                   or _STOP_SIGNAL is not None or step==args.steps
                                   or stale_checks>=args.patience)
                if should_checkpoint:
                    state.update({'phase':'training','arm_index':arm_index,'step':step,'arm':arm,
                        'arm_report':arm_report,'model_state':_copy_state_to_cpu(model),
                        'optimizer_state':optimizer.state_dict(),'best_model_state':best_state,
                        'best_validation':best_validation,'best_validation_metrics':best_validation_metrics,
                        'best_validation_step':best_validation_step,'stale_checks':stale_checks,
                        'rng_state':checkpointing.capture_rng_state(),
                        'completed_arms':list(report['models']),
                        'test_results':report.get('test_results',{}),
                        'run_started_unix':state.get('run_started_unix',run_started)})
                    checkpoint_started=time.perf_counter()
                    _save_state(output,state)
                    checkpoint_seconds=time.perf_counter()-checkpoint_started
                    _append_progress(output,'checkpoint',arm=arm,step=step,
                                     checkpoint_seconds=checkpoint_seconds,
                                     checkpoint_path=str(output/'training_state.pt'),
                                     **_gpu_memory(device))
                    _write_status(output,'checkpoint_saved',protocol_fingerprint=fingerprint,
                                  arm=arm,arm_index=arm_index,step=step,
                                  checkpoint_seconds=checkpoint_seconds,device=gpu_name,
                                  **_gpu_memory(device))

                if _STOP_SIGNAL is not None:
                    _append_progress(output,'graceful_stop',arm=arm,step=step,
                                     signal=_STOP_SIGNAL)
                    _write_status(output,'interrupted',protocol_fingerprint=fingerprint,
                                  arm=arm,arm_index=arm_index,step=step,
                                  signal=_STOP_SIGNAL,device=gpu_name)
                    return {'interrupted':True}
                if stale_checks >= args.patience:
                    break

            if best_state is None or best_validation_step is None:
                raise RuntimeError(f'{arm} arm reached completion without a validation-selected state')
            arm_report.update({'seconds':time.time()-arm_started,
                               'best_validation_selection_score':best_validation,
                               'validation_selected_at_step':best_validation_step,
                               'status':'validation checkpoint selected; test pending'})
            selected_payload = {
                'schema':HEAD_SCHEMA,'arm':arm,'model':best_state,
                'normalization_mean':mean,'normalization_std':std,
                'graph_schema':program.SCHEMA_V22,'overlay_schema':overlay.SCHEMA,
                'tensor_schema':tensors.SCHEMA,
                'candidate_demand_schema':action_demand.CANDIDATE_SCHEMA,
                'protocol_fingerprint':fingerprint,
                'protocol_commit':git_commit,'training_source_sha256':source_hashes,
                'validation':{'step':best_validation_step,'selection_score':best_validation,
                              'metrics':best_validation_metrics},
            }
            _atomic_selected_checkpoint(output/(arm+'_selected.pt'),selected_payload,fingerprint)
            report['models'][arm]=arm_report
            checkpointing.atomic_json(report_path,report)
            _append_progress(output,'arm_validation_selected',arm=arm,
                             validation_step=best_validation_step,
                             validation_score=best_validation,seconds=arm_report['seconds'])
            _write_status(output,'arm_selected',protocol_fingerprint=fingerprint,
                          arm=arm,arm_index=arm_index,step=current_step,
                          selected_checkpoint=str(output/(arm+'_selected.pt')),
                          validation_step=best_validation_step,device=gpu_name)
            next_arm=arm_index+1
            state.update({'phase':'training' if next_arm<len(arm_names) else 'test_evaluation',
                          'arm_index':next_arm,'step':0,'arm':None,'arm_report':None,
                          'model_state':None,'optimizer_state':None,'best_model_state':None,
                          'best_validation':float('inf'),'best_validation_metrics':None,
                          'best_validation_step':None,'stale_checks':0,'rng_state':None,
                          'completed_arms':list(report['models']),
                          'run_started_unix':state.get('run_started_unix',run_started)})
            _save_state(output,state)

        # Test is deliberately evaluated only after both validation-selected
        # checkpoints exist, preventing test observations from steering arm B.
        if args.skip_test:
            state.update({'phase':'smoke_complete','completed_arms':list(report['models'])})
            _save_state(output,state)
            _append_progress(output,'smoke_complete',arms=list(report['models']),
                             test_evaluation='skipped_by_explicit_preflight_flag')
            _write_status(output,'smoke_complete',protocol_fingerprint=fingerprint,
                          arms=list(report['models']),test_evaluation='skipped')
            return report
        evaluations_path=output/'evaluation.json'
        evaluations=json.loads(evaluations_path.read_text()) if evaluations_path.exists() else {}
        for arm in arm_names:
            if arm in evaluations:
                continue
            selected=torch.load(output/(arm+'_selected.pt'),map_location='cpu',weights_only=False)
            if selected.get('protocol_fingerprint')!=fingerprint:
                raise ValueError('Selected checkpoint protocol fingerprint mismatch')
            model=CandidateQoRModel().to(device)
            model.load_state_dict(selected['model'])
            test_started=time.perf_counter()
            test_metrics=evaluate(model,dataset,test_groups,mean,std,arm,device)
            evaluations[arm]={'test':test_metrics,'seconds':time.perf_counter()-test_started,
                              'checkpoint':str(output/(arm+'_selected.pt'))}
            checkpointing.atomic_json(evaluations_path,evaluations)
            report['models'][arm]['test']=test_metrics
            report['models'][arm]['status']='validation-selected checkpoint; test evaluated once after both selections'
            checkpointing.atomic_json(report_path,report)
            _append_progress(output,'test_evaluation_complete',arm=arm,
                             elapsed_seconds=evaluations[arm]['seconds'],metrics=test_metrics)
            _write_status(output,'test_evaluated',protocol_fingerprint=fingerprint,
                          arm=arm,device=gpu_name,**_gpu_memory(device))
            del model
            torch.cuda.empty_cache()
        state.update({'phase':'completed','test_results':evaluations,
                      'completed_arms':list(report['models'])})
        _save_state(output,state)
        checkpointing.atomic_json(output/'completed.json',{
            'schema':'salt-hls-v22-completed-run-v1','protocol_fingerprint':fingerprint,
            'completed_at':checkpointing.utc_now(),'arms':list(report['models']),
            'evaluation_path':str(evaluations_path)})
        _append_progress(output,'run_completed',elapsed_seconds=time.time()-run_started,
                         arms=list(report['models']))
        _write_status(output,'completed',protocol_fingerprint=fingerprint,
                      elapsed_seconds=time.time()-run_started,arms=list(report['models']),device=gpu_name)
        return report
    except BaseException as exc:
        if output.exists():
            _append_progress(output,'exception',exception_type=type(exc).__name__,
                             message=str(exc),traceback=traceback.format_exc())
            _write_status(output,'failed',exception_type=type(exc).__name__,
                          message=str(exc),traceback=traceback.format_exc(),
                          signal=_STOP_SIGNAL)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig,handler)


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
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--progress-every',type=int,default=25)
    parser.add_argument('--rank-weight',type=float,default=0.1)
    parser.add_argument('--expected-gpu',default='A2')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--skip-test',action='store_true',
                        help='Preflight only: stop after both validation selections without reading test metrics')
    args = parser.parse_args()
    if (args.steps < 1 or args.candidates < 2 or args.log_every < 1 or args.val_every < 1
            or args.patience < 1 or args.checkpoint_every < 1 or args.progress_every < 1
            or not math.isfinite(args.rank_weight) or args.rank_weight < 0):
        parser.error('steps/log interval must be positive and at least two candidates are required')
    result = run(args)
    if result.get('interrupted'):
        signal_number = _STOP_SIGNAL or signal.SIGTERM
        print(json.dumps({'status':'interrupted','output':str(args.output),
                          'signal':signal_number}),flush=True)
        raise SystemExit(128+signal_number)
    print(json.dumps({'output':str(args.output),'rows_by_split':result['rows_by_split'],
                      'models':list(result['models'])}, sort_keys=True),flush=True)


if __name__ == '__main__':
    main()
