"""Small cached three-arm target-conditioning probe on authenticated GNΩSIS data.

This uses the existing V4 CandidateQoRModel, loss and compiler graph APIs. It
is a diagnostic only; no test split or synthesis results are used.
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
from scipy.stats import kendalltau, spearmanr
from torch import nn
from torch.nn import functional as F

from experiments.v22_gnn_training import checkpointing, fast_probe, train

TRAIN_KERNELS = fast_probe.TRAIN_KERNELS
VALIDATION_KERNELS = fast_probe.VALIDATION_KERNELS
PROFILES = ('zcu104', 'u200')
CLOCKS = (3.33, 5.0, 10.0)
N_CANDIDATES = 16
SEED = 123
SCHEMA = 'salt-hls-v4-target-condition-small-probe-v1'
BUDGET_INDEPENDENT_SCHEMA = 'salt-hls-v4-physical-qor-budget-independent-v1'
TOPK_VALUES = (1, 3, 5, 10)
LATENCY_REL_TOLERANCE = 0.01
# Source CSV latencies are reported in ms to at most 1e-6 ms in this cohort.
# One 10 ns clock cycle is 1e-5 ms, so use one cycle at the largest measured
# period as the absolute floor for near-zero latency comparisons.
LATENCY_ABS_TOLERANCE_MS = 1e-5


def _assignment_intersection(groups, kernel):
    maps = [
        {row['config_hash']: row for row in groups.get((kernel, profile, clock), [])}
        for profile in PROFILES for clock in CLOCKS
    ]
    return set.intersection(*(set(item) for item in maps)) if maps and all(maps) else set()


class CandidateOnlyModel(nn.Module):
    """Small non-graph control with candidate/action and numeric context only."""
    def __init__(self):
        super().__init__()
        width = 64 * (train.CANDIDATE_DIM + train.CANDIDATE_DEMAND_DIM) + train.tensors.CONTEXT_DIM
        self.net = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, context, candidates, demands):
        n = candidates.shape[0]
        return self.net(torch.cat([candidates.flatten(1), demands.flatten(1),
                                   context.expand(n, -1)], dim=-1))


def _metric_rank(fn, actual, predicted):
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return None
    value = float(fn(actual, predicted)[0])
    return value if math.isfinite(value) else None


def _relative_regret(scores, selected_score):
    """Return relative regret only when the measured optimum is well-scaled."""
    best = float(np.min(scores))
    scale = max(float(np.max(np.abs(scores))), 1e-12)
    if abs(best) <= 1e-6 * scale:
        return None, 'near_zero_measured_optimum'
    return float((float(selected_score)-best)/abs(best)), None


def _absolute_regret(scores, selected_score):
    """Absolute measured-pool objective gap; defined even when optimum is zero."""
    return max(0.0, float(selected_score) - float(np.min(scores)))


def _feasible_ranked_shortlists(actual_scores, predicted_scores,
                                measured_feasible, predicted_feasible,
                                topk=TOPK_VALUES):
    """Feasibility-aware measured-pool selection without hiding failed shortlists."""
    actual_scores = np.asarray(actual_scores, dtype=np.float64)
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64)
    measured_feasible = np.asarray(measured_feasible, dtype=bool)
    predicted_feasible = np.asarray(predicted_feasible, dtype=bool)
    if not (actual_scores.shape == predicted_scores.shape == measured_feasible.shape == predicted_feasible.shape):
        raise ValueError('Score and feasibility vectors must have identical shapes')
    n_true = int(measured_feasible.sum())
    eligible = np.flatnonzero(predicted_feasible)
    order = eligible[np.argsort(predicted_scores[eligible], kind='stable')] if len(eligible) else []
    result = {'measured_feasible_candidate_count': n_true,
              'predicted_feasible_candidate_count': int(len(eligible)),
              'measured_pool_size': int(len(actual_scores)),
              'predicted_feasible_selection_empty': not bool(len(eligible)),
              'topk': {}}
    for k in topk:
        shortlist = list(order[:k])
        actual_good = [int(i) for i in shortlist if measured_feasible[i]]
        item = {'requested_k': int(k), 'effective_shortlist_size': len(shortlist),
                'shortlist_actual_feasible_count': len(actual_good),
                'has_measured_feasible_selection': bool(actual_good),
                'selected_index': shortlist[0] if shortlist else None,
                'selected_is_measured_feasible': (bool(measured_feasible[shortlist[0]])
                                                  if shortlist else None),
                'relative_measured_feasible_pool_regret': None,
                'absolute_measured_feasible_pool_regret': None,
                'regret_undefined_reason': None}
        if n_true == 0:
            item['regret_undefined_reason'] = 'no_measured_feasible_candidate_in_pool'
        elif not shortlist:
            item['regret_undefined_reason'] = 'no_predicted_feasible_candidate'
        elif not actual_good:
            item['regret_undefined_reason'] = 'shortlist_contains_no_measured_feasible_candidate'
        else:
            best = float(np.min(actual_scores[measured_feasible]))
            selected = float(np.min(actual_scores[actual_good]))
            relative, reason = _relative_regret(actual_scores[measured_feasible], selected)
            item['relative_measured_feasible_pool_regret'] = relative
            item['absolute_measured_feasible_pool_regret'] = _absolute_regret(
                actual_scores[measured_feasible], selected)
            item['regret_undefined_reason'] = reason
        result['topk'][str(k)] = item
    return result


def _features(data, records):
    graphs = {}
    candidates = {}
    demands = {}
    groups = defaultdict(list)
    for row in records:
        key = (row['kernel'], row['profile'], float(row['clock']))
        if key not in graphs:
            graphs[key] = {}
            for fraction in (1.0, 0.5, 0.1):
                hardware = data.graph_tensor(row, fraction, 'hardware').cpu()
                graphs[key][fraction] = {
                    'hardware': hardware,
                    'numerical': train.tensors.without_hardware_messages(hardware),
                }
        identity = (row['kernel'], row['profile'], float(row['clock']), row['config_hash'])
        if identity not in candidates:
            candidates[identity], demands[identity] = (x.cpu() for x in data.candidate_inputs(row))
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: row['config_hash'])
    return graphs, candidates, demands, groups


def _select_cohort(data):
    groups = data.groups
    selected = {}
    selection = {}
    for kernel in TRAIN_KERNELS:
        common = _assignment_intersection(groups, kernel)
        if len(common) < N_CANDIDATES:
            raise ValueError(f'{kernel}: only {len(common)} assignments common across all six training contexts')
        per = [groups[(kernel, p, c)] for p in PROFILES for c in CLOCKS]
        exemplar = {r['config_hash']: r for r in per[0]}
        seed_rows = [exemplar[h] for h in sorted(common)]
        chosen = fast_probe._select_training_candidates(seed_rows, N_CANDIDATES)
        hashes = [r['config_hash'] for r in chosen]
        for profile in PROFILES:
            for clock in CLOCKS:
                lookup = {r['config_hash']: r for r in groups[(kernel, profile, clock)]}
                selected[(kernel, profile, clock)] = [lookup[h] for h in hashes]
        selection[kernel] = {'split': 'train', 'common_all_six_available': len(common),
                             'selected_assignments': hashes,
                             'selection': 'max-Hamming assignment diversity with hash tie break; no QoR labels'}

    for kernel in VALIDATION_KERNELS:
        common = _assignment_intersection(groups, kernel)
        if len(common) >= 8:
            hashes = sorted(common)[:min(N_CANDIDATES, len(common))]
            method = 'hash-first exact assignment intersection across six contexts; QoR-blind'
            for profile in PROFILES:
                for clock in CLOCKS:
                    lookup = {r['config_hash']: r for r in groups[(kernel, profile, clock)]}
                    selected[(kernel, profile, clock)] = [lookup[h] for h in hashes]
        else:
            # No six-way assignment intersection: choose hash-first independently
            # in each target context, and retain exact pair intersections later.
            per_context = {}
            for profile in PROFILES:
                for clock in CLOCKS:
                    rows = groups[(kernel, profile, clock)]
                    chosen = sorted(rows, key=lambda r: r['config_hash'])[:N_CANDIDATES]
                    selected[(kernel, profile, clock)] = chosen
                    per_context[f'{profile}|{clock:g}'] = [r['config_hash'] for r in chosen]
            method = 'hash-first independently per context; QoR-blind; paired analyses intersect exact hashes'
            hashes = []
        selection[kernel] = {'split': 'validation', 'common_all_six_available': len(common),
                             'selection': method,
                             'selected_assignments': hashes if hashes else per_context}
    return selected, selection


def _collate(group, graph, candidates, demands, device):
    c, d = [], []
    for row in group:
        key = (row['kernel'], row['profile'], float(row['clock']), row['config_hash'])
        c.append(candidates[key]); d.append(demands[key])
    return torch.stack(c).to(device), torch.stack(d).to(device)


def _predict(model, arm, graph, records, candidates, demands, device):
    c, d = _collate(records, graph, candidates, demands, device)
    if arm == 'candidate_target':
        context = graph.hardware_context
        z = model(context, c, d)
    else:
        z = model(graph, c, d, arm=arm)
    return z


def _graph_at_requested_budget(graph_cache, key, fraction, arm, budget_independent=False):
    graph_fraction = 1.0 if budget_independent else fraction
    return graph_cache[key][graph_fraction][
        arm if arm in ('hardware', 'numerical') else 'hardware']


def _evaluate(model, arm, groups, graph_cache, candidates, demands, mean, std, device,
              budget_independent=False):
    model.eval()
    per_context = []
    physical_by_hash = {}
    with torch.no_grad():
        for key in sorted(groups):
            rows = groups[key]
            budget_predictions = {}
            for fraction in (1.0, 0.5, 0.1):
                graph = _graph_at_requested_budget(graph_cache, key, fraction, arm,
                                                   budget_independent).to(device)
                z = _predict(model, arm, graph, rows, candidates, demands, device)
                budget_predictions[fraction] = torch.expm1(z * std + mean).clamp_min(0).cpu().numpy()
            prediction = budget_predictions[1.0]
            actual = np.asarray([row['physical'] for row in rows], dtype=np.float64)
            capacities = np.asarray(rows[0]['capacities'], dtype=np.float64)
            area = np.mean(actual[:, 1:] / capacities, axis=1)
            pred_area = np.mean(prediction[:, 1:] / capacities, axis=1)
            scores = {
                'latency': (actual[:, 0], prediction[:, 0]),
                'area': (area, pred_area),
                'adp': (actual[:, 0] * area, prediction[:, 0] * pred_area),
            }
            ranking = {}
            feasible_selection = {}
            for name, (y, p) in scores.items():
                selected = int(np.argmin(p))
                order = np.argsort(p, kind='stable')
                top3 = order[:min(3, len(order))]
                top1_regret, regret_reason = _relative_regret(y, y[selected])
                top3_regret, _ = _relative_regret(y, float(np.min(y[top3])))
                ranking[name] = {
                    'spearman': _metric_rank(spearmanr, y, p),
                    'kendall': _metric_rank(kendalltau, y, p),
                    'top1_regret': top1_regret,
                    'top3_regret': top3_regret,
                    'top1_absolute_regret': _absolute_regret(y, y[selected]),
                    'top3_absolute_regret': _absolute_regret(y, float(np.min(y[top3]))),
                    'regret_undefined_reason': regret_reason,
                    'pool_size': len(rows),
                }
                feasible_selection[name] = {}
                for fraction in (1.0, .5, .1):
                    measured_feasible = np.all(
                        actual[:, 1:] <= (capacities * fraction)[None, :], axis=1)
                    fraction_prediction = budget_predictions[fraction]
                    predicted_feasible = np.all(
                        fraction_prediction[:, 1:] <= (capacities * fraction)[None, :], axis=1)
                    predicted_area = np.mean(fraction_prediction[:, 1:] /
                                             capacities[None, :], axis=1)
                    predicted_objective = {
                        'latency': fraction_prediction[:, 0],
                        'area': predicted_area,
                        'adp': fraction_prediction[:, 0] * predicted_area,
                    }[name]
                    feasible_selection[name][str(fraction)] = _feasible_ranked_shortlists(
                        y, predicted_objective, measured_feasible, predicted_feasible)
            feasibility = {}
            full_selected_hash = None
            for fraction in (1.0, 0.5, 0.1):
                pred_for_budget = budget_predictions[fraction]
                measured_feasible = np.all(actual[:, 1:] <= (capacities*fraction)[None,:],axis=1)
                predicted_feasible = np.all(pred_for_budget[:,1:] <= (capacities*fraction)[None,:],axis=1)
                selected = int(np.argmin(np.where(predicted_feasible, pred_for_budget[:,0], np.inf))) if predicted_feasible.any() else None
                if fraction == 1.0 and selected is not None:
                    full_selected_hash = rows[selected]['config_hash']
                tp = int(np.sum(measured_feasible & predicted_feasible))
                fp = int(np.sum(~measured_feasible & predicted_feasible))
                fn = int(np.sum(measured_feasible & ~predicted_feasible))
                tn = int(np.sum(~measured_feasible & ~predicted_feasible))
                feasibility[str(fraction)] = {
                    'accuracy': float(np.mean(measured_feasible==predicted_feasible)),
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'false_feasible': fp, 'false_infeasible': fn,
                    'false_feasible_rate': (float(fp / int((~measured_feasible).sum()))
                                             if (~measured_feasible).any() else None),
                    'false_infeasible_rate': (float(fn / int(measured_feasible.sum()))
                                              if measured_feasible.any() else None),
                    'precision': float(tp / (tp + fp)) if tp + fp else None,
                    'recall': float(tp / (tp + fn)) if tp + fn else None,
                    'measured_feasible_count': int(measured_feasible.sum()),
                    'predicted_feasible_count': int(predicted_feasible.sum()),
                    'always_feasible_baseline_accuracy': float(np.mean(measured_feasible)),
                    'selected_is_measured_feasible': bool(measured_feasible[selected]) if selected is not None else None,
                    'selected_config_hash': rows[selected]['config_hash'] if selected is not None else None,
                }
                feasibility[str(fraction)]['top1_regret'] = (
                    float((actual[selected,0]-np.min(actual[measured_feasible,0])) /
                          max(abs(float(np.min(actual[measured_feasible,0]))),1e-12))
                    if selected is not None and measured_feasible.any() and measured_feasible[selected] else None)
            per_context.append({
                'kernel': key[0], 'profile': key[1], 'clock_ns': key[2],
                'latency_mae_ms': float(np.mean(np.abs(prediction[:, 0]-actual[:, 0]))),
                'latency_mape': float(np.mean(np.abs(prediction[:, 0]-actual[:, 0])/
                                              np.maximum(actual[:, 0], 1e-12))),
                'latency_rmse_ms': float(np.sqrt(np.mean((prediction[:, 0]-actual[:, 0]) ** 2))),
                'predicted_latency_range_ms': [float(prediction[:, 0].min()),
                                               float(prediction[:, 0].max())],
                'measured_latency_range_ms': [float(actual[:, 0].min()),
                                              float(actual[:, 0].max())],
                'resource_mae_counts': np.mean(np.abs(prediction[:, 1:] - actual[:, 1:]), axis=0).tolist(),
                'resource_mae_fraction': np.mean(np.abs(prediction[:,1:]-actual[:,1:])/
                                                 capacities[None,:], axis=0).tolist(),
                'resource_mae_utilization_percentage_points': (
                    100 * np.mean(np.abs(prediction[:, 1:] - actual[:, 1:]), axis=0) /
                    capacities).tolist(),
                'ranking': ranking,
                'feasible_selection': feasible_selection,
                'feasibility_by_budget_fraction': feasibility,
                'mean_abs_budget_induced_physical_prediction_delta': np.mean(
                    np.abs(budget_predictions[1.0]-budget_predictions[0.5]),axis=0).tolist(),
                'budget_qor_delta_by_budget_pair': {
                    f'{left:g}_vs_{right:g}': {
                        'mean_absolute_physical_prediction_delta': np.mean(
                            np.abs(budget_predictions[left] - budget_predictions[right]), axis=0).tolist(),
                        'exactly_invariant': bool(np.array_equal(
                            budget_predictions[left], budget_predictions[right]))}
                    for left, right in ((1.0, .5), (1.0, .1))},
                'selected_config_hash_full_budget': full_selected_hash,
            })
            for row, pred in zip(rows, prediction):
                physical_by_hash[(key, row['config_hash'])] = pred
    return {'contexts': per_context, 'predictions': physical_by_hash}


def _validation_predictions(model, arm, groups, graph_cache, candidates, demands,
                            mean, std, device, budget_independent=False):
    """Return per-budget physical predictions keyed by exact assignment hash."""
    model.eval()
    result = {}
    with torch.no_grad():
        for key in sorted(groups):
            rows = groups[key]
            for fraction in (1.0, .5, .1):
                graph = _graph_at_requested_budget(graph_cache, key, fraction, arm,
                                                   budget_independent).to(device)
                z = _predict(model, arm, graph, rows, candidates, demands, device)
                values = torch.expm1(z * std + mean).clamp_min(0).cpu().numpy()
                result[(key, fraction)] = {row['config_hash']: value
                                           for row, value in zip(rows, values)}
    return result


def _fixed_training_diagnostic(model, arm, key, rows, graph_cache, candidates,
                               demands, mean, std, rank_scales, device):
    """Evaluate a frozen training-context set without updating model state."""
    model.eval()
    graph = graph_cache[key][1.0][arm if arm in ('hardware', 'numerical') else 'hardware'].to(device)
    with torch.no_grad():
        c, d = _collate(rows, graph, candidates, demands, device)
        z = _predict(model, arm, graph, rows, candidates, demands, device)
        y = torch.tensor([row['transformed'] for row in rows], dtype=torch.float32,
                         device=device)
        reg = F.smooth_l1_loss(z, (y - mean) / std)
        full_budget = rows[0]['capacities']
        rank, pairs = train.ranking_loss(z, rows, 'PARETO_LATENCY', full_budget,
                                         mean, std, return_pair_count=True,
                                         objective_scales=rank_scales)
        physical = torch.expm1(z * std + mean).clamp_min(0).cpu().numpy()
    actual = np.asarray([row['physical'] for row in rows])
    actual_best = int(np.argmin(actual[:, 0]))
    pred_best = int(np.argmin(physical[:, 0]))
    return {
        'regression_smoothl1': float(reg.cpu()),
        'ranking_loss': float(rank.cpu()),
        'ranking_pair_count': int(pairs),
        'latency_spearman': _metric_rank(spearmanr, actual[:, 0], physical[:, 0]),
        'latency_mae_ms': float(np.mean(np.abs(actual[:, 0] - physical[:, 0]))),
        'predicted_latency_range_ms': [float(physical[:, 0].min()), float(physical[:, 0].max())],
        'measured_latency_range_ms': [float(actual[:, 0].min()), float(actual[:, 0].max())],
        'predicted_best_config_hash': rows[pred_best]['config_hash'],
        'measured_best_config_hash': rows[actual_best]['config_hash'],
        'selected_measured_pool_regret_fraction': float(
            (actual[pred_best, 0] - actual[actual_best, 0]) / actual[actual_best, 0]),
    }


def _contrast_report(selected_groups, predictions):
    by_budget = {}
    for fraction in (1.0, .5, .1):
        selected = {(key): rows for key, rows in selected_groups.items()
                    if rows[0]['split'] == 'val'}
        pred = {(key, assignment): value
                for (key, budget), mapping in predictions.items() if budget == fraction
                for assignment, value in mapping.items()}
        rows = _target_contrasts(selected, pred)
        by_budget[str(fraction)] = {'groups': rows,
                                    'summary': _summarize_target_contrasts(rows)}
    return by_budget


def _verify_historical_cohort(old_manifest, new_manifest):
    keys = ('dataset_sha256', 'split_sha256', 'graph_archive_sha256',
            'training_kernels', 'validation_kernels', 'target_contexts', 'selection')
    mismatches = [key for key in keys if old_manifest.get(key) != new_manifest.get(key)]
    if mismatches:
        raise ValueError('Historical checkpoint cohort differs in: ' + ', '.join(mismatches))


def _target_contrasts(groups, predictions):
    records_by_key = {key: {row['config_hash']: row for row in rows}
                      for key, rows in groups.items()}
    reports = {'clock': [], 'device': []}
    for kernel in VALIDATION_KERNELS:
        for profile in PROFILES:
            for i, c1 in enumerate(CLOCKS):
                for c2 in CLOCKS[i+1:]:
                    k1, k2 = (kernel, profile, c1), (kernel, profile, c2)
                    left, right = records_by_key.get(k1, {}), records_by_key.get(k2, {})
                    common = sorted(set(left) & set(right))
                    if not common: continue
                    reports['clock'].append(_contrast_row(
                        'clock', kernel, f'{profile}:{c1:g}->{c2:g}ns', common,
                        k1, k2, left, right, predictions))
        for clock in CLOCKS:
            k1, k2 = (kernel, 'zcu104', clock), (kernel, 'u200', clock)
            left, right = records_by_key.get(k1, {}), records_by_key.get(k2, {})
            common = sorted(set(left) & set(right))
            if common:
                reports['device'].append(_contrast_row(
                    'device', kernel, f'{clock:g}ns:zcu104->u200', common,
                    k1, k2, left, right, predictions))
    return reports


def _meaningful_latency_delta(left_ms, right_ms, relative=LATENCY_REL_TOLERANCE,
                              absolute_ms=LATENCY_ABS_TOLERANCE_MS):
    """Classify paired latency changes using the original measurements.

    A delta is meaningful only when it exceeds both a one-percent relative
    threshold (relative to the larger endpoint) and the source-resolution
    floor. The absolute floor is one 10 ns cycle expressed in milliseconds.
    """
    left = np.asarray(left_ms, dtype=np.float64)
    right = np.asarray(right_ms, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError('Paired latency arrays must have the same shape')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Paired latency values must be finite')
    delta = right - left
    threshold = np.maximum(absolute_ms,
                           relative * np.maximum(np.abs(left), np.abs(right)))
    return delta, np.abs(delta) > threshold


def _contrast_row(kind, kernel, target_pair, common, k1, k2, left, right, predictions):
    actual, predicted = [], []
    for h in common:
        actual.append(np.asarray(right[h]['physical']) - np.asarray(left[h]['physical']))
        predicted.append(predictions[(k2, h)] - predictions[(k1, h)])
    if not common:
        return {'kernel': kernel, 'target_pair': target_pair,
                'matched_assignments': 0,
                'latency_tolerance': {'relative': LATENCY_REL_TOLERANCE,
                                     'absolute_ms': LATENCY_ABS_TOLERANCE_MS},
                'meaningful_positive_latency_delta_count': 0,
                'meaningful_negative_latency_delta_count': 0,
                'negligible_latency_delta_count': 0,
                'latency_sign_accuracy': None,
                'latency_delta_mae_ms': None,
                'latency_delta_rmse_ms': None,
                'meaningful_latency_delta_mae_ms': None,
                'meaningful_latency_delta_rmse_ms': None,
                'mean_absolute_paired_error_by_target': None,
                'paired_delta_rmse_by_target': None}
    a, p = np.asarray(actual), np.asarray(predicted)
    delta_ms, meaningful = _meaningful_latency_delta(
        np.asarray([left[h]['physical'][0] for h in common]),
        np.asarray([right[h]['physical'][0] for h in common]))
    sign = np.sign(delta_ms[meaningful]) == np.sign(p[meaningful, 0])
    paired_error = p - a
    meaningful_error = p[meaningful, 0] - delta_ms[meaningful]
    return {'kernel': kernel, 'target_pair': target_pair,
            'matched_assignments': len(common),
            'latency_tolerance': {'relative': LATENCY_REL_TOLERANCE,
                                  'absolute_ms': LATENCY_ABS_TOLERANCE_MS},
            'meaningful_positive_latency_delta_count': int(np.sum(delta_ms[meaningful] > 0)),
            'meaningful_negative_latency_delta_count': int(np.sum(delta_ms[meaningful] < 0)),
            'meaningful_latency_delta_count': int(meaningful.sum()),
            'negligible_latency_delta_count': int((~meaningful).sum()),
            'mean_absolute_paired_error_by_target': np.mean(np.abs(paired_error),axis=0).tolist(),
            'paired_delta_rmse_by_target': np.sqrt(np.mean(paired_error ** 2,axis=0)).tolist(),
            'latency_delta_mae_ms': float(np.mean(np.abs(paired_error[:, 0]))),
            'latency_delta_rmse_ms': float(np.sqrt(np.mean(paired_error[:, 0] ** 2))),
            'meaningful_latency_delta_mae_ms': (float(np.mean(np.abs(meaningful_error)))
                                                if meaningful.any() else None),
            'meaningful_latency_delta_rmse_ms': (float(np.sqrt(np.mean(meaningful_error ** 2)))
                                                 if meaningful.any() else None),
            'latency_sign_accuracy': float(np.mean(sign)) if len(sign) else None,
            'near_zero_measured_latency_contrast_count': int((~meaningful).sum())}


def _summarize_target_contrasts(reports):
    """Summarize by kernel first so overlapping assignment pairs are not treated as independent kernels."""
    result = {}
    for kind, comparisons in reports.items():
        by_kernel = defaultdict(list)
        for row in comparisons:
            by_kernel[row['kernel']].append(row)
        per_kernel = {}
        for kernel, rows in sorted(by_kernel.items()):
            eligible = [row for row in rows if row['meaningful_latency_delta_count']]
            all_mae = [row['latency_delta_mae_ms'] for row in rows
                       if row['latency_delta_mae_ms'] is not None]
            all_rmse = [row['latency_delta_rmse_ms'] for row in rows
                        if row['latency_delta_rmse_ms'] is not None]
            signs = [row['latency_sign_accuracy'] for row in eligible
                     if row['latency_sign_accuracy'] is not None]
            per_kernel[kernel] = {
                'target_pair_group_count': len(rows),
                'matched_assignment_pair_observations': sum(row['matched_assignments'] for row in rows),
                'meaningful_pair_observations': sum(row['meaningful_latency_delta_count'] for row in rows),
                'negligible_pair_observations': sum(row['negligible_latency_delta_count'] for row in rows),
                'macro_delta_mae_ms': float(np.mean(all_mae)) if all_mae else None,
                'macro_delta_rmse_ms': float(np.mean(all_rmse)) if all_rmse else None,
                'macro_meaningful_sign_accuracy': float(np.mean(signs)) if signs else None,
            }
        kernel_maes = [item['macro_delta_mae_ms'] for item in per_kernel.values()
                       if item['macro_delta_mae_ms'] is not None]
        kernel_rmses = [item['macro_delta_rmse_ms'] for item in per_kernel.values()
                        if item['macro_delta_rmse_ms'] is not None]
        kernel_signs = [item['macro_meaningful_sign_accuracy'] for item in per_kernel.values()
                        if item['macro_meaningful_sign_accuracy'] is not None]
        result[kind] = {
            'distinct_kernel_count': len(by_kernel),
            'comparison_group_count': len(comparisons),
            'macro_over_kernels_delta_mae_ms': float(np.mean(kernel_maes)) if kernel_maes else None,
            'macro_over_kernels_delta_rmse_ms': float(np.mean(kernel_rmses)) if kernel_rmses else None,
            'macro_over_kernels_meaningful_sign_accuracy': float(np.mean(kernel_signs)) if kernel_signs else None,
            'per_kernel': per_kernel,
            'pair_observations_overlap_across_target_groups': True,
        }
    return result


def run(args):
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = train.PreparedData(args.release_dir, out/'data_audit.json',
                              selected_kernels=TRAIN_KERNELS + VALIDATION_KERNELS)
    selected, selection = _select_cohort(data)
    records = [row for rows in selected.values() for row in rows]
    if any(row['split'] == 'test' for row in records): raise ValueError('Test records selected')
    prep = time.monotonic()-start
    if prep > args.preparation_limit_seconds: raise TimeoutError('Data audit/cohort selection exceeded prep cap')
    feature_start = time.monotonic()
    graph_cache, candidate_cache, demand_cache, all_groups = _features(data, records)
    features_seconds = time.monotonic()-feature_start
    if prep + features_seconds > args.preparation_limit_seconds:
        raise TimeoutError('Authenticated data plus feature preparation exceeded the preparation cap')
    selected_groups = {key: sorted(rows,key=lambda r:r['config_hash']) for key,rows in selected.items()}
    train_rows = [r for key,rows in selected_groups.items() if rows[0]['split']=='train' for r in rows]
    mean, std = train._fit_normalization(train_rows)
    rank_scales = train.fit_ranking_scales(train_rows)
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=train.ROOT,
                                         text=True).strip()
    git_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'],
                                              cwd=train.ROOT, text=True).strip())
    if git_dirty:
        raise ValueError('Refusing training from a dirty source worktree')
    run_schema = (BUDGET_INDEPENDENT_SCHEMA if args.budget_independent_physical_qor else SCHEMA)
    manifest = {
        'schema': run_schema, 'source_commit': git_commit,
        'physical_qor_budget_independent': bool(args.budget_independent_physical_qor),
        'seed': SEED, 'dataset_sha256': data.audit['dataset']['dataset_sha256'],
        'split_sha256': data.audit['dataset']['split_sha256'],
        'graph_archive_sha256': data.audit['graph_release']['archive_sha256'],
        'training_kernels': list(TRAIN_KERNELS), 'validation_kernels': list(VALIDATION_KERNELS),
        'target_contexts': [f'{p}|{c:g}ns' for p in PROFILES for c in CLOCKS],
        'selection': selection,
        'training_rows': len(train_rows), 'validation_rows': sum(len(r) for k,r in selected_groups.items() if r[0]['split']=='val'),
        'test_records_used_for_selection_or_fit': False,
        'normalization': 'log1p target mean/std fitted on selected training rows only',
        'physical_target_budget_invariance': 'same measured row reused under 100% and 50% request budgets; budget affects feasibility only',
        'feature_preparation_seconds': features_seconds,
        'source_hashes': fast_probe._source_hashes(),
        'target_condition_probe_sha256': fast_probe.overlay.file_hash(Path(__file__)),
    }
    manifest['sha256'] = fast_probe._canonical_hash(manifest)
    checkpointing.atomic_json(out/'cohort_manifest.json',manifest)

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    gpu=fast_probe._verify_idle_gpu(args.gpu_name,args.gpu_uuid)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Requires one isolated CUDA GPU')
    device=torch.device('cuda:0')
    if args.gpu_name.casefold() not in torch.cuda.get_device_name(device).casefold(): raise RuntimeError('GPU identity mismatch')
    gpu_start=time.monotonic(); deadline=gpu_start+args.gpu_limit_seconds
    mean,std=mean.to(device),std.to(device)
    historical_evaluation = None
    if args.compare_checkpoint_dir:
        if not args.compare_manifest:
            raise ValueError('--compare-manifest is required with --compare-checkpoint-dir')
        compare_dir = Path(args.compare_checkpoint_dir)
        old_manifest = json.loads(Path(args.compare_manifest).read_text())
        _verify_historical_cohort(old_manifest, manifest)
        old_models = {}
        old_metrics = {}
        for old_arm in ('numerical', 'hardware', 'candidate_target'):
            path = compare_dir / f'{old_arm}_probe.pt'
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if checkpoint.get('schema') != SCHEMA or checkpoint.get('arm') != old_arm:
                raise ValueError(f'Invalid historical checkpoint identity: {path}')
            if checkpoint.get('cohort_sha256') != old_manifest.get('sha256'):
                raise ValueError(f'Historical checkpoint/cohort mismatch: {path}')
            if not torch.equal(checkpoint['normalization_mean'].cpu(), mean.cpu()) or not torch.equal(
                    checkpoint['normalization_std'].cpu(), std.cpu()):
                raise ValueError(f'Historical normalization differs: {path}')
            cls = CandidateOnlyModel if old_arm == 'candidate_target' else train.CandidateQoRModel
            model = cls().to(device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            old_models[old_arm] = {
                'checkpoint_sha256': fast_probe.overlay.file_hash(path),
                'metrics': _evaluate(model, old_arm, val_groups, graph_cache,
                                     candidate_cache, demand_cache, mean, std, device,
                                     budget_independent=False),
                'predictions': _validation_predictions(model, old_arm, val_groups,
                                    graph_cache, candidate_cache, demand_cache,
                                    mean, std, device, budget_independent=False),
            }
            del model
        historical_evaluation = {
            'checkpoint_directory': str(compare_dir),
            'cohort_manifest_sha256': old_manifest.get('sha256'),
            'arms': {},
        }
        for old_arm, item in old_models.items():
            historical_evaluation['arms'][old_arm] = {
                'checkpoint_sha256': item['checkpoint_sha256'],
                'validation': item['metrics']['contexts'],
                'target_contrasts': _contrast_report(selected_groups, item['predictions']),
            }
        del old_models
    # Freeze identical update sampling before training any arm.
    rng=random.Random(SEED)
    train_keys=sorted(k for k,r in selected_groups.items() if r[0]['split']=='train')
    trace=[]
    for _ in range(args.updates):
        k=rng.choice(train_keys); rows=selected_groups[k]
        ix=sorted(rng.sample(range(len(rows)),min(args.candidates_per_update,len(rows))))
        trace.append({'group':list(k),'indices':ix,'objective':rng.choice(train.OBJECTIVES),
                      'budget':rng.choice(train.BUDGET_FRACTIONS)})
    init_seed=SEED
    random.seed(init_seed); np.random.seed(init_seed); torch.manual_seed(init_seed); torch.cuda.manual_seed_all(init_seed)
    base=train.CandidateQoRModel()
    initial_state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    init_hash=train.model_state_sha256(base)
    arms={}
    fixed_key = ('machsuite-stencil2d', 'zcu104', 10.0)
    fixed_rows = selected_groups[fixed_key]
    diagnostic_steps = {1, 25, 50, 100, 200, args.updates}
    for arm in ('numerical','hardware','candidate_target'):
        if time.monotonic()>deadline: raise TimeoutError('GPU time cap reached')
        torch.cuda.reset_peak_memory_stats(device)
        random.seed(init_seed); np.random.seed(init_seed); torch.manual_seed(init_seed); torch.cuda.manual_seed_all(init_seed)
        if arm=='candidate_target': model=CandidateOnlyModel().to(device)
        else:
            model=train.CandidateQoRModel().to(device); model.load_state_dict(initial_state)
        model_hash=train.model_state_sha256(model) if arm!='candidate_target' else None
        if arm!='candidate_target' and model_hash!=init_hash: raise ValueError('Graph arms initial states differ')
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        losses=[]; gradients=[]; relation_gradients=[]; fixed_curve=[]; arm_start=time.monotonic()
        fixed_curve.append({'step': 0, **_fixed_training_diagnostic(
            model, arm, fixed_key, fixed_rows, graph_cache, candidate_cache,
            demand_cache, mean, std, rank_scales, device)})
        for step,item in enumerate(trace,1):
            if time.monotonic()>deadline: raise TimeoutError('GPU time cap reached during training')
            key=tuple(item['group']); rows=selected_groups[key]
            batch=[rows[i] for i in item['indices']]
            graph=_graph_at_requested_budget(graph_cache, key, item['budget'], arm,
                        args.budget_independent_physical_qor).to(device)
            c,d=_collate(batch,graph,candidate_cache,demand_cache,device)
            target=(torch.tensor([r['transformed'] for r in batch],dtype=torch.float32,device=device)-mean)/std
            if arm=='candidate_target': z=model(graph.hardware_context,c,d)
            else: z=model(graph,c,d,arm=arm)
            reg=F.smooth_l1_loss(z,target)
            budget=[x*item['budget'] for x in batch[0]['capacities']]
            rank,pairs=train.ranking_loss(z,batch,item['objective'],budget,mean,std,
                                           return_pair_count=True,objective_scales=rank_scales)
            loss=reg+.1*rank
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            opt.zero_grad(set_to_none=True); loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
            relation_grad = (fast_probe._hardware_relation_gradient_norm(model)
                             if arm != 'candidate_target' else 0.0)
            relation_gradients.append(relation_grad)
            gradients.append({'step':step,'preclip_gradient_norm':float(norm),
                              'gradient_clipped':bool(float(norm)>1.0),
                              'ranking_pairs':pairs,
                              'weighted_rank_loss':float((.1*rank).detach()),'regression_loss':float(reg.detach())})
            opt.step(); losses.append(float(loss.detach()))
            if step in diagnostic_steps:
                fixed_curve.append({'step':step, **_fixed_training_diagnostic(
                    model, arm, fixed_key, fixed_rows, graph_cache, candidate_cache,
                    demand_cache, mean, std, rank_scales, device)})
        val_groups={k:r for k,r in selected_groups.items() if r[0]['split']=='val'}
        # Evaluate small selected pools at both requested budgets and record
        # budget-induced physical QoR prediction changes separately.
        report=_evaluate(model,arm,val_groups,graph_cache,candidate_cache,demand_cache,mean,std,device,
                         budget_independent=args.budget_independent_physical_qor)
        ablation=None
        if arm=='hardware':
            original={key:{frac:dict(arms) for frac,arms in by_frac.items()}
                      for key,by_frac in graph_cache.items()}
            for key in graph_cache:
                for frac in graph_cache[key]:
                    graph_cache[key][frac]['hardware']=graph_cache[key][frac]['numerical']
            disabled=_evaluate(model,arm,val_groups,graph_cache,candidate_cache,demand_cache,mean,std,device,
                               budget_independent=args.budget_independent_physical_qor)
            graph_cache.update(original)
            embedding_shifts=[]
            with torch.no_grad():
                for key, rows in val_groups.items():
                    full_graph=graph_cache[key][1.0]['hardware'].to(device)
                    no_edge_graph=graph_cache[key][1.0]['numerical'].to(device)
                    full_memory,full_pool=model.graph.memories(full_graph)
                    no_memory,no_pool=model.graph.memories(no_edge_graph)
                    active=torch.nonzero(full_graph.matched_mask[0],as_tuple=False).flatten().tolist()
                    if active:
                        a,b=full_memory[:,active],no_memory[:,active]
                        embedding_shifts.append({
                            'kernel':key[0],'profile':key[1],'clock_ns':key[2],
                            'relative_action_memory_l2':float((a-b).norm()/a.norm().clamp_min(1e-12)),
                            'pooled_memory_l2':float((full_pool-no_pool).norm())})
            ablation={'disabled_metrics':{'contexts':disabled['contexts']},
                      'enabled_metrics':{'contexts':report['contexts'],
                      'embedding_shifts':embedding_shifts,
                      'mean_relative_action_memory_l2':float(np.mean(
                          [x['relative_action_memory_l2'] for x in embedding_shifts])) if embedding_shifts else None}}
        ckpt=out/f'{arm}_probe.pt'
        checkpointing.atomic_torch_save(ckpt,{'schema':run_schema,'cohort_sha256':manifest['sha256'],
            'arm':arm,'model':{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
            'normalization_mean':mean.cpu(),'normalization_std':std.cpu()})
        arms[arm]={'initial_model_sha256':model_hash,'final_model_sha256':train.model_state_sha256(model),
            'updates':len(losses),'loss_first':losses[0],'loss_last':losses[-1],
            'loss_min':min(losses),'training_and_validation_seconds':time.monotonic()-arm_start,
            'peak_cuda_memory_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),
            'ranking_gradient_audit':gradients,
            'clipped_update_count':sum(item['gradient_clipped'] for item in gradients),
            'fixed_training_context_diagnostic':fixed_curve,
            'hardware_relation_gradient_norm_mean':float(np.mean(relation_gradients)),
            'hardware_relation_gradient_norm_max':float(np.max(relation_gradients)),
            'validation':{'contexts':report['contexts']},'hardware_edge_ablation':ablation,
            'checkpoint_sha256':fast_probe.overlay.file_hash(ckpt)}
        del model,opt; torch.cuda.empty_cache()
    elapsed=time.monotonic()-gpu_start
    if elapsed>args.gpu_limit_seconds: raise TimeoutError('GPU time cap exceeded')
    # Paired exact-assignment clock/device contrasts for each arm.
    contrasts={}
    for arm in ('numerical','hardware','candidate_target'):
        model_class=CandidateOnlyModel if arm=='candidate_target' else train.CandidateQoRModel
        ckpt=torch.load(out/f'{arm}_probe.pt',map_location=device,weights_only=False)
        if ckpt.get('schema') != run_schema:
            raise ValueError(f'Checkpoint schema mismatch: {arm}')
        model=model_class().to(device); model.load_state_dict(ckpt['model']); model.eval()
        pred={}
        with torch.no_grad():
            for key,rows in selected_groups.items():
                if rows[0]['split']!='val': continue
                for frac in (1.0,.5,.1):
                    graph=graph_cache[key][frac][arm if arm in ('hardware','numerical') else 'hardware'].to(device)
                    if args.budget_independent_physical_qor:
                        graph=graph_cache[key][1.0][arm if arm in ('hardware','numerical') else 'hardware'].to(device)
                    c,d=_collate(rows,graph,candidate_cache,demand_cache,device)
                    z=(model(graph.hardware_context,c,d) if arm=='candidate_target'
                       else model(graph,c,d,arm=arm))
                    physical=torch.expm1(z*std+mean).clamp_min(0).cpu().numpy()
                    pred[(key,frac)]={r['config_hash']:y for r,y in zip(rows,physical)}
        by_budget={}
        for frac in (1.0,.5,.1):
            by_budget[frac]=_target_contrasts(selected_groups,{(key,h):y
                for (key,b),m in pred.items() if b==frac for h,y in m.items()})
        budget_delta_pairs={}
        for left_fraction,right_fraction in ((1.0,.5),(1.0,.1)):
            deltas=[]
            for key in val_groups:
                common=set(pred[(key,left_fraction)])&set(pred[(key,right_fraction)])
                deltas += [np.abs(pred[(key,left_fraction)][h]-pred[(key,right_fraction)][h]).tolist()
                           for h in common]
            budget_delta_pairs[f'{left_fraction:g}_vs_{right_fraction:g}']={
                'mean_absolute_physical_prediction_delta':np.mean(deltas,axis=0).tolist() if deltas else None,
                'exactly_invariant':bool(deltas) and bool(np.array_equal(np.asarray(deltas),np.zeros_like(deltas))) }
        contrasts[arm]={'clock_device':by_budget,
                        'summary_by_budget':{str(frac):_summarize_target_contrasts(rows)
                                             for frac, rows in by_budget.items()},
                        'budget_qor_delta_by_budget_pair':budget_delta_pairs,
                        'mean_abs_budget_induced_physical_prediction_delta':
                            budget_delta_pairs['1_vs_0.5']['mean_absolute_physical_prediction_delta']}
    output={'status':'completed','manifest_sha256':manifest['sha256'],
        'prep_seconds':prep,'feature_seconds':features_seconds,'gpu_seconds':elapsed,
        'gpu_name':torch.cuda.get_device_name(device),'gpu_uuid':args.gpu_uuid,
        'gpu_inventory_before_launch':gpu,'updates_per_arm':args.updates,
        'candidates_per_update':args.candidates_per_update,
        'sampling_trace_sha256':fast_probe._canonical_hash(trace),
        'initial_graph_arm_state_sha256':init_hash,'arms':arms,'target_contrasts':contrasts}
    output['historical_100_update_comparison'] = historical_evaluation
    checkpointing.atomic_json(out/'results.json',output)
    checkpointing.atomic_json(out/'status.json',{'status':'completed','manifest_sha256':manifest['sha256'],
                               'gpu_seconds':elapsed})
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu-name',default='A2'); ap.add_argument('--gpu-uuid',default='GPU-1a87c5a7-652d-d582-29e8-a5fccb11ce35')
    ap.add_argument('--updates',type=int,default=100); ap.add_argument('--candidates-per-update',type=int,default=8)
    ap.add_argument('--preparation-limit-seconds',type=int,default=600); ap.add_argument('--gpu-limit-seconds',type=int,default=1200)
    ap.add_argument('--compare-checkpoint-dir',type=Path)
    ap.add_argument('--compare-manifest',type=Path)
    ap.add_argument('--budget-independent-physical-qor',action='store_true',
                    help='Versioned ablation: physical predictions use full-capacity graph context; budgets only filter feasibility/ranking pairs')
    args=ap.parse_args()
    if not 1<=args.updates<=600: ap.error('updates must be 1..600')
    result=run(args); print(json.dumps({'status':result['status'],'prep_seconds':result['prep_seconds'],
        'feature_seconds':result['feature_seconds'],'gpu_seconds':result['gpu_seconds'],
        'cohort':result['manifest_sha256']},indent=2))

if __name__=='__main__': main()
