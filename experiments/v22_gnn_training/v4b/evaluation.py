"""V4b validation diagnostics; all regrets refer to the fixed measured pool.

Budget views are correlated. Family macro metrics average contexts within a
family first. Oracle feasibility is diagnostic only and never changes inputs.
"""
from collections import defaultdict
from itertools import combinations
import numpy as np
import torch
from scipy.stats import spearmanr

from .models import pairwise_pool_order
from .preferences import _ambiguous_zero_area

BUDGETS = (1., .75, .5, .25, .1)


def modules(model):
    return list(model.values()) if isinstance(model, dict) else [model]


def alter_memory(memory, pooled, graph, mode):
    if mode == 'correct':
        return memory, pooled
    active = torch.where(graph.registered_mask[0] & graph.matched_mask[0])[0]
    result = memory.clone()
    if mode == 'zero':
        return torch.zeros_like(memory), torch.zeros_like(pooled)
    if len(active):
        if mode == 'shuffled':
            generator = torch.Generator().manual_seed(123)
            order = torch.randperm(len(active), generator=generator).to(active.device)
            result[:, active] = memory[:, active[order]]
        elif mode == 'global_mean':
            result[:, active] = memory[:, active].mean(1, keepdim=True)
        else:
            raise ValueError(mode)
    return result, pooled


def predict_pool(arm, model, prepared, group, device, mode='correct'):
    graph = prepared['graph_cache'][(group['kernel'], group['profile'], group['clock'])].clone().to(device)
    features = [prepared['inputs'][key] for key in group['identities']]
    candidates = torch.stack([x[0] for x in features]).to(device)
    demands = torch.stack([x[1] for x in features]).to(device)
    base = model['base'] if isinstance(model, dict) else model
    if arm == 'no_graph':
        memory = pooled = None
    else:
        physical_graph = graph
        if mode == 'target_mismatched':
            other = next((g for g in prepared['groups'] if g['kernel'] == group['kernel']
                          and g['profile'] != group['profile'] and g['clock'] == group['clock']), None)
            if other is None:
                raise ValueError('Target mismatch requires an exact alternate physical context')
            physical_graph = prepared['graph_cache'][(other['kernel'], other['profile'], other['clock'])].clone().to(device)
        if arm == 'target_film':
            profile = physical_graph if mode == 'target_mismatched' else graph
            profile_id = other['profile'] if mode == 'target_mismatched' else group['profile']
            memory, pooled = base.encode(profile, prepared['profile_features'][(profile_id, group['clock'])].clone().to(device))
        else:
            memory, pooled = base.encode(physical_graph)
        if mode != 'target_mismatched':
            memory, pooled = alter_memory(memory, pooled, graph, mode)
    def forward(fraction):
        return (base(graph, candidates, demands, fraction) if arm == 'no_graph' else
                base.from_memory(graph, memory, pooled, candidates, demands, fraction))
    initial = forward(1.)
    mean, std = prepared['mean'].to(device), prepared['std'].to(device)
    physical = torch.expm1(initial['qor_z'] * std + mean).clamp_min(0)
    if not torch.isfinite(physical).all():
        raise ValueError('Nonfinite physical prediction')
    n = len(group['rows'])
    pair_matrices, drift = {}, 0.
    for fraction in BUDGETS:
        output = forward(fraction)
        drift = max(drift, float((output['qor_z'] - initial['qor_z']).abs().max()))
        if isinstance(model, dict):
            indices = torch.triu_indices(n, n, 1, device=device)
            scores, _ = model['compare'](graph, memory, candidates[indices[0]], demands[indices[0]],
                                         candidates[indices[1]], demands[indices[1]], fraction)
            matrix = physical.new_zeros((n, n))
            matrix[indices[0], indices[1]] = scores
            matrix[indices[1], indices[0]] = -scores
        else:
            score = output['preference_score']
            matrix = score[None, :] - score[:, None]  # positive means left preferred
        pair_matrices[fraction] = matrix.cpu()
    if drift > 1e-7:
        raise AssertionError(f'Budget affected physical QoR: normalized drift {drift}')
    return physical.cpu().numpy(), pair_matrices, drift


def _selection(order, feasible, adp, reference):
    if not len(reference):
        return {'selected_hash_index': None, 'selected_feasible': None, 'top1_regret': None,
                'top3_regret': None, 'selection_failure': None}
    best = float(adp[reference].min())
    chosen = order[0] if order else None
    shortlist = [i for i in order[:3] if feasible[i]]
    ok = chosen is not None and bool(feasible[chosen])
    return {'selected_hash_index': chosen, 'selected_feasible': bool(ok),
            'top1_regret': float((adp[chosen] - best) / max(abs(best), 1e-12)) if ok else None,
            'top3_regret': float((min(adp[i] for i in shortlist) - best) / max(abs(best), 1e-12)) if shortlist else None,
            'selection_failure': float(not ok)}


def _mean(values):
    valid = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(valid)) if valid else None


def summarize(rows):
    keys = ('selection_failure', 'top1_regret', 'top3_regret', 'pair_accuracy',
            'feasibility_precision', 'feasibility_recall', 'false_feasible_rate',
            'false_infeasible_rate', 'oracle_top1_regret', 'oracle_top3_regret', 'adp_spearman')
    return {key: _mean([row.get(key) for row in rows]) for key in keys}


def evaluate(arm, model, prepared, device, mode='correct'):
    from .audit import _winner
    from .quality import adp_valid
    strict_quality = prepared.get('quality_policy') == 'reported-zero-adp-ambiguous-v1'
    for module in modules(model):
        module.eval()
    contexts, predictions, physical_rows = [], {}, []
    with torch.no_grad():
        for group in prepared['groups']:
            if group['split'] != 'val':
                continue
            predicted, matrices, drift = predict_pool(arm, model, prepared, group, device, mode)
            true = np.asarray([r['physical'] for r in group['rows']])
            caps = np.asarray(group['rows'][0]['capacities'])
            adp = true[:, 0] * (true[:, 1:] / caps).mean(1)
            pred_adp = predicted[:, 0] * (predicted[:, 1:] / caps).mean(1)
            valid = np.asarray([adp_valid(r) if strict_quality else not _ambiguous_zero_area(r) for r in group['rows']])
            ids = [r['config_hash'] for r in group['rows']]
            predictions[(group['kernel'], group['profile'], group['clock'])] = {
                h: (true[i], predicted[i]) for i, h in enumerate(ids)}
            rho = float(spearmanr(adp[valid], pred_adp[valid]).statistic) if (
                valid.sum() > 1 and np.ptp(adp[valid]) > 0 and np.ptp(pred_adp[valid]) > 0) else None
            physical_rows.append({'kernel': group['kernel'], 'family': group['family'],
                'device': group['profile'], 'clock_ns': group['clock'],
                'mae_latency_bram_dsp_ff_lut': np.abs(predicted - true).mean(0).tolist(),
                'budget_normalized_prediction_drift': drift,
                'physical_prediction': predicted.tolist(), 'assignment_hashes': ids})
            for fraction in BUDGETS:
                pf = (predicted[:, 1:] <= caps * fraction).all(1)
                mf = (true[:, 1:] <= caps * fraction).all(1)
                tp, fp, fn, tn = [int(x.sum()) for x in (pf & mf, pf & ~mf, ~pf & mf, ~pf & ~mf)]
                reference = np.flatnonzero(mf & valid)
                matrix = matrices[fraction]
                def rank(mask):
                    ix = np.flatnonzero(mask & valid).tolist()
                    if not ix:
                        return []
                    local = pairwise_pool_order(matrix[ix][:, ix], [ids[i] for i in ix])
                    return [ix[j] for j in local]
                deploy = _selection(rank(pf), mf, adp, reference)
                oracle = _selection(rank(mf), mf, adp, reference)
                pairs = group['budget_pairs'][fraction]
                pair_accuracy = _mean([float(matrix[a, b] > 0) for a, b, _ in pairs])
                winner, margin, count = _winner([r for i,r in enumerate(group['rows']) if valid[i]], fraction, 'PARETO_ADP')
                selected = deploy.pop('selected_hash_index')
                contexts.append({'context': group['key'], 'kernel': group['kernel'], 'family': group['family'],
                    'device': group['profile'], 'clock_ns': group['clock'], 'budget_fraction': fraction,
                    **deploy, 'oracle_top1_regret': oracle['top1_regret'], 'oracle_top3_regret': oracle['top3_regret'],
                    'selected_hash': ids[selected] if selected is not None else None,
                    'measured_winner': winner, 'measured_winner_margin': margin,
                    'measured_feasible_count': count, 'predicted_feasible_count': int(pf.sum()),
                    'pair_count': len(pairs), 'pair_accuracy': pair_accuracy, 'adp_spearman': rho,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'feasibility_precision': tp / (tp + fp) if tp + fp else None,
                    'feasibility_recall': tp / (tp + fn) if tp + fn else None,
                    'false_feasible_rate': fp / (fp + tn) if fp + tn else None,
                    'false_infeasible_rate': fn / (fn + tp) if fn + tp else None,
                    'constant_or_no_meaningful_pair': len(pairs) == 0})
    families = {family: summarize([r for r in contexts if r['family'] == family])
                for family in sorted({r['family'] for r in contexts})}
    macro = {key: _mean([r[key] for r in families.values()]) for key in next(iter(families.values()), {})}
    transitions = []
    for a, b in combinations(contexts, 2):
        if a['context'] != b['context']:
            continue
        meaningful = (a['measured_winner'] is not None and b['measured_winner'] is not None
                      and a['measured_winner'] != b['measured_winner']
                      and (a['measured_winner_margin'] or 0) > .05 and (b['measured_winner_margin'] or 0) > .05)
        if meaningful:
            transitions.append({'context': a['context'], 'budgets': [a['budget_fraction'], b['budget_fraction']],
                'correct': a['selected_hash'] == a['measured_winner'] and b['selected_hash'] == b['measured_winner']})
    deltas = []
    for left, right in combinations(predictions, 2):
        if left[0] != right[0] or not (left[1] == right[1] or left[2] == right[2]):
            continue
        common = sorted(predictions[left].keys() & predictions[right].keys())
        if common:
            error = [np.abs((predictions[right][h][1] - predictions[left][h][1]) -
                            (predictions[right][h][0] - predictions[left][h][0])) for h in common]
            deltas.append({'kernel': left[0], 'left': list(left[1:]), 'right': list(right[1:]),
                           'matched_assignments': len(common), 'delta_mae_latency_bram_dsp_ff_lut': np.mean(error, axis=0).tolist()})
    score = None if macro.get('selection_failure') is None else (
        macro['selection_failure'] + (macro['top1_regret'] if macro['top1_regret'] is not None else 1.))
    return {'family_macro': macro, 'per_family': families, 'contexts': contexts,
            'physical_contexts': physical_rows, 'paired_physical_delta_errors': deltas,
            'meaningful_budget_transitions': transitions, 'memory_mode': mode,
            'selection_score': score, 'validation_preference_accuracy': macro.get('pair_accuracy'),
            'feasible_top1_regret_macro': macro.get('top1_regret'),
            'feasible_top3_regret_macro': macro.get('top3_regret'),
            'test_kernels_loaded': [], 'scope': 'fixed measured-pool; two validation families; budgets correlated'}


def controlled_action_probe(model, prepared, device):
    """First hash-ordered measured training pair differing at one action.

    Selection reads directive tensors only. Perturbations measure score
    sensitivity, not attention causality. Missing support is explicit.
    """
    if not isinstance(model, dict):
        return {'status': 'not_an_action_difference_arm'}
    for group in prepared['groups']:
        if group['split'] != 'train':
            continue
        features = [prepared['inputs'][key] for key in group['identities']]
        for i, j in combinations(range(len(features)), 2):
            changed = (features[i][0] != features[j][0]).any(-1)
            if int(changed.sum()) != 1:
                continue
            slot = int(torch.where(changed)[0][0])
            graph = prepared['graph_cache'][(group['kernel'], group['profile'], group['clock'])].clone().to(device)
            if not bool(graph.matched_mask[0, slot]):
                continue
            other = torch.where(graph.registered_mask[0] & graph.matched_mask[0] & ~changed.to(device))[0]
            with torch.no_grad():
                memory, _ = model['base'].encode(graph)
                a, da = [x.unsqueeze(0).to(device) for x in features[i]]
                b, db = [x.unsqueeze(0).to(device) for x in features[j]]
                def compare(mem):
                    score, weights = model['compare'](graph, mem, a, da, b, db, 1.)
                    return float(score[0]), float(weights[0, slot])
                correct, attention = compare(memory)
                altered = memory.clone(); altered[0, slot] += .1
                aligned, _ = compare(altered)
                unrelated = None
                if len(other):
                    altered = memory.clone(); altered[0, int(other[0])] += .1
                    unrelated, _ = compare(altered)
                return {'status': 'measured_training_pair_diagnostic', 'context': group['key'],
                        'candidate_hashes': [group['rows'][k]['config_hash'] for k in (i, j)],
                        'changed_absolute_slot': slot, 'attention_to_changed_slot': attention,
                        'correct_score': correct, 'aligned_perturbation_score': aligned,
                        'unrelated_perturbation_score': unrelated,
                        'aligned_direction_changed': bool((correct > 0) != (aligned > 0)),
                        'perturbation': 'add 0.1 to each of 128 coordinates at one matched action',
                        'interpretation': 'controlled local score sensitivity; attention magnitude does not establish causal importance'}
    return {'status': 'unsupported_no_exact_single_action_pair_in_fixed_training_pool'}
