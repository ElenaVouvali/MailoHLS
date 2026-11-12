"""Objective ranking and evaluation shared by the paper GNN workflow.

No model, optimizer, or data loader lives here. Counts are physical quantities;
QoR is supplied by the authenticated paper measurement policy, never rebuilt
from a guessed resource average. Unknown feasibility is a third state.
"""
from __future__ import annotations

from collections import defaultdict
import math

OBJECTIVES = ('PARETO_LATENCY', 'PARETO_AREA', 'PARETO_ADP')


def feasibility(resources, budget, *, measured_success):
    if not measured_success:
        return None
    if len(resources) != 4 or len(budget) != 4:
        raise ValueError('Expected BRAM, DSP, FF, LUT in physical units')
    known = []
    for demand, limit in zip(resources, budget):
        if limit is None or not math.isfinite(limit) or limit < 0:
            raise ValueError('Budget must contain four finite nonnegative limits')
        if demand is None or not math.isfinite(demand) or demand < 0:
            known.append(None)
        else:
            known.append(demand <= limit)
    if False in known:
        return False  # A measured exceeding demand proves infeasibility.
    return True if all(x is True for x in known) else None


def objective_log2(log_latency, log_area, objective):
    """Inputs already follow the paper's sealed unit and area-floor policy."""
    if objective == 'PARETO_LATENCY':
        return log_latency
    if objective == 'PARETO_AREA':
        return log_area
    if objective == 'PARETO_ADP':
        return log_latency + log_area
    raise ValueError(objective)


def ranking_loss(prediction, measured, contexts, feasible, design_ids,
                 *, relative_tie=0.05, temperature=1.0):
    """Equal-context loss over unique, measured-feasible complete designs.

    Call separately for each budget, then average views within the physical
    kernel/device/clock context before averaging contexts. Do not expand the
    regression dataset by budget. Contexts must include device AND clock.
    """
    import torch
    from torch.nn import functional as F
    if temperature <= 0 or not 0 <= relative_tie < 1:
        raise ValueError('Invalid ranking policy')
    if not (len(prediction) == len(measured) == len(contexts) == len(feasible) == len(design_ids)):
        raise ValueError('Ranking arrays differ in length')
    groups = defaultdict(list)
    seen = set()
    for i, (context, design, valid) in enumerate(zip(contexts, design_ids, feasible)):
        key = (tuple(context), design)
        if key in seen:
            continue
        seen.add(key)
        if valid is True:
            groups[tuple(context)].append(i)
    losses = []
    for ids in groups.values():
        pred = prediction[ids].reshape(-1)
        truth = measured[ids].reshape(-1)
        delta = truth[:, None] - truth[None, :]
        valid = torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1)
        valid &= torch.isfinite(delta) & (delta.abs() > math.log2(1 + relative_tie))
        if valid.any():
            margin = pred[:, None] - pred[None, :]
            losses.append(F.softplus(-delta[valid].sign() * margin[valid] / temperature).mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0


def evaluate(rows, objective):
    """Evaluate validation pools, then aggregate contexts and kernel families.

    Each row needs family, kernel, device, clock, budget_id, design_id,
    measured_success, resources, predicted_resources, budget, log_latency,
    log_area, predicted_log_latency, predicted_log_area. Log measurements
    must be null when unknown. Timing qualification is reported separately.
    """
    from scipy.stats import kendalltau
    pools = defaultdict(dict)
    coverage = defaultdict(int)
    for row in rows:
        if row.get('split') != 'val':
            raise ValueError('Development evaluator accepts validation rows only')
        coverage['input_views'] += 1
        key = tuple(row[k] for k in ('family', 'kernel', 'device', 'clock', 'budget_id'))
        design = row['design_id']
        if design in pools[key]:
            if pools[key][design] != row:
                raise ValueError('Conflicting measurements for one candidate identity')
            coverage['duplicate_views'] += 1
            continue
        pools[key][design] = row
    reports = []
    for key, unique in sorted(pools.items()):
        candidates = []
        for design, row in sorted(unique.items()):
            measured = feasibility(row['resources'], row['budget'], measured_success=row['measured_success'])
            predicted = feasibility(row['predicted_resources'], row['budget'], measured_success=True)
            coverage['resource_' + {True: 'feasible', False: 'infeasible', None: 'unknown'}[measured]] += 1
            coverage['timing_' + str(row.get('timing_qualified', 'unknown'))] += 1
            names = ('log_latency', 'log_area', 'predicted_log_latency', 'predicted_log_area')
            needed = (0, 2) if objective == 'PARETO_LATENCY' else (1, 3) if objective == 'PARETO_AREA' else range(4)
            values = [row.get(name) for name in names]
            if not row['measured_success'] or any(values[i] is None or not math.isfinite(values[i]) for i in needed):
                coverage['unknown_qor'] += 1
                continue
            true = objective_log2(values[0], values[1], objective)
            pred = objective_log2(values[2], values[3], objective)
            candidates.append((design, true, pred, measured, predicted))
        known = [r for r in candidates if r[3] is True]
        allowed = [r for r in candidates if r[4] is True]
        selected = min(allowed, key=lambda r: (r[2], r[0])) if allowed else None
        # tau-b is scipy's default, including the historical GNN environment
        # whose kendalltau predates the explicit variant keyword.
        tau = kendalltau([r[1] for r in known], [r[2] for r in known]).correlation if len(known) > 1 else math.nan
        regret = None
        if known and selected and selected[3] is True:
            regret = math.expm1((selected[1] - min(r[1] for r in known)) * math.log(2))
        reports.append(dict(zip(('family', 'kernel', 'device', 'clock', 'budget_id'), key),
            candidates=len(candidates), measured_feasible=len(known),
            kendall_tau_b=float(tau) if math.isfinite(tau) else None,
            feasible_regret_at_1=regret,
            log_qor_mae=sum(abs(r[1]-r[2]) for r in candidates)/len(candidates) if candidates else None,
            feasible_selection_rate=float(selected is not None and selected[3] is True),
            selected_design=selected[0] if selected else None))
    metrics = ('kendall_tau_b', 'feasible_regret_at_1', 'log_qor_mae', 'feasible_selection_rate')
    def aggregate(items):
        return {m: sum(x[m] for x in items if x[m] is not None)/sum(x[m] is not None for x in items)
                if any(x[m] is not None for x in items) else None for m in metrics}
    # Average budget views inside contexts, then contexts inside kernels,
    # kernels inside families, and finally independent families.
    level = reports
    for fields in (('family','kernel','device','clock'), ('family','kernel'), ('family',)):
        groups = defaultdict(list)
        for row in level:
            groups[tuple(row[f] for f in fields)].append(row)
        level = [dict(zip(fields, key), **aggregate(items)) for key, items in sorted(groups.items())]
    return {'objective': objective, 'coverage': dict(coverage), 'pools': reports,
            'families': level, 'family_macro': aggregate(level),
            'regret_policy': 'conditional on measured-feasible selection; failures counted in feasible_selection_rate',
            'pool_policy': 'unique complete measured configurations within kernel/device/clock/requested budget; no synthesized or imputed labels',
            'feasibility_provenance': 'physical measured demands versus requested physical limits; timing separate',
            'log_units': 'log2 of sealed paper-policy QoR'}
