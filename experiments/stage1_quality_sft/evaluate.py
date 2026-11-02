"""Score constrained-decoder predictions only against exact validation measurements."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from experiments.stage1_quality_sft.core import assignment, parse, validate_complete_assignment
from experiments.stage1_quality_sft.support import sha, normalize_kname, constrained_rhs_candidates


def evaluate(requests, predictions, candidate_pools=None, registry=None):
    frozen_ids = {row['request_id'] for row in requests}
    unknown_ids = sorted({p['request_id'] for p in predictions} - frozen_ids)
    if unknown_ids:
        raise ValueError(f'Prediction is outside frozen validation requests: {unknown_ids[:3]}')
    if candidate_pools is not None:
        enriched = []
        util_keys = ('bram_util_%', 'dsp_util_%', 'ff_util_%', 'lut_util_%')
        budget_keys = ('budget_frac_bram_18k', 'budget_frac_dsp', 'budget_frac_ff', 'budget_frac_lut')
        for request in requests:
            pool = candidate_pools[request['candidate_pool_id']]
            candidates = []
            for candidate in pool:
                values = dict(candidate, assignment_id=candidate['candidate_id'])
                values['feasible'] = all(float(values['utilization'][util]) / 100. <= float(request['budget'][budget]) + 1e-9
                                         for util, budget in zip(util_keys, budget_keys))
                candidates.append(values)
            feasible = [c for c in candidates if c['feasible']]
            if not feasible:
                raise ValueError(f'Frozen supported validation request has no feasible pool: {request["request_id"]}')
            best = min(c['latency'] for c in feasible)
            accepted_ids = set(request['sensitivity']['0.01']['accepted_candidate_ids'])
            enriched.append(dict(request, measured_candidates=candidates, feasible_candidates=feasible,
                                 best_latency=best, accepted_candidate_ids=sorted(accepted_ids)))
        requests = enriched
    by_id = {row['request_id']: row for row in requests}
    seen = set()
    per_request, family_rows = [], defaultdict(list)
    for prediction in predictions:
        rid = prediction['request_id']
        if rid in seen:
            raise ValueError(f'Duplicate prediction request: {rid}')
        seen.add(rid)
        request = by_id.get(rid)
        if request is None:
            raise ValueError(f'Prediction is outside frozen validation requests: {rid}')
        output_text = prediction.get('prediction', prediction.get('target', ''))
        try:
            output = assignment(output_text) if isinstance(output_text, str) else tuple(sorted(output_text.items()))
        except ValueError:
            output = ()
        pool = {tuple(sorted(c['assignment'].items())): c for c in request['measured_candidates']}
        measured = pool.get(output)
        expected_fields = set(next(iter(pool))) if pool else set()
        complete = set(dict(output)) == expected_fields
        legal = None
        if registry is not None and complete:
            try:
                validate_complete_assignment(dict(kernel_name=request['request_identity'][0], target=output_text),
                    registry, constrained_rhs_candidates, normalize_kname, parse)
                legal = True
            except (ValueError, KeyError):
                legal = False
        if measured is None:
            row = dict(request_id=rid, kernel=request['request_identity'][0], family=prediction.get('family', request.get('family') or 'unknown'),
                       exact_measured_match=False, measured_feasible=None, latency=None,
                       best_feasible_latency=request['best_latency'], regret=None,
                       within_1pct=None, within_5pct=None, prediction_complete=complete,
                       source_legal=legal, within_primary_acceptable_set=False)
        else:
            best = float(request['best_latency'])
            latency = float(measured['latency'])
            regret = max(0., latency / best - 1.) if best > 0 else None
            row = dict(request_id=rid, kernel=request['request_identity'][0], family=prediction.get('family', request.get('family') or 'unknown'),
                       exact_measured_match=True, measured_feasible=bool(measured['feasible']),
                       latency=latency, best_feasible_latency=best, regret=regret,
                       within_1pct=bool(regret is not None and regret <= .01 + 1e-12),
                       within_5pct=bool(regret is not None and regret <= .05 + 1e-12),
                       prediction_complete=complete, source_legal=legal,
                       within_primary_acceptable_set=bool(measured and measured['assignment_id'] in request.get('accepted_candidate_ids', [])))
        per_request.append(row)
        family_rows[row['family']].append(row)
    unmatched_requests = sorted(set(by_id) - seen)
    scored = [r for r in per_request if r['exact_measured_match']]
    feasible = [r for r in scored if r['measured_feasible']]
    result = dict(schema='stage1-quality-prediction-evaluation-v1', requests=len(by_id),
        predictions=len(predictions), missing_predictions=len(unmatched_requests),
        exact_measured_matches=len(scored), exact_match_coverage=len(scored)/len(per_request) if per_request else None,
        unknown_qor=len(per_request)-len(scored), measured_feasible=len(feasible),
        complete_assignment_count=sum(r['prediction_complete'] for r in per_request),
        complete_assignment_rate=sum(r['prediction_complete'] for r in per_request)/len(per_request) if per_request else None,
        source_legal_count=sum(r.get('source_legal') is True for r in per_request),
        source_illegal_count=sum(r.get('source_legal') is False for r in per_request),
        primary_near_optimal_match_count=sum(r['within_primary_acceptable_set'] for r in per_request),
        measured_infeasible=sum(r['measured_feasible'] is False for r in scored),
        mean_latency_regret_feasible=sum(r['regret'] for r in feasible)/len(feasible) if feasible else None,
        within_1pct=sum(r['within_1pct'] for r in feasible)/len(feasible) if feasible else None,
        within_5pct=sum(r['within_5pct'] for r in feasible)/len(feasible) if feasible else None,
        family_macro=dict(family_count=len(family_rows),
            exact_match_coverage=_macro(family_rows, 'exact_measured_match'),
            feasible_regret=_macro_regret(family_rows),
            within_1pct=_macro(family_rows, 'within_1pct'),
            within_5pct=_macro(family_rows, 'within_5pct')),
        per_request=per_request, missing_request_ids=unmatched_requests,
        qos_source='exact complete directive assignment in authenticated raw validation candidate bank',
        unmatched_qor='unknown; no estimate assigned')
    result['budget_response'] = budget_response(requests, predictions)
    result['nondefault_recall_single_reference_requests'] = nondefault_recall(requests, predictions, candidate_pools)
    return result


def nondefault_recall(requests, predictions, candidate_pools):
    """Recall only on requests with one 1%-acceptable measured reference."""
    if candidate_pools is None:
        return dict(status='candidate_pools_unavailable')
    pool_by_id = {key: {c['candidate_id']: c for c in vals} for key, vals in candidate_pools.items()}
    pred_by_id = {p['request_id']: p for p in predictions}
    counts = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0))
    supported = 0
    for request in requests:
        accepted = request['sensitivity']['0.01']['accepted_candidate_ids']
        if len(accepted) != 1 or request['request_id'] not in pred_by_id:
            continue
        supported += 1
        reference = pool_by_id[request['candidate_pool_id']][accepted[0]]['assignment']
        value = pred_by_id[request['request_id']].get('prediction', '')
        prediction = parse(value) if isinstance(value, str) else value
        for field, actual in reference.items():
            kind = field.split('_', 2)[1]
            if kind not in ('PIPE', 'UNROLL'):
                continue
            truth, guess = str(actual) != '0', str(prediction.get(field, '')) != '0'
            counts[kind]['tp' if truth and guess else 'fn' if truth else 'fp' if guess else 'tn'] += 1
    return dict(status='measured singleton requests only', request_count=supported,
        by_kind={kind: dict(v, recall=(v['tp']/(v['tp']+v['fn']) if v['tp']+v['fn'] else None),
                            precision=(v['tp']/(v['tp']+v['fp']) if v['tp']+v['fp'] else None))
                 for kind, v in counts.items()})


def _macro(groups, key):
    values = []
    for rows in groups.values():
        subset = [r[key] for r in rows if r.get(key) is not None]
        if subset:
            values.append(sum(map(float, subset))/len(subset))
    return sum(values)/len(values) if values else None


def _macro_regret(groups):
    vals = []
    for rows in groups.values():
        x = [r['regret'] for r in rows if r.get('regret') is not None and r.get('measured_feasible')]
        if x: vals.append(sum(x)/len(x))
    return sum(vals)/len(vals) if vals else None


def budget_response(requests, predictions):
    pred_by_id = {p['request_id']: assignment(p.get('prediction', p.get('target', ''))) if isinstance(
        p.get('prediction', p.get('target', '')), str) else tuple(sorted(p.get('prediction', p.get('target', {})).items()))
        for p in predictions}
    groups = defaultdict(list)
    for req in requests:
        identity = req['request_identity']
        base = (identity[0], identity[1], identity[2], identity[3], identity[4], identity[6])
        groups[base].append(req)
    counts = defaultdict(int)
    for group in groups.values():
        group.sort(key=lambda r: r['budget']['budget_frac_bram_18k'])
        for i, left in enumerate(group):
            for right in group[i+1:]:
                if left['request_id'] not in pred_by_id or right['request_id'] not in pred_by_id:
                    counts['unsupported_missing_prediction'] += 1; continue
                a, b = pred_by_id[left['request_id']], pred_by_id[right['request_id']]
                pred_changed = a != b
                # Deterministic measured oracle relation; not claimed as a causal
                # effect because several near-optimal designs may exist.
                la = min(left['feasible_candidates'], key=lambda c: (c['latency'], c['assignment_id']))['assignment_id']
                rb = min(right['feasible_candidates'], key=lambda c: (c['latency'], c['assignment_id']))['assignment_id']
                oracle_changed = la != rb
                if pred_changed and oracle_changed: counts['correctly_changed'] += 1
                elif not pred_changed and not oracle_changed: counts['correctly_invariant'] += 1
                else: counts['relation_mismatch'] += 1
                pools = [{tuple(sorted(c['assignment'].items())): c for c in r['measured_candidates']}
                         for r in (left, right)]
                matches = [pools[0].get(a), pools[1].get(b)]
                if all(matches):
                    counts['measured_outputs'] += 1
                    if not all(m['feasible'] for m in matches): counts['measured_feasibility_violation'] += 1
                else: counts['unsupported_measured_output'] += 1
    return dict(counts=dict(counts), interpretation='Budget-response relation uses exact-assignment measured best-design identity; it is descriptive, and does not prove budget causality.')


def select_checkpoint(reports):
    """Predeclared lexicographic checkpoint choice; refuses incomplete/unknown-only evidence."""
    if not reports:
        raise ValueError('No checkpoint evaluation reports')
    identities = {(r['report']['input_sha256']['requests'],
                   r['report']['input_sha256']['candidate_pools'],
                   r['report']['requests'], r['report']['predictions'],
                   r['report']['missing_predictions']) for r in reports}
    if len(identities) != 1:
        raise ValueError('Checkpoint reports do not use identical complete evaluation inputs')
    if any(r['report']['missing_predictions'] != 0 for r in reports):
        raise ValueError('Checkpoint selection requires every frozen validation request')
    if not any(r['report']['exact_measured_matches'] for r in reports):
        return dict(status='no_hardware_supported_checkpoint', selected=None,
            reason='Every prediction has unknown measured QoR; generated-only scores cannot establish hardware quality.')
    def score(item):
        report = item['report']
        regret = report['family_macro']['feasible_regret']
        return (report['measured_feasible']/report['requests'],
                -(float(regret) if regret is not None else float('inf')),
                report['exact_match_coverage'] or 0.,
                report.get('source_legal_count', 0)/report['requests'],
                report.get('complete_assignment_count', 0)/report['requests'],
                -int(item['step']))
    chosen = max(reports, key=score)
    return dict(status='selected_by_frozen_validation_criterion', selected=chosen['checkpoint'],
        step=int(chosen['step']), score=list(score(chosen)),
        priority=['measured feasible fraction of all requests',
                  'lower family-macro measured-pool latency regret among feasible exact matches',
                  'exact measured-match coverage', 'source legality',
                  'complete directive assignments', 'earlier step tie-break'],
        reports=[dict(checkpoint=r['checkpoint'], step=r['step'], score=list(score(r))) for r in reports])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--requests', required=True); ap.add_argument('--candidate-pools', required=True)
    ap.add_argument('--predictions', required=True); ap.add_argument('--registry', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args()
    requests_path, predictions_path = Path(args.requests), Path(args.predictions)
    with gzip.open(requests_path, 'rt', encoding='utf-8') if str(requests_path).endswith('.gz') else requests_path.open() as handle:
        requests = [json.loads(x) for x in handle if x.strip()]
    pool_path = Path(args.candidate_pools)
    with gzip.open(pool_path, 'rt', encoding='utf-8') if str(pool_path).endswith('.gz') else pool_path.open() as handle:
        pool_records = [json.loads(x) for x in handle if x.strip()]
    candidate_pools = {record['candidate_pool_id']: record['candidates'] for record in pool_records}
    predictions = [json.loads(x) for x in predictions_path.read_text().splitlines() if x]
    from experiments.stage1_quality_sft.support import load_registry
    registry = load_registry(args.registry)
    result = evaluate(requests, predictions, candidate_pools, registry)
    result['input_sha256'] = dict(requests=sha(requests_path), candidate_pools=sha(args.candidate_pools), predictions=sha(predictions_path))
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('requests','predictions','exact_measured_matches','unknown_qor','mean_latency_regret_feasible','within_1pct','within_5pct')}, indent=2))


if __name__ == '__main__':
    main()
