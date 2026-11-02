"""Authenticated full-cohort and target-preference coverage audit for V4b.

This is a CPU-only descriptive audit. It uses the existing V4 PreparedData
contract and never loads the reserved Kalman test kernel.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from experiments.v22_gnn_training import train
from experiments.v22_gnn_training.v4b import SCHEMA
from experiments.v22_gnn_training.v4b.preferences import preference_pairs


ROOT = Path(__file__).resolve().parents[3]
PROFILES = ('zcu104', 'u200')
CLOCKS = (3.33, 5.0, 10.0)
BUDGETS = (1.0, 0.75, 0.5, 0.25, 0.1)
PRIMARY_GAP = 0.05
ZERO_AREA_AMBIGUITY = ('machsuite-viterbi', 'u200', 5.0)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _family_map(split):
    result = {}
    families = [('train', f) for f in split['train_families']]
    families += [('val', f) for f in split['val_families']]
    for split_name, family in families:
        norm = re.sub(r'[^a-z0-9]', '', family.lower())
        matches = []
        for kernel in split[split_name + '_kernels']:
            knorm = re.sub(r'[^a-z0-9]', '', kernel.lower())
            if knorm.startswith(norm):
                matches.append(kernel)
        for kernel in matches:
            if kernel in result and result[kernel] != family:
                raise ValueError(f'Ambiguous family assignment for {kernel}')
            result[kernel] = family
    return result


def _area(row):
    return sum(float(value) / float(capacity) for value, capacity in
               zip(row['physical'][1:], row['capacities'])) / 4.0


def _objective(row, name):
    latency = float(row['physical'][0])
    if name == 'PARETO_LATENCY':
        return latency
    area = _area(row)
    if name == 'PARETO_AREA':
        return area
    if name == 'PARETO_ADP':
        return latency * area
    raise ValueError(f'Unsupported objective {name!r}')


def _ambiguous_zero_area(row):
    return (row['kernel'], row['profile'], float(row['clock'])) == ZERO_AREA_AMBIGUITY and _area(row) == 0.0


def _relative_gap(a, b):
    denom = max(abs(float(a)), abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / denom


def _winner(rows, fraction, objective):
    if not rows:
        return None, None, 0
    physical = np.asarray([row['physical'] for row in rows], dtype=np.float64)
    capacities = np.asarray([row['capacities'] for row in rows], dtype=np.float64)
    eligible = np.all(physical[:, 1:] <= capacities * fraction, axis=1)
    if objective in ('PARETO_AREA', 'PARETO_ADP'):
        eligible &= ~np.asarray([_ambiguous_zero_area(row) for row in rows], dtype=bool)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return None, None, 0
    area = np.mean(physical[:, 1:] / capacities, axis=1)
    values = (physical[:, 0] if objective == 'PARETO_LATENCY' else area if objective == 'PARETO_AREA'
              else physical[:, 0] * area)
    hashes = np.asarray([row['config_hash'] for row in rows], dtype=str)
    order = indices[np.lexsort((hashes[indices], values[indices]))]
    best = float(values[order[0]])
    gap = None if len(order) < 2 else _relative_gap(best, float(values[order[1]]))
    return rows[order[0]]['config_hash'], gap, int(len(indices))


def _budget_winners(rows):
    """Evaluate all five budgets with one QoR sort per physical context."""
    candidates = [r for r in rows if not _ambiguous_zero_area(r)]
    if not candidates:
        return {f'{fraction:g}': (None, None, 0) for fraction in BUDGETS}
    physical = np.asarray([r['physical'] for r in candidates], dtype=np.float64)
    capacities = np.asarray([r['capacities'] for r in candidates], dtype=np.float64)
    area = np.mean(physical[:, 1:] / capacities, axis=1)
    objective = physical[:, 0] * area
    ranked = sorted(range(len(candidates)), key=lambda i: (objective[i], candidates[i]['config_hash']))
    result = {}
    for fraction in BUDGETS:
        feasible = [i for i in ranked if np.all(physical[i, 1:] <= capacities[i] * fraction)]
        if not feasible:
            result[f'{fraction:g}'] = (None, None, 0)
        else:
            gap = None if len(feasible) < 2 else _relative_gap(objective[feasible[0]], objective[feasible[1]])
            result[f'{fraction:g}'] = (candidates[feasible[0]]['config_hash'], gap, len(feasible))
    return result


def _target_pairs():
    result = []
    for clock in CLOCKS:
        result.append(('device', ('zcu104', clock), ('u200', clock)))
    for profile in PROFILES:
        for left, right in ((3.33, 5.0), (3.33, 10.0), (5.0, 10.0)):
            result.append(('clock', (profile, left), (profile, right)))
    return result


def _coverage(records, family_by_kernel):
    grouped = defaultdict(dict)
    for row in records:
        key = (row['kernel'], row['profile'], float(row['clock']))
        mapping = grouped[key]
        previous = mapping.get(row['config_hash'])
        if previous is not None:
            if not np.allclose(previous['physical'], row['physical'], rtol=0, atol=1e-12):
                raise ValueError('Repeated assignment has conflicting physical measurements')
            continue
        mapping[row['config_hash']] = row

    target_pairs = []
    family_rows = defaultdict(lambda: defaultdict(list))
    budget_pairs = []
    for (kernel, profile, clock), assignment_map in sorted(grouped.items()):
        rows = list(assignment_map.values())
        winners = _budget_winners(rows)
        for i, low in enumerate(BUDGETS):
            for high in BUDGETS[i + 1:]:
                low_w, low_gap, low_n = winners[f'{low:g}']
                high_w, high_gap, high_n = winners[f'{high:g}']
                changed = low_w is not None and high_w is not None and low_w != high_w
                budget_pairs.append({
                    'kernel': kernel, 'family': family_by_kernel.get(kernel),
                    'split': rows[0]['split'], 'profile': profile, 'clock_ns': clock,
                    'low_budget': low, 'high_budget': high,
                    'candidate_count': len(assignment_map),
                    'low_feasible_count': low_n, 'high_feasible_count': high_n,
                    'low_winner': low_w, 'high_winner': high_w,
                    'winner_changed': changed,
                    'meaningful_reversal_5pct': bool(
                        changed and low_gap is not None and high_gap is not None
                        and low_gap > PRIMARY_GAP and high_gap > PRIMARY_GAP),
                    'low_winner_margin': low_gap, 'high_winner_margin': high_gap,
                })
    for kernel_number, kernel in enumerate(sorted({row['kernel'] for row in records}), 1):
        for dimension, left_target, right_target in _target_pairs():
            left = grouped.get((kernel, *left_target), {})
            right = grouped.get((kernel, *right_target), {})
            common = sorted(set(left) & set(right))
            if not common:
                summary = {'kernel': kernel, 'family': family_by_kernel.get(kernel),
                           'split': next((r['split'] for r in records if r['kernel'] == kernel), None),
                           'dimension': dimension, 'left': list(left_target), 'right': list(right_target),
                           'matched_assignments': 0, 'status': 'unsupported_no_exact_assignment_intersection'}
                target_pairs.append(summary)
                family_rows[family_by_kernel.get(kernel, kernel)][dimension].append(summary)
                continue
            winner_comparisons = {}
            lrows = [left[h] for h in common]
            rrows = [right[h] for h in common]
            # Vectorize physical-contrast summaries. The original Python loop
            # repeatedly converted capacities and recomputed the same
            # relative-gap values for three sensitivity thresholds, making a
            # full 97k-row coverage audit needlessly slow.
            lphysical = np.asarray([r['physical'] for r in lrows], dtype=np.float64)
            rphysical = np.asarray([r['physical'] for r in rrows], dtype=np.float64)
            lcapacity = np.asarray([r['capacities'] for r in lrows], dtype=np.float64)
            rcapacity = np.asarray([r['capacities'] for r in rrows], dtype=np.float64)
            larea = np.mean(lphysical[:, 1:] / lcapacity, axis=1)
            rarea = np.mean(rphysical[:, 1:] / rcapacity, axis=1)
            lmetrics = {'latency_ms': lphysical[:, 0], 'area_fraction': larea,
                        'adp': lphysical[:, 0] * larea}
            rmetrics = {'latency_ms': rphysical[:, 0], 'area_fraction': rarea,
                        'adp': rphysical[:, 0] * rarea}
            deltas = {}
            meaningful = Counter()
            for name in lmetrics:
                av, bv = lmetrics[name], rmetrics[name]
                deltas[name] = bv - av
                relative = np.abs(bv - av) / np.maximum(
                    np.maximum(np.abs(av), np.abs(bv)), 1e-12)
                for threshold in (0.01, 0.05, 0.10):
                    meaningful[f'{name}_gt_{int(threshold*100)}pct'] = int(
                        np.count_nonzero(relative > threshold))
            for fraction in BUDGETS:
                lw, lgap, ln = _winner(lrows, fraction, 'PARETO_ADP')
                rw, rgap, rn = _winner(rrows, fraction, 'PARETO_ADP')
                changed = lw is not None and rw is not None and lw != rw
                meaningful_change = (changed and lgap is not None and rgap is not None
                                     and lgap > PRIMARY_GAP and rgap > PRIMARY_GAP)
                winner_comparisons[f'{fraction:g}'] = {
                    'left_winner': lw, 'right_winner': rw, 'winner_changed': changed,
                    'meaningful_reversal_5pct': bool(meaningful_change),
                    'left_winner_margin': lgap, 'right_winner_margin': rgap,
                    'left_feasible_candidates': ln, 'right_feasible_candidates': rn,
                    'empty_left_pool': ln == 0, 'empty_right_pool': rn == 0,
                }
            row = {'kernel': kernel, 'family': family_by_kernel.get(kernel),
                   'split': left[next(iter(common))]['split'], 'dimension': dimension,
                   'left': list(left_target), 'right': list(right_target),
                   'left_candidates': len(left), 'right_candidates': len(right),
                   'matched_assignments': len(common), 'status': 'paired',
                   'physical_delta_summary': {
                       name: {'mean_absolute': float(np.mean(np.abs(values))),
                              'median_absolute': float(np.median(np.abs(values))),
                              'mean_signed': float(np.mean(values)),
                              'meaningful_counts': {
                                  f'>{int(t*100)}%': int(meaningful[f'{name}_gt_{int(t*100)}pct'])
                                  for t in (0.01, 0.05, 0.10)}}
                       for name, values in deltas.items()},
                   'meaningful_counts_5pct': {k: v for k, v in meaningful.items() if k.endswith('_5pct')},
                   'adp_winners_by_budget': winner_comparisons,
                   'candidate_pool_hash': hashlib.sha256(
                       json.dumps(common, separators=(',', ':')).encode()).hexdigest()}
            target_pairs.append(row)
            family_rows[family_by_kernel.get(kernel, kernel)][dimension].append(row)
        if kernel_number % 5 == 0:
            print(f'V4b coverage: completed {kernel_number}/40 kernels', flush=True)

    family_summary = {}
    for family, dimensions in sorted(family_rows.items()):
        family_summary[family] = {}
        for dimension, rows in dimensions.items():
            paired = [r for r in rows if r.get('status') == 'paired']
            family_summary[family][dimension] = {
                'kernel_count': len({r['kernel'] for r in paired}),
                'target_pairs': len(rows),
                'paired_target_pairs': len(paired),
                'matched_assignment_total': sum(r['matched_assignments'] for r in paired),
                'meaningful_adp_winner_reversals_by_budget': {
                    budget: sum(bool(r.get('adp_winners_by_budget', {}).get(budget, {}).get(
                        'meaningful_reversal_5pct')) for r in paired)
                    for budget in map(lambda x: f'{x:g}', BUDGETS)},
            }
    return {'target_pairs': target_pairs, 'budget_pairs': budget_pairs,
            'family_summary': family_summary,
            'eligible_context_count': len(grouped),
            'exact_assignment_intersections': sum(r.get('matched_assignments', 0) for r in target_pairs)}


def _assignment_distance(a, b):
    keys = set(a) | set(b)
    return sum(a.get(k) != b.get(k) for k in keys)


def _select_qoblind(rows, count, support=None, seed_hashes=()):
    """Vectorized Hamming-diverse assignment sampling; never reads QoR labels."""
    unique = {}
    for row in rows:
        unique.setdefault(row['config_hash'], row)
    ordered = sorted(unique.values(), key=lambda r: (
        -int((support or {}).get(r['config_hash'], 0)), r['config_hash']))
    if len(ordered) <= count:
        return sorted(ordered, key=lambda r: r['config_hash'])
    keys = sorted({key for row in ordered for key in row['assignment']})
    matrix = np.asarray([[str(row['assignment'].get(key, '<missing>')) for key in keys]
                         for row in ordered], dtype=object)
    chosen = [i for i, row in enumerate(ordered) if row['config_hash'] in set(seed_hashes)]
    chosen = chosen[:count]
    if not chosen:
        chosen = [0]
    min_distance = np.full(len(ordered), np.iinfo(np.int32).max, dtype=np.int32)
    for index in chosen:
        min_distance = np.minimum(min_distance, np.count_nonzero(matrix != matrix[index], axis=1))
    min_distance[chosen] = -1
    while len(chosen) < count:
        maximum = int(min_distance.max())
        if maximum < 0:
            break
        # `ordered` already sorts support/hash, so argmax gives deterministic
        # support-prioritized then assignment-hash tie handling.
        index = int(np.flatnonzero(min_distance == maximum)[0])
        chosen.append(index)
        distance = np.count_nonzero(matrix != matrix[index], axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[chosen] = -1
    return sorted((ordered[i] for i in chosen), key=lambda r: r['config_hash'])


def build_report(release_dir, authenticated_audit, output_dir, candidates_per_context=16):
    """Load only audited non-test kernels and write immutable V4b audit artifacts."""
    start = time.monotonic()
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty V4b output directory {out}')
    audit = json.loads(Path(authenticated_audit).read_text())
    if audit.get('gate') not in ('pass', 'pass_with_explicit_exclusions'):
        raise ValueError('Authenticated V4 audit gate did not pass')
    split_path = ROOT / 'mailohls_runs/mailohls_final_family_split_s123.json'
    split = json.loads(split_path.read_text())
    test_kernels = set(split['test_kernels'])
    if test_kernels != {'serrano-kalman-filter'}:
        raise ValueError('Unexpected test kernel inventory; refusing V4b audit')
    audit_rows = audit['kernels']
    eligible = [r for r in audit_rows if not r['unmapped_csv_actions'] and r['family'] in ('train', 'val')]
    selected = sorted(Path(r['graph_member']).name.rsplit('-', 1)[0] for r in eligible)
    # Graph filenames end in a 12-character content suffix; recover names by
    # exact split inventory matching rather than assuming punctuation rules.
    all_non_test = set(split['train_kernels']) | set(split['val_kernels'])
    def match_kernel(member):
        found = [k for k in all_non_test if Path(member).name.startswith(k + '-')]
        if len(found) != 1:
            raise ValueError(f'Cannot uniquely map authenticated graph member {member}')
        return found[0]
    eligible_names = sorted(match_kernel(r['graph_member']) for r in eligible)
    family_map = _family_map(split)
    if any(name in test_kernels for name in eligible_names):
        raise ValueError('Kalman leaked into eligible cohort')
    train_names = [k for k in eligible_names if k in split['train_kernels']]
    val_names = [k for k in eligible_names if k in split['val_kernels']]
    if len(train_names) != 37 or len(val_names) != 3:
        raise ValueError(f'Frozen strict cohort changed: train={len(train_names)}, val={len(val_names)}')

    # Recheck excluded APL actions: source/action metadata does not encode the
    # measured pragma value, and no alternative authenticated table is joined
    # by the existing source/CSV contract. They remain unknown, not disabled.
    exclusions = []
    for row in audit_rows:
        if row['family'] == 'test':
            continue
        kernel = match_kernel(row['graph_member'])
        if row['unmapped_csv_actions']:
            exclusions.append({
                'kernel': kernel, 'split': row['family'],
                'missing_measured_action_ids': row['unmapped_csv_actions'],
                'affected_measured_rows': row['measured_rows'],
                'classification': 'C_unknown_directive_state',
                'recovery_evidence': 'No measured APL/CSV field in the authenticated selected preprocessing contract; source action registration and placeholder defaults do not establish an applied directive value.',
                'alternative_authenticated_mapping_found': False,
                'usable_for_complete_assignment_training': False,
            })

    data = train.PreparedData(release_dir, Path(output_dir) / 'unused_audit.json',
                              selected_kernels=eligible_names,
                              authenticated_audit_path=authenticated_audit)
    print(f'V4b audit: loaded {len(data.records)} eligible records across '
          f'{len(data.graphs)} authenticated kernels; computing preference coverage',
          flush=True)
    if any(r['kernel'] in test_kernels for r in data.records):
        raise ValueError('Prepared V4b data unexpectedly includes Kalman')
    family_by_kernel = family_map
    coverage = _coverage(data.records, family_by_kernel)
    eligible_row_counts = Counter(row['kernel'] for row in data.records)
    context_groups = defaultdict(list)
    for row in data.records:
        context_groups[(row['kernel'], row['profile'], row['clock'])].append(row)
    pool_manifest = {}
    kernels = sorted({key[0] for key in context_groups})
    for kernel in kernels:
        kernel_contexts = {key: rows for key, rows in context_groups.items() if key[0] == kernel}
        support = Counter()
        exemplars = {}
        for rows in kernel_contexts.values():
            for row in rows:
                support[row['config_hash']] += 1
                exemplars.setdefault(row['config_hash'], row)
        core = _select_qoblind(list(exemplars.values()), candidates_per_context, support)
        core_hashes = [row['config_hash'] for row in core]
        for (context_kernel, profile, clock), rows in sorted(kernel_contexts.items()):
            chosen = _select_qoblind(rows, candidates_per_context, support,
                                     seed_hashes=core_hashes)
            pool_manifest[f'{kernel}|{profile}|{clock:g}ns'] = {
                'kernel': kernel, 'family': family_map.get(kernel),
                'split': chosen[0]['split'] if chosen else None,
                'device': profile, 'clock_ns': clock,
                'available_complete_assignments': len({r['config_hash'] for r in rows}),
                'selected_candidate_count': len(chosen),
                'selected_assignment_hashes': [r['config_hash'] for r in chosen],
                'matched_core_assignment_count': sum(h in {r['config_hash'] for r in rows}
                                                     for h in core_hashes),
                'selection': 'support-prioritized core plus vectorized greedy directive Hamming diversity; QoR labels not read',
            }
    print(f'V4b coverage complete; selected-pool manifest built for {len(pool_manifest)} contexts',
          flush=True)
    pair_contexts = []
    pair_totals = defaultdict(Counter)
    for context_id, selected_context in sorted(pool_manifest.items()):
        key = (selected_context['kernel'], selected_context['device'],
               float(selected_context['clock_ns']))
        by_hash = {row['config_hash']: row for row in context_groups[key]}
        rows = [by_hash[h] for h in selected_context['selected_assignment_hashes']]
        for fraction in BUDGETS:
            pairs, counts = preference_pairs(rows, fraction, PRIMARY_GAP)
            pair_ids = [(rows[a]['config_hash'], rows[b]['config_hash'], reason)
                        for a, b, reason in pairs]
            pair_contexts.append({
                'context': context_id, 'kernel': selected_context['kernel'],
                'family': selected_context['family'], 'split': selected_context['split'],
                'device': selected_context['device'], 'clock_ns': selected_context['clock_ns'],
                'budget_fraction': fraction, 'objective': 'PARETO_ADP',
                'selected_candidate_count': len(rows),
                'candidate_pool_sha256': hashlib.sha256(json.dumps(
                    [row['config_hash'] for row in rows], separators=(',', ':')).encode()).hexdigest(),
                'pair_count': len(pair_ids),
                'pair_identity_sha256': hashlib.sha256(json.dumps(
                    pair_ids, separators=(',', ':')).encode()).hexdigest(),
                'pair_counts': counts,
                'measurement_keys': [{
                    'config_hash': row['config_hash'], 'preprocessed_row': row['preprocessed_row'],
                    'kernel': row['kernel'], 'device': row['device'], 'clock_ns': row['clock'],
                } for row in rows],
            })
            for category, count in counts.items():
                pair_totals[selected_context['split']][category] += count
            pair_totals[selected_context['split']]['preference_pairs'] += len(pair_ids)
    pair_audit = {
        'schema': SCHEMA + '-preference-pair-audit',
        'objective': 'PARETO_ADP', 'meaningful_gap': PRIMARY_GAP,
        'budget_fractions': list(BUDGETS),
        'feasibility_rule': 'measured physical resources <= fraction * authenticated device capacity',
        'both_infeasible': 'excluded', 'near_tie': 'excluded',
        'zero_area_ambiguity': 'Viterbi/U200/5ns ADP pairs involving all-zero utilization excluded; source labels unchanged',
        'pair_context_count': len(pair_contexts),
        'totals_by_split': {split_name: dict(values) for split_name, values in pair_totals.items()},
        'contexts': pair_contexts,
        'test_kernels_loaded': [],
    }
    eligibility = {
        'schema': SCHEMA + '-eligibility-audit',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'audit_script_sha256': _sha(__file__),
        'preference_builder_sha256': _sha(Path(__file__).with_name('preferences.py')),
        'model_implementation_sha256': _sha(Path(__file__).with_name('models.py')),
        'training_runner_sha256': _sha(Path(__file__).with_name('train_probe.py')),
        'action_demand_implementation_sha256': _sha(
            ROOT / 'GNN_branch/action_demand_v22.py'),
        'dataset_sha256': audit['dataset']['dataset_sha256'],
        'family_split_sha256': audit['dataset']['split_sha256'],
        'graph_archive_sha256': audit['graph_release']['archive_sha256'],
        'authenticated_audit_sha256': _sha(authenticated_audit),
        'full_inventory': {'kernels': audit['dataset']['kernels'],
                           'train_kernels': len(split['train_kernels']),
                           'validation_kernels': len(split['val_kernels']),
                           'test_kernels': len(test_kernels),
                           'aggregated_design_points': audit['dataset']['unique_measured_design_points'],
                           'strict_eligible_train_kernels': len(train_names),
                           'strict_eligible_validation_kernels': len(val_names),
                           'eligible_measured_rows_after_tuple_checks': len(data.records),
                           'missing_measurement_action_exclusions': len(exclusions)},
        'training_kernels': train_names, 'validation_kernels': val_names,
        'family_by_kernel': {kernel: family_map.get(kernel) for kernel in eligible_names},
        'test_kernels_loaded': [],
        'kernel_audit': [{
            'kernel': match_kernel(r['graph_member']),
            'family': family_map.get(match_kernel(r['graph_member'])),
            'split': r['family'],
            'graph_identity': r['graph_identity'], 'graph_member_sha256': r['graph_member_sha256'],
            'source_sha256': r['source_sha256'], 'apl_sha256': r['apl_sha256'],
            'preprocessed_csv_sha256': r['preprocessed_csv_sha256'],
            'registered_actions': r['registered_actions'], 'matched_actions': r['matched_actions'],
            'unresolved_actions': r['unresolved_actions'],
            'unmeasured_action_ids': r['unmapped_csv_actions'],
            'rows': r['measured_rows'],
            'source_valid_rows_before_policy_exclusions': r['candidate_rows_validated'],
            'eligible_rows_after_policy': eligible_row_counts.get(
                match_kernel(r['graph_member']), 0),
            'array_actions': r['array_actions'], 'loop_actions': r['loop_actions'],
            'eligibility': ('excluded_unknown_action_values' if r['unmapped_csv_actions'] else
                            'eligible_complete_assignment_contract'),
        } for r in audit_rows if r['family'] != 'test'],
        'excluded_action_assessment': exclusions,
        'cohort_policy': 'All eligible training kernels and all eligible validation kernels; complete source-legal assignments only; no Kalman; no imputation.',
        'candidate_pool_size': candidates_per_context,
        'candidate_pool_selection': 'QoR-independent complete-assignment diversity; per-context support varies; target pair analysis separately uses exact full intersections.',
        'elapsed_cpu_seconds': time.monotonic() - start,
    }
    # Remove implementation helper's unused audit path; the original data audit
    # is authenticated and supplied separately.
    out.mkdir(parents=True, exist_ok=True)
    cohort = {'schema': SCHEMA + '-cohort-manifest',
              'eligibility_manifest_sha256': hashlib.sha256(json.dumps(
                  eligibility, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'dataset_sha256': audit['dataset']['dataset_sha256'],
              'family_split_sha256': audit['dataset']['split_sha256'],
              'graph_archive_sha256': audit['graph_release']['archive_sha256'],
              'action_demand_implementation_sha256':
                  eligibility['action_demand_implementation_sha256'],
              'training_kernels': train_names, 'validation_kernels': val_names,
              'candidate_pool_size': candidates_per_context,
              'family_disjoint': True, 'test_kernels_loaded': [], 'contexts': pool_manifest}
    cohort_hash = hashlib.sha256(json.dumps(
        cohort, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cohort['manifest_sha256'] = cohort_hash
    pair_audit['cohort_manifest_sha256'] = cohort_hash
    coverage_report = {'schema': SCHEMA + '-target-coverage', **coverage,
                       'primary_meaningful_gap': PRIMARY_GAP,
                       'sensitivity_gaps': [0.01, 0.05, 0.10],
                       'budget_fractions': list(BUDGETS),
                       'objective': 'PARETO_ADP',
                       'interpretation': 'All-source-candidate coverage audit; target pair observations overlap and are not independent samples.'}
    for name, payload in (('eligibility.json', eligibility),
                          ('preference_cohort_manifest.json', cohort),
                          ('preference_pair_audit.json', pair_audit),
                          ('preference_target_coverage.json', coverage_report)):
        path = out / name
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return {'eligibility': eligibility, 'cohort': cohort, 'coverage': coverage_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--authenticated-audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidates-per-context', type=int, default=16)
    args = parser.parse_args()
    if args.candidates_per_context < 1:
        raise SystemExit('--candidates-per-context must be positive')
    result = build_report(args.release_dir, args.authenticated_audit, args.output_dir,
                          args.candidates_per_context)
    print(json.dumps({
        'inventory': result['eligibility']['full_inventory'],
        'target_pairs': len(result['coverage']['target_pairs']),
        'coverage_output': str(args.output_dir),
    }, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
