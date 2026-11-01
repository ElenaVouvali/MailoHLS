"""Reproducibly summarize authenticated assignment overlap across targets.

Only the predeclared five training and two validation kernels are included;
the Kalman test split is excluded from loaded measurement records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v22_gnn_training import checkpointing, train
from GNN_branch import target_graph_v22 as overlay


TRAIN = ('machsuite-gemm-ncubed', 'machsuite-spmv-ellpack',
         'machsuite-stencil2d', 'machsuite-md-knn',
         'rodinia_dilate_0_baseline_0')
VALIDATION = ('machsuite-viterbi', 'rodinia_pathfinder_0_baseline_0')
PROFILES = ('zcu104', 'u200')
CLOCKS = (3.33, 5.0, 10.0)


def _variation(pairs):
    if not pairs:
        return {'matched_assignment_count': 0}
    deltas = np.asarray([np.asarray(right['physical']) - np.asarray(left['physical'])
                         for left, right in pairs], dtype=np.float64)
    return {
        'matched_assignment_count': len(pairs),
        'mean_absolute_delta': np.mean(np.abs(deltas), axis=0).tolist(),
        'nonzero_delta_count_by_target': np.sum(np.abs(deltas) > 1e-12, axis=0).tolist(),
        'latency_delta_ms_range': [float(np.min(deltas[:, 0])), float(np.max(deltas[:, 0]))],
    }


def audit(release_dir: Path, output: Path):
    roster = TRAIN + VALIDATION
    data = train.PreparedData(release_dir, output.with_suffix('.full-audit.json'),
                              selected_kernels=roster)
    result = {
        'schema': 'salt-hls-v4-authenticated-target-coverage-audit-v1',
        'audit_script_sha256': overlay.file_hash(Path(__file__)),
        'graph_release_archive_sha256': data.audit['graph_release']['archive_sha256'],
        'dataset_sha256': data.audit['dataset']['dataset_sha256'],
        'family_split_sha256': data.audit['dataset']['split_sha256'],
        'training_kernels': list(TRAIN), 'validation_kernels': list(VALIDATION),
        'test_kernels_loaded_for_analysis': [],
        'context_axis': {'profiles': list(PROFILES), 'clock_ns': list(CLOCKS)},
        'budget_semantics': {
            'physical_measurement_keys': ['kernel', 'complete_assignment', 'profile', 'clock_ns'],
            'budget_is_not_part_of_physical_measurement_identity': True,
            'feasibility_only': True,
            'evidence': 'PreparedData.physical_targets stores measured QoR independent of budget; '
                        'train.evaluate recomputes feasibility against capacities*fraction without changing labels.'},
        'kernels': {}, 'exclusions': data.exclusions,
    }
    for kernel in roster:
        contexts = {}
        sets = {}
        for profile in PROFILES:
            for clock in CLOCKS:
                key = (kernel, profile, clock)
                rows = data.groups.get(key, [])
                lookup = {row['config_hash']: row for row in rows}
                contexts[f'{profile}|{clock:g}ns'] = {
                    'candidate_count': len(rows),
                    'distinct_assignments': len(lookup),
                    'distinct_latency_values': len({r['physical'][0] for r in rows}),
                    'full_capacity_feasible': sum(all(v <= c for v, c in zip(r['physical'][1:], r['capacities'])) for r in rows),
                    'half_capacity_feasible': sum(all(v <= .5*c for v, c in zip(r['physical'][1:], r['capacities'])) for r in rows),
                    'tenth_capacity_feasible': sum(all(v <= .1*c for v, c in zip(r['physical'][1:], r['capacities'])) for r in rows),
                }
                sets[(profile, clock)] = lookup
        clock_pairs = {}
        device_pairs = {}
        for profile in PROFILES:
            for i, left_clock in enumerate(CLOCKS):
                for right_clock in CLOCKS[i+1:]:
                    left, right = sets[(profile, left_clock)], sets[(profile, right_clock)]
                    shared = sorted(set(left) & set(right))
                    clock_pairs[f'{profile}|{left_clock:g}->{right_clock:g}ns'] = _variation(
                        [(left[h], right[h]) for h in shared])
        for clock in CLOCKS:
            left, right = sets[('zcu104', clock)], sets[('u200', clock)]
            shared = sorted(set(left) & set(right))
            device_pairs[f'{clock:g}ns|zcu104->u200'] = _variation(
                [(left[h], right[h]) for h in shared])
        six_way = set.intersection(*(set(sets[(profile, clock)])
                                     for profile in PROFILES for clock in CLOCKS))
        six_latency_spread = [
            max(sets[(profile, clock)][h]['physical'][0] for profile in PROFILES for clock in CLOCKS)
            - min(sets[(profile, clock)][h]['physical'][0] for profile in PROFILES for clock in CLOCKS)
            for h in six_way
        ]
        graph_item = next((item for item in data.audit['kernels']
                           if Path(item['graph_member']).name.startswith(kernel + '-')), None)
        result['kernels'][kernel] = {
            'split': 'train' if kernel in TRAIN else 'validation',
            'contexts': contexts, 'clock_pairs': clock_pairs,
            'device_pairs': device_pairs,
            'assignments_shared_across_all_six_contexts': len(six_way),
            'six_context_latency_spread_ms': {
                'mean': float(np.mean(six_latency_spread)) if six_latency_spread else None,
                'median': float(np.median(six_latency_spread)) if six_latency_spread else None,
                'max': float(max(six_latency_spread)) if six_latency_spread else None,
            },
            'graph_identity': graph_item.get('graph_identity') if graph_item else None,
            'graph_member_sha256': graph_item.get('graph_member_sha256') if graph_item else None,
            'source_sha256': graph_item.get('source_sha256') if graph_item else None,
            'metadata_sha256': graph_item.get('metadata_sha256') if graph_item else None,
            'registered_actions': graph_item.get('registered_actions') if graph_item else None,
            'matched_actions': graph_item.get('matched_actions') if graph_item else None,
            'array_actions': graph_item.get('array_actions') if graph_item else None,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpointing.atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.release_dir, args.output)
    print(json.dumps({'output': str(args.output), 'kernels': len(report['kernels']),
                      'dataset_sha256': report['dataset_sha256'],
                      'graph_release_archive_sha256': report['graph_release_archive_sha256']}, indent=2))


if __name__ == '__main__':
    main()
