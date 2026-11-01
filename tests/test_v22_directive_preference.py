from types import SimpleNamespace

import pytest
import torch

from Preprocessing import create_jsonl
from experiments.v22_gnn_training import directive_preference_probe as probe


def _record(*, latency=1.0, resources=(1.0, 1.0, 1.0, 1.0), kernel='toy'):
    return {'kernel': kernel, 'profile': 'zcu104', 'clock': 10.0,
            'physical': [latency, *resources], 'capacities': [10.0] * 4,
            'config_hash': f'{latency}-{resources}'}


def test_feasible_configuration_beats_infeasible_even_with_worse_latency():
    rows = [_record(latency=0.1, resources=(9, 1, 1, 1)),
            _record(latency=1.0, resources=(4, 1, 1, 1))]
    pairs, counts = probe._preference_pairs(rows, 0.5, 'PARETO_LATENCY')
    assert pairs == [(1, 0, 'feasibility')]
    assert counts['feasible_vs_infeasible'] == 1


def test_objective_pairs_exclude_near_ties_and_both_infeasible():
    rows = [_record(latency=1.0), _record(latency=1.2), _record(latency=1.001),
            _record(latency=0.2, resources=(9, 9, 9, 9)),
            _record(latency=0.3, resources=(8, 8, 8, 8))]
    pairs, counts = probe._preference_pairs(rows, 0.5, 'PARETO_LATENCY')
    assert pairs
    assert all(rows[a]['physical'][0] < rows[b]['physical'][0]
               for a, b, kind in pairs if kind == 'objective')
    assert counts['near_tie'] > 0
    assert counts.get('both_infeasible', 0) > 0


def test_ambiguous_zero_area_viterbi_rows_are_excluded_for_area_and_adp():
    ambiguous = _record(kernel='machsuite-viterbi', resources=(0, 0, 0, 0))
    ambiguous.update(profile='u200', clock=5.0)
    other = _record(kernel='machsuite-viterbi', latency=2.0)
    other.update(profile='u200', clock=5.0)
    pairs, counts = probe._preference_pairs([ambiguous, other], 1.0, 'PARETO_ADP')
    assert pairs == []
    assert counts['zero_area_ambiguous'] == 1


def test_target_response_ranks_only_the_authenticated_common_pool():
    kernel = 'machsuite-viterbi'
    left_key = f'{kernel}|zcu104|10ns|1|PARETO_ADP'
    right_key = f'{kernel}|u200|10ns|1|PARETO_ADP'
    pool_id = probe._sha(['b', 'c'])
    coverage = {'assignment_pools': {pool_id: ['b', 'c']},
        'axes': {'device_preference': {'rows': [{
        'kernel': kernel, 'family': 'family', 'pair': '10ns',
        'classification': 'meaningful_change',
        'left_request': {'profile': 'zcu104', 'clock_ns': 10.0,
                         'budget_fraction': 1.0, 'objective': 'PARETO_ADP'},
        'right_request': {'profile': 'u200', 'clock_ns': 10.0,
                          'budget_fraction': 1.0, 'objective': 'PARETO_ADP'},
        'matched_assignment_pool_sha256': pool_id,
        'left': {'winner': 'b'}, 'right': {'winner': 'c'},
    }]}}}
    result = {'contexts': [
        {'context': left_key, 'candidate_scores': [
            {'config_hash': 'a', 'score': -10.0},
            {'config_hash': 'b', 'score': 2.0}, {'config_hash': 'c', 'score': 3.0}]},
        {'context': right_key, 'candidate_scores': [
            {'config_hash': 'd', 'score': -20.0},
            {'config_hash': 'b', 'score': 4.0}, {'config_hash': 'c', 'score': 1.0}]},
    ]}
    report = probe.target_response_report(coverage, result)['device_preference']
    assert report['totals']['correct_changed_complete_winners'] == 1
    assert report['comparisons'][0]['left_selected'] == 'b'
    assert report['comparisons'][0]['right_selected'] == 'c'
    assert report['comparisons'][0]['matched_candidate_count'] == 2


def test_frozen_roster_is_family_disjoint_and_excludes_test_kernel():
    assert set(probe.TRAIN_KERNELS).isdisjoint(probe.VALIDATION_KERNELS)
    assert not any('kalman' in name.lower() for name in
                   probe.TRAIN_KERNELS + probe.VALIDATION_KERNELS)
    assert set(probe.OBJECTIVES) == {'PARETO_ADP', 'PARETO_LATENCY', 'PARETO_AREA'}


def test_selected_dataset_reconstruction_does_not_read_unselected_measurements(tmp_path, monkeypatch):
    import pandas as pd

    columns = ['Device', 'Clock_Period_nsec', 'Latency_msec', 'Area', 'Weight',
               'is_pareto', 'BRAM_Utilization_percentage',
               'DSP_Utilization_percentage', 'FF_Utilization_percentage',
               'LUT_Utilization_percentage']
    selected = tmp_path / 'preprocessed-train-kernel.csv'
    selected.write_text(','.join(columns) + '\nxczu7ev-ffvc1156-2-e,10,1,0.5,1,true,1,1,1,1\n')
    heldout = tmp_path / 'preprocessed-test-kernel.csv'
    heldout.write_text('this intentionally is not a valid CSV or measurement')
    source = tmp_path / 'kernel.cpp'
    source.write_text('void kernel() {}')
    read_paths = []
    original_read_csv = pd.read_csv

    def tracked_read_csv(path, *args, **kwargs):
        read_paths.append(str(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(create_jsonl, 'application_sources',
                        lambda kernels: {kernel: source for kernel in kernels})
    monkeypatch.setattr(create_jsonl, 'load_actions', lambda kernel, text: [])
    monkeypatch.setattr(create_jsonl, 'source_with_placeholders', lambda *args: 'template')
    monkeypatch.setattr(pd, 'read_csv', tracked_read_csv)
    examples, stats, _ = create_jsonl.build_examples(
        tmp_path, selected_kernels={'train-kernel'})
    assert len(examples) == 1 and len(stats) == 1
    assert examples[0]['kernel_name'] == 'train-kernel'
    assert read_paths == [str(selected)]


class _FixedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.full((1, 64, 128), 0.2))
        self.relation = torch.nn.Embedding(len(probe.tensors.RELATIONS), 2)

    def memories(self, graph):
        return self.bias + self.relation.weight[0, 0] * 0.0, None


def _graph():
    return SimpleNamespace(
        registered_mask=torch.ones((1, 64), dtype=torch.bool),
        matched_mask=torch.ones((1, 64), dtype=torch.bool),
        action_demand=torch.ones((1, 64, probe.tensors.ACTION_DEMAND_DIM)),
        action_kind_slot=torch.zeros((1, 64), dtype=torch.long),
    )


def test_target_film_is_zero_gated_candidate_independent_and_differentiable():
    model = probe.PreferenceMemoryModel('target_film', profile_feature_dim=8)
    model.encoder = _FixedEncoder()
    graph = _graph()
    profile = torch.ones(8)
    base, _ = model.encoder.memories(graph)
    first = model.encode_memory(graph, profile)
    second = model.encode_memory(graph, profile)
    assert torch.equal(first, base)
    assert torch.equal(first, second)
    first.sum().backward()
    assert model.gate.grad is not None
    assert torch.isfinite(model.gate.grad)
    assert torch.isfinite(model.modulation[0].weight.grad).all()


def test_preference_scorer_has_finite_candidate_and_memory_gradients():
    model = probe.PreferenceMemoryModel('compiler')
    model.encoder = _FixedEncoder()
    graph = _graph()
    memory, _ = model.encoder.memories(graph)
    candidates = torch.randn(2, 64, probe.train.CANDIDATE_DIM)
    demands = torch.randn(2, 64, probe.train.CANDIDATE_DEMAND_DIM)
    decision = torch.zeros(12)
    scores = model.score(graph, memory, candidates, demands, decision)
    loss = torch.nn.functional.softplus(scores[0] - scores[1])
    loss.backward()
    assert torch.isfinite(scores).all() and torch.isfinite(loss)
    assert model.encoder.bias.grad is not None
    assert model.scorer[-1].weight.grad is not None
    assert torch.isfinite(model.encoder.bias.grad).all()
    assert torch.isfinite(model.scorer[-1].weight.grad).all()
    breakdown = probe._gradient_breakdown(model)
    assert set(breakdown) == {'encoder_l2','hardware_relation_embedding_l2',
                              'target_conditioner_l2','preference_head_l2'}
    assert all(torch.isfinite(torch.tensor(value)) for value in breakdown.values())

