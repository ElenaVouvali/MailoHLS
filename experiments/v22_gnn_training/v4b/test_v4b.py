"""Focused CPU regressions for V4b physical and pairwise contracts."""
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import target_graph_tensor_v22 as tensors
from experiments.target_graph_readiness.experiment import CANDIDATE_DEMAND_DIM
from experiments.v22_gnn_training.v4b.audit import _coverage, _select_qoblind
from experiments.v22_gnn_training.v4b.models import (
    ActionDifferenceComparator,
    NoGraphControl,
    TargetFilmV4bModel,
    V4bGraphModel,
    feasibility_from_physical,
    hybrid_loss,
    pairwise_pool_order,
)
from experiments.v22_gnn_training.v4b.preferences import preference_pairs
from experiments.v22_gnn_training.v4b.train_probe import (
    _cached_candidate_inputs, _initial_models, _pair_schedule,
)
from experiments.v22_gnn_training.v4b.train_probe import require_isolated_gpu


def _fake_graph():
    registered = torch.zeros((1, 64), dtype=torch.bool)
    matched = torch.zeros_like(registered)
    registered[0, [0, 3, 9]] = True
    matched[0, [0, 3]] = True  # slot 9 represents a registered unresolved action
    return SimpleNamespace(
        registered_mask=registered,
        matched_mask=matched,
        action_kind_slot=torch.zeros((1, 64), dtype=torch.long),
        hardware_context=torch.zeros((1, 13)),
    )


def _tiny_tensor_graph():
    n = 2
    registered = torch.zeros((1, 64), dtype=torch.bool)
    matched = torch.zeros_like(registered)
    unresolved = torch.zeros_like(registered)
    return Data(
        x=torch.randn(n, tensors.NODE_DIM),
        edge_index=torch.tensor([[1], [0]], dtype=torch.long),
        edge_type=torch.zeros(1, dtype=torch.long),
        edge_attr=torch.randn(1, tensors.EDGE_DIM),
        node_type=torch.tensor([0, len(compiler.NODE_KINDS) + 1]),
        op_type=torch.zeros(n, dtype=torch.long), value_type=torch.zeros(n, dtype=torch.long),
        resource_type=torch.tensor([0, 1]), root_type=torch.zeros(n, dtype=torch.long),
        action_node_index=torch.empty(0, dtype=torch.long),
        action_slot=torch.empty(0, dtype=torch.long),
        action_kind_slot=torch.zeros((1, 64), dtype=torch.long),
        action_demand=torch.zeros(1, 64, tensors.ACTION_DEMAND_DIM),
        action_semantic=torch.zeros(1, 64, tensors.SEMANTIC_EDGE_DIM),
        action_semantic_mask=matched, action_demand_mask=matched,
        registered_mask=registered, matched_mask=matched, unresolved_mask=unresolved,
        hardware_context=torch.tensor([[0.] * 8 + [1.] * 4 + [0.]]),
    )


def test_physical_prediction_is_budget_independent_and_preferences_can_change():
    torch.manual_seed(123)
    graph = _fake_graph()
    graph.hardware_context[:, 8:12] = 1.0
    model = V4bGraphModel(hardware_messages=False)
    memory = torch.randn(1, 64, 128)
    pooled = torch.randn(1, 128)
    candidate = torch.randn(2, 64, 8)
    demand = torch.randn(2, 64, CANDIDATE_DEMAND_DIM)
    full = model.from_memory(graph, memory, pooled, candidate, demand, 1.0)
    tight = model.from_memory(graph, memory, pooled, candidate, demand, 0.25)
    assert torch.equal(full['qor_z'], tight['qor_z'])
    assert not torch.equal(full['preference_score'], tight['preference_score'])
    (full['qor_z'].sum() + full['preference_score'].sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_graph_arm_encoder_uses_only_selected_message_edges_and_full_capacity():
    torch.manual_seed(13)
    graph = _tiny_tensor_graph()
    numerical = V4bGraphModel(hardware_messages=False)
    memory, pooled = numerical.encode(graph)
    expected_memory, expected_pooled = numerical.graph.memories(
        tensors.without_hardware_messages(graph))
    assert torch.equal(memory, expected_memory)
    assert torch.equal(pooled, expected_pooled)
    assert memory.shape == (1, 64, 128)
    assert pooled.shape == (1, 128)
    graph.hardware_context[:, 8] = 0.5
    try:
        numerical.encode(graph)
    except ValueError as exc:
        assert 'full device capacity' in str(exc)
    else:
        raise AssertionError('Requested budget leaked into physical memory encoding')


def test_action_difference_is_antisymmetric_zero_for_identity_and_aligned():
    torch.manual_seed(456)
    graph = _fake_graph()
    memory = torch.randn(1, 64, 128)
    a = torch.zeros(1, 64, 8)
    b = a.clone()
    b[0, 3, 0] = 1.0
    demand_a = torch.zeros(1, 64, CANDIDATE_DEMAND_DIM)
    demand_b = demand_a.clone()
    model = ActionDifferenceComparator()
    score_ab, weights_ab = model(graph, memory, a, demand_a, b, demand_b, 0.5)
    score_ba, weights_ba = model(graph, memory, b, demand_b, a, demand_a, 0.5)
    score_aa, weights_aa = model(graph, memory, a, demand_a, a, demand_a, 0.5)
    assert torch.allclose(score_ab, -score_ba, atol=1e-7, rtol=0)
    assert torch.equal(score_aa, torch.zeros_like(score_aa))
    assert torch.allclose(weights_ab, weights_ba, atol=0, rtol=0)
    assert weights_ab[0, 3] == 1
    assert torch.equal(weights_ab[0, [0, 9]], torch.zeros(2))
    assert torch.equal(weights_aa, torch.zeros_like(weights_aa))


def test_unresolved_action_only_attends_when_its_candidate_changes():
    torch.manual_seed(7)
    graph = _fake_graph()
    memory = torch.randn(1, 64, 128)
    a = torch.zeros(1, 64, 8)
    b = a.clone()
    b[0, 9, 2] = 1.0
    demand = torch.zeros(1, 64, CANDIDATE_DEMAND_DIM)
    comparator = ActionDifferenceComparator()
    _, weights = comparator(graph, memory, a, demand, b, demand)
    assert weights[0, 9] == 1


def test_no_graph_control_and_hybrid_objective_are_finite():
    torch.manual_seed(9)
    graph = _fake_graph()
    graph.hardware_context[:, 8:12] = 1.0
    model = NoGraphControl()
    candidate = torch.randn(3, 64, 8)
    demand = torch.randn(3, 64, CANDIDATE_DEMAND_DIM)
    output = model(graph, candidate, demand, 0.5)
    target = torch.randn_like(output['qor_z'])
    loss, components = hybrid_loss(output['qor_z'], target,
                                   output['preference_score'][:2].diff(), [1])
    assert torch.isfinite(loss)
    assert torch.isfinite(components['physical_qor'])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_feasibility_and_pairwise_pool_ties():
    resources = torch.tensor([[5., 4., 3., 2.], [8., 4., 3., 2.]])
    capacity = torch.tensor([10., 10., 10., 10.])
    assert feasibility_from_physical(resources, capacity, 0.5).tolist() == [True, False]
    scores = torch.tensor([[0., 2., -1.], [-2., 0., 1.], [1., -1., 0.]])
    assert pairwise_pool_order(scores, ['b', 'a', 'c']) == [1, 0, 2]


def test_explicit_v100s_gpu_selection_requires_matching_idle_uuid(monkeypatch):
    from experiments.v22_gnn_training import launch

    expected = 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'
    monkeypatch.setattr(launch, 'gpu_inventory', lambda: [{
        'index': 1, 'uuid': expected, 'name': 'Tesla V100S-PCIE-32GB',
        'total_mib': 32768, 'used_mib': 1, 'free_mib': 32495,
        'utilization_percent': 0,
    }])
    monkeypatch.setattr(launch, 'compute_processes', lambda: [])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', expected)
    monkeypatch.setenv('SALT_HLS_GPU_UUID', expected)
    result = require_isolated_gpu('V100S', expected, 6000)
    assert result['uuid'] == expected

    monkeypatch.setattr(launch, 'compute_processes', lambda: [{
        'uuid': expected, 'pid': 17, 'name': 'other-job', 'used_mib': '1000',
    }])
    try:
        require_isolated_gpu('V100S', expected, 6000)
    except RuntimeError as exc:
        assert 'occupied V100S' in str(exc)
    else:
        raise AssertionError('Occupied V100S must be rejected')


def test_candidate_feature_cache_reuses_overlay_and_matches_prepared_data(monkeypatch):
    from experiments.v22_gnn_training import train

    class FakeData:
        graphs = {'kernel': (object(), object())}
        actions = {'kernel': {}}
        demand_cache = {}
        preprocessed_action_cache = {'kernel': {}}

    calls = []
    marker = object()
    monkeypatch.setattr(train.overlay, 'enrich',
                        lambda base, profile, clock, authentication=None: calls.append(
                            (profile, clock, authentication)) or marker)
    monkeypatch.setattr(train, 'candidate_demand_tensor',
                        lambda graph, assignment, actions: (torch.tensor([float(assignment['x'])]), None))
    data = FakeData()
    cache, candidate_cache = {}, {}
    row_a = {'kernel': 'kernel', 'profile': 'zcu104', 'clock': 10.,
             'assignment': {'x': 1}, 'config_hash': 'a'}
    row_b = {'kernel': 'kernel', 'profile': 'zcu104', 'clock': 10.,
             'assignment': {'x': 2}, 'config_hash': 'b'}
    cached_a = _cached_candidate_inputs(data, row_a, cache, candidate_cache)
    cached_b = _cached_candidate_inputs(data, row_b, cache, candidate_cache)
    assert len(calls) == 1

    same_assignment_other_target = dict(row_a, profile='u200', clock=3.33)
    cached_other_target = _cached_candidate_inputs(
        data, same_assignment_other_target, cache, candidate_cache)
    assert cached_other_target is cached_a
    assert len(calls) == 1

    # Match the existing uncached PreparedData path under the same controlled
    # feature implementations, and require tensor identity for both candidates.
    legacy_a = train.PreparedData.candidate_inputs(data, row_a)
    legacy_b = train.PreparedData.candidate_inputs(data, row_b)
    legacy_other_target = train.PreparedData.candidate_inputs(data, same_assignment_other_target)
    for cached, legacy in ((cached_a, legacy_a), (cached_b, legacy_b),
                           (cached_other_target, legacy_other_target)):
        assert torch.equal(cached[0], legacy[0])
        assert torch.equal(cached[1], legacy[1])


def test_full_assignment_target_coverage_detects_real_winner_reversal():
    def row(profile, config_hash, latency, resource, capacities):
        return {'kernel': 'k', 'profile': profile, 'clock': 10.0,
                'split': 'train', 'config_hash': config_hash,
                'physical': [latency, resource, resource, resource, resource],
                'capacities': capacities}

    rows = [
        row('zcu104', 'a', 1.0, 4.0, [10.0] * 4),
        row('zcu104', 'b', 2.0, 8.0, [10.0] * 4),
        row('u200', 'a', 2.0, 9.0, [20.0] * 4),
        row('u200', 'b', 1.1, 5.0, [20.0] * 4),
    ]
    result = _coverage(rows, {'k': 'family'})
    paired = next(r for r in result['target_pairs']
                  if r['kernel'] == 'k' and r['dimension'] == 'device'
                  and r['left'] == ['zcu104', 10.0])
    assert paired['matched_assignments'] == 2
    assert paired['adp_winners_by_budget']['1']['winner_changed'] is True
    assert paired['adp_winners_by_budget']['1']['meaningful_reversal_5pct'] is True
    assert len(result['budget_pairs']) == 20


def test_candidate_pool_selection_is_deterministic_support_and_qor_blind():
    rows = [
        {'config_hash': f'h{i:02d}', 'assignment': {f'field{j}': str((i >> j) & 1)
                                                       for j in range(5)},
         'physical': [float(i)]}
        for i in range(24)
    ]
    support = {row['config_hash']: (5 if row['config_hash'] == 'h23' else 1)
               for row in rows}
    a = _select_qoblind(rows, 8, support)
    b = _select_qoblind(list(reversed(rows)), 8, support)
    assert [row['config_hash'] for row in a] == [row['config_hash'] for row in b]
    assert 'h23' in {row['config_hash'] for row in a}
    assert len({row['config_hash'] for row in a}) == 8


def test_adp_preference_pairs_keep_feasibility_direction_and_drop_ties():
    def row(name, latency, resource):
        return {'kernel': 'k', 'profile': 'zcu104', 'clock': 10.0,
                'config_hash': name, 'physical': [latency, resource, resource, resource, resource],
                'capacities': [10.0] * 4}
    records = [row('a', 1.0, 4.0), row('b', 1.03, 4.0), row('c', 2.0, 8.0)]
    pairs, counts = preference_pairs(records, 0.5, meaningful_gap=0.05)
    assert (0, 1, 'adp') not in pairs  # five-percent near tie
    assert (0, 2, 'feasible_infeasible') in pairs
    assert counts['near_tie'] == 1
    assert counts['feasible_infeasible'] == 2
    assert counts['both_infeasible'] == 0


def test_adp_preference_excludes_ambiguous_zero_area_rows():
    def row(name, latency, resource):
        return {'kernel': 'machsuite-viterbi', 'profile': 'u200', 'clock': 5.0,
                'config_hash': name, 'physical': [latency, resource, resource, resource, resource],
                'capacities': [10.0] * 4}
    records = [row('zero', 1.0, 0.0), row('other', 2.0, 4.0)]
    pairs, counts = preference_pairs(records, 1.0)
    assert pairs == []
    assert counts['zero_area_ambiguous'] == 1
    tight_pairs, tight_counts = preference_pairs(records, 0.1)
    assert tight_pairs == []
    assert tight_counts['zero_area_ambiguous'] == 1


def test_initial_graph_arms_match_and_difference_control_shares_compiler_state():
    models = _initial_models(123, profile_feature_dim=5)
    compiler = models['compiler'].state_dict()
    hardware = models['hardware'].state_dict()
    film = models['target_film'].state_dict()
    difference = models['compiler_difference']['base'].state_dict()
    assert compiler.keys() == hardware.keys() == difference.keys()
    assert all(key in film and torch.equal(compiler[key], film[key]) for key in compiler)
    assert all(torch.equal(compiler[key], hardware[key]) for key in compiler)
    assert all(torch.equal(compiler[key], difference[key]) for key in compiler)


def test_pair_trace_is_deterministic_and_excludes_validation_contexts():
    groups = [
        {'split': 'train', 'family': 'train_family', 'budget_pairs': {
            1.0: [(0, 1, 'adp'), (1, 2, 'feasibility')],
            0.75: [(0, 2, 'adp')]}},
        {'split': 'val', 'family': 'validation_family', 'budget_pairs': {
            1.0: [(0, 1, 'adp')]}}
    ]
    first = _pair_schedule(groups, 30, 123)
    second = _pair_schedule(groups, 30, 123)
    assert first == second
    assert all(groups[index]['split'] == 'train' for index, _, _ in first)


def test_zero_gated_target_film_preserves_base_memory_and_receives_gradient():
    torch.manual_seed(31)
    graph = _tiny_tensor_graph()
    graph.registered_mask[0, 2] = True
    graph.matched_mask[0, 2] = True
    graph.action_demand = torch.randn(1, 64, tensors.ACTION_DEMAND_DIM)
    base = V4bGraphModel(hardware_messages=False)
    film = TargetFilmV4bModel(profile_feature_dim=5)
    film.load_state_dict(base.state_dict(), strict=False)
    profile = torch.randn(1, 5)
    base.eval(); film.eval()
    expected, pooled = base.encode(graph)
    actual, _ = film.encode(graph, profile)
    assert torch.equal(expected, actual)
    actual[0, 2].sum().backward()
    assert film.gate.grad is not None and torch.isfinite(film.gate.grad)
