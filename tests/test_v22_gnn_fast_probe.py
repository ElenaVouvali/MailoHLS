import copy

import pytest
import torch
from torch_geometric.data import Data

from experiments.v22_gnn_training import fast_probe
from GNN_branch import target_graph_tensor_v22 as tensors


def test_candidate_selection_uses_assignments_not_qor_labels():
    rows = [
        {'config_hash': 'a', 'assignment': {'x': 0}, 'physical': [1]},
        {'config_hash': 'b', 'assignment': {'x': 1}, 'physical': [100]},
        {'config_hash': 'c', 'assignment': {'x': 2}, 'physical': [2]},
    ]
    changed_labels = copy.deepcopy(rows)
    for row in changed_labels:
        row['physical'] = [-row['physical'][0]]
    assert [r['config_hash'] for r in fast_probe._select_training_candidates(rows, 2)] == [
        r['config_hash'] for r in fast_probe._select_training_candidates(changed_labels, 2)]


def test_candidate_distance_uses_complete_action_fields():
    assert fast_probe._assignment_distance({'PIPE_L1': 0, 'UNROLL_L1': 2},
                                           {'PIPE_L1': 1, 'UNROLL_L1': 2}) == 1
    assert fast_probe._assignment_distance({'PIPE_L1': 0},
                                           {'PIPE_L1': 0, 'UNROLL_L1': 4}) == 1


def test_graph_tensor_equality_checks_every_field():
    left = Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))
    right = Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))
    assert fast_probe._tensor_fields_equal(left, right)
    right.edge_index = torch.tensor([[0], [1]])
    assert not fast_probe._tensor_fields_equal(left, right)


def test_cache_keys_bind_graph_candidate_context_and_feature_identity():
    record = {'kernel': 'k', 'profile': 'zcu104', 'clock': 10.0,
              'config_hash': 'a', 'source_action_mapping_sha256': 'mapping',
              'feature_impl_sha256': 'features'}
    assert fast_probe._record_key(record, 'graph') == fast_probe._record_key(record, 'graph')
    assert fast_probe._record_key(record, 'graph') != fast_probe._record_key(
        {**record, 'config_hash': 'b'}, 'graph')
    assert fast_probe._record_key(record, 'graph') != fast_probe._record_key(
        {**record, 'source_action_mapping_sha256': 'other'}, 'graph')
    assert fast_probe._graph_key(record, 'graph', 1.0) != fast_probe._graph_key(
        record, 'graph', 0.5)
    assert fast_probe._graph_key(record, 'graph', 1.0) != fast_probe._graph_key(
        {**record, 'profile': 'u200'}, 'graph', 1.0)


def test_probe_roster_is_family_disjoint_by_construction():
    assert set(fast_probe.TRAIN_KERNELS).isdisjoint(fast_probe.VALIDATION_KERNELS)
    assert len(fast_probe.VALIDATION_KERNELS) >= 2


def test_probe_runtime_budget_is_bounded():
    with pytest.raises(ValueError, match='20-minute'):
        fast_probe.run('/nonexistent', '/tmp/unused', 'V100S', 'uuid',
                       runtime_limit_seconds=fast_probe.GPU_PHASE_LIMIT_SECONDS + 1)


def test_numerical_arm_is_derived_from_same_hardware_graph():
    # Hardware-edge filtering retains numerical context and node features.
    data = Data(x=torch.randn(3, tensors.NODE_DIM),
                node_type=torch.tensor([0, 0, len(tensors.NODE_TYPES)-1]),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_type=torch.tensor([0, 0]), edge_attr=torch.zeros(2, tensors.EDGE_DIM),
                hardware_context=torch.tensor([[1.0, 2.0]]))
    numerical = tensors.without_hardware_messages(data)
    assert torch.equal(numerical.x, data.x)
    assert torch.equal(numerical.hardware_context, data.hardware_context)
    assert numerical.edge_index.shape[1] <= data.edge_index.shape[1]


def test_nonempty_outputs_are_never_overwritten(tmp_path):
    prepared = tmp_path / 'existing-cache'
    prepared.mkdir()
    (prepared / 'keep.txt').write_text('original')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        fast_probe.prepare(tmp_path / 'not-used', prepared)
    run_output = tmp_path / 'existing-run'
    run_output.mkdir()
    (run_output / 'keep.txt').write_text('original')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        fast_probe.run(prepared, run_output, 'A2', 'uuid')
    assert (prepared / 'keep.txt').read_text() == 'original'
    assert (run_output / 'keep.txt').read_text() == 'original'
