"""Focused CPU checks for RC3 routing and the future XAttn adapter."""
from pathlib import Path

import pytest
import torch

from GNN_branch.mlir_graph_v2 import read_graph
from LLM_branch.common.structural_xattn import MaskedCrossAttention
from experiments.rc3_fusion_foundation.rc3_action_relations import build_action_relations
from experiments.rc3_fusion_foundation.structural_memory_v1 import (
    EVIDENCE_CAPACITY, adapt_for_xattn, pad_evidence_batch,
)

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / 'paper_figures/gemv_rc3_20261105/gemv.canonical.json'
VARIANT_GRAPH = Path(__file__).resolve().parent / 'benchmark/variants/reduce32/graph.json'
ARTIFACT = ROOT / 'artifacts/rc3_fusion_foundation/v1_20261107/gemv_rc3_original.pt'
EXPECTED = {
    'L1': ['L1', 'L4'], 'L2': ['L2', 'L4'], 'L3': ['L3', 'L4'],
    'L4': ['L1', 'L2', 'L3', 'L4', 'L5'], 'L5': ['L1', 'L2', 'L4', 'L5'],
}


@pytest.fixture(scope='module')
def pack():
    if not ARTIFACT.exists():
        pytest.skip('Run build_memory.py first')
    return torch.load(ARTIFACT, map_location='cpu', weights_only=False)


def test_relation_routes_are_compiler_derived_absolute_slots():
    graph = read_graph(GRAPH)
    rel = build_action_relations(graph, expected_graph_identity='42fe84cb8b8c2f5665dc016fcfb6a969f93b848a3407c7343100b6b4bcabae21')
    routes = {f'L{i+1}': [f'L{j+1}' for j, v in enumerate(row) if v]
              for i, row in enumerate(rel['eligibility']) if any(row)}
    assert routes == EXPECTED
    assert rel['absolute_slots']['valid_mask'][:5] == [True] * 5
    assert rel['absolute_slots']['valid_mask'][5:] == [False] * 59
    for row in rel['ledger']:
        if row['routing_eligible'] and row['query_action'] != row['key_action']:
            assert (row.get('supporting_edge') or row.get('supporting_edges')
                    or row.get('source_edge') or row.get('edge_key'))
            assert row['graph_identity'] == graph.graph['graph_identity']
    assert all(x['evidence_status'] in ('proven', 'possible', 'not_established')
               for x in rel['non_routing_evidence'])
    variant = build_action_relations(read_graph(VARIANT_GRAPH))
    assert variant['relation_schema_id'] == rel['relation_schema_id']
    assert variant['payload_sha256'] != rel['payload_sha256']


def test_a_b_c_keep_same_absolute_action_vectors_and_selective_evidence(pack):
    action = pack['action_memory']
    assert tuple(action.shape) == (1, 64, 128)
    a, b, c = (adapt_for_xattn(pack, mode)['memory'] for mode in ('A', 'B', 'C'))
    assert torch.equal(a, b)
    assert torch.equal(a, c[:, :64])
    assert pack['evidence_count'] <= EVIDENCE_CAPACITY
    assert pack['evidence_memory'].shape == (1, pack['evidence_count'], 128)
    assert pack['action_relation_mask'].shape == (1, 64, 64)
    assert pack['action_evidence_mask'].shape == (1, 64, pack['evidence_count'])
    counts = pack['action_evidence_mask'][0, :5].sum(-1).tolist()
    assert all(count > 0 for count in counts)
    assert len(set(counts)) > 1
    assert not pack['action_evidence_mask'][0, 5:].any()
    assert torch.equal(pack['evidence_valid_mask'][0], torch.ones(pack['evidence_count'], dtype=torch.bool))


def test_adapter_and_real_attention_mask_for_a_b_c(pack):
    torch.manual_seed(11)
    x = torch.randn(1, 3, 16)
    # Position 0 has no action anchor; 1/2 are L1 positions.
    slot_ids = torch.tensor([[0, 1, 1]])
    for mode in ('A', 'B', 'C'):
        adapted = adapt_for_xattn(pack, mode)
        memory, memory_mask = adapted['memory'], adapted['memory_mask']
        assert memory.shape[0] == 1 and memory.shape[2] == 128
        assert memory_mask.shape == memory.shape[:2]
        relation = adapted['action_relation_mask'] if mode != 'A' else None
        module = MaskedCrossAttention(dim=16, dim_memory=128, dim_head=8, heads=2,
                                      only_attend_immediate_memory=True, mask_mode='segment')
        out = module(x, memory, placeholder_slot_ids=slot_ids, memory_mask=memory_mask,
                     action_relation_mask=relation)
        assert out.shape == x.shape and torch.isfinite(out).all()
        perturbed = memory.clone()
        allowed = (relation[0, 0].clone() if relation is not None else
                   torch.nn.functional.one_hot(torch.tensor(0), num_classes=memory.shape[1]).bool())
        allowed &= memory_mask[0]
        perturbed[:, ~allowed] += 1000.0
        same = module(x, perturbed, placeholder_slot_ids=slot_ids, memory_mask=memory_mask,
                      action_relation_mask=relation)
        assert torch.allclose(out, same, atol=1e-6, rtol=1e-6)
        if mode == 'A':
            assert memory.shape[1] == 64
        if mode == 'B':
            assert relation[0, 0].nonzero().flatten().tolist() == [0, 3]
        if mode == 'C':
            expected = pack['action_evidence_mask'][0, 0].sum().item()
            assert relation[0, 0].sum().item() == 2 + expected
            assert not relation[0, 64:].any()


def test_zero_valid_memory_is_finite_and_silent():
    torch.manual_seed(12)
    module = MaskedCrossAttention(dim=16, dim_memory=8, dim_head=8, heads=2)
    x = torch.randn(1, 2, 16)
    memory = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    ids = torch.zeros(1, 2, dtype=torch.long)
    result = module(x, memory, placeholder_slot_ids=ids, memory_mask=mask)
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(result))


def test_active_action_with_zero_legal_keys_fails_loudly():
    module = MaskedCrossAttention(dim=16, dim_memory=8, dim_head=8, heads=2)
    with pytest.raises(RuntimeError, match='zero legal memory keys'):
        module(torch.randn(1, 1, 16), torch.randn(1, 4, 8),
               placeholder_slot_ids=torch.tensor([[1]]),
               memory_mask=torch.ones(1, 4, dtype=torch.bool),
               action_relation_mask=torch.zeros(1, 4, 4, dtype=torch.bool))


def test_batch_padding_is_right_padded_and_never_truncated(pack):
    one = {'evidence_memory': pack['evidence_memory'],
           'evidence_valid_mask': pack['evidence_valid_mask'],
           'evidence_known_mask': pack['evidence_known_mask']}
    empty = {'evidence_memory': torch.zeros(1, 0, 128),
             'evidence_valid_mask': torch.zeros(1, 0, dtype=torch.bool),
             'evidence_known_mask': torch.zeros(1, 0, len(pack['known_field_names']), dtype=torch.bool)}
    batched = pad_evidence_batch([one, empty])
    assert batched['evidence_memory'].shape == (2, pack['evidence_count'], 128)
    assert batched['evidence_valid_mask'][0].all()
    assert not batched['evidence_valid_mask'][1].any()
    with pytest.raises(ValueError, match='exceeds cap'):
        pad_evidence_batch([one], capacity=pack['evidence_count'] - 1)
