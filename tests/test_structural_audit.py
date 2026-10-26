"""CPU diagnostics with independent expectations; no vendor tools or large model."""
from argparse import Namespace
import copy
import json
from pathlib import Path

import pytest
import torch

from experiments.structural_audit import audit as a
from experiments.refinement import configurations as c, relations as legacy
from LLM_branch.common.directive_domains import SourceAction


SPECS = json.loads((a.HERE / 'fixtures/semantics.json').read_text())


@pytest.mark.parametrize('spec', SPECS, ids=lambda s: s['name'])
def test_independently_specified_semantic_fixtures(spec):
    assert a.fixture_checks(spec['name']) == spec['unit_expected']
    assert spec['program'] and spec['semantics']


def test_two_calls_same_helper_no_position_only_binding_or_false_direct_relation():
    graph = a.fixture_graph('shared_helper')
    before = list(graph.edges(keys=True, data=True))
    # The old implementation accepts the unscoped positional binding for the
    # actual passed in this loop. The stricter audit deliberately withholds proof.
    assert legacy.verify_pair(graph, 'L1', 'L9')['verified_relation_class'] == 'indirect_verified'
    own = a.strict_verify_pair(graph, 'L1', 'L9')
    other = a.strict_verify_pair(graph, 'L3', 'L9')
    assert own['verified_relation_class'] == other['verified_relation_class'] == 'conservative_unknown'
    assert own['rejected_binding_count'] == 2
    assert list(graph.edges(keys=True, data=True)) == before
    assert own['legacy_verifier_class'] == 'indirect_verified'


def test_explicit_callsite_must_match_actual_operand_callee_and_position():
    graph = a.fixture_graph('bound_helper')
    assert a.strict_verify_pair(graph, 'L1', 'L9')['verified_relation_class'] == 'indirect_verified'
    keys = [key for _, v, key, e in graph.out_edges('2', keys=True, data=True) if e.get('role') == 'actual_to_formal']
    edge = graph['2']['32'][keys[0]]
    for changes in ({'callsite': '21'}, {'position': 2}, {'callsite': 'missing'}):
        original = dict(edge); edge.update(changes)
        assert a.strict_verify_pair(graph, 'L1', 'L9')['verified_relation_class'] == 'conservative_unknown'
        edge.clear(); edge.update(original)
    graph.nodes['32']['function'] = 99
    assert a.strict_verify_pair(graph, 'L1', 'L9')['verified_relation_class'] == 'conservative_unknown'


def test_proven_dependence_direction_not_uncertain_reverse():
    observed = a.fixture_checks('dependencies')
    assert observed['proven_forward'] and not observed['uncertain_reverse']


def test_native_write_direction_is_recognized_without_mutating_legacy():
    graph = a.fixture_graph('direct_write')
    assert legacy.verify_pair(graph, 'L1', 'L9')['verified_relation_class'] == 'conservative_unknown'
    result = a.strict_verify_pair(graph, 'L1', 'L9')
    assert result['verified_relation_class'] == 'direct_verified'
    assert any(e['decision'] == 'recognize_native_directed_write' for e in result['access_evidence_adaptations'])
    # A lone reverse-role edge is not enough to invent the forward proof.
    graph.remove_edges_from([('11', '2')])
    assert a.strict_verify_pair(graph, 'L1', 'L9')['verified_relation_class'] == 'conservative_unknown'


def test_call_effect_summary_is_not_concrete_direct_memory_access():
    graph = a.fixture_graph('shared_helper')
    graph.add_edge('1', '11', flow=7, role='array_read')
    graph.add_edge('2', '11', flow=10, role='read', operand_index=0)
    assert legacy.verify_pair(graph, 'L1', 'L9')['verified_relation_class'] == 'direct_verified'
    result = a.strict_verify_pair(graph, 'L1', 'L9')
    assert result['verified_relation_class'] == 'conservative_unknown'
    assert any(e['decision'] == 'exclude_call_summary_or_cross_function_root_as_direct_proof' for e in result['access_evidence_adaptations'])


def test_unknowns_and_machine_drafts_never_inflate_reviewed_accuracy():
    base = {'kernel': 'toy', 'relationship_type': 'loop_array_access'}
    rows = [{**base, 'expected': True, 'observed': True, 'review_status': 'draft'},
            {**base, 'expected': False, 'observed': True, 'review_status': 'draft'},
            {**base, 'expected': None, 'observed': True}]
    out = a.compare_annotations(rows)[0]
    assert out['precision'] is None and out['recall'] is None
    assert out['draft_TP'] == out['draft_FP'] == out['unresolved'] == 1
    rows += [{**base, 'expected': True, 'observed': True, 'review_status': 'independently_reviewed', 'reviewer': 'fixture-reviewer'},
             {**base, 'expected': True, 'observed': False, 'review_status': 'independently_reviewed', 'reviewer': 'fixture-reviewer'}]
    out = a.compare_annotations(rows)[0]
    assert out['precision'] == 1 and out['recall'] == .5


def test_annotation_cannot_silently_change_source_or_claim_review(tmp_path):
    source = tmp_path / 'source.cpp'; source.write_text('L1: int a[8];\nL9: for(int i=0;i<8;++i){}\n')
    spec = {'kernel': 'toy', 'action_i': 'L1', 'action_j': 'L9', 'relationship_type': 'loop_array_access',
            'expected': True, 'source_quotes': ['L1:'], 'source_sha256': 'incorrect'}
    with pytest.raises(ValueError, match='source hash'):
        a.annotate(spec, source, None, {}, {}, 'hash')
    spec.update(source_sha256=c.sha256(source), review_status='independently_reviewed')
    with pytest.raises(ValueError, match='reviewer'):
        a.annotate(spec, source, None, {}, {}, 'hash')


def test_missing_graph_is_explicit():
    with pytest.raises(OSError):
        a.load_kernel('missing-kernel-for-audit', {})


def test_real_projector_sparse_slots_and_truncation():
    graph = a.fixture_graph('direct_read')
    actions = {k: SourceAction(kind='array', dimensions=((1, 8),)) for k in ('L1', 'L3')}
    actions.update({k: SourceAction(kind='loop', trip_count=8) for k in ('L9', 'L19')})
    _, active, mask, bits, _ = a.project(graph, actions, 20)
    assert [i + 1 for i, x in enumerate(active) if x] == [1, 3, 9, 19]
    assert not mask[1].any() and not bits[1].any()
    assert torch.equal(mask, bits.ne(0))
    with pytest.raises(ValueError, match='truncated'):
        a.project(graph, actions, 8)


def test_typed_relation_information_collapses_in_real_routing_interface():
    from LLM_branch.common.structural_memory import resolve_relation_mask
    first = torch.tensor([[1, 4, 0], [2, 1, 0], [0, 0, 0]], dtype=torch.int16)
    second = torch.tensor([[1, 16, 0], [32, 1, 0], [0, 0, 0]], dtype=torch.int16)
    assert not torch.equal(first, second)
    left = resolve_relation_mask({'relation_mask': first.ne(0), 'relation_bits': first}, 'compiler_relational')
    right = resolve_relation_mask({'relation_mask': second.ne(0), 'relation_bits': second}, 'compiler_relational')
    assert torch.equal(left, right) and not left[2].any() and left[0, 0]
    # Exercise the actual attention layer on tiny CPU tensors, not a language model.
    from LLM_branch.common.structural_xattn import MaskedCrossAttention
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        attention = MaskedCrossAttention(dim=8, dim_memory=4, dim_head=4, heads=1).eval()
        x, memory = torch.randn(1, 2, 8), torch.randn(1, 3, 4)
        kwargs = {'placeholder_slot_ids': torch.tensor([[1, 0]]),
                  'memory_mask': torch.tensor([[True, True, False]])}
        output = attention(x, memory, action_relation_mask=left[None], **kwargs)
        assert torch.equal(output, attention(x, memory, action_relation_mask=right[None], **kwargs))
        memory[:, 2] = 1e6
        assert torch.equal(output, attention(x, memory, action_relation_mask=left[None], **kwargs))


def test_saved_original_memory_and_encoder_tensors_exact_cpu_reproduction():
    name = 'machsuite-gemm-ncubed'
    root = c.ROOT / 'Checkpoints/gnn_final_absolute_direct_rank_s123/run1'
    schema_path = root / 'provenance/feature_schema.json'
    pack_path = c.ROOT / 'artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned' / (name + '.memory.pt')
    tensor_path = c.ROOT / 'GNN_branch/MLIR_dataset/all_kernels/graphs' / (name + '.pt')
    if not all(p.is_file() for p in (schema_path, pack_path, tensor_path)):
        pytest.skip('Saved research artifacts are local prerequisites, not required for unit fixtures')
    _, _, graph_path, graph, actions, _ = a.load_kernel(name, {})
    schema = json.loads(schema_path.read_text())
    encoded, _ = a.encoded_features(graph, schema)
    saved = torch.load(tensor_path, map_location='cpu', weights_only=False)
    assert torch.equal(encoded['node'], saved['x'])
    assert torch.equal(encoded['edge'], saved['edge_attr'])
    assert torch.equal(encoded['edge_index'], saved['edge_index'])
    pack = torch.load(pack_path, map_location='cpu', weights_only=False)
    before = c.sha256(pack_path)
    assert pack['source_gexf_sha256'] == c.sha256(graph_path)
    result = a.check_memory(pack, graph, actions, c.sha256(schema_path), c.sha256(root / 'gnn_checkpoint_contract.json'))
    assert result['shape'] == [64, 128] and result['active_slots'] == 3
    assert before == c.sha256(pack_path)
    broken = copy.deepcopy(pack); broken['labels'][0] = 3
    with pytest.raises(ValueError, match='alignment'):
        a.check_memory(broken, graph, actions, c.sha256(schema_path), c.sha256(root / 'gnn_checkpoint_contract.json'))
    broken = copy.deepcopy(pack); broken['action_relation_mask'][63, 0] = True
    with pytest.raises(ValueError, match='relation tensor'):
        a.check_memory(broken, graph, actions, c.sha256(schema_path), c.sha256(root / 'gnn_checkpoint_contract.json'))


def test_real_GAN_reference_comes_from_source_not_projection():
    if not a.DEFAULT_SHARED.is_file():
        pytest.skip('External GAN graph archive not installed')
    shared = json.loads(a.DEFAULT_SHARED.read_text())
    source, _, path, graph, actions, projected = a.load_kernel('GAN', shared)
    specs = [r for r in json.loads((a.HERE / 'fixtures/annotations.json').read_text())
             if r['kernel'] == 'GAN' and r['action_j'] == 'L11' and r['action_i'] in ('L2', 'L3')]
    rows = {r['action_i']: a.annotate(r, source, graph, actions, projected, c.sha256(path)) for r in specs}
    assert rows['L2']['expected'] is False and rows['L2']['observed'] is True
    assert rows['L2']['strict_verification']['verified_relation_class'] == 'conservative_unknown'
    # Source-positive, but old direct proof is a summarized func.call access.
    # Do not preserve the old confidence label by weakening the stricter rule.
    assert rows['L3']['expected'] is True and rows['L3']['observed'] is True
    assert rows['L3']['strict_verification']['legacy_verifier_class'] == 'direct_verified'
    assert rows['L3']['strict_verification']['verified_relation_class'] == 'conservative_unknown'
    witnesses = a.projection_witness(graph, 'L2', 'L11')
    assert witnesses and any(e.get('role') == 'calls' for w in witnesses for e in w['region_path'])


def test_fixture_mode_never_invokes_compiler_without_explicit_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, 'compiler_tools', lambda **kwargs: pytest.fail('Unexpected compiler invocation'))
    records = a.validate_fixtures(Namespace(output_dir=str(tmp_path), compile_fixtures=False,
                                          polygeist_build=None, mlir_python=None))
    assert len(records) == 8 and all(r['compiler_status'] == 'not_run' for r in records)


def test_real_audit_writes_outputs_preserves_inputs_and_freezes_manifest(tmp_path):
    if not a.DEFAULT_SHARED.is_file():
        pytest.skip('External graph archive not installed')
    annotations = tmp_path / 'annotations.json'
    selected = [r for r in json.loads((a.HERE / 'fixtures/annotations.json').read_text()) if r['kernel'] == 'GAN']
    annotations.write_text(json.dumps(selected))
    output = tmp_path / 'audit'
    args = Namespace(output_dir=str(output), shared_manifest=str(a.DEFAULT_SHARED),
                     annotations=str(annotations), kernels=['GAN', 'Kalman'], include_originals=False)
    historical_before = c.sha256(a.DEFAULT_SHARED)
    trace, rows = a.run_audit(args)
    assert len(rows) == 8 and trace['kernels']['GAN']['status'] == 'audited'
    if (c.ROOT / 'artifacts/gnn/absolute_direct_rank_epoch9_s123/multiscale_aligned/serrano-kalman-filter.memory.pt').is_file():
        assert trace['kernels']['Kalman']['memory']['status'] == 'validated'
        assert trace['kernels']['Kalman']['memory']['active_slots'] == 11
        assert trace['kernels']['Kalman']['raw_mlir_status'] == 'hash_verified'
        assert trace['kernels']['Kalman']['saved_tensor']['status'] == 'validated'
    for name in ('manifest.json', 'relationship_annotations.jsonl', 'relationship_comparison.csv',
                 'relation_errors.jsonl', 'representation_trace.json', 'structural_audit_report.md'):
        assert (output / name).is_file()
    assert historical_before == c.sha256(a.DEFAULT_SHARED)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['protected_historical_sha256']
    assert all(c.sha256(c.repo_path(p)) == digest for p, digest in manifest['inputs_sha256'].items())
    before = c.sha256(output / 'structural_audit_report.md')
    selected[0]['expected'] = False
    annotations.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match='inputs changed'):
        a.run_audit(args)
    assert c.sha256(output / 'structural_audit_report.md') == before
