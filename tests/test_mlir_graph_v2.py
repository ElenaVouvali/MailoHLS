"""Independent V2 semantics on CPU; hand graphs are never compiler evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import networkx as nx
import pytest

from GNN_branch import mlir_graph_v2 as v2
from GNN_branch.action_point_relations import get_action_point_relations as query, action_point_views, historical_relation


def hand_graph():
    graph = nx.MultiDiGraph(schema=v2.SCHEMA, view='canonical', graph_identity='hand-authored-unit-input',
                            provenance={'origin': 'hand_authored_not_compiler_output'})
    for symbol, fid in [('F', 0), ('H', 1)]:
        graph.add_node('op:' + symbol + ':function', category='operation', function=fid,
                       function_symbol=symbol, op_uid=symbol + ':function', text='func.func', operation_family='function')
    for label, ident in [('L1', 'A'), ('L3', 'B')]:
        value, root = 'value:' + ident, 'memory:' + ident
        graph.add_node(value, category='ssa_value', function=0, ssa_id=ident, is_memory=1, value_type='memref<8xf32>')
        graph.add_node(root, category='memory_object', function=0, root_value=value, root_kind='allocation')
        v2.add_relation(graph, value, root, 'resolves_to')
        graph.add_node('action:' + label, category='action_point', action_id=label, action_kind='array',
                       absolute_index=int(label[1:]), supported_directives=['ARRAY_PARTITION'])
        v2.add_relation(graph, 'action:' + label, root, 'targets')
    formal, formal_root = 'value:H:b1:a0', 'memory:H:b1:a0'
    graph.add_node(formal, category='ssa_value', function=1, ssa_id='H:b1:a0', function_argument_index=0,
                   is_memory=1, value_type='memref<8xf32>')
    graph.add_node(formal_root, category='memory_object', function=1, root_value=formal, root_kind='formal_parameter')
    v2.add_relation(graph, formal, formal_root, 'resolves_to')
    graph.add_node('op:H:op0', category='operation', function=1, op_uid='H:op0', text='memref.load', loop_context=[])
    v2.add_relation(graph, 'op:H:op0', formal_root, 'memory_access', access='read', concrete=True,
                    memory_operand=formal, operand_index=0)
    for i, (label, actual) in enumerate([('L9', 'A'), ('L19', 'B')]):
        loop, call = f'op:F:op{i}', f'op:F:op{i + 2}'
        graph.add_node(loop, category='operation', function=0, op_uid=f'F:op{i}', is_loop=1, loop_context=[])
        graph.add_node('action:' + label, category='action_point', action_id=label, action_kind='loop',
                       absolute_index=int(label[1:]), supported_directives=['PIPELINE', 'UNROLL'])
        v2.add_relation(graph, 'action:' + label, loop, 'targets')
        graph.add_node(call, category='operation', function=0, op_uid=f'F:op{i + 2}', text='func.call',
                       operation_family='call', call_identity=f'F:op{i + 2}', call_resolution='proven',
                       caller_function='F', callee_function='H', loop_context=[loop])
        v2.add_relation(graph, call, 'op:H:function', 'calls', call_operation=call, call_uid=f'F:op{i + 2}')
        v2.add_relation(graph, 'value:' + actual, call, 'operand', legacy_evidence={'position': 0})
        v2.add_relation(graph, 'value:' + actual, formal, 'actual_to_formal', call_operation=call,
                        call_uid=f'F:op{i + 2}', caller_function='F', callee_function='H', argument_index=0)
    return graph


def test_two_calls_different_arrays_are_distinct_not_shared_helper_reachability():
    graph = hand_graph(); v2.validate(graph)
    left, right = query(graph, 'L9'), query(graph, 'L19')
    assert {r['neighbor'] for r in left} == {'L1'}
    assert {r['neighbor'] for r in right} == {'L3'}
    assert left[0]['access_path'] == right[0]['access_path'] == 'through_call'
    assert left[0]['call_chain'] == ['op:F:op2'] and right[0]['call_chain'] == ['op:F:op3']
    assert left[0]['callsite_bindings'][0]['actual'] == 'value:A'
    assert right[0]['callsite_bindings'][0]['actual'] == 'value:B'


@pytest.mark.parametrize('field,value', [('call_uid', 'F:op3'), ('argument_index', 1), ('call_operation', 'missing')])
def test_no_argument_position_or_wrong_call_identity_fallback(field, value):
    graph = hand_graph()
    edge = next(e for _, _, e in graph.edges(data=True) if e['relation_type'] == 'actual_to_formal')
    edge[field] = value
    with pytest.raises(ValueError, match='Binding'):
        v2.validate(graph)


def test_wrong_callee_is_rejected():
    graph = hand_graph(); graph.nodes['op:F:op2']['callee_function'] = 'other'
    with pytest.raises(ValueError, match='callee'):
        v2.validate(graph)


def test_swapped_two_argument_bindings_preserve_each_call_context():
    graph = hand_graph()
    formal, root = 'value:H:b1:a1', 'memory:H:b1:a1'
    graph.add_node(formal, category='ssa_value', function=1, ssa_id='H:b1:a1', function_argument_index=1,
                   is_memory=1, value_type='memref<8xf32>')
    graph.add_node(root, category='memory_object', function=1, root_value=formal, root_kind='formal_parameter')
    v2.add_relation(graph, formal, root, 'resolves_to')
    for actual, call in [('value:B','op:F:op2'),('value:A','op:F:op3')]:
        v2.add_relation(graph, actual, call, 'operand', legacy_evidence={'position':1})
        v2.add_relation(graph, actual, formal, 'actual_to_formal', call_operation=call,
                        call_uid=graph.nodes[call]['op_uid'], argument_index=1, caller_function='F', callee_function='H')
    graph.add_node('op:H:op1', category='operation', function=1, op_uid='H:op1', text='memref.store', loop_context=[])
    v2.add_relation(graph, 'op:H:op1', root, 'memory_access', access='write', concrete=True,
                    memory_operand=formal, operand_index=1)
    v2.validate(graph)
    assert {(r['neighbor'],r['access']) for r in query(graph,'L9')} == {('L1','read'),('L3','write')}
    assert {(r['neighbor'],r['access']) for r in query(graph,'L19')} == {('L3','read'),('L1','write')}


@pytest.mark.parametrize('mode', ['read', 'write', 'readwrite'])
def test_local_memory_access_has_uniform_operation_to_object_direction(mode):
    graph = hand_graph(); loop = 'op:F:op0'
    graph.add_node('op:F:op4', category='operation', op_uid='F:op4', function=0, loop_context=[loop], text='memref.store')
    v2.add_relation(graph, 'op:F:op4', 'memory:A', 'memory_access', access=mode, concrete=True,
                    memory_operand='value:A', operand_index=0, access_path='local')
    v2.validate(graph)
    local = [r for r in query(graph, 'L9') if r['access_path'] == 'local']
    assert len(local) == 1 and local[0]['access'] == mode and local[0]['memory_root'] == 'memory:A'


def test_call_summaries_cannot_become_concrete_access():
    graph = hand_graph()
    v2.add_relation(graph, 'op:F:op2', 'memory:A', 'memory_access', access='read', concrete=True)
    with pytest.raises(ValueError, match='call summary'):
        v2.validate(graph)


def test_ambiguous_roots_do_not_import_another_calls_proven_binding():
    graph = hand_graph()
    v2.add_relation(graph, 'value:A', 'memory:B', 'resolves_to', 'possible')
    assert not [r for r in query(graph, 'L9') if r['relation_type'] == 'memory_access']
    assert {r['neighbor'] for r in query(graph, 'L19')} == {'L3'}


def test_recursive_or_unresolved_call_is_only_possible():
    graph = hand_graph(); graph.nodes['op:F:op2']['call_resolution'] = 'possible'
    graph.nodes['op:F:op2']['recursive_call'] = True
    assert not query(graph, 'L9')
    rows = query(graph, 'L9', 'all')
    assert rows and all(r['evidence_status'] == 'possible' for r in rows)


def test_shared_storage_is_not_a_proven_dependency_and_uncertain_direction_is_retained():
    graph = hand_graph()
    assert not [r for r in query(graph, 'L9', 'all') if r['relation_type'] == 'memory_dependence']
    for i, loop in [(4, 'op:F:op0'), (5, 'op:F:op1')]:
        graph.add_node(f'op:F:op{i}', category='operation', function=0, op_uid=f'F:op{i}', loop_context=[loop])
    v2.add_relation(graph, 'op:F:op5', 'op:F:op4', 'depends_on', 'possible', dependence_kind='RAW', source_analysis={'result': 'unknown'})
    assert not [r for r in query(graph, 'L9') if r['relation_type'] == 'memory_dependence']
    rows = [r for r in query(graph, 'L9', 'all') if r['relation_type'] == 'memory_dependence']
    assert rows[0]['dependence_direction'] == 'precedes'
    edge = next(e for _, _, e in graph.edges(data=True) if e['relation_type'] == 'depends_on')
    edge['evidence_status'] = 'proven'
    assert any(r.get('dependence_direction') == 'precedes' for r in query(graph, 'L9'))
    assert any(r.get('dependence_direction') == 'depends_on' for r in query(graph, 'L19'))


def test_alias_fact_does_not_create_an_access_or_must_alias_proof():
    graph = hand_graph()
    v2.add_relation(graph, 'value:A', 'value:B', 'alias', 'possible', alias_kind='may_alias', source_analysis={'result': 'may_alias'})
    assert not [r for r in query(graph, 'L1') if r['relation_type'] == 'memory_alias']
    assert any(r.get('alias_kind') == 'may_alias' for r in query(graph, 'L1', 'all'))
    assert {r['neighbor'] for r in query(graph, 'L9')} == {'L1'}


def test_sparse_absolute_action_point_views_and_hierarchy():
    graph = hand_graph(); v2.add_relation(graph, 'op:F:op0', 'op:F:op1', 'loop_child')
    views = action_point_views(graph, max_slots=20)
    assert [i + 1 for i, active in enumerate(views['active_mask']) if active] == [1, 3, 9, 19]
    assert views['labels'][1] == -1 and not any(views['legacy_relation_mask'][1])
    assert views['legacy_relation_bits'][8][18] & 4 and views['legacy_relation_bits'][18][8] & 2
    assert views['legacy_relation_bits'][8][0] & 8 and views['legacy_relation_bits'][0][8] & 8
    with pytest.raises(ValueError, match='truncated'): action_point_views(graph, 8)
    with pytest.raises(ValueError, match='inactive'): query(graph, 'L2')
    # Evidence keys identify an edge across the whole graph, not just one pair.
    keys = [key for _,_,key in graph.edges(keys=True)]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize('old', ['direct_verified', 'indirect_verified', 'conservative_unknown', 'no_evidenced_relation'])
def test_legacy_adapter_preserves_original_record_and_ambiguity(old):
    original = {'verified_relation_class': old, 'original_relation_types': ['loop_array_access'], 'evidence': ['original']}
    out = historical_relation(original)
    assert out['legacy_record'] == original and out['legacy_classification'] == old
    assert out['mapping_ambiguity'] and out['evidence_status'] != 'proven'
    if old == 'no_evidenced_relation': assert out['evidence_status'] == 'not_established'


def test_compaction_is_lossless_preserves_ssa_and_all_query_views(tmp_path):
    graph = hand_graph()
    graph.add_node('feature:shape', category='constant_or_feature', feature_role='auxiliary', feature_kind='memory_shape', text='rank_1')
    graph.add_node('literal:one', category='constant_or_feature', feature_role='semantic_constant', feature_kind='constant', text='1')
    v2.add_relation(graph, 'feature:shape', 'value:A', 'feature', legacy_evidence={'role': 'memory_shape_feature'})
    v2.add_relation(graph, 'literal:one', 'op:F:op2', 'literal')
    before = v2.graph_payload(graph)
    compact = v2.compact_view(graph)
    assert len(compact) == len(graph) - 1 and 'literal:one' in compact and 'value:A' in compact
    assert v2.graph_payload(v2.expand_compact(compact)) == before
    assert action_point_views(graph) == action_point_views(compact)
    assert v2.graph_payload(graph) == before  # queries and compaction did not mutate canonical data.
    path = tmp_path/'compact.json'; v2.write_graph(compact, path)
    assert v2.canonical_json(v2.graph_payload(v2.expand_compact(v2.read_graph(path)))) == v2.canonical_json(before)
    original = path.read_bytes(); v2.write_graph(compact, path); assert original == path.read_bytes()
    bad = deepcopy(compact); bad.nodes['value:A']['value_type'] = 'different'
    with pytest.raises(ValueError, match='overwrite'): v2.write_graph(bad, path)
    from experiments.refinement.relations import graph_relations
    with pytest.raises(ValueError, match='GEXF'):
        graph_relations(path, 'source-hash', 'metadata-hash', {})


@pytest.fixture(scope='module')
def native_tools():
    from experiments.refinement.relations import compiler_tools
    build = os.environ.get('POLYGEIST_BUILD')
    local = Path.home()/'tools/Polygeist/build-mailohls-assertions'
    try:
        return compiler_tools(build=build or (str(local) if local.is_dir() else None))
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope='module')
def compiled_cases(native_tools, tmp_path_factory):
    from experiments.structural_audit import compare_v2 as compare
    from experiments.refinement import configurations as c
    root = compare.audit.HERE/'results/current/fixtures'
    if not (root/'shared_helper/kernel.mlir').is_file(): pytest.skip('Compiled R1.2 fixtures are unavailable')
    output = tmp_path_factory.mktemp('compiler_v2')
    cases = {}
    for name in compare.FIXTURE_EXPECTATIONS:
        p = root/name
        source_hashes = {f: c.sha256(p/f) for f in ('kernel.cpp', 'kernel_info.txt', 'kernel.mlir', 'kernel.gexf')}
        graph, record = compare._compile_case(name, p/'kernel.cpp', p/'kernel_info.txt', p/'kernel.mlir', p/'kernel.gexf', output, native_tools)
        assert all(c.sha256(p/f) == digest for f, digest in source_hashes.items())
        cases[name] = (graph, record)
    return cases


def test_actual_compiler_fixtures_and_deterministic_generation(compiled_cases, native_tools, tmp_path):
    from experiments.structural_audit.compare_v2 import FIXTURE_EXPECTATIONS
    for name, (graph, record) in compiled_cases.items():
        assert record['preservation']['exact_canonical_reconstruction']
        assert record['call_bindings'] == record['explicit_call_identity_bindings']
        assert not record['compatibility_mapping_unresolved']
        for point, neighbor, kind, expected, role in FIXTURE_EXPECTATIONS[name]:
            rows = [r for r in query(graph, point) if r['neighbor'] == neighbor and r['relation_type'] == kind]
            assert bool(rows) == expected, (name, point, neighbor)
            if expected: assert any(role in (r.get('access_path'),r.get('hierarchy'),r.get('dependence_direction')) for r in rows)
    first, _ = compiled_cases['shared_helper']
    from experiments.structural_audit import compare_v2 as compare
    root = compare.audit.HERE/'results/current/fixtures/shared_helper'
    second, _ = compare._compile_case('again', root/'kernel.cpp', root/'kernel_info.txt', root/'kernel.mlir', root/'kernel.gexf', tmp_path, native_tools)
    assert v2.canonical_json(v2.graph_payload(first)) == v2.canonical_json(v2.graph_payload(second))


def test_real_GAN_context_precision(native_tools, tmp_path):
    from experiments.structural_audit import compare_v2 as compare
    if not compare.audit.DEFAULT_SHARED.is_file(): pytest.skip('Authenticated external GAN archive unavailable')
    shared = json.loads(compare.audit.DEFAULT_SHARED.read_text())
    source, metadata, old_path, old, _, _ = compare.audit.load_kernel('GAN', shared)
    raw = compare.audit.raw_artifacts('GAN', old_path, old)
    graph, record = compare._compile_case('GAN', source, metadata, compare.c.repo_path(raw['raw_mlir']), old_path, tmp_path, native_tools)
    rows = query(graph, 'L11')
    assert not any(r['neighbor'] == 'L2' and r['relation_type'] == 'memory_access' for r in rows)
    correct = [r for r in rows if r['neighbor'] == 'L3' and r['relation_type'] == 'memory_access']
    assert correct and all(r['access_path'] == 'through_call' and r['callsite_bindings'] for r in correct)
    assert record['active_action_points'] == [f'L{i}' for i in range(1,12)]


def test_fresh_frontend_and_canonical_global_storage(native_tools, tmp_path):
    source = tmp_path/'kernel.cpp'
    source.write_text('float g[8]={1,2,3,4,5,6,7,8};\n'
        'extern "C" __attribute__((noinline)) float first(){return g[0];}\n'
        'extern "C" __attribute__((noinline)) float second(){return g[1];}\n'
        'extern "C" float top(){float s=0;\nL9: for(int i=0;i<8;i++){s+=first()+second();}\nreturn s;}\n')
    source.with_name('kernel_info.txt').write_text('top\nL9,loop,8\n')
    path = tmp_path/'global.v2.json'
    env = {**os.environ,'PYTHONPATH':native_tools['pythonpath'],'PYTHONHASHSEED':'0'}
    result = subprocess.run([native_tools['python'],'-B','-m','GNN_branch.mlir_graph_v2',
        '--source',str(source),'--cgeist',native_tools['cgeist'],'--output',str(path)],
        env=env,cwd=Path(v2.__file__).resolve().parents[1],capture_output=True,text=True,timeout=120)
    assert result.returncode == 0, result.stderr
    graph = v2.read_graph(path)
    globals_ = [(n,a) for n,a in graph.nodes(data=True) if a.get('root_kind')=='global']
    assert len(globals_) == 1 and len(globals_[0][1]['root_values']) >= 2
    definition = globals_[0][1]['global_definition']
    assert definition['status'] == 'compiler_recorded' and 'initial_value' in definition['attributes']
    assert 'action:L9' in graph and graph.nodes['action:L9']['absolute_index'] == 9
