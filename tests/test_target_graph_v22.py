"""Target-overlay tests on authenticated snapshots and explicitly hand-authored units.

No GPU, native MLIR installation, training, labels or synthesis is required.
Hand-authored graph evidence exercises consumer logic, not compiler correctness.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

from GNN_branch import mlir_graph_v2 as v2
from GNN_branch import target_graph_v22 as target
from GNN_branch.action_point_relations import get_action_point_relations, get_loop_semantics


@pytest.fixture(scope='module')
def release():
    manifest = json.loads(target.RELEASE.read_text())
    with target.authenticated_release(target.ROOT / manifest['archive']) as (root, _):
        yield root


@pytest.fixture
def snapshot(release):
    def load(prefix='spcl_example_03-', view='canonical'):
        path = next((release / view).glob(prefix + '*.json'))
        return target.authenticated_input(path)
    return load


def program_payload(graph):
    return v2.graph_payload(target.strip_overlay(graph))


def overlay_edges(graph, relation=None):
    return [(u, v, k, e) for u, v, k, e in graph.edges(keys=True, data=True)
            if e.get('_target_overlay') and (relation is None or e['relation_type'] == relation)]


def literal_assignment(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == name for n in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_verified_capacity_units_match_existing_contracts():
    registry = target.load_registry()
    gnn = literal_assignment(target.ROOT / 'GNN_branch/mlir_data.py', 'TARGET_DEVICE_CAPACITIES')
    llm = literal_assignment(target.ROOT / 'LLM_branch/common/mailohls_contract.py', 'DEVICE_RESOURCES')
    for key in ('zcu104', 'u200'):
        p = registry['profiles'][key]
        raw = {r: a['value'] for r, a in p['resources'].items()}
        assert raw == llm[p['part']]
        assert tuple(raw[r] for r in ('BRAM_18K', 'DSP', 'FF', 'LUT')) == gnn[p['part']]
    u50 = json.loads((target.ROOT / 'experiments/u50_adaptation/config.json').read_text())
    assert {r: p['value'] for r, p in registry['profiles']['u50']['resources'].items()} == u50['u50_available_resources']
    assert registry['capabilities']['BRAM_18K']['properties']['units_per_36k_block']['value'] == 2
    assert registry['capabilities']['LUT']['properties']['available_distributed_storage_luts']['value'] is None


@pytest.mark.parametrize('field,value', [
    ('value', 0), ('value', -1), ('value', True), ('value', float('nan')),
    ('unit', '36-Kibit block'), ('meaning', 'synthesized_consumption'),
    ('known', False), ('sources', ['missing']),
])
def test_bad_physical_profiles_are_rejected(field, value):
    registry = target.load_registry()
    registry['profiles']['zcu104']['resources']['BRAM_18K'][field] = value
    with pytest.raises(ValueError):
        target.validate_registry(registry)


def test_qor_and_candidate_fields_are_not_profile_features():
    for field in ('measured_latency', 'gold_directives', 'pareto_rank'):
        registry = target.load_registry()
        registry['profiles']['zcu104'][field] = 1
        with pytest.raises(ValueError, match='Unknown profile fields'):
            target.validate_registry(registry)


@pytest.mark.parametrize('clock', [3.33, 5, 10, 6.67, 0.125])
def test_arbitrary_positive_numeric_clock_supported(clock):
    p = target.load_registry()['profiles']['u50']
    assert target.deployment_context(p, clock)['clock_period_ns'] == clock


@pytest.mark.parametrize('clock', [0, -1, float('nan'), float('inf'), -float('inf'), True, '10'])
def test_invalid_clock_rejected(clock):
    with pytest.raises(ValueError):
        target.deployment_context(target.load_registry()['profiles']['zcu104'], clock)


@pytest.mark.parametrize('budget', [-1, 1.5, float('nan'), float('inf'), True, 1729])
def test_invalid_resource_budget_rejected(budget):
    with pytest.raises(ValueError):
        target.deployment_context(target.load_registry()['profiles']['zcu104'], 10, {'DSP': budget})


def test_platform_allocation_request_and_physical_capacity_stay_distinct(snapshot):
    program, auth = snapshot()
    registry = target.load_registry()
    before = deepcopy(registry)
    full = target.enrich(program, 'zcu104', 10, authentication=auth)
    restricted = target.enrich(program, 'zcu104', 10, {'DSP': 0, 'BRAM_18K': 128},
                               platform_available={'DSP': 800, 'BRAM_18K': 300}, authentication=auth)
    target.validate(restricted)
    for resource in target.RESOURCES:
        a = full.nodes[target._resource(resource)]
        b = restricted.nodes[target._resource(resource)]
        assert a['physical_capacity'] == b['physical_capacity']
    dsp = restricted.nodes[target._resource('DSP')]
    assert dsp['physical_capacity']['value'] == 1728
    assert dsp['platform_available']['value'] == 800
    assert dsp['requested_budget']['value'] == dsp['available_capacity']['value'] == 0
    assert dsp['requested_budget']['known'] is True
    assert full.nodes[target._resource('DSP')]['requested_budget']['known'] is False
    assert program_payload(full) == program_payload(restricted) == v2.graph_payload(program)
    assert registry == before
    with pytest.raises(ValueError):
        target.enrich(program, 'zcu104', 10, {'DSP': 801}, platform_available={'DSP': 800})


def test_three_devices_and_new_clock_preserve_all_program_facts(snapshot):
    program, auth = snapshot()
    before = v2.graph_payload(program)
    graphs = [target.enrich(program, d, c, authentication=auth)
              for d, c in [('zcu104', 10), ('u200', 10), ('u50', 10), ('zcu104', 6.67)]]
    assert len({g.graph['target_context_identity'] for g in graphs}) == 4
    assert len({g.graph['program_payload_sha256'] for g in graphs}) == 1
    assert all(program_payload(g) == before for g in graphs)
    assert v2.graph_payload(program) == before
    assert all(g.graph['action_slots'] == graphs[0].graph['action_slots'] for g in graphs)
    for graph in graphs:
        target.validate(graph)


def test_compact_semantics_and_exact_restoration(snapshot):
    original, auth = snapshot()
    compact, cauth = snapshot(view='compact')
    first = target.enrich(original, 'u50', 6.67, authentication=auth)
    second = target.enrich(compact, 'u50', 6.67, authentication=cauth)
    assert target.overlay_signature(first) == target.overlay_signature(second)
    assert program_payload(first) == v2.graph_payload(original)
    assert program_payload(second) == v2.graph_payload(compact)
    for label, a in first.graph['action_annotations'].items():
        assert get_action_point_relations(original, label, 'all') == get_action_point_relations(compact, label, 'all')
        if original.nodes['action:' + label]['action_kind'] == 'loop':
            assert a['loop_semantics']['sha256'] == target.fingerprint(get_loop_semantics(original, label))


def test_deterministic_serialization_and_roundtrip(snapshot, tmp_path):
    original, auth = snapshot()
    graph = target.enrich(original, 'u50', 6.67, {'DSP': 1024}, authentication=auth)
    other = target.enrich(original, 'u50', 6.67, {'DSP': 1024}, authentication=auth)
    assert target.serialized(graph) == target.serialized(other)
    out = tmp_path / 'joint.json'
    target.write_graph(graph, out)
    assert target.serialized(target.read_graph(out)) == out.read_text()
    with pytest.raises(ValueError, match='overwrite'):
        target.write_graph(target.enrich(original, 'u200', 10), out)
    with pytest.raises(ValueError, match='V2 schema'):
        v2.validate(graph)


def test_all_relationships_have_separate_evidence_and_stable_endpoints(snapshot):
    original, _ = snapshot()
    graph = target.enrich(original, 'zcu104', 10)
    assert set(graph) - set(original) == {target.HARDWARE, *(target._resource(r) for r in target.RESOURCES)}
    for u, v, _, e in overlay_edges(graph):
        assert e['hardware_evidence']['status'] == 'documented'
        assert e['hardware_evidence']['sources']
        assert e['compatibility']['status'] in ('possible', 'not_applicable')
        assert e['compatibility']['actual_binding'] == 'unknown'
        assert e['compatibility']['interpretation']
        assert not any(original.nodes[n]['category'] == 'constant_or_feature' for n in (u, v) if n in original)
        assert all(n in original for n in e['compiler_evidence']['nodes'])
    assert all(original.nodes[u]['root_kind'] == 'allocation'
               for u, _, _, _ in overlay_edges(graph, 'target:storage_compatible'))


@pytest.mark.parametrize('mutation', ['program_node', 'program_edge', 'program_metadata', 'proven_mapping',
                                     'witness', 'capacity', 'context', 'implementation', 'authentication'])
def test_corruption_is_rejected(snapshot, mutation):
    program, auth = snapshot()
    graph = target.enrich(program, 'zcu104', 10, authentication=auth)
    if mutation == 'program_node':
        graph.nodes['action:L5']['absolute_index'] = 1
    elif mutation == 'program_edge':
        u, v, k = next(iter(program.edges(keys=True)))
        graph[u][v][k]['evidence_status'] = 'not_established'
    elif mutation == 'program_metadata':
        graph.graph['program_metadata']['provenance']['source_sha256'] = '0' * 64
    elif mutation == 'proven_mapping':
        overlay_edges(graph, 'target:arithmetic_compatible')[0][3]['compatibility']['status'] = 'proven'
    elif mutation == 'witness':
        overlay_edges(graph, 'target:arithmetic_compatible')[0][3]['compiler_evidence']['edges'] = ['invented']
    elif mutation == 'capacity':
        graph.nodes[target._resource('DSP')]['physical_capacity']['value'] = 1
    elif mutation == 'context':
        graph.graph['target_context_identity'] = '0' * 64
    elif mutation == 'implementation':
        graph.graph['implementation']['version'] = 'unknown'
    else:
        graph.graph['source_authentication']['file_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        target.validate(graph)


@pytest.mark.parametrize('prefix,unresolved', [
    ('serrano-kalman-filter-96', ['L7', 'L8']),
    ('rodinia-hotspot-2-', ['L1', 'L2', 'L3', 'L4', 'L6']),
    ('rodinia-hotspot-3-', ['L1', 'L2', 'L3', 'L4', 'L6']),
])
def test_unresolved_actions_never_gain_hardware_targets(snapshot, prefix, unresolved):
    program, auth = snapshot(prefix)
    graph = target.enrich(program, 'u50', 6.67, authentication=auth)
    for label in unresolved:
        assert 'action:' + label not in graph
        index = int(label[1:]) - 1
        assert graph.graph['action_slots']['registered_mask'][index]
        assert graph.graph['action_slots']['unresolved_mask'][index]
        assert not graph.graph['action_slots']['matched_mask'][index]
    assert graph.graph['program_metadata']['action_point_coverage'] == program.graph['action_point_coverage']
    assert program_payload(graph) == v2.graph_payload(program)


def test_lud_invocation_receipts_are_retained(snapshot):
    program, auth = snapshot('rodinia_lud_1_')
    graph = target.enrich(program, 'u200', 10, authentication=auth)
    loop = graph.graph['action_annotations']['L2']['compiler_target']
    relevant = [e for u, _, _, e in overlay_edges(graph, 'target:loop_resource_relevance') if u == loop]
    witnesses = [w for e in relevant for w in e['compiler_evidence']['witnesses'] if w['kind'] == 'memory_access']
    assert any(w.get('invocation_context') == ['op:lud_diagonal:op5'] and w.get('callsite_bindings') for w in witnesses)
    assert not [r for r in get_action_point_relations(program, 'L2', call_context=()) if r['neighbor'] == 'L5']


def test_analysis_stub_types_are_never_hardware_exact(snapshot):
    program, _ = snapshot('serrano-kalman-filter-96')
    graph = target.enrich(program, 'zcu104', 10)
    edges = overlay_edges(graph, 'target:arithmetic_compatible')
    assert edges
    for _, _, _, e in edges:
        f = e['compiler_evidence']['numerical_fidelity']
        assert f['hardware_exact'] is not True
        assert f['source_correspondence'] == 'not_established'
        assert 'dsp_count' not in e and 'achieved_latency' not in e
        assert e['compatibility']['status'] == 'possible'


@pytest.mark.parametrize('name,types,expected', [
    ('arith.muli', ['i32', 'i32'], True), ('arith.mulf', ['f32', 'f32'], True),
    ('math.exp', ['f32', 'f32'], False), ('arith.muli', ['index', 'index'], False),
    ('arith.mulf', ['vector<4xf32>', 'vector<4xf32>'], False),
    ('arith.muli', ['i16', 'i32'], False),
])
def test_scalar_operator_eligibility_is_conservative(name, types, expected):
    assert target._arithmetic_eligible({'text': name, 'operand_types': types, 'result_types': [types[-1]]}) == expected


def hand_program(*, ambiguous=False, unresolved=False, alias_kind='may_alias'):
    """Consumer unit input, never claimed to be translated C++ or native evidence."""
    graph = nx.MultiDiGraph(schema=v2.SCHEMA_V22, semantic_revision='rc3', view='canonical',
                            graph_identity='hand-authored-not-compiler-output',
                            provenance={'origin': 'hand_authored_not_compiler_output'},
                            compiler_analysis={'aliases': [], 'dependences': []},
                            numerical_fidelity={'hardware_exact': False})
    loop = 'op:F:op0'
    graph.add_node('op:F:function', category='operation', function=0, text='func.func', op_uid='F:function')
    graph.add_node(loop, category='operation', function=0, text='scf.for', is_loop=1, op_uid='F:op0',
                   parent_operation='op:F:function', loop_context=[], trip_count=8, trip_count_static=1)
    graph.add_node('action:L9', category='action_point', action_id='L9', absolute_index=9, function=0,
                   action_kind='loop', supported_directives=['PIPELINE', 'UNROLL'], metadata_value=8,
                   source_loop_metadata_value=8, compiler_proven_trip_count=8, tcad_trip_count=8)
    v2.add_relation(graph, 'action:L9', loop, 'targets')
    for i, name in enumerate(('A', 'B')):
        operation, value, memory = f'op:F:op{i + 1}', 'value:F:' + name, 'memory:' + name
        graph.add_node(operation, category='operation', function=0, text='memref.alloca', op_uid=f'F:op{i + 1}',
                       parent_operation='op:F:function', loop_context=[])
        graph.add_node(value, category='ssa_value', function=0, ssa_id='F:' + name, is_memory=1, value_type='memref<8xi32>')
        graph.add_node(memory, category='memory_object', function=0, root_kind='allocation', root_value=value,
                       root_values=[value], memory_shape=[8], value_type='memref<8xi32>')
        v2.add_relation(graph, operation, value, 'defines', legacy_evidence={'position': 0})
        v2.add_relation(graph, value, memory, 'resolves_to')
    native_alias = {'lhs': 'F:A', 'rhs': 'F:B', 'result': alias_kind}
    graph.graph['compiler_analysis']['aliases'].append(native_alias)
    v2.add_relation(graph, 'value:F:A', 'value:F:B', 'alias',
                    'proven' if alias_kind == 'no_alias' else 'possible', alias_kind=alias_kind,
                    source_analysis=native_alias, symmetric=True)
    operand = 'value:F:A'
    roots = [('memory:A', 'proven')]
    if ambiguous or unresolved:
        operand = 'value:F:P'
        graph.add_node(operand, category='ssa_value', function=0, ssa_id='F:P', is_memory=1,
                       value_type='memref<8xi32>', region_root_evidence={'all_alternatives_resolved': False})
        if unresolved:
            graph.add_node('memory:unknown', category='memory_object', function=0, root_kind='unresolved', root_value=operand)
            roots = [('memory:unknown', 'not_established')]
        else:
            roots = [('memory:A', 'possible'), ('memory:B', 'possible')]
        for memory, certainty in roots:
            v2.add_relation(graph, operand, memory, 'resolves_to', certainty)
    access = 'op:F:op4'
    graph.add_node(access, category='operation', function=0, text='memref.load', op_uid='F:op4',
                   parent_operation=loop, loop_context=[loop], mlir_attrs={})
    v2.add_relation(graph, operand, access, 'operand', legacy_evidence={'position': 0})
    effect = {'function': 'F', 'ordinal': 4,
              'effects': [{'kind': 'read', 'value': graph.nodes[operand]['ssa_id'], 'operand_index': 0}]}
    for memory, certainty in roots:
        v2.add_relation(graph, access, memory, 'memory_access', certainty, concrete=True,
                        memory_operand=operand, operand_index=0, access='read', origin_loop=loop,
                        access_indices={'ssa_operands': [], 'induction_loops': [], 'affine_map': None}, source_analysis=effect)
    dep = {'result': 'unknown', 'kind': 'RAW', 'source': {'function': 'F', 'ordinal': 4},
           'target': {'function': 'F', 'ordinal': 4}, 'reason': 'hand-authored uncertain-dependence unit'}
    graph.graph['compiler_analysis']['dependences'].append(dep)
    v2.add_relation(graph, access, access, 'depends_on', 'possible', source_analysis=dep,
                    dependence_kind='RAW', common_affine_loops=[], carrying_loop=None)
    graph.graph['action_point_coverage'] = {'records': [
        {'action_id': 'L9', 'action_point_node': 'action:L9', 'compiler_target': loop,
         'compiler_matched': True, 'metadata_registered': True, 'source_labeled': True,
         'metadata_value': 8, 'compiler_proven_trip_count': 8},
        {'action_id': 'L17', 'action_point_node': None, 'compiler_matched': False,
         'metadata_registered': True, 'source_labeled': True}]}
    v2.validate(graph)
    return graph


def test_possible_aliases_incomplete_roots_and_dependencies_stay_uncertain():
    program = hand_program(ambiguous=True)
    graph = target.enrich(program, 'u50', 6.67)
    target.validate(graph)
    bram = [e for _, v, _, e in overlay_edges(graph, 'target:loop_resource_relevance') if v == target._resource('BRAM_18K')]
    assert bram and all(e['compiler_evidence']['status'] == 'possible' for e in bram)
    assert all(w['root_candidates_complete'] is False for e in bram for w in e['compiler_evidence']['witnesses'])
    action = [e for _, v, _, e in overlay_edges(graph, 'target:action_resource_relevance') if v == target._resource('BRAM_18K')]
    assert action and all(e['compiler_evidence']['status'] == 'possible' for e in action)
    assert graph.graph['action_annotations']['L9']['loop_semantics']['dependence_counts'] == {'possible': 1}
    assert graph.graph['action_annotations']['L9']['absolute_slot'] == 8
    assert graph.graph['action_slots']['unresolved_mask'][16]
    assert program_payload(graph) == v2.graph_payload(program)


def test_unresolved_root_does_not_create_storage_or_loop_resource_claim():
    program = hand_program(unresolved=True)
    graph = target.enrich(program, 'zcu104', 10)
    assert not [e for u, _, _, e in overlay_edges(graph) if u == 'memory:unknown']
    assert not overlay_edges(graph, 'target:loop_resource_relevance')
    assert graph.graph['action_annotations']['L9']['loop_semantics']['access_evidence_counts'] == {'not_established': 1}


def test_negative_alias_evidence_does_not_create_positive_connectivity():
    program = hand_program(unresolved=True, alias_kind='no_alias')
    graph = target.enrich(program, 'zcu104', 10)
    assert not overlay_edges(graph, 'target:loop_resource_relevance')
    assert any(e.get('alias_kind') == 'no_alias' for _, _, e in target.strip_overlay(graph).edges(data=True))


def test_authentication_rejects_changed_original_and_archive(snapshot, tmp_path):
    program, auth = snapshot()
    auth['file_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='authentication'):
        target.enrich(program, 'zcu104', 10, authentication=auth)
    fake = tmp_path / 'fake.tar.gz'
    fake.write_bytes(b'not the release')
    with pytest.raises(ValueError, match='Archive hash'):
        with target.authenticated_release(fake):
            pass


def test_cli_compose_validate_and_inspect(release, tmp_path):
    path = next((release / 'compact').glob('spcl_example_03-*.json'))
    output = tmp_path / 'cli.json'
    compose = subprocess.run([sys.executable, '-B', '-m', 'GNN_branch.target_graph_v22', 'compose',
                              '--graph', str(path), '--target', 'u50', '--clock-period-ns', '6.67',
                              '--available-dsp', '1024', '--output', str(output)],
                             cwd=target.ROOT, capture_output=True, text=True)
    assert compose.returncode == 0, compose.stderr
    assert json.loads(compose.stdout)['added_nodes'] == 5
    for command in ('validate', 'inspect'):
        result = subprocess.run([sys.executable, '-B', '-m', 'GNN_branch.target_graph_v22', command,
                                 '--graph', str(output)], cwd=target.ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['schema'] == target.SCHEMA
