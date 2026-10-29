"""Semantic edge tensor tests: synthetic proof cases plus authenticated RC3 graphs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import networkx as nx
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('torch_geometric')

from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch import semantic_tensor_features_v22 as semantic
from GNN_branch import target_graph_tensor_v22 as tensors
from GNN_branch import target_graph_v22 as overlay


ROOT = Path(__file__).resolve().parents[1]
MEMBERS = (
    'canonical/machsuite-stencil2d-624ee88fef89.json',
    'canonical/machsuite-gemm-ncubed-8692fbcbe7b2.json',
    'canonical/rodinia_lud_1_tiling_0-e35f0c4cd5d4.json',
    'canonical/chstone-aes-883c223fddcf.json',
    'canonical/serrano-kalman-filter-96b4d608969f.json',
)


def _idx(name):
    return semantic.FEATURE_NAMES.index(name)


def _pair(data, name):
    at = _idx(name)
    return data[at:at + 2]


def _affine_fixture(map_text, *, bad_binding=False):
    graph = nx.MultiDiGraph()
    graph.add_node('op:kernel:op1', category='operation', operation_family='memory_read', loop_context=['op:kernel:op0'])
    graph.add_node('memory:kernel:root', category='memory_object', root_kind='formal_parameter')
    graph.add_node('op:kernel:op0', category='operation', operation_family='loop', loop_depth=0, action_id='L1')
    input_names = re_inputs(map_text)
    for i, name in enumerate(input_names):
        value = f'value:kernel:b0:a{i}'
        position = i + 1
        graph.add_node(value, category='ssa_value')
        graph.add_edge(value, 'op:kernel:op1', key=f'op{i}', relation_type='operand',
                       legacy_evidence={'position': position + int(bad_binding and i == 0), 'role': 'operand'})
    graph.add_edge('op:kernel:op1', 'memory:kernel:root', key='ma', relation_type='memory_access',
                   access='read', evidence_status='proven',
                   access_indices={'status': 'compiler_recorded', 'affine_map': map_text,
                                   'ssa_operands': [f'value:kernel:b0:a{i}' for i in range(len(input_names))],
                                   'operand_positions': [i + 1 for i in range(len(input_names))],
                                   'induction_loops': ['op:kernel:op0']})
    return graph


def re_inputs(text):
    parsed = semantic.parse_linear_affine_map(text)
    if parsed is None:
        # The malformed/unsupported fixture still has one documented input.
        return ['d0']
    return parsed['inputs']


def _one_edge(graph, relation):
    return next((u, v, e) for u, v, e in graph.edges(data=True) if e.get('relation_type') == relation)


def test_affine_accesses_bind_coefficients_to_exact_ssa_operands():
    maps = {
        'A[i]': 'affine_map<(d0) -> (d0)>',
        'A[i+1]': 'affine_map<(d0) -> (d0 + 1)>',
        'A[2*i]': 'affine_map<(d0) -> (2*d0)>',
        'A[i][j] flattened': 'affine_map<(d0, d1) -> (d0 + 16*d1)>',
    }
    got = {}
    for label, text in maps.items():
        graph = _affine_fixture(text)
        bindings = semantic._index_bindings(graph)
        edge = _one_edge(graph, 'memory_access')
        row = semantic.edge_features(graph, *edge[:2], edge[2], bindings)
        got[label] = row
    assert _pair(got['A[i]'], 'affine_offset_0_value') != _pair(got['A[i+1]'], 'affine_offset_0_value')
    g = _affine_fixture(maps['A[i][j] flattened'])
    binds = semantic._index_bindings(g)
    first = semantic.edge_features(g, 'value:kernel:b0:a0', 'op:kernel:op1',
                                   g['value:kernel:b0:a0']['op:kernel:op1']['op0'], binds)
    second = semantic.edge_features(g, 'value:kernel:b0:a1', 'op:kernel:op1',
                                    g['value:kernel:b0:a1']['op:kernel:op1']['op1'], binds)
    assert first[_idx('index_role')] == second[_idx('index_role')] == 1
    assert first[_idx('index_coeff_result_0_value')] != second[_idx('index_coeff_result_0_value')]
    assert _pair(second, 'index_input_ordinal_value') == semantic._pair(1)
    consecutive = _affine_fixture('affine_map<(d0) -> (d0)>')
    stride_two = _affine_fixture('affine_map<(d0) -> (2*d0)>')
    def operand_coeff(g):
        u, v, e = _one_edge(g, 'operand')
        return semantic.edge_features(g, u, v, e, semantic._index_bindings(g))[_idx('index_coeff_result_0_value')]
    assert operand_coeff(consecutive) != operand_coeff(stride_two)
    def action_summary(g):
        g.add_node('action:L1',category='action_point',action_id='L1')
        bindings, maps = semantic._index_context(g)
        rows, records = [], []
        for u,v,key,attrs in g.edges(keys=True,data=True):
            if attrs.get('relation_type') == 'no_alias':
                continue
            rows.append(semantic.edge_features(g,u,v,attrs,bindings,maps))
            records.append((u,v,str(key),attrs))
        return semantic.action_edge_summary(g,'action:L1','memory:kernel:root',rows,records)[0]
    assert action_summary(consecutive) != action_summary(stride_two)


def test_affine_unknown_or_wrong_ssa_binding_is_masked_not_zero_stride():
    unsupported = _affine_fixture('affine_map<(d0) -> (d0 floordiv 4)>')
    edge = _one_edge(unsupported, 'memory_access')
    row = semantic.edge_features(unsupported, *edge[:2], edge[2], semantic._index_bindings(unsupported))
    assert row[_idx('affine_map_supported')] == 0
    mismatch = _affine_fixture('affine_map<(d0) -> (d0)>', bad_binding=True)
    edge = _one_edge(mismatch, 'memory_access')
    row = semantic.edge_features(mismatch, *edge[:2], edge[2], semantic._index_bindings(mismatch))
    assert row[_idx('affine_map_supported')] == 0
    assert row[_idx('affine_map_overflow')] == 0  # malformed binding is unknown, not truncation

    over_cap = _affine_fixture('affine_map<(d0, d1, d2, d3, d4) -> (d0+d1+d2+d3+d4)>')
    edge = _one_edge(over_cap, 'memory_access')
    row = semantic.edge_features(over_cap, *edge[:2], edge[2], semantic._index_bindings(over_cap))
    assert row[_idx('affine_map_known')] == 1
    assert row[_idx('affine_map_supported')] == 0
    assert row[_idx('affine_map_overflow')] == 1

    # Multiple compiler witnesses for one operation must agree before any
    # operand edge receives a coefficient. Never select the first witness.
    conflict = _affine_fixture('affine_map<(d0) -> (d0)>')
    conflict.add_edge('op:kernel:op1', 'memory:kernel:root', key='conflicting-map',
                      relation_type='memory_access', access='read', evidence_status='proven',
                      access_indices={'status': 'compiler_recorded',
                                      'affine_map': 'affine_map<(d0) -> (2*d0)>',
                                      'ssa_operands': ['value:kernel:b0:a0'],
                                      'operand_positions': [1],
                                      'induction_loops': ['op:kernel:op0']})
    bindings, maps = semantic._index_context(conflict)
    assert not bindings
    assert maps['op:kernel:op1']['conflict']
    operand = _one_edge(conflict, 'operand')
    edge_features = semantic.edge_features(conflict, *operand[:2], operand[2], bindings, maps)
    assert edge_features[_idx('index_map_supported')] == 0


def test_dependence_interval_exactness_carrying_scope_and_forwarding_roles():
    graph = nx.MultiDiGraph()
    graph.add_node('op:k:loop', category='operation', loop_depth=2, action_id='L5')
    graph.add_node('op:k:a', category='operation')
    graph.add_node('op:k:b', category='operation')
    dep = {'relation_type': 'depends_on', 'dependence_kind': 'RAW', 'evidence_status': 'possible',
           'loop_carried': True, 'carrying_loop': 'op:k:loop', 'carrying_loop_status': 'proven',
           'dependence_depth': 1, 'distance_bounds': [{'loop': 'op:k:loop', 'lower': -2, 'upper': 4}],
           'distance_components': [{'loop': 'op:k:loop', 'lower': -2, 'upper': 4}], 'distance_exact': [None]}
    row = semantic.edge_features(graph, 'op:k:a', 'op:k:b', dep)
    assert row[_idx('dependence_bounds_known')] == 1
    assert _pair(row, 'dependence_distance_lower_value')[1] == 1
    assert _pair(row, 'dependence_distance_upper_value')[1] == 1
    assert _pair(row, 'dependence_exact_value')[1] == 0
    assert row[_idx('carrying_loop_is_action')] == 1
    dep['distance_bounds'] = [{'loop': 'op:k:loop', 'lower': 3, 'upper': 3}]
    dep['distance_components'] = [{'loop': 'op:k:loop', 'lower': 3, 'upper': 3}]
    dep['distance_exact'] = [3]
    exact = semantic.edge_features(graph, 'op:k:a', 'op:k:b', dep)
    assert _pair(exact, 'dependence_exact_value')[1] == 1
    dep['distance_bounds'] = []
    dep['distance_components'] = []
    dep['distance_exact'] = []
    missing = semantic.edge_features(graph, 'op:k:a', 'op:k:b', dep)
    assert _pair(missing, 'dependence_distance_lower_value')[1] == 0
    assert _pair(missing, 'dependence_exact_value')[1] == 0
    dep['distance_bounds'] = [{'loop': 'op:k:loop', 'lower': 'unknown', 'upper': 3}]
    invalid = semantic.edge_features(graph, 'op:k:a', 'op:k:b', dep)
    assert invalid[_idx('dependence_bounds_known')] == 0
    for kind in ('RAW','WAR','WAW'):
        attrs = {'relation_type':'depends_on','dependence_kind':kind,'evidence_status':'proven'}
        base = tensors._edge_features(attrs)
        at = len(tensors.EVIDENCE)+tensors.EDGE_NUMERICAL.index(kind)
        assert base[at] == 1
    for role in ('iter_init', 'loop_backedge', 'loop_result', 'future_role'):
        edge = {'relation_type': 'loop_carried', 'evidence_status': 'proven', 'forwarding_role': role,
                'state_kind': 'scalar', 'evidence_origin': 'verified_MLIR_region_SSA'}
        encoded = semantic.edge_features(graph, 'op:k:a', 'op:k:b', edge)
        category = role if role in semantic.FORWARDING_ROLES[:-1] else 'unknown'
        assert encoded[_idx('forwarding_role_' + category)] == 1


def test_views_retain_operation_offset_semantics_dimensions_and_dynamic_masks():
    graph = nx.MultiDiGraph()
    base = {'relation_type': 'references', 'view_mapping': {
        'operation_kind': 'memref.subview', 'offset_semantics': 'relative_to_source_view',
        'offsets': [2, -9223372036854775808], 'sizes': [8, 4], 'strides': [2, 1],
        'source_shape': [32, 8], 'result_shape': [8], 'element_overlap': 'not_established'}}
    subview = semantic.edge_features(graph, 'value:a', 'value:b', base)
    cast = deepcopy(base); cast['view_mapping']['operation_kind'] = 'memref.reinterpret_cast'
    cast['view_mapping']['offset_semantics'] = 'relative_to_underlying_storage'
    reinterpret = semantic.edge_features(graph, 'value:a', 'value:b', cast)
    assert subview[_idx('view_kind_memref.subview')] == 1
    assert reinterpret[_idx('view_kind_memref.reinterpret_cast')] == 1
    assert subview[_idx('view_offset_semantics_relative_to_source_view')] == 1
    assert reinterpret[_idx('view_offset_semantics_relative_to_underlying_storage')] == 1
    assert _pair(subview, 'view_offset_0_value')[1] == 1
    assert _pair(subview, 'view_offset_1_value')[1] == 0
    assert _pair(subview, 'view_size_1_value')[1] == 1
    base['view_mapping']['offsets'] = [0, 0, 0, 0, 0]
    assert semantic.edge_features(graph, 'value:a', 'value:b', base)[_idx('view_dimension_overflow')] == 1


def test_calls_encode_argument_position_and_resolution_without_identifiers():
    graph = nx.MultiDiGraph()
    graph.add_node('op:k:call', category='operation', call_resolution='possible')
    attrs = {'relation_type': 'actual_to_formal', 'call_operation': 'op:k:call',
             'argument_index': 2, 'evidence_status': 'possible'}
    row = semantic.edge_features(graph, 'value:k:actual', 'value:helper:formal', attrs)
    assert row[_idx('call_resolution_possible')] == 1
    base = tensors._edge_features(attrs)
    at = len(tensors.EVIDENCE)+tensors.EDGE_NUMERICAL.index('argument_position_known')
    assert base[at] == 1
    assert all('op:k:call' not in name for name in semantic.FEATURE_NAMES)
    graph.add_node('op:k:call2',category='operation',call_resolution='proven')
    graph.add_node('value:caller:a1',category='ssa_value')
    graph.add_node('value:caller:a2',category='ssa_value')
    graph.add_node('value:helper:formal',category='ssa_value')
    graph.add_edge('value:caller:a1','value:helper:formal',relation_type='actual_to_formal',
                   call_operation='op:k:call',argument_index=1,evidence_status='proven')
    graph.add_edge('value:caller:a2','value:helper:formal',relation_type='actual_to_formal',
                   call_operation='op:k:call2',argument_index=1,evidence_status='proven')
    assert graph.number_of_edges('value:caller:a1','value:helper:formal') == 1
    assert graph.number_of_edges('value:caller:a2','value:helper:formal') == 1
    assert graph.has_node('value:caller:a1') and graph.has_node('value:caller:a2')


def test_evidence_certainty_and_negative_alias_stay_distinct():
    graph = nx.MultiDiGraph()
    for status in ('proven','possible','not_established'):
        attrs = {'relation_type':'resolves_to','evidence_status':status}
        row = tensors._edge_features(attrs) + semantic.edge_features(graph,'value:a','memory:a',attrs)
        at = tensors.EVIDENCE.index(status)
        assert row[at] == 1
    no_alias = {'relation_type':'alias','alias_kind':'no_alias','evidence_status':'proven'}
    row = tensors._edge_features(no_alias) + semantic.edge_features(graph,'memory:a','memory:b',no_alias)
    assert row[len(tensors.EVIDENCE)+tensors.EDGE_NUMERICAL.index('no_alias')] == 1
    assert not any(row[tensors.BASE_EDGE_DIM:])


def _authenticated_graphs(tmp_path):
    manifest = json.loads(overlay.RELEASE.read_text())
    archive = ROOT / manifest['archive']
    assert overlay.file_hash(archive) == manifest['archive_sha256']
    out = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in MEMBERS:
            raw = tar.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest() == manifest['files_sha256'][member]
            path = tmp_path / Path(member).name
            path.write_bytes(raw)
            out[member] = compiler.read_graph(path)
    return out


def test_authenticated_coverage_target_independence_compaction_and_kalman_masks(tmp_path):
    graphs = _authenticated_graphs(tmp_path)
    key = MEMBERS[0]
    original = graphs[key]
    before = compiler.canonical_json(compiler.graph_payload(original))
    zcu = tensors.tensorize(overlay.enrich(original, 'zcu104', 10))
    repeated = tensors.tensorize(overlay.enrich(original, 'zcu104', 10))
    assert torch.equal(zcu.edge_attr, repeated.edge_attr)
    assert torch.equal(zcu.action_semantic, repeated.action_semantic)
    assert zcu.audit['schema'] == 'salt-hls-target-tensors-v3-semantic-experimental'
    assert zcu.audit['previous_tensor_schema'] == tensors.LEGACY_SCHEMA
    assert compiler.canonical_json(compiler.graph_payload(original)) == before
    for profile, clock in [('u200', 10), ('u50', 10), ('zcu104', 6.67)]:
        other = tensors.tensorize(overlay.enrich(original, profile, clock))
        assert torch.equal(zcu.action_semantic, other.action_semantic)
        assert torch.equal(zcu.action_demand, other.action_demand)
    compact_path = tmp_path / 'compact.json'
    manifest = json.loads(overlay.RELEASE.read_text())
    member = MEMBERS[0].replace('canonical/', 'compact/')
    with tarfile.open(ROOT / manifest['archive'], 'r:gz') as tar:
        raw = tar.extractfile(member).read()
    assert hashlib.sha256(raw).hexdigest() == manifest['files_sha256'][member]
    compact_path.write_bytes(raw)
    compact = tensors.tensorize(overlay.enrich(compiler.read_graph(compact_path), 'zcu104', 10))
    for name, value in zcu:
        if torch.is_tensor(value):
            assert torch.equal(value, compact[name]), name
    kalman = tensors.tensorize(overlay.enrich(graphs[MEMBERS[-1]], 'zcu104', 10))
    assert kalman.unresolved_mask[0, 6:8].all()
    assert not kalman.action_semantic_mask[0, 6:8].any()
    assert kalman.audit['semantic_edge_features']['feature_names'] == list(semantic.FEATURE_NAMES)


def test_gnn_consumes_semantic_edges_and_action_summary_with_gradient(tmp_path):
    graphs = _authenticated_graphs(tmp_path)
    data = tensors.tensorize(overlay.enrich(graphs[MEMBERS[0]], 'zcu104', 10))
    assert data.edge_attr.shape[-1] == tensors.BASE_EDGE_DIM + len(semantic.FEATURE_NAMES)
    assert data.action_semantic.shape == (1, 64, len(semantic.FEATURE_NAMES))
    assert torch.isfinite(data.edge_attr).all() and torch.isfinite(data.action_semantic).all()
    torch.manual_seed(211)
    model = tensors.SmallGraphEncoder().eval()
    edge_attr = data.edge_attr.detach().clone().requires_grad_(True)
    data.edge_attr = edge_attr
    output = model(data)
    action_rows = data.action_node_index
    loss = output[action_rows].square().sum()
    loss.backward()
    sem_grad = edge_attr.grad[:, tensors.BASE_EDGE_DIM:]
    assert torch.isfinite(sem_grad).all() and sem_grad.abs().sum() > 0
    action_grad = model.action_semantic_input.weight.grad
    assert action_grad is not None and torch.isfinite(action_grad).all() and action_grad.abs().sum() > 0
    memory, _ = model.memories(data)
    assert memory.shape == (1, 64, 128)
    assert torch.isfinite(memory).all()
    # A same-weight, one-feature perturbation proves semantic channels can alter
    # an action-aligned embedding; it is not a QoR or accuracy experiment.
    changed = data.clone()
    changed.edge_attr = data.edge_attr.detach().clone()
    row = changed.action_semantic.clone()
    slot = int(changed.action_semantic_mask[0].nonzero()[0])
    row[0, slot, _idx('affine_map_supported')] += 1
    changed.action_semantic = row
    with torch.no_grad():
        baseline_memory, _ = model.memories(data)
        perturbed_memory, _ = model.memories(changed)
    assert (baseline_memory[0, slot] - perturbed_memory[0, slot]).abs().sum() > 0


def test_representative_kernels_all_support_finite_gnn_forward_backward(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(307)
    model = tensors.SmallGraphEncoder().eval()
    graphs = _authenticated_graphs(tmp_path)
    for member in MEMBERS:
        data = tensors.tensorize(overlay.enrich(graphs[member], 'zcu104', 10))
        model.zero_grad(set_to_none=True)
        output = model(data)
        memory, _ = model.memories(data, output)
        assert memory.shape == (1,64,128)
        assert torch.isfinite(output).all() and torch.isfinite(memory).all()
        output[data.action_node_index].square().sum().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), member
        assert model.edge_input.weight.grad is not None, member
        if data.action_semantic_mask.any():
            action_grad = model.action_semantic_input.weight.grad
            assert action_grad is not None and action_grad.abs().sum() > 0, member


def test_semantic_payload_remains_static_only_and_negative_alias_is_separate(tmp_path):
    graph = _authenticated_graphs(tmp_path)[MEMBERS[0]]
    joint = overlay.enrich(graph, 'zcu104', 10)
    data = tensors.tensorize(joint)
    assert 'candidate' not in data.audit['semantic_edge_schema']
    assert not hasattr(data, 'candidate_demand')
    no_alias = [e for _, _, e in joint.edges(data=True) if e.get('alias_kind') == 'no_alias']
    if no_alias:
        assert data.negative_edge_index.shape[1] >= len(no_alias)
    assert data.action_demand.shape[1] == data.action_semantic.shape[1] == 64
