"""CPU-only target overlay for immutable V2.2 RC3 graphs; no neural encoding.

The program graph is retained verbatim. Compatibility is always possible,
never a synthesis binding. Public queries come from the current checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time

import networkx as nx

from GNN_branch import action_point_relations as queries
from GNN_branch import mlir_graph_v2 as program_graph

ROOT = Path(__file__).resolve().parents[1]
PROFILES = Path(__file__).with_name('targets') / 'fpga_profiles.json'
RELEASE = ROOT / 'docs/graph-release-v22-rc3.json'
SCHEMA = 'salt-hls-target-enriched-v22-v1'
PROFILE_SCHEMA = 'salt-hls-fpga-profiles-v1'
IMPLEMENTATION_VERSION = '1.0.0'
RESOURCES = ('BRAM_18K', 'DSP', 'LUT', 'FF')
UNITS = dict(zip(RESOURCES, ('18-Kibit RAM equivalent', 'DSP slice', 'LUT', 'flip-flop')))
HARDWARE = 'hardware:fpga'
RELATIONS = {
    'hardware:belongs_to', 'target:arithmetic_compatible',
    'target:storage_compatible', 'target:loop_resource_relevance',
    'target:action_resource_relevance', 'target:requested_clock',
}
ARITHMETIC = {'arith.addi', 'arith.subi', 'arith.muli',
              'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf'}


def canonical(value):
    return program_graph.canonical_json(value)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, name, maximum=None):
    _require(type(value) is int and value >= 0, f'{name}: finite nonnegative integer required')
    _require(maximum is None or value <= maximum, f'{name}: exceeds available capacity')
    return value


def validate_registry(registry):
    """Validate units, physical meanings, source references and missingness."""
    _require(registry.get('schema') == PROFILE_SCHEMA, 'Unsupported target-profile schema')
    _require(set(registry) == {'schema', 'registry_version', 'sources', 'capabilities', 'profiles'},
             'Unknown registry fields; labels and candidate configurations are not profile features')
    _require(bool(registry.get('registry_version')), 'Missing registry version')
    sources = registry.get('sources', {})
    _require(bool(sources), 'Missing hardware sources')
    for name, source in sources.items():
        _require(set(source) <= {'title', 'scope', 'url', 'uri', 'selector', 'excerpt',
                                 'excerpt_sha256', 'tool_version'}, f'{name}: unknown provenance fields')
        _require(source.get('title') and source.get('scope'), f'{name}: incomplete source')
        _require(source.get('url') or source.get('uri'), f'{name}: missing source location')
        if 'excerpt' in source:
            _require(set(source['excerpt']) == {'part', 'AvailableResources'} and
                     set(source['excerpt']['AvailableResources']) == set(RESOURCES),
                     'Only device capacities may enter vendor excerpts')
            _require(fingerprint(source['excerpt']) == source.get('excerpt_sha256'),
                     f'{name}: capacity excerpt hash mismatch')
    _require(set(registry.get('capabilities', {})) == set(RESOURCES), 'Four resource classes required')
    for name, cap in registry['capabilities'].items():
        _require(set(cap) <= {'primitive', 'sources', 'properties', 'supported_modes'}, 'Unknown capability fields')
        _require(cap.get('primitive') and cap.get('sources'), f'{name}: missing capability evidence')
        _require(set(cap['sources']) <= sources.keys(), f'{name}: unknown capability source')
        for field, prop in cap.get('properties', {}).items():
            _require(set(prop) == {'value', 'unit', 'meaning', 'known', 'sources'}, 'Unknown numerical-property fields')
            value = prop.get('value')
            _require(prop.get('known') is (value is not None), f'{name}/{field}: missingness mismatch')
            _require(prop.get('unit') and prop.get('meaning') == 'physical_primitive_characteristic',
                     f'{name}/{field}: missing units/meaning')
            _require(prop.get('sources') and set(prop['sources']) <= sources.keys(),
                     f'{name}/{field}: missing sources')
            if value is not None:
                _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                         f'{name}/{field}: invalid capability value')
    _require(bool(registry.get('profiles')), 'No device profiles')
    for name, profile in registry['profiles'].items():
        _require(set(profile) == {'profile_version', 'part', 'family', 'identity_sources', 'resources',
                                  'platform_available_resources', 'platform_availability_status',
                                  'external_memory', 'unmodeled'}, 'Unknown profile fields; no QoR/candidate features allowed')
        _require(profile.get('profile_version') and profile.get('part') and profile.get('family'),
                 f'{name}: missing profile identity')
        _require(profile.get('identity_sources') and set(profile['identity_sources']) <= sources.keys(),
                 f'{name}: missing identity evidence')
        _require(set(profile.get('resources', {})) == set(RESOURCES), f'{name}: four capacities required')
        for resource, prop in profile['resources'].items():
            _require(set(prop) <= {'value', 'unit', 'meaning', 'known', 'sources', 'verification'},
                     'Unknown capacity-property fields')
            _require(_count(prop.get('value'), resource) > 0, 'Physical capacity must be positive')
            _require(prop.get('unit') == UNITS[resource] and prop.get('known') is True
                     and prop.get('meaning') == 'physical_device_capacity', 'Invalid capacity semantics')
            _require(prop.get('sources') and set(prop['sources']) <= sources.keys(), 'Missing capacity source')
            for source in (sources[s] for s in prop['sources']):
                if 'excerpt' in source:
                    _require(source['excerpt']['part'] == profile['part'] and
                             source['excerpt']['AvailableResources'][resource] == prop['value'],
                             'Profile disagrees with vendor capacity excerpt')
        _require(profile.get('platform_available_resources') is None,
                 'Registry profiles describe physical devices; supply platform allocations separately')
        external = profile['external_memory']
        _require(set(external) == {'architecture', 'sources', 'kernel_binding', 'kernel_binding_known',
                                  'bandwidth_bytes_per_second', 'bandwidth_known'}, 'Unknown external-memory fields')
        _require(external['kernel_binding'] is None and external['kernel_binding_known'] is False and
                 external['bandwidth_bytes_per_second'] is None and external['bandwidth_known'] is False,
                 'No benchmark memory binding or bandwidth model is established')
    canonical(registry)  # Reject non-JSON/nonfinite fields, including provenance.
    return registry


def load_registry(path=PROFILES):
    return validate_registry(json.loads(Path(path).read_text()))


def deployment_context(profile, clock_period_ns, budgets=None, platform_available=None):
    _require(type(clock_period_ns) in (int, float) and math.isfinite(clock_period_ns)
             and clock_period_ns > 0, 'Clock period must be finite and positive, in ns')
    physical = {r: profile['resources'][r]['value'] for r in RESOURCES}
    requested = dict(budgets or {})
    platform = dict(platform_available or {})
    _require(set(requested) <= set(RESOURCES) and set(platform) <= set(RESOURCES),
             'Unknown resource budget key')
    for r, v in platform.items():
        _count(v, r, physical[r])
    for r, v in requested.items():
        _count(v, r, platform.get(r, physical[r]))
    available = {r: requested.get(r, platform.get(r, physical[r])) for r in RESOURCES}
    return {
        'clock_period_ns': float(clock_period_ns), 'clock_unit': 'ns', 'clock_source': 'user_requested',
        'requested_resources': {r: requested.get(r) for r in RESOURCES},
        'platform_available_resources': {r: platform.get(r) for r in RESOURCES},
        'available_resources': available,
        'available_fractions': {r: available[r] / physical[r] for r in RESOURCES},
        'availability_basis': {r: 'user_budget' if r in requested else 'platform_available'
                               if r in platform else 'physical_full_capacity_convention' for r in RESOURCES},
        'resource_units': UNITS,
    }


def implementation_identity():
    paths = [Path(__file__), Path(queries.__file__), Path(program_graph.__file__),
             Path(__file__).with_name('mlir_semantics.py'),
             Path(__file__).with_name('mlir_semantics_v22.py')]
    return {'version': IMPLEMENTATION_VERSION,
            'source_sha256': {p.name: file_hash(p) for p in paths}}


def _edge(graph, source, target, relation, compiler, hardware_sources, *, details=None, mapping='possible'):
    """Three separate evidence axes: compiler facts, hardware facts, compatibility."""
    attrs = {
        '_target_overlay': True, 'relation_type': relation,
        'compiler_evidence': deepcopy(compiler),
        'hardware_evidence': {'status': 'documented', 'sources': sorted(set(hardware_sources))},
        'compatibility': {'status': mapping, 'actual_binding': 'unknown',
                          'interpretation': details or 'Possible implementation compatibility; no resource cost or binding inferred'},
    }
    key = 'target:' + fingerprint([source, target, attrs])[:24]
    graph.add_edge(source, target, key=key, **attrs)
    return key


def _compiler(nodes, edges=(), status='proven', **extra):
    return {'status': status, 'nodes': sorted(set(nodes)), 'edges': sorted(set(edges)), **extra}


def _resource(name):
    return 'hardware:resource:' + name


def _arithmetic_eligible(attrs):
    name = attrs.get('text')
    if name not in ARITHMETIC:
        return False
    types = attrs.get('operand_types', []) + attrs.get('result_types', [])
    if len(attrs.get('operand_types', [])) != 2 or len(attrs.get('result_types', [])) != 1:
        return False
    pattern = r'i[1-9][0-9]*' if name.endswith('i') else r'f(?:16|32|64)'
    return len(set(types)) == 1 and re.fullmatch(pattern, types[0]) is not None


def _storage_eligible(graph, node, attrs):
    """Only exact local, static, ordinary scalar memref allocations in v1."""
    if attrs.get('root_kind') != 'allocation':
        return False
    value = attrs.get('root_value')
    if value not in graph or not re.fullmatch(r'memref<(?:[1-9][0-9]*x)+(?:i[1-9][0-9]*|f16|f32|f64)>',
                                             attrs.get('value_type') or ''):
        return False
    definitions = [(u, k) for u, _, k, e in graph.in_edges(value, keys=True, data=True)
                   if e['relation_type'] == 'defines' and graph.nodes[u].get('text') in ('memref.alloc', 'memref.alloca')]
    return len(definitions) == 1 and any(v == node and e['relation_type'] == 'resolves_to'
                                       and e['evidence_status'] == 'proven'
                                       for _, v, e in graph.out_edges(value, data=True))


def _scoped_computation(graph, loop, eligible):
    """Traverse compiler ownership and proven call edges; perform no root analysis."""
    bodies = defaultdict(list)
    for n, a in graph.nodes(data=True):
        if a['category'] == 'operation' and a.get('op_uid'):
            bodies[a['function']].append(n)
    local = [n for n, a in graph.nodes(data=True) if loop in a.get('loop_context', [])]
    records = []

    def visit(nodes, chain, witnesses, visited):
        for node in sorted(nodes):
            if node in eligible:
                records.append({'operation': node, 'call_chain': list(chain),
                                'supporting_call_edges': list(witnesses),
                                'scope': 'invoked' if chain else 'compiler_loop_context'})
            for _, target, key, edge in graph.out_edges(node, keys=True, data=True):
                if edge['relation_type'] != 'calls' or edge['evidence_status'] != 'proven':
                    continue
                if graph.nodes[node].get('call_resolution') != 'proven':
                    continue
                function = graph.nodes[target]['function']
                if function in visited:
                    continue
                visit(bodies[function], chain + (node,), witnesses + (str(key),), visited | {function})

    visit(local, (), (), {graph.nodes[loop]['function']})
    return sorted(records, key=canonical)


def _access_witness(row, query_name):
    fields = ('source_operation', 'memory_root', 'supporting_edge', 'memory_operand',
              'evidence_status', 'access', 'access_path', 'call_chain', 'callsite_bindings',
              'root_candidates_complete', 'invocation_context', 'access_indices', 'memory_views')
    return {'query': query_name, 'query_record_sha256': fingerprint(row),
            **{k: deepcopy(row[k]) for k in fields if k in row}}


def _construct(program, profile_id, registry, deployment, authentication):
    profile = registry['profiles'][profile_id]
    graph = deepcopy(program)
    metadata = deepcopy(program.graph)
    identity = implementation_identity()
    context_id = fingerprint({'profile': profile, 'registry_sha256': fingerprint(registry),
                              'deployment': deployment})
    graph.graph.clear()
    graph.graph.update(schema=SCHEMA, program_metadata=metadata,
                       program_payload_sha256=fingerprint(program_graph.graph_payload(program)),
                       program_graph_identity=metadata['graph_identity'],
                       target_profile_id=profile_id, target_profile=deepcopy(profile),
                       profile_sha256=fingerprint(profile), registry_sha256=fingerprint(registry),
                       target_context_identity=context_id, deployment=deepcopy(deployment),
                       implementation=identity, source_authentication=deepcopy(authentication),
                       tensor_compatibility='future_encoder_required; no V1 or Stage-2 compatibility asserted')
    graph.add_node(HARDWARE, category='hardware_target', _target_overlay=True,
                   part=profile['part'], family=profile['family'], profile_version=profile['profile_version'],
                   identity_sources=profile['identity_sources'], clock_period_ns=deployment['clock_period_ns'],
                   clock_unit='ns', clock_meaning='user_requested_constraint',
                   external_memory=deepcopy(profile['external_memory']),
                   platform_availability_known=all(v is not None for v in deployment['platform_available_resources'].values()))
    for r in RESOURCES:
        node = _resource(r)
        graph.add_node(node, category='hardware_resource', _target_overlay=True, resource_class=r,
                       physical_capacity=deepcopy(profile['resources'][r]),
                       available_capacity={'value': deployment['available_resources'][r], 'unit': UNITS[r],
                                           'known': True, 'meaning': 'deployment_available_capacity',
                                           'source': deployment['availability_basis'][r]},
                       available_fraction={'value': deployment['available_fractions'][r], 'unit': 'fraction',
                                           'known': True, 'meaning': 'deployment_available_over_physical'},
                       platform_available={'value': deployment['platform_available_resources'][r],
                                           'unit': UNITS[r], 'known': deployment['platform_available_resources'][r] is not None,
                                           'meaning': 'platform_available_capacity'},
                       requested_budget={'value': deployment['requested_resources'][r], 'unit': UNITS[r],
                                         'known': deployment['requested_resources'][r] is not None,
                                         'meaning': 'user_requested_budget'},
                       capability=deepcopy(registry['capabilities'][r]))
        _edge(graph, node, HARDWARE, 'hardware:belongs_to', _compiler([], status='not_applicable'),
              profile['resources'][r]['sources'], mapping='not_applicable',
              details='Documented resource class belongs to this physical FPGA; no program binding')
    compatible = defaultdict(dict)
    arithmetic = set()
    excluded = Counter()
    fidelity = deepcopy(program.graph.get('numerical_fidelity', {}))
    for node, attrs in sorted(program.nodes(data=True)):
        if attrs['category'] == 'operation' and _arithmetic_eligible(attrs):
            arithmetic.add(node)
            refs = [(u, v, str(k)) for u, v, k, e in program.edges(node, keys=True, data=True)
                    if e['relation_type'] == 'defines']
            refs += [(u, v, str(k)) for u, v, k, e in program.in_edges(node, keys=True, data=True)
                     if e['relation_type'] == 'operand']
            compiler = _compiler([node] + [n for u, v, _ in refs for n in (u, v)],
                                 [k for _, _, k in refs], operation=attrs['text'],
                                 operand_types=attrs['operand_types'], result_types=attrs['result_types'],
                                 numerical_fidelity={'hardware_exact': fidelity.get('hardware_exact'),
                                                     'source_correspondence': 'not_established',
                                                     'interpretation': 'Native lowered types only; no width-to-DSP-count inference'})
            # UG1399 2021.1 supports fabric/DSP add/sub/mul; floating division fabric only.
            for r in ('LUT',) if attrs['text'] == 'arith.divf' else ('DSP', 'LUT'):
                compatible[node][r] = _edge(graph, node, _resource(r), 'target:arithmetic_compatible',
                                            compiler, ['hls_bind_op'] + registry['capabilities'][r]['sources'])
        elif attrs['category'] == 'operation' and attrs.get('operation_family') == 'arithmetic':
            excluded['unsupported_or_untyped_arithmetic'] += 1
        elif attrs['category'] == 'memory_object':
            if not _storage_eligible(program, node, attrs):
                excluded['storage_' + attrs.get('root_kind', 'unknown') + '_not_eligible'] += 1
                continue
            value = attrs['root_value']
            refs = [(u, v, str(k)) for u, v, k, e in program.in_edges(value, keys=True, data=True)
                    if e['relation_type'] == 'defines']
            refs += [(u, v, str(k)) for u, v, k, e in program.out_edges(value, keys=True, data=True)
                     if e['relation_type'] == 'resolves_to' and v == node]
            compiler = _compiler([node, value] + [u for u, _, _ in refs], [k for _, _, k in refs],
                                 storage_identity='exact_local_allocation', lowered_type=attrs['value_type'],
                                 hardware_exact=fidelity.get('hardware_exact'), source_correspondence='not_established')
            for r in ('BRAM_18K', 'LUT', 'FF'):
                compatible[node][r] = _edge(graph, node, _resource(r), 'target:storage_compatible', compiler,
                                            ['hls_partition'] + registry['capabilities'][r]['sources'],
                                            details='Possible local storage implementation; packing, banking, binding and element overlap unknown')
    annotations, cache = {}, {}
    for action, attrs in sorted(program.nodes(data=True)):
        if attrs['category'] != 'action_point':
            continue
        label = attrs['action_id']
        _, target, target_key, _ = next((u, v, k, e) for u, v, k, e in program.out_edges(action, keys=True, data=True)
                                       if e['relation_type'] == 'targets')
        rows = queries.get_action_point_relations(program, label, 'all', _access_cache=cache)
        annotations[label] = {'absolute_slot': int(label[1:]) - 1, 'compiler_target': target,
                              'supported_directives': deepcopy(attrs['supported_directives']),
                              'all_evidence_query_sha256': fingerprint(rows),
                              'query_counts': dict(Counter((r['relation_type'] + ':' + r['evidence_status'] +
                                                           (':' + r['alias_kind'] if 'alias_kind' in r else '')) for r in rows))}
        if attrs['action_kind'] == 'loop':
            semantics = queries.get_loop_semantics(program, label)
            annotations[label]['loop_semantics'] = {
                'sha256': fingerprint(semantics),
                'compiler_proven_trip_count': semantics['compiler_proven_trip_count'],
                'trip_count_provenance': semantics['trip_count_provenance'],
                'ssa_state_supporting_edges': [r['supporting_edge'] for r in semantics['ssa_state']],
                'dependence_counts': dict(Counter(r['evidence_status'] for r in semantics['memory_dependences'])),
                'dependence_supporting_edges': [r['supporting_edge'] for r in semantics['memory_dependences']],
                'access_evidence_counts': dict(Counter(r['evidence_status'] for r in semantics['accesses'])),
                'dependence_completeness': 'not_established',
                'achievable_ii': None,
            }
            relevant = defaultdict(list)
            for record in _scoped_computation(program, target, arithmetic):
                for r, key in compatible[record['operation']].items():
                    relevant[r].append({'kind': 'contained_or_invoked_computation', 'evidence_status': 'proven',
                                        'compatibility_edge': key, **record})
            accesses = [(r, 'get_loop_semantics.accesses') for r in semantics['accesses']]
            accesses += [(r, 'get_action_point_relations') for r in rows if r['relation_type'] == 'memory_access']
            for access, query_name in accesses:
                if access['evidence_status'] not in ('proven', 'possible') or not access.get('concrete'):
                    continue
                for r, key in compatible.get(access['memory_root'], {}).items():
                    relevant[r].append({'kind': 'memory_access', 'compatibility_edge': key,
                                        **_access_witness(access, query_name)})
            for r, witnesses in sorted(relevant.items()):
                witnesses = [json.loads(w) for w in sorted({canonical(w) for w in witnesses})]
                status = 'proven' if any(w['evidence_status'] == 'proven' for w in witnesses) else 'possible'
                edge = _edge(graph, target, _resource(r), 'target:loop_resource_relevance',
                             _compiler([target], status=status, witnesses=witnesses),
                             registry['capabilities'][r]['sources'],
                             details='Resource relevance through compiler-supported computation/access; no demand, II or throughput inferred')
                compatible[target][r] = edge
            _edge(graph, action, HARDWARE, 'target:requested_clock', _compiler([action, target], [target_key]),
                  ['hls_clock'], mapping='not_applicable',
                  details='Requested global clock constrains this loop action; feasibility and achievable II unknown')
        for r, support in sorted(compatible.get(target, {}).items()):
            support_status = graph.get_edge_data(target, _resource(r), support)['compiler_evidence']['status']
            _edge(graph, action, _resource(r), 'target:action_resource_relevance',
                  _compiler([action, target], [target_key], status=support_status, compatibility_support=support,
                            supported_directives=attrs['supported_directives']),
                  registry['capabilities'][r]['sources'],
                  details='Possible relevance for a registered action; directive values and synthesis binding absent')
    coverage = {r['action_id']: r for r in metadata.get('action_point_coverage', {}).get('records', [])}
    count = max((int(label[1:]) for label in set(coverage) | set(annotations)), default=0)
    labels = [f'L{k + 1}' for k in range(count)]
    slots = {'slot_rule': 'Lk -> k-1; no renumbering', 'labels': labels,
             'registered_mask': [bool(coverage.get(label, {}).get('metadata_registered')) for label in labels],
             'matched_mask': [label in annotations for label in labels],
             'unresolved_mask': [bool(coverage.get(label, {}).get('metadata_registered')) and label not in annotations
                                 for label in labels],
             'source_only_mask': [bool(coverage.get(label, {}).get('source_labeled'))
                                  and not coverage.get(label, {}).get('metadata_registered', False) for label in labels]}
    graph.graph.update(action_annotations=annotations, action_slots=slots,
                       excluded_relationships=dict(sorted(excluded.items())))
    graph.graph['overlay_identity'] = fingerprint({'program_payload_sha256': graph.graph['program_payload_sha256'],
                                                  'target_context_identity': context_id, 'implementation': identity})
    return graph


def enrich(program, target, clock_period_ns, budgets=None, *, platform_available=None,
           registry=None, authentication=None):
    registry = load_registry() if registry is None else validate_registry(registry)
    _require(target in registry['profiles'], f'Unknown FPGA profile: {target}')
    _require(program.graph.get('schema') == program_graph.SCHEMA_V22 and
             program.graph.get('semantic_revision') == 'rc3', 'V2.2 RC3 input required')
    program_graph.validate(program)
    if authentication is not None:
        release = json.loads(RELEASE.read_text())
        member = authentication.get('member')
        expected = release['files_sha256'].get(member)
        payload_bytes = (canonical(program_graph.graph_payload(program)) + '\n').encode()
        _require(authentication.get('manifest_sha256') == file_hash(RELEASE)
                 and authentication.get('archive_sha256') == release['archive_sha256']
                 and authentication.get('file_sha256') == expected
                 and hashlib.sha256(payload_bytes).hexdigest() == expected,
                 'Source authentication does not match the original release payload')
    _require(not any(str(n).startswith('hardware:') or a.get('_target_overlay') for n, a in program.nodes(data=True)),
             'Program node collides with hardware namespace')
    _require(not any(e.get('_target_overlay') for _, _, e in program.edges(data=True)), 'Already enriched graph')
    context = deployment_context(registry['profiles'][target], clock_period_ns, budgets, platform_available)
    return _construct(program, target, registry, context, authentication)


def strip_overlay(graph):
    _require(graph.graph.get('schema') == SCHEMA, 'Not a target-enriched graph')
    result = nx.MultiDiGraph()
    result.graph.update(deepcopy(graph.graph['program_metadata']))
    for node, attrs in graph.nodes(data=True):
        if not str(node).startswith('hardware:'):
            result.add_node(node, **deepcopy(attrs))
    for u, v, k, attrs in graph.edges(keys=True, data=True):
        if not attrs.get('_target_overlay'):
            _require(u in result and v in result, 'Original edge has hardware endpoint')
            result.add_edge(u, v, key=k, **deepcopy(attrs))
    _require(fingerprint(program_graph.graph_payload(result)) == graph.graph.get('program_payload_sha256'),
             'Original program payload changed')
    return result


def overlay_signature(graph):
    """Semantic overlay comparison independent of canonical/compact auxiliary leaves."""
    return {'nodes': sorted((n, a) for n, a in graph.nodes(data=True) if a.get('_target_overlay')),
            'edges': sorted((u, v, k, a) for u, v, k, a in graph.edges(keys=True, data=True) if a.get('_target_overlay')),
            'action_annotations': graph.graph['action_annotations'], 'action_slots': graph.graph['action_slots']}


def validate(graph, registry=None):
    """Restore/validate V2.2, then independently rederive and compare the overlay."""
    registry = load_registry() if registry is None else validate_registry(registry)
    program = strip_overlay(graph)
    _require(len([n for n in graph if str(n).startswith('hardware:')]) == 5, 'Exactly five hardware nodes required')
    for u, v, key, edge in graph.edges(keys=True, data=True):
        if not edge.get('_target_overlay'):
            continue
        _require(edge.get('relation_type') in RELATIONS and u in graph and v in graph, 'Invalid overlay endpoints/relation')
        _require(edge.get('compiler_evidence', {}).get('status') in
                 ('proven', 'possible', 'not_established', 'not_applicable'), 'Missing compiler evidence axis')
        _require(edge.get('hardware_evidence', {}).get('status') == 'documented', 'Missing hardware evidence axis')
        _require(edge.get('compatibility', {}).get('status') in ('possible', 'not_applicable') and
                 edge['compatibility'].get('actual_binding') == 'unknown', 'Unsupported actual/proven hardware mapping')
    d = graph.graph['deployment']
    expected = enrich(program, graph.graph['target_profile_id'], d['clock_period_ns'],
                      {r: v for r, v in d['requested_resources'].items() if v is not None},
                      platform_available={r: v for r, v in d['platform_available_resources'].items() if v is not None},
                      registry=registry, authentication=graph.graph.get('source_authentication'))
    _require(program_graph.graph_payload(expected) == program_graph.graph_payload(graph),
             'Overlay differs from registry, deployment, current implementation or compiler witnesses')
    return summary(graph)


def summary(graph):
    edges = [e for _, _, e in graph.edges(data=True) if e.get('_target_overlay')]
    coverage = graph.graph['program_metadata'].get('action_point_coverage', {}).get('records', [])
    return {'schema': SCHEMA, 'program_nodes': len(graph) - 5,
            'program_edges': graph.number_of_edges() - len(edges), 'added_nodes': 5, 'added_edges': len(edges),
            'relations': dict(sorted(Counter(e['relation_type'] for e in edges).items())),
            'compiler_evidence': dict(sorted(Counter(e['compiler_evidence']['status'] for e in edges).items())),
            'hardware_evidence': dict(Counter(e['hardware_evidence']['status'] for e in edges)),
            'compatibility': dict(Counter(e['compatibility']['status'] for e in edges)),
            'registered_actions': sum(bool(r['metadata_registered']) for r in coverage),
            'matched_actions': sum(bool(r['compiler_matched']) for r in coverage),
            'unresolved_actions': [r['action_id'] for r in coverage if r['metadata_registered'] and not r['compiler_matched']],
            'source_only_actions': [r['action_id'] for r in coverage if r['source_labeled'] and not r['metadata_registered']],
            'target_context_identity': graph.graph['target_context_identity'],
            'program_payload_sha256': graph.graph['program_payload_sha256']}


def serialized(graph):
    return canonical(program_graph.graph_payload(graph)) + '\n'


def write_graph(graph, path, registry=None):
    validate(graph, registry)
    path = Path(path)
    text = serialized(graph)
    _require(not path.exists() or path.read_text() == text, f'Refusing to overwrite a different graph: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read_graph(path, registry=None):
    payload = json.loads(Path(path).read_text())
    _require(payload.get('schema') == SCHEMA and payload['metadata'].get('schema') == SCHEMA, 'Wrong overlay schema')
    graph = nx.MultiDiGraph()
    graph.graph.update(payload['metadata'])
    for node in payload['nodes']:
        _require(node['id'] not in graph, 'Duplicate node ID')
        graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    for edge in payload['edges']:
        u, v, key = edge['source'], edge['target'], edge['key']
        _require(u in graph and v in graph and not graph.has_edge(u, v, key), 'Dangling/duplicate edge')
        graph.add_edge(u, v, key=key, **{k: val for k, val in edge.items() if k not in ('source', 'target', 'key')})
    validate(graph, registry)
    return graph


@contextmanager
def authenticated_release(archive, manifest=RELEASE):
    """Authenticate every bundled member, then safely extract an owned temporary copy."""
    release = json.loads(Path(manifest).read_text())
    _require(file_hash(archive) == release['archive_sha256'], 'Archive hash mismatch')
    expected = release['files_sha256']
    with tempfile.TemporaryDirectory(prefix='salt-v22-authenticated-') as directory:
        root, seen = Path(directory), set()
        with tarfile.open(archive) as stream:
            for member in stream:
                path = root / member.name
                _require(path.resolve().is_relative_to(root.resolve()) and not Path(member.name).is_absolute(),
                         'Unsafe archive member')
                _require(member.isfile() or member.isdir(), 'Archive links/devices are not accepted')
                if member.isdir():
                    continue
                _require(member.name not in seen, 'Duplicate archive member')
                seen.add(member.name)
                data = stream.extractfile(member).read()
                if member.name in expected:
                    _require(hashlib.sha256(data).hexdigest() == expected[member.name], f'Member hash mismatch: {member.name}')
                else:
                    _require(member.name in ('release.json', 'checksums.sha256'),
                             f'Unmanifested archive member: {member.name}')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        _require(set(expected) <= seen, 'Missing authenticated members')
        _require(json.loads((root / 'release.json').read_text())['files_sha256'] == expected,
                 'Bundled and external content manifests disagree')
        checksum_names = set()
        for line in (root / 'checksums.sha256').read_text().splitlines():
            checksum, name = line.split(maxsplit=1)
            _require(name in set(expected) | {'release.json'} and name not in checksum_names,
                     'Unexpected/duplicate bundled checksum')
            checksum_names.add(name)
            _require(file_hash(root / name) == checksum, 'Bundled checksum list disagrees')
        _require(checksum_names == set(expected) | {'release.json'}, 'Incomplete bundled checksum list')
        _require(len(list((root / 'canonical').glob('*.json'))) == release['graph_count'] == 63,
                 'Unexpected release cohort')
        yield root, release


def authenticated_input(path, manifest=RELEASE):
    release = json.loads(Path(manifest).read_text())
    graph = program_graph.read_graph(path)
    member = graph.graph['view'] + '/' + Path(path).name
    _require(member in release['files_sha256'] and file_hash(path) == release['files_sha256'][member],
             'Input is not an authenticated release graph')
    return graph, {'archive_sha256': release['archive_sha256'], 'manifest_sha256': file_hash(manifest),
                   'member': member, 'file_sha256': file_hash(path)}


def validate_corpus(archive, manifest, registry, output):
    start = time.perf_counter()
    rows, failures, variations = [], [], []
    with authenticated_release(archive, manifest) as (root, release):
        for path in sorted((root / 'canonical').glob('*.json')):
            began = time.perf_counter()
            try:
                original, auth = authenticated_input(path, manifest)
                compact, cauth = authenticated_input(root / 'compact' / path.name, manifest)
                _require(program_graph.graph_payload(original) == program_graph.graph_payload(program_graph.expand_compact(compact)),
                         'Canonical/compact reconstruction differs')
                graph = enrich(original, 'zcu104', 10, registry=registry, authentication=auth)
                other = enrich(compact, 'zcu104', 10, registry=registry, authentication=cauth)
                validate(graph, registry)
                validate(other, registry)
                _require(overlay_signature(graph) == overlay_signature(other), 'Canonical/compact overlay semantics differ')
                _require(serialized(graph) == serialized(enrich(original, 'zcu104', 10, registry=registry, authentication=auth)),
                         'Repeated enrichment is nondeterministic')
                for base, composed in ((original, graph), (compact, other)):
                    _require(program_graph.graph_payload(base) == program_graph.graph_payload(strip_overlay(composed)),
                             'Program restoration failed')
                text = serialized(graph)
                row = {'file': path.name, 'status': 'pass', **summary(graph),
                       'compact_program_nodes': len(compact), 'compact_program_edges': compact.number_of_edges(),
                       'serialized_bytes': len(text.encode()), 'serialized_sha256': hashlib.sha256(text.encode()).hexdigest(),
                       'runtime_seconds': time.perf_counter() - began}
                if path.name.startswith(('spcl_example_03-', 'rodinia_lud_1_', 'chstone-aes-',
                                         'serrano-kalman-filter-96', 'rodinia-hotspot-2-')):
                    for target in ('u200', 'u50'):
                        alternate = enrich(original, target, 10, registry=registry, authentication=auth)
                        validate(alternate, registry)
                        _require(program_graph.graph_payload(strip_overlay(alternate)) == program_graph.graph_payload(original),
                                 'Target variation changed compiler facts')
                        variations.append({'file': path.name, 'target': target, 'status': 'pass', **summary(alternate)})
                rows.append(row)
                print(path.name + ' PASS', flush=True)
            except Exception as exc:
                failures.append({'file': path.name, 'status': 'failed', 'reason': str(exc)})
                print(path.name + ' FAILED: ' + str(exc), flush=True)
        report = {'schema': 'salt-hls-target-overlay-validation-v1', 'archive_sha256': release['archive_sha256'],
                  'manifest_sha256': file_hash(manifest), 'authenticated_members': len(release['files_sha256']),
                  'registry_sha256': fingerprint(registry), 'implementation': implementation_identity(),
                  'graphs': len(rows) + len(failures), 'passed': len(rows), 'failed': len(failures),
                  'blocked': 0, 'runtime_seconds': time.perf_counter() - start,
                  'rows': rows, 'failures': failures, 'profile_variations': variations,
                  'independent_adjudication': 'pending', 'native_analysis_rerun': False,
                  'training_or_synthesis_performed': False}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', default=str(PROFILES))
    sub = parser.add_subparsers(dest='command', required=True)
    compose = sub.add_parser('compose')
    compose.add_argument('--graph', required=True, help='Canonical or compact JSON; authenticated against manifest')
    compose.add_argument('--manifest', default=str(RELEASE))
    compose.add_argument('--target', required=True)
    compose.add_argument('--clock-period-ns', type=float, required=True)
    for name in ('bram18k', 'dsp', 'lut', 'ff'):
        compose.add_argument('--available-' + name, type=int)
    compose.add_argument('--platform-resources-json', help='Optional JSON object with BRAM_18K/DSP/LUT/FF counts')
    compose.add_argument('--output', required=True)
    for name in ('validate', 'inspect'):
        command = sub.add_parser(name)
        command.add_argument('--graph', required=True)
    corpus = sub.add_parser('corpus')
    corpus.add_argument('--archive', required=True)
    corpus.add_argument('--manifest', default=str(RELEASE))
    corpus.add_argument('--output', required=True)
    extract = sub.add_parser('authenticate')
    extract.add_argument('--archive', required=True)
    extract.add_argument('--manifest', default=str(RELEASE))
    extract.add_argument('--output-dir', required=True, help='New directory for authenticated release copy')
    args = parser.parse_args(argv)
    registry = load_registry(args.profiles)
    if args.command == 'compose':
        program, auth = authenticated_input(args.graph, args.manifest)
        budgets = {r: getattr(args, 'available_' + option) for r, option in
                   zip(RESOURCES, ('bram18k', 'dsp', 'lut', 'ff'))
                   if getattr(args, 'available_' + option) is not None}
        platform = json.loads(Path(args.platform_resources_json).read_text()) if args.platform_resources_json else None
        graph = enrich(program, args.target, args.clock_period_ns, budgets, registry=registry,
                       platform_available=platform, authentication=auth)
        write_graph(graph, args.output, registry)
        print(json.dumps(summary(graph), indent=2, sort_keys=True))
    elif args.command in ('validate', 'inspect'):
        graph = read_graph(args.graph, registry)
        result = summary(graph)
        if args.command == 'inspect':
            result.update(deployment=graph.graph['deployment'], action_annotations=graph.graph['action_annotations'],
                          excluded_relationships=graph.graph['excluded_relationships'])
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == 'authenticate':
        import shutil
        with authenticated_release(args.archive, args.manifest) as (root, release):
            _require(not Path(args.output_dir).exists(), 'Extraction destination must be new')
            shutil.copytree(root, args.output_dir)
            print(json.dumps({'graphs': release['graph_count'], 'authenticated_members': len(release['files_sha256']),
                              'archive_sha256': release['archive_sha256']}, sort_keys=True))
    else:
        report = validate_corpus(args.archive, args.manifest, registry, args.output)
        print(json.dumps({k: report[k] for k in ('graphs', 'passed', 'failed', 'runtime_seconds')}, sort_keys=True))
        if report['failed']:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
