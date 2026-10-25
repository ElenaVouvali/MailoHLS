"""Authenticate MLIR graphs and verify action relationships without model inference."""
from __future__ import annotations
import ast
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from itertools import combinations
from pathlib import Path
from .configurations import ROOT, sha256, display_path, repo_path
RELATIONS = {
    1: 'self',
    2: 'parent_loop',
    4: 'child_loop',
    8: 'loop_array_access',
    16: 'proven_dependence_forward',
    32: 'proven_dependence_reverse'
}

SCHEMA = 'mailohls-action-relations-v1'

BUILDER = ROOT / 'GNN_branch/build_structural_memory.py'

def existing_projector():
    """Compile the unchanged helper definitions, as existing helper tests do.

    The exporter's top level parses required training/checkpoint CLI arguments.
    AST extraction avoids those side effects, torch_geometric, and GPU/model
    loading. The projection algorithm itself is not copied or approximated.
    """
    import torch
    names = {
        '_edge_records',
        '_has_edge',
        '_numeric_node_map',
        '_loop_action_scope_pseudo',
        '_loop_region_distances',
        'build_action_relation_mask'
    }
    tree = ast.parse(BUILDER.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Existing relation projector changed; review refinement adapter')
    constants = {
        'NODE_TYPE_OP',
        'NODE_TYPE_PSEUDO_BLOCK',
        'NODE_TYPE_PRAGMA',
        'NODE_TYPE_ARRAY_SCOPE',
        'FLOW_CONTROL',
        'FLOW_CALL',
        'FLOW_PSEUDO_BLOCK',
        'FLOW_LOOP_HIERARCHY',
        'FLOW_ARRAY_SCOPE',
        'FLOW_REGION',
        'FLOW_MEMORY_DEPENDENCE',
        'REL_SELF',
        'REL_PARENT',
        'REL_CHILD',
        'REL_ARRAY',
        'REL_DEP_FORWARD',
        'REL_DEP_REVERSE'
    }
    assignments = [
        n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and (n.targets[0].id in constants)
    ]
    namespace = {'torch': torch}
    exec(compile(ast.Module(assignments + nodes, type_ignores=[]), str(BUILDER), 'exec'), namespace)
    return namespace['build_action_relation_mask']

def pairs_from_bits(bits, active, labels, actions, evidence, inactive_actions=()):
    """Absolute Lk index alignment; only real active, non-self relations qualify."""
    n = len(active)
    if len(labels) != n or len(bits) != n or any((len(row) != n for row in bits)):
        raise ValueError('Inconsistent relation tensor shapes')
    for i in range(n):
        if bool(active[i]):
            if labels[i] != i + 1 or f'L{i + 1}' not in actions:
                raise ValueError('Sparse absolute Lk slot alignment mismatch')
        for j in range(n):
            value = bits[i][j]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value & ~63:
                raise ValueError('Unknown/noninteger relation bits')
            if value and (not active[i] or not active[j]):
                raise ValueError('Inactive slots contain relation bits')
        if active[i] and (not bits[i][i] & 1):
            raise ValueError('Active action lost its self relation')
    observed = {f'L{i + 1}' for i in range(n) if active[i]}
    if set(inactive_actions) - set(actions) or observed != set(actions) - set(inactive_actions):
        raise ValueError('Relations do not cover exactly the source actions')
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not active[i] or not active[j]:
                continue
            (forward, reverse) = (bits[i][j] & ~1, bits[j][i] & ~1)
            combined = forward | reverse
            if not combined:
                continue
            if bool(forward & 8) != bool(reverse & 8) or bool(forward & 2) != bool(reverse & 4) or bool(forward & 4) != bool(reverse & 2):
                raise ValueError('Asymmetric compiler relation semantics')
            if bool(forward & 16) != bool(reverse & 32) or bool(forward & 32) != bool(reverse & 16):
                raise ValueError('Asymmetric proven dependence semantics')
            kinds = {actions[f'L{i + 1}'].kind, actions[f'L{j + 1}'].kind}
            if combined & 8 and kinds != {'loop', 'array'}:
                raise ValueError('Loop-array relation has incompatible action kinds')
            if combined & (2 | 4 | 16 | 32) and kinds != {'loop'}:
                raise ValueError('Loop hierarchy/dependence relation requires two loops')
            pairs.append(
                {
                    'action_i': f'L{i + 1}',
                    'action_j': f'L{j + 1}',
                    'relation_type': [name for (bit, name) in RELATIONS.items() if bit != 1 and combined & bit],
                    'relation_bits': {'i_to_j': forward, 'j_to_i': reverse},
                    **evidence
                }
            )
    return pairs

def graph_relations(path, source_hash, metadata_hash, actions, relation_metadata_path=None):
    import networkx as nx
    if Path(path).suffix != '.gexf':
        raise ValueError(
            'Refinement requires an authenticated MLIR GEXF, not a dense attention or unverified memory pack'
        )
    try:
        graph = nx.read_gexf(path)
    except Exception as exc:
        if isinstance(exc, OSError):
            raise
        raise ValueError(f'Unreadable MLIR graph: {exc}') from exc
    name = graph.graph.get('name', '')
    if not name.startswith('mailohls-meta-v1:'):
        raise ValueError('Not a provenance-bearing MLIR graph (LLVM/HARP is not accepted)')
    metadata = json.loads(name.split(':', 1)[1])
    relation_actions = actions
    if relation_metadata_path:
        from LLM_branch.common import directive_domains as dd
        relation_metadata_path = Path(relation_metadata_path)
        relation_actions = dd.load_source_actions('relation-source', relation_metadata_path.parent)
        if any(
            (
                label not in actions or action != actions[label] for (label, action) in relation_actions.items()
            )
        ):
            raise ValueError('Compiler-active metadata is not an exact subset of the synthesis action contract')
        metadata_hash = sha256(relation_metadata_path)
    if metadata.get('source_sha256') != source_hash or metadata.get('action_sha256') != metadata_hash:
        raise ValueError('MLIR graph source/metadata hashes do not match the source snapshot')
    if metadata.get('compiler_analysis_schema') != 'mailohls-native-analysis-v3' or metadata.get('mlir_level') != 'affine+scf+memref+arith+func':
        raise ValueError('Native compiler analysis at the required MLIR level is missing')
    n = max((int(label[1:]) for label in actions))
    (node_ids, active, labels) = ([-1] * n, [False] * n, [-1] * n)
    declared = {
        str(a.get('action_id')) for (_, a) in graph.nodes(data=True) if int(a.get('type', -1)) == 100
    }
    if declared != set(relation_actions):
        raise ValueError(
            f'Graph action set mismatch: graph={sorted(declared)}, compiler-active contract={sorted(relation_actions)}'
        )
    for (node, attrs) in graph.nodes(data=True):
        label = str(attrs.get('action_id', ''))
        if label not in relation_actions:
            continue
        loop = int(attrs.get('is_loop', 0)) == 1
        array = int(attrs.get('type', -1)) == 104
        if not loop and (not array):
            continue
        if (actions[label].kind == 'loop') != loop:
            raise ValueError(f'{label}: semantic anchor kind mismatch')
        index = int(label[1:]) - 1
        if active[index]:
            raise ValueError(f'Duplicate semantic anchor for {label}')
        (node_ids[index], active[index], labels[index]) = (int(node), True, index + 1)
    if {f'L{i + 1}' for (i, flag) in enumerate(active) if flag} != set(relation_actions):
        raise ValueError('Source action has no semantic MLIR anchor')
    (_, bits, stats) = existing_projector()(graph, node_ids=node_ids, active_mask=active, max_slots=n)
    evidence = {
        'compiler_evidence_source': 'GNN_branch/build_structural_memory.py:build_action_relation_mask',
        'relation_schema': SCHEMA,
        'projector_sha256': sha256(BUILDER),
        'graph_or_memory_artifact': display_path(path),
        'artifact_hash': sha256(path),
        'compiler_analysis_schema': metadata['compiler_analysis_schema'],
        'dependence_policy': 'Only certainty=proven flow-12 edges; uncertain analyses excluded'
    }
    inactive = sorted(set(actions) - set(relation_actions), key=lambda label: int(label[1:]))
    evidence['compiler_active_metadata'] = display_path(relation_metadata_path) if relation_metadata_path else None
    evidence['compiler_active_metadata_sha256'] = metadata_hash
    return {
        'status': 'available',
        'pairs': pairs_from_bits(bits.tolist(), active, labels, actions, evidence, inactive),
        'artifact': display_path(path),
        'artifact_hash': sha256(path),
        'action_relation_bits': bits.tolist(),
        'active_mask': active,
        'labels': labels,
        'graph_metadata': metadata,
        'stats': stats,
        'inactive_actions': inactive,
        'inactive_policy': 'Unrepresented actions retain their complete directive assignment, but never become edit sites',
        'compiler_active_metadata_path': display_path(relation_metadata_path) if relation_metadata_path else None,
        'compiler_active_metadata_sha256': metadata_hash
    }

def compiler_tools(build=None, python=None):
    """Find the existing native pipeline; never install/replace a toolchain."""
    selected = build or os.environ.get('POLYGEIST_BUILD')
    found = shutil.which(os.environ.get('CGEIST', 'cgeist'))
    builds = [Path(selected)] if selected else []
    if found:
        default = Path(found).resolve().parents[1]
        builds += [default, default.parent / 'build-mailohls-assertions']
    for directory in dict.fromkeys(builds):
        libs = directory / 'tools/mlir/python_packages/mlir_core'
        extensions = list(libs.glob('mlir/_mlir_libs/_mailohls_analysis*.so'))
        cgeist = directory / 'bin/cgeist'
        if not cgeist.is_file() or not extensions:
            continue
        choices = [
            python or os.environ.get('MLIR_PYTHON'),
            str(ROOT / '.toolchain-python/bin/python'),
            str(Path.home() / '.mlir-python311/bin/python'),
            shutil.which('python3.11'),
            sys.executable
        ]
        for interpreter in dict.fromkeys((p for p in choices if p and Path(p).is_file())):
            env = {**os.environ, 'PYTHONPATH': str(libs), 'PYTHONHASHSEED': '0'}
            probe = subprocess.run(
                [interpreter, '-c', 'from mlir import ir; from mlir._mlir_libs import _mailohls_analysis'],
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            if probe.returncode == 0:
                return {
                    'build': str(directory),
                    'python': interpreter,
                    'pythonpath': str(libs),
                    'cgeist': str(cgeist),
                    'cgeist_sha256': sha256(cgeist),
                    'analysis_extension_sha256': sha256(extensions[0]),
                    'lock_file_sha256': sha256(ROOT / 'toolchain/lock.env')
                }
    raise FileNotFoundError(
        'Native MLIR/Polygeist pipeline unavailable. Build toolchain/bootstrap.sh, then set POLYGEIST_BUILD and MLIR_PYTHON.'
    )

def regenerate_graph(source, output, tools, timeout=600, compiler_symbol=None):
    command = [
        tools['python'],
        str(ROOT / 'GNN_branch/mlir_graph_gen.py'),
        str(source),
        '--cgeist',
        tools['cgeist'],
        '--output',
        str(output),
        '--mlir-output',
        str(output.with_suffix('.mlir')),
        '--analysis-output',
        str(output.with_suffix('.analysis.json'))
    ]
    if compiler_symbol is not None:
        command += ['--compiler-symbol', compiler_symbol]
    env = {**os.environ, 'PYTHONPATH': tools['pythonpath'], 'PYTHONHASHSEED': '0'}
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.generation.log')
    with log.open('w') as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'MLIR generation timed out; see {log}') from exc
    if result.returncode or not output.is_file():
        raise RuntimeError(f'MLIR generation failed (exit {result.returncode}); see {log}')
    return {
        'argv': command,
        'environment': {k: env[k] for k in ('PYTHONPATH', 'PYTHONHASHSEED')},
        'log': display_path(log)
    }

def _edges(graph, node):
    return graph.out_edges(node, data=True)

def local_region(graph, loop):
    """Existing loop-region traversal restricted to the owning function; no calls."""
    owner = graph.nodes[loop].get('function')
    if owner is None:
        return set()
    pending = deque(
        (
            target for (_, target, edge) in _edges(graph, loop) if int(edge.get('flow', -1)) == 8 and edge.get('role') == 'region_entry'
        )
    )
    found = set()
    while pending:
        node = pending.popleft()
        if node in found or graph.nodes[node].get('function') != owner:
            continue
        found.add(node)
        for (_, target, edge) in _edges(graph, node):
            (flow, role) = (int(edge.get('flow', -1)), edge.get('role'))
            if flow == 8 and role == 'region_entry' or (flow == 0 and role in {'next_in_block', 'block_successor'}):
                pending.append(target)
    return found

def _anchor(graph, label, kind):
    matches = [
        node for (node, a) in graph.nodes(data=True) if a.get('action_id') == label and (
            kind == 'loop' and int(a.get('is_loop', 0)) == 1 or (kind == 'array' and int(a.get('type', -1)) == 104)
        )
    ]
    if len(matches) != 1:
        raise ValueError(f'Missing/ambiguous semantic anchor for {label}')
    node = matches[0]
    if graph.nodes[node].get('action_resolution') != 'source_location':
        raise ValueError(f'Non-exact source binding for {label}')
    return node

def _operation(graph, node):
    a = graph.nodes[node]
    return {
        'graph_node': str(node),
        **{
            key: a[key] for key in (
                'op_uid',
                'text',
                'full_text',
                'function',
                'block',
                'source_file',
                'source_line',
                'source_column'
            ) if key in a
        }
    }

def _root_access(graph, root, operation):
    """The compiler's concrete root -> access edge, never just SSA reachability."""
    return [
        dict(edge) for (_, target, edge) in _edges(graph, root) if target == operation and int(edge.get('flow', -1)) == 10 and (edge.get('role') in {'read', 'write', 'readwrite'})
    ]

def verify_pair(graph, array_label, loop_label, original=None, artifact_hash=None):
    (array, loop) = (_anchor(graph, array_label, 'array'), _anchor(graph, loop_label, 'loop'))
    roots = [
        target for (_, target, edge) in _edges(graph, array) if int(edge.get('flow', -1)) == 7 and edge.get('role') == 'array_root'
    ]
    accesses = {
        target for (_, target, edge) in _edges(graph, array) if int(edge.get('flow', -1)) == 7 and edge.get('role') in {'array_read', 'array_write', 'array_readwrite'}
    }
    owned = local_region(graph, loop)
    full = set(existing_projector().__globals__['_loop_region_distances'](graph, loop))
    (supporting, callsites, direct, indirect) = ([], [], [], [])
    valid_root = len(roots) == 1 and int(graph.nodes[roots[0]].get('is_memory_root', 0)) == 1
    if valid_root:
        root = roots[0]
        for operation in sorted(accesses & owned, key=str):
            bindings = _root_access(graph, root, operation)
            if bindings:
                direct.append(operation)
                supporting.append(_operation(graph, operation))
                callsites.append(
                    {
                        'call_or_access_node': str(operation),
                        'root_node': str(root),
                        'root_access_edges': bindings,
                        'binding_rule': "compiler root-access edge in the loop's own function/region"
                    }
                )
        for call in sorted(owned, key=str):
            if graph.nodes[call].get('text') != 'func.call':
                continue
            actuals = {
                n: int(edge['position']) for (n, _, edge) in graph.in_edges(call, data=True) if int(edge.get('flow', -1)) == 1 and edge.get('role') == 'operand' and ('position' in edge)
            }
            callees = {
                graph.nodes[target].get('function') for (_, target, edge) in _edges(graph, call) if int(edge.get('flow', -1)) == 2 and edge.get('role') == 'calls'
            }
            for (actual, position) in actuals.items():
                if actual != root and (
                    not any(
                        (
                            int(e.get('flow', -1)) == 9 and e.get('certainty') in {'proven', 'must_alias'} for (_, target, e) in _edges(graph, root) if target == actual
                        )
                    )
                ):
                    continue
                for (_, formal, binding) in _edges(graph, actual):
                    if int(binding.get('flow', -1)) != 2 or binding.get('role') != 'actual_to_formal' or graph.nodes[formal].get('function') not in callees:
                        continue
                    if 'callsite' in binding:
                        matched = str(binding['callsite']) == str(call)
                    else:
                        matched = int(binding.get('position', -1)) == 1 + position
                    if not matched:
                        continue
                    for operation in sorted(accesses & full, key=str):
                        if _root_access(graph, formal, operation):
                            indirect.append(operation)
                            supporting.append(_operation(graph, operation))
                            callsites.append(
                                {
                                    'call_node': str(call),
                                    'actual_node': str(actual),
                                    'actual_operand_position': position,
                                    'formal_node': str(formal),
                                    'callee_function': graph.nodes[formal].get('function'),
                                    'binding_edge': dict(binding)
                                }
                            )
    shared = sorted((accesses & full) - owned, key=str)
    if direct:
        (classification, reason) = (
            'direct_verified',
            "Concrete compiler root-access edge and exact source action bindings in the loop's own region."
        )
    elif indirect:
        (classification, reason) = (
            'indirect_verified',
            'Call-site-specific actual/formal binding and callee memory access verified.'
        )
    elif accesses & full or (original and any(original.get('relation_bits', {}).values())) or (not valid_root):
        (classification, reason) = (
            'conservative_unknown',
            'Only shared helper reachability, unresolved root/access bindings, or legacy relation evidence; no context-specific proof.'
        )
    else:
        (classification, reason) = (
            'no_evidenced_relation',
            'No qualifying access/binding established; this is not proof of independence.'
        )
    if not supporting:
        supporting = [_operation(graph, node) for node in shared]
    bindings = {
        label: {
            key: graph.nodes[node].get(key) for key in (
                'action_id',
                'action_resolution',
                'action_source_file',
                'action_source_line',
                'action_source_column',
                'array_var'
            )
        } for (label, node) in [(array_label, array), (loop_label, loop)]
    }
    return {
        'action_i': array_label,
        'action_j': loop_label,
        'original_relation_bits': (original or {}).get('relation_bits', {'i_to_j': 0, 'j_to_i': 0}),
        'original_relation_types': (original or {}).get('relation_type', []),
        'verified_relation_class': classification,
        'source_action_bindings': bindings,
        'relevant_memory_root': [_operation(graph, node) for node in roots],
        'supporting_mlir_operations': supporting,
        'supporting_graph_nodes': [op['graph_node'] for op in supporting],
        'shared_helper_nodes': shared,
        'callsite_provenance': callsites,
        'evidence_artifact_hash': artifact_hash,
        'reason': reason
    }

def audit_all(graph, actions, legacy_pairs, artifact_hash):
    by_pair = {frozenset((p['action_i'], p['action_j'])): p for p in legacy_pairs}
    return [
        verify_pair(graph, array, loop, by_pair.get(frozenset((array, loop))), artifact_hash) for (array, a) in actions.items() if a.kind == 'array' for (loop, b) in actions.items() if b.kind == 'loop'
    ]

def audit_kernel(kernel, actions):
    import networkx as nx
    record = kernel['relations']
    if record['status'] != 'available':
        return ([], [])
    path = repo_path(record['artifact'])
    validated = graph_relations(
        path,
        kernel['source_provenance']['source_sha256'],
        kernel['source_provenance']['metadata_sha256'],
        actions,
        repo_path(record['compiler_active_metadata_path']) if record.get('compiler_active_metadata_path') else None
    )
    active = {
        label: action for (label, action) in actions.items() if label not in validated['inactive_actions']
    }
    graph = nx.read_gexf(path)
    result = audit_all(graph, active, validated['pairs'], sha256(path))
    result += audit_loop_pairs(graph, active, validated['pairs'], sha256(path))
    return (result, validated['inactive_actions'])

def audit_loop_pairs(graph, active, legacy_pairs, artifact_hash):
    """Only exact same-function regions or proven same-function dependencies."""
    result = []
    lookup = {frozenset((p['action_i'], p['action_j'])): p for p in legacy_pairs}
    loops = {
        label: _anchor(graph, label, 'loop') for (label, action) in active.items() if action.kind == 'loop'
    }
    regions = {label: local_region(graph, node) for (label, node) in loops.items()}
    for (left, right) in combinations(loops, 2):
        original = lookup.get(frozenset((left, right)), {})
        (a, b) = (loops[left], loops[right])
        nesting = graph.nodes[a].get('function') == graph.nodes[b].get('function') and (b in regions[left] or a in regions[right])
        dependencies = []
        for source in regions[left]:
            for (_, target, edge) in graph.out_edges(source, data=True):
                if target in regions[right] and graph.nodes[source].get('function') == graph.nodes[target].get('function') and (int(edge.get('flow', -1)) == 12) and (edge.get('certainty') == 'proven'):
                    dependencies.append(
                        {
                            'source': _operation(graph, source),
                            'target': _operation(graph, target),
                            'edge': dict(edge)
                        }
                    )
        for source in regions[right]:
            for (_, target, edge) in graph.out_edges(source, data=True):
                if target in regions[left] and graph.nodes[source].get('function') == graph.nodes[target].get('function') and (int(edge.get('flow', -1)) == 12) and (edge.get('certainty') == 'proven'):
                    dependencies.append(
                        {
                            'source': _operation(graph, source),
                            'target': _operation(graph, target),
                            'edge': dict(edge)
                        }
                    )
        classification = 'direct_verified' if nesting or dependencies else 'conservative_unknown' if original else 'no_evidenced_relation'
        result.append(
            {
                'action_i': left,
                'action_j': right,
                'verified_relation_class': classification,
                'verified_relation_types': ['loop_nesting'] if nesting else ['proven_memory_dependence'] if dependencies else [],
                'original_relation_bits': original.get('relation_bits', {'i_to_j': 0, 'j_to_i': 0}),
                'original_relation_types': original.get('relation_type', []),
                'evidence_artifact_hash': artifact_hash,
                'supporting_graph_nodes': [a, b],
                'supporting_mlir_operations': [_operation(graph, a), _operation(graph, b)],
                'dependence_evidence': dependencies,
                'source_action_bindings': {
                    label: {
                        key: graph.nodes[node].get(key) for key in ('action_id', 'action_source_file', 'action_source_line', 'action_resolution')
                    } for (label, node) in ((left, a), (right, b))
                },
                'reason': 'Verified same-function nested loop region or explicit proven dependence edge.' if nesting or dependencies else 'No direct region/dependence proof; absence is not independence.'
            }
        )
    return result
