"""Fixed, evidence-preserving semantic edge features for V2.2 RC3 tensors.

This module tensorizes compiler-recorded attributes only. It does not run a
compiler analysis or mutate the graph. Unknown and over-capacity structures
remain masked; no identifier string is used as a learned feature.
"""
from __future__ import annotations

import math
import re

SIGNED_SCALE = 24.0
# Input operands are represented by one edge per verified SSA operand, so an
# input-count cap would discard valid map bindings without saving tensor width.
MAX_AFFINE_INPUTS = None
MAX_AFFINE_RESULTS = 3
MAX_VIEW_DIMS = 4

FORWARDING_ROLES = ('iter_init', 'loop_backedge', 'loop_result', 'unknown')
STATE_KINDS = ('scalar', 'memory', 'unknown')
VIEW_KINDS = ('memref.subview', 'memref.reinterpret_cast', 'memref.cast',
              'llvm.getelementptr', 'polygeist.memref2pointer',
              'polygeist.pointer2memref', 'polygeist.subindex', 'other', 'unknown')
OFFSET_SEMANTICS = ('relative_to_source_view', 'relative_to_underlying_storage',
                    'shape_layout_compatible_identity', 'base_pointer_with_element_indices',
                    'aligned_pointer_plus_source_descriptor_offset_or_pointer_ABI_cast',
                    'view_of_source_pointer', 'mapping_retained_not_normalized',
                    'not_applicable', 'unknown')
CALL_RESOLUTION = ('proven', 'possible', 'unresolved', 'unknown')
CARRYING_STATUS = ('proven', 'possible', 'not_established', 'not_applicable', 'unknown')


def _pair(value):
    if value is None:
        return [0.0, 0.0]
    if type(value) not in (int, float) or not math.isfinite(value):
        return [0.0, 0.0]
    scaled = math.copysign(math.log2(1 + abs(float(value))) / SIGNED_SCALE, value)
    return [scaled, 1.0]


def _one_hot(value, vocabulary):
    row = [0.0] * len(vocabulary)
    row[vocabulary.index(value if value in vocabulary else vocabulary[-1])] = 1.0
    return row


def _split_top_level(text):
    values, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in '([':
            depth += 1
        elif char in ')]':
            depth -= 1
            if depth < 0:
                return None
        elif char == ',' and depth == 0:
            values.append(text[start:i].strip())
            start = i + 1
    if depth:
        return None
    tail = text[start:].strip()
    if tail or text.strip():
        values.append(tail)
    return values


def parse_linear_affine_map(text):
    """Parse only additive integer-affine expressions over dN/sN identifiers.

    This intentionally rejects floor/ceil division, modulo, min/max and other
    MLIR affine expressions. Parsing is accepted only after SSA bindings have
    separately been matched to the compiler-recorded operand positions.
    """
    match = re.fullmatch(
        r'\s*affine_map<\(([^()]*)\)(?:\[([^\[\]]*)\])?\s*->\s*\((.*)\)>\s*',
        str(text))
    if not match:
        return None
    dims = [v.strip() for v in match.group(1).split(',') if v.strip()]
    symbols = [v.strip() for v in (match.group(2) or '').split(',') if v.strip()]
    if dims != [f'd{i}' for i in range(len(dims))] or symbols != [f's{i}' for i in range(len(symbols))]:
        return None
    exprs = _split_top_level(match.group(3))
    if exprs is None:
        return None
    inputs = dims + symbols
    results = []
    for expr in exprs:
        if not expr:
            return None
        tokens = re.findall(r'[+-]?\s*(?:\d+\s*\*\s*[ds]\d+|[ds]\d+\s*\*\s*\d+|[ds]\d+|\d+)', expr)
        if not tokens or ''.join(tokens).replace(' ', '') != expr.replace(' ', ''):
            return None
        coeff = {name: 0 for name in inputs}
        offset = 0
        for token in tokens:
            token = token.replace(' ', '')
            sign = -1 if token.startswith('-') else 1
            body = token.lstrip('+-')
            if '*' in body:
                left, right = body.split('*')
                var, factor = (left, int(right)) if left[:1] in ('d', 's') else (right, int(left))
                if var not in coeff:
                    return None
                coeff[var] += sign * factor
            elif body[:1] in ('d', 's'):
                if body not in coeff:
                    return None
                coeff[body] += sign
            else:
                offset += sign * int(body)
        results.append({'coefficients': coeff, 'offset': offset})
    return {'dimensions': dims, 'symbols': symbols, 'inputs': inputs, 'results': results}


def _index_context(graph):
    """Build one verified map/binding index to avoid reparsing per message edge."""
    bindings, access_maps = {}, {}
    for operation, memory, key, edge in graph.edges(keys=True, data=True):
        if edge.get('relation_type') != 'memory_access':
            continue
        info = edge.get('access_indices') or {}
        if info.get('status') != 'compiler_recorded':
            continue
        valid = _validated_map_inputs(graph, operation, info)
        if valid is None:
            continue
        parsed, values, positions = valid
        map_row = {'parsed': parsed, 'values': values, 'positions': positions,
                   'supported': len(parsed['results']) <= MAX_AFFINE_RESULTS,
                   'witness': str(key)}
        prior_map = access_maps.get(operation)
        if prior_map is not None and prior_map.get('conflict'):
            continue
        if prior_map is not None and (prior_map['parsed'] != parsed or prior_map['values'] != values
                                      or prior_map['positions'] != positions):
            access_maps[operation] = {'conflict': True}
        elif prior_map is None:
            access_maps[operation] = map_row
        if not map_row['supported']:
            continue
        if len(parsed['results']) > MAX_AFFINE_RESULTS:
            continue
        # One map may be repeated on multiple resolved-root edges. A conflict
        # invalidates the binding instead of selecting one arbitrary witness.
        for ordinal, (value, position) in enumerate(zip(values, positions)):
            ident = (value, operation, position)
            row = {'input_ordinal': ordinal, 'input_kind': 'dimension' if ordinal < len(parsed['dimensions']) else 'symbol',
                   'coefficients': [parsed['results'][i]['coefficients'][parsed['inputs'][ordinal]]
                                    for i in range(len(parsed['results']))],
                   'result_count': len(parsed['results']), 'input_count': len(parsed['inputs']),
                   'symbol_count': len(parsed['symbols']), 'supported': True, 'witness': str(key)}
            prior = bindings.get(ident)
            if prior is not None and any(prior.get(field) != row.get(field)
                                         for field in ('input_ordinal','input_kind','coefficients',
                                                       'result_count','input_count','symbol_count')):
                bindings[ident] = {'supported': False}
            elif prior is None:
                bindings[ident] = row
    conflicted_operations = {operation for operation, row in access_maps.items()
                             if row.get('conflict')}
    if conflicted_operations:
        bindings = {key: value for key, value in bindings.items()
                    if key[1] not in conflicted_operations}
    return bindings, access_maps


def _index_bindings(graph):
    """Compatibility helper for tests/consumers needing just operand bindings."""
    return _index_context(graph)[0]


def _validated_map_inputs(graph, operation, info):
    if info.get('status') != 'compiler_recorded':
        return None
    parsed = parse_linear_affine_map(info.get('affine_map'))
    values, positions = info.get('ssa_operands'), info.get('operand_positions')
    if (parsed is None or not isinstance(values, list) or not isinstance(positions, list)
            or len(values) != len(positions) or len(values) != len(parsed['inputs'])
            or len(set(positions)) != len(positions)):
        return None
    actual = {}
    for value, _, _, operand in graph.in_edges(operation, keys=True, data=True):
        if operand.get('relation_type') != 'operand':
            continue
        position = (operand.get('legacy_evidence') or {}).get('position')
        if type(position) is int:
            actual.setdefault(position, set()).add(value)
    if any(type(pos) is not int or actual.get(pos) != {value}
           for value, pos in zip(values, positions)):
        return None
    return parsed, values, positions


def _call_resolution(graph, attrs):
    call = attrs.get('call_operation')
    if call is None and attrs.get('relation_type') == 'calls':
        call = attrs.get('source')
    if call not in graph:
        return 'unknown'
    value = graph.nodes[call].get('call_resolution')
    return value if value in CALL_RESOLUTION[:-1] else 'unknown'


def _distance(edge):
    carrying = edge.get('carrying_loop')
    rows = edge.get('distance_bounds') or []
    selected = next((row for row in rows if row.get('loop') == carrying), None) if carrying else None
    if selected is None and len(rows) == 1:
        selected = rows[0]
    exact_values = edge.get('distance_exact') or []
    component_rows = edge.get('distance_components') or []
    component_index = next((i for i, row in enumerate(component_rows)
                            if carrying and row.get('loop') == carrying), None)
    if component_index is None and len(component_rows) == 1:
        component_index = 0
    exact = exact_values[component_index] if component_index is not None and component_index < len(exact_values) else None
    if selected is not None and exact is None and selected.get('lower') == selected.get('upper'):
        # Equal bounds are an exact mathematical interval only when the native
        # exactness record is absent; retain bounds and mark exactness unknown.
        exact = None
    return selected, exact


def feature_names():
    names = ['operand_position_value', 'operand_position_known',
             'index_role', 'index_role_known', 'index_map_supported', 'index_map_overflow',
             'index_input_is_dimension', 'index_input_is_symbol',
             'index_input_ordinal_value', 'index_input_ordinal_known',
             'index_input_count_value', 'index_input_count_known',
             'index_symbol_count_value', 'index_symbol_count_known',
             'index_result_rank_value', 'index_result_rank_known']
    for i in range(MAX_AFFINE_RESULTS):
        names += [f'index_coeff_result_{i}_value']
    names += ['affine_map_known', 'affine_map_supported', 'affine_map_overflow']
    names += ['affine_input_count_value', 'affine_input_count_known',
              'affine_result_count_value', 'affine_result_count_known',
              'affine_symbol_count_value', 'affine_symbol_count_known']
    for i in range(MAX_AFFINE_RESULTS):
        names += [f'affine_offset_{i}_value']
    names += ['dependence_distance_lower_value', 'dependence_distance_lower_known',
              'dependence_distance_upper_value', 'dependence_distance_upper_known',
              'dependence_exact_value', 'dependence_exact_known',
              'dependence_bounds_known', 'dependence_depth_value', 'dependence_depth_known',
              'carrying_loop_known', 'carrying_loop_is_action', 'carrying_loop_depth_value',
              'carrying_loop_depth_known']
    names += [f'carrying_status_{x}' for x in CARRYING_STATUS]
    names += [f'forwarding_role_{x}' for x in FORWARDING_ROLES]
    names += [f'state_kind_{x}' for x in STATE_KINDS]
    names += ['forwarding_evidence_origin_verified']
    names += [f'view_kind_{x}' for x in VIEW_KINDS]
    names += [f'view_offset_semantics_{x}' for x in OFFSET_SEMANTICS]
    names += ['view_source_rank_value', 'view_source_rank_known',
              'view_result_rank_value', 'view_result_rank_known', 'view_dimension_overflow']
    for i in range(MAX_VIEW_DIMS):
        for what in ('offset', 'size', 'stride'):
            names += [f'view_{what}_{i}_value', f'view_{what}_{i}_known']
    names += [f'call_resolution_{x}' for x in CALL_RESOLUTION]
    return tuple(names)


FEATURE_NAMES = feature_names()


def edge_features(graph, source, target, attrs, bindings=None, access_maps=None):
    """Return a deterministic fixed-size semantic edge feature vector."""
    if bindings is None or access_maps is None:
        found_bindings, found_maps = _index_context(graph)
        bindings = found_bindings if bindings is None else bindings
        access_maps = found_maps if access_maps is None else access_maps
    values = {name: 0.0 for name in FEATURE_NAMES}

    position = (attrs.get('legacy_evidence') or {}).get('position')
    if attrs.get('relation_type') == 'operand':
        pos = _pair(position)
        values['operand_position_value'], values['operand_position_known'] = pos
        bound = bindings.get((source, target, position))
        access_map = access_maps.get(target)
        if access_map is not None and not access_map.get('conflict'):
            access_values, access_positions, parsed = access_map['values'], access_map['positions'], access_map['parsed']
            values['index_role_known'] = 1.0
            values['index_role'] = float(position in access_positions)
            if position in access_positions:
                input_ordinal = access_positions.index(position)
                values['index_input_is_dimension'] = float(input_ordinal < len(parsed['dimensions']))
                values['index_input_is_symbol'] = 1.0-values['index_input_is_dimension']
                pair = _pair(input_ordinal)
                values['index_input_ordinal_value'], values['index_input_ordinal_known'] = pair
            if not access_map['supported']:
                values['index_map_overflow'] = 1.0
        if bound is not None:
            if bound.get('supported'):
                values['index_role'] = 1.0
                values['index_map_supported'] = 1.0
                values['index_input_is_dimension'] = float(bound['input_kind'] == 'dimension')
                values['index_input_is_symbol'] = float(bound['input_kind'] == 'symbol')
                for name, number in [('index_input_ordinal', bound['input_ordinal']),
                                     ('index_input_count', bound['input_count']),
                                     ('index_symbol_count', bound['symbol_count']),
                                     ('index_result_rank', bound['result_count'])]:
                    pair = _pair(number)
                    values[name+'_value'], values[name+'_known'] = pair
                for i, coefficient in enumerate(bound['coefficients']):
                    pair = _pair(coefficient)
                    values[f'index_coeff_result_{i}_value'] = pair[0]

    if attrs.get('relation_type') == 'memory_access':
        info = attrs.get('access_indices') or {}
        parsed = parse_linear_affine_map(info.get('affine_map')) if info.get('status') == 'compiler_recorded' else None
        access_map = access_maps.get(source)
        binding_verified = bool(access_map is not None and not access_map.get('conflict')
                                and access_map['parsed'] == parsed
                                and access_map['values'] == info.get('ssa_operands')
                                and access_map['positions'] == info.get('operand_positions'))
        values['affine_map_known'] = float(binding_verified)
        is_bound = bool(binding_verified and access_map.get('supported')
                        and len(info.get('ssa_operands') or []) == len(parsed['inputs'])
                        and len(info.get('operand_positions') or []) == len(parsed['inputs']))
        values['affine_map_supported'] = float(is_bound)
        values['affine_map_overflow'] = float(parsed is not None and
            len(parsed['results']) > MAX_AFFINE_RESULTS)
        if is_bound:
            for name, number in [('affine_input_count', len(parsed['inputs'])),
                                 ('affine_result_count', len(parsed['results'])),
                                 ('affine_symbol_count', len(parsed['symbols']))]:
                pair = _pair(number)
                values[name+'_value'], values[name+'_known'] = pair
            for i, result in enumerate(parsed['results']):
                pair = _pair(result['offset'])
                values[f'affine_offset_{i}_value'] = pair[0]

    if attrs.get('relation_type') == 'depends_on':
        selected, exact = _distance(attrs)
        if selected is not None:
            low, high = selected.get('lower'), selected.get('upper')
            for name, value in [('dependence_distance_lower', low), ('dependence_distance_upper', high)]:
                pair = _pair(value)
                values[name+'_value'], values[name+'_known'] = pair
            values['dependence_bounds_known'] = float(
                _pair(low)[1] == 1.0 and _pair(high)[1] == 1.0)
        pair = _pair(exact)
        values['dependence_exact_value'], values['dependence_exact_known'] = pair
        pair = _pair(attrs.get('dependence_depth'))
        values['dependence_depth_value'], values['dependence_depth_known'] = pair
        loop = attrs.get('carrying_loop')
        values['carrying_loop_known'] = float(loop in graph)
        values['carrying_loop_is_action'] = float(loop in graph and bool(graph.nodes[loop].get('action_id')))
        if loop in graph:
            pair = _pair(graph.nodes[loop].get('loop_depth'))
            values['carrying_loop_depth_value'], values['carrying_loop_depth_known'] = pair
        status = attrs.get('carrying_loop_status')
        status = status if status in CARRYING_STATUS[:-1] else 'unknown'
        values.update(dict(zip([f'carrying_status_{x}' for x in CARRYING_STATUS], _one_hot(status, CARRYING_STATUS))))

    if attrs.get('relation_type') == 'loop_carried':
        role = attrs.get('forwarding_role')
        role = role if role in FORWARDING_ROLES[:-1] else 'unknown'
        values.update(dict(zip([f'forwarding_role_{x}' for x in FORWARDING_ROLES], _one_hot(role, FORWARDING_ROLES))))
        state = attrs.get('state_kind')
        state = state if state in STATE_KINDS[:-1] else 'unknown'
        values.update(dict(zip([f'state_kind_{x}' for x in STATE_KINDS], _one_hot(state, STATE_KINDS))))
        values['forwarding_evidence_origin_verified'] = float(attrs.get('evidence_origin') == 'verified_MLIR_region_SSA')
        loop = attrs.get('carrying_loop')
        if loop in graph:
            pair = _pair(graph.nodes[loop].get('loop_depth'))
            values['carrying_loop_depth_value'], values['carrying_loop_depth_known'] = pair

    mapping = attrs.get('view_mapping')
    if isinstance(mapping, dict):
        kind = mapping.get('operation_kind')
        kind = kind if kind in VIEW_KINDS[:-2] else 'other' if kind else 'unknown'
        values.update(dict(zip([f'view_kind_{x}' for x in VIEW_KINDS], _one_hot(kind, VIEW_KINDS))))
        semantics = mapping.get('offset_semantics')
        semantics = semantics if semantics in OFFSET_SEMANTICS[:-1] else 'unknown'
        values.update(dict(zip([f'view_offset_semantics_{x}' for x in OFFSET_SEMANTICS], _one_hot(semantics, OFFSET_SEMANTICS))))
        source_shape, result_shape = mapping.get('source_shape'), mapping.get('result_shape')
        for prefix, shape in (('view_source_rank', source_shape), ('view_result_rank', result_shape)):
            pair = _pair(len(shape) if isinstance(shape, list) else None)
            values[prefix+'_value'], values[prefix+'_known'] = pair
        arrays = [mapping.get(name) for name in ('offsets', 'sizes', 'strides')]
        max_rank = max((len(v) for v in arrays if isinstance(v, list)), default=0)
        values['view_dimension_overflow'] = float(max_rank > MAX_VIEW_DIMS)
        for i in range(min(max_rank, MAX_VIEW_DIMS)):
            for what, array in zip(('offset', 'size', 'stride'), arrays):
                item = array[i] if isinstance(array, list) and i < len(array) else None
                # Only MLIR's kDynamic sentinel is missing. Other signed values
                # are retained as reported compiler attributes.
                if type(item) is not int or item == -(1 << 63):
                    item = None
                pair = _pair(item)
                values[f'view_{what}_{i}_value'], values[f'view_{what}_{i}_known'] = pair

    relation = attrs.get('relation_type')
    if relation in ('calls', 'actual_to_formal', 'returns_to'):
        status = _call_resolution(graph, attrs)
        values.update(dict(zip([f'call_resolution_{x}' for x in CALL_RESOLUTION], _one_hot(status, CALL_RESOLUTION))))
    return [values[name] for name in FEATURE_NAMES]


def semantic_metadata():
    return {'schema': 'salt-hls-semantic-edge-features-v2', 'feature_names': list(FEATURE_NAMES),
            'signed_log_scale': SIGNED_SCALE, 'max_affine_inputs': MAX_AFFINE_INPUTS,
            'max_affine_results': MAX_AFFINE_RESULTS, 'max_view_dimensions': MAX_VIEW_DIMS,
            'affine_parser': 'additive integer-linear dN/sN expressions with exact SSA operand binding; arbitrary verified input count'}


def action_edge_summary(graph, action_node, action_target, semantic_rows, edge_records):
    """Mean-pool supported compiler-semantic edge rows relevant to one action.

    Loop scope comes from existing `loop_context`; array scope starts from the
    compiler-bound memory object and its directly observed access operations.
    This is a compact neural summary, not a new compiler query or new graph edge.
    """
    if action_target not in graph:
        return [0.0] * len(FEATURE_NAMES), False
    target_attrs = graph.nodes[action_target]
    related_ops = set()
    if target_attrs.get('category') == 'operation' and target_attrs.get('operation_family') == 'loop':
        related_ops = {node for node, attrs in graph.nodes(data=True)
                       if attrs.get('category') == 'operation'
                       and (node == action_target or action_target in attrs.get('loop_context', []))}
    elif target_attrs.get('category') == 'memory_object':
        for u, v, attrs in graph.edges(data=True):
            if attrs.get('relation_type') == 'memory_access' and action_target in (u, v):
                related_ops.add(u if graph.nodes[u].get('category') == 'operation' else v)
    selected = []
    witnesses = []
    for row, (u, v, key, attrs) in zip(semantic_rows, edge_records):
        rel = attrs.get('relation_type')
        incident = action_target in (u, v) or bool(related_ops.intersection((u, v)))
        carried = attrs.get('carrying_loop') == action_target
        if (incident or carried) and any(row):
            selected.append(row)
            witnesses.append(str(key))
    if not selected:
        return [0.0] * len(FEATURE_NAMES), False, []
    return ([sum(row[i] for row in selected) / len(selected) for i in range(len(FEATURE_NAMES))],
            True, sorted(set(witnesses)))
