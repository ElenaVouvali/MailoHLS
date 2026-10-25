"""Complete configurations, legal neighborhoods and factorial attribution."""
from __future__ import annotations
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import replace, asdict
from itertools import product
from pathlib import Path
from LLM_branch.common import directive_domains as legacy, mailohls_contract as mc
from LLM_branch.common.directive_domains import tcad
ROOT = Path(__file__).resolve().parents[2]
RESOURCE_KEYS = mc.RESOURCE_KEYS
PART = "xczu7ev-ffvc1156-2-e"
CAPACITIES = dict(mc.DEVICE_RESOURCES[PART])
CLOCK_NS = 10.0
HLS_VERSION = "2021.1"
SETTINGS = [
    "config_array_partition -complete_threshold 0 -throughput_driven off",
    "config_compile -pipeline_loops 0"
]
SPECS = {
    'covariance': {
        'source': 'evaluation_set/covariance/kernel.cpp',
        'metadata': 'evaluation_set/covariance/kernel_info.txt',
        'top': 'kernel_covariance'
    },
    'GAN': {
        'source': 'evaluation_set/GAN/network.cpp',
        'metadata': 'evaluation_set/GAN/code_to_memory_outputs/meta/kernel_info.txt',
        'top': 'forward_propagation',
        'compiler_symbol': '_Z19forward_propagationPfS_'
    },
    'Kalman': {
        'source': 'evaluation_set/serrano-kalman-filter/kernel.cpp',
        'metadata': 'evaluation_set/serrano-kalman-filter/kernel_info.txt',
        'top': 'krnl_KALMAN',
        'relations': 'GNN_branch/MLIR_graphs/serrano-kalman-filter.gexf',
        'relation_metadata': 'Data/ApplicationDataset/serrano-kalman-filter/kernel_info.txt',
        'relation_source': 'Data/ApplicationDataset/serrano-kalman-filter/kernel.cpp'
    }
}

CONTEXT = {
    'device': 'ZCU104',
    'part': PART,
    'requested_clock_ns': CLOCK_NS,
    'resource_budget': CAPACITIES,
    'device_capacities': CAPACITIES,
    'objective': 'ADP',
    'area_definition': 'mean of four device-capacity utilization percentages',
    'adp_definition': 'latency_cycles * area_proxy',
    'hls_version': HLS_VERSION,
    'flow_target': 'vivado',
    'synthesis_settings': SETTINGS
}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def display_path(path):
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)

def repo_path(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

ASSIGNMENT = re.compile('auto\\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_(L[1-9][0-9]*)\\}\\s*=\\s*(\\S+)', re.I)

FIELD = re.compile('AUTO\\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_(L[1-9][0-9]*)\\}')

KINDS = legacy.KINDS

def field_key(kind, label):
    return f'AUTO{{_{kind}_{label}}}'

def ordered_fields(actions):
    return [field_key(kind, label) for (label, action) in actions.items() for kind in KINDS[action.kind]]

def parse_assignment(text):
    values = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = ASSIGNMENT.fullmatch(line.strip())
        if not match:
            raise ValueError(f'Malformed directive assignment: {line!r}')
        (kind, label, rhs) = match.groups()
        lhs = field_key(kind.upper(), label.upper())
        if lhs in values:
            raise ValueError(f'Duplicate directive field: {lhs}')
        rhs = rhs.lower()
        if kind.upper() != 'ARRAY_T' and (not re.fullmatch('0|[1-9][0-9]*', rhs)):
            raise ValueError(f'Noncanonical integer: {rhs!r}')
        values[lhs] = rhs
    if not values:
        raise ValueError('Empty directive assignment')
    mc.validate_directive_assignments(values)
    return values

def serialize(values, actions):
    expected = ordered_fields(actions)
    if set(values) != set(expected):
        raise ValueError('Incomplete assignment or action/schema mismatch')
    return '\n'.join((f'auto{{{lhs[5:-1]}}} = {values[lhs]}' for lhs in expected))

def validate(values, actions, domains):
    expected = set(ordered_fields(actions))
    if set(values) != expected or set(domains) != expected:
        raise ValueError(
            f'Incomplete/schema mismatch: missing={sorted(expected - set(values))}, extra={sorted(set(values) - expected)}'
        )
    mc.validate_directive_assignments(values)
    violations = {
        lhs: {'value': rhs, 'allowed': domains[lhs]} for (lhs, rhs) in values.items() if rhs not in domains[lhs]
    }
    if violations:
        raise DomainViolation(violations)

class DomainViolation(ValueError):

    def __init__(self, violations):
        self.violations = violations
        super().__init__(
            'Configuration violates source-derived domains: ' + json.dumps(violations, sort_keys=True)
        )

def from_paper(row, actions):
    values = {}
    seen = set()
    for directive in row['directives']:
        label = directive['action_id']
        if label in seen or label not in actions or directive['kind'] != actions[label].kind:
            raise ValueError(f'Duplicate, unknown, or mismatched action: {label}')
        seen.add(label)
        if directive['kind'] == 'loop':
            data = {'PIPE': directive['pipeline_ii'], 'UNROLL': directive['unroll_factor']}
        else:
            data = {
                'ARRAY_T': directive['partition_type'],
                'ARRAY_F': directive['partition_factor'],
                'ARRAY_D': directive['partition_dim']
            }
        for (kind, value) in data.items():
            values[field_key(kind, label)] = str(value).lower()
    return parse_assignment(serialize(values, actions))

def signature(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def changed(base, candidate, actions):
    fields = [lhs for lhs in ordered_fields(actions) if base[lhs] != candidate[lhs]]
    labels = [label for label in actions if any((FIELD.fullmatch(lhs)[2] == label for lhs in fields))]
    return (labels, fields)

def legacy_domains(actions, directory):
    text = '\n'.join(ordered_fields(actions))
    return legacy.source_site_domains('source', text, directory)

def compiler_actions(kernel, actions):
    """Resolve loop sizes from authenticated MLIR, never upper-bound metadata.

    Caller first authenticates the graph through the relation loader.
    Array extents remain source metadata; inactive actions stay in assignments.
    """
    from .relations import _anchor
    graph = None
    if kernel['relations']['status'] == 'available':
        import networkx as nx
        graph = nx.read_gexf(repo_path(kernel['relations']['artifact']))
    result = dict(actions)
    evidence = []
    for (label, action) in actions.items():
        if action.kind != 'loop':
            continue
        node = None
        attributes = {}
        if graph is not None:
            try:
                node = _anchor(graph, label, 'loop')
                attributes = graph.nodes[node]
            except ValueError:
                pass
        count = int(attributes.get('trip_count', -1))
        resolved = count > 0 and int(attributes.get('trip_count_static', 0)) == 1 and (attributes.get('trip_count_source') == 'mlir_affine_analysis')
        result[label] = replace(action, trip_count=count if resolved else None)
        evidence.append(
            {
                'action_id': label,
                'metadata_size': action.trip_count,
                'verified_trip_count': count if resolved else None,
                'graph_node': node,
                'graph_sha256': kernel['relations'].get('artifact_hash'),
                'compiler_operation': attributes.get('full_text'),
                'status': 'static_agrees' if resolved and count == action.trip_count else 'static_metadata_mismatch' if resolved else 'unresolved_no_enabled_factor_proposals'
            }
        )
    kernel['domain_action_contract'] = {label: asdict(action) for (label, action) in result.items()}
    kernel['loop_domain_evidence'] = evidence
    return result

def recorded_actions(kernel, fallback):
    if 'domain_action_contract' not in kernel:
        return fallback
    return {
        label: legacy.SourceAction(
            kind=row['kind'],
            trip_count=row.get('trip_count'),
            dimensions=tuple((tuple(x) for x in row.get('dimensions', ())))
        ) for (label, row) in kernel['domain_action_contract'].items()
    }

def admissible(values, actions, native, policy):
    try:
        if policy == 'legacy':
            validate(values, actions, native)
        else:
            tcad.validate_assignments(values, actions)
        return True
    except (ValueError, KeyError):
        return False

def policy_record(values, baseline, actions, native, policy='tcad'):
    tcad_valid = admissible(values, actions, native, 'tcad')
    legacy_valid = admissible(values, actions, native, 'legacy')
    if policy == 'tcad':
        exceptions = tcad.validate_assignments(values, actions, baseline, allow_baseline_exceptions=True)
    elif policy == 'legacy':
        exceptions = {}
        for (label, action) in actions.items():
            keys = [field_key(k, label) for k in KINDS[action.kind]]
            if any((values[k] not in native[k] for k in keys)):
                if all((values[k] == baseline[k] for k in keys)):
                    exceptions[label] = {k: values[k] for k in keys}
                else:
                    raise ValueError('Modified action is outside selected legacy proposal policy')
    else:
        raise ValueError('Unknown domain policy')
    (_, fields) = changed(baseline, values, actions)
    return {
        'domain_policy': tcad.SOURCE_DOMAIN_POLICY if policy == 'tcad' else legacy.SOURCE_DOMAIN_POLICY,
        'classification': 'baseline_only_exception' if exceptions else policy,
        'legacy_admissible': legacy_valid,
        'tcad_admissible': tcad_valid,
        'baseline_exception_required': bool(exceptions),
        'baseline_exceptions': exceptions,
        'changed_values_new_under_tcad': {
            key: values[key] for key in fields if values[key] not in native[key] and values[key] in tcad.domains_from_actions(actions)[key]
        }
    }

def audit_domains(output, sft_path=None):
    """Audit all source metadata plus representative measured SFT labels."""
    (records, contracts) = ([], {})
    for info in sorted((ROOT / 'Data/ApplicationDataset').glob('*/kernel_info.txt')):
        try:
            actions = legacy.load_source_actions(info.parent.name, info.parent)
            old = legacy_domains(actions, info.parent)
            contracts[info.parent.name] = (actions, old)
        except (ValueError, OSError) as exc:
            records.append({'kernel': info.parent.name, 'status': 'unverifiable', 'reason': str(exc)})
            continue
        for (label, action) in actions.items():
            sizes = [('trip_count', action.trip_count)] if action.kind == 'loop' else list(action.dimensions)
            key = field_key('UNROLL' if action.kind == 'loop' else 'ARRAY_F', label)
            old_values = {int(x) for x in old[key] if int(x) > 0}
            for (dimension, size) in sizes:
                new = set(tcad.factor_values(size)) if size is not None else set()
                records.append(
                    {
                        'kernel': info.parent.name,
                        'metadata_sha256': sha256(info),
                        'action_id': label,
                        'kind': action.kind,
                        'dimension': dimension,
                        'size': size,
                        'legacy_factors': sorted(old_values),
                        'tcad_factors': sorted(new),
                        'legacy_count': len(old_values),
                        'tcad_count': len(new),
                        'added': sorted(new - old_values),
                        'removed': sorted(old_values - new),
                        'new_non_power_of_two_divisors': sorted((x for x in new - old_values if x & x - 1 and size % x == 0)),
                        'legacy_values_exceeding_extent': sorted((x for x in old_values if size is not None and x > size))
                    }
                )
    datasets = {}
    path = Path(sft_path or ROOT / 'artifacts/llm/mailohls_sft.jsonl')
    if path.is_file():
        (counts, cache, per_kernel, examples) = (Counter(), {}, {}, [])
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                (name, target) = (row.get('kernel_name'), row.get('target'))
                counts['rows'] += 1
                key = (name, target)
                if key not in cache:
                    if name not in contracts or not isinstance(target, str):
                        result = (False, False, 'unverifiable', 'missing source metadata/target')
                    else:
                        (actions, old) = contracts[name]
                        try:
                            values = parse_assignment(target)
                            (a, b) = (admissible(values, actions, old, 'legacy'), admissible(values, actions, old, 'tcad'))
                            result = (a, b, 'verified', None)
                            if not b:
                                try:
                                    tcad.validate_assignments(values, actions)
                                except ValueError as exc:
                                    result = (a, b, 'verified', str(exc))
                        except ValueError as exc:
                            result = (False, False, 'verified', str(exc))
                    cache[key] = result
                (a, b, state, reason) = cache[key]
                counts['legacy_admissible'] += a
                counts['tcad_admissible'] += b
                counts['tcad_invalid'] += not b and state != 'unverifiable'
                counts['unverifiable'] += state == 'unverifiable'
                per_kernel.setdefault(name, Counter()).update(
                    rows=1,
                    legacy_admissible=int(a),
                    tcad_admissible=int(b),
                    unverifiable=int(state == 'unverifiable')
                )
                if not b and len(examples) < 12:
                    examples.append(
                        {
                            'kernel': name,
                            'target': target,
                            'reason': reason or 'not representable under source-aware tcad'
                        }
                    )
        datasets['sft'] = {
            **dict(counts),
            'path': str(path),
            'sha256': sha256(path),
            'distinct_configurations': len(cache),
            'legacy_fraction_all_rows': counts['legacy_admissible'] / counts['rows'],
            'tcad_fraction_all_rows': counts['tcad_admissible'] / counts['rows'],
            'per_kernel': {key: dict(value) for (key, value) in per_kernel.items()},
            'invalid_examples': examples,
            'scope': 'All rows of existing measured/preprocessed SFT corpus; QoR fields are never read for policy construction.'
        }
    else:
        datasets['sft'] = {'status': 'unavailable', 'path': str(path)}
    paper = ROOT / 'paper_plots/SALT_experiments.jsonl'
    total = Counter()
    per_kernel = {}
    paper_contracts = {}
    if paper.is_file():
        for row in read_jsonl(paper):
            if not row.get('directives'):
                continue
            total['rows'] += 1
            name = row['kernel']
            resolved = 'covariance' if name.lower() == 'covariance' else name
            try:
                if resolved not in paper_contracts:
                    if resolved in SPECS:
                        directory = repo_path(SPECS[resolved]['metadata']).parent
                    else:
                        root = ROOT / 'evaluation_set' / resolved
                        available = [
                            p for p in (root / 'kernel_info.txt', root / 'code_to_memory_outputs/meta/kernel_info.txt') if p.is_file()
                        ]
                        if not available:
                            raise ValueError('No exact source-directory metadata mapping')
                        if len({sha256(p) for p in available}) > 1:
                            raise ValueError('Conflicting evaluation metadata')
                        directory = available[0].parent
                    actions = legacy.load_source_actions(resolved, directory)
                    paper_contracts[resolved] = (actions, legacy_domains(actions, directory))
                (actions, old) = paper_contracts[resolved]
                values = from_paper(row, actions)
                (a, b) = (admissible(values, actions, old, 'legacy'), admissible(values, actions, old, 'tcad'))
                total.update(legacy_admissible=int(a), tcad_admissible=int(b))
                per_kernel.setdefault(name, Counter()).update(rows=1, legacy_admissible=int(a), tcad_admissible=int(b))
            except (ValueError, KeyError, OSError):
                total['unverifiable'] += 1
        datasets['paper_complete_configurations'] = {
            **dict(total),
            'path': str(paper),
            'sha256': sha256(paper),
            'per_kernel': {k: dict(v) for (k, v) in per_kernel.items()},
            'scope': 'All nonempty complete directive rows against current matching evaluation-source metadata; missing/conflicting mappings counted unverifiable. This does not authenticate historical source snapshots or reuse their QoR.'
        }
    for dataset in datasets.values():
        if dataset.get('rows'):
            verified = dataset['rows'] - dataset.get('unverifiable', 0)
            dataset['legacy_fraction_all_rows'] = dataset.get('legacy_admissible', 0) / dataset['rows']
            dataset['tcad_fraction_all_rows'] = dataset.get('tcad_admissible', 0) / dataset['rows']
            dataset['legacy_fraction_verified_rows'] = dataset.get('legacy_admissible', 0) / verified if verified else None
            dataset['tcad_fraction_verified_rows'] = dataset.get('tcad_admissible', 0) / verified if verified else None
    result = {
        'policy': tcad.SOURCE_DOMAIN_POLICY,
        'legacy_policy': legacy.SOURCE_DOMAIN_POLICY,
        'definition': 'powers of two <= n UNION integer divisors >=2; no independent caps',
        'examples': {str(n): list(tcad.factor_values(n)) for n in (30, 50, 32, 1)},
        'metadata_records': records,
        'datasets': datasets,
        'summary': {
            'legacy_factor_candidates_per_action_dimension': sum((r.get('legacy_count', 0) for r in records)),
            'tcad_factor_candidates_per_action_dimension': sum((r.get('tcad_count', 0) for r in records)),
            'added_non_power_divisor_candidates': sum((len(r.get('new_non_power_of_two_divisors', [])) for r in records))
        },
        'manuscript': '05__Detailed_Design.tex not found locally; exact formula supplied in task used. No manuscript edits.'
    }
    write_json(output, result)
    return result

def _tie(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()

def _options(label, action, baseline, native, policy):
    kinds = KINDS[action.kind]
    keys = [field_key(kind, label) for kind in kinds]
    tuples = tcad.action_tuples(action) if policy == 'tcad' else [
        t for t in product(*(native[k] for k in keys)) if mc._valid_action_values(dict(zip(kinds, t)))
    ]
    result = []
    for values in tuples:
        update = {key: value for (key, value) in zip(keys, values) if baseline[key] != value}
        if not update:
            continue
        changed_kinds = [FIELD.fullmatch(key)[1] for key in update]
        family = changed_kinds[0] if len(changed_kinds) == 1 else 'tuple'
        distance = sum(
            (
                1 if kind == 'ARRAY_T' else abs(math.log2(max(1, int(value)) / max(1, int(baseline[key])))) for (key, value) in update.items() for kind in [FIELD.fullmatch(key)[1]]
            )
        )
        decrease = sum(
            (
                int(value) < int(baseline[key]) for (key, value) in update.items() if FIELD.fullmatch(key)[1] in {'UNROLL', 'ARRAY_F'}
            )
        )
        result.append(
            {
                'action': label,
                'updates': update,
                'family': family,
                'distance': distance,
                'decrease': decrease,
                'field_count': len(update)
            }
        )
    return result

def _nearest(options, family, seed):
    candidates = [r for r in options if r['family'] == family]
    return min(candidates, key=lambda r: (r['decrease'], r['distance'], _tie(seed, str(r['updates'])))) if candidates else None

def select_design(name, baseline, actions, native, relations, inactive=(), policy='tcad', seed=123):
    options = {
        label: _options(label, action, baseline, native, policy) for (label, action) in actions.items() if label not in inactive
    }
    pool = [
        {**option, 'consideration': 'single_action', 'selected': False} for values in options.values() for option in values
    ]
    independent = []
    for family in ['UNROLL', 'ARRAY_F', 'PIPE', 'ARRAY_T'] * 4:
        eligible = [
            r for values in options.values() for r in values if r['family'] == family and r['action'] not in {x['action'] for x in independent}
        ]
        if not eligible:
            continue
        chosen = min(
            eligible,
            key=lambda r: (
                r['decrease'],
                r['field_count'],
                r['distance'],
                _tie(seed, r['action'] + str(r['updates']))
            )
        )
        independent.append(chosen)
        if len(independent) == 4:
            break
        paired = []
    for relation in relations:
        (i, j) = (relation['action_i'], relation['action_j'])
        if i == j or i not in options or j not in options:
            continue
        if {actions[i].kind, actions[j].kind} == {'array', 'loop'}:
            loop = i if actions[i].kind == 'loop' else j
            array = j if loop == i else i
            (left, right) = (_nearest(options[loop], 'UNROLL', seed), _nearest(options[array], 'ARRAY_F', seed))
            pair_type = 'loop_unroll_array_factor'
        elif actions[i].kind == actions[j].kind == 'loop':
            (left, right) = (_nearest(options[i], 'UNROLL', seed), _nearest(options[j], 'UNROLL', seed))
            pair_type = 'loop_unroll_loop_unroll'
        else:
            continue
        if not left or not right:
            pool.append(
                {
                    'consideration': 'pair',
                    'actions': [i, j],
                    'selected': False,
                    'reason': 'No supported numeric alternative at both endpoints'
                }
            )
            continue
        row = {
            'left': left,
            'right': right,
            'relation': relation,
            'pair_type': pair_type,
            'distance': left['distance'] + right['distance']
        }
        paired.append(row)
        pool.append(
            {
                'consideration': 'pair',
                **row,
                'selected': False,
                'eligible_verified': relation['verified_relation_class'] in {'direct_verified', 'indirect_verified'}
            }
        )
    structural = []
    eligible = [
        r for r in paired if r['relation']['verified_relation_class'] in {'direct_verified', 'indirect_verified'}
    ]
    while eligible and len(structural) < 2:
        used = {m['action'] for row in structural for m in (row['left'], row['right'])}
        chosen = min(
            eligible,
            key=lambda r: (
                r['relation']['verified_relation_class'] != 'direct_verified',
                r['pair_type'] != 'loop_unroll_array_factor',
                sum((m['action'] in used for m in (r['left'], r['right']))),
                r['distance'],
                _tie(seed, str((r['left']['updates'], r['right']['updates'])))
            )
        )
        structural.append(chosen)
        eligible.remove(chosen)
    control = None
    controls = [
        r for r in paired if r['relation']['verified_relation_class'] not in {'direct_verified', 'indirect_verified'}
    ]
    if structural:
        reference = structural[0]
        eligible = [
            r for r in controls if r['pair_type'] == reference['pair_type'] and (r['left']['field_count'], r['right']['field_count']) == (reference['left']['field_count'], reference['right']['field_count'])
        ]

        def score(row):
            new_flags = lambda r: sum(
                (
                    value not in native[key] for m in (r['left'], r['right']) for (key, value) in m['updates'].items()
                )
            )
            shared = len(
                {m['action'] for m in (row['left'], row['right'])} & {m['action'] for m in (reference['left'], reference['right'])}
            )
            return (
                abs(new_flags(row) - new_flags(reference)),
                row['relation']['verified_relation_class'] != 'no_evidenced_relation',
                -shared,
                abs(row['left']['distance'] - reference['left']['distance']) + abs(row['right']['distance'] - reference['right']['distance']),
                _tie(seed, str((row['left']['updates'], row['right']['updates'])))
            )
        if eligible:
            control = min(eligible, key=score)
            control = {
                **control,
                'match_score': list(score(control)[:-1]),
                'control_label': 'no_verified_relation',
                'matching_note': 'Same action types/field counts; source-derived factor magnitudes and policy novelty matched as closely as available. Array extents may differ. No independence claim.'
            }
    factorials = []
    for (index, row) in enumerate(structural, 1):
        factorials.append({'id': f'{name}/structural-{index}', 'group': 'structural', **row})
    if control:
        factorials.append({'id': f'{name}/control', 'group': 'control', **control})
    configs = {}

    def record(values, membership):
        policy_info = policy_record(values, baseline, actions, native, policy)
        digest = signature(values)
        if digest not in configs:
            (labels, fields) = changed(baseline, values, actions)
            configs[digest] = {
                'kernel': name,
                'candidate_id': f'{name.lower()}-{digest[:16]}',
                'canonical_signature': digest,
                'assignment': values,
                'target': serialize(values, actions),
                'candidate_group': 'baseline' if not labels else 'refinement',
                'changed_action_ids': labels,
                'changed_directive_fields': fields,
                'changed_action_count': len(labels),
                'changed_field_count': len(fields),
                'policy': policy_info,
                'memberships': [],
                'functional_validation': 'not_tested',
                'static_validation': 'pragma-only materialization; not functional equivalence'
            }
        configs[digest]['memberships'].append(membership)
        if configs[digest]['changed_action_count'] == 1 and membership['group'] in {'structural', 'control'}:
            configs[digest]['memberships'].append(
                {
                    'group': 'independent',
                    'origin': 'factorial_single',
                    'factorial': membership['factorial'],
                    'cell': membership['cell']
                }
            )
        return configs[digest]['candidate_id']
    record(dict(baseline), {'group': 'baseline'})
    for row in independent:
        record({**baseline, **row['updates']}, {'group': 'independent', 'family': row['family']})
    for experiment in factorials:
        (left, right) = (experiment['left'], experiment['right'])
        if left['action'] == right['action']:
            raise ValueError('Paired edits must affect distinct sites')
        experiment['cells'] = {}
        for (cell, update) in [
            ('d0', {}),
            ('dA', left['updates']),
            ('dB', right['updates']),
            ('dAB', {**left['updates'], **right['updates']})
        ]:
            values = {**baseline, **update}
            (labels, fields) = changed(baseline, values, actions)
            if len(labels) != {'d0': 0, 'dA': 1, 'dB': 1, 'dAB': 2}[cell] or set(fields) != set(update):
                raise ValueError('Wrong action/field distance')
            experiment['cells'][cell] = record(values, {'group': experiment['group'], 'factorial': experiment['id'], 'cell': cell})
    selected_updates = {str(row['updates']) for row in independent}
    selected_updates.update((str(row[side]['updates']) for row in factorials for side in ('left', 'right')))
    selected_pairs = {str((row['left']['updates'], row['right']['updates'])) for row in factorials}
    for row in pool:
        row['selected'] = str(row.get('updates')) in selected_updates if row['consideration'] == 'single_action' else str((row.get('left', {}).get('updates'), row.get('right', {}).get('updates'))) in selected_pairs
        row.setdefault(
            'reason',
            'Included in frozen independent/factorial design' if row['selected'] else 'Outside deterministic diversity/pair quota; no QoR used'
        )
        row['selection_policy'] = 'QoR-blind; nearest supported larger numeric factor (fallback smaller), action-family diversity; verified UF priority; seed-123 SHA ties'
    limitations = []
    if len(structural) < 2:
        limitations.append(f'Only {len(structural)} suitable verified structural pairs available')
    if control is None:
        limitations.append('No type/field-matched control pair available')
    return (list(configs.values()), factorials, pool, limitations)

def analyze(measurements):
    """Every cell must be successful, feasible, authenticated and comparable."""
    names = ('d0', 'dA', 'dB', 'dAB')
    if any((not measurements.get(name) for name in names)):
        return {
            'G_joint': None,
            'G_joint_pct': None,
            'joint_better_than_each_single': None,
            'status': 'unavailable',
            'reason': 'missing measurements'
        }
    rows = [measurements[name] for name in names]
    if any(
        (
            row.get('authentication') not in {'authenticated_reuse', 'authenticated_new'} or row.get('synthesis_status') != 'success' or row.get('feasible') is not True or (row.get('ADP') is None) for row in rows
        )
    ):
        return {
            'G_joint': None,
            'G_joint_pct': None,
            'joint_better_than_each_single': None,
            'status': 'unavailable',
            'reason': 'unverified, failed, incomplete or infeasible measurement'
        }
    contexts = {row.get('comparison_context_id') for row in rows}
    if None in contexts or len(contexts) != 1:
        return {
            'G_joint': None,
            'G_joint_pct': None,
            'joint_better_than_each_single': None,
            'status': 'unavailable',
            'reason': 'mismatched evaluation contexts'
        }
    (j0, jx, jy, jxy) = (row['ADP'] for row in rows)
    if j0 <= 0 or any((not math.isfinite(j) for j in (j0, jx, jy, jxy))):
        return {
            'G_joint': None,
            'G_joint_pct': None,
            'joint_better_than_each_single': None,
            'status': 'unavailable',
            'reason': 'invalid objective values'
        }
    interaction = jxy - jx - jy + j0
    epsilon = 32 * math.ulp(float(max((abs(j) for j in (j0, jx, jy, jxy))))) if hasattr(math, 'ulp') else 1e-10
    if abs(interaction) <= epsilon:
        interaction = 0.0
    gain = (min(j0, jx, jy) - jxy) / j0
    return {
        'G_joint': gain,
        'G_joint_pct': 100 * gain,
        'joint_better_than_each_single': gain > 0,
        'status': 'measured',
        'interaction_ADP': interaction,
        'interaction_pct_of_baseline': 100 * interaction / j0,
        'interaction_class': 'favorable_nonadditive' if interaction < 0 else 'unfavorable_nonadditive' if interaction > 0 else 'additive',
        'improvement_pct': {name: 100 * (j0 - measurements[name]['ADP']) / j0 for name in names},
        'ADP_deltas': {name: measurements[name]['ADP'] - j0 for name in names},
        'latency_deltas': {name: measurements[name]['latency_cycles'] - rows[0]['latency_cycles'] for name in names},
        'resource_deltas': {
            name: {
                key: measurements[name]['resources'][key] - rows[0]['resources'][key] for key in RESOURCE_KEYS
            } for name in names
        },
        'interpretation_limit': 'Finite-difference ADP attribution, not proof of a physical hardware mechanism.'
    }
