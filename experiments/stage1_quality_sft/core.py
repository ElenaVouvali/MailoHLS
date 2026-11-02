"""Measured-only multi-reference targets and deterministic request sampling.

This module intentionally has no dependency on a model checkpoint.  QoR labels
are used only to construct targets from the selected TRAIN rows.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Mapping

BUDGET = ('budget_frac_bram_18k', 'budget_frac_dsp', 'budget_frac_ff', 'budget_frac_lut')
UTILIZATION = ('bram_util_%', 'dsp_util_%', 'ff_util_%', 'lut_util_%')


def assignment(text):
    """Pure parser matching the existing recovery assignment identity format."""
    return tuple(sorted(parse(text).items()))


def parse(text):
    pairs = re.findall(r'(auto\{_[A-Z_]+_L\d+\})\s*=\s*(\w+)', text, flags=re.IGNORECASE)
    result = {}
    for lhs, rhs in pairs:
        key = lhs.upper()
        if key in result:
            raise ValueError(f'Duplicate directive assignment: {key}')
        result[key] = rhs
    return result


def physical_context(row):
    return (row['kernel_name'], row.get('source_key'), row.get('source_file'),
            str(row['device']).lower(), round(float(row['clock_period']), 2))


def candidate_banks(rows):
    """Same exact duplicate/conflict policy as stage1_recovery.ambiguity."""
    from collections import defaultdict
    banks, repeats = defaultdict(dict), 0
    for row in rows:
        context, design = physical_context(row), assignment(row['target'])
        if not design:
            raise ValueError('Measured configuration has no directive assignment')
        qos = tuple(float(row[k]) for k in ('latency', *UTILIZATION))
        old = banks[context].get(design)
        if old is not None:
            if old['qos'] != qos:
                raise ValueError(f'Conflicting measured QoR for repeated design: {(context, design)}')
            repeats += 1
            continue
        banks[context][design] = dict(qos=qos, row_id=int(row['_jsonl_idx']))
    return banks, repeats


def near_optimal(request, bank, epsilon):
    limits = tuple(float(request[k]) for k in BUDGET)
    feasible = {design: receipt for design, receipt in bank.items()
        if all(receipt['qos'][i + 1] / 100. <= limits[i] + 1e-9 for i in range(4))}
    if not feasible:
        raise ValueError('Authenticated selected request has no feasible measured design')
    optimum = min(x['qos'][0] for x in feasible.values())
    if optimum <= 0:
        raise ValueError('Expected positive measured latency')
    accepted = {design: item for design, item in feasible.items()
        if item['qos'][0] <= (1. + epsilon) * optimum + 1e-12}
    return feasible, accepted, optimum

EPSILONS = (0.0, 0.005, 0.01)


def request_key(row: Mapping) -> tuple:
    """Identity includes the complete deployment and optimization context."""
    return (row['kernel_name'], row.get('source_key'), row.get('source_file'),
            str(row['device']).lower(), round(float(row['clock_period']), 2),
            tuple(round(float(row[k]), 8) for k in BUDGET),
            str(row.get('objective', row.get('obj_mode', 'PARETO_LATENCY'))))


def request_id(row: Mapping) -> str:
    packed = json.dumps(request_key(row), separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(packed.encode()).hexdigest()


def build_request(row: Mapping, bank: Mapping, epsilon: float = 0.01) -> dict:
    if epsilon not in EPSILONS:
        raise ValueError(f'epsilon must be one of {EPSILONS}')
    candidates = bank.get(physical_context(row), {})
    feasible, accepted, best = near_optimal(row, candidates, epsilon)
    # Assignment identity, not row order, defines exact tie breaking and sampling.
    choices = []
    for assignment_tuple, receipt in sorted(accepted.items()):
        choices.append(dict(row_id=int(receipt['row_id']),
                            assignment=dict(assignment_tuple),
                            assignment_id=hashlib.sha256(repr(assignment_tuple).encode()).hexdigest(),
                            latency=float(receipt['qos'][0])))
    feasible_choices = []
    for assignment_tuple, receipt in sorted(feasible.items()):
        feasible_choices.append(dict(row_id=int(receipt['row_id']),
            assignment=dict(assignment_tuple),
            assignment_id=hashlib.sha256(repr(assignment_tuple).encode()).hexdigest(),
            latency=float(receipt['qos'][0]),
            utilization={name: float(receipt['qos'][i + 1]) for i, name in enumerate(UTILIZATION)}))
    all_choices = []
    for assignment_tuple, receipt in sorted(candidates.items()):
        all_choices.append(dict(row_id=int(receipt['row_id']),
            assignment=dict(assignment_tuple),
            assignment_id=hashlib.sha256(repr(assignment_tuple).encode()).hexdigest(),
            latency=float(receipt['qos'][0]),
            utilization={name: float(receipt['qos'][i + 1]) for i, name in enumerate(UTILIZATION)},
            feasible=assignment_tuple in feasible))
    if not choices:
        raise ValueError('No measured feasible target')
    return dict(schema='stage1-quality-request-v1', request_id=request_id(row),
        request_identity=request_key(row), physical_context=physical_context(row),
        family=row.get('_family'),
        objective=str(row.get('objective', row.get('obj_mode', 'PARETO_LATENCY'))),
        budget={key: float(row[key]) for key in BUDGET}, epsilon=float(epsilon),
        best_latency=float(best), feasible_count=len(feasible), accepted_count=len(choices),
        accepted=choices, feasible_candidates=feasible_choices,
        measured_candidates=all_choices)


def candidate_pool_id(row: Mapping) -> str:
    packed = json.dumps(physical_context(row), separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(packed.encode()).hexdigest()


def candidate_pool_record(context, bank):
    identity = hashlib.sha256(json.dumps(context, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    candidates = []
    for design, receipt in sorted(bank.items()):
        candidates.append(dict(candidate_id=hashlib.sha256(repr(design).encode()).hexdigest(),
            row_id=int(receipt['row_id']), assignment=dict(design), latency=float(receipt['qos'][0]),
            utilization={name: float(receipt['qos'][i + 1]) for i, name in enumerate(UTILIZATION)}))
    return dict(candidate_pool_id=identity, physical_context=context, candidates=candidates)


def build_compact_request(row: Mapping, bank: Mapping) -> dict:
    """Request references a deduplicated pool; it does not copy measured QoR rows."""
    per_eps = {}
    for epsilon in EPSILONS:
        feasible, accepted, best = near_optimal(row, bank.get(physical_context(row), {}), epsilon)
        ids = sorted(hashlib.sha256(repr(design).encode()).hexdigest() for design in accepted)
        per_eps[str(epsilon)] = dict(accepted_candidate_ids=ids,
            feasible_count=len(feasible), accepted_count=len(ids))
    primary = per_eps['0.01']
    best_feasible, _, best_latency = near_optimal(row, bank.get(physical_context(row), {}), 0.)
    best_id = min((hashlib.sha256(repr(design).encode()).hexdigest(), design)
                  for design, receipt in best_feasible.items() if receipt['qos'][0] == best_latency)[0]
    return dict(schema='stage1-quality-request-v1', request_id=request_id(row),
        request_identity=request_key(row), family=row.get('_family'),
        physical_context=physical_context(row), candidate_pool_id=candidate_pool_id(row),
        objective=str(row.get('objective', row.get('obj_mode', 'PARETO_LATENCY'))),
        budget={key: float(row[key]) for key in BUDGET}, primary_epsilon=.01,
        best_latency=float(best_latency), best_candidate_id=best_id,
        feasible_count=primary['feasible_count'], accepted_count=primary['accepted_count'],
        sensitivity=per_eps)


def single_reference(request: Mapping) -> dict:
    """Choose the best-latency complete design; ties use authenticated identity."""
    return min(request['accepted'], key=lambda c: (float(c['latency']), c['assignment_id']))


def sample_target(request: Mapping, seed: int, exposure: int) -> dict:
    """Uniform deterministic choice from complete designs, stable across resume."""
    if exposure < 0:
        raise ValueError('exposure must be nonnegative')
    choices = request['accepted']
    if not choices:
        raise ValueError('Empty measured target set')
    digest = hashlib.sha256(
        f'{int(seed)}:{request["request_id"]}:{int(exposure)}'.encode()).digest()
    # Rejection sampling avoids modulo bias while retaining stateless resume.
    span = 1 << 64
    limit = span - span % len(choices)
    counter = 0
    while True:
        word = int.from_bytes(digest[:8], 'big') if counter == 0 else int.from_bytes(
            hashlib.sha256(digest + counter.to_bytes(8, 'big')).digest()[:8], 'big')
        if word < limit:
            return choices[word % len(choices)]
        counter += 1


def build_banks(raw_rows: Iterable[Mapping]):
    return candidate_banks(raw_rows)


def validate_complete_assignment(row: Mapping, registry: Mapping, candidates, normalize, parse):
    """Validate every assignment field against the existing source domain."""
    kernel = normalize(row['kernel_name'])
    if kernel not in registry:
        raise ValueError(f'No source/action registry for {row["kernel_name"]}')
    expected = {str(k).upper() for k in registry[kernel]}
    actual = {str(k).upper(): str(v) for k, v in parse(row['target']).items()}
    if set(actual) != expected:
        raise ValueError(f'Incomplete/extra directive fields for {kernel}: '
                         f'missing={sorted(expected-set(actual))}, extra={sorted(set(actual)-expected)}')
    for field, value in actual.items():
        legal = registry[kernel].get(field, [])
        if value not in legal:
            raise ValueError(f'Illegal measured directive: {kernel}/{field}={value}')
    # The shared production contract defines legal complete loop/array tuples.
    # This is equivalent to sequential constrained candidate filtering for a
    # complete assignment, without enumerating Cartesian companion domains.
    from LLM_branch.common.mailohls_contract import validate_directive_assignments
    validate_directive_assignments(actual)
    return actual


def observability_record(row: Mapping, source_fields: Iterable[str], registry: Mapping,
                         normalize, parse) -> dict:
    """Classify target fields as measured labels; source template proves marker coverage.

    Preprocessed GNΩSIS row provenance is authenticated by its dataset manifest.
    This deliberately does not classify unrelated graph actions as labels.
    """
    target = {str(k).upper(): str(v) for k, v in parse(row['target']).items()}
    source = {str(k).upper() for k in source_fields}
    if set(target) != source:
        raise ValueError(f'Source/action observability mismatch for {row["kernel_name"]}')
    return dict(kernel=row['kernel_name'], source_key=row.get('source_key'),
        action_count=len(source), actions=[dict(action=k, classification='directly_measured_label',
            provenance='authenticated GNΩSIS JSONL target row; exact source-marker identity')
            for k in sorted(source)])
