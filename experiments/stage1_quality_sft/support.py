"""CPU-only adapters around the repository's shared source/domain contracts."""
from __future__ import annotations

import re
from pathlib import Path
import json
import hashlib

from LLM_branch.common import mailohls_contract


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_kname(value):
    return re.sub(r'[-\s]+', '_', str(value).strip().lower()).replace('-', '_')


def family_id_from_kernel_name(value):
    return mailohls_contract.family_id_from_kernel_name(value)


def load_registry(path):
    payload = json.loads(Path(path).read_text())
    if 'kernels' in payload:
        if payload.get('schema') != 'mailohls-directive-domain-registry-v1':
            raise ValueError('Unsupported directive-domain registry schema')
        payload = payload['kernels']
    result = {}
    for kernel, sites in payload.items():
        key = normalize_kname(kernel)
        if key in result:
            raise ValueError(f'Duplicate normalized registry kernel: {key}')
        result[key] = {str(k).upper(): [str(v) for v in values]
                       for k, values in sites.items()}
    return result


def constrained_rhs_candidates(kernel, lhs, registry, chosen):
    key = normalize_kname(kernel)
    sites = registry[key]
    lhs = str(lhs).upper()
    original = list(sites[lhs])
    kind = re.search(r'_([A-Z]+)_L\d+\}', lhs).group(1)
    if kind in ('PIPE', 'UNROLL'):
        return original, original
    return original, mailohls_contract.filter_semantic_candidates(lhs, original, dict(chosen), sites)


def source_action_plan(source):
    result, seen = [], set()
    for match in re.finditer(r'auto\{_[A-Z0-9_]+_L\d+\}', source, re.IGNORECASE):
        lhs = match.group(0).upper()
        if lhs not in seen:
            seen.add(lhs)
            label = re.search(r'_L(\d+)\}', lhs).group(1)
            result.append((f'L{label}', lhs))
    if not result:
        raise ValueError('Source template has no action directive markers')
    return result


def source_fields(source):
    return [lhs for _, lhs in source_action_plan(source)]
