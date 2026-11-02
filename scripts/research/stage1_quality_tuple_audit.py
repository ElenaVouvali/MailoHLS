#!/usr/bin/env python3
"""CPU-only audit of measured candidate tuples in Stage-1 quality-SFT banks.

Fail-closed: unrecognized record shapes are rejected, not silently skipped.
The check is semantic tuple validity; measurement provenance must be audited
separately in experiments/stage1_quality_sft/observability.py.
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
from pathlib import Path

# Allow direct invocation as scripts/research/stage1_quality_tuple_audit.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from LLM_branch.common.mailohls_contract import validate_directive_assignments


def unpack(candidate: dict) -> dict[str, str]:
    for name in ('assignment', 'directives', 'target'):
        if name not in candidate:
            continue
        value = candidate[name]
        if isinstance(value, dict):
            return {str(k).strip().upper(): str(v).strip() for k, v in value.items()}
        if isinstance(value, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in value):
            return {str(k).strip().upper(): str(v).strip() for k, v in value}
        if isinstance(value, str):
            from re import fullmatch
            fields: dict[str, str] = {}
            for line in value.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = fullmatch(r'(AUTO\{_[A-Za-z0-9_]+_L\d+\})\s*=\s*(\S+)', line, flags=2)
                if not m:
                    raise ValueError(f'Unrecognized directive line: {line[:100]}')
                key = m.group(1).upper()
                if key in fields:
                    raise ValueError(f'Duplicate field: {key}')
                fields[key] = m.group(2)
            return fields
        raise ValueError(f'Unsupported candidate {name} shape: {type(value).__name__}')
    raise ValueError(f'Candidate contains no assignment/directives/target; keys={sorted(candidate)}')


def inspect_file(path: Path) -> dict:
    opener = gzip.open if path.suffix == '.gz' else open
    count = 0
    invalid: list[dict] = []
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('candidates'), list):
                raise ValueError(f'Expected pool candidates[] at {path}:{line_num}')
            for j, candidate in enumerate(row['candidates']):
                count += 1
                try:
                    values = unpack(candidate)
                    if not values:
                        raise ValueError('Empty directive assignment')
                    validate_directive_assignments(values)
                except (ValueError, KeyError, TypeError) as e:
                    if len(invalid) < 12:
                        invalid.append({'line': line_num, 'candidate_index': j, 'error': str(e)})
    return {'file': str(path), 'candidates_checked': count,
            'invalid_examples': invalid, 'valid': not invalid}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.prepared.glob('*candidate_pools.jsonl*'))
    if not paths:
        print('FAIL: no *candidate_pools.jsonl* files under prepared directory', file=sys.stderr)
        return 2
    results = []
    for path in paths:
        try:
            results.append(inspect_file(path))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            results.append({'file': str(path), 'valid': False, 'error': str(exc)})
    print(json.dumps(results, indent=2))
    return 2 if any(not x['valid'] for x in results) else 0


if __name__ == '__main__':
    sys.exit(main())
