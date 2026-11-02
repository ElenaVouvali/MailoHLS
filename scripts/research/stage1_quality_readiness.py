#!/usr/bin/env python3
"""Read-only, fail-closed readiness checks for MailoHLS Stage-1 quality SFT.

No GPU access, model loading, package installation, filesystem cleanup,
training, or synthesis. Run from a dedicated stage1-refinement worktree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PKG = Path('experiments/stage1_quality_sft')
EXPECTED_FILES = (
    '__init__.py', '__main__.py', 'core.py', 'build.py', 'support.py',
    'observability.py', 'training.py', 'evaluate.py', 'inference.py',
    'select_checkpoints.py', 'gpu_preflight.py',
)
DATASET = Path('artifacts/llm/mailohls_sft.jsonl')
SPLIT = Path('mailohls_runs/mailohls_final_family_split_s123.json')
DATASET_SHA = 'afdf4af8e6defc559871a6ec92185e6321313979ff3f1495806e5b037d5ab820'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(root: Path, *args: str) -> tuple[int, str]:
    result = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True,
                            check=False)
    return result.returncode, result.stdout.strip()


def inspect(root: Path, prepared: Path | None) -> dict:
    report: dict = {'schema': 'mailohls-stage1-quality-readiness-v1',
                    'root': str(root), 'blocked': [], 'warnings': [], 'checks': {}}
    def block(label: str, info: object) -> None:
        report['blocked'].append({'name': label, 'detail': info})
    def warn(label: str, info: object) -> None:
        report['warnings'].append({'name': label, 'detail': info})

    rc, branch = git(root, 'branch', '--show-current')
    report['checks']['branch'] = branch if rc == 0 else '<unavailable>'
    if branch != 'research/stage1-refinement':
        block('wrong_branch', 'Must run in the research/stage1-refinement worktree')
    rc, head = git(root, 'rev-parse', 'HEAD')
    report['checks']['head'] = head if rc == 0 else '<unavailable>'
    rc, status = git(root, 'status', '--short')
    report['checks']['uncommitted_entries'] = len(status.splitlines()) if rc == 0 and status else 0
    if rc != 0:
        block('git_status_failed', 'Could not inspect git status')

    missing = [p for p in EXPECTED_FILES if not (root / PKG / p).is_file()]
    report['checks']['quality_package_missing'] = missing
    if missing:
        block('package_incomplete', missing)
    test_file = root / 'LLM_branch/tests/test_stage1_quality_sft.py'
    if not test_file.is_file():
        block('missing_tests', str(test_file.relative_to(root)))

    for name, file in [('dataset', DATASET), ('family_split', SPLIT)]:
        p = root / file
        if not p.is_file():
            block(f'missing_{name}', str(file))
            continue
        digest = sha256(p)
        report['checks'][f'{name}_sha256'] = digest
        if name == 'dataset' and digest != DATASET_SHA:
            block('dataset_changed', 'Frozen GNOSIS dataset digest differs')
    split = root / SPLIT
    if split.is_file():
        try:
            data = json.loads(split.read_text())
            memberships = [set(map(int, data[f'{n}_jsonl_idx'])) for n in ('train', 'val', 'test')]
            counts = dict(zip(('train', 'val', 'test'), map(len, memberships)))
            report['checks']['split_membership_counts'] = counts
            if any(memberships[i] & memberships[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                block('split_overlap', 'Train/validation/test row IDs overlap')
        except (ValueError, KeyError, TypeError) as exc:
            block('invalid_split_manifest', str(exc))

    trainer = root / 'LLM_branch/train/train_SFT_xattn_new.py'
    wrapper = root / PKG / 'training.py'
    if trainer.is_file():
        source = trainer.read_text()
        report['checks']['production_reselects_objectives'] = 'select_objectives(' in source
        report['checks']['production_reaugments_budget'] = 'augment_rows_with_random_resource_budgets(' in source
        report['checks']['explicit_quality_sft_hook'] = 'stage1_quality_sft' in source or 'quality_sft_preselected' in source
        if not report['checks']['explicit_quality_sft_hook']:
            block('target_selection_not_proven',
                  'Production trainer still selects winners/budgets. Verify a real bypass before GPU training; '
                  'exporting JSONL alone is insufficient.')
    else:
        block('production_trainer_missing', str(trainer.relative_to(root)))
    if wrapper.is_file():
        w = wrapper.read_text()
        report['checks']['quality_wrapper_mentions_preselected'] = any(
            k in w for k in ('preselected', 'select_objectives', 'quality_sft_prepared'))
        if '.local-archive/' in w:
            warn('nonportable_training_path', 'Wrapper references a local archived path')
    build = root / PKG / 'build.py'
    if build.is_file():
        b = build.read_text()
        if '.local-archive/' in b or 'Path.home()' in b and 'retired-worktree' in b:
            block('nonportable_observability_path',
                  'Build script depends on a personal retired-worktree archive; use authenticated paths')

    if prepared is not None:
        report['checks']['prepared'] = str(prepared)
        if not prepared.is_dir():
            block('prepared_missing', str(prepared))
        else:
            files = sorted(p.name for p in prepared.iterdir() if p.is_file())
            report['checks']['prepared_files'] = files
            for candidate in ('manifest.json', 'protocol.json', 'preparation_manifest.json'):
                m = prepared / candidate
                if m.is_file():
                    try:
                        obj = json.loads(m.read_text())
                        report['checks']['manifest_name'] = candidate
                        report['checks']['manifest_sha256'] = sha256(m)
                        report['checks']['manifest_keys'] = sorted(obj)
                        if obj.get('schema') is None:
                            warn('missing_manifest_schema', candidate)
                    except (OSError, ValueError) as exc:
                        block('invalid_manifest', str(exc))
                    break
            else:
                block('manifest_missing', 'No manifest.json, protocol.json or preparation_manifest.json')
            if not any('train' in p and 'pool' in p for p in files):
                warn('training_candidate_pool_not_found', files)
            if not any('validation' in p and 'request' in p for p in files):
                warn('validation_request_manifest_not_found', files)

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--prepared', type=Path)
    ap.add_argument('--json-out', type=Path,
                    help='Optional NEW report file. Existing files are never overwritten.')
    ap.add_argument('--strict', action='store_true', help='Exit 2 if any blocking item exists')
    args = ap.parse_args()
    root = args.root.resolve()
    report = inspect(root, args.prepared.resolve() if args.prepared else None)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        with args.json_out.open('x') as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write('\n')
    return 2 if args.strict and report['blocked'] else 0


if __name__ == '__main__':
    sys.exit(main())
