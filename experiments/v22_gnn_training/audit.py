"""Authenticate the full GNΩSIS/V2.2 design-point correspondence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from GNN_branch import target_graph_v22 as overlay
from GNN_branch import mlir_graph_v2 as program_graph
from LLM_branch.common import directive_domains as domains
from Preprocessing.create_jsonl import (
    application_sources,
    build_examples,
    file_sha256,
    validate_preprocessing_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / 'mailohls_runs/mailohls_final_family_split_s123.json'
PREPROCESSED = ROOT / 'LLM_branch/Data/preprocessed_CSVS'
RAW_CSVS = ROOT / 'Data/CSVS'
APL_ROOT = ROOT / 'Data/ApplicationAPLMapping'
RELEASE_MANIFEST = ROOT / 'docs/graph-release-v22-rc3.json'
DEVICE_TO_PROFILE = {
    'xczu7ev-ffvc1156-2-e': 'zcu104',
    'xcu200-fsgd2104-2-e': 'u200',
}
EXPECTED_CLOCKS = (3.33, 5.0, 10.0)
TARGET_RE = re.compile(
    r'^\s*(AUTO\{_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*\})'
    r'\s*=\s*(.*?)\s*$', re.IGNORECASE
)


def _candidate_values(target: str) -> dict[str, str]:
    result = {}
    for line in target.splitlines():
        match = TARGET_RE.fullmatch(line)
        if not match:
            raise ValueError(f'Unrecognized serialized directive row: {line!r}')
        key, value = match.groups()
        key = key.upper()
        if key in result:
            raise ValueError(f'Duplicate serialized directive field: {key}')
        result[key] = value.strip().lower()
    return result


def _family_split(split: dict, row_count: int):
    index_to_split = {}
    for name in ('train', 'val', 'test'):
        indices = split.get(name + '_jsonl_idx')
        if not isinstance(indices, list):
            raise ValueError(f'Missing {name} row indices from family split')
        for index in indices:
            index = int(index)
            if index < 0 or index >= row_count or index in index_to_split:
                raise ValueError(f'Invalid or overlapping family split index: {index}')
            index_to_split[index] = name
    if set(index_to_split) != set(range(row_count)):
        raise ValueError('Seed-123 family split does not cover reconstructed dataset rows')
    kernel_to_split = {}
    for name in ('train', 'val', 'test'):
        for kernel in split.get(name + '_kernels', []):
            if kernel in kernel_to_split:
                raise ValueError(f'Kernel crosses split boundaries: {kernel}')
            kernel_to_split[kernel] = name
    return index_to_split, kernel_to_split


def audit_dataset(release_dir: Path, output: Path | None = None) -> dict:
    """Verify all rows, source/action mappings, exact graphs and split membership.

    Reconstructs the absent compact JSONL in memory from its tracked tables and
    manifests, then verifies that its canonical bytes match the split's SHA256.
    No training examples or source artifacts are rewritten.
    """
    release_dir = Path(release_dir)
    split = json.loads(SPLIT_PATH.read_text())
    selected_manifest_path = ROOT / 'artifacts/llm/mailohls_sft.manifest.json'
    selected_manifest = json.loads(selected_manifest_path.read_text())
    pre_manifest_path = validate_preprocessing_manifest(PREPROCESSED)
    pre_manifest = json.loads(pre_manifest_path.read_text())
    if file_sha256(pre_manifest_path) != selected_manifest['preprocessing_manifest_sha256']:
        raise ValueError('Tracked preprocessing manifest differs from the selected dataset contract')
    if selected_manifest['output_sha256'] != split['dataset_sha256']:
        raise ValueError('Family split and selected dataset manifest have different dataset hashes')

    examples, per_kernel, source_templates = build_examples(PREPROCESSED)
    dataset_hash = hashlib.sha256()
    for row in examples:
        dataset_hash.update((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())
    if dataset_hash.hexdigest() != split['dataset_sha256']:
        raise ValueError('Reconstructed GNΩSIS JSONL does not match seed-123 split dataset SHA256')
    if len(examples) != selected_manifest['examples'] or len(per_kernel) != selected_manifest['kernels']:
        raise ValueError('Reconstructed GNΩSIS table count differs from selected dataset manifest')

    index_split, kernel_split = _family_split(split, len(examples))
    row_counts_by_split = Counter()
    for index, row in enumerate(examples):
        membership = index_split[index]
        if kernel_split.get(row['kernel_name']) != membership:
            raise ValueError(f'JSONL index {index} has inconsistent kernel/family membership')
        row_counts_by_split[membership] += 1
    if set(kernel_split) != {row['kernel_name'] for row in examples}:
        raise ValueError('Family split kernel inventory differs from reconstructed dataset')

    registry = overlay.load_registry()
    release = json.loads((release_dir / 'release.json').read_text())
    source_manifest = json.loads(RELEASE_MANIFEST.read_text())
    if release.get('schema') != program_graph.SCHEMA_V22:
        raise ValueError('Unsupported extracted graph release manifest')
    archive_path = ROOT / source_manifest['archive']
    if overlay.file_hash(archive_path) != source_manifest['archive_sha256']:
        raise ValueError('Source graph archive differs from its published checksum')
    if (release.get('files_sha256') != source_manifest.get('files_sha256')
            or release.get('graph_count') != source_manifest.get('graph_count')):
        raise ValueError('Extracted graph archive identity differs from the repository release')
    canonical_dir = release_dir / 'canonical'
    if not canonical_dir.is_dir():
        raise FileNotFoundError(canonical_dir)
    manifest_by_kernel = {item['kernel']: item for item in per_kernel}
    raw_stats = {item['kernel']: item for item in pre_manifest['kernels']}
    graph_matches = {}
    source_actions_by_kernel = {}
    for kernel in sorted(manifest_by_kernel):
        info = manifest_by_kernel[kernel]
        source_path = ROOT / info['source']
        metadata_path = source_path.parent / 'kernel_info.txt'
        apl_path = APL_ROOT / f'{kernel}.txt'
        raw_csv = RAW_CSVS / f'{kernel}.csv'
        prep_csv = PREPROCESSED / f'preprocessed-{kernel}.csv'
        if not all(path.is_file() for path in (source_path, metadata_path, apl_path, raw_csv, prep_csv)):
            raise FileNotFoundError(f'Incomplete source/metadata/APL/CSV chain for {kernel}')
        source_hash = file_sha256(source_path)
        metadata_hash = file_sha256(metadata_path)
        row_manifest = raw_stats[kernel]
        if row_manifest['input_sha256'] != file_sha256(raw_csv):
            raise ValueError(f'Raw CSV hash mismatch for {kernel}')
        if row_manifest['output_sha256'] != file_sha256(prep_csv):
            raise ValueError(f'Preprocessed CSV hash mismatch for {kernel}')
        if source_hash != info['source_sha256']:
            raise ValueError(f'Source hash differs from the reconstructed data row for {kernel}')

        # Filenames narrow the search only. A usable match requires exact source,
        # metadata, local-header, archive-manifest and graph-validator checks.
        matches = []
        for path in sorted(canonical_dir.glob(kernel + '-*.json')):
            graph, authentication = overlay.authenticated_input(path)
            provenance = graph.graph['provenance']
            if provenance.get('source_sha256') != source_hash or provenance.get('metadata_sha256') != metadata_hash:
                continue
            header_hashes = provenance.get('local_header_sha256', {})
            header_status = {}
            for name, expected in header_hashes.items():
                header_path = source_path.parent / name
                header_status[name] = header_path.is_file() and file_sha256(header_path) == expected
            if not all(header_status.values()):
                raise ValueError(f'Local-header provenance mismatch for {kernel}: {header_status}')
            matches.append((path, graph, authentication, header_status))
        if len(matches) != 1:
            raise ValueError(f'Expected one authenticated compiler graph for {kernel}; found {len(matches)}')
        graph_path, graph, authentication, header_status = matches[0]
        if not isinstance(graph.graph.get('graph_identity'), str) or not graph.graph['graph_identity']:
            raise ValueError(f'Missing graph identity for {kernel}')

        coverage = graph.graph['action_point_coverage']['records']
        coverage_by_id = {record['action_id']: record for record in coverage}
        source_actions = domains.tcad.load_source_actions(kernel, ROOT / 'Data/ApplicationDataset')
        source_actions_by_kernel[kernel] = source_actions
        source_ids = set(source_actions)
        registered_ids = {r['action_id'] for r in coverage if r.get('metadata_registered')}
        if source_ids != registered_ids:
            raise ValueError(f'Source registration and authenticated graph action inventory disagree for {kernel}: '
                             f'missing={sorted(source_ids-registered_ids)}, extra={sorted(registered_ids-source_ids)}')
        from Preprocessing.create_jsonl import load_actions
        actions = {action.action_id: action for action in load_actions(
            kernel, source_path.read_text(encoding='utf-8'))}
        if set(actions) != source_ids:
            raise ValueError(f'CSV/source action objects differ for {kernel}')

        unresolved = sorted(r['action_id'] for r in coverage
                            if r.get('metadata_registered') and not r.get('compiler_matched'))
        unmapped_columns = sorted(label for label, action in actions.items() if action.csv_column is None)
        graph_matches[kernel] = {
            'family': kernel_split[kernel],
            'graph_member': str(graph_path.relative_to(release_dir)),
            'graph_identity': graph.graph['graph_identity'],
            'graph_member_sha256': authentication['file_sha256'],
            'source_sha256': source_hash,
            'metadata_sha256': metadata_hash,
            'apl_sha256': file_sha256(apl_path),
            'raw_csv_sha256': file_sha256(raw_csv),
            'preprocessed_csv_sha256': file_sha256(prep_csv),
            'local_headers': header_status,
            'measured_rows': row_manifest['output_rows'],
            'raw_rows': row_manifest['raw_rows'],
            'valid_measurement_rows': row_manifest['valid_rows'],
            'aggregated_repeats': row_manifest['valid_rows'] - row_manifest['output_rows'],
            'loop_actions': sum(a.kind == 'loop' for a in actions.values()),
            'array_actions': sum(a.kind == 'array' for a in actions.values()),
            'registered_actions': len(source_ids),
            'matched_actions': sum(r.get('compiler_matched', False) for r in coverage),
            'unresolved_actions': unresolved,
            'unmapped_csv_actions': unmapped_columns,
            'action_columns': {label: action.csv_column for label, action in sorted(actions.items())},
            'contexts': {},
            'candidate_rows_validated': 0,
            'candidate_rows_excluded_invalid': 0,
            'candidate_exclusion_reasons': {},
            'rows_with_active_unresolved_action': 0,
            'unresolved_action_value_counts': {label: Counter() for label in unresolved},
        }

    # Parse and validate every complete assignment, including source-registered
    # but compiler-unresolved actions. They remain candidate labels, never fake
    # compiler nodes. A missing APL column is explicitly counted as unknown.
    assignment_failures = Counter()
    validation_cache = {}
    for index, row in enumerate(examples):
        kernel = row['kernel_name']
        summary = graph_matches[kernel]
        profile = DEVICE_TO_PROFILE.get(row['device'].lower())
        clock = float(row['clock_period'])
        context_key = f'{row["device"]}@{clock:g}ns'
        summary['contexts'][context_key] = summary['contexts'].get(context_key, 0) + 1
        if profile is None or clock not in EXPECTED_CLOCKS:
            # Valid external target/clock rows are audited and retained in the
            # report, but reserved from the current ZCU104/U200 training set.
            summary.setdefault('reserved_external_context_rows', 0)
            summary['reserved_external_context_rows'] += 1
        actions = source_actions_by_kernel[kernel]
        missing_column_actions = summary['unmapped_csv_actions']
        signature = (kernel, row['target'])
        if signature not in validation_cache:
            try:
                candidate_values = _candidate_values(row['target'])
                domains.tcad.validate_assignments(candidate_values, actions)
                validation_cache[signature] = (candidate_values, None)
            except (TypeError, ValueError) as exc:
                validation_cache[signature] = (None, str(exc).split(':', 1)[0])
        assignments, error = validation_cache[signature]
        if error is not None:
            reason = error
            assignment_failures[reason] += 1
            summary['candidate_rows_excluded_invalid'] += 1
            summary['candidate_exclusion_reasons'][reason] = \
                summary['candidate_exclusion_reasons'].get(reason, 0) + 1
            continue
        summary['candidate_rows_validated'] += 1
        if missing_column_actions:
            reason = 'source_action_has_no_measured_apl_column'
            assignment_failures[reason] += 1
            summary['candidate_rows_excluded_invalid'] += 1
            summary['candidate_exclusion_reasons'][reason] = \
                summary['candidate_exclusion_reasons'].get(reason, 0) + 1
        active_unresolved = []
        for label in summary['unresolved_actions']:
            if label in missing_column_actions:
                continue
            values = [value for key, value in assignments.items() if key.endswith('_' + label + '}')]
            if values:
                summary['unresolved_action_value_counts'][label][','.join(values)] += 1
                if any(value not in ('0', 'none') for value in values):
                    active_unresolved.append(label)
        if active_unresolved:
            summary['rows_with_active_unresolved_action'] += 1

    failures = dict(sorted(assignment_failures.items()))
    # Known, policy-defined exclusions are not an audit defect: unmeasured APL
    # fields cannot be relabeled as disabled pragmas, and source-invalid joint
    # tuples cannot be silently repaired. Any parse/target error remains a hard
    # block; the training reader must separately filter and report these rows.
    permitted_exclusions = {'source_action_has_no_measured_apl_column'}
    allowed_reasons = permitted_exclusions | {
        reason for reason in failures
        if reason.startswith('Source-aware TCAD tuple violation at ')
    }
    if set(failures) <= allowed_reasons:
        audit_gate = 'pass_with_explicit_exclusions' if failures else 'pass'
    else:
        audit_gate = 'blocked_candidate_or_target_rows_require_review'

    kernels = []
    for kernel in sorted(graph_matches):
        row = graph_matches[kernel]
        row['candidate_exclusion_reasons'] = dict(sorted(row['candidate_exclusion_reasons'].items()))
        row['unresolved_action_value_counts'] = {
            label: dict(sorted(counts.items()))
            for label, counts in row['unresolved_action_value_counts'].items()
        }
        kernels.append(row)
    result = {
        'schema': 'salt-hls-v22-training-data-audit-v1',
        'implementation': {
            'base_git_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'working_tree_code_sha256': {
                'audit.py': file_sha256(Path(__file__)),
                'target_graph_v22.py': file_sha256(ROOT / 'GNN_branch/target_graph_v22.py'),
                'mlir_graph_v2.py': file_sha256(ROOT / 'GNN_branch/mlir_graph_v2.py'),
                'directive_domains.py': file_sha256(ROOT / 'LLM_branch/common/directive_domains.py'),
            },
        },
        'dataset': {
            'dataset_sha256': dataset_hash.hexdigest(),
            'selected_manifest_sha256': file_sha256(selected_manifest_path),
            'preprocessing_manifest_sha256': file_sha256(pre_manifest_path),
            'split_sha256': file_sha256(SPLIT_PATH),
            'reconstructed_rows': len(examples),
            'kernels': len(per_kernel),
            'raw_csv_rows': sum(item['raw_rows'] for item in pre_manifest['kernels']),
            'valid_raw_measurement_rows': sum(item['valid_rows'] for item in pre_manifest['kernels']),
            'unique_measured_design_points': sum(item['output_rows'] for item in pre_manifest['kernels']),
            'replicated_valid_measurements_aggregated': sum(item['valid_rows']-item['output_rows'] for item in pre_manifest['kernels']),
            'compact_jsonl_materialized_in_repository': (ROOT / 'artifacts/llm/mailohls_sft.jsonl').is_file(),
            'raw_measurement_rows_excluded_by_existing_preprocessing': sum(item['raw_rows']-item['valid_rows'] for item in pre_manifest['kernels']),
            'split_rows': dict(row_counts_by_split),
            'split_family_counts': {name: len(split[name+'_families']) for name in ('train','val','test')},
            'split_kernel_counts': {name: len(split[name+'_kernels']) for name in ('train','val','test')},
            'split_kernels': {name: split[name+'_kernels'] for name in ('train','val','test')},
            'split_seed': split['seed'],
            'index_membership_verified_against_reconstructed_rows': True,
        },
        'graph_release': {
            'archive_sha256': source_manifest['archive_sha256'],
            'release_manifest_sha256': file_sha256(RELEASE_MANIFEST),
            'authenticated_graph_snapshots': release.get('graph_count'),
            'matched_gnosis_kernels': len(graph_matches),
        },
        'candidate_validation': {
            'complete_assignments_validated': sum(x['candidate_rows_validated'] for x in graph_matches.values()),
            'unique_complete_assignment_signatures_validated': len(validation_cache),
            'excluded_or_blocked_rows': sum(x['candidate_rows_excluded_invalid'] for x in graph_matches.values()),
            'reasons': failures,
            'unresolved_action_rows_retained_with_source_values_and_no_compiler_embedding': sum(
                x['rows_with_active_unresolved_action'] for x in graph_matches.values()),
            'reserved_external_context_rows': sum(x.get('reserved_external_context_rows', 0)
                                                   for x in graph_matches.values()),
            'policy': 'source-authenticated candidate values remain eligible when only compiler matching is unresolved; missing APL values are not inferred',
        },
        'gate': audit_gate,
        'kernels': kernels,
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_dataset(args.release_dir, args.output)
    print(json.dumps({
        'gate': result['gate'],
        'dataset': result['dataset'],
        'graph_release': result['graph_release'],
        'candidate_validation': result['candidate_validation'],
        'report': str(args.output),
    }, sort_keys=True, indent=2))
    if result['gate'] not in ('pass', 'pass_with_explicit_exclusions'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
