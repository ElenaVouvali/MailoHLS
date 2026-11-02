"""Independent APL/CSV/source action observability audit for GNΩSIS labels."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from experiments.stage1_quality_sft.core import parse
from experiments.stage1_quality_sft.support import sha, source_fields


def _mapping(path):
    result = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        column, action = [x.strip() for x in line.split(',', 1)]
        if action in result:
            raise ValueError(f'Duplicate APL action mapping {action} in {path}')
        result[action] = column
    return result


def _action_defs(path):
    result = {}
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    for line in lines[1:]:
        cols = [x.strip() for x in line.split(',')]
        if len(cols) < 3 or not re.fullmatch(r'L[1-9]\d*', cols[0]):
            raise ValueError(f'Malformed source action record: {line}')
        result[cols[0].upper()] = cols[1].lower()
    return result


def _directive_tuple(kind, value, action):
    value = str(value or '').strip().lower()
    if value in ('', 'ndir', 'none', 'auto'):
        value = ''
    if kind == 'loop':
        pipe = unroll = '0'
        if value:
            m = re.fullmatch(r'(pipeline|unroll)_(\d+)', value)
            if not m:
                raise ValueError(f'Unsupported canonical loop value: {value}')
            if m.group(1) == 'pipeline': pipe = m.group(2)
            else: unroll = m.group(2)
        return {f'AUTO{{_PIPE_{action}}}': pipe, f'AUTO{{_UNROLL_{action}}}': unroll}
    part, factor, dim = 'none', '0', '0'
    if value:
        m = re.fullmatch(r'(block|cyclic)_(\d+)_(\d+)', value)
        complete = re.fullmatch(r'complete_(\d+)', value)
        if m:
            part, factor, dim = m.group(1), m.group(2), m.group(3)
        elif complete:
            part, dim = 'complete', complete.group(1)
        else:
            raise ValueError(f'Unsupported canonical array value: {value}')
    return {f'AUTO{{_ARRAY_T_{action}}}': part,
            f'AUTO{{_ARRAY_F_{action}}}': factor,
            f'AUTO{{_ARRAY_D_{action}}}': dim}


def audit(rows, dataset_manifest, preprocessing_manifest, template_map, repository_root,
          preprocessed_dir):
    root, preprocessed_dir = Path(repository_root), Path(preprocessed_dir)
    prep = json.loads(Path(preprocessing_manifest).read_text())
    by_kernel = {record['kernel']: record for record in prep['kernels']}
    dataset_by_kernel = {record['kernel']: record for record in dataset_manifest['kernel_statistics']}
    rows_by_kernel = {}
    for row in rows:
        rows_by_kernel.setdefault(row['kernel_name'], []).append(row)
    result, eligible = {}, set()
    for kernel in sorted(rows_by_kernel):
        if kernel not in by_kernel or kernel not in dataset_by_kernel:
            raise ValueError(f'No authenticated preprocessing manifest for {kernel}')
        prep_record, data_record = by_kernel[kernel], dataset_by_kernel[kernel]
        raw_path = root / 'Data/CSVS' / f'{kernel}.csv'
        source_path = root / data_record['source']
        processed_path = preprocessed_dir / f'preprocessed-{kernel}.csv'
        proof = dict(kernel=kernel, raw_csv=str(raw_path), source=str(source_path),
            processed_csv=str(processed_path), raw_csv_expected_sha256=prep_record['input_sha256'],
            processed_csv_expected_sha256=prep_record['output_sha256'],
            source_expected_sha256=data_record['source_sha256'], actions=[])
        if not raw_path.is_file() or sha(raw_path) != prep_record['input_sha256']:
            proof['block_reason'] = 'raw_csv_missing_or_hash_mismatch'; result[kernel] = proof; continue
        if not source_path.is_file() or sha(source_path) != data_record['source_sha256']:
            proof['block_reason'] = 'source_missing_or_hash_mismatch'; result[kernel] = proof; continue
        if not processed_path.is_file() or sha(processed_path) != prep_record['output_sha256']:
            proof['block_reason'] = 'processed_csv_missing_or_hash_mismatch'; result[kernel] = proof; continue
        proof.update(raw_csv_actual_sha256=sha(raw_path), source_actual_sha256=sha(source_path),
                     processed_csv_actual_sha256=sha(processed_path))
        action_map = _mapping(root / 'Data/ApplicationAPLMapping' / f'{kernel}.txt')
        actions = _action_defs(root / 'Data/ApplicationDataset' / kernel / 'kernel_info.txt')
        expected_fields = {field for action, kind in actions.items() for field in (
            (f'AUTO{{_PIPE_{action}}}', f'AUTO{{_UNROLL_{action}}}') if kind == 'loop' else
            (f'AUTO{{_ARRAY_T_{action}}}', f'AUTO{{_ARRAY_F_{action}}}', f'AUTO{{_ARRAY_D_{action}}}'))}
        fields_in_source = set(source_fields(template_map[kernel]))
        if expected_fields != fields_in_source:
            raise ValueError(f'{kernel}: source markers do not cover source action metadata')
        with raw_path.open(newline='', encoding='utf-8-sig') as f:
            raw_reader = csv.DictReader(f)
            raw_header = set(raw_reader.fieldnames or ())
            raw_active = {column: False for column in action_map.values()}
            for record in raw_reader:
                for column in raw_active:
                    if column in record and str(record[column]).strip().lower() not in ('', 'ndir', 'none', 'auto'):
                        raw_active[column] = True
        with processed_path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            processed_header = list(reader.fieldnames or ())
            processed_rows = list(reader)
        action_status = {}
        for action, kind in actions.items():
            column = action_map.get(action)
            status = 'directly_measured'
            reason = 'authenticated_raw_csv_column_and_preprocessed_row'
            if not column:
                status, reason = 'unknown', 'source_action_has_no_APL_column'
            elif column not in raw_header:
                status, reason = 'unknown', 'APL_column_missing_from_raw_CSV'
            elif not raw_active.get(column, False):
                status, reason = 'unknown', 'raw_CSV_column_never_contains_an_active_directive'
            elif column not in processed_header:
                status, reason = 'unknown', 'active_APL_column_missing_from_preprocessed_CSV'
            action_status[action] = status
            proof['actions'].append(dict(action_id=action, kind=kind, csv_column=column,
                status=status, provenance=reason,
                raw_column_present=bool(column and column in raw_header),
                raw_column_has_active_observation=bool(column and raw_active.get(column, False)),
                processed_column_present=bool(column and column in processed_header)))
        # Exact row-level equality verifies no unsupported default was injected
        # into an eligible label between the processed table and the JSONL target.
        for row in rows_by_kernel[kernel]:
            pos = int(row['preprocessed_row'])
            if pos < 0 or pos >= len(processed_rows):
                raise ValueError(f'{kernel}: processed row index outside authenticated CSV')
            table_row = processed_rows[pos]
            expected = {}
            for action, kind in actions.items():
                column = action_map.get(action)
                if column and column in processed_header:
                    expected.update(_directive_tuple(kind, table_row.get(column), action))
                else:
                    # The dataset builder's fallback is represented only for audit;
                    # this kernel will be excluded if it is not independently known.
                    expected.update(_directive_tuple(kind, '', action))
            observed = parse(row['target'])
            if observed != expected:
                raise ValueError(f'{kernel}: target does not match authenticated processed row {pos}')
        unknown = [action for action, status in action_status.items() if status != 'directly_measured']
        proof['eligible'] = not unknown
        proof['unknown_action_ids'] = unknown
        if not unknown:
            eligible.add(kernel)
        else:
            proof['block_reason'] = 'one_or_more_action_values_are_preprocessing_defaults_or_unobservable'
        result[kernel] = proof
    return result, eligible
