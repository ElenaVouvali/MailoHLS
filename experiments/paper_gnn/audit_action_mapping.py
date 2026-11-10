"""Audit action/column provenance without parsing locked-test measurement rows.

This reports mapping evidence and the current whole-kernel exclusion policy.
It never creates default directive labels or changes dataset eligibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from LLM_branch.common.mailohls_contract import family_id_from_kernel_name

ROOT = Path(__file__).resolve().parents[2]
DIRECTIVE_COLUMN = re.compile(r'^(Array_\d+|OuterLoop_\d+|InnerLoop_\d+_\d+)$')
NO_DIRECTIVE = {'', 'ndir', 'none', 'auto', 'nan'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [line.strip().split(',') for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def columns(path, parse_rows):
    active = Counter()
    malformed = 0
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if parse_rows:
            candidates = [c for c in header if DIRECTIVE_COLUMN.fullmatch(c)]
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    malformed += 1
                for col in candidates:
                    if str(row.get(col) or '').strip().lower() not in NO_DIRECTIVE:
                        active[col] += 1
    return {'header': header, 'active_counts': dict(active), 'malformed_rows': malformed,
            'duplicate_headers': [c for c,n in Counter(header).items() if n>1],
            'measurement_rows_parsed': parse_rows}


def audit(root):
    split_path = root/'mailohls_runs/mailohls_final_family_split_s123.json'
    split = json.loads(split_path.read_text())
    pre_dir = root/'LLM_branch/Data/preprocessed_CSVS'
    manifest_path = pre_dir/'preprocessing_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {}
    with (root/'Data/ApplicationInformation.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            name = row['file_name'].strip()
            if not Path(name).suffix:
                name += '.'+row['file_name_extension'].strip().lstrip('.')
            sources[row['app_name'].strip()] = root/'Data/ApplicationDataset'/row['app_name'].strip()/name
    rows = []
    for entry in sorted(manifest['kernels'],key=lambda r:r['kernel']):
        kernel = entry['kernel']
        cohort = next(n for n in ('train','val','test') if kernel in split[n+'_kernels'])
        info_path = root/'Data/ApplicationDataset'/kernel/'kernel_info.txt'
        mapping_path = root/'Data/ApplicationAPLMapping'/(kernel+'.txt')
        source_path = sources[kernel]
        definitions = records(info_path)[1:]
        mapping = records(mapping_path)
        declared = {r[0]:r[1] for r in definitions}
        valid_mapping = {column:label for column,label in mapping if label in declared}
        source_lines = source_path.read_text().splitlines()
        missing = sorted(set(declared)-set(valid_mapping.values()),key=lambda x:int(x[1:]))
        raw_path = root/'Data/CSVS'/(kernel+'.csv')
        processed_path = pre_dir/('preprocessed-'+kernel+'.csv')
        # Metadata/header checks cover test; no test measurement rows are read.
        raw = columns(raw_path,cohort!='test')
        processed = columns(processed_path,cohort!='test')
        provenance_status = {'raw_csv_matches_manifest':sha(raw_path)==entry['input_sha256'],
                             'processed_csv_matches_manifest':sha(processed_path)==entry['output_sha256']}
        details = []
        analysis_path = source_path.parent/'src_info.json'
        analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
        legacy_loops = {label.strip() for label in analysis.get('loops',{})}
        for label in missing:
            locations = [{'line':i+1,'text':line.strip()} for i,line in enumerate(source_lines)
                         if re.search(rf'\b{label}\s*:',line)]
            details.append({'action':label,'kind':declared[label],'source_locations':locations,
                            'historical_applied_directive':'unproven; no default inferred'})
        rows.append({'kernel':kernel,'family':family_id_from_kernel_name(kernel),'split':cohort,'preprocessed_rows':entry['output_rows'],
            'csv_to_action':valid_mapping,
            'declared_actions':len(definitions),'mapped_actions':len(set(valid_mapping.values())),
            'unmapped_source_actions':missing,'unmapped_source_details':details,
            'mapped_columns_missing_raw_header':sorted(set(valid_mapping)-set(raw['header'])),
            'mapped_columns_missing_processed_header':sorted(set(valid_mapping)-set(processed['header'])),
            'active_unmapped_raw_columns':sorted(set(raw['active_counts'])-set(valid_mapping)),
            'active_unmapped_processed_columns':sorted(set(processed['active_counts'])-set(valid_mapping)),
            'stale_mapping_entries':[[c,l] for c,l in mapping if l not in declared],
            'duplicate_mapping_columns':[c for c,n in Counter(c for c,l in mapping).items() if n>1],
            'duplicate_mapped_labels':[l for l,n in Counter(valid_mapping.values()).items() if n>1],
            'kind_mismatches':[[c,l] for c,l in valid_mapping.items()
                               if ('array' if c.startswith('Array_') else 'loop')!=declared[l]],
            'raw_column_summary':raw,'processed_column_summary':processed,
            'guard_excludes':bool(missing) if cohort!='test' else None,
            'legacy_loop_analysis':{
                'path':str(analysis_path.relative_to(root)),
                'sha256':sha(analysis_path) if analysis_path.exists() else None,
                'unmapped_actions_present':[a for a in missing if a in legacy_loops],
                'mapped_loops_absent':[a for c,a in valid_mapping.items()
                                       if declared[a]=='loop' and a not in legacy_loops]},
            'provenance_status':provenance_status,
            'expected_raw_csv_sha256':entry['input_sha256'],
            'expected_processed_csv_sha256':entry['output_sha256'],
            'provenance':{str(p.relative_to(root)):sha(p) for p in
                          (info_path,mapping_path,source_path,raw_path,processed_path)}})
    summary = {}
    for name in ('train','val','test'):
        selected = [r for r in rows if r['split']==name]
        affected = [r for r in selected if r['unmapped_source_actions']]
        summary[name]={'kernels':len(selected),'kernels_with_unmapped_actions':len(affected),
                       'affected_preprocessed_rows':sum(r['preprocessed_rows'] for r in affected),
                       'locked_families':len({r['family'] for r in selected}),
                       'families_after_mapping_guard':len({r['family'] for r in selected if not r['unmapped_source_actions']}),
                       'rows_parsed':name!='test'}
    return {'schema':'salt-paper-action-mapping-audit-v1','summary':summary,'kernels':rows,
        'split_sha256':sha(split_path),'preprocessing_manifest_sha256':sha(manifest_path),
        'interpretation':'Mapping absence does not prove a fixed no-directive state or invalidate measured QoR. Current PaperData rejects the whole kernel.',
        'upstream_default_risk':'create_jsonl.cell_text(None) returns an empty directive, and action_assignments emits zero/default choices. This is a software default, not measurement evidence.',
        'test_measurements_parsed':False,'mappings_modified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Source-action mapping audit','',
        'No mappings changed. Missing applied states have not been assumed to be no-directive. Locked-test measurement rows were not parsed.','',
        '| Split | Kernel | Unmapped actions | Rows excluded by current guard |',
        '|---|---|---|---:|']
    for row in result['kernels']:
        if row['unmapped_source_actions']:
            lines.append(f"| {row['split']} | {row['kernel']} | {', '.join(row['unmapped_source_actions'])} | {row['preprocessed_rows']:,} |")
    total_missing = sum(len(r['unmapped_source_actions']) for r in result['kernels'])
    legacy_missing = sum(len(r['legacy_loop_analysis']['unmapped_actions_present']) for r in result['kernels'])
    mapped_not_legacy = sum(len(r['legacy_loop_analysis']['mapped_loops_absent']) for r in result['kernels'])
    lines += ['', '## Structural inventory versus measurement inventory','',
        f'{total_missing-legacy_missing} of {total_missing} unmapped source actions are absent from their historical src_info.json loop inventories. {mapped_not_legacy} mapped loops are absent from those inventories.',
        'This supports an inventory mismatch when mapped loops are retained but source-only loops are absent; it does not by itself prove a per-design fixed directive state.',
        '', '## Cohort impact','']
    for name in ('train','val'):
        s=result['summary'][name]
        lines.append(f"- {name}: {s['kernels_with_unmapped_actions']} of {s['kernels']} kernels and {s['affected_preprocessed_rows']:,} rows excluded; {s['families_after_mapping_guard']} of {s['locked_families']} families remain.")
    lines += ['', '## Additional header/provenance issues','']
    for row in result['kernels']:
        for field,title in (('mapped_columns_missing_raw_header','raw'),('mapped_columns_missing_processed_header','preprocessed')):
            if row[field]:
                names=', '.join(c+' -> '+row['csv_to_action'][c] for c in row[field])
                lines.append(f"- {row['kernel']} ({row['split']}): mapped {names} absent from {title} header.")
        if not all(row['provenance_status'].values()):
            lines.append(f"- {row['kernel']}: manifest hash checks {row['provenance_status']}; no test measurement rows were parsed.")
        if row['stale_mapping_entries']:
            lines.append(f"- {row['kernel']}: stale mapping entries {row['stale_mapping_entries']}; undeclared labels require review. Active unmapped columns: {row['active_unmapped_raw_columns']}.")
    lines += ['', 'The JSON report records source locations, file hashes, header gaps, stale/duplicate mappings and active unmapped CSV columns.',
        '', 'Upstream JSONL creation defaults missing action columns to no-directive values; those defaults do not prove the historical synthesis configuration. Recover an authenticated fixed-state/generator trace before reinstating a kernel.']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
