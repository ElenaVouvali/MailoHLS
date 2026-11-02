"""Reauthenticate the full non-test inventory without parsing Kalman labels."""
import argparse
import hashlib
import json
from pathlib import Path
from Preprocessing.create_jsonl import build_examples, validate_preprocessing_manifest
from experiments.v22_gnn_training import train

ROOT = Path(__file__).resolve().parents[3]


def authenticate(release, audit_path):
    audit = json.loads(Path(audit_path).read_text())
    split = json.loads(train.SPLIT.read_text())
    if split['test_kernels'] != ['serrano-kalman-filter']:
        raise ValueError('Unexpected reserved test inventory')
    names = sorted(split['train_kernels'] + split['val_kernels'])
    if len(names) != 54 or len(set(names)) != 54:
        raise ValueError('Original non-test inventory changed')
    preprocessing_path = validate_preprocessing_manifest(train.PREPROCESSED)  # hashes only, including reserved table
    preprocessing = json.loads(preprocessing_path.read_text())
    raw_records = {r['kernel']: r for r in preprocessing['kernels']}
    graph_release = json.loads((ROOT / 'docs/graph-release-v22-rc3.json').read_text())
    if train.overlay.file_hash(ROOT / graph_release['archive']) != graph_release['archive_sha256']:
        raise ValueError('Frozen RC3 archive bytes changed')
    examples, manifests, _ = build_examples(train.PREPROCESSED, selected_kernels=set(names))
    source_rows = {r['kernel']: r for r in manifests}
    # PreparedData verifies source/table/split/validator and graph identities for
    # every non-test kernel; unknown action rows remain excluded, never imputed.
    data = train.PreparedData(release, '/tmp/v4b-unused-audit.json',
                              selected_kernels=names, authenticated_audit_path=audit_path)
    assessments = []
    for name in names:
        row = next(r for r in audit['kernels'] if Path(r['graph_member']).name.startswith(name + '-'))
        checks = {'source_sha256': ROOT / source_rows[name]['source'],
                  'apl_sha256': ROOT / 'Data/ApplicationAPLMapping' / (name + '.txt'),
                  'preprocessed_csv_sha256': train.PREPROCESSED / ('preprocessed-' + name + '.csv'),
                  'graph_member_sha256': Path(release) / row['graph_member']}
        for field, path in checks.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[field]:
                raise ValueError(f'Authenticated {field} changed: {name}')
        if train.overlay.file_hash(ROOT / 'Data/CSVS' / (name + '.csv')) != raw_records[name]['input_sha256']:
            raise ValueError(f'Raw CSV bytes changed: {name}')
        provenance = data.graphs[name][0].graph['provenance']
        source_dir = checks['source_sha256'].parent
        if train.overlay.file_hash(source_dir / 'kernel_info.txt') != provenance['metadata_sha256']:
            raise ValueError(f'Source action metadata changed: {name}')
        for header, expected in provenance.get('local_header_sha256', {}).items():
            if train.overlay.file_hash(source_dir / header) != expected:
                raise ValueError(f'Local source header changed: {name}/{header}')
        missing = row['unmapped_csv_actions']
        if missing:
            source = checks['source_sha256'].read_text()
            apl = checks['apl_sha256'].read_text()
            assessments.append({'kernel': name, 'classification': 'C_unknown_directive_state',
                'missing_action_ids': missing,
                'source_action_lines': [line.strip() for line in source.splitlines()
                                        if any(str(action) in line for action in missing)],
                'authenticated_apl_text': apl,
                'fixed_state_proven': False,
                'alternative_mapping': 'No alternative registered in the authenticated selected manifest; unauthenticated alternatives cannot recover labels.',
                'recovery_requirement': 'Authenticated synthesis configuration trace proving missing action states, or new synthesis under a separately versioned protocol.',
                'measured_subset_usable': False,
                'reason': 'QoR describes the whole configuration; source registration/placeholder alone does not prove the applied state.'})
    counts = {s: len({r['kernel'] for r in data.records if r['split'] == s}) for s in ('train', 'val')}
    if counts != {'train': 37, 'val': 3}:
        raise ValueError(f'Frozen eligibility changed: {counts}')
    reserved_count = next(r['measured_rows'] for r in audit['kernels'] if r['family'] == 'test')
    if len(examples) + reserved_count != 150026:
        raise ValueError('Historical inventory count changed')
    return {'schema': 'v4b-non-test-reauthentication-v2', 'original_inventory_kernels': 55,
            'non_test_kernels_reconstructed': len(names), 'non_test_rows_reconstructed': len(examples),
            'original_aggregated_rows': len(examples) + reserved_count,
            'reserved_count_provenance': 'authenticated historical audit, no test labels parsed',
            'eligible_kernel_counts': counts, 'eligible_non_test_records': len(data.records),
            'test_labels_parsed': False, 'excluded_action_assessment': assessments,
            'authentication': 'source/APL/raw CSV/preprocessed table/graph bytes verified; metadata/local headers verified; RC3 archive bytes verified; preprocessing manifest verified'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir', required=True)
    p.add_argument('--authenticated-audit', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = authenticate(args.release_dir, args.authenticated_audit)
    with open(args.output, 'x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print({k: v for k, v in report.items() if k != 'excluded_action_assessment'})


if __name__ == '__main__':
    main()
