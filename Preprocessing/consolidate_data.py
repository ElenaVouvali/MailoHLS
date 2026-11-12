"""Recover authenticated measurements and maintain one canonical dataset registry.

Run with ``python -m Preprocessing.consolidate_data {history,restore,index,verify,tables,verify-cache}``.
Historical training snapshots are never edited. Missing action states stay unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = '25dfef26f3e9d8a4f6356e672e8c552563a76a7b'
STRIPPED = '4f3c6baf038a3c7413b0e80f9a4943477d2b6866'
RECOVER = (
    'rodinia_cfd_flux_0_baseline_0', 'rodinia_lavaMD_0_baseline',
    'rodinia_lavaMD_1_tiling_0', 'rodinia_lavaMD_1_tiling_1',
    'rodinia_lavaMD_2_pipeline_0', 'rodinia_pathfinder_0_baseline_0',
)
RESOURCES = ('BRAM_Utilization_percentage', 'DSP_Utilization_percentage',
             'FF_Utilization_percentage', 'LUT_Utilization_percentage')
DIRECTIVE = re.compile(r'^(Array_\d+|OuterLoop_\d+|InnerLoop_\d+_\d+)$')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.PIPE)


def blob(revision, path):
    return git('show', revision + ':' + str(path))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def header(raw):
    return next(csv.reader(io.StringIO(raw.decode())), [])


def records(raw):
    return [list(map(str.strip, r)) for r in csv.reader(io.StringIO(raw.decode()))
            if r and r[0].strip() and not r[0].lstrip().startswith('#')]


def applications():
    result = {}
    with (ROOT/'Data/ApplicationInformation.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            k = row['app_name'].strip()
            name = row['file_name'].strip()
            if not Path(name).suffix:
                name += '.' + row['file_name_extension'].strip().lstrip('.')
            source = Path('Data/ApplicationDataset')/k/name
            if (ROOT/source).is_file() and (ROOT/f'Data/CSVS/{k}.csv').is_file():
                if k in result:
                    raise ValueError('Duplicate application: ' + k)
                result[k] = source
    return dict(sorted(result.items()))


def split_identity():
    path = ROOT/'mailohls_runs/mailohls_final_family_split_s123.json'
    split = json.loads(path.read_text())
    sets = [set(split[n+'_families']) for n in ('train', 'val', 'test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Family overlap in locked split')
    return split, {'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes())}


def historical_inventory():
    kernels = []
    for k in applications():
        paths = [f'Data/ApplicationAPLMapping/{k}.txt',
                 f'Data/ApplicationDataset/{k}/kernel_info.txt', f'Data/CSVS/{k}.csv']
        revisions = sorted(set(git('log', '--all', '--format=%H', '--', *paths).decode().splitlines()))
        now = set(header((ROOT/paths[2]).read_bytes()))
        current_labels = {r[0] for r in records((ROOT/paths[1]).read_bytes())[1:]}
        states, extra_columns, extra_labels = {}, [], []
        for rev in revisions:
            try:
                mapping, info, raw = [blob(rev, p) for p in paths]
            except subprocess.CalledProcessError:
                continue
            labels = {r[0] for r in records(info)[1:]}
            mapped = {r[1] for r in records(mapping)}
            missing = tuple(sorted(labels-mapped))
            states.setdefault(missing, []).append(rev)
            extra = sorted(set(header(raw))-now)
            if extra:
                extra_columns.append({'commit': rev, 'extra_columns': extra, 'raw_sha256': sha(raw)})
            if labels-current_labels:
                extra_labels.append({'commit': rev, 'labels': sorted(labels-current_labels)})
        kernels.append({'kernel': k, 'revisions_scanned': len(revisions),
                        'mapping_states': [{'missing_actions': list(key), 'commits': value}
                                           for key, value in states.items()],
                        'historical_extra_columns': extra_columns, 'historical_extra_labels': extra_labels})
    refs = git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/remotes/origin').decode().splitlines()
    return {'schema': 'salt-hls-data-history-search-v1', 'refs': refs, 'kernels': kernels,
            'scope': 'Fetched origin branches/tags and all reachable local Git history; tracked Data raw CSV/APL/metadata revisions. Not deleted unreachable GitHub objects or other repositories.',
            'test_measurement_rows_parsed': False}


def compare_rows(old, new):
    def keyed(raw):
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        keys = {(r['Version'], r['Device'], float(r['Clock_Period_nsec'])): r for r in rows}
        if len(keys) != len(rows):
            raise ValueError('Ambiguous design identity')
        return keys
    a, b = keyed(old), keyed(new)
    differences = Counter()
    for key in a.keys() & b.keys():
        for column in a[key].keys() & b[key].keys():
            x, y = a[key][column], b[key][column]
            if x == y:
                continue
            try:
                if (math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-12)
                        or (math.isnan(float(x)) and math.isnan(float(y)))):
                    continue
            except ValueError:
                pass
            differences[column] += 1
    return {'original_rows': len(a), 'stripped_rows': len(b), 'same_design_keys': a.keys() == b.keys(),
            'shared_column_difference_counts': dict(sorted(differences.items())),
            'numeric_comparison_tolerance': {'relative': 1e-12, 'absolute': 1e-12}}


def label_spacing(raw):
    # The only source edit for tiling-0 changed indentation AFTER labels.
    return re.sub(rb'(?m)^([ \t]*L[0-9]+:)[ \t]*', rb'\1', raw)


def restore():
    sources = applications()
    split, _ = split_identity()
    changes, evidence = [], []
    for k in RECOVER:
        if k in split['test_kernels']:
            raise ValueError('Cannot restore locked-test inputs through this operation')
        source = sources[k]
        source_proof = []
        for p in sorted((ROOT/source.parent).iterdir()):
            if p.suffix.lower() not in {'.c', '.cc', '.cpp', '.cxx', '.h', '.hpp'}:
                continue
            rel = p.relative_to(ROOT)
            old, current = blob(ORIGINAL, rel), p.read_bytes()
            if label_spacing(old) != label_spacing(current):
                raise ValueError('Source changed beyond label spacing: ' + str(rel))
            source_proof.append({'path': str(rel), 'original_sha256': sha(old),
                                 'current_sha256': sha(current), 'equal_after_label_spacing_only': True})
        raw_path = f'Data/CSVS/{k}.csv'
        old, stripped = blob(ORIGINAL, raw_path), blob(STRIPPED, raw_path)
        comparison = compare_rows(old, stripped)
        metadata_path = str(source.parent/'kernel_info.txt')
        original_records = {r[0]: r for r in records(blob(ORIGINAL, metadata_path))[1:]}
        stripped_records = {r[0]: r for r in records(blob(STRIPPED, metadata_path))[1:]}
        if any(original_records[a] != stripped_records[a]
               for a in original_records.keys() & stripped_records.keys()):
            raise ValueError('Common action definition changed: ' + k)
        comparison['common_action_records_unchanged'] = True
        if k == 'rodinia_lavaMD_1_tiling_0':
            peer = blob(STRIPPED, 'Data/CSVS/rodinia_lavaMD_1_tiling_1.csv')
            if stripped != peer:
                raise ValueError('Expected historical cross-kernel CSV duplication not verified')
            comparison['reason_for_replacement'] = 'Stripped tiling-0 table is byte-identical to tiling-1 despite distinct labeled sources; restore original source-associated table, not a column join.'
            comparison['duplicated_peer_sha256'] = sha(peer)
        elif not comparison['same_design_keys'] or comparison['shared_column_difference_counts']:
            raise ValueError('Historical measurements conflict: ' + k)
        files = []
        for rel in (raw_path, str(source.parent/'kernel_info.txt'), f'Data/ApplicationAPLMapping/{k}.txt'):
            old, previous = blob(ORIGINAL, rel), blob(STRIPPED, rel)
            current = (ROOT/rel).read_bytes()
            if current not in (old, previous):
                raise ValueError('Refusing to overwrite unrelated edits: ' + rel)
            if old == previous:
                continue
            files.append({'path': rel, 'previous_sha256': sha(previous), 'restored_sha256': sha(old),
                          'original_git_blob': git('rev-parse', ORIGINAL+':'+rel).decode().strip(),
                          'previous_git_blob': git('rev-parse', STRIPPED+':'+rel).decode().strip()})
            if current != old:
                changes.append((ROOT/rel, old))
        evidence.append({'kernel': k, 'row_identity_check': comparison, 'source_proof': source_proof,
                         'recovered_columns': sorted(set(header(blob(ORIGINAL, raw_path)))-set(header(stripped))),
                         'files': files})
    # Validate every proposed file before applying any mutation.
    for path, content in changes:
        path.write_bytes(content)
    receipt = {'schema': 'salt-hls-authenticated-data-restoration-v1', 'source_commit': ORIGINAL,
               'stripped_commit': STRIPPED, 'kernels': evidence, 'test_measurement_rows_parsed': False,
               'historical_checkpoints_or_snapshots_modified': False,
               'unknown_directive_states_imputed': False}
    write_json(ROOT/'Data/provenance/restoration_20261110.json', receipt)
    print(f'Restored {len(changes)} files for {len(evidence)} kernels; receipt in Data/provenance/')


def graph_provenance():
    release_path = ROOT/'docs/graph-release-v22-rc3.json'
    release = json.loads(release_path.read_text())
    archive = ROOT/release['archive']
    if sha(archive.read_bytes()) != release['archive_sha256']:
        raise ValueError('RC3 archive identity changed')
    result = {}
    names = applications()
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.name.startswith('canonical/') or not member.name.endswith('.json'):
                continue
            matches = [k for k in names if Path(member.name).name.startswith(k+'-')]
            if len(matches) == 1:
                raw = tar.extractfile(member).read()
                if sha(raw) != release['files_sha256'][member.name]:
                    raise ValueError('RC3 member identity changed')
                result[matches[0]] = json.loads(raw)['metadata']['provenance']
    return result


def canonical_index():
    from Preprocessing import gnosis_recovery
    split, split_record = split_identity()
    graphs = graph_provenance()
    # Frozen snapshots may be restored separately from the artifact store.
    # Preserve their registered references when they are not materialized;
    # active canonical measurements are always checked from disk below.
    previous_path = ROOT/'Data/manifest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    snapshot_refs = {k['kernel']: k.get('historical_snapshots', []) for k in previous.get('kernels', [])}
    kernels = []
    for k, source in applications().items():
        cohort = next(n for n in ('train', 'val', 'test') if k in split[n+'_kernels'])
        paths = {'source': str(source), 'metadata': str(source.parent/'kernel_info.txt'),
                 'mapping': f'Data/ApplicationAPLMapping/{k}.txt', 'measurements': f'Data/CSVS/{k}.csv'}
        files = {role: {'path': p, 'sha256': sha((ROOT/p).read_bytes())} for role, p in paths.items()}
        definitions = {r[0]: r[1] for r in records((ROOT/paths['metadata']).read_bytes())[1:]}
        mapping_rows = records((ROOT/paths['mapping']).read_bytes())
        mapping = {c: a for c, a in mapping_rows}
        if len(mapping) != len(mapping_rows) or len(set(mapping.values())) != len(mapping):
            raise ValueError('Duplicate action mapping: ' + k)
        labels = set(re.findall(r'\b(L[1-9][0-9]*)\s*:', (ROOT/source).read_text()))
        extra_files = [p for p in sorted((ROOT/source.parent).iterdir())
                       if p != ROOT/source and p.suffix.lower() in {'.c', '.cc', '.cpp', '.cxx', '.h', '.hpp'}]
        files.update({'local_source:'+p.name: {'path': str(p.relative_to(ROOT)), 'sha256': sha(p.read_bytes())}
                      for p in extra_files})
        counts, active = Counter(), Counter()
        with (ROOT/paths['measurements']).open(newline='') as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if len(columns) != len(set(columns)):
                raise ValueError('Duplicate CSV headers: ' + k)
            if cohort != 'test':
                for row in reader:
                    counts['raw_rows'] += 1
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError('Malformed CSV row: ' + k)
                    for c in columns:
                        if DIRECTIVE.fullmatch(c) and row[c].strip().lower() not in {'', 'ndir', 'none', 'auto', 'nan'}:
                            active[c] += 1
                    try:
                        latency = float(row['Latency_msec'])
                        resources = [float(row[c]) for c in RESOURCES]
                        complete = math.isfinite(latency) and latency > 0 and all(math.isfinite(v) and v >= 0 for v in resources)
                    except ValueError:
                        complete = False
                    counts['finite_positive_latency_and_known_resources' if complete else 'invalid_or_unknown_measurement_rows'] += 1
                    if complete and any(v >= 100 for v in resources):
                        counts['measured_resource_at_or_above_capacity_rows'] += 1
        valid = {c: a for c, a in mapping.items() if a in definitions and a in labels and c in columns}
        csv_uncovered = sorted((labels | set(definitions))-set(valid.values()), key=lambda a: int(a[1:]))
        recovery = gnosis_recovery.validated_entry(k, ROOT)
        proven = set(recovery['missing_actions']) if recovery and recovery['canonical_eligible'] else set()
        if not proven.issubset(csv_uncovered):
            raise ValueError('Recovery conflicts with canonical CSV/APL coverage: ' + k)
        unknown = sorted(set(csv_uncovered)-proven, key=lambda a: int(a[1:]))
        defects = {'mapped_columns_missing_raw': sorted(set(mapping)-set(columns)),
                   'mapping_labels_missing_metadata': sorted(set(mapping.values())-set(definitions)),
                   'mapping_labels_missing_source': sorted(set(mapping.values())-labels),
                   'active_unmapped_columns': sorted(set(active)-set(valid))}
        provenance = graphs.get(k, {})
        compatible = (provenance.get('source_sha256') == files['source']['sha256']
                      and provenance.get('metadata_sha256') == files['metadata']['sha256'])
        snapshots = []
        for folder in ('LLM_branch/Data/preprocessed_CSVS', 'GNN_branch/Data/preprocessed_CSVS'):
            path = ROOT/folder/f'preprocessed-{k}.csv'
            if path.exists():
                snapshots.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes()),
                                  'role': 'historical_frozen_derived_snapshot',
                                  'mapped_columns_absent': sorted(set(mapping)-set(header(path.read_bytes())))})
            else:
                snapshots.extend(s for s in snapshot_refs.get(k, [])
                                 if s['path'] == str(path.relative_to(ROOT)))
        kernels.append({'kernel': k, 'split': cohort, 'files': files, 'csv_to_action': valid,
                        'csv_uncovered_source_actions': csv_uncovered,
                        'unproven_source_actions': unknown, 'source_labels_absent_metadata': sorted(labels-set(definitions)),
                        'defects': defects, 'measurement_counts': dict(counts), 'test_rows_parsed': False,
                        'raw_csv_complete_source_configuration_eligible': not csv_uncovered and not any(defects.values()),
                        'complete_source_configuration_eligible': not unknown and not any(defects.values()),
                        'complete_source_configuration_contract': (
                            gnosis_recovery.CONTRACT if proven else 'canonical-csv-apl-v1' if not unknown else 'partial-csv-apl-v1'),
                        'action_state_recovery': None if recovery is None else {
                            'registry': str(gnosis_recovery.REGISTRY),
                            'registry_sha256': sha((ROOT/gnosis_recovery.REGISTRY).read_bytes()),
                            'states': recovery['states'], 'canonical_source_compatible': recovery['canonical_eligible'],
                            'proven_canonical_actions': sorted(proven, key=lambda a: int(a[1:]))},
                        'rc3_release_source_and_metadata_match': compatible, 'historical_snapshots': snapshots})
    return {'schema': 'salt-hls-canonical-data-v1', 'split': split_record, 'kernels': kernels,
            'canonical_raw_root': 'Data/CSVS', 'shared_derived_cache': 'Data/preprocessed_CSVS',
            'policies': {'unknown_action_state': 'unproven; never impute no-directive',
                         'split_reuse': 'family/kernel membership only; historical row indices and fitted normalization do not apply to regenerated tables',
                         'locked_test': 'headers and hashes only; no measurement-row parsing',
                         'resource_overflow': 'retain finite measured demands; assess feasibility per requested budget',
                         'unknown_measurements': 'not evidence of resource infeasibility',
                         'timing': 'requested clock is context; these CSVs do not prove achieved timing qualification',
                         'snapshots': 'preserved for replay; not canonical live inputs'},
            'summary': {'kernels': len(kernels),
                        'complete_source_configuration_eligible_kernels': sum(k['complete_source_configuration_eligible'] for k in kernels),
                        'raw_csv_complete_source_configuration_eligible_kernels': sum(k['raw_csv_complete_source_configuration_eligible'] for k in kernels),
                        'kernels_with_unproven_source_actions': sum(bool(k['unproven_source_actions']) for k in kernels),
                        'unproven_source_actions': sum(len(k['unproven_source_actions']) for k in kernels),
                        'rc3_graphs_requiring_regeneration': [k['kernel'] for k in kernels if not k['rc3_release_source_and_metadata_match']]}}


def verify():
    expected = json.loads((ROOT/'Data/manifest.json').read_text())
    actual = canonical_index()
    if expected != actual:
        raise ValueError('Canonical data drift detected; audit changes before running index')
    print(json.dumps(actual['summary'], indent=2))
    return actual


def tables():
    """One optional shared derived cache, using the maintained preprocessor."""
    manifest = verify()
    from Preprocessing.data_preprocess import preprocess_kernel
    output = ROOT/manifest['shared_derived_cache']
    output.mkdir(exist_ok=True)
    existing = output/'manifest.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get('schema') != 'salt-hls-canonical-derived-v1':
            raise ValueError('Refusing to overwrite an unrelated dataset cache')
        for entry in previous['kernels']:
            if sha((ROOT/entry['path']).read_bytes()) != entry['sha256']:
                raise ValueError('Unrecorded cache edits: ' + entry['path'])
    elif any(output.iterdir()):
        raise ValueError('Refusing to overwrite an unregistered dataset cache')
    receipt = {'schema': 'salt-hls-canonical-derived-v1', 'canonical_manifest_sha256': sha((ROOT/'Data/manifest.json').read_bytes()),
               'mode': 'all_measured_targets_train_val', 'unknown_actions_imputed': False,
               'preprocessor_sha256': sha((ROOT/'Preprocessing/data_preprocess.py').read_bytes()),
               'consolidator_sha256': sha(Path(__file__).read_bytes()),
               'parameters': {'minimum_weight': .1, 'gamma': 2., 'all_targets': True,
                              'retain_measured_overflow': True, 'preserve_measured_actions': True},
               'test_rows_read': False, 'kernels': []}
    for entry in manifest['kernels']:
        if entry['split'] == 'test':
            continue
        if any(entry['defects'].values()):
            raise ValueError('Unresolved measured-control mapping: ' + entry['kernel'])
        table, stats = preprocess_kernel(ROOT/entry['files']['measurements']['path'],
                                         'gnn', '', 10., .1, 2., all_targets=True,
                                         retain_measured_overflow=True,
                                         preserve_measured_actions=True)
        path = output/('preprocessed-'+entry['kernel']+'.csv')
        table.to_csv(path, index=False, float_format='%.12g')
        # This cache is a CSV/APL projection, not a joined source-action sidecar.
        # Never promote its rows merely because the separate recovery is complete.
        stats.update({'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes()),
                      'complete_source_configuration_eligible': entry['raw_csv_complete_source_configuration_eligible'],
                      'authenticated_complete_source_configuration_eligible': entry['complete_source_configuration_eligible'],
                      'unproven_source_actions': entry['csv_uncovered_source_actions']})
        receipt['kernels'].append(stats)
        print(entry['kernel'], stats['output_rows'], flush=True)
    write_json(output/'manifest.json', receipt)


def verify_cache():
    manifest = verify()
    entries = {k['kernel']: k for k in manifest['kernels'] if k['split'] != 'test'}
    output = ROOT/manifest['shared_derived_cache']
    cache = json.loads((output/'manifest.json').read_text())
    if cache['canonical_manifest_sha256'] != sha((ROOT/'Data/manifest.json').read_bytes()):
        raise ValueError('Derived cache uses another canonical registry')
    if {k['kernel'] for k in cache['kernels']} != set(entries):
        raise ValueError('Derived cache must contain exactly the train/val kernels')
    expected_paths = {str(output/('preprocessed-'+k+'.csv')) for k in entries}
    if {str(p) for p in output.glob('*.csv')} != expected_paths:
        raise ValueError('Unregistered or locked-test table in shared cache')
    totals = Counter(tables=0, unique_design_rows=0, known_measured_raw_rows=0,
                     rows_with_resource_at_or_above_capacity=0, complete_source_eligible_rows=0)
    for entry in cache['kernels']:
        k = entries[entry['kernel']]
        path = ROOT/entry['path']
        if path != output/('preprocessed-'+entry['kernel']+'.csv') or sha(path.read_bytes()) != entry['sha256']:
            raise ValueError('Derived cache identity changed: ' + entry['kernel'])
        if entry['valid_rows'] != k['measurement_counts']['finite_positive_latency_and_known_resources']:
            raise ValueError('Measured rows were lost: ' + entry['kernel'])
        totals['tables'] += 1
        totals['known_measured_raw_rows'] += entry['valid_rows']
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not set(k['csv_to_action']).issubset(reader.fieldnames or []):
                raise ValueError('Recorded control columns lost: ' + entry['kernel'])
            rows = 0
            for row in reader:
                rows += 1
                resource = [float(row[c]) for c in RESOURCES]
                if not (float(row['Clock_Period_nsec']) > 0 and float(row['Latency_msec']) > 0
                        and all(math.isfinite(v) and v >= 0 for v in resource)):
                    raise ValueError('Invalid physical measurement in derived cache')
                if any(v >= 100 for v in resource):
                    totals['rows_with_resource_at_or_above_capacity'] += 1
            if rows != entry['output_rows']:
                raise ValueError('Derived row count differs: ' + entry['kernel'])
            totals['unique_design_rows'] += rows
            if entry['complete_source_configuration_eligible'] != k['raw_csv_complete_source_configuration_eligible']:
                raise ValueError('CSV projection mislabeled as joined complete-action supervision')
            if entry['complete_source_configuration_eligible']:
                totals['complete_source_eligible_rows'] += rows
    print(json.dumps(dict(totals), indent=2))
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('history', 'restore', 'index', 'verify', 'tables', 'verify-cache'))
    args = parser.parse_args()
    if args.command == 'history':
        write_json(ROOT/'Data/provenance/history_search_20261110.json', historical_inventory())
    elif args.command == 'restore':
        restore()
    elif args.command == 'index':
        write_json(ROOT/'Data/manifest.json', canonical_index())
    elif args.command == 'verify':
        verify()
    elif args.command == 'verify-cache':
        verify_cache()
    else:
        tables()


if __name__ == '__main__':
    main()
