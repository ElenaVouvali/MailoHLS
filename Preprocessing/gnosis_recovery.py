"""Recover omitted GNOSIS states from pinned SQLite vectors, never from defaults.

The sidecars are a separate source-level supervision contract. Legacy CSV/APL
tables remain partial until explicitly joined through ``complete_rows``.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import pickle
import re
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROOF = Path('Data/provenance/gnosis_17_recovery_20261113')
REGISTRY = PROOF/'recovery.json'
PIN = '8888db17a6f344307de0a27cb431582f3f6fd093'
CONTRACT = 'canonical-csv-apl-plus-authenticated-sqlite-states-v1'
QOR = {'Latency_msec':'latency', 'Synthesis_Time_sec':'synth_time',
       'BRAM_Utilization_percentage':'util_bram', 'DSP_Utilization_percentage':'util_dsp',
       'FF_Utilization_percentage':'util_ff', 'LUT_Utilization_percentage':'util_lut'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')


def same_number(a, b):
    try:
        x,y=float(a),float(b)
        return (math.isnan(x) and math.isnan(y)) or math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12)
    except (ValueError,TypeError):
        return str(a)==str(b)


class MetricUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Unexpected executable pickle global: '+module+'.'+name)


def decode_vector(key, domains):
    if not re.fullmatch(r'\[\s*\d+(?:\s+\d+)*\s*\]',key):
        raise ValueError('Malformed original directive vector')
    indices=list(map(int,key.strip('[]').split()))
    if len(indices)!=len(domains):
        raise ValueError('Directive vector length differs from original action inventory')
    if any(not 0<=v<len(domains[i]) for i,v in enumerate(indices)):
        raise ValueError('Out-of-range directive index; clamping is forbidden')
    return [domain[index] for domain,index in zip(domains,indices)]


def source_equivalence(old, new):
    if old==new:
        return 'byte-identical'
    normalize=lambda b:re.sub(rb'(?m)^([ \t]*L[0-9]+:)[ \t]*',rb'\1',b)
    if normalize(old)==normalize(new):
        return 'label-spacing-only'
    return 'unproven-source-equivalence'


def key_of(row):
    return (row['Version'],row['Device'],float(row['Clock_Period_nsec']))


def load_registry(root=ROOT):
    path=root/REGISTRY
    if not path.exists():
        return {}
    data=json.loads(path.read_text())
    if data['schema']!='salt-hls-gnosis-action-recovery-v1':
        raise ValueError('Unknown recovery provenance schema')
    return {e['kernel']:e for e in data['kernels']}


def validated_entry(kernel, root=ROOT):
    entry=load_registry(root).get(kernel)
    if entry is None:
        return None
    for path,digest in entry['canonical_files'].items():
        if sha(root/path)!=digest:
            raise ValueError('Recovered states bound to different canonical input: '+path)
    if sha(root/entry['states']['path'])!=entry['states']['sha256']:
        raise ValueError('Recovered action-state sidecar changed: '+kernel)
    return entry


def complete_rows(kernel, root=ROOT):
    """Explicitly join measured APL states and recovered states for source-safe kernels.

    Locked-test membership is checked before opening any measured rows. A
    partial view, source mismatch, duplicate, missing row, or QoR drift fails.
    """
    split=json.loads((root/'mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    if kernel in split['test_kernels']:
        raise ValueError('Locked-test rows cannot be loaded by the recovery join')
    entry=validated_entry(kernel,root)
    if entry is not None and not entry['canonical_eligible']:
        from Preprocessing.gnosis_compilation_recovery import canonical_entry
        entry=canonical_entry(entry,root)
    if entry is None or not entry['canonical_eligible']:
        raise ValueError('No authenticated complete canonical recovery for '+kernel)
    yield from _joined_rows(entry, root, entry.get('contract', CONTRACT))


def _joined_rows(entry, root, contract):
    """The same strict design/QoR/action join for explicitly bound source versions."""
    path=root/entry['canonical_measurements']
    with path.open(newline='') as handle:
        raw=list(csv.DictReader(handle))
    records=[]
    with gzip.open(root/entry['states']['path'],'rt') as stream:
        for line in stream:
            records.append(json.loads(line))
    by_key={tuple([r['version'],r['device'],float(r['clock'])]):r for r in records}
    if len(by_key)!=len(records) or len(raw)!=len(records) or {key_of(r) for r in raw}!=set(by_key):
        raise ValueError('Recovery sidecar does not cover every original measured configuration uniquely')
    mapping=entry['csv_to_action']
    for row in raw:
        recovered=by_key[key_of(row)]
        if recovered['qor_sha256']!=object_sha({c:row[c] for c in QOR}):
            raise ValueError('QoR association changed in recovered configuration')
        states={action:row[column].strip() for column,action in mapping.items()}
        if set(recovered['states'])!=set(entry['missing_actions']) or set(states)&set(recovered['states']):
            raise ValueError('Missing or conflicting recovered action identity')
        states.update(recovered['states'])
        if set(states)!=set(entry['source_actions']) or any(not v for v in states.values()):
            raise ValueError('Partial action configuration cannot be called complete')
        yield {'measurement':row,'source_action_states':states,'contract':contract,
               'sqlite_key':recovered['sqlite_key'],'sqlite_rowid':recovered['sqlite_rowid']}


def recover(upstream, root=ROOT):
    if (root/REGISTRY).exists():
        raise FileExistsError('Preserve the existing recovery receipt; reproduce from its isolated base or verify the recorded sidecars')
    for record in json.loads((root/PROOF/'code_blobs.json').read_text()):
        path=root/record['destination'];raw=path.read_bytes()
        if sha(path)!=record['sha256'] or hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()!=record['blob']:
            raise ValueError('Pinned upstream dictionary/generator code changed')
    before=json.loads((root/PROOF/'before/canonical_manifest.json').read_text())
    split=json.loads((root/before['split']['path']).read_text())
    blobs=json.loads((root/PROOF/'upstream_blobs.json').read_text())
    artifacts={}
    for record in blobs:
        path=upstream/record['path'];data=path.read_bytes()
        actual=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
        if actual!=record['blob'] or len(data)!=record['size']:
            raise ValueError('Upstream pinned Git blob mismatch: '+record['path'])
        artifacts[record['path']]={**record,'sha256':sha(path),'upstream_commit':PIN}
    code=root/PROOF/'upstream_code/GNWSIS/modules/directivesManipularor.py'
    # This is the pinned, recorded exporter dictionary, not a new directive policy.
    spec=importlib.util.spec_from_file_location('pinned_gnosis_directives',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    dictionary=module.DirectivesManipulator.__new__(module.DirectivesManipulator)
    entries=[];metadata_repairs=[]
    for entry in before['kernels']:
        if entry['complete_source_configuration_eligible']:
            continue
        k=entry['kernel']
        if k in split['test_kernels']:
            raise ValueError('Recovery cannot inspect locked-test measurements')
        source=root/entry['files']['source']['path'];original=upstream/'data/ApplicationDataset'/k
        info=original/'kernel_info.txt';info_lines=info.read_text().splitlines()
        labels=[line.split(',')[0].strip() for line in info_lines[1:]]
        if labels!=['L'+str(i+1) for i in range(len(labels))]:
            raise ValueError('Original vector indexing cannot be tied to contiguous source labels')
        # Replay only the inserter's label scan; never invoke its synthesis code.
        found=[];next_label=1
        for line_no,line in enumerate((original/source.name).read_text().splitlines(),1):
            compact=line.replace(' ','').replace('\t','')
            label='L'+str(next_label)
            if label in compact:
                if not re.search(r'\b'+label+r'\s*:',line):
                    raise ValueError('Original sequential inserter would hit a non-label')
                found.append((label,line_no,line.strip()));next_label+=1
        if [x[0] for x in found]!=labels:
            raise ValueError('Original source insertion order differs from directive vector')
        files={};compatibility=[]
        for path in [source,*sorted(source.parent.glob('*.h'))]:
            peer=original/path.name
            match=source_equivalence(peer.read_bytes(),path.read_bytes()) if peer.exists() else 'missing-upstream-file'
            compatibility.append(dict(path=str(path.relative_to(root)),canonical_sha256=sha(path),
                upstream_path=str(peer.relative_to(upstream)),upstream_sha256=sha(peer) if peer.exists() else None,proof=match))
            files[str(path.relative_to(root))]=sha(path)
        source_safe=all(r['proof'] in ('byte-identical','label-spacing-only') for r in compatibility)
        canonical_metadata=source.parent/'kernel_info.txt'
        if canonical_metadata.read_bytes()!=info.read_bytes():
            old=subprocess.check_output(['git','-C',str(root),'show','25dfef26f3e9d8a4f6356e672e8c552563a76a7b:'+str(canonical_metadata.relative_to(root))])
            if old!=info.read_bytes():
                raise ValueError('Metadata does not match authenticated historical MailoHLS version')
            old_records={l.split(',')[0]:l for l in canonical_metadata.read_text().splitlines()[1:] if l.strip()}
            all_records={l.split(',')[0]:l for l in info_lines[1:]}
            if any(all_records.get(a)!=v for a,v in old_records.items()) or not source_safe:
                raise ValueError('Metadata restoration would change an existing definition or source identity')
            metadata_repairs.append(dict(path=str(canonical_metadata.relative_to(root)),before_sha256=sha(canonical_metadata),
                after_sha256=sha(info),added_records=[v for a,v in all_records.items() if a not in old_records],
                historical_commit='25dfef26f3e9d8a4f6356e672e8c552563a76a7b'))
            # Defer canonical mutations until every database and source passes audit.
        files[str(canonical_metadata.relative_to(root))]=sha(info) if source_safe else sha(canonical_metadata)
        for role in ('mapping','measurements'):
            rec=entry['files'][role];files[rec['path']]=sha(root/rec['path'])
            if files[rec['path']]!=rec['sha256']:
                raise ValueError('Canonical input drift before recovery')
        with (root/entry['files']['measurements']['path']).open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        indexed={key_of(r):r for r in rows}
        if len(indexed)!=len(rows):raise ValueError('Ambiguous original design key')
        domains={size:dictionary._get_directives_per_action_point(str(info),size) for size in (128,1024)}
        counter=0;records=[];contexts=[];hist={a:Counter() for a in entry['unproven_source_actions']}
        for device in ('xczu7ev-ffvc1156-2-e','xcu200-fsgd2104-2-e'):
            for period in ('10','5','3.33'):
                rel=f'data/ApplicationDatabases/{device}_{period}/{k}_{device}_{period}.sqlite'
                db=upstream/rel;context={**artifacts[rel],'device':device,'clock':period,'rows':0}
                connection=sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True)
                size=1024 if device.startswith('xczu') and period=='3.33' else 128
                for rowid,key,raw in connection.execute('select rowid,key,value from unnamed order by rowid'):
                    identity=(info_lines[0].strip()+'_'+str(counter)+source.suffix,device,float(period));counter+=1
                    if identity not in indexed:raise ValueError('Original version/device/clock does not match canonical CSV')
                    row=indexed[identity];metrics=MetricUnpickler(io.BytesIO(raw)).load()
                    if not isinstance(metrics,dict) or any(not same_number(row[c],metrics[v]) for c,v in QOR.items()):
                        raise ValueError('Original QoR values differ; recovery rejected')
                    translated=decode_vector(key,domains[size]);states={}
                    for label,directive in zip(labels,translated):
                        states[label]=(dictionary._translate_array_directive(directive) if 'array' in directive else dictionary._translate_loop_directive(directive))
                    if any(row[c]!=states[a] for c,a in entry['csv_to_action'].items()):
                        raise ValueError('Existing measured directive differs; recovery rejected')
                    missing={a:states[a] for a in hist}
                    for a,v in missing.items():hist[a][v]+=1
                    records.append(dict(version=identity[0],device=device,clock=period,sqlite_rowid=rowid,
                        sqlite_key=key,states=missing,qor_sha256=object_sha({c:row[c] for c in QOR})))
                    context['rows']+=1
                connection.close();contexts.append(context)
        if counter!=len(rows) or {key_of(r) for r in rows}!={(r['version'],r['device'],float(r['clock'])) for r in records}:
            raise ValueError('Incomplete original measurement coverage')
        path=PROOF/'states'/(k+'.jsonl.gz');(root/path).parent.mkdir(parents=True,exist_ok=True)
        payload=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records).encode()
        (root/path).write_bytes(gzip.compress(payload,mtime=0))
        details=[];located={label:(line,text) for label,line,text in found}
        for action,counts in hist.items():
            line,text=located[action]
            details.append(dict(action=action,upstream_source_line=line,upstream_statement=text,
                metadata_record=info_lines[labels.index(action)+1],dse_vector_index=labels.index(action),
                observed_states=dict(sorted(counts.items())),historically_variable=len(counts)>1,
                canonical_state_proven=source_safe,
                root_cause='Full SQLite DSE vector projected through incomplete compiler-feature APL; exporter skips unmapped labels'))
        entries.append(dict(kernel=k,split=entry['split'],canonical_eligible=source_safe,
            contract=CONTRACT if source_safe else 'historical-upstream-only-source-identity-blocked',
            missing_actions=entry['unproven_source_actions'],source_actions=labels,csv_to_action=entry['csv_to_action'],
            canonical_measurements=entry['files']['measurements']['path'],canonical_files=files,
            source_identity=compatibility,actions=details,states=dict(path=str(path),sha256=sha(root/path),rows=len(records)),
            databases=contexts,matching_qor_and_existing_directives=True,clamped_indices=0,imputed_states=0))
        print(k,'rows='+str(counter),'canonical_safe='+str(source_safe),flush=True)
    for repair in metadata_repairs:
        target=root/repair['path'];up=upstream/'data/ApplicationDataset'/target.parent.name/'kernel_info.txt'
        if sha(target)!=repair['before_sha256']:raise ValueError('Metadata changed during audit')
        target.write_bytes(up.read_bytes())
    receipt=dict(schema='salt-hls-gnosis-action-recovery-v1',base_commit='5fa470038b93427f24e2582a075c1eafe74eef35',
        upstream_repository='https://github.com/aferikoglou/GNWSIS',upstream_commit=PIN,
        generator_repository='https://github.com/aferikoglou/GenHLSOptimizer',generator_commit='7acb82f43f1bd94cf502e4fb384ca6c648e5a11c',
        directive_dictionary_sha256=sha(code),split=before['split'],metadata_repairs=metadata_repairs,
        test_rows_read=False,raw_measurements_modified=False,imputed_states=0,kernels=entries,
        summary=dict(audited_kernels=len(entries),matched_rows=sum(e['states']['rows'] for e in entries),
            historical_action_histories=sum(len(e['actions']) for e in entries),
            newly_canonical_eligible=sum(e['canonical_eligible'] for e in entries),
            canonical_recovered_actions=sum(len(e['actions']) for e in entries if e['canonical_eligible']),
            blocked_source_actions=sum(len(e['actions']) for e in entries if not e['canonical_eligible']),
            fixed_or_non_dse_actions=0))
    write_json(root/REGISTRY,receipt)
    for entry in entries:
        if entry['canonical_eligible']:
            if sum(1 for _ in complete_rows(entry['kernel'],root))!=entry['states']['rows']:
                raise ValueError('Recovered complete-row loader did not reproduce every configuration')
    print(json.dumps(receipt['summary'],indent=2))


def verify_recovery(root=ROOT):
    registry=load_registry(root)
    before=json.loads((root/PROOF/'before/file_hashes.json').read_text())
    receipt=json.loads((root/REGISTRY).read_text())
    allowed={r['path']:r['after_sha256'] for r in receipt['metadata_repairs']}
    protected={p:d for p,d in before.items() if p not in ('Data/manifest.json','Data/preprocessed_CSVS/manifest.json')}
    for path,digest in protected.items():
        if sha(root/path)!=allowed.get(path,digest):
            raise ValueError('Unexpected original input mutation: '+path)
    for record in json.loads((root/PROOF/'code_blobs.json').read_text()):
        raw=(root/record['destination']).read_bytes()
        if hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()!=record['blob']:
            raise ValueError('Pinned upstream code blob changed')
    complete=0;historical=0
    for k,entry in registry.items():
        validated_entry(k,root)
        with gzip.open(root/entry['states']['path'],'rt') as stream:
            records=[json.loads(line) for line in stream]
        with (root/entry['canonical_measurements']).open(newline='') as handle:
            measured=list(csv.DictReader(handle))
        indexed={key_of(r):r for r in measured}
        seen=set()
        for r in records:
            key=(r['version'],r['device'],float(r['clock']))
            if key in seen or key not in indexed or r['qor_sha256']!=object_sha({c:indexed[key][c] for c in QOR}):
                raise ValueError('Invalid original measurement association')
            if set(r['states'])!=set(entry['missing_actions']) or any(not v for v in r['states'].values()):
                raise ValueError('Missing historical action state')
            seen.add(key)
        if seen!=set(indexed):raise ValueError('Incomplete sidecar coverage')
        historical+=len(records)
        if entry['canonical_eligible']:
            complete+=sum(1 for _ in complete_rows(k,root))
    result=dict(status='PASS',historical_measured_rows_verified=historical,canonical_complete_rows_verified=complete,
        all_55_raw_csvs_byte_unchanged=True,source_and_headers_byte_unchanged=True,
        metadata_files_restored=len(allowed),metadata_records_restored=sum(len(r['added_records']) for r in receipt['metadata_repairs']),
        split_byte_unchanged=True,locked_test_rows_read=False,imputed_states=0)
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify:verify_recovery()
    elif args.upstream is None:p.error('--upstream is required for historical reconstruction')
    else:recover(args.upstream)


if __name__=='__main__':main()
