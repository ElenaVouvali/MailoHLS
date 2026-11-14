"""Fail-closed qualification of recovered Stage-2 supervision and frozen FiLM data."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'film-gmt-recovered-supervision-v1'
ORIGINAL_BANK_SHA = '2b5745ab029a32f3f0088367b80a7478212249912a7ee8e7b890fdaffc2cbf0c'
PARENT_REGISTRY_SHA = '41d0f6f90db4c2370bdc02fbcdda4e114ec65cf57b89a567bfab56606d8d0c1c'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def source_inventory(root=ROOT):
    manifest=json.loads((root/'Data/manifest.json').read_text())
    split=json.loads((root/manifest['split']['path']).read_text())
    if sha(root/manifest['split']['path'])!=manifest['split']['sha256']:
        raise ValueError('Family split identity changed')
    selected={e['kernel']:e for e in manifest['kernels']
              if e['complete_source_configuration_eligible'] and e['split']!='test'}
    if collections.Counter(e['split'] for e in selected.values())!={'train':49,'val':4}:
        raise ValueError('Expanded cohort must contain exactly 49 train and four validation sources')
    for k,e in selected.items():
        if k not in split[e['split']+'_kernels']:raise ValueError('Split mismatch: '+k)
        for record in e['files'].values():
            if sha(root/record['path'])!=record['sha256']:raise ValueError('Input hash mismatch: '+record['path'])
    return manifest,selected


def qualification(parent_registry,out,root=ROOT):
    """Only hashes/metadata are read here; no locked or non-test QoR rows are parsed."""
    if Path(out).exists():raise FileExistsError('Qualification output must be fresh')
    if sha(parent_registry)!=PARENT_REGISTRY_SHA:raise ValueError('Frozen V4 registry changed')
    manifest,selected=source_inventory(root)
    registry=json.loads(Path(parent_registry).read_text())['kernels']
    blockers={};coverage={}
    for k,e in sorted(selected.items()):
        definitions=[line.split(',') for line in (root/e['files']['metadata']['path']).read_text().splitlines()[1:] if line.strip()]
        expected={f'AUTO{{_{kind}_{a}}}' for a,t,*rest in definitions
                  for kind in (('PIPE','UNROLL') if t=='loop' else ('ARRAY_T','ARRAY_F','ARRAY_D'))}
        actual=registry.get(k.lower().replace('-','_'),{})
        missing=sorted(expected-set(actual))
        if missing:blockers[k]={'reason':'source_action_missing_from_frozen_V4_registry','missing_lhs':missing,'split':e['split']}
        coverage[k]={'split':e['split'],'actions':len(definitions),'lhs_count':len(expected),
                     'source_sha256':e['files']['source']['sha256'],
                     'recovery_contract':e['complete_source_configuration_contract'],
                     'parent_missing_lhs':missing}
    result={'schema':'film-gmt-expanded-qualification-v1',
        'status':'NOT_READY' if blockers else 'SOURCE_DOMAINS_PASS',
        'expected_train_kernels':49,'expected_val_kernels':4,'kernel_inventory':coverage,
        'blockers':blockers,'canonical_manifest_sha256':sha(root/'Data/manifest.json'),
        'parent_registry_sha256':sha(parent_registry),'locked_test_qor_rows_read':False,
        'non_test_qor_rows_read':False,'aggregation_or_selection_started':False,
        'separate_measured_knn':'Preserved at Data/provenance/gnosis_5_compilation_20261113; excluded from canonical bank',
        'decision':'Do not invent domains, drop source actions, or silently subset the recovered cohort.'}
    write(Path(out)/'qualification.json',result)
    return result


def joined_configurations(kernel,root=ROOT):
    """Complete assignments MUST be joined before any downstream aggregation."""
    from Preprocessing import gnosis_recovery as rec
    manifest=json.loads((root/'Data/manifest.json').read_text())
    entry=next(e for e in manifest['kernels'] if e['kernel']==kernel)
    if entry['split']=='test':raise ValueError('Locked-test QoR rows must not be opened')
    if not entry['complete_source_configuration_eligible']:raise ValueError('Not a complete canonical source')
    if entry['action_state_recovery']:
        yield from rec.complete_rows(kernel,root)
        return
    source=(root/entry['files']['source']['path']).read_text()
    import re
    expected=set(re.findall(r'\b(L[1-9][0-9]*)\s*:',source))
    item=entry['files']['measurements'];path=root/item['path']
    if sha(path)!=item['sha256']:raise ValueError('Raw measurement hash mismatch')
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            states={a:row[c].strip() for c,a in entry['csv_to_action'].items()}
            if set(states)!=expected or any(not value for value in states.values()):
                raise ValueError('Unknown or incomplete measured source assignment')
            yield {'measurement':row,'source_action_states':states,
                   'contract':entry['complete_source_configuration_contract']}


def verify_recovered_bank(path,parent_registry,root=ROOT):
    """Additional contract; never relaxes authentication of the selected Stage-1."""
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['schema']!=SCHEMA or m['status']!='QUALIFIED_COMPLETE_COHORT':
        raise ValueError('Not a qualified expanded recovery bank')
    if m['stage1_training_bank_manifest_sha256']!=ORIGINAL_BANK_SHA:
        raise ValueError('Recovered bank references a different selected FiLM Stage-1')
    if sha(parent_registry)!=PARENT_REGISTRY_SHA or m['parent_registry_sha256']!=PARENT_REGISTRY_SHA:
        raise ValueError('Original V4 domain authentication changed')
    _,selected=source_inventory(root)
    for name,digest in m['inputs_sha256'].items():
        p=Path(name);p=p if p.is_absolute() else root/p
        if sha(p)!=digest:raise ValueError('Recovery provenance mismatch: '+name)
    for name,digest in m['outputs_sha256'].items():
        if sha(path/name)!=digest:raise ValueError('Recovered bank file mismatch: '+name)
    if set(m['source_kernels'])!=set(selected):raise ValueError('Incomplete recovered cohort')
    for k,e in selected.items():
        if m['source_kernels'][k]['source_sha256']!=e['files']['source']['sha256']:
            raise ValueError('Recovered source identity mismatch: '+k)
        if m['source_kernels'][k]['contract']!=e['complete_source_configuration_contract']:
            raise ValueError('Recovered action-history contract mismatch: '+k)
    return m


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent-registry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=qualification(a.parent_registry,a.out)
    print(json.dumps({'status':result['status'],'blocked_kernels':list(result['blockers'])},indent=2))
    raise SystemExit(2 if result['blockers'] else 0)
