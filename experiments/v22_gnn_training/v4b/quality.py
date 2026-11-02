"""V4b reported-zero ADP validity v1, distinct from frozen legacy v2.

Literal all-zero utilization percentages do not establish exact zero area.
Keep physical labels and pool identities, but do not order ADP using these
interval/precision-ambiguous values. No missing resource count is invented.
"""
from collections import Counter
import numpy as np
from .preferences import preference_pairs

POLICY = 'reported-zero-adp-ambiguous-v1'


def adp_valid(row):
    return bool(np.isfinite(row['physical']).all() and sum(row['physical'][1:]) > 0)


def apply_quality(prepared):
    counts=Counter(); details=[]
    for group in prepared['groups']:
        valid=[i for i,r in enumerate(group['rows']) if adp_valid(r)]
        counts[group['split']+'_observations'] += len(group['rows'])
        counts[group['split']+'_ambiguous_zero_area'] += len(group['rows'])-len(valid)
        for fraction in group['budget_pairs']:
            pairs,_=preference_pairs([group['rows'][i] for i in valid],fraction)
            group['budget_pairs'][fraction]=[(valid[a],valid[b],reason) for a,b,reason in pairs]
        details.append({'context':group['key'],'split':group['split'],'family':group['family'],
                        'adp_valid_candidates':len(valid),'physical_candidates':len(group['rows']),
                        'has_any_meaningful_pair':any(group['budget_pairs'].values()),
                        'ambiguous_hashes':[r['config_hash'] for r in group['rows'] if not adp_valid(r)]})
    prepared['quality_policy']=POLICY
    return {'policy':POLICY,'counts':dict(counts),'contexts':details,
            'physical_labels_changed':False,'candidate_pool_identities_changed':False,
            'test_kernels_loaded':[]}



def full_audit(release, authenticated_audit, output):
    import csv, json
    from pathlib import Path
    from experiments.v22_gnn_training import train, checkpointing
    from .audit import _coverage, _family_map, _target_pairs
    split=json.loads(train.SPLIT.read_text())
    eligibility=json.loads((Path(__file__).parent/'receipts_v2/full_cohort/eligibility.json').read_text())
    names=eligibility['training_kernels']+eligibility['validation_kernels']
    data=train.PreparedData(release,'/tmp/v4b-unused-quality-audit.json',selected_kernels=names,
                            authenticated_audit_path=authenticated_audit)
    counts=Counter(); per_kernel={}; valid=[]
    for name in names:
        rows=[r for r in data.records if r['kernel']==name]
        zeros=[r for r in rows if not adp_valid(r)]
        per_kernel[name]={'split':data.kernel_split[name],'physical_records':len(rows),
                          'adp_positive_records':len(rows)-len(zeros),'ambiguous_zero_area_records':len(zeros)}
        counts[data.kernel_split[name]+'_ambiguous_zero_area_records'] += len(zeros)
        valid.extend(r for r in rows if adp_valid(r))
    coverage=_coverage(valid,_family_map(split))
    present={(r['kernel'],r['dimension'],tuple(r['left']),tuple(r['right'])) for r in coverage['target_pairs']}
    families=_family_map(split)
    for name in names:
        for dimension,left,right in _target_pairs():
            if (name,dimension,left,right) not in present:
                coverage['target_pairs'].append({'kernel':name,'family':families.get(name),'split':data.kernel_split[name],
                    'dimension':dimension,'left':list(left),'right':list(right),'matched_assignments':0,
                    'status':'unsupported_no_positive_adp_measurements'})
    source_evidence=[]
    # Trace every selected PathFinder zero to original literal source fields.
    for name in split['val_kernels']:
        if 'pathfinder' not in name: continue
        path=train.ROOT/'Data/CSVS'/f'{name}.csv'
        if not path.exists(): continue
        with path.open() as stream:
            for row in csv.DictReader(stream):
                fields=('BRAM_Utilization_percentage','DSP_Utilization_percentage',
                        'FF_Utilization_percentage','LUT_Utilization_percentage')
                if all(float(row[k])==0 for k in fields):
                    source_evidence.append({'kernel':name,'version':row['Version'],'device':row['Device'],
                        'clock_ns':row['Clock_Period_nsec'],'latency_ms':row['Latency_msec'],
                        'literal_resource_strings':[row[k] for k in fields]})
    report={'schema':'v4b-global-zero-area-quality-audit-v1','policy':POLICY,'counts':dict(counts),
            'audit_script_sha256':train.overlay.file_hash(Path(__file__)),
            'authenticated_audit_sha256':train.overlay.file_hash(authenticated_audit),
            'dataset_sha256':data.audit['dataset']['dataset_sha256'],
            'family_split_sha256':data.audit['dataset']['split_sha256'],
            'graph_archive_sha256':data.audit['graph_release']['archive_sha256'],
            'per_kernel':per_kernel,'positive_adp_coverage':coverage,'raw_pathfinder_zero_records':source_evidence,
            'original_physical_labels_preserved':True,'test_kernels_loaded':[],
            'interpretation':'Source percentage zeros lack sub-percent counts/precision metadata; exclude from ADP comparisons, retain reported physical regression labels. Positive-ADP coverage is a separately versioned validity sensitivity.'}
    checkpointing.atomic_json(Path(output),report)
    print(dict(counts))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-dir',required=True);p.add_argument('--authenticated-audit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();full_audit(a.release_dir,a.authenticated_audit,a.output)
