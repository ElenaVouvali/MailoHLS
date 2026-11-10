"""Read-only reconciliation of source, raw measurements and frozen candidate identities."""
import argparse
from collections import Counter
import json
from pathlib import Path
from experiments.paper_gnn.data import ROOT,sha
from experiments.paper_gnn.train import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    split=json.loads((ROOT/'mailohls_runs/mailohls_final_family_split_s123.json').read_text());manifest=json.loads((root/'cohort/corrected_cohort_v1/preprocessing_manifest.json').read_text());frozen=json.loads((root/'cohort/candidate_pool_manifest.json').read_text());tables=root/'cohort/corrected_cohort_v1'
    expected=set(split['train_kernels']+split['val_kernels']);assert {r['kernel'] for r in manifest['kernels']}==expected
    assert not expected.intersection(split['test_kernels'])
    assert sha(tables/'preprocessing_manifest.json')==frozen['corrected_manifest_sha256'];assert sha(root/'cohort/candidate_pool.jsonl')==frozen['candidate_pool_sha256']
    totals={};checks=[]
    for record in manifest['kernels']:
        kernel=record['kernel'];assert sha(ROOT/'Data/CSVS'/(kernel+'.csv'))==record['input_sha256'];assert sha(tables/('preprocessed-'+kernel+'.csv'))==record['output_sha256'];assert sha(tables/(kernel+'.raw_provenance.jsonl'))==record['raw_provenance_sha256']
        checks.append({'kernel':kernel,'raw_table_hash_matches':True,'corrected_table_hash_matches':True,'all_raw_measurement_provenance_hash_matches':True})
    pool=[json.loads(line) for line in (root/'cohort/candidate_pool.jsonl').read_text().splitlines()]
    identities=[tuple(r[n] for n in ('kernel','device','clock','design_id')) for r in pool];assert len(set(identities))==len(identities)
    for tag in ('train','val'):
        rows=[r for r in manifest['kernels'] if r['split']==tag];accepted=[r for r in pool if r['split']==tag]
        totals[tag]={name:sum(r[name] for r in rows) for name in ('raw_rows','successful_measured_rows','recovered_overcapacity_rows','legacy_preprocessed_rows','output_rows')}
        rejected=Counter()
        for r in rows:rejected.update(r['rejected_by_reason'])
        totals[tag].update(rejected_by_reason=dict(rejected),accepted_unique_observed_configurations=len(accepted),eligible_kernels=len({r['kernel'] for r in accepted}),eligible_families=len({r['family'] for r in accepted}),full_split_kernels=len(split[tag+'_kernels']),full_split_families=len(split[tag+'_families']))
    write_json({'status':'passed','source_commit':'20028e7','split_sha256':sha(ROOT/'mailohls_runs/mailohls_final_family_split_s123.json'),'release_manifest_sha256':sha(ROOT/'docs/graph-release-v22-rc3.json'),'archive_sha256':json.loads((ROOT/'docs/graph-release-v22-rc3.json').read_text())['archive_sha256'],'totals':totals,'per_kernel_checks':checks,'duplicate_candidate_identities':0,'locked_test_measurements_deserialized':False,'sort_radix_qualification':'ten observed action assignments, L6 explicitly ignored by user; not an authenticated eleventh directive'},root/'data_provenance_report.json')
    print(json.dumps(totals,indent=2))
