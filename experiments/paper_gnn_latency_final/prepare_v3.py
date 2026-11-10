"""Reuse corrected measurements; freeze only the required input compatibility repair."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from experiments.paper_gnn.data import sha,fingerprint
from experiments.paper_gnn.train import write_json
from experiments.paper_gnn_latency_final.dataset_v3 import CompletePaperData

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--measurements',type=Path,required=True);a=p.parse_args();root=a.root;root.mkdir();cohort=root/'cohort';cohort.mkdir()
    (cohort/'corrected_cohort_v1').symlink_to(a.measurements/'corrected_cohort_v1',target_is_directory=True)
    (cohort/'cohort_reconciliation.json').symlink_to(a.measurements/'cohort_reconciliation.json')
    os.environ['SALT_CORRECTED_COHORT']=str(cohort/'corrected_cohort_v1')
    data=CompletePaperData(cohort)
    pool=[]
    for key,records in sorted(data.groups.items()):
        for r in records:pool.append({k:r[k] for k in ('split','kernel','family','device','clock','design_id','row_index')})
    with (cohort/'candidate_pool.jsonl').open('w') as f:
        for r in pool:f.write(json.dumps(r,sort_keys=True)+'\n')
    coverage=[dict(split=r[0]['split'],kernel=k[0],family=r[0]['family'],device=k[1],clock=k[2],unique_configurations=len(r)) for k,r in sorted(data.groups.items())]
    manifest={'schema':'salt-final-latency-frozen-candidate-pools-v3','candidate_pool_sha256':sha(cohort/'candidate_pool.jsonl'),'corrected_manifest_sha256':sha(cohort/'corrected_cohort_v1/preprocessing_manifest.json'),'policy_sha256':sha(Path(__file__).parent/'predeclared_policy.json'),'coverage':coverage,'data_identity_sha256':fingerprint(data.identity),'full_tcad_split':{tag:len(data.split[tag+'_jsonl_idx']) for tag in ('train','val','test')},'eligible_counts':dict(Counter(r['split'] for r in pool)),'source_only_unbound_nodes':data.source_only_nodes,'locked_test_labels_read':False,'supersedes_unqualified_attempt':'A v2 trained but failed full registered-source XAttn routing qualification; optimizer/loss/architecture/data/seed not tuned','training_measurements_reused_unchanged':True}
    write_json(data.identity,cohort/'data_identity.json');write_json(manifest,cohort/'candidate_pool_manifest.json')
    amendment=root/'evaluation_amendment';amendment.mkdir()
    val=[r for r in pool if r['split']=='val'];write_json({'schema':'salt-final-four-kernel-evaluation-pretraining-v3','candidate_identity_sha256':fingerprint(sorted((r['kernel'],r['device'],r['clock'],r['design_id']) for r in val)),'unique_observed_configurations':len(val),'kernel_counts':dict(Counter(r['kernel'] for r in val)),'families':sorted({r['family'] for r in val}),'ignored_action':{'machsuite-sort-radix':['L6']},'frozen_before_both_final_arms':True},amendment/'manifest.json')
    print(json.dumps(manifest,indent=2))
