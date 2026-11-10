"""Create immutable corrected train/development tables and pretraining pool receipts."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pandas as pd
from Preprocessing import data_preprocess as pre
from Preprocessing.create_jsonl import application_sources, load_actions
from experiments.paper_gnn.data import ROOT, PaperData, sha, fingerprint, NUMERIC
from experiments.paper_gnn.train import write_json


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'candidate_pool_manifest.json').exists():
        raise ValueError('Frozen candidate manifest already exists; do not mutate cohort')
    split=json.loads((ROOT/'mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    selected=set(split['train_kernels']+split['val_kernels'])
    cohort=out/'corrected_cohort_v1';cohort.mkdir()
    manifest={'schema':'mailohls-qor-preprocessing-v1','mode':'llm',
        'utilization_policy':'reported_percentages_unchanged',
        'area_metric':'arithmetic_mean_of_bram_dsp_ff_lut_percentages',
        'retain_measured_overflow':True,'kernels':[],
        'split_sha256':sha(ROOT/'mailohls_runs/mailohls_final_family_split_s123.json'),
        'locked_test_read':False,
        'measurement_policy':'positive finite latency <=100000ms, all nonnegative finite resource labels, positive finite synthesis duration; above range ambiguous, no invented failure latency',
        'replicas':'one physical design; medians of authentic repeated measurements following original preprocessor; individual values preserved in raw_provenance.jsonl',
        'resource_units':'raw rounded reported percentages; physical resource supervision uses fractional capacity-scaled estimates, never exact integer counts'}
    sources=application_sources(selected); mapping=[]
    for kernel in sorted(selected):
        raw=ROOT/'Data/CSVS'/(kernel+'.csv')
        processed,stats=pre.preprocess_kernel(raw,'llm','',10.,.1,2.,retain_measured_overflow=True)
        table=cohort/('preprocessed-'+kernel+'.csv');processed.to_csv(table,index=False,float_format='%.12g')
        stats.update(input_sha256=sha(raw),output_sha256=sha(table),split='train' if kernel in split['train_kernels'] else 'val')
        legacy=ROOT/'LLM_branch/Data/preprocessed_CSVS'/table.name
        stats['legacy_preprocessed_rows']=sum(1 for _ in legacy.open())-1
        actions=load_actions(kernel,sources[kernel].read_text())
        missing=[a.action_id for a in actions if not a.csv_column]
        stats['missing_action_columns']=missing
        stats['complete_mapping_rows_before_domain_checks']=len(processed) if not missing else 0
        mapping.append({'kernel':kernel,'missing_actions':missing,'source_sha256':sha(sources[kernel]),'metadata_sha256':sha(sources[kernel].parent/'kernel_info.txt'),'apl_sha256':sha(ROOT/'Data/ApplicationAPLMapping'/(kernel+'.txt'))})
        manifest['kernels'].append(stats)
        # Preserve actual measured numbers, row identity, directives and raw strings.
        import csv
        with raw.open() as src,(cohort/(kernel+'.raw_provenance.jsonl')).open('w') as dest:
            for i,row in enumerate(csv.DictReader(src)):
                dest.write(json.dumps({'source_sha256':stats['input_sha256'],'raw_row_index':i,'raw_record':row},sort_keys=True)+'\n')
        stats['raw_provenance_sha256']=sha(cohort/(kernel+'.raw_provenance.jsonl'))
        print(json.dumps(stats),flush=True)
    write_json(manifest,cohort/'preprocessing_manifest.json')
    os.environ['SALT_CORRECTED_COHORT']=str(cohort)
    data=PaperData(out)
    write_json(data.identity,out/'data_identity.json')
    pool=[]
    for key,records in sorted(data.groups.items()):
        for r in records:
            pool.append({k:r[k] for k in ('split','kernel','family','device','clock','design_id','row_index')})
    with (out/'candidate_pool.jsonl').open('w') as f:
        for r in pool:f.write(json.dumps(r,sort_keys=True)+'\n')
    coverage=[]
    for key,records in sorted(data.groups.items()):
        coverage.append({'split':records[0]['split'],'kernel':key[0],'family':records[0]['family'],'device':key[1],'clock':key[2],'unique_configurations':len(records)})
    write_json({'schema':'salt-final-latency-frozen-candidate-pools-v1','candidate_pool_sha256':sha(out/'candidate_pool.jsonl'),'corrected_manifest_sha256':sha(cohort/'preprocessing_manifest.json'),'policy_sha256':sha(Path(__file__).parent/'predeclared_policy.json'),'coverage':coverage,'data_identity_sha256':fingerprint(data.identity),'full_tcad_split':{tag:len(split[tag+'_jsonl_idx']) for tag in ('train','val','test')},'eligible_counts':dict(Counter(r['split'] for r in pool)),'locked_test_labels_read':False},out/'candidate_pool_manifest.json')
    write_json({'kernels':manifest['kernels'],'directive_mapping':mapping,'coverage':coverage,'deduplication':data.coverage,'eligibility':'all source actions must have an authentic APL column plus legal complete assignments','sort_radix':'L6 init_1 loop absent from original APL mapping; raw table has extra columns but no authenticated identity to L6. Historical csvrow indices do not authenticate a missing directive. Excluded; never assume NDIR.'},out/'cohort_reconciliation.json')
    # Audit the exact authenticated fields reaching the shared numerical path.
    audit=[]
    for k,g in sorted(data.graphs.items()):
        available=Counter(name for _,a in g.nodes(data=True) for name in a)
        audit.append({'kernel':k,'graph_sha256':data.graph_hashes[k],'nodes':len(g),'edges':g.number_of_edges(),'available_fields':dict(available),'numerical_fields_present':{n:available[n] for n in NUMERIC},'node_features':data.tensors[k].x.shape[1],'edge_features':data.tensors[k].edge_attr.shape[1],'action_slots':data.tensors[k].action_slots.tolist(),'relations':dict(Counter(a['relation_type'] for _,_,a in g.edges(data=True)))})
    write_json({'old_defects':['loop_lower/upper/step requested using nonexistent aliases','ssa_use_count requested as use_count','authenticated memory shape magnitude omitted','SSA and memory categories retained as generic node category only'], 'shared_repairs':'correct exact field names; encode authenticated SSA/type/action/root/feature categories and static memory volume; candidate-independent allowlist; no architecture change','checks':audit},out/'structural_audit.json')
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
