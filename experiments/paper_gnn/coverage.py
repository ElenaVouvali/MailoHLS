"""Measurement/cohort coverage without parsing locked-test measurements."""
import argparse
import json
from pathlib import Path
from Preprocessing.create_jsonl import application_sources, load_actions
from LLM_branch.common.mailohls_contract import family_id_from_kernel_name
from experiments.paper_gnn.data import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    identity=json.loads(a.identity.read_text())
    split=json.loads((ROOT/'mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    manifest_path=ROOT/'LLM_branch/Data/preprocessed_CSVS/preprocessing_manifest.json'
    manifest=json.loads(manifest_path.read_text());table={r['kernel']:r for r in manifest['kernels']}
    selected=set(split['train_kernels']+split['val_kernels']);sources=application_sources(selected)
    eligible={k:not any(not x.csv_column for x in load_actions(k,sources[k].read_text())) for k in selected}
    cohorts={}
    for name in ('train','val'):
        kernels=split[name+'_kernels'];used=[k for k in kernels if eligible[k]]
        cohorts[name]={'locked_split_kernels':kernels,'eligible_kernels':used,
            'eligible_families':sorted({family_id_from_kernel_name(k) for k in used}),
            'raw_rows':sum(table[k]['raw_rows'] for k in kernels),
            'valid_measurements_before_replica_aggregation':sum(table[k]['valid_rows'] for k in kernels),
            'preprocessed_unique_measurements':sum(table[k]['output_rows'] for k in kernels),
            'rows_excluded_missing_action_columns':sum(table[k]['output_rows'] for k in kernels if not eligible[k])}
    report={'schema':'salt-paper-gnn-coverage-v1','preprocessing_manifest_sha256':sha(manifest_path),
        'data_identity_sha256':sha(a.identity),'cohorts':cohorts,'accepted':identity['coverage'],
        'invalid_unknown_policy':'Raw validity-filter exclusions are not resource-infeasibility labels. The upstream manifest does not separate failed-HLS versus unknown-resource reason counts.',
        'timing':'Not qualified by these tables; requested clock is not an achieved-timing receipt.',
        'resources':'Accepted rows have finite reported resource percentages; physical counts are reconstructed from authenticated capacities.',
        'locked_test_measurements_parsed':False}
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
