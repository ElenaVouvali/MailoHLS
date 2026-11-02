"""Summarize completed v2 runs without pooling budget views as replications."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def summary(run):
    root=Path(run)
    receipt=json.loads((root/'run_receipt.json').read_text())
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Only completed runs may be summarized as results')
    arms={}
    for arm, report in receipt['training'].items():
        selected=report['selected_validation']
        history=report['history']
        recent=history[-min(250,len(history)):]
        ablations={}
        baseline={(r['context'],r['budget_fraction']):r for r in selected['contexts']}
        for mode, value in report['ablations'].items():
            changed=sum(r['selected_hash'] != baseline[(r['context'],r['budget_fraction'])]['selected_hash']
                        for r in value['contexts'])
            ablations[mode]={'family_macro':value['family_macro'], 'selection_score':value['selection_score'],
                             'selection_changed_contexts':changed, 'context_count':len(value['contexts'])}
        physical=selected['physical_contexts']
        family_errors={family:np.mean([r['mae_latency_bram_dsp_ff_lut'] for r in physical if r['family']==family],axis=0).tolist()
                       for family in sorted({r['family'] for r in physical})}
        arms[arm]={'selected_step':report['selected_step'], 'parameter_count':report['parameter_count'],
                   'elapsed_seconds':report['elapsed_seconds'], 'peak_gpu_allocated_bytes':report['peak_gpu_allocated_bytes'],
                   'family_macro':selected['family_macro'], 'per_family':selected['per_family'],
                   'selection_score':selected['selection_score'],
                   'physical_mae_family_macro_latency_bram_dsp_ff_lut':np.mean(list(family_errors.values()),axis=0).tolist(),
                   'sampled_last250_preupdate_pair_accuracy':float(np.mean([r['pairwise_loss']<math.log(2) for r in recent])),
                   'sampled_last250_physical_loss':float(np.mean([r['physical_qor_loss'] for r in recent])),
                   'initial_sampled_loss':history[0]['loss'], 'last_sampled_loss':history[-1]['loss'],
                   'meaningful_budget_transition_count':len(selected['meaningful_budget_transitions']),
                   'correct_meaningful_budget_transitions':sum(r['correct'] for r in selected['meaningful_budget_transitions']),
                   'ablations':ablations,'controlled_action_probe':report['controlled_action_probe']}
    return {'schema':'v4b-evaluation-summary-v2','run':str(root),'source_commit':receipt['source_commit'],
            'protocol_fingerprint':receipt['protocol_fingerprint'],'manifest_sha256':receipt['manifest_sha256'],
            'sample_trace_sha256':receipt['sample_trace_sha256'],'sample_coverage':receipt['sample_coverage'],
            'arms':arms,'test_kernels_loaded':[],
            'scope':'family-macro fixed measured-pool selection; two held-out families, one seed; budget views correlated'}


def compare(primary, reference):
    p=json.loads((Path(primary)/'protocol.json').read_text())
    r=json.loads((Path(reference)/'protocol.json').read_text())
    for key in ('version','seed','updates','validate_every','pair_weight','lr','weight_decay','source_sha256','torch','gpu_uuid'):
        if p[key] != r[key]:
            raise ValueError(f'Unmatched protocol field: {key}')
    a=summary(primary); b=summary(reference)
    for arm in a['arms']:
        pa=json.loads((Path(primary)/f'{arm}_report.json').read_text())['selected_validation']['contexts']
        pb=json.loads((Path(reference)/f'{arm}_report.json').read_text())['selected_validation']['contexts']
        # Pool identity is verified from physical-context hashes, independently
        # of selected winners or budget outcomes.
        xa=json.loads((Path(primary)/f'{arm}_report.json').read_text())['selected_validation']['physical_contexts']
        xb=json.loads((Path(reference)/f'{arm}_report.json').read_text())['selected_validation']['physical_contexts']
        identity=lambda rows:sorted((x['kernel'],x['device'],x['clock_ns'],tuple(x['assignment_hashes'])) for x in rows)
        if identity(xa) != identity(xb):
            raise ValueError('Held-out candidate pools differ')
    return {'protocol_matched':True,'validation_pools_identical':True,'primary':a,'reference':b}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--reference-run'); p.add_argument('--output',required=True)
    args=p.parse_args()
    result=compare(args.run,args.reference_run) if args.reference_run else summary(args.run)
    with open(args.output,'x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    arms=result.get('arms',result.get('primary',{}).get('arms',{}))
    for arm,r in arms.items():
        print(arm,r['selected_step'],r['selection_score'],r['family_macro'])


if __name__=='__main__':
    main()
