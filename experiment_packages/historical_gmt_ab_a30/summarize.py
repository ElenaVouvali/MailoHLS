#!/usr/bin/env python3
"""Conservative A/B analysis of matched Stage-2 validation-selection JSON.

Never manufacture missing results or silently replace a failed arm.
"""
from pathlib import Path
import argparse,json
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
assert (a.run/'prepared/PREPARATION.json').exists(),'Missing authenticated preparation receipt'
prep=json.loads((a.run/'prepared/PREPARATION.json').read_text())
if prep['status']!='READY_FOR_MATCHED_STAGE2':raise RuntimeError('Invalid preparation receipt')
rows=[]
for seed_folder in sorted(a.run.glob('seed_*')):
    seed=seed_folder.name.removeprefix('seed_')
    for arm in ('A','B'):
        directory=seed_folder/arm
        candidates=list(directory.glob('val_selection_step_*.json'))
        selected=directory/'best_custom_stage1'/'best_selection_metrics.json'
        if not selected.exists():
            selected=directory/'best_custom_stage2'/'best_selection_metrics.json'
        if not selected.exists() or not candidates:
            rows.append({'seed':seed,'arm':arm,'status':'INCOMPLETE',
                         'evaluations':len(candidates),'best_path':str(selected)})
            continue
        item=json.loads(selected.read_text())
        summary=item.get('summary') or item
        # Field names are maintained by train_SFT_xattn_new, not guessed.
        fields={name:summary.get(name) for name in (
            'selection_score','joint_action_score','mean_decision_site_accuracy',
            'minimum_kernel_decision_accuracy','mean_value_acc','exact_design_accuracy',
            'budget_counterfactual_changed_site_accuracy',
            'budget_counterfactual_transition_accuracy','budget_counterfactual_decision_accuracy',
            'candidate_margin_mean','teacher_forced_top1_accuracy',
            'directive_semantic_compliance','schema_compliance','per_kernel_decision_accuracy') if name in summary}
        rows.append({'seed':seed,'arm':arm,'status':'COMPLETE','evaluations':len(candidates),
                     'selected_step':item.get('step',item.get('global_step')),
                     'metrics':fields,'best_path':str(selected),
                     'raw_summary_keys':sorted(summary)})
paired=[]
by={(x['seed'],x['arm']):x for x in rows}
for seed in sorted({x['seed'] for x in rows}):
    A=by.get((seed,'A'));B=by.get((seed,'B'))
    if not A or not B or A['status']!='COMPLETE' or B['status']!='COMPLETE':continue
    common=sorted(set(A['metrics'])&set(B['metrics']))
    differences={k:float(B['metrics'][k])-float(A['metrics'][k])
                 for k in common if isinstance(A['metrics'][k],(int,float)) and isinstance(B['metrics'][k],(int,float))}
    paired.append({'seed':seed,'B_minus_A':differences})
report={'schema':'historical-gmt-matched-stage2-report-v1',
        'interpretation':'Decision evidence only. No independent physical QoR synthesis, no final production promotion.',
        'parent':'frozen V4 reference; conclusions must be replayed if final Stage-1 changes',
        'preparation':{'train_families':prep['train_families'],
                       'validation_families':prep['validation_families'],
                       'gnn_sha256':prep['gnn_sha256'],'gmt_sha256':prep['gmt_sha256']},
        'arms':rows,'paired_differences':paired,
        'decision':'NEEDS_REVIEW' if paired else 'INCOMPLETE_NO_DECISION',
        'validity_limits':['Two held-out families may give insufficient independent evidence',
          'HLS physical synthesis and timing closure not measured',
          'GMT summaries are trained on compiler-signature reconstruction, then frozen',
          'Run B-ON/B-summary-masked and generation comparisons before claiming structural-causal improvement']}
out=a.run/'architecture_comparison.json'
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'decision':report['decision'],'paired_seeds':len(paired),
  'completed':[(r['seed'],r['arm'],r['status']) for r in rows],
  'output':str(out)},indent=2))
