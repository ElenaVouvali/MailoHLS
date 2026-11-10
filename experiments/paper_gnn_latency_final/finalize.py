"""Shared matched-pool reevaluation, scientific decision, memory export and report."""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import torch
from experiments.paper_gnn.data import PaperData,sha,fingerprint
from experiments.paper_gnn.train import backend,write_json
from experiments.paper_gnn_latency_final.train import development,gpu_receipt
from experiments.paper_gnn_latency_final.memory_export import export
from GNN_branch.paper_protocol import evaluate

METRICS=('kendall_tau_b','feasible_regret_at_1','log_qor_mae','feasible_selection_rate')

def rowkey(r):return tuple(r[n] for n in ('kernel','device','clock','design_id'))

def save_rows(rows,path):
    with path.open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')

def intervals(report):
    families=report['families'];rng=random.Random(123);result={}
    for metric in METRICS:
        vals=[f[metric] for f in families if f[metric] is not None]
        if not vals:result[metric]=None;continue
        boot=sorted(sum(rng.choice(vals) for _ in vals)/len(vals) for _ in range(2000))
        result[metric]={'lower':boot[49],'upper':boot[1949],'independent_families':len(vals),'warning':'exploratory family bootstrap; few independent families cannot support precise confidence claims'}
    return result

def comparable_improvement(new,old):
    """Conservative implementation of the frozen joint scientific policy."""
    a=new['family_macro'];b=old['family_macro']
    if any(a[m] is None or b[m] is None for m in METRICS):return False,{'reason':'undefined primary metrics'}
    feasibility=a['feasible_selection_rate']>=b['feasible_selection_rate']-.05
    calibration=a['log_qor_mae']<=b['log_qor_mae']*1.1
    ranking=(a['kendall_tau_b']>=b['kendall_tau_b'] and a['feasible_regret_at_1']<=b['feasible_regret_at_1'] and (a['kendall_tau_b']>b['kendall_tau_b'] or a['feasible_regret_at_1']<b['feasible_regret_at_1']))
    per_family={}
    oldfamilies={f['family']:f for f in old['families']}
    for f in new['families']:
        o=oldfamilies.get(f['family'])
        per_family[f['family']]=bool(o and f['feasible_selection_rate'] is not None and o['feasible_selection_rate'] is not None and f['feasible_selection_rate']>=o['feasible_selection_rate']-.05 and ((f['kendall_tau_b'] is not None and o['kendall_tau_b'] is not None and f['kendall_tau_b']>o['kendall_tau_b']) or (f['feasible_regret_at_1'] is not None and o['feasible_regret_at_1'] is not None and f['feasible_regret_at_1']<o['feasible_regret_at_1'])))
    passed=feasibility and calibration and ranking and all(per_family.values())
    return passed,dict(feasibility_nonsevere=feasibility,mae_nonsevere=calibration,ranking_or_selection_gain=ranking,independent_family_support=per_family)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    os.environ['SALT_CORRECTED_COHORT']=str(root/'cohort/corrected_cohort_v1')
    out=root/'final_evaluation';out.mkdir(exist_ok=True)
    data=PaperData(out)
    from experiments.paper_gnn_latency_final.evaluation_cohort import include_radix_ignoring_l6
    amendment=include_radix_ignoring_l6(data,root/'cohort/corrected_cohort_v1');write_json(amendment,out/'evaluation_cohort_amendment.json')
    Model,seed=backend();torch.set_num_threads(4)
    expected=json.loads((root/'B/gpu_receipt.json').read_text())['device'];gpu_receipt('V100' if 'V100' in expected else 'A30')
    new_schema=data.schema;new_tensors=data.tensors;new_mean=data.mean;new_std=data.std
    preliminary=Path(os.environ.get('SALT_PRELIMINARY_LATENCY',str(root.parent/'paper_gnn_20261110/latency/selected.pt')))
    assert sha(preliminary)=='79540c9f27e4807e0c3badbd876322bbde5badd9d4a04f601b316ee75e944ee8'
    rows_by={};states={};checkpoint_paths={'preliminary':preliminary,'A':root/'A/selected.pt','B':root/'B/selected.pt'}
    from experiments.paper_gnn_latency_final.legacy_tensorize import tensorize as old_tensorize
    for name,path in checkpoint_paths.items():
        state=torch.load(path,map_location='cpu',weights_only=False);states[name]=state
        if name=='preliminary':
            data.schema=state['schema'];data.tensors={k:old_tensorize(g,state['schema']['vocabulary']) for k,g in data.graphs.items()}
        else:data.schema=new_schema;data.tensors=new_tensors
        data.mean=state['data_mean'];data.std=state['data_std']
        first=next(iter(data.tensors.values()));model=Model(first.x.shape[1],first.edge_attr.shape[1]).cuda();model.load_state_dict(state['model'],strict=True)
        report,rows=development(model,data,'PARETO_LATENCY','cuda')
        for row in rows:
            if row['kernel']=='machsuite-sort-radix':row.update(ignored_actions=['L6'],candidate_observation_status='observed assignment excluding L6, per user amendment')
        rows_by[name]=rows
        save_rows(rows,out/(name+'_predictions.jsonl'));write_json(report,out/(name+'_all_context_metrics.json'))
        del model
    data.schema=new_schema;data.tensors=new_tensors;data.mean=new_mean;data.std=new_std
    # All three current estimators are reevaluated on the complete corrected pool.
    pools=[{rowkey(r) for r in rows_by[k]} for k in ('preliminary','A','B')]
    amended=json.loads((root/'evaluation_amendment/manifest.json').read_text())
    assert fingerprint(sorted(pools[0]))==amended['candidate_identity_sha256']
    assert pools[0]==pools[1]==pools[2]
    old_rows=[json.loads(l) for l in (preliminary.parent/'selected_predictions.jsonl').read_text().splitlines()]
    original_intersection=pools[0]&{rowkey(r) for r in old_rows}
    historical_path=root/'historical_v1/replayed_predictions.jsonl'
    historical_receipt=root/'historical_v1/replay_receipt.json'
    legacy_rows=[json.loads(l) for l in historical_path.read_text().splitlines()] if historical_receipt.exists() else []
    common_v1=pools[0]&{rowkey(r) for r in legacy_rows}
    protocols={
        'corrected_all_contexts':(None,('preliminary','A','B')),
        'corrected_zcu104_10ns':('slice',('preliminary','A','B')),
        'original_legacy_cohort_intersection':(original_intersection,('preliminary','A','B')),
        'v1_matched_zcu104_10ns_intersection':(common_v1,('preliminary','A','B','V1')),
    }
    if legacy_rows:rows_by['V1']=legacy_rows
    table=[];reports={};pool_manifest={}
    for protocol,(subset,names) in protocols.items():
        reports[protocol]={}
        for name in names:
            if name not in rows_by or (isinstance(subset,set) and not subset):continue
            rows=rows_by[name]
            if subset=='slice':rows=[r for r in rows if r['device']=='xczu7ev-ffvc1156-2-e' and r['clock']==10.]
            elif isinstance(subset,set):rows=[r for r in rows if rowkey(r) in subset]
            if name=='V1':
                # Use the same corrected measured values for all estimators after identity matching.
                truth={rowkey(r):r for r in rows_by['A']}
                for r in rows:
                    ref=truth[rowkey(r)]
                    if not math.isclose(r['log_latency'],ref['log_latency'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Historical/current measured label disagreement')
                    for field in ('resources','log_latency','log_area','measured_resource_precision'):r[field]=ref[field]
            report=evaluate(rows,'PARETO_LATENCY');report['family_grouped_intervals']=intervals(report)
            write_json(report,out/(protocol+'_'+name+'.json'));reports[protocol][name]=report
            ids=sorted({rowkey(r) for r in rows});pool_manifest[protocol]={'unique_physical_candidates':len(ids),'identity_sha256':fingerprint(ids),'coverage':[dict(zip(('kernel','device','clock'),key),configurations=count) for key,count in sorted(Counter(x[:3] for x in ids).items())]}
            table.append({'protocol':protocol,'model':name,'physical_candidates':len(ids),**report['family_macro'],'conditional_regret_pools':report['conditional_regret_pool_count'],'total_budget_pools':len(report['pools']),'no_measured_feasible_candidate_pools':report['impossible_pool_count']})
    write_json(pool_manifest,out/'matched_candidate_pool_manifest.json')
    with (out/'four_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write_json(table,out/'four_metrics.json')
    allreports=reports['corrected_all_contexts'];b_over_a,bdiag=comparable_improvement(allreports['B'],allreports['A']);candidate='B' if b_over_a else 'A'
    upgraded,upgrade_diag=comparable_improvement(allreports[candidate],allreports['preliminary'])
    slice_ok,slice_diag=comparable_improvement(reports['corrected_zcu104_10ns'][candidate],reports['corrected_zcu104_10ns']['preliminary'])
    v1reports=reports['v1_matched_zcu104_10ns_intersection']
    historical_ok,historical_diag=comparable_improvement(v1reports[candidate],v1reports['V1']) if 'V1' in v1reports else (False,{'reason':'No authenticated matched V1 replay; comparative historical upgrade not established'})
    upgraded=upgraded and slice_ok and historical_ok
    selected=candidate if upgraded else 'preliminary';decision='FREEZE_RECOMMENDED' if upgraded else 'FREEZE_NOT_YET_JUSTIFIED'
    # Paired scientific integrity and exposure receipts.
    pa=states['A']['protocol'];pb=states['B']['protocol'];assert pa['initial_weights_sha256']==pb['initial_weights_sha256'];assert pa['data_identity_sha256']==pb['data_identity_sha256']
    sampler=[]
    for epoch in range(min(len(json.loads((root/'A/curves.json').read_text())),len(json.loads((root/'B/curves.json').read_text())))):
        x=json.loads((root/'A'/('sampling_epoch_%02d.json'%epoch)).read_text());y=json.loads((root/'B'/('sampling_epoch_%02d.json'%epoch)).read_text())
        assert x['batches']==y['batches'];sampler.append({'epoch':epoch,'identical_sampled_physical_designs':True})
    write_json({'identical_fresh_initialization_sha256':pa['initial_weights_sha256'],'identical_corrected_data_identity_sha256':pa['data_identity_sha256'],'shared_epoch_exposure':sampler,'early_stopping_may_change_total_exposure':'same prespecified exposure and stopping policy; actual totals reported separately','locked_test_read':False},out/'matched_training_integrity.json')
    # Export chosen estimator under its exact tensor vocabulary and original weights.
    state=states[selected]
    data.schema=state['schema'];data.mean=state['data_mean'];data.std=state['data_std']
    data.tensors=({k:old_tensorize(g,state['schema']['vocabulary']) for k,g in data.tensor_graphs.items()} if selected=='preliminary' else new_tensors)
    if selected=='preliminary':
        data.schema={**state['schema'],'source_only_unbound_action_nodes':new_schema['source_only_unbound_action_nodes'],'export_input_adapter':'complete registered source action coverage v3; same frozen preliminary weights, versioned added static source-only slots'}
    first=next(iter(data.tensors.values()));model=Model(first.x.shape[1],first.edge_attr.shape[1]).cuda();model.load_state_dict(state['model'],strict=True);model.eval()
    selected_dir=root/'selected_export';selected_dir.mkdir(exist_ok=True)
    export(model,data,selected_dir,checkpoint_paths[selected],'PARETO_LATENCY',state['protocol'])
    # Audit access from authenticated compiler features to the actual prediction path.
    probes=[]
    for key,records in sorted(data.groups.items()):
        if records[0]['split']!='val' or key[1]!='xczu7ev-ffvc1156-2-e' or key[2]!=10.:continue
        graph=data.tensors[key[0]].clone().cuda();graph.x.requires_grad_(True);graph.edge_attr.requires_grad_(True)
        prediction=model(graph,records[0]['candidate'].unsqueeze(0).cuda(),data.context(records[0],'cuda'))
        gx,ge=torch.autograd.grad(prediction.sum(),(graph.x,graph.edge_attr))
        probes.append({'kernel':key[0],'node_feature_prediction_gradient_norm':float(gx.norm()),'edge_feature_prediction_gradient_norm':float(ge.norm()),'candidate_independent_static_memory':True})
    write_json(probes,selected_dir/'prediction_path_probes.json')
    from experiments.paper_gnn.compatibility import check
    check(selected_dir/'memory',selected_dir/'xattn_check.json','V100' if 'V100' in expected else 'A30')
    # Preserve all exact receipts and training states; no production registration.
    receipts={}
    for name in ('A','B'):
        completion=json.loads((root/name/'completion.json').read_text());last=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
        receipts[name]={**completion,'last_checkpoint_sha256':sha(root/name/'last.pt'),'selected_checkpoint_sha256':sha(root/name/'selected.pt'),'protocol_sha256':sha(root/name/'protocol.json'),'curves_sha256':sha(root/name/'curves.json'),'rng_saved':['torch_rng','cuda_rng','sampler_rng'],'optimizer_and_scheduler_saved':True,'total_epochs':last['epoch']+1,'total_optimizer_steps':last['steps'],'selected_epoch_one_based':completion['selected_epoch']+1}
    write_json(receipts,out/'training_receipts.json')
    decision_receipt={'status':decision,'selected':selected,'new_arm_candidate':candidate,'B_vs_A':bdiag,'candidate_vs_V1_matched':historical_diag,'candidate_vs_preliminary_all_contexts':upgrade_diag,'candidate_vs_preliminary_zcu104_10ns':slice_diag,'selected_checkpoint_sha256':sha(checkpoint_paths[selected]),'selected_checkpoint_path':str(checkpoint_paths[selected]),'selected_memory_manifest_sha256':sha(selected_dir/'memory/memory_manifest.json'),'xattn_status':'passed','production_promoted':False,'test_used':False,'downstream_llm_improvement_verified':False,'limits':['three development families; limited independent-family uncertainty','Sort-radix L6 explicitly ignored per user amendment; not authenticated','all-context training is not historical single-target training reproduction','resource counts are percentage-derived estimates; exact capacity equality unknown','ambiguous above-range raw latencies excluded conservatively']}
    write_json(decision_receipt,root/'decision.json')
    plot_curves(root,out)
    make_report(root,out,reports,table,decision_receipt,receipts)
    write_json({'status':'completed','decision':decision,'decision_sha256':sha(root/'decision.json'),'report_sha256':sha(root/'SCIENTIFIC_DECISION.md'),'no_promotion':True,'locked_test_read':False},root/'completion.json')
    print(json.dumps(decision_receipt),flush=True)


def plot_curves(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for name in ('A','B'):
        curves=json.loads((root/name/'curves.json').read_text());epochs=[x['epoch']+1 for x in curves]
        for ax,metric in zip(axes.flatten(),('qor_loss','resource_loss','rank_loss','delta_loss','feasible_regret_at_1','log_qor_mae')):
            ax.plot(epochs,[x.get(metric) for x in curves],label=name);ax.set(xlabel='Epoch',ylabel=metric);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=160);fig.savefig(out/'loss_curves.pdf');plt.close(fig)
    gradient_rows=[]
    for name in ('A','B'):
        for file in sorted((root/name).glob('sampling_epoch_*.json')):
            r=json.loads(file.read_text())
            for component in ('qor','resource','ranking','delta'):
                values=[v['component_gradient_norms'][component] for v in r['gradient_samples']]
                weight={'qor':1.,'resource':.1,'ranking':.05,'delta':r['delta_weight']}[component]
                gradient_rows.append({'arm':name,'epoch':r['epoch']+1,'component':component,'mean_raw_gradient_norm':sum(values)/len(values),'weight':weight,'mean_weighted_gradient_norm':weight*sum(values)/len(values),'nonzero_kernel_probes':sum(v>0 for v in values)})
    with (out/'gradient_norms.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(gradient_rows[0]));writer.writeheader();writer.writerows(gradient_rows)
    fig,ax=plt.subplots(figsize=(9,5))
    for name in ('A','B'):
        for component in ('ranking','delta'):
            rr=[r for r in gradient_rows if r['arm']==name and r['component']==component]
            ax.plot([r['epoch'] for r in rr],[r['mean_weighted_gradient_norm'] for r in rr],label=name+' '+component)
    ax.set(xlabel='Epoch',ylabel='Mean weighted component gradient norm');ax.legend();fig.tight_layout();fig.savefig(out/'gradient_curves.png',dpi=160);fig.savefig(out/'gradient_curves.pdf');plt.close(fig)


def make_report(root,out,reports,table,decision,receipts):
    reconciliation=json.loads((root/'cohort/cohort_reconciliation.json').read_text());manifest=json.loads((root/'cohort/candidate_pool_manifest.json').read_text())
    lines=['# Final latency-GNN controlled study','',decision['status'],'',f"Selected: **{decision['selected']}**. Candidate checkpoint SHA256: `{decision['selected_checkpoint_sha256']}`. No model or production registration was promoted.",'','The two new arms use the same corrected data, fresh seed-123 initialization, architecture, optimizer and sampling schedule. B adds only delayed within-context delta supervision. Actual early-stopping totals are reported below. The fixed four-layer edge-aware TransformerConv, width 64, JK-max, one post-directive stage and 128-D [first-layer | JK] memory are preserved. No Stage-1/Stage-2 training or architecture search was performed.','','## Data reconciliation','','| Split | Raw rows | Successful measured under documented policy | Recovered >=100% raw rows | Legacy unique rows | Corrected unique rows | GNN eligible unique rows |','|---|---:|---:|---:|---:|---:|---:|']
    for tag in ('train','val'):
        rows=[r for r in reconciliation['kernels'] if r['split']==tag]
        sums=[sum(r[n] for r in rows) for n in ('raw_rows','successful_measured_rows','recovered_overcapacity_rows','legacy_preprocessed_rows','output_rows')]
        lines.append('| '+tag+' | '+' | '.join(str(n) for n in sums+[manifest['eligible_counts'][tag]])+' |')
    lines+=['','Successfully measured means finite positive latency within the established <=100000 ms range, nonnegative finite resource percentages and finite positive synthesis duration. Finite above-range values (including repeated 1,000,000 values) are ambiguous without outcome records; they are excluded rather than declared failures or successful latency labels. This conservative exclusion remains a data limitation. Original percentages, including zeros and overflow, are preserved in corrected tables and row-level raw provenance. Repeated physical designs are aggregated once using the historical median policy, with every raw measurement retained. Capacity-scaled percentages are fractional estimates, never exact resource counts.','','The full TCAD split has 130,310 training, 13,011 development and 6,705 locked-test configurations across 55 source-resolvable kernels. Those figures are not the eligible GNN cohort. The original frozen training-time development cohort excluded Sort-radix because L6 has no authentic APL column. The user explicitly requested final metrics on all four validation kernels, ignoring only L6 in Sort-radix; this was frozen before both final qualified arms. Final evaluation therefore covers four kernels in three independent families; the corrected training measurements remain unchanged and both final arms use the same four-kernel checkpoint-selection pools. L6 is unobserved and its candidate coordinates are masked, with no invented directive assignment. The original 9,070-candidate cohort and the amended broader evaluation are separately manifested. No locked-test labels were parsed. Exact exclusions, domains, coverage, hashes and per-context counts are in cohort_reconciliation.json and candidate_pool_manifest.json.','','## Matched evaluation','','ZCU104/10 ns is reported separately from all-measured-target contexts. The new models train on all measured contexts; evaluating their ZCU104/10 ns slice does not reproduce historical single-target training. All rows in each comparison table use the identical complete candidate pool, measured physical context, budgets and shared evaluation code. V1 is evaluated only on its authenticated single-target candidate intersection.','','| Protocol | Checkpoint | Unique physical candidates | Tau-b | Feasible Regret@1 (ratio) | Log2 latency MAE | Feasible selection |','|---|---|---:|---:|---:|---:|']
    def fmt(value):return 'undefined' if value is None else f'{value:.6f}'
    for r in table:lines.append('| '+r['protocol']+' | '+r['model']+' | '+str(r['physical_candidates'])+' | '+' | '.join(fmt(r[m]) for m in METRICS)+' |')
    lines+=['','Regret is selected measured latency / best measured-feasible latency - 1 within the measured pool; multiply the ratio by 100 for percent regret. It is conditional on successful measured-feasible selection, with coverage reported in each JSON. Selection filters by predicted resource requirements only; measured feasibility is checked afterward. Rounded percentage equality at a capacity limit is unknown. No measured-feasible-candidate pools and undefined tau are reported separately. Timing qualification is unknown. Nominal rounded utilization does not prove exact post-route resource or timing closure. Tau-b follows the inherited measured-feasible budget-pool convention; MAE uses all measured candidates. Budget views average within contexts, contexts within kernels, kernels within families, then independent families equally.','','### Family results on complete corrected pools','','| Checkpoint | Family | Tau-b | Regret ratio | Log2 MAE | Feasible selection |','|---|---|---:|---:|---:|---:|']
    for name,report in reports['corrected_all_contexts'].items():
        for f in report['families']:lines.append('| '+name+' | '+f['family']+' | '+' | '.join(fmt(f[m]) for m in METRICS)+' |')
    lines+=['','Family bootstrap intervals in the evaluation JSONs are exploratory; three independent eligible families cannot provide reliable narrow confidence intervals. Per-context and per-kernel results are retained, so a Pathfinder outlier cannot be hidden by the macro-average.','','### Published TCAD reference (not a newly replayed result)','','| Representation | Published latency tau-b | Published latency Regret@1 | Log2 MAE | Feasible selection |','|---|---:|---:|---|---|','| HARP representation | 0.201 | 0.404 | Not reported | Not reported |','| Structured MLIR | 0.200 | 0.355 | Not reported | Not reported |','','These publication values have their historical candidate protocol; they are not directly ranked against the new matched tables. The V1 replay authenticates the saved implementation/schema, checkpoint, cached directive vectors, scope injection and measured labels, and converts log2(ms+epsilon) predictions to log2(ms). It does not reconstruct missing Sort-radix directives.','','## Execution and scientific decision','','| Arm | Epochs run | Optimizer steps | Selected epoch (1-based) | Selected checkpoint SHA256 |','|---|---:|---:|---:|---|']
    for name,r in receipts.items():lines.append(f"| {name} | {r['total_epochs']} | {r['total_optimizer_steps']} | {r['selected_epoch_one_based']} | `{r['selected_checkpoint_sha256']}` |")
    lines+=['','Earlier Davinci attempts are preserved separately. Both final matched arms ran sequentially on the Liono V100S using the same prepared environment and corrected data. GPU receipts identify process ownership. All executed code and checkpoint states, optimizer/scheduler, sampler/Torch/CUDA RNG states, selected/last checkpoints, pair counts and component gradients are retained. Kernel exposure is equal at eight context updates per eligible kernel per epoch. Device/clock contexts rotate deterministically. Complete distinct candidates are sampled once per physical batch; regression is never replicated by budget. Ranking/delta near-ties within 5% are excluded. Nonzero ranking gradients are required, and B requires nonzero delta gradients after activation. B delta coefficients are 0,0,0,0.025,0.05 for epochs 1-5; the final coefficient remains 0.05. The loss uses unstandardized log2 latency differences.','','Decision checks:','', '```json',json.dumps({k:decision[k] for k in ('B_vs_A','candidate_vs_preliminary_all_contexts','candidate_vs_preliminary_zcu104_10ns','candidate_vs_V1_matched')},indent=2),'```','','## Structural representation and Stage-2 compatibility','','The RC3 adapter repairs wrong bound/use-count field names and transmits authenticated SSA/type/action/root/feature categories and static memory volume. Tensor schema v3 is versioned; both final arms receive the same correction, including isolated static nodes for registered source actions absent from the RC3 graph. These source-only nodes contain authenticated action metadata and have no invented compiler bindings or dependence edges. Their memory has identity visibility only. The earlier trained v2 A attempt is preserved and excluded from final checkpoint qualification because it failed full-source XAttn routing. The authenticated compiler graph topology, compiler relations, registered source actions and GNN architecture are unchanged; only the tensor adapter adds isolated source-only slots. Structural audit distinguishes archive availability from encoded features. Compiler dependence/evidence relation types enter edge-aware messages; detailed relation metadata also accompanies memory export. Static memory contains neither candidate values, device/clock context, budgets nor measured QoR. All source action identities, active/unbound masks, exact [conv1 | JK] layout, padding and training-only kernel-balanced RMS normalization are checked. The selected bank passes the maintained loader and real XAttn forward/backward with fresh fusion weights. Historical XAttn weights are not reused.','','QoR prediction improvement and feasible-ranking improvement are distinct in the four-metric tables. Valid memory export and XAttn compatibility establish representation integration only. **Downstream LLM improvement remains unverified until objective-matched Stage-2 training.** No complete Stage-2 run was launched.']
    (root/'SCIENTIFIC_DECISION.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
