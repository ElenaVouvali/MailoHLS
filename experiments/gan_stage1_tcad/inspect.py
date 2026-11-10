"""Inspectable raw GAN assignments, request differences and offline tool checks.
No inference edits, synthesis, QoR estimation or checkpoint selection.
"""
import argparse,csv,hashlib,html,importlib,json,re,sys
from pathlib import Path
from LLM_branch.common import directive_domains as domains,mailohls_contract as mc
from experiments.refinement import configurations as c,synthesis as hls

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--label',default='Stage1');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--inference',type=Path,required=True);p.add_argument('--policy-package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Inspection output already exists')
 execution=json.loads((a.inference/'execution.json').read_text());predictions=[json.loads(s) for s in (a.inference/'predictions.jsonl').read_text().splitlines()];requests=[json.loads(s) for s in (a.prepared/'requests.jsonl').read_text().splitlines()];assert len(predictions)==len(requests)==18 and execution['completed_cases']==18;assert sha(a.inference/'predictions.jsonl')==execution['predictions_sha256']
 actions=domains.load_source_actions('GAN',a.prepared/'sources');fields=list(domains.legacy_site_domains('GAN',requests[0]['input'],a.prepared/'sources'));assert len(fields)==25
 sys.path.insert(0,str(a.policy_package.parent));tool=importlib.import_module(a.policy_package.name+'.source_tool_policy');policy=tool.load_policy(a.policy_package/'policy.json');records=[];configs={}
 for request,prediction in zip(requests,predictions):
  assert request['context_id']==prediction['context_id'];candidate=prediction['candidates'][0];raw=candidate['prediction'];assignment=c.parse_assignment('\n'.join(line for line in raw.splitlines() if not re.fullmatch(r'<L[1-9][0-9]*>',line.strip())));assert set(assignment)==set(fields)
  assert all(assignment[f] in domains.legacy_site_domains('GAN',request['input'],a.prepared/'sources')[f] for f in fields),'Value outside native source proposal domain';mc.validate_directive_assignments(assignment);key=json.dumps(assignment,sort_keys=True);config=configs.setdefault(key,'C'+str(len(configs)+1));issues=[]
  for label,action in actions.items():
   if action.kind=='array':
    triple=tuple(assignment[f'AUTO{{_{kind}_{label}}}'] for kind in ('ARRAY_T','ARRAY_F','ARRAY_D'));reason=tool.tuple_rejection(triple,action,policy)
    if reason:issues.append({'action':label,'tuple':triple,'reason':reason})
   elif int(assignment[f'AUTO{{_UNROLL_{label}}}'])>action.trip_count:issues.append({'action':label,'reason':'unroll_exceeds_trip_count'})
  records.append({'context_id':request['context_id'],'device':'ZCU104' if request['device'].startswith('xczu') else 'U200','part':request['device'],'clock_ns':request['selected_clock_period'],'budget_percent':round(100*request['resource_budget_frac']),'absolute_budgets':prediction['resource_budget'],'configuration':config,'assignment':assignment,'raw_prediction':raw,'source_tool_v2_pass':not issues,'source_tool_v2_issues':issues,'prompt_token_count':prediction['prompt_token_count'],'site_margins':[{k:site.get(k) for k in ('lhs','chosen_rhs','chosen_score','forced_by_semantics','static_candidate_count')}|{'margin':site['top_candidates'][0]['score']-site['top_candidates'][1]['score'] if len(site['top_candidates'])>=2 else None} for site in candidate['site_debug']],'HLS_QoR':None})
 lookup={(r['device'],r['clock_ns'],r['budget_percent']):r for r in records};reference=lookup[('ZCU104',10.,100)]
 def distance(left,right):return sum(left['assignment'][f]!=right['assignment'][f] for f in fields)
 for r in records:
  r.update(changed_fields_vs_reference=[f for f in fields if r['assignment'][f]!=reference['assignment'][f]],distance_reference=distance(r,reference),distance_same_target_full_budget=distance(r,lookup[(r['device'],r['clock_ns'],100)]),distance_same_device_budget_10ns=distance(r,lookup[(r['device'],10.,r['budget_percent'])]),distance_matched_ZCU104=distance(r,lookup[('ZCU104',r['clock_ns'],r['budget_percent'])]))
 a.out.mkdir(parents=True);(a.out/'requests').mkdir();(a.out/'materialized_sources').mkdir();md=['# GAN: '+a.label+' directives for the TCAD requests','', 'PARETO_LATENCY; frozen V4 parent, '+a.label+'. Raw native-checkpoint decoder outputs are preserved.','', '**No synthesis has been launched.** Offline source/tool checks are separate from raw predictions and do not establish resource feasibility or timing.','', 'Array cells: `type/factor/dimension`. Loop cells: `pipeline/unroll`. Factor0 denotes the checkpoint encoding for disabled unroll/partial partition or the complete-partition sentinel, as applicable.','', '|Device|Clock(ns)|Budget|Config|L1 xbuf|L2 layer_1_out|L3 layer_2_out|L4 P/U|L5 P/U|L6 P/U|L7 P/U|L8 P/U|L9 P/U|L10 P/U|L11 P/U|Tool check|','|'+'---|'*16]
 def cell(r,label):
  assignment=r['assignment'];kinds=('ARRAY_T','ARRAY_F','ARRAY_D') if actions[label].kind=='array' else ('PIPE','UNROLL');return '/'.join(assignment[f'AUTO{{_{kind}_{label}}}'] for kind in kinds)
 for idx,r in enumerate(records,1):
  columns=[r['device'],str(r['clock_ns']),str(r['budget_percent'])+'%',r['configuration']]+[cell(r,'L'+str(i)) for i in range(1,12)]+['PASS' if r['source_tool_v2_pass'] else 'FAIL'];md.append('|'+'|'.join(columns)+'|');(a.out/'requests'/f'{idx:02d}_{r["device"]}_{r["clock_ns"]}ns_{r["budget_percent"]}pct.txt').write_text(r['raw_prediction']+'\n');(a.out/'materialized_sources'/f'{idx:02d}_{r["configuration"]}.cpp').write_text(hls.materialize(requests[idx-1]['input'],'\n'.join(f'{k} = {v}' for k,v in r['assignment'].items())))
 md+=['','## Full directives for every request','']
 for r in records:
  md+=['### '+r['context_id']+' — '+r['configuration'],'','Absolute BRAM18K/DSP/FF/LUT limits: `'+json.dumps(r['absolute_budgets'],sort_keys=True)+'`.','', '```text',r['raw_prediction'],'```','']
  if r['source_tool_v2_issues']:md+=['Offline source/tool issues: `'+json.dumps(r['source_tool_v2_issues'])+'`. Raw values above have not been repaired.','']
 md+=['## Interpretation','',f'{len(configs)} distinct complete assignments across18 requests; {sum(r["source_tool_v2_pass"] for r in records)}/18 pass the separate pinned source/tool check.','', 'Directive differences are descriptive, not measured hardware gains. The historical paper heatmap latency/utilization values belong to its own checkpoints/receipts and are not assigned to this run.','']
 (a.out/'DIRECTIVES.md').write_text('\n'.join(md))
 with (a.out/'directives.csv').open('w') as f:
  keys=['context_id','configuration','source_tool_v2_pass',*fields];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({'context_id':r['context_id'],'configuration':r['configuration'],'source_tool_v2_pass':r['source_tool_v2_pass'],**r['assignment']} for r in records)
 payload={'model_label':a.label,'schema':'gan-checkpoint-tcad-inspection-v2','status':'INFERENCE_COMPLETED_SYNTHESIS_NOT_STARTED','objective':'PARETO_LATENCY','requests':18,'unique_configurations':len(configs),'source_tool_v2_pass_count':sum(r['source_tool_v2_pass'] for r in records),'offline_policy_sha256':sha(a.policy_package/'policy.json'),'input_manifest_sha256':sha(a.prepared/'manifest.json'),'execution':execution,'records':records,'scope':'GAN external descriptive inference, no model selection, optimizer update or synthesis. Raw legacy-domain predictions are unchanged; source/tool checks are post-hoc.'};(a.out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 body=['<!doctype html><meta charset="utf-8"><title>GAN Stage1 directive inspection</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}pre{background:#f3f4f6;padding:16px;overflow:auto}summary{cursor:pointer;padding:10px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style><h1>GAN Stage1: eighteen TCAD requests</h1><p>PARETO_LATENCY · selected V4 · no synthesis run</p><p>Raw predictions are preserved; source/tool checks are offline.</p>']
 for r in records:body.append('<details><summary>'+html.escape(r['context_id']+' | '+r['configuration']+' | source/tool '+('PASS' if r['source_tool_v2_pass'] else 'FAIL'))+'</summary><p>Limits: '+html.escape(str(r['absolute_budgets']))+'</p><pre>'+html.escape(r['raw_prediction'])+'</pre><p>'+html.escape(str(r['source_tool_v2_issues']))+'</p></details>')
 (a.out/'inspect.html').write_text('\n'.join(body).replace('GAN Stage1','GAN '+html.escape(a.label)).replace('GAN Stage1:','GAN '+html.escape(a.label)+':'));print(json.dumps({'requests':18,'unique_configurations':len(configs),'source_tool_pass':payload['source_tool_v2_pass_count']},indent=2))
if __name__=='__main__':main()
