"""Deterministic descriptive evaluation of completed constrained-decoder outputs.
No synthesis interpolation, locked-test labels or checkpoint selection occur here.
"""
import argparse,collections,csv,hashlib,json,statistics
from pathlib import Path

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scores(candidate):return {s['lhs']:{x['rhs']:x['score'] for x in s['top_candidates']} for s in candidate['site_debug']}
def differences(a,b):
 aa,bb=scores(a),scores(b);values=[abs(v-bb[k][rhs]) for k,opts in aa.items() if k in bb for rhs,v in opts.items() if rhs in bb[k]]
 return {'common_topk_scores':len(values),'mean_absolute_score_change':statistics.mean(values) if values else None,'max_absolute_score_change':max(values,default=None)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,required=True);p.add_argument('--evidence-controls',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 models={n:{r['context_id']:r for r in read(a.generated/(n+'.jsonl'))} for n in ('stage1','stage2','sft','stage3')};assert all(set(rows)==set(models['stage1']) for rows in models.values());records=[];macro={}
 for name,rows in models.items():
  families=collections.defaultdict(list)
  for key,r in rows.items():
   c=r['candidates'][0];base=models['stage1'][key]['candidates'][0];family='viterbi' if 'viterbi' in r['kernel_name'] else 'pathfinder';families[family].append(c['value_accuracy_over_expected'])
   records.append({'model':name,'context_id':key,'kernel':r['kernel_name'],'family':family,'part':r['device'],'clock_ns':r['clock_period_ns'],'budget':r['resource_budget'],'schema_compliant':c['schema_compliant'],'expected_key_match':c['expected_key_match'],'exact_reference_match':c['exact_design_match'],'reference_field_accuracy':c['value_accuracy_over_expected'],'changed_assignment_vs_stage1':c['prediction']!=base['prediction'],'assignment':c['prediction'],'stage1_candidate_score_changes':differences(c,base),'physical_feasibility':None,'physical_latency':None,'physical_scope':'No exact physical receipt joined; identical assignment alone is not a new measured hardware outcome.'})
  macro[name]={'family_mean_reference_field_accuracy':{f:statistics.mean(v) for f,v in families.items()},'family_macro_reference_field_accuracy':statistics.mean(statistics.mean(v) for v in families.values()),'changed_assignments_vs_stage1':sum(r['changed_assignment_vs_stage1'] for r in records if r['model']==name),'requests':len(rows),'families':len(families)}
 controls=[]
 for r in read(a.evidence_controls):
  kernel=r['variants']['authentic']['kernel_name'];variants=r['variants'];baseline=variants['authentic']['candidates'][0]
  ledger=r['control_ledgers']['cross_route_evidence'][kernel];controls.append({'context_id':r['context_id'],'kernel':kernel,'evidence_records_moved':len(ledger),'moved_between_routes':sum(x['route_changed'] for x in ledger),'mean_relative_vector_change':statistics.mean(x['relative_vector_change'] for x in ledger) if ledger else 0,'variants':{name:{'changed_assignment':v['candidates'][0]['prediction']!=baseline['prediction'],**differences(v['candidates'][0],baseline)} for name,v in variants.items()},'scope':'A2 NF4 FP16 supplementary selected-Step50 control; natural prefixes identical when all chosen values match; eligibility/types/status preserved but cross-route content moved.'})
 result={'schema':'v3-completed-decoder-descriptive-v1','inputs':{str(a.generated/(n+'.jsonl')):sha(a.generated/(n+'.jsonl')) for n in models}|{str(a.evidence_controls):sha(a.evidence_controls)},'summary':macro,'per_request':records,'evidence_controls':controls,'scope':'Three validation kernels, two families, nine correlated requests; no locked test. Teacher-reference field agreement is not QoR. Truncated top-k score comparisons cover only shared returned candidates; full score parity was checked separately. No architecture promoted.'}
 (a.out/'decoder_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 with (a.out/'decoder_comparison.csv').open('w') as f:
  fields=['model','context_id','family','schema_compliant','expected_key_match','reference_field_accuracy','changed_assignment_vs_stage1','physical_feasibility','physical_latency'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in records)
 print(json.dumps({'summary':macro,'controls':controls},indent=2))
if __name__=='__main__':main()
