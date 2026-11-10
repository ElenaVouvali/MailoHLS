"""Exact directive-field separation across the18 requests and four frozen policies.
All distances describe requested directives, not physical QoR or independent
semantic decisions. Only identical preceding values permit local score contrasts.
"""
import argparse,collections,csv,hashlib,json,statistics
from itertools import combinations
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True,help='Ordered label -> completed inspection results.json');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 files=json.loads(a.reports.read_text());reports={label:json.loads(Path(path).read_text()) for label,path in files.items()};first=next(iter(reports.values()));order=[r['context_id'] for r in first['records']];fields=list(first['records'][0]['assignment']);assert len(order)==18 and len(fields)==25;maps={label:{r['context_id']:r for r in report['records']} for label,report in reports.items()};assert all(set(rows)==set(order) for rows in maps.values());globalids={};globalconfigs={}
 def signature(record):return json.dumps(record['assignment'],sort_keys=True)
 def distance(a,b):return sum(a['assignment'][k]!=b['assignment'][k] for k in fields)
 def axis_pairs(rows,axis):
  return [(x,y) for x,y in combinations(rows,2) if ((x['device']==y['device'] and x['clock_ns']==y['clock_ns']) if axis=='budget' else (x['device']==y['device'] and x['budget_percent']==y['budget_percent']) if axis=='clock' else (x['clock_ns']==y['clock_ns'] and x['budget_percent']==y['budget_percent']))]
 summaries={};cases=[]
 for label,rows in maps.items():
  ordered=[rows[k] for k in order];pairs=list(combinations(ordered,2));distances=[distance(x,y) for x,y in pairs];axes={}
  for axis in ('budget','clock','device'):
   ps=axis_pairs(ordered,axis);ds=[distance(x,y) for x,y in ps];axes[axis]={'matched_pairs':len(ps),'different_assignment_pairs':sum(x>0 for x in ds),'mean_field_distance':statistics.mean(ds),'max_field_distance':max(ds),'device_contrast_scope':'full-device fractions, not common absolute resource limits' if axis=='device' else None}
  summaries[label]={'requests':18,'distinct_assignments':len({signature(r) for r in ordered}),'all_pairs':len(pairs),'identical_pairs':sum(d==0 for d in distances),'different_pairs':sum(d>0 for d in distances),'mean_field_distance':statistics.mean(distances),'mean_fraction_different':statistics.mean(d/25 for d in distances),'max_field_distance':max(distances),'changed_requests_vs_stage1':sum(distance(rows[k],next(iter(maps.values()))[k])>0 for k in order),'source_tool_pass':sum(r['source_tool_v2_pass'] for r in ordered),'single_axis_comparisons':axes,'distance_matrix_fields':[[distance(x,y) for y in ordered] for x in ordered]}
  for r in ordered:
   key=signature(r)
   if key not in globalids:globalids[key]='D'+str(len(globalids)+1);globalconfigs[globalids[key]]=r['raw_prediction']
   cases.append({'model':label,'context_id':r['context_id'],'configuration':globalids[key],'changed_fields_vs_stage1':[k for k in fields if r['assignment'][k]!=next(iter(maps.values()))[r['context_id']]['assignment'][k]],'source_tool_v2_pass':r['source_tool_v2_pass']})
 a.out.mkdir(parents=True);md=['# GAN: Stage1 / Stage2 / Stage3 request separation','', '**Inference only; synthesis is stopped pending user inspection.**','', 'Stage1 selected = final315. Stage2-50 and Stage2-150 are corrected-supervision checkpoints. Stage3 is DPO34 trained from Stage2-50, not Stage2-150.','', 'A distance of n/25 counts changed raw directive fields. Conditional/forced fields are not independent decisions. Same D-ID means the complete25-field assignment is identical, across models as well as requests.','', '|Request|'+'|'.join(maps)+'|','|---|'+'---|'*len(maps)]
 for context in order:md.append('|'+context+'|'+'|'.join(globalids[signature(maps[label][context])] for label in maps)+'|')
 md+=['','## Separation within each checkpoint','', '|Model|Unique assignments|Identical pairs /153|Different pairs /153|Mean field distance|Max changed fields|Requests changed vsStage1|','|---|---:|---:|---:|---:|---:|---:|']
 for label,s in summaries.items():md.append(f'|{label}|{s["distinct_assignments"]}|{s["identical_pairs"]}|{s["different_pairs"]}|{100*s["mean_fraction_different"]:.2f}%|{s["max_field_distance"]}/25|{s["changed_requests_vs_stage1"]}/18|')
 md+=['','## Every distinct full assignment','']
 for name,text in globalconfigs.items():md+=['### '+name,'','```text',text,'```','']
 md+=['## Limits','', 'These are one-program descriptive results, not independent generalization samples. Target-dependent directives or score changes do not establish improved measured latency, resource feasibility or timing. All models share the same source, requests, candidate policy and frozen Stage1 lineage. Memory C is target-independent; AP arithmetic in its compiler graph is analysis-only, not bit-true hardware evidence.','']
 (a.out/'COMPARISON.md').write_text('\n'.join(md));result={'schema':'gan-frozen-checkpoint-request-separation-v1','fields':25,'request_order':order,'summary':summaries,'cases':cases,'configurations':globalconfigs,'input_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files.values()},'synthesis_started':False,'hardware_QoR':None};(a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 with (a.out/'request_configurations.csv').open('w') as f:
  keys=['context_id',*maps];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({'context_id':key,**{label:globalids[signature(maps[label][key])] for label in maps}} for key in order)
 print(json.dumps({k:{f:v for f,v in r.items() if f!='distance_matrix_fields'} for k,r in summaries.items()},indent=2))
if __name__=='__main__':main()
