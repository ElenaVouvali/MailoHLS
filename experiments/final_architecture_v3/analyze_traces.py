"""Reconcile immutable controlled-prefix captures; no model loading or tuning."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(trace):
 rows=[json.loads(s) for s in trace.read_text().splitlines() if s];effects=[];targets=[];states=[];by={r['context_id']:r for r in rows}
 for row in rows:
  records={(r['field'],r['mode']):r for r in row['records']}
  for (field,mode),a in records.items():
   if mode!='authentic':continue
   if a['semantically_forced']:continue
   for other in ('zero_evidence','permuted_evidence','gate_off'):
    b=records[field,other];assert a['prior_output_sha256']==b['prior_output_sha256'];assert a['candidate_domain_sha256']==b['candidate_domain_sha256']
    sa={s['rhs']:s['score'] for s in a['scores']};sb={s['rhs']:s['score'] for s in b['scores']};assert sa.keys()==sb.keys()
    effects.append({'context':row['context_id'],'kernel':row['kernel'],'field':field,'control':other,'winner_changed':a['scores'][0]['rhs']!=b['scores'][0]['rhs'],'max_abs_candidate_score_change':max(abs(sa[k]-sb[k]) for k in sa),'margin_change':a['margin']-b['margin']})
   for layer in a['layers']:states.append({'context':row['context_id'],'field':field,**{k:layer[k] for k in ('layer','relative_update','effective_gate','eligible_keys','attention_entropy','category_mass')}})
 for kernel in sorted({r['kernel'] for r in rows}):
  rr=[r for r in rows if r['kernel']==kernel];anchor=next(r for r in rr if ':xczu7ev-ffvc1156-2-e:10.0' in r['context_id'])
  def load(row):
   p=Path(row['tensors']);p=p if p.is_absolute() else trace.parent.parent.parent/p
   if not p.is_file():p=trace.parent/Path(row['tensors']).name
   assert sha(p)==row['tensor_sha256'];return torch.load(p,map_location='cpu',weights_only=True)
  ta=load(anchor);ar={r['field']:r for r in anchor['records'] if r['mode']=='authentic' and not r['semantically_forced']}
  for dest in rr:
   if dest is anchor:continue
   tb=load(dest)
   for b in dest['records']:
    if b['mode']!='authentic' or b['semantically_forced']:continue
    a=ar[b['field']];assert a['prior_output_sha256']==b['prior_output_sha256'] and a['candidate_domain_sha256']==b['candidate_domain_sha256']
    sa={s['rhs']:s['score'] for s in a['scores']};sb={s['rhs']:s['score'] for s in b['scores']}
    for la,lb in zip(a['layers'],b['layers']):
     va,vb=ta[la['tensor_index']],tb[lb['tensor_index']];ha,hb=va['before'],vb['before'];pa,pb=va['attention'],vb['attention'];assert torch.equal(va['allowed'],vb['allowed'])
     targets.append({'kernel':kernel,'context_a':anchor['context_id'],'context_b':dest['context_id'],'field':b['field'],'layer':la['layer'],'winner_changed':a['scores'][0]['rhs']!=b['scores'][0]['rhs'],'max_abs_score_change':max(abs(sa[k]-sb[k]) for k in sa),'margin_change':b['margin']-a['margin'],'before_relative_l2':float((hb-ha).norm()/ha.norm()),'before_cosine':float(torch.nn.functional.cosine_similarity(ha,hb,dim=0)),'attention_mean_total_variation':float((pa-pb).abs().sum(-1).mean()/2)})
 aggregate={}
 for c in ('zero_evidence','permuted_evidence','gate_off'):
  e=[r for r in effects if r['control']==c];aggregate[c]={'unforced_field_comparisons':len(e),'winner_changes':sum(r['winner_changed'] for r in e),'mean_max_abs_candidate_score_change':statistics.mean(r['max_abs_candidate_score_change'] for r in e),'max_abs_candidate_score_change':max(r['max_abs_candidate_score_change'] for r in e)}
 return {'schema':'v3-fixed-prefix-analysis-v1','trace_sha256':sha(trace),'contexts':len(rows),'independent_kernels':len({r['kernel'] for r in rows}),'controls':aggregate,'effects':effects,'target_contrasts':targets,'states':states,'limitations':['historical equal-field/top1 checkpoint150; not corrected-supervision result','two validation families, no generalization probe','score changes do not establish correct semantic use or physical quality','no budget contrast in these six contexts','post-attention boundary captures are not complete multi-token candidate likelihoods']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();report=summarize(a.trace);a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({k:v for k,v in report.items() if k not in ('effects','target_contrasts','states')},indent=2))
if __name__=='__main__':main()
