"""Descriptive GAN field comparison; never assigns QoR to model outputs."""
import argparse,collections,csv,hashlib,itertools,json,re,statistics,sys,importlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--v4',type=Path,required=True);p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 if args.out.exists():raise FileExistsError(args.out)
 models={};configs={};fields=None
 from LLM_branch.common import directive_domains as domains,mailohls_contract as contract
 policy_package=Path(__file__).resolve().parents[2]/'artifacts/stage1_target_film/policy/salt_qualification_contract'
 sys.path.insert(0,str(policy_package.parent));tool=importlib.import_module(policy_package.name+'.source_tool_policy');policy=tool.load_policy(policy_package/'policy.json');actions=domains.load_source_actions('GAN',args.v4.parent.parent/'prepared/sources')
 for label,path in [('V4',args.v4),('A',args.a),('B',args.b)]:
  rows=[json.loads(s) for s in path.read_text().splitlines()];assert len(rows)==18;model={}
  for row in rows:
   text=row['candidates'][0]['prediction'];assignment={k.upper():v for k,v in re.findall(r'(?im)^(auto\{[^}]+\})\s*=\s*(\S+)\s*$',text)};assert len(assignment)==25
   contract.validate_directive_assignments(assignment);tool.validate_source_assignment(assignment,actions);tool.validate_complete_assignment(assignment,actions,policy)
   canonical=json.dumps(assignment,sort_keys=True,separators=(',',':'));identity=hashlib.sha256(canonical.encode()).hexdigest();configs[identity]=assignment;model[row['context_id']]={'hash':identity,'assignment':assignment};fields=sorted(assignment)
  models[label]=model
 order=list(models['V4']);assert all(set(rows)==set(order) for rows in models.values())
 def distance(x,y):return [f for f in fields if x['assignment'][f]!=y['assignment'][f]]
 summary={};records=[]
 for label,rows in models.items():
  ds=[len(distance(rows[x],rows[y])) for x,y in itertools.combinations(order,2)];axes={}
  for axis,index in [('device',1),('clock',2),('budget',3)]:
   pairs=[(x,y) for x,y in itertools.combinations(order,2) if all(x.split(':')[j]==y.split(':')[j] for j in (1,2,3) if j!=index)];axes[axis]={'pairs':len(pairs),'different':sum(rows[x]['hash']!=rows[y]['hash'] for x,y in pairs),'scope':'matched full-device fractions; capacities differ' if axis=='device' else 'one request axis changed'}
  summary[label]={'unique':len({r['hash'] for r in rows.values()}),'mean_pairwise_fraction':statistics.mean(ds)/25,'max_pairwise_fields':max(ds),'axes':axes}
  for context in order:records.append({'model':label,'request':context,'hash':rows[context]['hash'],'changed_fields_vs_V4':distance(rows[context],models['V4'][context]),'A_B_changed_fields':distance(models['A'][context],models['B'][context])})
 ids={h:'C'+str(i+1) for i,h in enumerate(configs)};md=['# GAN: frozen V4 versus residual LoRA and LoRA + FiLM','','Inference only. GAN was excluded from training and checkpoint selection. No HLS synthesis or physical QoR claim.','','|FPGA|Clock|Budget|V4|Residual LoRA A|LoRA + FiLM B|','|---|---|---|---|---|---|']
 for context in order:
  _,device,clock,budget=context.split(':');md.append('|'+ '|'.join([device,clock,budget,*[ids[models[l][context]['hash']] for l in models]])+'|')
 md+=['','|Model|Distinct|Mean pairwise changed fields|Max distance|','|---|---:|---:|---:|']
 for l,s in summary.items():md.append(f'|{l}|{s["unique"]}|{s["mean_pairwise_fraction"]:.2%}|{s["max_pairwise_fields"]}/25|')
 for h,assignment in configs.items():md+=['',f'## {ids[h]}','',f'SHA-256: `{h}`','','```text',*['%s = %s'%x for x in sorted(assignment.items())],'```']
 md+=['','Every request references a complete 25-field configuration above. Full per-request assignments, changed-field identities and axis counts are in JSON/CSV. Conditional fields count in Hamming distances; they are not independent choices.']
 args.out.mkdir(parents=True);(args.out/'GAN_COMPARISON.md').write_text('\n'.join(md)+'\n');(args.out/'gan_comparison.json').write_text(json.dumps({'summary':summary,'records':records,'models':models,'configurations':configs,'input_sha256':{l:hashlib.sha256(p.read_bytes()).hexdigest() for l,p in [('V4',args.v4),('A',args.a),('B',args.b)]},'HLS_calls':0,'offline_source_tool_policy_sha256':hashlib.sha256((policy_package/'policy.json').read_bytes()).hexdigest(),'source_tool_domain_validity':'all54 proposed assignments passed; no physical feasibility inferred'},indent=2)+'\n')
 with (args.out/'gan_directives.csv').open('w') as f:
  w=csv.DictWriter(f,['model','request','configuration_sha256',*fields]);w.writeheader();w.writerows({'model':l,'request':c,'configuration_sha256':r['hash'],**r['assignment']} for l,rows in models.items() for c,r in rows.items())
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
