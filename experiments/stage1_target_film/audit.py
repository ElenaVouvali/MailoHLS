"""Audit selected pair directions; never equate generic pair counts with switches."""
import argparse,collections,hashlib,itertools,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();report={}
 for fold in ('train','val'):
  path=a.pairs/(fold+'_pairs.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines()];counts=collections.Counter();physical=set();families=set()
  for r in rows:
   families.add(r['request']['family'])
   for side in ('preferred','rejected'):
    m=r[side]['measurement'];physical.add((m['kernel'],m['device'],m['clock'],m['config_hash']))
  for x,y in itertools.combinations(rows,2):
   cx,cy=x['request']['physical_context'],y['request']['physical_context']
   if cx[:3]!=cy[:3]:continue
   bx,by=x['request']['budget'],y['request']['budget']
   changes=[axis for axis,changed in [('budget',bx!=by),('device',cx[3]!=cy[3]),('clock',cx[4]!=cy[4])] if changed]
   if len(changes)!=1:continue
   axis=changes[0];px=x['preferred']['candidate_id'];py=y['preferred']['candidate_id'];sx={px,x['rejected']['candidate_id']};sy={py,y['rejected']['candidate_id']}
   if sx!=sy:counts[axis+'_unmatched_candidate_pair']+=1
   elif px==py:counts[axis+'_same_pair_stay']+=1
   else:counts[axis+'_same_pair_switch']+=1
  report[fold]={'generic_same_context_pairs':len(rows),'independent_families':len(families),'unique_physical_designs_in_pairs':len(physical),'contrasts':dict(counts),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 report.update(exact_resource_counts=False,timing_qualified=False,scope='Selected 128/54 pair bank. Matching requires identical complete candidate hashes; unmatched pairs are not switches. Budget evidence is proxy-only. No locked test or GAN loaded.')
 with a.out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
