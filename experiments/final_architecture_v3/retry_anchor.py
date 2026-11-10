"""One prospectively capped exact-baseline retry after censored refinement.
No policy proposal or architecture selection; preserve every failed attempt.
"""
import argparse,json,subprocess,time
from pathlib import Path
from experiments.refinement import configurations as c
from experiments.refinement_search.coroni import CoroniOracle

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--active-ssh-pid',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 request=json.loads(a.request.read_text());baseline=c.read_jsonl(a.run/'coordinate/measurements.jsonl')[0]['result']['physical_identity'];ledger=c.read_jsonl(a.run/'physical/physical_ledger.jsonl');prior=23+len(ledger)
 protocol={'schema':'v3-exact-stage3-anchor-timeout-retry-v1','question':'Can the exact generated baseline yield finite HLS latency/resources/timing under one1200s confirmation?','reason':'Repeated300s timeouts made the policy comparison uninformative. Cancel own policy queue; preserve active job. One longer same-physical-design retry replaces remaining neighbours.','physical_identity':baseline,'prior_calls':prior,'maximum_new_calls':1,'timeout_seconds':1200,'selected_from':'Actual Stage3 model output fixed before synthesis; no quality-selected substitute','limits':'A successful HLS result is not C/RTL correctness or post-route timing. A timeout is unknown, not infeasible.'}
 print(json.dumps(protocol,indent=2),flush=True)
 if not a.execute:return
 a.out.mkdir(parents=True,exist_ok=True);file=a.out/'protocol.json'
 if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('Retry protocol changed')
 c.write_json(file,protocol)
 # Leave the former queue's current SSH child and bounded HLS process untouched.
 started=time.monotonic()
 while True:
  proc=Path('/proc')/str(a.active_ssh_pid)
  if not proc.exists():break
  state=(proc/'stat').read_text().split(') ')[1].split()[0]
  if state=='Z':break
  if time.monotonic()-started>330:raise TimeoutError('Previous bounded Coroni child still active; no concurrent retry launched')
  time.sleep(2)
 pending=ledger[-1]['key'];dest=a.run/'physical'/pending/'results';dest.mkdir(exist_ok=True)
 subprocess.run(['rsync','-a','bonus24@coroni.microlab.ntua.gr:architecture_v3_stage3_refinement_20261110/'+pending+'/results/',str(dest)+'/'],check=True)
 result=dest/'query/result.json';c.write_json(a.run/'active_query_after_cancel.json',{'charged_query_key':pending,'result':json.loads(result.read_text()),'result_sha256':c.sha256(result),'scope':'Active job completed after local orchestration cancellation; original controller pending journal remains untouched for exact resume.'})
 template=Path(request['source_template']);oracle=CoroniOracle(request,template,[template.parent/h for h in request['headers_sha256']],a.runner,a.out/'physical','architecture_v3_stage3_anchor_confirmation_20261110',max_new_calls=1,prior_calls=prior,timeout=1200)
 entry=oracle(baseline);c.write_json(a.out/'result.json',entry);print('[ANCHOR]',entry['measurement'],flush=True)
if __name__=='__main__':main()
