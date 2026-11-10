"""One log-supported existing coordinate proposal, not a new search engine.
Prospective after observing parent-pipeline-driven512-way unroll and timeouts.
"""
import argparse,json,time,re
from pathlib import Path
from LLM_branch.common import directive_domains as domains
from experiments.refinement import configurations as c
from experiments.refinement_search import policies
from experiments.refinement_search.coroni import CoroniOracle
from experiments.refinement_search.deployment import physical_identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--confirmation',type=Path,required=True);p.add_argument('--original-run',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();request=json.loads(a.request.read_text());initial=c.read_jsonl(a.original_run/'coordinate/measurements.jsonl')[0];base=initial['result']['physical_identity']['assignment'];directory=Path(request['source_directory']);actions=domains.load_source_actions(request['kernel'],directory);problem={'baseline':base,'actions':actions,'native_domains':c.legacy_domains(actions,directory),'relations':[],'inactive_actions':[]}
 proposals=policies.single_proposals(problem,base);matches=[r for r in proposals if r['reference_changed_fields']==['AUTO{_PIPE_L1}'] and r['assignment']['AUTO{_PIPE_L1}']=='0']
 if len(matches)!=1:raise ValueError('No unique existing legal coordinate proposal for parent pipeline off')
 proposal=matches[0];baseline_entry=next((a.original_run/'physical').glob('*/oracle.json'));source_entry=json.loads(baseline_entry.read_text());log=next(Path(x) for x in source_entry['report_sha256'] if x.endswith('synthesis.log'));lines=[line for line in log.read_text().splitlines() if ('Unrolling all sub-loops' in line or 'completely with a factor of 512' in line or 'Ignored partial unroll' in line)];assert len(lines)>=3
 protocol={'schema':'v3-log-supported-pipeline-coordinate-diagnostic-v1','baseline_assignment':base,'proposal':proposal,'evidence_log_sha256':c.sha256(log),'compiler_messages':lines,'max_new_calls':1,'prior_calls':30,'timeout_seconds':300,'qualification':'Exploratory single development case after timeout diagnosis; not a matched policy comparison or proof of structural-fusion benefit. No model update or test-label access.'};print(json.dumps(protocol,indent=2),flush=True)
 if not a.execute:return
 if a.out.exists():raise FileExistsError('Preserve previous diagnostic')
 a.out.mkdir(parents=True);c.write_json(a.out/'protocol.json',protocol);start=time.monotonic()
 while not (a.confirmation/'result.json').exists():
  if time.monotonic()-start>1250:raise TimeoutError('Baseline confirmation unavailable; no concurrent synthesis launched')
  time.sleep(2)
 template=Path(request['source_template']);oracle=CoroniOracle(request,template,[directory/h for h in request['headers_sha256']],a.runner,a.out/'physical','architecture_v3_pipeline_diagnostic_20261110',max_new_calls=1,prior_calls=30,timeout=300)
 entry=oracle(physical_identity(request,proposal['assignment']));c.write_json(a.out/'result.json',entry);print('[PIPELINE_DIAGNOSTIC]',entry['measurement'],flush=True)
if __name__=='__main__':main()
