"""Authenticate probe receipts; separate completed synthesis from rankable QoR."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/architecture_finalization'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze():
 cases=[]
 for name,bundle in [('probe_results','reference_probe'),('probe2_results','reference_probe2')]:
  root=BASE/name;manifest=BASE/bundle/'manifest.json';m=json.loads(manifest.read_text());prov=json.loads((root/'batch_provenance.json').read_text())
  assert prov['manifest_sha256']==sha(manifest);assert prov['runner_sha256']==sha(BASE/bundle/'run_batch.py')
  for j in m['jobs']:
   p=root/j['job_id'];d=json.loads((p/'result.json').read_text());receipt=json.loads((p/'receipt.json').read_text())
   assert sha(p/'receipt.json')==d['receipt_sha256'];assert sha(p/'synthesis.log')==d['log_sha256'];assert sha(p/'run_hls.tcl')==receipt['tcl_sha256'];assert receipt['source_sha256']==j['materialized_source_sha256']
   r={'job_id':j['job_id'],'kernel':j['kernel'],'part':j['part'],'clock_ns':j['clock_ns'],'runner_status':d['status'],'runtime_seconds':d['runtime_seconds'],'error':d['error'],'assignment':j['assignment'],'source_sha256':j['original_source_sha256'],'headers':j['headers'],'materialized_source_sha256':j['materialized_source_sha256'],'receipt_sha256':sha(p/'receipt.json'),'synthesis_completed':False,'rankable_latency':False,'latency_cycles':None,'resources':None,'resource_feasible':None,'timing_qualified':None}
   report=p/'hls_project/solution1/syn/report'/(j['top']+'_csynth.xml')
   if d['returncode']==0 and report.is_file():
    assert sha(report)==d['report_sha256'];t=ET.parse(report)
    for key,value in [('Part',j['part']),('TopModelName',j['top'])]:assert t.findtext('.//UserAssignments/'+key)==value
    assert t.findtext('.//UserAssignments/FlowTarget')=='vivado';assert t.findtext('.//UserAssignments/unit')=='ns';assert t.findtext('.//SummaryOfOverallLatency/unit')=='clock cycles'
    assert {k:int(t.findtext('.//AreaEstimates/AvailableResources/'+k)) for k in j['capacities']}==j['capacities']
    assert t.findtext('.//ReportVersion/Version')=='2021.1';assert float(t.findtext('.//UserAssignments/TargetClockPeriod'))==j['clock_ns']
    log=(p/'synthesis.log').read_text();assert 'Finished Command csynth_design' in log and 'Exiting vitis_hls' in log
    r['synthesis_completed']=True;r['latency_tag']=t.findtext('.//SummaryOfOverallLatency/Worst-caseLatency')
    rr={k:int(t.findtext('.//AreaEstimates/Resources/'+k)) for k in j['capacities']};r['resources']=rr
    r['resource_feasible']=all(rr[k]<=j['budget'][k] for k in rr);r['resource_headroom']={k:1-rr[k]/j['budget'][k] for k in rr}
    r['estimated_clock_ns']=float(t.findtext('.//SummaryOfTimingAnalysis/EstimatedClockPeriod'));r['timing_qualified']=r['estimated_clock_ns']<=j['clock_ns']
    r['rankable_latency']=str(r['latency_tag']).isdigit() and int(r['latency_tag'])>0
    if r['rankable_latency']:r['latency_cycles']=int(r['latency_tag'])
   cases.append(r)
 result={'schema':'architecture-qualification-results-v1','cases':cases,'new_unique_hls_jobs':len(cases),'completed_csynth':sum(x['synthesis_completed'] for x in cases),'rankable_latency_cases':sum(x['rankable_latency'] for x in cases),'switch_pairs':0,'stay_pairs':0,'cohort_status':'unresolved: no paired rankable latency measurements','gpu_optimizer_updates':0,'gpu_model_forwards':0,'gpu_execution_blocker':'Operational baseline preflight refused Ray worker holding 29782 MiB on assigned V100S; no process interrupted','post_route_checks':0,'promotion':'none; retained registry paths and defaults','hls_runtime_seconds':sum(x['runtime_seconds'] for x in cases),'unexecuted_sealed_jobs':10,'test_outcomes_accessed':0}
 out=ROOT/'docs/architecture-finalization/results.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 return result
if __name__=='__main__':analyze()
