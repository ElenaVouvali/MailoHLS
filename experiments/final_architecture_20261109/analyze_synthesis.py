"""Strict partial-QoR receipts: undefined latency is never synthesis failure."""
import json,hashlib,re,xml.etree.ElementTree as ET
from pathlib import Path
from .audit import ROOT,OUT,sha

def batch(bundle,results):
 manifest=json.loads((bundle/'manifest.json').read_text());prov=json.loads((results/'batch_provenance.json').read_text());assert prov['manifest_sha256']==sha(bundle/'manifest.json');assert prov['runner_sha256']==sha(bundle/'run_batch.py')
 rows=[]
 for j in manifest['jobs']:
  d=results/j['job_id'];r=json.loads((d/'result.json').read_text());receipt=json.loads((d/'receipt.json').read_text())
  assert sha(d/'receipt.json')==r['receipt_sha256'] and sha(d/'synthesis.log')==r['log_sha256']
  assert sha(d/'run_hls.tcl')==receipt['tcl_sha256']==j['expected_tcl_sha256'];assert receipt['source_sha256']==j['materialized_source_sha256']
  row={'job_id':j['job_id'],'checkpoint_step':j.get('attributions',[{}])[0].get('checkpoint_step'),'kernel':j['kernel'],'part':j['part'],'clock_ns':j['clock_ns'],'budget':j['budget'],'assignment':j['assignment'],'attributions':j.get('attributions',[]),'source_sha256':j['original_source_sha256'],'headers':j['headers'],'materialized_source_sha256':j['materialized_source_sha256'],'runner_status':r['status'],'runtime_seconds':r['runtime_seconds'],'error':r['error'],'csynth_completed':False,'latency_cycles':None,'resources':None,'resource_feasible':None,'hls_estimated_timing_qualified':None,'receipt_sha256':r['receipt_sha256'],'log_sha256':r['log_sha256'],'report_sha256':r['report_sha256'],'batch_provenance_sha256':sha(results/'batch_provenance.json')}
  report=d/'hls_project/solution1/syn/report'/(j['top']+'_csynth.xml')
  if r['returncode']==0 and report.exists():
   assert sha(report)==r['report_sha256'];tree=ET.parse(report)
   for path,value in [('.//UserAssignments/Part',j['part']),('.//UserAssignments/TopModelName',j['top']),('.//ReportVersion/Version','2021.1'),('.//UserAssignments/FlowTarget','vivado'),('.//UserAssignments/unit','ns'),('.//SummaryOfOverallLatency/unit','clock cycles')]:assert tree.findtext(path)==value
   assert float(tree.findtext('.//UserAssignments/TargetClockPeriod'))==j['clock_ns'];assert 'Finished Command csynth_design' in (d/'synthesis.log').read_text()
   def integer(path):
    value=tree.findtext(path);assert value is not None and re.fullmatch('[0-9]+',value.strip());return int(value)
   capacities={k:integer('.//AreaEstimates/AvailableResources/'+k) for k in j['capacities']};assert capacities==j['capacities']
   row['csynth_completed']=True;row['latency_tag']=tree.findtext('.//SummaryOfOverallLatency/Worst-caseLatency');row['resources']={k:integer('.//AreaEstimates/Resources/'+k) for k in capacities};row['resource_feasible']=all(row['resources'][k]<=j['budget'][k] for k in capacities);row['headroom']={k:1-row['resources'][k]/j['budget'][k] for k in capacities}
   row['estimated_clock_ns']=float(tree.findtext('.//SummaryOfTimingAnalysis/EstimatedClockPeriod'));row['hls_estimated_timing_qualified']=row['estimated_clock_ns']<=j['clock_ns']
   if row['latency_tag'].isdigit() and int(row['latency_tag'])>0:row['latency_cycles']=int(row['latency_tag'])
  rows.append(row)
 return rows

def main():
 rows=batch(OUT/'checkpoint_0_150_synthesis_v3_reference',OUT/'coroni_checkpoint_results')+batch(OUT/'checkpoint_0_150_u200_pathfinder',OUT/'coroni_u200_results');sim=[]
 for job in ('cp_01','cp_02'):
  d=OUT/'coroni_cosim_results'/job;r=json.loads((d/'simulation_receipt.json').read_text());source_row=next(x for x in rows if x['job_id']==job);assert r['source_job_receipt_sha256']==source_row['receipt_sha256'];assert r['materialized_source_sha256']==source_row['materialized_source_sha256']
  for phase in r['results']:
   assert sha(d/(phase['mode']+'.log'))==phase['log_sha256'];assert sha(d/('run_'+phase['mode']+'.tcl'))==phase['tcl_sha256'];assert phase['returncode']==0
  log=(d/'cosim.log').read_text();assert 'C/RTL co-simulation finished: PASS' in log and 'all 1024 outputs matched' in log
  report=d/'hls_project/solution1/sim/report/workload_cosim.rpt';line=next(x for x in report.read_text().splitlines() if '|   Verilog|' in x);fields=[x.strip() for x in line.split('|') if x.strip()];assert fields[1]=='Pass';cycles=int(fields[2]);assert cycles==int(fields[3])==int(fields[4])
  sim.append({'job_id':job,'checkpoint_step':source_row['checkpoint_step'],'RTL_testcase_cycles':cycles,'functional_testcase_pass':True,'input_cells':1048576,'checked_outputs':1024,'testcases':1,'report_sha256':sha(report),'simulation_receipt_sha256':sha(d/'simulation_receipt.json'),'testbench_sha256':r['testbench_sha256'],'scope':r['scope']})
 result={'schema':'architecture-r2-matched-physical-comparison-v1','cases':rows,'simulations':sim,'paired_RTL_speedup':sim[0]['RTL_testcase_cycles']/sim[1]['RTL_testcase_cycles'],'RTL_latency_improvement_percent':100*(sim[0]['RTL_testcase_cycles']-sim[1]['RTL_testcase_cycles'])/sim[0]['RTL_testcase_cycles'],'hardware_superiority':'not demonstrated','independent_families_with_paired_finite_latency':1,'total_new_CSYNTH_calls_this_revision':7,'prior_revision_unique_CSYNTH_calls':2,'RTL_simulations':2,'C_simulations':2,'post_route_runs':0,'new_complete_models_promoted':0,'limitations':['HLS baseline latency undefined for Pathfinder','Viterbi both matched20minute timeouts: unknown QoR','only one deterministic RTL testcase and default AXI BFM; no routed timing','U200 prompt budgets50%, ZCU104 full; model comparison matched within context, not isolated target-axis experiment','no locked-test outcomes used']}
 out=ROOT/'experiments/final_architecture_20261109/results/checkpoint_physical_comparison.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'RTL_speedup':result['paired_RTL_speedup'],'cases':[(x['job_id'],x['latency_cycles'],x['resources']) for x in rows]},indent=2))
if __name__=='__main__':main()
