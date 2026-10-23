import concurrent.futures,json,os,re,subprocess,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
def run(row):
 d=ROOT/row['candidate']; start=time.time();(d/'STARTED').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
 print('START',row['candidate'],flush=True)
 env=os.environ.copy();tmp=d/'tmp';tmp.mkdir(exist_ok=True);env.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
 with (d/'run.log').open('w') as log:
  rc=subprocess.call(['/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls','-f','run_hls.tcl'],cwd=str(d),stdout=log,stderr=subprocess.STDOUT,env=env)
 (d/'EXIT_CODE').write_text(str(rc)+'\n')
 result=dict(row,returncode=rc,elapsed_seconds=round(time.time()-start,2),run_directory=str(d),implementation_status='not_run')
 report=d/'project/solution1/syn/report'/(row['top']+'_csynth.xml')
 result.update(synthesis_status='failed',cosimulation_status='not_completed',latency=None)
 if report.exists():
  x=ET.parse(str(report)).getroot()
  assert x.findtext('UserAssignments/Part')=='xczu7ev-ffvc1156-2-e'
  assert float(x.findtext('UserAssignments/TargetClockPeriod'))==10.0
  def numeric(path):
   v=x.findtext(path)
   try:return int(v)
   except:return None
  result.update(synthesis_status='completed',synthesis_latency_cycles={k:numeric('PerformanceEstimates/SummaryOfOverallLatency/'+v) for k,v in [('min','Best-caseLatency'),('max','Worst-caseLatency')]},resources={k:numeric('AreaEstimates/Resources/'+v) for k,v in [('bram','BRAM_18K'),('dsp','DSP'),('ff','FF'),('lut','LUT')]},resources_source='hls_synthesis_estimate',estimated_clock_period_ns=float(x.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod')),synthesis_report=str(report))
 cr=d/'project/solution1/sim/report'/(row['top']+'_cosim.rpt')
 if cr.exists():
  m=re.search(r'\|\s*Verilog\s*\|\s*Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',cr.read_text())
  if m: result.update(cosimulation_status='passed',latency=int(m[1]),latency_source='rtl_cosimulation_min_cycles',cosimulation_latency_cycles=dict(min=int(m[1]),avg=int(m[2]),max=int(m[3])),cosimulation_report=str(cr))
 result['status']='success' if rc==0 and result['cosimulation_status']=='passed' else 'incomplete'
 (d/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('END',row['candidate'],result['status'],result['latency'],flush=True)
 return result
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(run,row) for row in manifest]
 for f in concurrent.futures.as_completed(futures):
  try: results.append(f.result())
  except Exception: traceback.print_exc()
  (ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
(ROOT/'COMPLETE').write_text(str(len(results))+' jobs finished\n')
