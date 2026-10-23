import pathlib,json,xml.etree.ElementTree as ET,hashlib
root=pathlib.Path('/home/bonus24/requested_latency_20260907')
results=[]
for kernel,top in [('covariance','kernel_covariance'),('chstone-jpeg','chstone_jpeg')]:
 case=root/kernel
 if not (case/'EXIT_CODE').exists():
  print(kernel,'RUNNING'); continue
 row=json.loads((case/'input.json').read_text())
 row['returncode']=int((case/'EXIT_CODE').read_text()); row['run_directory']=str(case)
 report=case/'hls_project/solution1/syn/report'/f'{top}_csynth.xml'
 row['status']='success' if row['returncode']==0 and report.exists() else 'failed'
 if report.exists():
  xml=ET.parse(report).getroot()
  assert xml.findtext('UserAssignments/Part')=='xczu7ev-ffvc1156-2-e'
  assert float(xml.findtext('UserAssignments/TargetClockPeriod'))==10
  row['latency_cycles']={k:xml.findtext('PerformanceEstimates/SummaryOfOverallLatency/'+v) for k,v in [('best','Best-caseLatency'),('worst','Worst-caseLatency')]}
  row['estimated_clock_period_ns']=float(xml.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod'))
  row['resources']={n:int(xml.findtext('AreaEstimates/Resources/'+n)) for n in ['BRAM_18K','DSP','FF','LUT','URAM']}
 row['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir() if p.suffix in ['.c','.cpp','.tcl']}
 (case/'result.json').write_text(json.dumps(row,indent=2)+'\n'); results.append(row)
 print(kernel,json.dumps({k:v for k,v in row.items() if k!='directives'}))
(root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
