import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

rows = [json.loads(line) for line in Path('Target_general_experiments.jsonl').read_text().splitlines() if json.loads(line)['kernel'] == 'covariance']
script = r'''
import json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
rows=json.loads(__ROWS__); root=Path('/home/bonus24/covariance_target_resynth_20260907'); root.mkdir(exist_ok=True)
parts={'ZCU104':'xczu7ev-ffvc1156-2-e','U50':'xcu50-fsvh2104-2-e'}
source=Path('/home/bonus24/Application/covariance/kernel_placeholders.cpp').read_text()
for idx,row in enumerate(rows,1):
 d=root/('%02d_%s_%s'%(idx,row['device'],str(row['clock_period_ns']).replace('.','p'))); d.mkdir(exist_ok=True)
 txt=re.sub(r'^\s*#pragma HLS (?:pipeline|unroll|array_partition)[^\n]*\n','',source,flags=re.M)
 for directive in row['directives']:
  aid=directive['action_id']; m=re.search(r'(/\*'+aid+r':\*/.*?)(?=/\*L\d+:\*/|\Z)',txt,re.S); assert m,aid
  block=m.group(1); pragmas=[]
  if directive['kind']=='loop':
   if directive['pipeline_ii']: pragmas.append('#pragma HLS pipeline II=%d'%directive['pipeline_ii'])
   if directive['unroll_factor']: pragmas.append('#pragma HLS unroll factor=%d'%directive['unroll_factor'])
   pos=block.index('{')+1
  new=block if not pragmas else block[:pos]+'\n'+'\n'.join(pragmas)+'\n'+block[pos:]
  txt=txt[:m.start()]+new+txt[m.end():]
 (d/'kernel_placeholders.cpp').write_text(txt)
 (d/'run_hls.tcl').write_text('\n'.join(['open_project -reset project_covariance','set_top kernel_covariance','add_files {kernel_placeholders.cpp}','open_solution -reset solution1 -flow_target vivado','set_part {'+parts[row['device']]+'}','create_clock -period '+str(row['clock_period_ns'])+' -name default','config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0','csynth_design','exit'])+'\n')
 with (d/'synthesis.log').open('w') as log: rc=subprocess.run(['/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls','-f','run_hls.tcl'],cwd=d,stdout=log,stderr=subprocess.STDOUT).returncode
 rpt=d/'project_covariance/solution1/syn/report/kernel_covariance_csynth.xml'; out=dict(row,resynthesis_returncode=rc,resynthesis_status='failed',resynthesis_run_directory=str(d))
 if rpt.exists():
  x=ET.parse(rpt).getroot(); n=lambda p:int(x.findtext(p)) if (x.findtext(p) or '').isdigit() else None
  out.update(resynthesis_status='success' if rc==0 else 'report_only',resynthesis_latency_best_cycles=n('PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency'),resynthesis_latency_worst_cycles=n('PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency'),resynthesis_resources={'bram':n('AreaEstimates/Resources/BRAM_18K'),'dsp':n('AreaEstimates/Resources/DSP'),'ff':n('AreaEstimates/Resources/FF'),'lut':n('AreaEstimates/Resources/LUT')},estimated_clock_period_ns=float(x.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod')),resynthesis_report=str(rpt))
 (d/'result.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,separators=(',',':')),flush=True)
'''.replace('__ROWS__', repr(json.dumps(rows)))
Path('/tmp/run_covariance_remote.py').write_text(script)
subprocess.run(['scp', '/tmp/run_covariance_remote.py', 'bonus24@coroni.microlab.ntua.gr:/tmp/run_covariance_remote.py'], check=True)
subprocess.run(['ssh', 'bonus24@coroni.microlab.ntua.gr', 'python3 /tmp/run_covariance_remote.py'], check=True)
