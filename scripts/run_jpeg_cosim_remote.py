import json, subprocess
from pathlib import Path

cases = [
    ("ZCU104_10_full_jpeg", "xczu7ev-ffvc1156-2-e", 10.0),
    ("ZCU104_6p67_full_jpeg", "xczu7ev-ffvc1156-2-e", 6.67),
    ("U50_10_full_jpeg", "xcu50-fsvh2104-2-e", 10.0),
]
remote = r'''import pathlib,shutil,re,json,subprocess
cases=json.loads(%r); root=pathlib.Path('/home/bonus24/target_general_20260907')
dirs_high={'L1':'P','L2':'P','L3':'U8','L4':'U8','L5':'C8','L6':'COMP','L7':'COMP','L8':'U3','L9':'U3','L10':'P','L11':'U3','L12':'P','L13':'U4','L14':'U3','L15':'NONE','L16':'NONE','L17':'P','L18':'P','L19':'P','L20':'U4','L21':'P'}
dirs_low={'L1':'P','L2':'P','L3':'U2','L4':'U2','L5':'C2','L6':'NONE','L7':'NONE','L8':'U3','L9':'U3','L10':'P','L11':'U3','L12':'N','L13':'U4','L14':'U3','L15':'NONE','L16':'NONE','L17':'P','L18':'P','L19':'P','L20':'U4','L21':'P'}
vars={'L5':'QuantBuff','L6':'HuffBuff','L7':'IDCTBuff','L15':'huffsize','L16':'huffcode'}
def materialize(txt):
 dirs = dirs_high if 'U50' in name else dirs_low
 for aid,val in dirs.items():
  m=re.search(r'((?:/\*)?'+aid+r':(?:\*/)?.*?)(?=(?:/\*)?L\d+:(?:\*/)?|\Z)',txt,re.S)
  if not m: raise RuntimeError(aid)
  c=m.group(1)
  if val=='NONE': c=re.sub(r'\n?#pragma HLS array_partition[^\n]*','',c)
  elif val=='COMP': c=re.sub(r'#pragma HLS array_partition[^\n]*',f'#pragma HLS array_partition variable={vars[aid]} type=complete factor=1 dim=1',c)
  elif val.startswith('C'): c=re.sub(r'#pragma HLS array_partition[^\n]*',f'#pragma HLS array_partition variable={vars[aid]} type=cyclic factor={int(val[1:])} dim=1',c)
  else:
   c=re.sub(r'(#pragma HLS pipeline II=)\d+',r'\g<1>'+str(1 if val=='P' else 0),c)
   c=re.sub(r'(#pragma HLS unroll factor=)\d+',r'\g<1>'+str(0 if val in ('P','N') else int(val[1:])),c)
  txt=txt[:m.start()]+c+txt[m.end():]
 return txt
out=[]
for name,part,clk in cases:
 d=root/name; src=pathlib.Path('/home/bonus24/Application/chstone-jpeg/kernel_placeholders.c'); (d/'kernel_placeholders.c').write_text(materialize(src.read_text())); shutil.copy2('/home/bonus24/Application/chstone-jpeg/tb_jpeg.c',d/'tb_jpeg.c')
 tcl='\n'.join(['open_project -reset project_jpeg','set_top chstone_jpeg','add_files {kernel_placeholders.c}','add_files -tb {tb_jpeg.c}','open_solution -reset solution1 -flow_target vivado','set_part {'+part+'}','create_clock -period '+str(clk)+' -name default','config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0','#csim_design','csynth_design','cosim_design -rtl verilog -tool xsim -trace_level none','exit'])+'\n'; (d/'run_hls.tcl').write_text(tcl)
 log=(d/'cosim_synthesis.log').open('w'); rc=subprocess.run(['/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls','-f','run_hls.tcl'],cwd=d,stdout=log,stderr=subprocess.STDOUT).returncode; log.close()
 rpt=d/'project_jpeg/solution1/sim/report/chstone_jpeg_cosim.rpt'; latency=None
 if rpt.exists():
  for line in rpt.read_text().splitlines():
   if 'Verilog' in line and 'Pass' in line: latency=int(re.findall(r'\d+',line)[0]); break
 out.append({'name':name,'returncode':rc,'status':'success' if rc==0 and latency is not None else 'failed','latency':latency,'cosim_report':str(rpt)})
(root/'jpeg_cosim_results.json').write_text(json.dumps(out,indent=2)+'\n')
''' % json.dumps(cases)
Path('/tmp/jpeg_cosim_remote.py').write_text(remote)
subprocess.run(['scp','/tmp/jpeg_cosim_remote.py','bonus24@coroni.microlab.ntua.gr:/tmp/jpeg_cosim_remote.py'],check=True)
subprocess.run(['ssh','bonus24@coroni.microlab.ntua.gr','python3 /tmp/jpeg_cosim_remote.py'],check=True)
subprocess.run(['scp','bonus24@coroni.microlab.ntua.gr:/home/bonus24/target_general_20260907/jpeg_cosim_results.json','jpeg_cosim_results.json'],check=True)
