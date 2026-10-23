import json, subprocess, textwrap
from pathlib import Path

cases = [
 ("ZCU104_10_full_GAN", "ZCU104", 10.0, "full", "GAN", "network.cpp", "forward_propagation", {"L1":"C2","L2":"C15","L3":"C25","L4":"P","L5":"P","L6":"U30","L7":"U30","L8":"P","L9":"U15","L10":"P","L11":"U25"}),
 ("ZCU104_10_full_jpeg", "ZCU104", 10.0, "full", "chstone-jpeg", "kernel_placeholders.c", "chstone_jpeg", {"L1":"P","L2":"P","L3":"U2","L4":"U2","L5":"C2","L6":"NONE","L7":"NONE","L8":"U3","L9":"U3","L10":"P","L11":"U3","L12":"N","L13":"U4","L14":"U3","L15":"NONE","L16":"NONE","L17":"P","L18":"P","L19":"P","L20":"U4","L21":"P"}),
 ("ZCU104_6p67_full", "ZCU104", 6.67, "full", "serrano-kalman-filter", "kernel.cpp", "krnl_KALMAN", {"L1":"C4","L2":"C4","L3":"P","L4":"U4","L5":"B2","L6":"B2","L7":"C2","L8":"C2","L9":"U2","L10":"N","L11":"P","L12":"P","L13":"P"}),
 ("ZCU104_6p67_full_GAN", "ZCU104", 6.67, "full", "GAN", "network.cpp", "forward_propagation", {"L1":"C2","L2":"C15","L3":"C25","L4":"P","L5":"P","L6":"U30","L7":"U30","L8":"P","L9":"U15","L10":"P","L11":"U25"}),
 ("ZCU104_6p67_full_jpeg", "ZCU104", 6.67, "full", "chstone-jpeg", "kernel_placeholders.c", "chstone_jpeg", {"L1":"P","L2":"P","L3":"U2","L4":"U2","L5":"C2","L6":"NONE","L7":"NONE","L8":"U3","L9":"U3","L10":"P","L11":"U3","L12":"N","L13":"U4","L14":"U3","L15":"NONE","L16":"NONE","L17":"P","L18":"P","L19":"P","L20":"U4","L21":"P"}),
 ("ZCU104_6p67_full_covariance", "ZCU104", 6.67, "full", "covariance", "kernel_placeholders.cpp", "kernel_covariance", {"L1":"P","L2":"U10","L3":"P","L4":"P","L5":"P","L6":"P","L7":"P"}),
 ("U50_10_full", "U50", 10.0, "full", "serrano-kalman-filter", "kernel.cpp", "krnl_KALMAN", {"L1":"C8","L2":"C8","L3":"P","L4":"U8","L5":"B4","L6":"B4","L7":"C4","L8":"C4","L9":"U4","L10":"N","L11":"P","L12":"P","L13":"P"}),
 ("U50_10_full_GAN", "U50", 10.0, "full", "GAN", "network.cpp", "forward_propagation", {"L1":"C2","L2":"C30","L3":"C50","L4":"P","L5":"P","L6":"U30","L7":"U30","L8":"P","L9":"U30","L10":"P","L11":"U50"}),
 ("U50_10_full_jpeg", "U50", 10.0, "full", "chstone-jpeg", "kernel_placeholders.c", "chstone_jpeg", {"L1":"P","L2":"P","L3":"U8","L4":"U8","L5":"C8","L6":"COMP","L7":"COMP","L8":"U3","L9":"U3","L10":"P","L11":"U3","L12":"P","L13":"U4","L14":"U3","L15":"NONE","L16":"NONE","L17":"P","L18":"P","L19":"P","L20":"U4","L21":"P"}),
 ("U50_10_full_covariance", "U50", 10.0, "full", "covariance", "kernel_placeholders.cpp", "kernel_covariance", {"L1":"P","L2":"U10","L3":"P","L4":"P","L5":"P","L6":"P","L7":"P"}),
]

payload = json.dumps(cases, separators=(',', ':'))
remote = r'''import json,re,shutil,subprocess, pathlib, xml.etree.ElementTree as ET
cases=json.loads(%r); root=pathlib.Path('/home/bonus24/target_general_20260907'); root.mkdir(exist_ok=True)
parts={'ZCU104':'xczu7ev-ffvc1156-2-e','U50':'xcu50-fsvh2104-2-e'}
vars={'serrano-kalman-filter':{'L1':'in_local','L2':'out_local','L5':'u_hat_arr','L6':'p_arr_1','L7':'k_arr','L8':'calc_temp_arr'},'GAN':{'L1':'xbuf','L2':'layer_1_out','L3':'layer_2_out'},'chstone-jpeg':{'L5':'QuantBuff','L6':'HuffBuff','L7':'IDCTBuff','L15':'huffsize','L16':'huffcode'}}
def apply(src, ds):
    txt=src.read_text(); out=[]
    for aid,val in ds.items():
      m=re.search(r'((?:/\*)?'+aid+r':(?:\*/)?.*?)(?=(?:/\*)?L\d+:(?:\*/)?|\Z)',txt,re.S)
      if not m: raise RuntimeError('missing '+aid)
      c=m.group(1)
      if val == 'NONE':
        c=re.sub(r'\n?#pragma HLS array_partition[^\n]*','',c)
      elif (val[0] in 'CB' and val[1:].isdigit()) or val == 'COMP':
        typ={'C':'cyclic','B':'block','NONE':'none','COMP':'complete'}[val[0] if val[0] in 'CB' else val]
        fac=int(val[1:]) if val[0] in 'CB' and val[1:].isdigit() else (1 if val == 'COMP' else 0)
        c=re.sub(r'#pragma HLS array_partition[^\n]*',f'#pragma HLS array_partition variable={vars[app][aid]} type={typ} factor={fac} dim=1',c)
      else:
        ii=1 if val=='P' else 0; uf=0 if val in ('P','N') else int(val[1:])
        c=re.sub(r'(#pragma HLS pipeline II=)\d+',r'\g<1>'+str(ii),c)
        c=re.sub(r'(#pragma HLS unroll factor=)\d+',r'\g<1>'+str(uf),c)
      txt=txt[:m.start()]+c+txt[m.end():]
    return txt
results=[]
for name,dev,clk,budget,app,fn,top,ds in cases:
  d=root/name; d.mkdir(exist_ok=True); src=pathlib.Path('/home/bonus24/Application')/app/fn; out=d/fn
  txt=apply(src,ds)
  if app=='GAN':
    out.write_text(txt); [shutil.copy2(pathlib.Path('/home/bonus24/Application/GAN')/x,d/x) for x in ('network.h','weight_definitions.h','tanh.h')]
    add='add_files {network.cpp} -cflags {-I.}'
  elif app=='serrano-kalman-filter':
    out.write_text(txt); [shutil.copy2(pathlib.Path('/home/bonus24/Application/serrano-kalman-filter')/x,d/x) for x in ('ap_fixed.h','ap_int.h') if (pathlib.Path('/home/bonus24/Application/serrano-kalman-filter')/x).exists()]
    add='add_files {kernel.cpp} -cflags {-I.}'
  else: out.write_text(txt); add='add_files {'+fn+'}'
  tcl='\n'.join(['open_project -reset project_'+app.replace('-','_'),'set_top '+top,add,'open_solution -reset solution1 -flow_target vivado','set_part {'+parts[dev]+'}','create_clock -period '+str(clk)+' -name default','config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0','csynth_design','exit'])+'\n'; (d/'run_hls.tcl').write_text(tcl)
  project='project_'+app.replace('-','_'); report=d/project/'solution1/syn/report'/(top+'_csynth.xml')
  if report.exists() and app != 'chstone-jpeg': rc=0
  else:
    log=(d/'synthesis.log').open('w'); rc=subprocess.run(['/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls','-f','run_hls.tcl'],cwd=d,stdout=log,stderr=subprocess.STDOUT).returncode; log.close()
  row={'kernel':app if app!='serrano-kalman-filter' else 'Kalman','device':dev,'clock_period_ns':clk,'resource_budget':budget,'directives':ds,'status':'success' if rc==0 and report.exists() else 'failed'}
  if report.exists():
    x=ET.parse(report).getroot()
    def num(path):
      v=x.findtext(path)
      try: return int(v)
      except (TypeError,ValueError): return None
    row.update(latency=num('PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency'),resources={'bram':num('AreaEstimates/Resources/BRAM_18K'),'dsp':num('AreaEstimates/Resources/DSP'),'ff':num('AreaEstimates/Resources/FF'),'lut':num('AreaEstimates/Resources/LUT')})
  results.append(row); (d/'result.json').write_text(json.dumps(row,indent=2)+'\n'); print(name,row.get('status'),flush=True)
(root/'Target_general_experiments.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in results))
''' % payload
Path('/tmp/target_general_remote.py').write_text(remote)
subprocess.run(['scp','/tmp/target_general_remote.py','bonus24@coroni.microlab.ntua.gr:/tmp/target_general_remote.py'],check=True)
subprocess.run(['ssh','bonus24@coroni.microlab.ntua.gr','python3 /tmp/target_general_remote.py'],check=True)
subprocess.run(['scp','bonus24@coroni.microlab.ntua.gr:/home/bonus24/target_general_20260907/Target_general_experiments.jsonl','Target_general_experiments.jsonl'],check=True)
