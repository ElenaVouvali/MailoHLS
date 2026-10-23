import json,re,pathlib,subprocess
root=pathlib.Path('/home/bonus24/requested_latency_20260907')
root.mkdir(exist_ok=False)
configs={ 'covariance':[(1,0),(0,10),(1,0),(0,10),(1,0),(1,0),(0,10)], 'chstone-jpeg':[(1,0),(1,0),(0,4),(0,4),('QuantBuff','cyclic',4,1),('HuffBuff','complete',0,1),('IDCTBuff','complete',0,1),(0,3),(0,3),(1,0),(0,3),(1,0),(0,2),(0,3),('huffsize','none',0,0),('huffcode','none',0,0),(1,0),(1,0),(1,0),(0,4),(1,0)]}
for kernel,values in configs.items():
 case=root/kernel; case.mkdir()
 ext='cpp' if kernel=='covariance' else 'c'
 source=pathlib.Path('/home/bonus24/Application')/kernel/('kernel_placeholders.'+ext)
 text=source.read_text(); directives=[]
 for i,v in enumerate(values,1):
  aid='L'+str(i)
  pattern=r'(/\*'+aid+r':\*/.*?)(?=/\*L\d+:\*/|\Z)'
  match=re.search(pattern,text,re.S); assert match,aid
  chunk=match.group()
  if len(v)==2:
   d=dict(action_id=aid,kind='loop',pipeline_ii=v[0],unroll_factor=v[1])
   chunk,n=re.subn(r'(#pragma HLS pipeline II=)\d+',lambda m:m[1]+str(v[0]),chunk); assert n==1
   chunk,n=re.subn(r'(#pragma HLS unroll factor=)\d+',lambda m:m[1]+str(v[1]),chunk); assert n==1
  else:
   d=dict(action_id=aid,kind='array',variable=v[0],partition_type=v[1],partition_factor=v[2],partition_dim=v[3])
   chunk,n=re.subn(r'#pragma HLS array_partition[^\n]*',f'#pragma HLS array_partition variable={v[0]} type={v[1]} factor={v[2]} dim={v[3]}',chunk); assert n==1
  text=text[:match.start()]+chunk+text[match.end():]; directives.append(d)
 (case/('kernel.'+ext)).write_text(text)
 (case/'input.json').write_text(json.dumps(dict(kernel=kernel,stage='stage-3',objective='LATENCY',target_context=dict(device='ZCU104',clock_period_ns=10.0,budget='full'),directives=directives),indent=2)+'\n')
 top='kernel_covariance' if kernel=='covariance' else 'chstone_jpeg'
 (case/'run_hls.tcl').write_text(f'''open_project -reset hls_project
set_top {top}
add_files kernel.{ext}
open_solution -reset solution1 -flow_target vivado
set_part {{xczu7ev-ffvc1156-2-e}}
create_clock -period 10 -name default
config_array_partition -complete_threshold 0 -throughput_driven off
config_compile -pipeline_loops 0
csynth_design
exit
''')
runner='''#!/bin/bash
source /opt/Xilinx/Vitis_HLS/2021.1/settings64.sh
run_one() {
 cd "$1" || return
 date -Is > STARTED
 vitis_hls -f run_hls.tcl > synthesis.log 2>&1
 rc=$?
 echo "$rc" > EXIT_CODE
 date -Is > FINISHED
}
run_one /home/bonus24/requested_latency_20260907/covariance &
run_one /home/bonus24/requested_latency_20260907/chstone-jpeg &
wait
'''
(root/'run.sh').write_text(runner)
with (root/'launcher.log').open('w') as log:
 p=subprocess.Popen(['nohup','bash',str(root/'run.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
(root/'PID').write_text(str(p.pid)+'\n')
print(root, 'launcher PID', p.pid)
