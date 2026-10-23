import json,re,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs'; OUT.mkdir(exist_ok=True)
salt=[json.loads(s) for s in Path('SALT_experiments.jsonl').read_text().splitlines()]
(ROOT/'SALT_before.jsonl').write_text(Path('SALT_experiments.jsonl').read_text())
def base(k,obj): return next(r for r in salt if r['kernel']==k and r.get('objective')==obj)['directives']
def loops(vals): return [dict(action_id='L'+str(i),kind='loop',pipeline_ii=int(v=='P'),unroll_factor=int(v[1:]) if v.startswith('U') else 0) for i,v in enumerate(vals,1)]
def mod(k,obj,changes):
 ds=json.loads(json.dumps(base(k,obj)))
 for d in ds:
  v=changes.get(d['action_id'])
  if v is None: continue
  if d['kind']=='loop': d.update(pipeline_ii=int(v=='P'),unroll_factor=int(v[1:]) if v.startswith('U') else 0)
  else: d.update(partition_type='none' if v=='N' else 'cyclic',partition_factor=0 if v=='N' else int(v[1:]),partition_dim=0 if v=='N' else d['partition_dim'])
 return ds
CASES=[
 ('GAN-AREA-BANKED','GAN','AREA',mod('GAN','AREA',{'L1':'C2','L2':'C5','L3':'C5'})),
 ('AES-U4-NOPIPE','chstone-aes','ADP',loops(['N','N','U4','U4','U4','U4'])),
 ('JAC-U2','jacobi-2d','LATENCY',loops(['N','U2','P','U2','P'])),
 ('ROS-U2-NOBANK','rosetta-3d-rendering','ADP',mod('rosetta-3d-rendering','AREA',{'L2':'U2','L5':'U2','L8':'U2'})),
 ('TRMM-U10','trmm-opt','LATENCY',loops(['U10','P','N'])),
 ('TRMM-P3','trmm-opt','ADP',loops(['U5','P','P'])),
 ('AES-U2-NOPIPE','chstone-aes','ADP',loops(['N','N','U2','U2','U2','U2'])),
 ('SYR-2222','syr2k','ADP',loops(['U2','U2','U2','U2'])),
 ('KAL-BRAM8','Kalman','LATENCY',mod('Kalman','LATENCY',{'L1':'C8','L2':'C8','L4':'N'})),
]
SPECS={
 'GAN':('predicted_directives_18_20260904/area/GAN/kernel_placeholders.cpp','forward_propagation'),
 'chstone-aes':('corrected_changed_20260904/area/chstone-aes/kernel_placeholders.c','chstone_aes'),
 'jacobi-2d':('corrected_changed_20260904/area/jacobi-2d/kernel_placeholders.cpp','kernel_jacobi_2d'),
 'rosetta-3d-rendering':('corrected_changed_20260904/area/rosetta-3d-rendering/kernel_placeholders.cpp','rendering'),
 'trmm-opt':('Application/trmm-opt/kernel.cpp','kernel_trmm'),
 'syr2k':('Application/syr2k/kernel.cpp','kernel_syr2k'),
 'Kalman':('Application/serrano-kalman-filter/kernel.cpp','krnl_KALMAN')}
# Remove only action-controlled optimization pragmas. Preserve the upstream
# fixed pragmas before L1 (notably Rosetta's projection pipeline).
def clean(s): return re.sub(r'^[ \t]*(?://[ \t]*)?#pragma\s+HLS\s+(?:pipeline|unroll|array_partition)[^\n]*\n','',s,flags=re.M|re.I)
def materialize(src,ds):
 marks=list(re.finditer(r'/\*L\d+:\*/|(?m:^[ \t]*L\d+:)',src))
 assert len(marks)==len(ds),(len(marks),len(ds))
 result=src[:marks[0].start()]
 for i,(m,d) in enumerate(zip(marks,ds)):
  assert re.search(r'L\d+',m.group())[0]==d['action_id']
  block=clean(src[m.start():marks[i+1].start() if i+1<len(marks) else len(src)])
  pragmas=[]
  if d['kind']=='loop':
   if d['pipeline_ii']: pragmas.append('#pragma HLS pipeline II=%d'%d['pipeline_ii'])
   if d['unroll_factor']: pragmas.append('#pragma HLS unroll factor=%d'%d['unroll_factor'])
   header=re.search(r'for\s*\([^\n]*\)[^\n]*',block); assert header,d
   pos=header.end()
  else:
   if d['partition_type']!='none':
    pragma='#pragma HLS array_partition variable=%s type=%s dim=%d'%(d['variable'],d['partition_type'],d['partition_dim'])
    if d['partition_type']!='complete': pragma+=' factor=%d'%d['partition_factor']
    pragmas.append(pragma)
   pos=block.index(';')+1
  if pragmas: block=block[:pos]+'\n'+'\n'.join(pragmas)+'\n'+block[pos:]
  result+=block
 # Optimization-only change: all other source text is preserved.
 assert re.sub(r'\s+','',clean(result))==re.sub(r'\s+','',clean(src))
 return result
BENCH={
'chstone-aes':'''#include <stdio.h>
int chstone_aes(void);
int main(void) { int r=chstone_aes(); printf("AES errors=%d\\n",r); return r!=0; }
''',
'jacobi-2d':'''#include <cmath>
#include <cstdio>
extern "C" void kernel_jacobi_2d(int,int,double[90][90],double[90][90]);
int main(){static double a[90][90],b[90][90],ra[90][90],rb[90][90];
for(int i=0;i<90;i++)for(int j=0;j<90;j++){a[i][j]=ra[i][j]=(i*7+j*3+1)/100.0;b[i][j]=rb[i][j]=(i+j+2)/200.0;}
for(int t=0;t<40;t++){for(int i=1;i<89;i++)for(int j=1;j<89;j++)rb[i][j]=0.2*(ra[i][j]+ra[i][j-1]+ra[i][j+1]+ra[i+1][j]+ra[i-1][j]);for(int i=1;i<89;i++)for(int j=1;j<89;j++)ra[i][j]=0.2*(rb[i][j]+rb[i][j-1]+rb[i][j+1]+rb[i+1][j]+rb[i-1][j]);}
kernel_jacobi_2d(40,90,a,b);int errors=0;for(int i=0;i<90;i++)for(int j=0;j<90;j++)if(!std::isfinite(a[i][j])||!std::isfinite(b[i][j])||fabs(a[i][j]-ra[i][j])>1e-8||fabs(b[i][j]-rb[i][j])>1e-8)errors++;
printf("Jacobi mismatches=%d\\n",errors);return errors!=0;}
''',
 'trmm-opt':'''#include <cmath>
#include <cstdio>
extern "C" void kernel_trmm(double,double[60][60],double[60][80]);
int main(){static double a[60][60],b[60][80],ref[60][80];for(int i=0;i<60;i++){for(int k=0;k<60;k++)a[i][k]=((i+k)%7+1)/100.0;for(int j=0;j<80;j++)b[i][j]=ref[i][j]=((i*3+j)%13+1)/20.0;}
for(int i=0;i<60;i++)for(int j=0;j<80;j++){double s=ref[i][j];for(int k=i+1;k<60;k++)s+=a[k][i]*ref[k][j];ref[i][j]=1.25*s;}
kernel_trmm(1.25,a,b);int errors=0;for(int i=0;i<60;i++)for(int j=0;j<80;j++)if(!std::isfinite(b[i][j])||fabs(b[i][j]-ref[i][j])>1e-8)errors++;printf("TRMM mismatches=%d\\n",errors);return errors!=0;}
''',
 'syr2k':'''#include <cmath>
#include <cstdio>
extern "C" void kernel_syr2k(double,double,double[80][80],double[80][60],double[80][60]);
int main(){static double a[80][60],b[80][60],c[80][80],ref[80][80];for(int i=0;i<80;i++){for(int k=0;k<60;k++){a[i][k]=((i+k)%11+1)/100.0;b[i][k]=((i*3+k)%7+1)/100.0;}for(int j=0;j<80;j++)c[i][j]=ref[i][j]=(i+j+1)/200.0;}
for(int i=0;i<80;i++){for(int j=0;j<=i;j++)ref[i][j]*=0.75;for(int k=0;k<60;k++)for(int j=0;j<=i;j++)ref[i][j]+=a[j][k]*1.25*b[i][k]+b[j][k]*1.25*a[i][k];}
kernel_syr2k(1.25,0.75,c,a,b);int errors=0;for(int i=0;i<80;i++)for(int j=0;j<80;j++)if(!std::isfinite(c[i][j])||fabs(c[i][j]-ref[i][j])>1e-8)errors++;printf("SYR2K mismatches=%d\\n",errors);return errors!=0;}
''',
'GAN':'''#include <cmath>
#include <cstdio>
#include "network.h"
void reference_forward_propagation(float*,float*);
int main(){float x[392],y[392],ref[392];for(int i=0;i<392;i++){x[i]=((i*7)%31-15)/64.0f;y[i]=ref[i]=0;}
reference_forward_propagation(x,ref);forward_propagation(x,y);int errors=0;for(int i=0;i<392;i++)if(!std::isfinite(y[i])||fabs(y[i]-ref[i])>1e-6)errors++;printf("GAN mismatches=%d\\n",errors);return errors!=0;}
''',
'Kalman':'''#include "ap_int.h"
#include <cstdio>
extern "C" void krnl_KALMAN(ap_uint<256>*,ap_uint<256>*);
extern "C" void reference_krnl_KALMAN(ap_uint<256>*,ap_uint<256>*);
int main(){static ap_uint<256> in[32768],out[32768],ref[32768];union {float f;unsigned u;} v;
for(int i=0;i<32768;i++){in[i]=0;out[i]=ref[i]=0;for(int j=0;j<8;j++){v.f=((i*8+j)%127)/128.0f;in[i].range(j*32+31,j*32)=v.u;}}
reference_krnl_KALMAN(in,ref);krnl_KALMAN(in,out);int errors=0;for(int i=0;i<32768;i++)if(out[i]!=ref[i])errors++;printf("Kalman mismatches=%d\\n",errors);return errors!=0;}
'''}
manifest=[]
for name,k,obj,ds in CASES:
 d=OUT/name;d.mkdir(exist_ok=True)
 srcname,top=SPECS[k]; source=ROOT/srcname; src=source.read_text(); ext=source.suffix
 (d/('kernel'+ext)).write_text(materialize(src,ds))
 extra=[]
 if k=='GAN':
  for n in ['ap_fixed.h','network.h','weight_definitions.h','tanh.h']: shutil.copy2(source.parent/n,d/n)
  ref=clean(src)
  for symbol in ['forward_propagation','ReLU','tanh']: ref=re.sub(r'\b'+symbol+r'\b','reference_'+symbol,ref)
  (d/'reference.cpp').write_text(ref);extra=['reference.cpp']
 if k=='Kalman':
  ref=re.sub(r'\bkrnl_KALMAN\b','reference_krnl_KALMAN',clean(src));(d/'reference.cpp').write_text(ref);extra=['reference.cpp']
 if k=='rosetta-3d-rendering':
  for n in ['tb_rendering.cpp','input_data.h','outputs_golden.txt']: shutil.copy2(ROOT/'Application'/k/n,d/n)
  tb='tb_rendering.cpp';extra=['outputs_golden.txt']
 else:
  tb='tb.c' if ext=='.c' else 'tb.cpp';(d/tb).write_text(BENCH[k])
 tcl=['open_project -reset project','set_top '+top,'add_files {kernel'+ext+'} -cflags {-I. -D__gmp_const=const}', 'add_files -tb {'+tb+'} -cflags {-I. -D__gmp_const=const}']
 tcl+=['add_files -tb {'+n+'} -cflags {-I. -D__gmp_const=const}' for n in extra]
 tcl+=['open_solution -reset solution1 -flow_target vivado','set_part {xczu7ev-ffvc1156-2-e}','create_clock -period 10 -name default','config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0']
 if k=='GAN': tcl+=['set_directive_interface -depth 392 '+top+' '+p for p in ['x','y']]
 if k=='Kalman': tcl+=['set_directive_interface -depth 32768 '+top+' '+p for p in ['in','out']]
 tcl+=['#csim_design','csynth_design','cosim_design -rtl verilog -tool xsim -trace_level none','exit']
 (d/'run_hls.tcl').write_text('\n'.join(tcl)+'\n')
 row=dict(candidate=name,kernel=k,stage='final-probe',objective=obj,target_context=dict(device='ZCU104',clock_period_ns=10.0,budget='full'),directives=ds,top=top,reference_source=srcname,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()},arithmetic_headers='SALT local ap_fixed compatibility header' if k=='GAN' else 'Vitis HLS vendor headers')
 (d/'input.json').write_text(json.dumps(row,indent=2)+'\n');manifest.append(row)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared and verified',len(manifest),'directive configurations')
