#!/usr/bin/env python3
"""Concurrent local directive search for four latency/speedup bands on Coroni."""
import argparse, concurrent.futures, datetime, hashlib, json, os, re, shutil, signal, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET

BASELINES={'chstone-jpeg':404293,'covariance':196127,'Kalman':10426447,'GAN':299204}
TARGETS={'chstone-jpeg':(0.0,0.696), 'covariance':(2.788,3.28835), 'Kalman':(25.0,30.0), 'GAN':(90.0,100.0)}
SPEC={'GAN':('GAN','network.cpp','forward_propagation'),'chstone-jpeg':('chstone-jpeg','kernel_placeholders.c','chstone_jpeg'),'Kalman':('serrano-kalman-filter','kernel_placeholders.cpp','krnl_KALMAN'),'covariance':('covariance','kernel_placeholders.cpp','kernel_covariance')}
PH=re.compile(r'auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+)\}',re.I)
ASSIGN=re.compile(r'^auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+)\}\s*=\s*(\S+)$',re.I)
def target(d): return '\n'.join('auto{%s} = %s'%(k,v) for k,v in d.items())
def parse_target(t):
 d={}
 for line in t.splitlines():
  m=ASSIGN.match(line.strip())
  if m:d[m.group(1).upper()]=m.group(2).lower()
 return d
def from_directives(ds):
 d={}
 for x in ds:
  a=x['action_id'];
  if x['kind']=='loop': d['_PIPE_'+a]=str(x.get('pipeline_ii') or 0);d['_UNROLL_'+a]=str(x.get('unroll_factor') or 0)
  else:d['_ARRAY_T_'+a]=x.get('partition_type','none');d['_ARRAY_F_'+a]=str(x.get('partition_factor') or 0);d['_ARRAY_D_'+a]=str(x.get('partition_dim') or 0)
 return d
def variants(kernel,base):
 out=[base]
 # Legal conservative one-action neighbors: pipeline toggle, and power-of-two
 # unroll/partition changes around each measured base value.
 labels=sorted({k.rsplit('_',1)[-1] for k in base})
 for label in labels:
  if '_PIPE_'+label in base:
   for value in ('0','1'):
    if value!=base['_PIPE_'+label]:
     d=dict(base);d['_PIPE_'+label]=value;out.append(d)
  if '_UNROLL_'+label in base:
   old=int(base['_UNROLL_'+label]); vals={0,1,2,4,8,16,30,50}
   for value in sorted(vals):
    if value!=old and value<=max(64,old*2):
     d=dict(base);d['_UNROLL_'+label]=str(value);out.append(d)
  if '_ARRAY_T_'+label in base:
   old=int(base['_ARRAY_F_'+label]); vals={0,1,2,4,8,16,32,64}
   for value in sorted(vals):
    typ=base['_ARRAY_T_'+label]
    if typ=='none': continue
    if value==0 and typ in ('block','cyclic'): continue
    if value!=old:
     d=dict(base);d['_ARRAY_F_'+label]=str(value);out.append(d)
 # deduplicate while preserving search order
 seen=set(); result=[]
 for d in out:
  key=json.dumps(d,sort_keys=True)
  if key not in seen:seen.add(key);result.append(d)
 return result
def materialize(src,t):
 text=src.read_text(); d=parse_target(t); ph={x.upper() for x in PH.findall(text)}
 if ph:
  if ph-set(d):raise ValueError('missing placeholders '+str(sorted(ph-set(d))))
  for k,v in d.items():
   text=text.replace('auto{'+k+'}',v)
  # A zero value denotes an inactive directive in the dataset.  It must not
  # become an HLS pragma with II=0 or factor=0; remove those lines entirely.
  text=re.sub(r'^\s*#pragma HLS pipeline II=0\s*\n?', '', text, flags=re.M|re.I)
  text=re.sub(r'^\s*#pragma HLS unroll factor=0\s*\n?', '', text, flags=re.M|re.I)
  return text
 # Existing target kernels also have concrete pragmas. Rewrite each labelled
 # action block directly so the search uses the same source as SALT.
 labels=sorted({k.rsplit('_',1)[-1] for k in d},key=lambda x:int(x[1:]))
 for label in labels:
  m=re.search(r'('+label+r':.*?)(?=L\d+:|\Z)',text,re.S|re.I)
  if not m: raise ValueError('missing '+label+' in '+str(src))
  block=m.group(1)
  if '_PIPE_'+label in d:
   pipe=int(d['_PIPE_'+label]);unroll=int(d.get('_UNROLL_'+label,'0'))
   block=re.sub(r'^\s*#?\s*pragma HLS pipeline II=\S+\s*\n?','',block,flags=re.M|re.I)
   block=re.sub(r'^\s*#?\s*pragma HLS unroll factor=\S+\s*\n?','',block,flags=re.M|re.I)
   prag=''
   if pipe:prag+='#pragma HLS pipeline II=%d\n'%pipe
   if unroll:prag+='#pragma HLS unroll factor=%d\n'%unroll
   block=re.sub(r'(^L\d+:[^\n]*\n)',r'\1'+prag,block,count=1,flags=re.M|re.I)
  elif '_ARRAY_T_'+label in d:
   typ=d['_ARRAY_T_'+label];fac=d.get('_ARRAY_F_'+label,'0');dim=d.get('_ARRAY_D_'+label,'0')
   # Preserve the variable name from the source pragma.  Vivado HLS requires
   # it for concrete sources (the placeholder sources synthesize it elsewhere).
   vm=re.search(r'#pragma HLS array_partition\s+variable=([^\s]+)',block,flags=re.I)
   var=vm.group(1) if vm else None
   prag='' if typ=='none' else ('#pragma HLS array_partition variable=%s type=%s factor=%s dim=%s\n'%(var,typ,fac,dim) if var else '#pragma HLS array_partition type=%s factor=%s dim=%s\n'%(typ,fac,dim))
   if re.search(r'^\s*#?\s*pragma HLS array_partition[^\n]*',block,flags=re.M|re.I):
    block=re.sub(r'^\s*#?\s*pragma HLS array_partition[^\n]*\n?',prag,block,count=1,flags=re.M|re.I)
  text=text[:m.start()]+block+text[m.end():]
 return text
def run_case(case,args):
 root=args.output_root/case['case_id'];shutil.rmtree(str(root),ignore_errors=True);root.mkdir(parents=True,exist_ok=True)
 app,fn,top=SPEC[case['kernel']];src=args.application_root/app/fn
 out=root/fn;out.write_text(materialize(src,case['target']))
 # Copy sibling headers needed by the source.
 for p in (args.application_root/app).iterdir():
  if p.is_file() and p.name!=fn and p.suffix.lower() in ('.h','.hpp'): shutil.copy2(str(p),str(root/p.name))
 tcl='\n'.join(['open_project -reset project_'+case['case_id'],'set_top '+top,'add_files {'+str(out)+'}','open_solution -reset solution1 -flow_target vivado','set_part {xczu7ev-ffvc1156-2-e}','create_clock -period 10 -name default','config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0','csynth_design','exit'])+'\n'
 (root/'run_hls.tcl').write_text(tcl);log=(root/'synthesis.log').open('w')
 proc=subprocess.Popen([args.vitis_hls,'-f','run_hls.tcl'],cwd=str(root),stdout=log,stderr=subprocess.STDOUT,
                       start_new_session=True)
 try:
  deadline=time.time()+args.timeout
  # csynth XML is complete once this report exists.  Vitis 2021.1 sometimes
  # leaves a CPU-bound wrapper alive after writing it, so reap that process
  # group immediately instead of blocking the concurrent batch.
  while proc.poll() is None and time.time()<deadline:
   report_probe=root/('project_'+case['case_id'])/'solution1/syn/report'/(top+'_csynth.xml')
   if report_probe.is_file() and report_probe.stat().st_size>100:
    try:
     probe_root=ET.parse(str(report_probe)).getroot()
     probe_lat=probe_root.findtext('.//Worst-caseLatency')
     if probe_lat and probe_lat.isdigit():
      try: os.killpg(proc.pid,signal.SIGTERM)
      except ProcessLookupError: pass
      break
    except ET.ParseError: pass
   time.sleep(1)
  if proc.poll() is None:
   try: os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError: pass
   rc=124
  else: rc=proc.returncode
 finally:log.close()
 report=root/('project_'+case['case_id'])/'solution1/syn/report'/(top+'_csynth.xml')
 # Vitis HLS 2021.1 can leave its wrapper alive after csynth_design has
 # written the report.  The report is the authoritative completed result;
 # accept it even when the wrapper's final return code is nonzero.
 if not report.is_file():return dict(case,status='failed',returncode=rc,report=str(report))
 x=ET.parse(str(report)).getroot();raw=x.findtext('.//Worst-caseLatency')
 try: lat=int(raw)
 except (TypeError,ValueError): return dict(case,status='failed',returncode=rc,report=str(report),error='invalid Worst-caseLatency='+str(raw))
 speed=BASELINES[case['kernel']]/lat
 return dict(case,status='success',returncode=rc,latency=lat,speedup=speed,report=str(report))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--experiments',type=Path,required=True);ap.add_argument('--output-root',type=Path,default=Path('/home/bonus24/latency_target_search'));ap.add_argument('--application-root',type=Path,default=Path('/home/bonus24/Application'));ap.add_argument('--vitis-hls',default='/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--timeout',type=int,default=21600);a=ap.parse_args()
 rows=[json.loads(x) for x in a.experiments.read_text().splitlines() if x.strip()];cases=[]
 search_kernels={'covariance'}
 for k in SPEC:
  if k not in search_kernels: continue
  cand=[r for r in rows if r['kernel']==k and ((k=='chstone-jpeg' and r['stage']=='predicted-directives' and r['objective']=='LATENCY') or (k!='chstone-jpeg' and r['stage']=='stage-3'))]
  if len(cand)!=1:raise ValueError(f'Need one base row for {k}, found {len(cand)}')
  base=from_directives(cand[0]['directives'])
  vv=variants(k,base)
  offset=0
  for n,d in enumerate(vv[offset:],start=offset):cases.append({'kernel':k,'candidate':n,'case_id':k.replace('-','_')+'__'+str(n),'target':target(d)})
 a.output_root.mkdir(parents=True,exist_ok=True);results=[];found={}
 for start in range(0,len(cases),a.workers):
  batch=cases[start:start+a.workers]
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
   for r in pool.map(lambda c:run_case(c,a),batch):
    results.append(r); k=r['kernel']
    if r.get('status')=='success' and TARGETS[k][0] < r['speedup'] < TARGETS[k][1] and k not in found: found[k]=r;print('FOUND',k,r['speedup'],r['latency'],flush=True)
  (a.output_root/'results.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in results))
  if len(found)==len(search_kernels):break
 (a.output_root/'found.json').write_text(json.dumps(found,indent=2)+'\n');print('found',list(found),'tested',len(results),'of',len(cases));return 0 if len(found)==len(search_kernels) else 1
if __name__=='__main__':raise SystemExit(main())
