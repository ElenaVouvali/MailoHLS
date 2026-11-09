"""One sealed, resumable physical query on the authorized Coroni runner only."""
import fcntl,importlib.util,json,re,shlex,shutil,subprocess
from pathlib import Path
from experiments.refinement import configurations as c,synthesis as hls
from .deployment import ExactReceiptOracle
HOST='bonus24@coroni.microlab.ntua.gr'
RUNNER_SHA='ec05f6b97479a30ebc1c3b45eea7758881889cc5c87e64234ae85b610e7f1414'
class CoroniOracle:
 def __init__(self,request,template,headers,runner,root,remote_base,*,cache=None,max_new_calls=8,prior_calls=11,timeout=1200):
  if not re.fullmatch(r'[A-Za-z0-9_/-]+',remote_base) or remote_base.startswith('/'):raise ValueError('Use a safe relative run-specific Coroni directory')
  if c.sha256(runner)!=RUNNER_SHA:raise ValueError('Unrecognized sealed synthesis runner')
  if c.sha256(template)!=request['source_sha256']:raise ValueError('Source template changed')
  if {Path(p).name:c.sha256(p) for p in headers}!=request['headers_sha256']:raise ValueError('Headers changed')
  if not isinstance(request['tool_identity'],dict):raise ValueError('Coroni requires a pinned executable/version/vendor-header identity')
  self.request=request;self.template=Path(template);self.headers=list(map(Path,headers));self.runner=Path(runner);self.root=Path(root);self.remote=remote_base;self.cache=ExactReceiptOracle(cache) if cache else None;self.max_calls=max_new_calls;self.prior=prior_calls;self.timeout=timeout
  self.root.mkdir(parents=True,exist_ok=True)
 def __call__(self,identity):
  if self.cache:
   hit=self.cache(identity)
   if hit:return hit
  key=c.object_hash(identity);base=self.root/key;bundle=base/'bundle';out=base/'results';resultfile=base/'oracle.json'
  if resultfile.exists():
   entry=json.loads(resultfile.read_text())
   for p,digest in entry['report_sha256'].items():
    if c.sha256(p)!=digest:raise ValueError('Cached physical report changed')
   return {**entry,'new_physical_call':False}
  if not bundle.exists():
   bundle.mkdir(parents=True);jobdir=bundle/'jobs'/'query';jobdir.mkdir(parents=True);source=jobdir/self.template.name
   target='\n'.join(f'{k} = {v}' for k,v in identity['assignment'].items());source.write_text(hls.materialize(self.template.read_text(),target))
   for path in self.headers:shutil.copyfile(path,jobdir/path.name)
   top=self.request['top']
   if not re.fullmatch(r'[A-Za-z_]\w*',top):raise ValueError('Invalid HLS top')
   job={'job_id':'query','kernel':self.request['kernel'],'arm':'stage3_refinement','request_number':1,'source':source.name,'top':top,'part':identity['part'],'clock_ns':identity['clock_ns'],'capacities':self.request['capacities'],'budget':self.request['budgets'],'materialized_source_sha256':c.sha256(source),'original_source_sha256':self.request['source_sha256'],'metadata_sha256':self.request['action_metadata_sha256'],'headers':self.request['headers_sha256'],'assignment':identity['assignment']}
   tcl='\n'.join(['open_project -reset hls_project','set_top '+top,'add_files -cflags {-I"/opt/Xilinx/Vitis_HLS/2021.1/include"} {'+source.name+'}','open_solution -reset solution1 -flow_target vivado','set_part {'+identity['part']+'}','create_clock -period '+str(identity['clock_ns'])+' -name default',*identity['flow_settings'],'csynth_design','exit','']);(jobdir/'run_hls.tcl').write_text(tcl);job['expected_tcl_sha256']=c.sha256(jobdir/'run_hls.tcl')
   shutil.copyfile(self.runner,bundle/'run_batch.py');c.write_json(bundle/'manifest.json',{'schema':'salt-v3-user-refinement-v1','jobs':[job],'settings':identity['flow_settings'],'objective':'PARETO_LATENCY','runner_sha256':RUNNER_SHA,'physical_identity':identity})
   (bundle/'SHA256SUMS').write_text(''.join(c.sha256(p)+'  '+str(p.relative_to(bundle))+'\n' for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
  # Physical reservations survive cancellation; the same key resumes its sealed job.
  with (self.root/'ledger.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);path=self.root/'physical_ledger.jsonl';ledger=c.read_jsonl(path) if path.exists() else []
   if key not in {e['key'] for e in ledger}:
    if len(ledger)>=self.max_calls or len(ledger)+self.prior>=48:raise RuntimeError('Physical HLS budget exhausted')
    ledger.append({'key':key,'status':'reserved','bundle_sha256':c.sha256(bundle/'manifest.json')});c.write_jsonl(path,ledger)
  remote=self.remote+'/'+key;quote=shlex.quote
  subprocess.run(['ssh',HOST,'mkdir -p '+quote(remote+'/bundle')],check=True)
  subprocess.run(['rsync','-a',str(bundle)+'/',HOST+':'+remote+'/bundle/'],check=True)
  command='cd '+quote(remote+'/bundle')+' && sha256sum -c SHA256SUMS && python3 run_batch.py --bundle . --output ../results --tool /opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls --timeout '+str(int(self.timeout))
  subprocess.run(['ssh',HOST,command],check=True)
  out.mkdir(exist_ok=True);subprocess.run(['rsync','-a',HOST+':'+remote+'/results/',str(out)+'/'],check=True)
  provenance=json.loads((out/'batch_provenance.json').read_text());actual_tool={k:provenance[k] for k in ('executable_sha256','version_output','vendor_headers')}
  if actual_tool!=self.request['tool_identity']:raise ValueError('Coroni tool build/header identity differs from request')
  result=json.loads((out/'query/result.json').read_text());report=out/'query/hls_project/solution1/syn/report'/(self.request['top']+'_csynth.xml')
  if result.get('report_sha256') and c.sha256(report)!=result['report_sha256']:raise ValueError('Returned report hash mismatch')
  witnesses={str(p.resolve()):c.sha256(p) for p in (out/'query/result.json',out/'batch_provenance.json',out/'query/synthesis.log')}
  if report.exists():witnesses[str(report.resolve())]=c.sha256(report)
  metrics=result.get('metrics') or {};measurement={'synthesis_status':'success' if result['status']=='success' else result['status'],'latency_cycles':metrics.get('latency_cycles'),'resources':metrics.get('resources'),'estimated_clock_ns':metrics.get('estimated_clock_ns')}
  entry={'physical_identity':identity,'measurement':measurement,'report_sha256':witnesses,'new_physical_call':True,'coroni':remote,'result':result}
  c.write_json(resultfile,entry);return entry
