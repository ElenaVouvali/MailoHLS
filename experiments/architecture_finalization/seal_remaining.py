"""Prepare fresh unattempted jobs with explicit expected Tcl; never overwrite receipts."""
import argparse,hashlib,json,shutil
from pathlib import Path
from .prepare_bank import ROOT,sha
INCLUDE='/opt/Xilinx/Vitis_HLS/2021.1/include'
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/architecture_finalization/bank');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 m=json.loads((a.bundle/'manifest.json').read_text());attempted={'bank_01','bank_04'};m['jobs']=[j for j in m['jobs'] if j['job_id'] not in attempted]
 a.out.mkdir(parents=True);shutil.copy2(a.bundle/'run_batch.py',a.out/'run_batch.py')
 for j in m['jobs']:
  d=a.out/'jobs'/j['job_id'];shutil.copytree(a.bundle/'jobs'/j['job_id'],d)
  tcl='\n'.join(['open_project -reset hls_project','set_top '+j['top'],'add_files -cflags {-I"'+INCLUDE+'"} {'+j['source']+'}','open_solution -reset solution1 -flow_target vivado','set_part {'+j['part']+'}','create_clock -period '+str(j['clock_ns'])+' -name default',*m['settings'],'csynth_design','exit',''])
  (d/'run_hls.tcl').write_text(tcl);j['expected_tcl_sha256']=sha(d/'run_hls.tcl')
 m['excluded_attempted_jobs']=sorted(attempted);m['qualification_gate']='Do not launch before anchor finite-latency protocol is resolved';(a.out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 (a.out/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(a.out))+'\n' for f in sorted(a.out.rglob('*')) if f.is_file()))
 print(json.dumps({'jobs':len(m['jobs']),'manifest_sha256':sha(a.out/'manifest.json'),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
