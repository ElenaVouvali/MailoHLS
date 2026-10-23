#!/usr/bin/env python3
"""Run the 80 train/validation U50 calibration syntheses on Coroni.

Run this file on Coroni.  The input JSONL has one record per design point:

  {"kernel_name":"machsuite-gemm-blocked", "origin":"zcu_pareto_area",
   "target":"auto{_PIPE_L1} = 1\nauto{_UNROLL_L1} = 0", ...}

The target is the canonical MailoHLS completion.  The source directory is
found under --application-root and must contain exactly one C/C++ file with
MailoHLS placeholders.  Results are resumable and written as JSONL with
absolute post-synthesis resources and worst-case latency cycles.
"""
import argparse, datetime, json, os, re, shutil, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET

PART = "xcu50-fsvh2104-2-e"
KERNELS = {
 "machsuite-gemm-blocked","machsuite-md-knn","machsuite-spmv-ellpack",
 "machsuite-stencil2d","machsuite-stencil3d","rodinia-backprop-0-baseline-forward",
 "rodinia_cfd_flux_0_baseline_0","rodinia_cfd_step_factor_4_doublebuffer_0",
 "rodinia_dilate_2_pipeline_0","rodinia-hotspot-2-pipeline","rodinia-kmeans-1-tiling",
 "rodinia-knn-4-doublebuffer","rodinia_lavaMD_2_pipeline_0","rodinia_lc_gicov_0_baseline_0",
 "rodinia_lud_1_tiling_0","rodinia_streamcluster_3_doublebuffer_0",
 "machsuite-sort-radix","machsuite-viterbi","rodinia_pathfinder_0_baseline_0",
 "rodinia_pathfinder_4_doublebuffer_0",
}
ASSIGN = re.compile(r"^auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+)\}\s*=\s*(\S+)\s*$", re.I)
PH = re.compile(r"auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+)\}", re.I)

def parse_target(text):
    out={}
    for line in text.splitlines():
        if not line.strip(): continue
        m=ASSIGN.match(line.strip())
        if not m: raise ValueError(f"Malformed target line: {line!r}")
        key,val=m.group(1).upper(),m.group(2).lower()
        if key in out and out[key]!=val: raise ValueError(f"Conflicting assignment: {key}")
        out[key]=val
    return out

def materialize(source, target):
    assignments=parse_target(target); text=source.read_text()
    placeholders={x.upper() for x in PH.findall(text)}
    if not placeholders:
        # Some Coroni application snapshots contain concrete or commented
        # pragmas instead of auto{...} placeholders. Materialize by L-label.
        labels=sorted({key.rsplit('_',1)[-1] for key in assignments}, key=lambda x:int(x[1:]))
        for label in labels:
            block_re=rf"({label}:.*?)(?=L\d+:|\Z)"
            match=re.search(block_re,text,re.S|re.I)
            if not match: raise ValueError(f"{source}: missing block {label}")
            block=match.group(1)
            if f'_PIPE_{label}' in assignments:
                pipe=int(assignments[f'_PIPE_{label}']); unroll=int(assignments.get(f'_UNROLL_{label}','0'))
                block=re.sub(r'^\s*#?\s*pragma HLS pipeline II=\S+\s*\n?', '', block, flags=re.M|re.I)
                block=re.sub(r'^\s*#?\s*pragma HLS unroll factor=\S+\s*\n?', '', block, flags=re.M|re.I)
                directives=''
                if pipe: directives += f'#pragma HLS pipeline II={pipe}\n'
                if unroll: directives += f'#pragma HLS unroll factor={unroll}\n'
                block= re.sub(r'(^L\d+:[^\n]*\n)', r'\1'+directives, block, count=1, flags=re.M|re.I)
            elif f'_ARRAY_T_{label}' in assignments:
                typ=assignments[f'_ARRAY_T_{label}']; fac=assignments.get(f'_ARRAY_F_{label}','0'); dim=assignments.get(f'_ARRAY_D_{label}','0')
                replacement='' if typ=='none' else f'#pragma HLS array_partition type={typ} factor={fac} dim={dim}\n'
                if re.search(r'^\s*#?\s*pragma HLS array_partition[^\n]*', block, flags=re.M|re.I):
                    block=re.sub(r'^\s*#?\s*pragma HLS array_partition[^\n]*\n?', replacement, block, count=1, flags=re.M|re.I)
                elif replacement:
                    block=re.sub(r'(^L\d+:[^\n]*\n)', r'\1'+replacement, block, count=1, flags=re.M|re.I)
            text=text[:match.start()]+block+text[match.end():]
        return text
    missing=placeholders-set(assignments)
    extra=set(assignments)-placeholders
    if missing or extra: raise ValueError(f"{source}: placeholder mismatch missing={sorted(missing)} extra={sorted(extra)}")
    for key,val in assignments.items():
        token=re.escape("auto{"+key+"}")
        if key.startswith("_PIPE_"):
            if not val.isdigit(): raise ValueError(f"{key}: pipeline must be integer")
            text=re.sub(rf"(pipeline II=){token}",r"\g<1>" + str(int(val)),text,flags=re.I)
        elif key.startswith("_UNROLL_"):
            if not val.isdigit(): raise ValueError(f"{key}: unroll must be integer")
            text=re.sub(rf"(unroll factor=){token}",r"\g<1>" + str(int(val)),text,flags=re.I)
        elif key.endswith(tuple(f"_L{i}" for i in range(1,65))):
            # Array sites carry independent type/factor/dimension assignments.
            label=key.rsplit("_",1)[-1]
            def get(kind, default): return assignments.get(f"_ARRAY_{kind}_{label}",default)
            typ=get("T","none"); fac=get("F","0"); dim=get("D","0")
            if typ not in {"none","block","cyclic","complete"} or not fac.isdigit() or not dim.isdigit():
                raise ValueError(f"Invalid array tuple at {label}: {typ},{fac},{dim}")
            if key.startswith("_ARRAY_T_"):
                for kind, value in (("T",typ),("F",str(int(fac))),("D",str(int(dim)))):
                    token = re.escape("auto{_ARRAY_" + kind + "_" + label + "}")
                    text = re.sub(token, value, text, flags=re.I)
    if PH.search(text): raise ValueError(f"Unresolved placeholders remain in {source}")
    return text

def find_source(app_root,kernel):
    directory=app_root/kernel
    if not directory.is_dir(): raise FileNotFoundError(directory)
    candidates=[]
    for p in directory.iterdir():
        if p.suffix.lower() in {".c",".cc",".cpp",".cxx"}:
            candidates.append(p)
    placeholder_candidates=[p for p in candidates if PH.search(p.read_text(errors="ignore"))]
    if placeholder_candidates: candidates=placeholder_candidates
    if len(candidates)!=1: raise ValueError(f"{kernel}: expected one source, found {candidates}")
    info=directory/'kernel_info.txt'
    if not info.is_file(): raise FileNotFoundError(info)
    top=info.read_text().splitlines()[0].strip()
    if not top: raise ValueError(f"{kernel}: empty kernel_info.txt")
    return directory,candidates[0],top

def copytree_merge(src, dst):
    """Python 3.7-compatible equivalent of copytree(dirs_exist_ok=True)."""
    if not dst.exists(): dst.mkdir(parents=True)
    for item in src.iterdir():
        target=dst/item.name
        if item.is_dir(): copytree_merge(item,target)
        else: shutil.copy2(str(item),str(target))

def parse_report(path, expected_clock):
    root=ET.parse(path).getroot()
    if root.findtext('.//Part') != PART: raise ValueError(f"wrong part in {path}")
    clock=float(root.findtext('.//TargetClockPeriod'))
    if abs(clock-expected_clock)>0.01: raise ValueError(f"wrong clock in {path}: {clock}")
    def integer(xpath): return int(root.findtext(xpath))
    return {'latency_cycles':integer('.//Worst-caseLatency'), 'resources':{
      'BRAM_18K':integer('.//Resources/BRAM_18K'),'DSP':integer('.//Resources/DSP'),
      'FF':integer('.//Resources/FF'),'LUT':integer('.//Resources/LUT')}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--application-root',type=Path,default=Path('/home/bonus24/Application'))
    ap.add_argument('--output-root',type=Path,default=Path('/home/bonus24/mailohls_u50_adaptation'))
    ap.add_argument('--vitis-hls',default='/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls'); ap.add_argument('--timeout-sec',type=int,default=21600)
    args=ap.parse_args(); rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    if len(rows)!=80: raise ValueError(f'Expected 80 train/validation designs, got {len(rows)}')
    if {r.get('device','xcu50-fsvh2104-2-e').lower() for r in rows} != {PART}: raise ValueError('Manifest must contain only U50 rows')
    if any(float(r.get('clock_period_ns',10))!=10.0 for r in rows): raise ValueError('Manifest must contain only 10 ns rows')
    if {r['kernel_name'] for r in rows}-KERNELS: raise ValueError('Manifest contains an unapproved kernel')
    if len({(r['kernel_name'],r['origin']) for r in rows})!=80 or any(sum(x['kernel_name']==k for x in rows)!=4 for k in KERNELS if k in {x['kernel_name'] for x in rows}):
        raise ValueError('Expected four unique origins per kernel')
    args.output_root.mkdir(parents=True,exist_ok=True); results_path=args.output_root/'synthesis_results.jsonl'; done={}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                r=json.loads(line); done[r['case_id']]=r
    for i,row in enumerate(rows,1):
        case_id=f"{row['kernel_name']}__{row['origin']}"
        if case_id in done and done[case_id].get('status')=='success': continue
        app_dir,source,top=find_source(args.application_root,row['kernel_name']); case_dir=args.output_root/case_id; case_dir.mkdir(parents=True,exist_ok=True)
        copytree_merge(app_dir,case_dir/'source')
        materialized=case_dir/'source'/source.name; materialized.write_text(materialize(source,row['target']))
        tcl='\n'.join([f'open_project -reset project_{row["kernel_name"]}',f'set_top {top}',f'add_files {{{materialized}}}', 'open_solution -reset solution1 -flow_target vivado',f'set_part {{{PART}}}','create_clock -period 10 -name default','config_array_partition -complete_threshold 0 -throughput_driven off','config_compile -pipeline_loops 0','csynth_design','exit'])+'\n'
        (case_dir/'run_hls.tcl').write_text(tcl); log=case_dir/'vitis_hls.log'; started=datetime.datetime.utcnow().isoformat()+'Z'
        try:
            with log.open('w') as stream: rc=subprocess.run([args.vitis_hls,'-f','run_hls.tcl'],cwd=case_dir,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout_sec).returncode
            report=case_dir/f'project_{row["kernel_name"]}/solution1/syn/report/{top}_csynth.xml'; metrics=parse_report(report,10.0) if rc==0 and report.is_file() else {}
            result={**row,'case_id':case_id,'status':'success' if rc==0 and metrics else 'failed','returncode':rc,'report':str(report),'log':str(log),'started_utc':started,**metrics}
        except subprocess.TimeoutExpired: result={**row,'case_id':case_id,'status':'timeout','returncode':124,'log':str(log),'started_utc':started}
        done[case_id]=result; results_path.write_text(''.join(json.dumps(done[k],separators=(',',':'))+'\n' for k in sorted(done))); print(i,case_id,result['status'],flush=True)
    failed=[r for r in done.values() if r.get('status')!='success']; print(f"completed={len(done)} failed={len(failed)} results={results_path}"); return 1 if failed else 0
if __name__=='__main__': raise SystemExit(main())
