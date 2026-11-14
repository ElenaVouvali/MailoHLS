#!/usr/bin/env python3
"""SALT nine-kernel JSONL -> sealed Coroni Vitis HLS 2021.1 batches.

Python 3.6+ and stdlib only. Prepare is safe offline; run requires Vitis HLS.
Original source/metadata are authoritative, NEVER compact inference prompt views.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

RESOURCES = ('BRAM_18K', 'DSP', 'FF', 'LUT')
CAPS = {
    'xczu7ev-ffvc1156-2-e': dict(zip(RESOURCES, (624, 1728, 460800, 230400))),
    'xcu200-fsgd2104-2-e': dict(zip(RESOURCES, (4320, 6840, 2364480, 1182240))),
}
PART_ALIAS = {'zcu104':'xczu7ev-ffvc1156-2-e', 'u200':'xcu200-fsgd2104-2-e'}
KERNELS = ('GAN','Kalman','chstone-aes','chstone-jpeg','covariance','jacobi-2d',
           'rosetta-3d-rendering','syr2k','trmm-opt')
SETTINGS = ('config_array_partition -complete_threshold 0 -throughput_driven off',
            'config_compile -pipeline_loops 0')
VENDOR = ('ap_fixed.h','ap_int.h')
PH = re.compile(r'auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*)\}', re.I)
LABEL = re.compile(r'^\s*(?:/\*\s*(L[1-9][0-9]*)\s*:\s*\*/|(L[1-9][0-9]*)\s*:)', re.I)
SAFE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
SOURCE_SUFFIX = ('.c','.cpp','.cc','.cxx')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path, obj):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def read_jsonl(path):
    rows=[]
    for number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except ValueError as e:raise ValueError('%s line %d: %s'%(path,number,e))
    if not rows:raise ValueError('No JSONL records: %s'%path)
    return rows


def read_metadata(path):
    lines=[x.strip() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(lines)<2 or not SAFE.fullmatch(lines[0]):raise ValueError('Malformed kernel_info.txt: '+str(path))
    top=lines[0];actions={}
    for line in lines[1:]:
        cells=[x.strip() for x in line.split(',')]
        if len(cells)<3 or not re.fullmatch(r'L[1-9][0-9]*',cells[0],re.I):
            raise ValueError('Malformed action: '+line)
        action=cells[0].upper()
        if action in actions:raise ValueError('Duplicate action '+action)
        kind=cells[1].lower()
        if kind=='loop':
            if len(cells)!=3 or int(cells[2])<1:raise ValueError('Invalid loop '+line)
            actions[action]=dict(kind=kind,trip_count=int(cells[2]))
        elif kind=='array':
            if len(cells)<5 or (len(cells)-3)%2:raise ValueError('Invalid array '+line)
            dims=dict((int(cells[i]),int(cells[i+1])) for i in range(3,len(cells),2))
            if len(dims)!=(len(cells)-3)//2 or any(k<1 or v<1 for k,v in dims.items()):raise ValueError('Invalid array shape '+line)
            actions[action]=dict(kind=kind,variable=cells[2],dims=dims)
        else:raise ValueError('Unsupported action kind '+kind)
    return top,actions


def choose_source(folder):
    for name in ('source.cpp','source.c','source.cc','source.cxx'):
        p=folder/name
        if p.is_file():return p
    candidates=[p for p in folder.iterdir() if p.is_file() and p.suffix in SOURCE_SUFFIX]
    if len(candidates)==1:return candidates[0]
    raise ValueError('%s: require one canonical source.c/source.cpp (got %s)'%(folder,[p.name for p in candidates]))


def source_template(src,actions):
    """Match GNN_branch.insert_placeholders source-line logic, without dependencies."""
    code=src.read_text(encoding='utf-8')
    if PH.search(code):
        templ=code
    else:
        parts=[]
        for line in code.splitlines(True):
            match=LABEL.match(line.rstrip('\n'))
            parts.append(line)
            if not match:continue
            action=(match.group(1) or match.group(2)).upper()
            if action not in actions:continue
            after=line.rstrip('\n')[match.end():]
            if re.search(r'\bfor\s*\(',after):
                parts.extend(('#pragma HLS pipeline II=auto{_PIPE_%s}\n'%action,
                              '#pragma HLS unroll factor=auto{_UNROLL_%s}\n'%action))
            else:
                array=re.search(r'^(?:\s*[A-Za-z_]\w*\s*:\s*)?\s*([A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)*)\s+([A-Za-z_]\w*)\s*(\[[^\]]+\])+\s*(?:=\s*[^;]+)?\s*;',after)
                if array:
                    parts.append('#pragma HLS array_partition variable=%s type=auto{_ARRAY_T_%s} factor=auto{_ARRAY_F_%s} dim=auto{_ARRAY_D_%s}\n'%(array.group(2),action,action,action))
        templ=''.join(parts)
    observed=['AUTO{%s}'%m.group(1).upper() for m in PH.finditer(templ)]
    expected=[]
    for action,info in actions.items():
        tags=('PIPE','UNROLL') if info['kind']=='loop' else ('ARRAY_T','ARRAY_F','ARRAY_D')
        expected.extend('AUTO{_%s_%s}'%(x,action) for x in tags)
    if set(observed)!=set(expected) or len(observed)!=len(set(observed)):
        raise ValueError('Source/action placeholder mismatch: missing=%s extra=%s duplicate=%s'%(sorted(set(expected)-set(observed)),sorted(set(observed)-set(expected)),len(observed)!=len(set(observed))))
    for line in templ.splitlines():
        if re.match(r'\s*#\s*pragma\s+HLS\s+(pipeline|unroll|array_partition)\b',line,re.I) and not PH.search(line):
            raise ValueError('Concrete optimization pragma in canonical source: '+line[:120])
    return templ


def intval(value,name):
    if isinstance(value,bool) or not re.fullmatch(r'[0-9]+',str(value)):
        raise ValueError('Invalid nonnegative integer %s=%r'%(name,value))
    return int(value)


def get_assignments(row, actions):
    supplied=row.get('directives',row.get('predicted_directives'))
    if supplied is None and isinstance(row.get('prediction'),dict):
        supplied=row['prediction'].get('directives')
    if supplied is None:raise ValueError('Missing directives/predicted_directives')
    if isinstance(supplied,dict):
        if all(str(k).upper().startswith('AUTO{') for k in supplied):
            return normalize_field_dict(supplied,actions)
        supplied=[dict(v,action_id=k) for k,v in supplied.items()]
    if not isinstance(supplied,list):raise ValueError('directives must be list or dictionary')
    by_id={}
    for directive in supplied:
        if not isinstance(directive,dict):raise ValueError('Directive must be dictionary')
        name=str(directive.get('action_id','')).upper()
        if name in by_id or name not in actions:raise ValueError('Duplicate or unexpected action '+name)
        if directive.get('kind')!=actions[name]['kind']:
            raise ValueError('%s: predicted kind differs from metadata'%name)
        by_id[name]=directive
    if set(by_id)!=set(actions):raise ValueError('Incomplete action set: missing=%s extra=%s'%(set(actions)-set(by_id),set(by_id)-set(actions)))
    fields={}
    for action,info in actions.items():
        d=by_id[action]
        if info['kind']=='loop':
            pipe=intval(d.get('pipeline_ii'),action+'.pipeline_ii')
            factor=intval(d.get('unroll_factor'),action+'.unroll_factor')
            # II=0 disables the pipeline; positive II values are legal source choices.
            fields['AUTO{_PIPE_%s}'%action]=str(pipe)
            fields['AUTO{_UNROLL_%s}'%action]=str(factor)
        else:
            if d.get('variable') and d['variable']!=info['variable']:
                raise ValueError('%s: variable differs from source metadata'%action)
            typ=str(d.get('partition_type','')).lower()
            if typ not in ('none','block','cyclic','complete'):raise ValueError('%s: bad partition type'%action)
            factor=intval(d.get('partition_factor'),action+'.partition_factor')
            dim=intval(d.get('partition_dim'),action+'.partition_dim')
            if typ=='none':
                if factor!=0:raise ValueError('%s: none must use factor=0'%action)
            elif typ=='complete':
                if factor not in (0,1):raise ValueError('%s: complete factor sentinel should be 0/1'%action)
                if dim not in info['dims']:raise ValueError('%s: invalid partition dimension'%action)
            else:
                if factor<1 or dim not in info['dims']:raise ValueError('%s: invalid partition factor/dim'%action)
            fields.update({'AUTO{_ARRAY_T_%s}'%action:typ,'AUTO{_ARRAY_F_%s}'%action:str(factor),'AUTO{_ARRAY_D_%s}'%action:str(dim)})
    return fields


def normalize_field_dict(vals,actions):
    fields={str(k).upper():str(v).lower() for k,v in vals.items()}
    expected=set()
    for a,info in actions.items():
        for tag in (('PIPE','UNROLL') if info['kind']=='loop' else ('ARRAY_T','ARRAY_F','ARRAY_D')):
            expected.add('AUTO{_%s_%s}'%(tag,a))
    if set(fields)!=expected:raise ValueError('Field assignment mismatch')
    # Route dictionary back through strict per-action validation.
    lst=[]
    for a,info in actions.items():
        if info['kind']=='loop':
            lst.append(dict(action_id=a,kind='loop',pipeline_ii=fields['AUTO{_PIPE_%s}'%a],unroll_factor=fields['AUTO{_UNROLL_%s}'%a]))
        else:
            lst.append(dict(action_id=a,kind='array',variable=info['variable'],partition_type=fields['AUTO{_ARRAY_T_%s}'%a],partition_factor=fields['AUTO{_ARRAY_F_%s}'%a],partition_dim=fields['AUTO{_ARRAY_D_%s}'%a]))
    return get_assignments({'directives':lst},actions)


def materialize(template,assignments):
    actual=['AUTO{%s}'%m.group(1).upper() for m in PH.finditer(template)]
    if set(actual)!=set(assignments) or len(actual)!=len(set(actual)):
        raise ValueError('Template/assignment fields differ')
    text=PH.sub(lambda m: assignments['AUTO{%s}'%m.group(1).upper()],template)
    out=[]
    for line in text.splitlines(True):
        if re.match(r'\s*#\s*pragma\s+HLS\s+(?:pipeline\s+II=0|unroll\s+factor=0)\s*$',line,re.I):continue
        if re.match(r'\s*#\s*pragma\s+HLS\s+array_partition\b',line,re.I):
            if re.search(r'\btype=none\b',line,re.I):continue
            if re.search(r'\btype=complete\b',line,re.I):line=re.sub(r'\s+factor=\d+\b','',line,flags=re.I)
        out.append(line)
    result=''.join(out)
    if PH.search(result):raise ValueError('Unresolved placeholders')
    return result


def quote_tcl(value):
    s=str(value)
    if any(c in s for c in '{}\r\n\\') or not s:raise ValueError('Unsafe TCL value '+repr(s))
    return '{'+s+'}'


def make_tcl(source_name,top,part,clock,cflags):
    if not SAFE.fullmatch(top) or not re.fullmatch(r'[\w.\-]+',part):raise ValueError('Invalid TCL top/part')
    if Path(source_name).name!=source_name:raise ValueError('Unsafe source name')
    if not 0<float(clock)<1000:raise ValueError('Invalid clock')
    return '\n'.join(['open_project -reset hls_project','set_top '+top,
        'add_files -cflags '+quote_tcl(cflags)+' '+quote_tcl(source_name),
        'open_solution -reset solution1 -flow_target vivado','set_part '+quote_tcl(part),
        'create_clock -period %s -name default'%format(clock,'.12g'),*SETTINGS,'csynth_design','exit',''])


def load_targets(a):
    if a.targets_json:
        targets=json.loads(Path(a.targets_json).read_text(encoding='utf-8'))
        if not isinstance(targets,list) or not targets:raise ValueError('--targets-json must contain nonempty array')
    else:
        targets=[dict(device=a.device,clock_ns=a.clock_ns,budget_percent=a.budget_percent)]
    result=[]
    for t in targets:
        if not isinstance(t,dict):raise ValueError('Invalid target entry')
        requested=t.get('part',t.get('device'))
        part=PART_ALIAS.get(str(requested).lower(),str(requested))
        capacities=t.get('capacities',CAPS.get(part))
        if capacities is None:raise ValueError('Unknown part %s: supply explicit capacities in --targets-json'%part)
        if set(capacities)!=set(RESOURCES) or any(int(capacities[k])!=capacities[k] or int(capacities[k])<=0 for k in RESOURCES):
            raise ValueError('Invalid four-resource capacity map for '+part)
        capacities={k:int(capacities[k]) for k in RESOURCES}
        has_period=t.get('clock_ns') is not None
        has_freq=t.get('frequency_mhz') is not None
        if has_period==has_freq:raise ValueError('Specify exactly one of clock_ns/frequency_mhz')
        clock=float(t['clock_ns']) if has_period else 1000./float(t['frequency_mhz'])
        if not math.isfinite(clock) or clock<=0:raise ValueError('Invalid target clock')
        pct=float(t.get('budget_percent',100))
        if not math.isfinite(pct) or pct<=0 or pct>100:raise ValueError('Invalid budget percentage')
        budget={k:int(math.floor(capacities[k]*pct/100.)) for k in RESOURCES}
        result.append(dict(part=part,clock_ns=clock,budget_percent=pct,budget=budget,capacities=capacities))
    return result


def prepare(a):
    root=Path(a.out).resolve()
    if root.exists():raise FileExistsError('Fresh bundle required: '+str(root))
    predictions=Path(a.predictions).resolve()
    sources=Path(a.sources_root).resolve()
    rows=read_jsonl(predictions)
    targets=load_targets(a)
    seen=set();prepared=[]
    for lineno,row in enumerate(rows,1):
        if not isinstance(row,dict):raise ValueError('JSONL row must be object')
        kernel=row.get('kernel',row.get('kernel_name'))
        if kernel not in KERNELS:raise ValueError('Unsupported kernel %r at line %d'%(kernel,lineno))
        folder=sources/kernel
        src=choose_source(folder)
        meta=folder/'kernel_info.txt'
        top,actions=read_metadata(meta)
        labels=[]
        for line in src.read_text(encoding='utf-8').splitlines():
            m=LABEL.match(line)
            if m:labels.append((m.group(1) or m.group(2)).upper())
        if set(labels)!=set(actions) or len(labels)!=len(actions):
            raise ValueError('%s: canonical source Lk markers mismatch metadata'%kernel)
        assignment=get_assignments(row,actions)
        template=source_template(src,actions)
        synthesized=materialize(template,assignment)
        if kernel=='GAN' and 'AP_FIXED_STUB_H' in synthesized:
            raise ValueError('GAN stub header is embedded, not excluded by filename; inspect full source')
        if '\x00' in synthesized:raise ValueError('NUL in source')
        prepared.append((lineno,row,kernel,folder,src,meta,top,actions,assignment,synthesized))
        seen.add(kernel)
    if a.require_nine and seen!=set(KERNELS):raise ValueError('Expected all nine kernels; present=%s'%sorted(seen))
    # Make no output until every input is vetted.
    (root/'jobs').mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve(),root/'coroni_batch.py')
    jobs=[];contexts=[]
    for lineno,row,kernel,folder,src,meta,top,actions,assign,text in prepared:
        for target in targets:
            signature=hashlib.sha256(json.dumps([kernel,assign,target['part'],target['clock_ns']],sort_keys=True).encode()).hexdigest()[:12]
            name='%03d_%s_%s'%(len(jobs)+1,kernel.replace('-','_'),signature)
            d=root/'jobs'/name;d.mkdir()
            out_name='kernel'+src.suffix
            (d/out_name).write_text(text,encoding='utf-8')
            headers={}
            for h in sorted(folder.iterdir()):
                if h.is_file() and h.suffix.lower() in ('.h','.hpp') and h.name not in VENDOR:
                    shutil.copyfile(h,d/h.name);headers[h.name]=sha(d/h.name)
            cflags='-I/opt/Xilinx/Vitis_HLS/2021.1/include -I.'
            (d/'run_hls.tcl').write_text(make_tcl(out_name,top,target['part'],target['clock_ns'],cflags),encoding='utf-8')
            rec=dict(job_id=name,request_index=lineno,kernel=kernel,top=top,source=out_name,
                part=target['part'],clock_ns=target['clock_ns'],budget_percent=target['budget_percent'],
                budget=target['budget'],capacities=target['capacities'],
                original_source_sha256=sha(src),metadata_sha256=sha(meta),
                rendered_source_sha256=sha(d/out_name),tcl_sha256=sha(d/'run_hls.tcl'),headers=headers,
                action_ids=list(actions),assignment=assign,checkpoint_step=row.get('checkpoint_step'),
                checkpoint_sha256=row.get('checkpoint_sha256'),
                source_view_note='Canonical unabridged synthesis source; independent of LLM compact prompt')
            jobs.append(rec)
            contexts.append((kernel,target['part'],target['clock_ns']))
    manifest=dict(schema='salt-coroni-eval9-batch-v1',created_utc=datetime.datetime.utcnow().isoformat()+'Z',
        predictions_sha256=sha(predictions),original_sources=str(sources),
        target_policy='explicit cross-product: one synthesis per JSONL record and target',
        expected_tool_version='2021.1',settings=list(SETTINGS),runner_sha256=sha(root/'coroni_batch.py'),
        jobs=jobs,tool='/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls',
        design_validation='source/action completeness only; full synthesis/cosimulation not yet performed')
    dump(root/'manifest.json',manifest)
    digest=''.join('%s  %s\n'%(sha(p),p.relative_to(root)) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS')
    (root/'SHA256SUMS').write_text(digest,encoding='utf-8')
    print('PREPARED %d jobs from %d JSONL rows x %d targets -> %s'%(len(jobs),len(prepared),len(targets),root))
    print('No synthesis was launched.')


def verify_bundle(root):
    root=Path(root).resolve()
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if m.get('schema')!='salt-coroni-eval9-batch-v1':raise ValueError('Wrong bundle schema')
    if sha(root/'coroni_batch.py')!=m['runner_sha256']:raise ValueError('Runner SHA mismatch')
    for j in m['jobs']:
        d=root/'jobs'/j['job_id']
        if sha(d/j['source'])!=j['rendered_source_sha256'] or sha(d/'run_hls.tcl')!=j['tcl_sha256']:
            raise ValueError('Synthesis source/TCL changed for '+j['job_id'])
        for name,h in j['headers'].items():
            if sha(d/name)!=h:raise ValueError('Header changed for '+j['job_id']+' '+name)
    return m


def run_one(root,j,tool,timeout,output):
    d=root/'jobs'/j['job_id']
    r=output/j['job_id']
    if r.exists():
        print('[SKIP] existing job directory %s (never overwrite)'%r,flush=True)
        return
    r.mkdir(parents=True)
    # Self-contained source/TCL, no shared project paths.
    for name in [j['source'],'run_hls.tcl']+list(j['headers']):shutil.copyfile(d/name,r/name)
    started=time.time();status='unknown';code=None
    log=r/'synthesis.log'
    with log.open('w',encoding='utf-8') as out:
        proc=subprocess.Popen([str(tool),'-f','run_hls.tcl'],cwd=str(r),stdout=out,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
        offset=0; timed_out=False
        while proc.poll() is None:
            time.sleep(1)
            with log.open('r',encoding='utf-8',errors='replace') as inp:
                inp.seek(offset);more=inp.read();offset=inp.tell()
                if more:sys.stdout.write('[%s] %s'%(j['job_id'],more));sys.stdout.flush()
            if time.time()-started>timeout:
                timed_out=True
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                break
        code=proc.wait()
        with log.open('r',encoding='utf-8',errors='replace') as inp:
            inp.seek(offset);more=inp.read()
            if more:sys.stdout.write('[%s] %s'%(j['job_id'],more));sys.stdout.flush()
        status='timed_out' if timed_out else ('finished_command' if code==0 else 'process_failed')
    report=r/'hls_project/solution1/syn/report'/(j['top']+'_csynth.xml')
    result=dict(status=status,returncode=code,elapsed_seconds=time.time()-started,
                synthesis_log_sha256=sha(log),report_sha256=sha(report) if report.is_file() else None)
    dump(r/'run_receipt.json',result)
    print('[JOB] %s %s, elapsed=%.1fs'%(j['job_id'],status,result['elapsed_seconds']),flush=True)


def run(a):
    root=Path(a.bundle).resolve();manifest=verify_bundle(root)
    tool=Path(a.tool or manifest['tool'])
    if not tool.is_file():raise FileNotFoundError('Vitis executable unavailable: '+str(tool))
    version=subprocess.check_output([str(tool),'-version'],stderr=subprocess.STDOUT,universal_newlines=True)
    if manifest['expected_tool_version'] not in version:raise ValueError('Vitis tool version mismatch: '+version[:300])
    vendor=tool.parents[1]/'include'
    if not vendor.is_dir():vendor=Path('/opt/Xilinx/Vitis_HLS/2021.1/include')
    # Critical: avoid fake source-side AP stubs.
    for name in VENDOR:
        data=(vendor/name).read_text(encoding='latin-1')
        if 'AP_FIXED_STUB_H' in data or 'AP_INT_STUB_H' in data:raise ValueError('Vendor header replaced by stub: '+name)
    out=Path(a.results).resolve();out.mkdir(parents=True,exist_ok=True)
    provenance=dict(schema='salt-coroni-eval9-run-v1',manifest_sha256=sha(root/'manifest.json'),
        runner_sha256=sha(root/'coroni_batch.py'),tool=str(tool),version=version,
        tool_sha256=sha(tool),vendor_headers={n:sha(vendor/n) for n in VENDOR})
    if (out/'batch_provenance.json').exists():
        old=json.loads((out/'batch_provenance.json').read_text(encoding='utf-8'))
        if old!=provenance:raise ValueError('Results provenance differs from current bundle/tool')
    else:dump(out/'batch_provenance.json',provenance)
    for job in manifest['jobs']:
        if a.job and a.job!=job['job_id']:continue
        run_one(root,job,tool,a.timeout_seconds,out)
    print('HLS PROCESSING COMPLETE; run collect for validated results')


def parse_report(path,job,version='2021.1'):
    root=ET.parse(str(path)).getroot()
    checks={'.//UserAssignments/Part':job['part'],
            './/UserAssignments/TopModelName':job['top'],
            './/ReportVersion/Version':version,
            './/UserAssignments/FlowTarget':'vivado',
            './/UserAssignments/unit':'ns',
            './/SummaryOfOverallLatency/unit':'clock cycles'}
    for xpath,expected in checks.items():
        actual=root.findtext(xpath)
        if actual!=expected:raise ValueError('Report mismatch %s=%r expected %r'%(xpath,actual,expected))
    clk=float(root.findtext('.//UserAssignments/TargetClockPeriod'))
    if abs(clk-job['clock_ns'])>1e-4:raise ValueError('Report target period mismatch')
    def number(path):
        s=root.findtext(path)
        if s is None or not re.fullmatch(r'[0-9]+',s.strip()):raise ValueError('Missing/noninteger report field '+path+' = '+repr(s))
        return int(s.strip())
    used={k:number('.//AreaEstimates/Resources/'+k) for k in RESOURCES}
    avail={k:number('.//AreaEstimates/AvailableResources/'+k) for k in RESOURCES}
    if avail!=job['capacities']:
        raise ValueError('Report capacities differ from pinned target: '+str(avail))
    raw=root.findtext('.//SummaryOfOverallLatency/Worst-caseLatency')
    cycles=int(raw) if raw and re.fullmatch(r'[0-9]+',raw.strip()) and int(raw)>0 else None
    # 'undef', '?', absent -> explicitly undefined, never zero or timeout.
    est=root.findtext('.//SummaryOfTimingAnalysis/EstimatedClockPeriod')
    clock_est=float(est) if est is not None else None
    if clock_est is not None and not math.isfinite(clock_est):clock_est=None
    return dict(latency_cycles=cycles,raw_latency_tag=raw,resources=used,
                percentages={k:100.*used[k]/avail[k] for k in RESOURCES},
                budget_feasible=all(used[k]<=job['budget'][k] for k in RESOURCES),
                estimated_clock_ns=clock_est)


def collect(a):
    bundle=Path(a.bundle).resolve();manifest=verify_bundle(bundle)
    results=Path(a.results).resolve()
    prov=json.loads((results/'batch_provenance.json').read_text(encoding='utf-8'))
    if prov['manifest_sha256']!=sha(bundle/'manifest.json') or prov['runner_sha256']!=manifest['runner_sha256']:
        raise ValueError('Batch/tool result identity mismatch')
    rows=[]
    for job in manifest['jobs']:
        d=results/job['job_id']
        rpt=d/'hls_project/solution1/syn/report'/(job['top']+'_csynth.xml')
        rec=d/'run_receipt.json';log=d/'synthesis.log'
        row={k:job.get(k) for k in ('job_id','kernel','request_index','part','clock_ns','budget_percent','checkpoint_step','checkpoint_sha256')}
        row.update(dict(status='not_run',latency_cycles=None,bram_pct=None,dsp_pct=None,ff_pct=None,lut_pct=None))
        if rec.is_file():
            receipt=json.loads(rec.read_text(encoding='utf-8'))
            if not log.is_file() or sha(log)!=receipt['synthesis_log_sha256']:
                raise ValueError('Log hash mismatched: '+job['job_id'])
            if rpt.is_file() and sha(rpt)!=receipt['report_sha256']:
                raise ValueError('Report hash mismatched: '+job['job_id'])
            row['status']=receipt['status']
            if receipt['returncode']==0 and rpt.is_file() and receipt['status']=='finished_command':
                parsed=parse_report(rpt,job,manifest['expected_tool_version'])
                row['status']='success' if parsed['latency_cycles'] is not None else 'latency_undefined'
                row['latency_cycles']=parsed['latency_cycles']
                row['bram_pct']=parsed['percentages']['BRAM_18K']
                row['dsp_pct']=parsed['percentages']['DSP']
                row['ff_pct']=parsed['percentages']['FF']
                row['lut_pct']=parsed['percentages']['LUT']
                row['budget_feasible']=parsed['budget_feasible']
                row['estimated_clock_ns']=parsed['estimated_clock_ns']
                row['raw_latency_tag']=parsed['raw_latency_tag']
        rows.append(row)
    out=Path(a.summary).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'results.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    cols=('job_id','kernel','checkpoint_step','part','clock_ns','budget_percent','status','latency_cycles','bram_pct','dsp_pct','ff_pct','lut_pct')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print('JOB | KERNEL | PART @ NS | STATUS | LATENCY CYCLES | BRAM% | DSP% | FF% | LUT%')
    for r in rows:
        fmt=lambda v: '—' if v is None else ('%.2f'%v)
        print('%s | %s | %s @ %g | %s | %s | %s | %s | %s | %s'%(r['job_id'],r['kernel'],r['part'],r['clock_ns'],r['status'],r['latency_cycles'] if r['latency_cycles'] is not None else '—',fmt(r['bram_pct']),fmt(r['dsp_pct']),fmt(r['ff_pct']),fmt(r['lut_pct'])))
    print('WROTE',out/'summary.csv',out/'results.jsonl')


def cli():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command')
    a=sub.add_parser('prepare',help='Offline: JSONL+full sources -> immutable HLS batch')
    a.add_argument('--predictions',required=True);a.add_argument('--sources-root',required=True)
    a.add_argument('--out',required=True);a.add_argument('--device',default='zcu104')
    a.add_argument('--clock-ns',type=float,default=10);a.add_argument('--budget-percent',type=float,default=100)
    a.add_argument('--targets-json',help='List of device/period/budget contexts; overrides CLI default target')
    a.add_argument('--require-nine',action='store_true')
    b=sub.add_parser('run',help='Coroni only: execute jobs with Vitis HLS')
    b.add_argument('--bundle',required=True);b.add_argument('--results',required=True)
    b.add_argument('--tool');b.add_argument('--timeout-seconds',type=int,default=3600)
    b.add_argument('--job',help='Optional single job ID')
    c=sub.add_parser('collect',help='Parse Coroni XML into provenance-safe CSV/JSONL summary')
    c.add_argument('--bundle',required=True);c.add_argument('--results',required=True);c.add_argument('--summary',required=True)
    return p.parse_args()


def main():
    args=cli()
    if args.command=='prepare':prepare(args)
    elif args.command=='run':run(args)
    elif args.command=='collect':collect(args)
    else:raise SystemExit('Specify prepare, run, or collect; see --help')

if __name__=='__main__':main()
