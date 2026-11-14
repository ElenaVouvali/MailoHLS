"""Prepare, run on Coroni (Python 3.6+), and summarize the sealed GAN grid."""
import argparse
import concurrent.futures
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
import time
import xml.etree.ElementTree as ET

RESOURCES = ('BRAM_18K', 'DSP', 'FF', 'LUT')
SETTINGS = ['config_array_partition -complete_threshold 0 -throughput_driven off',
            'config_compile -pipeline_loops 0']
TOOL = '/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls'
TOP = 'forward_propagation'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    p = Path(p)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(p)


def strip_pragmas(text):
    return re.sub(r'^\s*#\s*pragma\b[^\n]*(?:\n|$)', '', text, flags=re.M)


def tokens(text):
    text = strip_pragmas(text)
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    return re.findall(r'\w+|[^\s]', text)


def materialize(template, assignment):
    pattern = re.compile(r'auto\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+)\}', re.I)
    observed = ['AUTO{'+m[1].upper()+'}' for m in pattern.finditer(template)]
    assert len(observed) == len(set(observed)) == 25 and set(observed) == set(assignment)
    source = pattern.sub(lambda m: assignment['AUTO{'+m[1].upper()+'}'], template)
    lines = []
    for line in source.splitlines(True):
        if re.match(r'\s*#\s*pragma HLS (pipeline II=0|unroll factor=0)\s*$', line):
            continue
        if '#pragma HLS array_partition' in line and 'type=complete' in line:
            line = re.sub(r'\s+factor=0\b', '', line)
        lines.append(line)
    result = ''.join(lines)
    assert 'auto{' not in result and tokens(result) == tokens(template)
    return result


def prepare(a):
    predictions = json.loads((a.predictions/'predicted_directives.json').read_text())
    inference = json.loads((a.predictions/'manifest.json').read_text())
    assert inference['status'] == 'INFERENCE_COMPLETED' and len(predictions) == 18
    original = json.loads((a.prepared/'manifest.json').read_text())
    template_file = a.prepared/'sources/GAN/GAN_placeholders.cpp'
    assert sha(template_file) == original['source_sha256']
    template = template_file.read_text()
    baseline = strip_pragmas(template)
    assert 'auto{' not in baseline and tokens(baseline) == tokens(template)
    a.root.mkdir(parents=True, exist_ok=False)
    headers = {}
    for p in sorted((a.prepared/'sources/GAN').glob('*.h')):
        assert sha(p) == original['header_sha256'][p.name]
        if p.name not in ('ap_fixed.h', 'ap_int.h'):
            headers[p.name] = dict(sha256=sha(p), data=p.read_text())
    requests, jobs, signatures = [], [], {}

    def job_for(part, clock, role, source):
        signature = hashlib.sha256(json.dumps([part, clock, source, SETTINGS], sort_keys=True).encode()).hexdigest()
        if signature in signatures:
            return signatures[signature]
        device = 'zcu104' if part.startswith('xczu') else 'u200'
        job_id = '{}_{}ns_{}_{}'.format(device, str(clock).replace('.', 'p'), role, signature[:10])
        directory = a.root/'jobs'/job_id
        directory.mkdir(parents=True)
        (directory/'source.cpp').write_text(source)
        for name, record in headers.items():
            (directory/name).write_text(record['data'])
        tcl = '\n'.join(['open_project -reset hls_project', 'set_top '+TOP,
            'add_files -cflags {-I/opt/Xilinx/Vitis_HLS/2021.1/include} {source.cpp}',
            'open_solution -reset solution1 -flow_target vivado', 'set_part {'+part+'}',
            'create_clock -period {} -name default'.format(clock), *SETTINGS, 'csynth_design', 'exit', ''])
        (directory/'run_hls.tcl').write_text(tcl)
        files = {p.name: sha(p) for p in directory.iterdir()}
        jobs.append(dict(job_id=job_id, part=part, clock_ns=clock, role=role,
                         signature=signature, files=files))
        signatures[signature] = job_id
        return job_id

    for row in predictions:
        assert row['film_enabled'] and row['source_domain_complete']
        part, clock = row['device'], row['clock_ns']
        base_id = job_for(part, clock, 'baseline', baseline)
        pred_id = job_for(part, clock, 'predicted', materialize(template, row['directives']))
        requests.append(dict(context_id=row['context_id'], part=part, clock_ns=clock,
            budget_pct=row['budget_pct'], baseline_job=base_id, predicted_job=pred_id,
            directives=row['directives']))
    assert len(jobs) == 18 and sum(j['role']=='baseline' for j in jobs)==6
    manifest = dict(status='PREPARED', created_utc=datetime.datetime.utcnow().isoformat()+'Z',
        source_sha256=sha(template_file), predictions_sha256=sha(a.predictions/'predicted_directives.json'),
        inference_manifest_sha256=sha(a.predictions/'manifest.json'), inference_identity={
            key: inference[key] for key in ('checkpoint_step', 'parent_sha256', 'film_sha256', 'lora_sha256')},
        headers={name:r['sha256'] for name,r in headers.items()},
        vendor_header_policy='analysis ap_fixed.h stub excluded; use genuine Vitis 2021.1 AP headers',
        pragma_free_baseline='all pragma lines removed from top source; identical application headers and HLS settings',
        disabled_directives='pipeline II=0 and unroll factor=0 omitted; complete partition factor=0 sentinel omitted',
        settings=SETTINGS, top=TOP, tool=TOOL, requests=requests, jobs=jobs,
        speedup_definition='baseline worst-case cycles / predicted worst-case cycles, matched part and requested period',
        area_definition='100 * mean(resource count / full device capacity) for BRAM18K,DSP,FF,LUT',
        functional_simulation='NOT RUN', post_route_timing='NOT RUN')
    shutil.copyfile(__file__, a.root/'gan_tcad_synthesis.py')
    manifest['runner_sha256'] = sha(a.root/'gan_tcad_synthesis.py')
    dump(a.root/'manifest.json', manifest)
    print('Prepared 12 distinct predicted syntheses and 6 pragma-free baselines.', flush=True)


def parse_report(path, job):
    root = ET.parse(str(path)).getroot()
    for query, expected in [('.//UserAssignments/Part', job['part']), ('.//UserAssignments/TopModelName', TOP),
                             ('.//ReportVersion/Version', '2021.1'), ('.//UserAssignments/FlowTarget', 'vivado')]:
        assert root.findtext(query) == expected, (query, root.findtext(query), expected)
    assert abs(float(root.findtext('.//UserAssignments/TargetClockPeriod')) - job['clock_ns']) < 1e-6
    def count(query):
        s = root.findtext(query)
        assert s is not None and s.strip().isdigit(), (query,s)
        return int(s)
    latency = count('.//SummaryOfOverallLatency/Worst-caseLatency')
    assert latency > 0
    used = {r: count('.//AreaEstimates/Resources/'+r) for r in RESOURCES}
    caps = {r: count('.//AreaEstimates/AvailableResources/'+r) for r in RESOURCES}
    assert all(caps[r] > 0 for r in RESOURCES)
    estimated = float(root.findtext('.//SummaryOfTimingAnalysis/EstimatedClockPeriod'))
    assert math.isfinite(estimated) and estimated > 0
    return dict(latency_cycles=latency, resources=used, capacities=caps, estimated_clock_ns=estimated,
                hls_estimated_timing_met=estimated <= job['clock_ns'],
                mean_normalized_utilization_pct=25.*sum(used[r]/caps[r] for r in RESOURCES),
                resource_utilization_pct={r:100.*used[r]/caps[r] for r in RESOURCES})


def run(a):
    root = a.root.resolve()
    m = json.loads((root/'manifest.json').read_text())
    assert sha(root/'gan_tcad_synthesis.py') == m['runner_sha256']
    version = subprocess.check_output([TOOL, '-version'], stderr=subprocess.STDOUT, universal_newlines=True)
    assert '2021.1' in version
    vendor = {}
    for name in ('ap_fixed.h','ap_int.h'):
        p = Path('/opt/Xilinx/Vitis_HLS/2021.1/include')/name
        text = p.read_text()
        assert 'Xilinx' in text and 'AP_FIXED_STUB_H' not in text and 'AP_INT_STUB_H' not in text
        vendor[name] = sha(p)
    tool = dict(version=version, executable_sha256=sha(TOOL), vendor_header_sha256=vendor)
    dump(root/'tool.json', tool)

    def one(job):
        d = root/'jobs'/job['job_id']
        for name, digest in job['files'].items():
            assert sha(d/name) == digest
        if (d/'result.json').exists():
            result = json.loads((d/'result.json').read_text())
            if result['status']=='success':
                assert sha(d/'hls_project/solution1/syn/report'/ (TOP+'_csynth.xml')) == result['report_sha256']
                return result
            raise RuntimeError('Preserve failed job; explicit new attempt required: '+job['job_id'])
        assert not (d/'synthesis.log').exists(), 'Existing unsealed attempt: '+job['job_id']
        tick = time.monotonic()
        print('START', job['job_id'], flush=True)
        result = dict(job_id=job['job_id'], part=job['part'], clock_ns=job['clock_ns'], role=job['role'],
                      status='failed', started_utc=datetime.datetime.utcnow().isoformat()+'Z')
        with (d/'synthesis.log').open('w') as log:
            process = subprocess.Popen([TOOL,'-f','run_hls.tcl'], cwd=str(d), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = process.wait(timeout=a.timeout)
                result['returncode'] = rc
                if rc != 0:
                    result['error'] = 'Vitis returned '+str(rc)
                else:
                    report = d/'hls_project/solution1/syn/report'/(TOP+'_csynth.xml')
                    try:
                        result.update(parse_report(report, job), status='success', report_sha256=sha(report))
                    except Exception as e:
                        result['error'] = repr(e)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result.update(status='timeout', error='Exceeded {} seconds'.format(a.timeout))
        result.update(seconds=time.monotonic()-tick, source_sha256=sha(d/'source.cpp'),
                      tcl_sha256=sha(d/'run_hls.tcl'), log_sha256=sha(d/'synthesis.log'))
        dump(d/'result.json', result)
        print('FINISH',job['job_id'],result['status'],round(result['seconds'],1),flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        results = list(executor.map(one,m['jobs']))
    dump(root/'execution.json', dict(status='COMPLETED' if all(x['status']=='success' for x in results) else 'FAILED_JOBS',
        successful=sum(x['status']=='success' for x in results), total=len(results), results=results))
    summarize(root)


def summarize(root):
    m = json.loads((root/'manifest.json').read_text())
    rows = []
    for request in m['requests']:
        r = dict(request)
        pred = json.loads((root/'jobs'/r['predicted_job']/'result.json').read_text())
        base = json.loads((root/'jobs'/r['baseline_job']/'result.json').read_text())
        r.update(prediction=pred, baseline=base, speedup=None, mean_normalized_utilization_pct=None)
        if pred['status']=='success' and base['status']=='success':
            assert pred['part']==base['part']==r['part'] and pred['clock_ns']==base['clock_ns']==r['clock_ns']
            assert pred['capacities']==base['capacities']
            r.update(speedup=base['latency_cycles']/pred['latency_cycles'],
                mean_normalized_utilization_pct=pred['mean_normalized_utilization_pct'],
                budget_met=all(pred['resources'][k] <= round(pred['capacities'][k]*r['budget_pct']/100.) for k in RESOURCES))
        rows.append(r)
    dump(root/'request_results.json', rows)
    columns=['context_id','part','clock_ns','budget_pct','mean_normalized_utilization_pct','speedup',
             'baseline_cycles','predicted_cycles','BRAM_18K','DSP','FF','LUT','budget_met','estimated_clock_ns','hls_estimated_timing_met']
    with (root/'request_results.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:
            x={k:r.get(k) for k in columns}
            x.update(baseline_cycles=r['baseline'].get('latency_cycles'),predicted_cycles=r['prediction'].get('latency_cycles'),
                     **r['prediction'].get('resources',{}))
            for k in ('estimated_clock_ns','hls_estimated_timing_met'):x[k]=r['prediction'].get(k)
            w.writerow(x)
    text=['# GAN FiLM: Vitis HLS 2021.1 synthesis of 18 requests','',
        'Area (%) = 100 × mean(BRAM18K/capacity, DSP/capacity, FF/capacity, LUT/capacity).',
        'Speedup = pragma-free baseline worst-case cycles / predicted worst-case cycles at the same part and requested clock.',
        'Both use the same application headers and synthesis settings. Budget-only duplicates share physical measurements.',
        'Timing is an HLS estimate, not post-route timing; no functional simulation was run.','',
        '|Device|Clock ns|Budget %|Mean utilization %|Speedup|Baseline cycles|Predicted cycles|Budget met|Estimated clock ns|HLS timing met|',
        '|'+'---|'*10]
    for r in rows:
        pred,base=r['prediction'],r['baseline']
        if r['speedup'] is None:
            text.append('|{}|{}|{}|FAILED|—|{}|{}|—|—|—|'.format(r['part'],r['clock_ns'],r['budget_pct'],base['status'],pred['status']))
        else:
            text.append('|{}|{}|{}|{:.4f}|{:.3f}×|{}|{}|{}|{}|{}|'.format(
                'ZCU104' if r['part'].startswith('xczu') else 'U200',r['clock_ns'],r['budget_pct'],
                r['mean_normalized_utilization_pct'],r['speedup'],base['latency_cycles'],pred['latency_cycles'],
                r['budget_met'],pred['estimated_clock_ns'],pred['hls_estimated_timing_met']))
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print('Summarized {} requests; {} have paired measurements.'.format(len(rows),sum(r['speedup'] is not None for r in rows)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','run','summarize'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predictions',type=Path)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--timeout',type=int,default=21600)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='run':run(a)
    else:summarize(a.root)


if __name__=='__main__':
    main()
