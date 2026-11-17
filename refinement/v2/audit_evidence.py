#!/usr/bin/env python3
"""Verify the v2 snapshot without running HLS or modifying recorded results."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location('frozen_cg_bdr_v2',HERE/'cg_bdr.py')
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)
CONTROLLER = '01e3a18aac4aa92531c1b54893f2b242a015b1c3c9af7acb0d6bc4e2211988ff'
SEMANTICS = 'a8e2f4b59957eefa258695f17ab4e7ed57b07a411a10101f04dcad277f103a81'
NAMES = ('covariance','jacobi-2d','syr2k','trmm-opt','Kalman')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def unpacked_digest(path):
    digest = hashlib.sha256()
    with gzip.open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root=HERE):
    root = Path(root).resolve()
    require(cg.digest(root/'cg_bdr.py') == CONTROLLER,'V2 controller bytes changed')
    require(cg.digest(root/'semantic_v2.py') == SEMANTICS,'V2 semantics bytes changed')
    ledger = cg.jload(root/'evidence.sha256.json')
    actual = {str(p.relative_to(root)) for p in (root/'evidence').rglob('*') if p.is_file()}
    require(actual == set(ledger),'Evidence file inventory changed')
    for name,expected in ledger.items():
        path = root/name
        require(root in path.resolve().parents,'Evidence path escapes package')
        require(cg.digest(path) == expected,'Evidence hash mismatch: '+name)
    counts = 0
    rows = []
    trials = []
    baseline_origins = cg.jload(root/'baseline_report_origins.json')
    for name in NAMES:
        bundle = root/'evidence'/(name+'_LATENCY_10ns_4eval')
        m = cg.jload(bundle/'manifest.json')
        state = cg.jload(bundle/'state.json')
        summary = cg.jload(bundle/'summary.json')
        require(m['schema'] == 'cg-bdr-v2-research' and m['controller_sha256'] == CONTROLLER and
                m['semantics_sha256'] == SEMANTICS,'Wrong sealed v2 code identity')
        require(m['kernel'] == name and m['part'] == cg.PART and m['clock_ns'] == 10 and
                m['budget_percent'] == 100 and m['max_evaluations'] == 4,'Wrong experiment context')
        require(m['objective_metric'] == summary['objective_metric'] == 'best_case_latency_cycles',
                'Wrong v2 selection metric')
        for filename,expected in m['input_hashes'].items():
            path = bundle/'inputs'/filename
            require(path.resolve().parent == (bundle/'inputs').resolve(),'Unsafe input filename')
            actual = cg.digest(path) if path.is_file() else unpacked_digest(path.with_suffix(path.suffix+'.gz'))
            require(actual == expected,'Unpacked input mismatch: '+str(path))
        for action in m['actions'].values():
            if action['kind'] == 'array':
                action['dims'] = {int(k):v for k,v in action['dims'].items()}
        template = (bundle/'inputs/template.cpp').read_text()
        source_spans = cg.loop_spans(template,m['actions'])
        require(source_spans == m['loop_spans'],'Loop spans differ from template')
        # Reproduce source-order iteration; sealed JSON sorts dictionary keys.
        derived = cg.semantic_v2.analyze(template,m['actions'],source_spans,m['compiler_graph'])
        require(derived == m['semantic_v2'],'Source-supported semantics differ from sealed manifest')
        history = state['evaluations']
        require(1 <= len(history) <= 4 and summary['charged_evaluations'] == len(history),'Budget mismatch')
        require(history[0]['directives'] == m['initial_directives'],'Initial configuration changed')
        for record in history:
            cfg = cg.validate(record['directives'],m['actions'])
            require(cfg == record['directives'] and cg.signature(cfg) == record['signature'] ==
                    record['selection']['signature'],'Directive signature mismatch')
            result = record['result']
            work = bundle/result['work_dir']
            require(bundle.resolve() in work.resolve().parents,'Unsafe evaluation path')
            require(cg.jload(work/'result.json') == result,'State/result receipt mismatch')
            require(result['tool_mode'] == 'VITIS_HLS' and result['synthesis_status'] == 'success' and
                    not result['error'],'Synthetic or unsuccessful result in native evidence')
            source = work/('candidate'+m.get('template_suffix','.cpp'))
            log = work/'synthesis.log'
            report = work/'hls_project/solution1/syn/report'/(m['top']+'_csynth.xml')
            for p,key in [(source,'source_sha256'),(log,'log_sha256'),
                          (report,'report_sha256'),(work/'run_hls.tcl','tcl_sha256')]:
                require(cg.digest(p) == result[key],'Native file hash mismatch: '+str(p))
            require(source.read_text() == cg.materialize(template,cfg),'Materialized candidate mismatch')
            text = log.read_text(errors='replace')
            require('Finished Command csynth_design' in text and 'Exiting vitis_hls' in text and
                    not re.search(r'^ERROR:',text,re.M),'Incomplete HLS synthesis log')
            require(cg.read_report(report,m) == result['metrics'],'Native metric mismatch')
            require(cg.parse_schedule_results(text) == result['schedule_results'],'Scheduling II evidence mismatch')
            require(cg.parse_diagnostics(text,template,m) == result['diagnostics'],'Diagnostic attribution mismatch')
            require(cg.semantic_v2.action_constraint_graph(m,result) == result['action_constraint_graph'],
                    'Attributed constraint graph mismatch')
            xml = ET.parse(str(report)).getroot()
            for resource,tag in [('bram','BRAM_18K'),('dsp','DSP'),('ff','FF'),('lut','LUT')]:
                require(int(xml.findtext('.//AreaEstimates/AvailableResources/'+tag)) == cg.CAPACITIES[resource],
                        'Native device capacity mismatch')
            counts += 1
            trials.append(dict(kernel=name,evaluation=result['index'],kind=record['selection']['kind'],
                               best_cycles=result['metrics']['latency_best_cycles'],
                               worst_cycles=result['metrics']['latency_worst_cycles']))
        good = [r for r in history if cg.eligible(r['result']['metrics'])]
        require(bool(good),'No eligible measured incumbent')
        winner = min(good,key=cg.objective_value)
        best = summary['best']
        require(best['best_evaluation'] == winner['result']['index'] and
                best['best_directives'] == winner['directives'] and
                best['best_latency_cycles'] == cg.objective_value(winner) and
                best['best_latency_worst_cycles'] == winner['result']['metrics']['latency_worst_cycles'],
                'Incorrect selected incumbent')
        require(best['initial_latency_cycles'] == cg.objective_value(history[0]),'Wrong initial metric')
        speedup = cg.objective_value(history[0])/cg.objective_value(winner)
        require(abs(best['improvement_percent'] - 100*(1-1/speedup)) < 1e-9,'Wrong improvement statistic')
        rows.append(dict(kernel=name,recorded_evaluations=len(history),initial_best_cycles=cg.objective_value(history[0]),
                         selected_best_cycles=cg.objective_value(winner),selected_evaluation=winner['result']['index'],
                         selected_worst_cycles=winner['result']['metrics']['latency_worst_cycles'],speedup=speedup,
                         initial_report_matches_earlier_snapshot=(history[0]['result']['report_sha256'] ==
                             baseline_origins[bundle.name])))
    require(counts == 9,'Expected nine recorded native evaluations')
    return dict(controller_sha256=CONTROLLER,semantics_sha256=SEMANTICS,verified_evaluations=counts,
                selection_metric='Best-caseLatency',new_hls_calls=0,gan_v2_run_present=False,kernels=rows,trials=trials)


def extract(destination):
    destination = Path(destination).resolve()
    require(not destination.exists(),'Extraction destination already exists')
    shutil.copytree(str(HERE/'evidence'),str(destination))
    for path in destination.rglob('llvm_graph.gexf.gz'):
        with gzip.open(path,'rb') as source,path.with_suffix('').open('xb') as target:
            shutil.copyfileobj(source,target)
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-to',help='Restore full bundle graphs into a fresh directory after verification')
    args = parser.parse_args()
    result = audit()
    if args.extract_to:
        result['extracted_to'] = extract(args.extract_to)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
