#!/usr/bin/env python3
"""Read-only verification of the eleven archived native Vitis evaluations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import cg_bdr as cg

HERE = Path(__file__).resolve().parent
CONTROLLER_SHA256 = 'c5a7c7077c24adde96b32ee5eee34afdc0d2faeb6698b34151abb27815c56b32'
NAMES = ('covariance', 'jacobi-2d', 'syr2k', 'trmm-opt', 'GAN', 'Kalman')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unpacked_digest(path):
    digest = hashlib.sha256()
    with gzip.open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root=HERE):
    root = Path(root).resolve()
    require(cg.digest(root/'cg_bdr.py') == CONTROLLER_SHA256, 'Tested controller bytes changed')
    ledger = cg.jload(root/'evidence.sha256.json')
    actual_files = {str(p.relative_to(root)) for p in (root/'evidence').rglob('*') if p.is_file()}
    require(set(ledger) == actual_files, 'Evidence file inventory changed')
    for name, expected in ledger.items():
        path = root/name
        require(root in path.resolve().parents, 'Evidence path escapes package')
        require(cg.digest(path) == expected, 'Packed evidence hash mismatch: '+name)
    publication = cg.jload(root/'PUBLICATION_AUDIT.json')
    require(cg.digest(root/'evidence_plot_rows.jsonl') == publication['after_sha256'], 'Plot snapshot changed')
    rows = [json.loads(line) for line in (root/'evidence_plot_rows.jsonl').read_text().splitlines()]
    require(len(rows) == 18, 'Expected eighteen plotting rows')
    results = []
    count = 0
    for name in NAMES:
        bundle = root/'evidence'/(name+'_LATENCY_10ns_4eval')
        m = cg.jload(bundle/'manifest.json')
        state = cg.jload(bundle/'state.json')
        summary = cg.jload(bundle/'summary.json')
        require(m['kernel'] == name and m['part'] == cg.PART and m['clock_ns'] == 10 and
                m['budget_percent'] == 100 and m['max_evaluations'] == 4, 'Unexpected campaign context')
        require(m['schema'] == 'cg-bdr-v0.1-research' and
                m['controller_sha256'] == CONTROLLER_SHA256, 'Unknown controller or manifest schema')
        for filename, expected in m['input_hashes'].items():
            path = bundle/'inputs'/filename
            require(path.resolve().parent == (bundle/'inputs').resolve(), 'Unsafe input name')
            actual = cg.digest(path) if path.exists() else unpacked_digest(path.with_suffix(path.suffix+'.gz'))
            require(actual == expected, 'Unpacked input hash mismatch: '+str(path))
        for action in m['actions'].values():
            if action['kind'] == 'array':
                action['dims'] = {int(k):v for k,v in action['dims'].items()}
        history = state['evaluations']
        require(1 <= len(history) <= 4 and summary['charged_evaluations'] == len(history), 'Budget mismatch')
        initial = next(r for r in rows if r['kernel'] == name and r['objective'] == 'LATENCY')
        require(initial['directives'] == m['initial_directives'] == history[0]['directives'], 'Initial assignment changed')
        checked = {}
        for record in history:
            cfg = cg.validate(record['directives'], m['actions'])
            require(cfg == record['directives'] and cg.signature(cfg) == record['signature'] ==
                    record['selection']['signature'], 'Directive signature mismatch')
            result = record['result']
            work = bundle/result['work_dir']
            require(bundle.resolve() in work.resolve().parents, 'Unsafe evaluation path')
            require(cg.jload(work/'result.json') == result, 'Result differs from state')
            require(result['tool_mode'] == 'VITIS_HLS' and result['synthesis_status'] == 'success'
                    and not result['error'], 'Synthetic or unsuccessful evaluation in measured evidence')
            source = work/('candidate'+m.get('template_suffix','.cpp'))
            report = work/'hls_project/solution1/syn/report'/(m['top']+'_csynth.xml')
            log = work/'synthesis.log'
            for path, key in [(source,'source_sha256'), (report,'report_sha256'),
                              (log,'log_sha256'), (work/'run_hls.tcl','tcl_sha256')]:
                require(cg.digest(path) == result[key], 'Native evaluation hash mismatch: '+str(path))
            require(source.read_text() == cg.materialize((bundle/'inputs/template.cpp').read_text(),cfg),
                    'Candidate differs from frozen template and directives')
            text = log.read_text(errors='replace')
            require('Finished Command csynth_design' in text and 'Exiting vitis_hls' in text and
                    not re.search(r'^ERROR:',text,re.M), 'Incomplete or erroneous HLS execution')
            native = cg.read_report(report,m)
            require(native == result['metrics'], 'Native HLS metrics differ from receipt')
            xml = ET.parse(str(report)).getroot()
            for resource, tag in [('bram','BRAM_18K'),('dsp','DSP'),('ff','FF'),('lut','LUT')]:
                require(int(xml.findtext('.//AreaEstimates/AvailableResources/'+tag)) == cg.CAPACITIES[resource],
                        'Unexpected device capacity')
            best = int(xml.findtext('.//SummaryOfOverallLatency/Best-caseLatency'))
            worst = int(xml.findtext('.//SummaryOfOverallLatency/Worst-caseLatency'))
            require(0 < best <= worst == native['latency_cycles'], 'Latency bound mismatch')
            checked[result['index']] = (best,worst)
            count += 1
        good = [r for r in history if cg.eligible(r['result']['metrics'])]
        require(bool(good), 'No eligible measured candidate')
        winner = min(good,key=lambda r:r['result']['metrics']['latency_cycles'])
        index = winner['result']['index']
        require(summary['best']['best_evaluation'] == index and
                summary['best']['best_directives'] == winner['directives'] and
                summary['best']['best_latency_cycles'] == checked[index][1], 'Wrong measured incumbent')
        selected = publication['selected'][name]
        for filename, field in [('manifest.json','manifest_sha256'),('state.json','state_sha256'),('summary.json','summary_sha256')]:
            require(cg.digest(bundle/filename) == selected[field], 'Publication snapshot mismatch')
        refined = next(r for r in rows if r['kernel'] == name and r['objective'] == 'LATENCY_refinement')
        require(refined['directives'] == winner['directives'] and refined['latency'] == checked[index][0],
                'Plot row differs from selected source or best-case bound')
        require(refined['resources'] == winner['result']['metrics']['resources'], 'Plot resources differ')
        results.append(dict(kernel=name,charged_evaluations=len(history),selected_evaluation=index,
                            initial_worst_cycles=history[0]['result']['metrics']['latency_cycles'],
                            best_case_cycles=checked[index][0],worst_case_cycles=checked[index][1]))
    require(count == publication['evaluation_records_verified'] == 11, 'Missing native trials')
    return dict(controller_sha256=CONTROLLER_SHA256,verified_evaluations=count,kernels=results,
                new_hls_calls=0,selection_metric='Worst-caseLatency',plot_metric='Best-caseLatency')


def extract(root, destination):
    destination = Path(destination).resolve()
    require(not destination.exists(), 'Extraction destination already exists')
    shutil.copytree(str(root/'evidence'),str(destination))
    for path in destination.rglob('llvm_graph.gexf.gz'):
        with gzip.open(path,'rb') as source, path.with_suffix('').open('xb') as target:
            shutil.copyfileobj(source,target)
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-to', help='Optionally restore full bundles into a fresh directory after verification')
    args = parser.parse_args()
    result = audit()
    if args.extract_to:
        result['extracted_to'] = extract(HERE,args.extract_to)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
