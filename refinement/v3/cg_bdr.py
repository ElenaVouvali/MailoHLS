#!/usr/bin/env python3
"""CG-BDR v3.0: sealed, manually gated best-case-latency experiments (Python 3.8+)."""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import gzip
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
sys.dont_write_bytecode = True
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent/'v2'))
spec = importlib.util.spec_from_file_location('frozen_v2_helpers', HERE.parent/'v2/cg_bdr.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
import semantic_v3
import rank_v3

SCHEMA = 'cg-bdr-v3.0-best-case-sealed'
PART = base.PART
VITIS = '/opt/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls'
FLAGS = ['-I/opt/Xilinx/Vitis_HLS/2021.1/include']
CODE = ('cg_bdr.py','semantic_v3.py','rank_v3.py','runner_v3.py','../v2/cg_bdr.py','../v2/semantic_v2.py')
jload, jdump, digest, signature, die = base.jload, base.jdump, base.digest, base.signature, base.die


def code_hashes():
    return {name: digest(HERE/name) for name in CODE}


def emit(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), flush=True)


def check(condition, message):
    if not condition:
        die(message)


@contextmanager
def writer(bundle):
    # Existing state inode is stable; atomic state replacement needs a separate lock.
    lock = (bundle/'.writer.lock').open('a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            die('One writer per bundle: another writer holds the lock')
        yield
    finally:
        lock.close()


def unpack_hash(path):
    import hashlib
    h = hashlib.sha256()
    with gzip.open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def context(m):
    return {key: m[key] for key in ('kernel','top','part','clock_ns','budget_percent',
                                   'capacities','hls_version','tool_flags','settings','initial_directives','input_hashes')}


def read_report(path, m):
    xml = ET.parse(str(path)).getroot()
    check(xml.findtext('.//ReportVersion/Version') == '2021.1', 'Missing/wrong HLS 2021.1 report identity')
    for resource, tag in [('bram','BRAM_18K'),('dsp','DSP'),('ff','FF'),('lut','LUT')]:
        check(xml.findtext('.//AreaEstimates/AvailableResources/'+tag) == str(base.CAPACITIES[resource]),
              'Device capacity mismatch: '+resource)
    result = base.read_report(path, m)  # Numeric best and worst validated independently.
    clock = result['estimated_clock_ns']
    check(clock is None or (math.isfinite(clock) and clock > 0), 'Invalid estimated clock')
    result['schedule_margin_warning'] = clock > 7.3 if clock is not None else None
    result['timing_claim'] = 'HLS estimate only; routed timing closure unknown'
    return result


def eligible(metrics):
    if not metrics:
        return False
    best, worst, clock = (metrics.get(k) for k in ('latency_best_cycles','latency_worst_cycles','estimated_clock_ns'))
    return bool(type(best) is int and type(worst) is int and 0 < best <= worst and
                isinstance(clock, (int,float)) and math.isfinite(clock) and 0 < clock <= 10 and
                metrics.get('budget_feasible') is True and
                metrics.get('hls_estimated_clock_within_target') is True and
                all(type(metrics.get('resources', {}).get(k)) is int and
                    0 <= metrics['resources'][k] <= cap for k, cap in base.CAPACITIES.items()))


def best_row(history):
    good = [r for r in history if eligible(r['result'].get('metrics')) and
            r['result'].get('tool_mode') == 'VITIS_HLS']
    return min(good, key=lambda r: (r['result']['metrics']['latency_best_cycles'], r['result']['index'])) if good else None


def enrich(result, m, template, log):
    evidence_m = m
    if m['policy'] == 'feedback-only':
        evidence_m = dict(m, compiler_graph={'anchored_actions': list(m['actions'])})
    result['diagnostics'] = base.parse_diagnostics(log, template, evidence_m)
    if m['policy'] == 'feedback-only':
        for diagnostic in result['diagnostics']:
            diagnostic['attribution'] = 'source_line+source_Lk_anchor' if diagnostic['affected_loops'] else 'unmapped'
    result['schedule_results'] = []
    result['compiler_replication'] = []
    for line_no, line in enumerate(log.splitlines(), 1):
        result['schedule_results'].extend(dict(row, log_line=line_no) for row in base.parse_schedule_results(line))
        match = re.search(r"Unrolling loop '(L\d+)'.*partially with a factor of (\d+)",line)
        if match:
            result['compiler_replication'].append(dict(loop=match[1],factor=int(match[2]),raw=line,log_line=line_no))
    result['bottleneck_families'] = semantic_v3.warning_groups(result['diagnostics'])
    result['schedule_groups'] = semantic_v3.schedule_groups(result['schedule_results'])
    result.pop('action_constraint_graph', None)
    result['evidence_claim'] = 'Observed Vitis facts + source attribution; interventions remain hypotheses'
    return result


def prepare(args):
    out = Path(args.bundle).expanduser().resolve()
    check(not out.exists(), 'Bundle exists; use a new v3 directory')
    srcdir = Path(args.source_dir).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    info = Path(args.kernel_info).expanduser().resolve()
    graph = Path(args.graph).expanduser().resolve()
    initial = Path(args.initial).expanduser().resolve()
    check(template.parent == srcdir and template.suffix in ('.cpp','.c'), 'Template must be in source-dir')
    top, actions = base.parse_info(info.read_text())
    check(re.fullmatch(r'[A-Za-z_]\w*',top), 'Unsupported top function name')
    row = jload(initial)
    check(isinstance(row, dict), 'Initial prediction needs complete deployment context')
    for key, value in [('kernel',args.kernel),('objective','LATENCY'),('device',PART),
                       ('clock_period_ns',10),('budget_percent',100)]:
        check(row.get(key) == value, 'Initial prediction context mismatch: '+key)
    cfg = base.validate(row['directives'], actions)
    text = template.read_text()
    expected = {(typ, lid) for lid, a in actions.items() for typ in
                (('PIPE','UNROLL') if a['kind']=='loop' else ('ARRAY_T','ARRAY_F','ARRAY_D'))}
    check(set(base.PH.findall(text)) == expected, 'Template action placeholder coverage mismatch')
    base.materialize(text, cfg)
    policy = args.policy
    # Do not even read graph topology in feedback-only mode.
    graph_data = (base.graph_index(graph, actions) if policy == 'graph' else
                  dict(sha256=digest(graph), ablation='not_read_or_used_for_filtering'))
    if policy == 'graph':
        check(not graph_data['missing_actions'], 'Missing graph anchors')
    spans = base.loop_spans(text, actions)
    semantic = semantic_v3.analyze(base, text, actions, spans, graph_data, policy=='feedback-only')
    relations = base.direct_accesses(text, actions, spans, graph_data) if policy=='graph' else []
    out.mkdir(parents=True)
    (out/'inputs').mkdir()
    for p in srcdir.iterdir():
        if p.is_file() and p.suffix.lower() in ('.h','.hpp','.inc') and p.name not in base.VENDOR_HEADERS:
            shutil.copy2(str(p), str(out/'inputs'/p.name))
    for src, name in [(template,'template.cpp'),(info,'kernel_info.txt'),(graph,'llvm_graph.gexf')]:
        shutil.copy2(str(src), str(out/'inputs'/name))
    # Prediction is sealed separately; preserve original bytes without treating it as a source header.
    shutil.copy2(str(initial), str(out/'initial_prediction.json'))
    m = dict(schema=SCHEMA, kernel=args.kernel, top=top, template_suffix=template.suffix,
             part=PART, clock_ns=10, budget_percent=100, capacities=base.CAPACITIES,
             hls_version='2021.1', tool_flags=FLAGS, settings=list(base.SETTINGS),
             max_evaluations=4, objective_metric='best_case_latency_cycles', policy=policy,
             compiler_graph=graph_data, actions=actions, loop_spans=spans,
             loop_array_relations=relations, semantic_v3=semantic, initial_directives=cfg,
             input_hashes={p.name:digest(p) for p in (out/'inputs').iterdir()},
             initial_prediction_sha256=digest(initial), code_hashes=code_hashes(),
             extraction_provenance=getattr(args, 'extraction_provenance', None) or
                 dict(template_path=str(template), graph_path=str(graph), graph_sha256=digest(graph)),
             artifact_note='Preparation only; zero HLS calls')
    m['frozen_experiment_signature'] = signature(context(m))
    jdump(out/'manifest.json', m)
    jdump(out/'seal.json', dict(manifest_sha256=digest(out/'manifest.json')))
    jdump(out/'state.json', dict(evaluations=[], stop_reason=None))
    return out


def verify_bundle(bundle, verify_records=True):
    bundle = Path(bundle)
    m = jload(bundle/'manifest.json')
    check(m.get('schema') == SCHEMA, 'Not a v3 bundle; v0.1/v2 cannot be resumed')
    check(jload(bundle/'seal.json')['manifest_sha256'] == digest(bundle/'manifest.json'), 'Immutable manifest seal mismatch')
    check(m['code_hashes'] == code_hashes(), 'Sealed controller/semantic/ranker/runner/helper code changed; create new bundle')
    check(m['part']==PART and m['clock_ns']==10 and m['max_evaluations']==4 and
          m['settings']==list(base.SETTINGS) and m['tool_flags']==FLAGS and
          m['hls_version']=='2021.1' and m['capacities']==base.CAPACITIES and m['budget_percent']==100,
          'Frozen protocol context mismatch')
    check({p.name for p in (bundle/'inputs').iterdir() if p.is_file()} == set(m['input_hashes']), 'Input inventory mismatch')
    for name, expected in m['input_hashes'].items():
        check(Path(name).name==name and digest(bundle/'inputs'/name)==expected, 'Input hash mismatch: '+name)
    check(digest(bundle/'initial_prediction.json')==m['initial_prediction_sha256'], 'Prediction changed')
    check(signature(context(m))==m['frozen_experiment_signature'], 'Experiment context seal mismatch')
    for a in m['actions'].values():
        if a['kind']=='array':
            a['dims'] = {int(k):v for k,v in a['dims'].items()}
    state = jload(bundle/'state.json')
    hist = state['evaluations']
    if (bundle/'pending.json').exists():
        pending = jload(bundle/'pending.json')
        check(1 <= pending['index'] <= 4 and pending['index'] in (len(hist),len(hist)+1), 'Invalid charged reservation')
    check(len(hist)<=4 and len({r['signature'] for r in hist})==len(hist), 'Budget/duplicate signature violation')
    if hist:
        check(hist[0]['directives']==m['initial_directives'], 'Immutable baseline changed')
    for idx, r in enumerate(hist, 1):
        check(r['signature']==signature(base.validate(r['directives'], m['actions'])) and r['result']['index']==idx,
              'Candidate signature/order mismatch')
        if verify_records:
            verify_record(bundle, m, r)
    return m


def verify_record(bundle, m, record):
    result = record['result']
    work = bundle/result['work_dir']
    check(bundle.resolve() in work.resolve().parents, 'Record path escapes bundle')
    check(jload(work/'result.json')==result, 'State/receipt mismatch')
    for filename, expected in result['artifact_hashes'].items():
        path = work/filename
        check(work.resolve() in path.resolve().parents and digest(path)==expected, 'Measurement artifact hash mismatch: '+filename)
    check((work/('candidate'+m['template_suffix'])).read_text()==base.materialize((bundle/'inputs/template.cpp').read_text(),record['directives']),
          'Candidate differs from frozen source/directives')
    if result.get('tool_mode')=='VITIS_HLS':
        check((work/'run_hls.tcl').read_text()==expected_tcl(m), 'Native Tcl/tool flags mismatch')
    if result.get('metrics'):
        native = read_report(work/'hls_project/solution1/syn/report'/(m['top']+'_csynth.xml'), m)
        check(native==result['metrics'], 'Native XML/metric mismatch')
        log = (work/'synthesis.log').read_text(errors='replace')
        check(('2021.1' in log or result['tool_mode']=='SYNTHETIC_TEST_ONLY') and 'Finished Command csynth_design' in log and 'Exiting vitis_hls' in log and not re.search(r'^ERROR:',log,re.M),
              'Incomplete/error native HLS log')
        derived = enrich({}, m, (bundle/'inputs/template.cpp').read_text(), log)
        for key in ('schedule_results','schedule_groups','diagnostics','bottleneck_families','compiler_replication'):
            check(result[key]==derived[key], 'Derived diagnostic/schedule mismatch')
    check(result.get('tool_mode') in ('VITIS_HLS','SYNTHETIC_TEST_ONLY'), 'Unknown measurement authority')
    if record['reuse']:
        origin = record['reuse']
        check(not record['new_hls_call'] and origin['frozen_experiment_signature']==m['frozen_experiment_signature'],
              'Cross-experiment reuse forbidden')


# Tcl is exact, including vendor headers; accepting an extra command is forbidden.
def expected_tcl(m):
    return '\n'.join(['open_project -reset hls_project', 'set_top '+m['top'],
        'add_files -cflags {'+FLAGS[0]+'} {candidate'+m['template_suffix']+'}',
        'open_solution -reset solution1 -flow_target vivado', 'set_part {'+PART+'}',
        'create_clock -period 10 -name default', *base.SETTINGS, 'csynth_design','exit',''])


def artifacts(work):
    return {str(p.relative_to(work)):digest(p) for p in work.rglob('*') if p.is_file() and p.name!='result.json'}


def import_baseline(bundle, archive):
    bundle, archive = Path(bundle).resolve(), Path(archive).resolve()
    with writer(bundle):
        m = verify_bundle(bundle)
        state = jload(bundle/'state.json')
        check(not state['evaluations'] and not (bundle/'pending.json').exists(), 'Import requires a fresh empty v3 bundle')
        # Only packaged native archives with a frozen inventory are trusted by this importer.
        package = archive.parent.parent
        ledger = jload(package/'evidence.sha256.json')
        for p in archive.rglob('*'):
            if p.is_file():
                check(digest(p)==ledger.get(str(p.relative_to(package))), 'Archived inventory hash mismatch: '+str(p))
        old = jload(archive/'manifest.json')
        history = jload(archive/'state.json')['evaluations']
        row = history[0]
        check(old['kernel']==m['kernel'] and old['top']==m['top'] and old['part']==PART and
              old['clock_ns']==10 and old['budget_percent']==100 and old['max_evaluations']==4 and
              old['settings']==m['settings'], 'Archived deployment context mismatch')
        check(old['input_hashes']==m['input_hashes'], 'Archived source/graph/header hashes mismatch')
        for name, expected in old['input_hashes'].items():
            path = archive/'inputs'/name
            actual = digest(path) if path.is_file() else unpack_hash(path.with_suffix(path.suffix+'.gz'))
            check(actual==expected, 'Archived input mismatch: '+name)
        check(row['directives']==old['initial_directives']==m['initial_directives'], 'Initial prediction mismatch')
        check(row['signature']==signature(m['initial_directives']), 'Archived signature mismatch')
        result = row['result']
        work = archive/result['work_dir']
        check(archive in work.resolve().parents, 'Unsafe archived work path')
        check(result['tool_mode']=='VITIS_HLS' and result['synthesis_status']=='success' and not result['error'], 'Not successful native Vitis evidence')
        check(jload(work/'result.json')==result, 'Archived receipt mismatch')
        for filename, key in [('candidate'+m['template_suffix'],'source_sha256'),('run_hls.tcl','tcl_sha256'),
                              ('synthesis.log','log_sha256'),('hls_project/solution1/syn/report/'+m['top']+'_csynth.xml','report_sha256')]:
            check(digest(work/filename)==result[key], 'Archived artifact mismatch: '+filename)
        check((work/'run_hls.tcl').read_text()==expected_tcl(m), 'Archived commands/tool flags mismatch')
        check((work/('candidate'+m['template_suffix'])).read_text()==base.materialize((bundle/'inputs/template.cpp').read_text(),m['initial_directives']), 'Archived candidate/source mismatch')
        log = (work/'synthesis.log').read_text(errors='replace')
        check('2021.1' in log and 'Finished Command csynth_design' in log and 'Exiting vitis_hls' in log and
              not re.search(r'^ERROR:',log,re.M), 'Incomplete/wrong toolchain archived log')
        metric = read_report(work/'hls_project/solution1/syn/report'/(m['top']+'_csynth.xml'),m)
        target = bundle/'projects/eval-01'
        target.parent.mkdir(exist_ok=True)
        shutil.copytree(str(work), str(target))
        result = enrich(copy.deepcopy(result), m, (bundle/'inputs/template.cpp').read_text(), log)
        result.update(metrics=metric, artifact_hashes=artifacts(target), work_dir='projects/eval-01')
        origin = dict(annotation='reuse: verified archived initial baseline; not a fresh invocation',
                      archive=str(archive), archive_manifest_sha256=digest(archive/'manifest.json'),
                      archive_state_sha256=digest(archive/'state.json'), frozen_experiment_signature=m['frozen_experiment_signature'],
                      source_matching=True, commands_matching=True)
        record = dict(signature=row['signature'], directives=m['initial_directives'],
                      selection=dict(family='initial_mailohls', kind='initial_mailohls'),
                      result=result, new_hls_call=False, reuse=origin)
        jdump(target/'result.json', result)
        state['evaluations'].append(record)
        jdump(bundle/'state.json',state)
        return origin


def counters(bundle, state):
    hist = state['evaluations']
    pending = jload(bundle/'pending.json') if (bundle/'pending.json').exists() else None
    uncommitted = pending and pending['index']>len(hist)
    return dict(total_evaluations_charged=len(hist)+int(bool(uncommitted)),
                actual_new_vitis_calls=sum(bool(r['new_hls_call']) and r['result']['tool_mode']=='VITIS_HLS' for r in hist)+
                    int(bool(uncommitted and pending.get('invocation_started') and not pending.get('synthetic'))),
                reused_verified_evaluations=sum(bool(r['reuse']) for r in hist),
                synthesis_failures=sum(r['result']['synthesis_status']!='success' for r in hist),
                pending_reservations=int(bool(pending)), pending_incomplete=bool(uncommitted))


def snapshot(row):
    if not row:
        return None
    return dict(evaluation=row['result']['index'], signature=row['signature'],
                directives=row['directives'], metrics=row['result']['metrics'], family=row['selection']['family'])


def summary(bundle):
    bundle = Path(bundle).expanduser().resolve()
    m = verify_bundle(bundle)
    state = jload(bundle/'state.json')
    hist = state['evaluations']
    winner = best_row(hist)
    initial = hist[0] if hist else None
    budget = []
    for n in range(1,5):
        r = best_row(hist[:n])
        budget.append(dict(budget=n, best=snapshot(r), carried_forward=n>len(hist)))
    improvement = (100*(1-winner['result']['metrics']['latency_best_cycles']/initial['result']['metrics']['latency_best_cycles'])
                   if winner and initial and initial['result'].get('metrics') else None)
    return dict(kernel=m['kernel'], protocol=SCHEMA, policy=m['policy'], source_sha256=m['input_hashes']['template.cpp'],
                frozen_experiment_signature=m['frozen_experiment_signature'], initial=snapshot(initial), selected=snapshot(winner),
                improvement_percent=improvement, incumbent_after_budget=budget,
                stall_summary=winner['result']['schedule_groups'] if winner else None,
                trajectory=[dict(evaluation=r['result']['index'], signature=r['signature'], family=r['selection']['family'],
                                 metrics=r['result'].get('metrics'), new_hls_call=r['new_hls_call'], reuse=r['reuse'],
                                 outcome=r.get('outcome'), error=r['result'].get('error')) for r in hist],
                stop_reason=preview(bundle)['stop_reason'], **counters(bundle,state))


def preview(bundle, enable_scout=False):
    bundle = Path(bundle).expanduser().resolve()
    m = verify_bundle(bundle)
    state = jload(bundle/'state.json')
    hist = state['evaluations']
    counts = counters(bundle,state)
    proposals, rejections = [], []
    reason = state.get('stop_reason')
    incumbent = best_row(hist)
    if not incumbent:
        # Measured infeasible context is only for recovery, never an incumbent.
        measured = [r for r in hist if r['result'].get('metrics') and r['result']['tool_mode']=='VITIS_HLS']
        incumbent = measured[0] if measured else None
    if counts['pending_reservations']:
        reason = 'Pending reservation blocks further calls; preserve charged slot for investigation'
    elif counts['total_evaluations_charged']>=4:
        reason = 'Four-evaluation budget exhausted'
    elif not reason and not hist:
        proposals = [dict(directives=m['initial_directives'], signature=signature(m['initial_directives']),
                          family='initial_mailohls', kind='initial_mailohls', changed_actions=[], changed={},
                          exploratory=False, reason='Frozen initial prediction requires one approved baseline synthesis', rank=1)]
    elif not reason and incumbent:
        proposals, rejections = rank_v3.propose(base,m,incumbent,hist,enable_scout)
        if not proposals:
            reason = 'Abstain: no sufficiently supported untried latency repair under this policy'
    elif not reason:
        reason = 'Abstain: no native synthesized measurement for diagnosis'
    return dict(kernel=m['kernel'], mode='scout' if enable_scout else 'strict', policy=m['policy'],
                source_sha256=m['input_hashes']['template.cpp'], source_matching='sealed inputs verified',
                baseline=snapshot(hist[0]) if hist else None, selected=snapshot(best_row(hist)),
                bottleneck_families=incumbent['result']['bottleneck_families'] if incumbent else [],
                schedule_groups=incumbent['result']['schedule_groups'] if incumbent else {},
                eligible_candidate_count=len(proposals), candidates=proposals[:3], rejections=rejections,
                stop_reason=reason, **counts,
                changed_from_v2=['all schedule variants retained; grouped warnings do not amplify rank',
                                 'verified enclosing unroll considered; explicit lane-loss proxy',
                                 'pipeline relocation disabled; no feasible area-only relief',
                                 'lexicographic evidence/leverage/tradeoff/risk/scope/signature ranking'])


def approval_token(bundle, preview_result):
    return 'RUN {} {} {}'.format(preview_result['kernel'], preview_result['total_evaluations_charged']+1,
                                preview_result['candidates'][0]['signature'])


def synthesize(bundle, m, cfg, index, executable, timeout, test_mock=False):
    """The only Vitis process boundary. Caller has locked, approved and reserved."""
    work = bundle/'projects'/('eval-%02d'%index)
    work.mkdir(parents=True,exist_ok=False)
    src = work/('candidate'+m['template_suffix'])
    src.write_text(base.materialize((bundle/'inputs/template.cpp').read_text(),cfg))
    for path in (bundle/'inputs').iterdir():
        if path.suffix in ('.h','.hpp','.inc') and path.name not in base.VENDOR_HEADERS:
            shutil.copy2(str(path),str(work/path.name))
    (work/'run_hls.tcl').write_text(expected_tcl(m))
    logpath = work/'synthesis.log'
    start = time.monotonic()
    error, metrics, started = None, None, False
    with logpath.open('w') as stream:
        try:
            proc = subprocess.Popen([str(executable),'-f','run_hls.tcl'],cwd=str(work),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            started = True
            reservation = jload(bundle/'pending.json')
            reservation.update(invocation_started=True,pid=proc.pid)
            jdump(bundle/'pending.json',reservation)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                rc, error = -9, 'timeout'
            except BaseException:
                try:
                    os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                raise
        except OSError as ex:
            rc, error = -1, 'launch_failed: '+str(ex)
            stream.write(error+'\n')
    report = work/'hls_project/solution1/syn/report'/(m['top']+'_csynth.xml')
    text = logpath.read_text(errors='replace')
    if rc==0 and report.is_file() and 'Finished Command csynth_design' in text and 'Exiting vitis_hls' in text:
        try:
            metrics = read_report(report,m)
            check('2021.1' in text or test_mock, 'Missing toolchain version in log')
            check(not re.search(r'^ERROR:',text,re.M), 'HLS error in log')
        except (RuntimeError,ValueError,ET.ParseError) as ex:
            metrics, error = None, 'untrusted_report: '+str(ex)
    elif not error:
        error = 'vitis_failed_or_no_complete_report, returncode='+str(rc)
    return dict(index=index,work_dir=str(work.relative_to(bundle)),metrics=metrics,error=error,
                synthesis_status='success' if metrics else 'failed',duration_seconds=round(time.monotonic()-start,3),
                invocation_started=started,source_sha256=digest(src),tcl_sha256=digest(work/'run_hls.tcl'),
                report_sha256=digest(report) if report.is_file() else None,log_sha256=digest(logpath),
                tool=str(executable),tool_mode='SYNTHETIC_TEST_ONLY' if test_mock else 'VITIS_HLS')


def run(args):
    bundle = Path(args.bundle).expanduser().resolve()
    if args.no_hls:
        return preview(bundle,args.enable_one_scout)
    with writer(bundle):
        m = verify_bundle(bundle)
        p = preview(bundle,args.enable_one_scout)
        if p['stop_reason']:
            return p
        selection = p['candidates'][0]
        emit(dict(proposed=selection, budget='{}/4 charged'.format(p['total_evaluations_charged']),
                  reused=p['reused_verified_evaluations'], source_sha256=p['source_sha256'],
                  caveat='Latency is uncertain; resource use may grow. HLS timing is not routed closure.'))
        token = approval_token(bundle,p)
        approved = args.approval
        if approved is None:
            try:
                approved = input('Approve exactly one call by typing '+token+'\n> ')
            except EOFError:
                approved = ''
        check(approved == token, 'Explicit fresh typed candidate approval required; bare yes is insufficient')
        exe = Path(args.vitis_hls).resolve()
        check(exe.is_file() and os.access(str(exe),os.X_OK), 'Missing executable before reservation')
        if args.test_mock:
            check(exe==(HERE/'tests/mock_vitis.py').resolve(), 'Mock mode accepts only the bundled synthetic fixture')
        else:
            check(str(exe)==VITIS, 'Pinned Vitis 2021.1 path required')
            check(all((exe.parent.parent/'include'/h).is_file() for h in ('ap_int.h','ap_fixed.h')), 'Genuine vendor headers missing')
            # Version is checked by report and log; no unapproved -version process.
        check(shutil.disk_usage(str(bundle)).free>=args.min_free_gb*1024**3, 'Insufficient disk before reservation')
        check(0 < args.timeout <= 7200, 'Timeout must be 1..7200 seconds')
        state = jload(bundle/'state.json')
        check(not any(r['signature']==selection['signature'] for r in state['evaluations']), 'Duplicate signature')
        index = p['total_evaluations_charged']+1
        reservation = dict(index=index, signature=selection['signature'], selection=selection,
                           approval=token, created_at_utc=time.time(), invocation_started=False, synthetic=args.test_mock)
        jdump(bundle/'pending.json',reservation)  # Irrevocable charge before entering synthesis.
        try:
            result = synthesize(bundle,m,selection['directives'],index,exe,args.timeout,args.test_mock)
            work = bundle/result['work_dir']
            log = (work/'synthesis.log').read_text(errors='replace')
            if result['metrics']:
                try:
                    result['metrics'] = read_report(work/'hls_project/solution1/syn/report'/(m['top']+'_csynth.xml'),m)
                    check('2021.1' in log or args.test_mock, 'Toolchain version missing in log')
                    check(not re.search(r'^ERROR:',log,re.M), 'HLS error in log')
                except (RuntimeError,ValueError,ET.ParseError) as error:
                    result.update(metrics=None, error=str(error), synthesis_status='failed')
            result = enrich(result,m,(bundle/'inputs/template.cpp').read_text(),log)
            # Only immutable source/Tcl/log/top report are part of the receipt inventory.
            result['artifact_hashes'] = {name:digest(work/name) for name in
                ['candidate'+m['template_suffix'],'run_hls.tcl','synthesis.log']+
                [p.name for p in work.iterdir() if p.suffix in ('.h','.hpp','.inc')]+
                (['hls_project/solution1/syn/report/'+m['top']+'_csynth.xml'] if result.get('report_sha256') else [])}
            old = best_row(state['evaluations'])
            outcome = 'failed' if not result['metrics'] else 'measured'
            if old and result['metrics']==old['result']['metrics'] and result['schedule_groups']==old['result']['schedule_groups']:
                outcome = 'hardware-insensitive / neutral'
            record = dict(signature=selection['signature'], directives=selection['directives'],
                          selection=selection, result=result, new_hls_call=result.get('invocation_started',False) and not args.test_mock, reuse=None, outcome=outcome)
            jdump(work/'result.json',result)
            state['evaluations'].append(record)
            jdump(bundle/'state.json',state)
            (bundle/'pending.json').unlink()  # Charged slot is now in durable history, never reclaimed.
        except BaseException:
            print('Interrupted: pending reservation remains charged; no automatic retry.',file=sys.stderr)
            raise
    return summary(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    for name in ('bundle','kernel','source-dir','template','kernel-info','graph','initial'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--policy',choices=['graph','feedback-only'],default='graph')
    p = sub.add_parser('import-baseline')
    p.add_argument('--bundle',required=True)
    p.add_argument('--archive',required=True)
    for name in ('preview','summary','run'):
        p = sub.add_parser(name)
        p.add_argument('--bundle',required=True)
        if name != 'summary':
            p.add_argument('--enable-one-scout',action='store_true')
        if name=='preview':
            p.add_argument('--explain',action='store_true',help='All previews include explanation; read-only')
        if name=='run':
            p.add_argument('--approval',help='Exact per-candidate token typed by operator, e.g. through runner')
            p.add_argument('--no-hls',action='store_true')
            p.add_argument('--vitis-hls',default=VITIS)
            p.add_argument('--timeout',type=int,default=7200)
            p.add_argument('--min-free-gb',type=float,default=20)
            p.add_argument('--test-mock',action='store_true',help='Synthetic fixtures only; cannot select as native incumbent')
    args = parser.parse_args()
    if args.command=='prepare':
        emit(dict(bundle=str(prepare(args)), actual_new_vitis_calls=0))
    elif args.command=='import-baseline':
        emit(import_baseline(args.bundle,args.archive))
    elif args.command=='preview':
        emit(preview(args.bundle,args.enable_one_scout))
    elif args.command=='summary':
        emit(summary(args.bundle))
    else:
        result = run(args)
        print('V3_RESULT')
        emit(result)


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,ValueError,OSError) as error:
        print('V3 ERROR: '+str(error),file=sys.stderr)
        sys.exit(2)
