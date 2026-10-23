#!/usr/bin/env python3
"""Sequential, resumable implementation of the validated target experiment manifest.

Defaults target coroni. Preview by default; --run executes the pending entries.
Uses the existing export_impl.tcl, opening each project by its physical metadata name.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def status_read(path):
    if not path.exists():
        return {}
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def successful(path, report):
    data = status_read(path)
    return (data.get('status') == 'impl_ok' and data.get('exit_code') == '0'
            and data.get('implementation_report') == str(report)
            and report.is_file() and report.stat().st_size > 0)


def write_status(path, data):
    temp = path.with_suffix('.status.tmp')
    temp.write_text(''.join('{}={}\n'.format(k, v) for k, v in data.items()))
    temp.replace(path)


def utc():
    return datetime.datetime.utcnow().isoformat() + 'Z'


def prepare(hls, impl, experiments):
    rows = [json.loads(line) for line in experiments.read_text().splitlines() if line.strip()]
    projects = json.loads((hls / 'project_manifest.json').read_text())['projects']
    if len(rows) != 12 or len(projects) != 12:
        raise ValueError('Expected exactly 12 experiment rows and 12 manifest entries')
    entries = []
    seen = set()
    for number, row in enumerate(rows, 1):
        matches = [p for p in projects if p['row'] == number and p['experiment'] == row]
        if len(matches) != 1:
            raise ValueError('Manifest does not exactly match experiment row {}'.format(number))
        p = matches[0]
        project = hls / p['name']
        physical = project.resolve(strict=True)
        if physical != Path(p['target']).resolve(strict=True):
            raise ValueError('Manifest target mismatch: ' + str(project))
        app = ET.parse(str(physical / 'hls.app')).getroot()
        # Avoid the alias/physical-name mismatch that broke the first GAN export.
        export_project = physical.parent / app.attrib['name']
        if export_project.resolve(strict=True) != physical:
            raise ValueError('Metadata project path mismatch: ' + str(export_project))
        solution = project / p['solution']
        report = solution / 'syn/report' / (app.attrib['top'] + '_csynth.rpt')
        xml = ET.parse(str(report.with_suffix('.xml'))).getroot()
        part = {'ZCU104': 'xczu7ev-ffvc1156-2-e', 'U50': 'xcu50-fsvh2104-2-e'}[row['device']]
        if xml.findtext('.//Part') != part or abs(float(xml.findtext('.//TargetClockPeriod')) - row['clock_period_ns']) > 0.001:
            raise ValueError('Synthesis target/clock mismatch: ' + str(report))
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError('Missing synthesis report: ' + str(report))
        if not (solution / (p['solution'] + '.aps')).is_file():
            raise ValueError('Missing solution metadata: ' + str(solution))
        if str(solution.resolve()) in seen:
            raise ValueError('Duplicate physical solution: ' + str(solution))
        seen.add(str(solution.resolve()))
        case = '{}__{}__{}'.format(p['name'], p['solution'], hashlib.sha256(str(solution).encode()).hexdigest()[:10])
        entries.append(dict(row=number, case=case, project=project, solution=p['solution'],
                            export_project=export_project, report=report,
                            impl_report=solution / 'impl/report/verilog/export_impl.rpt',
                            status=impl / 'logs' / (case + '.status')))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute; default is a read-only preview')
    parser.add_argument('--hls-root', type=Path, default=Path(os.environ.get('TG_HLS_ROOT', '/home/bonus24/mailohls_target_generalization_hls')))
    parser.add_argument('--impl-root', type=Path, default=Path(os.environ.get('TG_IMPL_ROOT', '/home/bonus24/mailohls_target_generalization_impl')))
    parser.add_argument('--experiments', type=Path)
    args = parser.parse_args()
    impl = args.impl_root.absolute()
    entries = prepare(args.hls_root.absolute(), impl, args.experiments or impl / 'Target_general_experiments.queue.jsonl')
    pending = [e for e in entries if not successful(e['status'], e['impl_report'])]
    for e in entries:
        print('{:02d} {} {}'.format(e['row'], 'RUN ' if e in pending else 'SKIP', e['project'].name), flush=True)
    print('{} total; {} completed; {} pending'.format(len(entries), len(entries)-len(pending), len(pending)), flush=True)
    if not args.run:
        return 0
    timeout = int(os.environ.get('IMPL_TIMEOUT_SEC', '21600'))
    if timeout <= 0:
        raise ValueError('IMPL_TIMEOUT_SEC must be positive')
    for command in ('vitis_hls', 'timeout', '/usr/bin/time'):
        if not shutil.which(command):
            raise ValueError('Command unavailable: ' + command + '; source the Xilinx settings first')
    tcl = impl / 'scripts/export_impl.tcl'
    if not tcl.is_file():
        raise ValueError('Missing export Tcl: ' + str(tcl))
    (impl / 'logs').mkdir(exist_ok=True)
    lock = (impl / 'queue.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('Another implementation queue is running')
    failures = 0
    for e in entries:
        if successful(e['status'], e['impl_report']):
            continue
        stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')
        log = impl / 'logs' / (e['case'] + '.' + stamp + '.log')
        if e['status'].exists():
            shutil.copy2(str(e['status']), str(e['status']) + '.' + stamp + '.bak')
        data = dict(case_id=e['case'], project=e['project'], solution=e['solution'],
                    csynth_report=e['report'], start_utc=utc(), status='impl_running', log_file=log)
        write_status(e['status'], data)
        print('{} START {} log={}'.format(utc(), e['project'].name, log), flush=True)
        env = dict(os.environ, MAILOHLS_PROJECT=str(e['export_project']), MAILOHLS_SOLUTION=e['solution'])
        command = ['/usr/bin/time', '-v', 'timeout', '--signal=TERM', '--kill-after=300', str(timeout),
                   'vitis_hls', '-f', str(tcl), '-tclargs', str(e['export_project']), e['solution']]
        previous_mtime = e['impl_report'].stat().st_mtime_ns if e['impl_report'].is_file() else None
        with log.open('w') as output:
            # Pass the queue lock to the child so it also guards an orphaned running job.
            rc = subprocess.call(command, env=env, cwd=str(impl), stdout=output,
                                 stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
        valid_report = (e['impl_report'].is_file() and e['impl_report'].stat().st_size > 0
                        and e['impl_report'].stat().st_mtime_ns != previous_mtime)
        if rc == 0 and not valid_report:
            rc = 1
        outcome = 'impl_ok' if rc == 0 else ('impl_timeout' if rc in (124, 137) else 'impl_failed')
        data.update(end_utc=utc(), exit_code=rc, status=outcome,
                    implementation_report=str(e['impl_report']) if valid_report else '')
        write_status(e['status'], data)
        failures += rc != 0
        print('{} END {} {} exit_code={}'.format(utc(), e['project'].name, outcome, rc), flush=True)
    print('Queue finished; {} failed/timed out in this run.'.format(failures), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, ET.ParseError) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        sys.exit(2)
