"""Isolated Vitis synthesis, TCAD metrics and strict measurement authentication."""
from __future__ import annotations
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from . import configurations as c
from .configurations import ROOT, PART, CAPACITIES, CLOCK_NS, HLS_VERSION, SETTINGS, RESOURCE_KEYS, CONTEXT, sha256, object_hash, repo_path, display_path, read_jsonl, write_json, write_jsonl
parse_assignment = c.parse_assignment
def evaluate_metrics(latency_cycles, resources, capacities=None, budget=None):
    capacities = capacities or CAPACITIES
    budget = budget or capacities
    if set(capacities) != set(RESOURCE_KEYS) or set(budget) != set(RESOURCE_KEYS):
        raise ValueError('Capacities and budgets must specify all four resources')
    if any(
        (
            not math.isfinite(float(capacities[k])) or capacities[k] <= 0 or (not math.isfinite(float(budget[k]))) or (budget[k] < 0) for k in RESOURCE_KEYS
        )
    ):
        raise ValueError('Invalid capacities/budgets')
    missing = {
        'latency_cycles': None,
        'resources': resources,
        'normalized_utilization': None,
        'area_proxy': None,
        'ADP': None,
        'feasible': None,
        'rankable': False
    }
    if latency_cycles is None or resources is None or any((resources.get(k) is None for k in RESOURCE_KEYS)):
        return missing
    numbers = [latency_cycles, *(resources[k] for k in RESOURCE_KEYS)]
    if any(
        (
            isinstance(n, bool) or not isinstance(n, (int, float)) or (not math.isfinite(n)) or (n < 0) or (int(n) != n) for n in numbers
        )
    ) or latency_cycles <= 0:
        raise ValueError('QoR must be finite nonnegative integer counts, with positive latency')
    util = {k: 100.0 * resources[k] / capacities[k] for k in RESOURCE_KEYS}
    area = sum(util.values()) / 4.0
    feasible = all((resources[k] <= budget[k] for k in RESOURCE_KEYS))
    return {
        'latency_cycles': int(latency_cycles),
        'resources': dict(resources),
        'normalized_utilization': util,
        'area_proxy': area,
        'ADP': latency_cycles * area,
        'feasible': feasible,
        'rankable': feasible
    }

def best_feasible(results):
    valid = [
        r for r in results if r.get('synthesis_status') == 'success' and r.get('rankable') is True and (r.get('ADP') is not None)
    ]
    return min(valid, key=lambda r: (r['ADP'], r['candidate_id'])) if valid else None

def improvement_pct(baseline, candidate):
    if not baseline or not candidate or (not candidate.get('rankable')) or (baseline.get('ADP', 0) <= 0):
        return None
    return 100.0 * (baseline['ADP'] - candidate['ADP']) / baseline['ADP']

VENDOR_HEADERS = {'ap_fixed.h', 'ap_int.h'}

PH = re.compile('auto\\{(_(?:PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L[1-9][0-9]*)\\}', re.I)

def materialize(template, target):
    assignments = parse_assignment(target)
    for line in template.splitlines():
        if re.match('\\s*#\\s*pragma\\s+HLS\\s+(pipeline|unroll|array_partition)\\b', line, re.I) and (not PH.search(line)):
            raise ValueError('Ambiguous concrete optimization pragma outside the complete placeholder assignment')
    observed = ['AUTO{' + match[1].upper() + '}' for match in PH.finditer(template)]
    if set(observed) != set(assignments) or len(observed) != len(set(observed)):
        raise ValueError('Source/assignment placeholders are missing, extra, or duplicated')
    text = PH.sub(lambda match: assignments['AUTO{' + match[1].upper() + '}'], template)
    if 'auto{' in text.lower():
        raise ValueError('Unresolved source placeholder')
    output = []
    for line in text.splitlines(keepends=True):
        if re.match('\\s*#\\s*pragma\\s+HLS\\s+(?:pipeline\\s+II=0|unroll\\s+factor=0)\\s*$', line, re.I):
            continue
        if re.match('\\s*#\\s*pragma\\s+HLS\\s+array_partition\\b', line, re.I):
            if re.search('\\btype=none\\b', line, re.I):
                continue
            if re.search('\\btype=complete\\b', line, re.I):
                line = re.sub('\\s+factor=0\\b', '', line, flags=re.I)
        output.append(line)
    return ''.join(output)

def tcl_quote(value):
    value = str(value)
    if any((c in value for c in '{}\n\r')):
        raise ValueError('Unsafe Tcl path/value')
    return '{' + value + '}'

def make_tcl(source_name, top, part=PART, clock=CLOCK_NS, project="hls_project"):
    if not re.fullmatch('[A-Za-z_]\\w*', top) or not re.fullmatch('[A-Za-z0-9_]+', project) or part != PART or float(clock) != CLOCK_NS:
        raise ValueError('Invalid top or unsupported deployment context')
    return '\n'.join(
        [
            'open_project -reset ' + project,
            'set_top ' + top,
            'add_files ' + tcl_quote(source_name),
            'open_solution -reset solution1 -flow_target vivado',
            'set_part ' + tcl_quote(part),
            f'create_clock -period {float(clock):g} -name default',
            *SETTINGS,
            'csynth_design',
            'exit',
            ''
        ]
    )

class ReportError(ValueError):
    pass

def parse_report(path, top, part=PART, clock=CLOCK_NS, version=HLS_VERSION):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ReportError(f'Unreadable/partial XML: {exc}') from exc

    def text(query):
        return root.findtext(query)
    checks = {
        './/UserAssignments/Part': part,
        './/UserAssignments/TopModelName': top,
        './/ReportVersion/Version': version,
        './/UserAssignments/FlowTarget': 'vivado'
    }
    for (query, expected) in checks.items():
        if text(query) != expected:
            raise ReportError(f'Report provenance mismatch: {query}={text(query)!r}, expected {expected!r}')
    if text('.//UserAssignments/unit') != 'ns' or text('.//SummaryOfOverallLatency/unit') != 'clock cycles':
        raise ReportError('Report timing units do not match nanoseconds / clock cycles')
    try:
        actual_clock = float(text('.//UserAssignments/TargetClockPeriod'))
        if not math.isfinite(actual_clock) or abs(actual_clock - clock) > 1e-06:
            raise ValueError('changed target clock')
    except (ValueError, TypeError) as exc:
        raise ReportError('Missing/wrong requested clock') from exc

    def count(query):
        raw = text(query)
        if raw is None or not re.fullmatch('[0-9]+', raw.strip()):
            raise ReportError(f'Missing/invalid count: {query}={raw!r}')
        return int(raw)
    latency = count('.//SummaryOfOverallLatency/Worst-caseLatency')
    if latency <= 0:
        raise ReportError('Nonpositive worst-case latency')
    resources = {
        name: count('.//AreaEstimates/Resources/' + name) for name in ('BRAM_18K', 'DSP', 'FF', 'LUT')
    }
    estimated = text('.//SummaryOfTimingAnalysis/EstimatedClockPeriod')
    try:
        estimated = float(estimated)
        if not math.isfinite(estimated) or estimated <= 0:
            estimated = None
    except (ValueError, TypeError):
        estimated = None
    return {
        **evaluate_metrics(latency, resources),
        'estimated_clock_ns': estimated,
        'timing_status': 'unknown' if estimated is None else 'HLS estimate only; not post-route timing',
        'estimated_clock_le_requested': None if estimated is None else estimated <= clock,
        'requested_clock_ns': actual_clock,
        'part': part,
        'top': top,
        'hls_version': version
    }

def tool_info(executable):
    path = shutil.which(str(executable))
    if path is None:
        raise FileNotFoundError(
            f'Vitis HLS not found: {executable}. Preparation/summarization need no HLS; run on a Vitis HLS 2021.1 host with --vitis-hls /path/to/vitis_hls.'
        )
    outputs = []
    for flag in ('-version', '--version'):
        result = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        outputs.append(output)
        if result.returncode == 0 and re.search('(?:Vitis|Vivado|Version).*?\\b(?:v)?2021\\.1\\b', output, re.I):
            headers = {}
            for parent in list(Path(path).resolve().parents)[:4]:
                for name in VENDOR_HEADERS:
                    header = parent / 'include' / name
                    if header.is_file():
                        payload = header.read_text(errors='replace')
                        if 'Xilinx' not in payload and 'Advanced Micro Devices' not in payload or 'AP_FIXED_STUB_H' in payload or 'AP_INT_STUB_H' in payload:
                            continue
                        headers.setdefault(name, {'path': str(header), 'sha256': sha256(header)})
            return {
                'executable': str(Path(path).resolve()),
                'version': HLS_VERSION,
                'executable_sha256': sha256(path),
                'version_output': output.strip(),
                'vendor_headers': headers
            }
    raise ValueError('The executable did not identify as Vitis HLS 2021.1: ' + '\n'.join(outputs))

def verify_source(kernel):
    relation = kernel.get('relations', {})
    if relation.get('status') == 'available':
        if sha256(repo_path(relation['artifact'])) != relation['artifact_hash']:
            raise ValueError('Compiler relation artifact changed since preparation')
        path = relation.get('compiler_active_metadata_path')
        if path and sha256(repo_path(path)) != relation['compiler_active_metadata_sha256']:
            raise ValueError('Compiler-active metadata changed since preparation')
    if sha256(repo_path(kernel['template_path'])) != kernel['template_sha256']:
        raise ValueError('Prepared source template changed')
    for entry in kernel['source_provenance']['files']:
        path = repo_path(entry['snapshot'])
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f'Source snapshot changed since preparation: {path}')

def run_candidate(candidate, kernel, output, tool, timeout=21600):
    verify_source(kernel)
    root = Path(output) / 'projects' / candidate['candidate_id']
    number = 1
    while (root / f'attempt-{number:03}').exists():
        number += 1
    work = root / f'attempt-{number:03}'
    work.mkdir(parents=True, exist_ok=False)
    source_name = Path(kernel['source_provenance']['source']).name
    template = repo_path(kernel['template_path']).read_text()
    source_text = materialize(template, candidate['target'])
    (work / source_name).write_text(source_text)
    analysis_headers = []
    for entry in kernel['source_provenance']['files']:
        path = repo_path(entry['snapshot'])
        if path.suffix.lower() not in {'.h', '.hpp', '.hh', '.inc'}:
            continue
        if path.name in VENDOR_HEADERS:
            header = tool.get('vendor_headers', {}).get(path.name)
            if not header or sha256(header['path']) != header['sha256']:
                raise ValueError(
                    f'Cannot verify genuine Vitis header {path.name}; analysis stubs are forbidden for synthesis'
                )
            analysis_headers.append(path.name)
            continue
        destination = work / path.relative_to(repo_path(kernel['source_provenance']['snapshot_source']).parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    tcl = make_tcl(source_name, kernel['top'])
    if analysis_headers:
        include_dirs = sorted({str(Path(tool['vendor_headers'][name]['path']).parent) for name in analysis_headers})
        flags = ' '.join(('-I"' + directory + '"' for directory in include_dirs))
        tcl = tcl.replace(
            'add_files ' + tcl_quote(source_name),
            'add_files -cflags ' + tcl_quote(flags) + ' ' + tcl_quote(source_name)
        )
    (work / 'run_hls.tcl').write_text(tcl)
    receipt = {
        'project_name': 'hls_project',
        'candidate_id': candidate['candidate_id'],
        'canonical_signature': candidate['canonical_signature'],
        'evaluation_context_hash': candidate['evaluation_context_hash'],
        'materialized_source_sha256': sha256(work / source_name),
        'tcl_sha256': sha256(work / 'run_hls.tcl'),
        'tool': tool,
        'analysis_stubs_not_copied': analysis_headers,
        'header_policy': 'Vitis supplies ap_fixed.h/ap_int.h; remaining source headers copied exactly'
    }
    write_json(work / 'receipt.json', receipt)
    log = work / 'synthesis.log'
    report = work / 'hls_project/solution1/syn/report' / (kernel['top'] + '_csynth.xml')
    assert not report.exists()
    start = time.monotonic()
    (status, error, returncode) = ('failed', None, None)
    metrics = {**evaluate_metrics(None, None), 'estimated_clock_ns': None, 'timing_status': 'unknown'}
    with log.open('w') as stream:
        process = subprocess.Popen(
            [tool['executable'], '-f', 'run_hls.tcl'],
            cwd=work,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            (status, error) = ('timeout', f'Exceeded {timeout} seconds')
    runtime = time.monotonic() - start
    if status != 'timeout':
        if returncode != 0:
            error = f'Vitis exited with code {returncode}; report not ranked even if present'
        elif not report.is_file():
            (status, error) = ('missing_report', 'Vitis returned without the top-level csynth report')
        else:
            try:
                metrics = parse_report(report, kernel['top'])
                status = 'success'
            except ReportError as exc:
                (status, error) = ('invalid_report', str(exc))
    result = {
        'candidate_id': candidate['candidate_id'],
        'kernel': candidate['kernel'],
        'candidate_group': candidate['candidate_group'],
        'synthesis_status': status,
        'evaluation_context_hash': candidate['evaluation_context_hash'],
        'tool_hash': object_hash(tool),
        'attempt_count': number,
        'hls_runtime_seconds': runtime,
        'report_path': display_path(report),
        'log_path': display_path(log),
        'receipt_path': display_path(work / 'receipt.json'),
        'returncode': returncode,
        'error': error,
        'requested_clock_ns': CLOCK_NS,
        'part': PART,
        'top': kernel['top'],
        'report_sha256': sha256(report) if report.is_file() else None,
        **metrics
    }
    if status != 'success':
        result['rankable'] = False
    write_json(work / 'result.json', result)
    return result

def tool_identity(tool):
    if tool.get('version') != '2021.1' or not tool.get('executable_sha256'):
        raise ValueError('Unverified HLS version/executable')
    headers = tool.get('vendor_headers', {})
    if any((not headers.get(name, {}).get('sha256') for name in VENDOR_HEADERS)):
        raise ValueError('Missing vendor-header provenance')
    return {
        'version': tool['version'],
        'executable_sha256': tool['executable_sha256'],
        'vendor_headers': {name: headers[name]['sha256'] for name in sorted(headers)}
    }

def source_context(kernel):
    prov = kernel['source_provenance']
    source_parent = Path(prov['snapshot_source']).parent
    headers = {
        str(Path(f['snapshot']).relative_to(source_parent)): f['sha256'] for f in prov['files'] if Path(f['snapshot']).suffix.lower() in {'.h', '.hpp', '.hh', '.inc'}
    }
    return {
        'deployment': CONTEXT,
        'top': kernel['top'],
        'source_sha256': prov['source_sha256'],
        'metadata_sha256': prov['metadata_sha256'],
        'template_sha256': kernel['template_sha256'],
        'snapshot_headers': headers,
        'header_policy': 'genuine verified vendor AP headers; other headers copied exactly'
    }

def comparison_id(kernel, tool):
    return object_hash({'source_context': source_context(kernel), 'tool': tool_identity(tool)})

def expected_tcl(source_name, top, files, tool, project="hls_project"):
    text = make_tcl(source_name, top, project=project)
    required = {Path(f['snapshot']).name for f in files if Path(f['snapshot']).name in VENDOR_HEADERS}
    if required:
        paths = sorted({str(Path(tool['vendor_headers'][name]['path']).parent) for name in required})
        flags = ' '.join(('-I"' + path + '"' for path in paths))
        text = text.replace(
            'add_files ' + tcl_quote(source_name),
            'add_files -cflags ' + tcl_quote(flags) + ' ' + tcl_quote(source_name)
        )
    return text

def authenticate_work(candidate, result, kernel, resolve, *, expected_evaluation_hash):
    """Reparse actual XML and verify source, headers, receipts, Tcl and completion."""
    if result.get('synthesis_status') != 'success' or result.get('returncode') != 0:
        raise ValueError('Synthesis did not successfully complete')
    signature = c.signature(candidate['assignment'])
    if signature != candidate['canonical_signature'] or c.parse_assignment(candidate['target']) != candidate['assignment']:
        raise ValueError('Candidate signature/serialization mismatch')
    if result.get('evaluation_context_hash') != expected_evaluation_hash:
        raise ValueError('Evaluation context hash mismatch')
    for f in kernel['source_provenance']['files']:
        if sha256(resolve(f['snapshot'])) != f['sha256']:
            raise ValueError('Source/metadata/header snapshot hash mismatch')
    prov = kernel['source_provenance']
    if sha256(resolve(prov['snapshot_source'])) != prov['source_sha256'] or sha256(resolve(prov['snapshot_metadata'])) != prov['metadata_sha256']:
        raise ValueError('Source or metadata identity mismatch')
    template_path = resolve(kernel['template_path'])
    if sha256(template_path) != kernel['template_sha256']:
        raise ValueError('Source template hash mismatch')
    (receipt_path, report_path, log_path) = (resolve(result[field]) for field in ('receipt_path', 'report_path', 'log_path'))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('candidate_id') != candidate['candidate_id'] or receipt.get('canonical_signature') != signature or receipt.get('evaluation_context_hash') != expected_evaluation_hash:
        raise ValueError('Receipt identity mismatch')
    tool = receipt['tool']
    identity = tool_identity(tool)
    if result.get('tool_hash') != object_hash(tool):
        raise ValueError('Tool receipt hash mismatch')
    work = receipt_path.parent
    source_name = Path(prov['source']).name
    source = work / source_name
    if sha256(source) != receipt['materialized_source_sha256'] or source.read_text() != materialize(template_path.read_text(), candidate['target']):
        raise ValueError('Materialized source differs from exact complete assignment')
    actual_tcl = (work / 'run_hls.tcl').read_text()
    match = re.fullmatch(r'open_project -reset ([A-Za-z0-9_]+)', actual_tcl.splitlines()[0])
    if not match or receipt.get('project_name', match[1]) != match[1]:
        raise ValueError('Invalid synthesis project receipt')
    if sha256(work / 'run_hls.tcl') != receipt['tcl_sha256'] or (work / 'run_hls.tcl').read_text() != expected_tcl(source_name, kernel['top'], prov['files'], tool, project=match[1]):
        raise ValueError('Tcl synthesis settings differ')
    required = set()
    for entry in prov['files']:
        snapshot = Path(entry['snapshot'])
        if snapshot.suffix.lower() not in {'.h', '.hpp', '.hh', '.inc'}:
            continue
        relative = snapshot.relative_to(Path(prov['snapshot_source']).parent)
        if snapshot.name in VENDOR_HEADERS:
            required.add(snapshot.name)
            if (work / relative).exists():
                raise ValueError('Analysis AP stub was copied into synthesis work')
        elif sha256(work / relative) != entry['sha256']:
            raise ValueError('Synthesis dependency header mismatch')
    if required != set(receipt.get('analysis_stubs_not_copied', [])):
        raise ValueError('Unverified vendor header substitution')
    log = log_path.read_text(errors='replace')
    if 'Finished Command csynth_design' not in log or 'Exiting vitis_hls' not in log:
        raise ValueError('Missing completed synthesis log evidence')
    if sha256(report_path) != result.get('report_sha256'):
        raise ValueError('Report integrity mismatch')
    parsed = parse_report(report_path, kernel['top'])
    xml = ET.parse(report_path).getroot()
    available = {
        key: int(xml.findtext('.//AreaEstimates/AvailableResources/' + key)) for key in CONTEXT['device_capacities']
    }
    if available != CONTEXT['device_capacities']:
        raise ValueError('Report device capacities differ')
    for key in (
        'latency_cycles',
        'resources',
        'ADP',
        'area_proxy',
        'normalized_utilization',
        'feasible',
        'rankable',
        'part',
        'requested_clock_ns',
        'top',
        'hls_version'
    ):
        if result.get(key) != parsed[key]:
            raise ValueError(f'Stored QoR/provenance differs from actual XML: {key}')
    return {
        **parsed,
        'tool_identity': identity,
        'comparison_context_id': comparison_id(kernel, tool),
        'evidence': {
            str(p): sha256(p) for p in (receipt_path, report_path, log_path, source, work / 'run_hls.tcl')
        }
    }
