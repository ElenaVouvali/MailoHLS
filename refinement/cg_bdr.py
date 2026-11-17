#!/usr/bin/env python3
"""CG-BDR: synthesis-feedback, action-grounded prototype (Python 3.8+).

Research prototype: no model weights, no learned QoR oracle, no arbitrary search.
Synthesis authority is Vitis HLS. Graph anchors and source bindings are *not*
proof of LLVM alias/dependence; this tool only implements a conservative subset.
"""
import argparse
import copy
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

PART = 'xczu7ev-ffvc1156-2-e'
CAPACITIES = {'bram': 624, 'dsp': 1728, 'ff': 460800, 'lut': 230400}
SETTINGS = ('config_array_partition -complete_threshold 0 -throughput_driven off',
            'config_compile -pipeline_loops 0')
VENDOR_HEADERS = {'ap_int.h', 'ap_fixed.h', 'hls_stream.h', 'ap_axi_sdata.h'}
PH = re.compile(r'auto\{_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_(L\d+)\}')
ANCHOR = re.compile(r'/\*L(\d+):\*/')
REF = re.compile(r'\b(L\d+)\b')


def die(s):
    raise RuntimeError(s)


def jdump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(temp), str(p))


def jload(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def natural(label):
    return int(label[1:])


def parse_info(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        die('Empty kernel_info.txt')
    actions = {}
    for line in lines[1:]:
        cols = [x.strip() for x in line.split(',')]
        if len(cols) < 3 or not re.fullmatch(r'L[1-9]\d*', cols[0]):
            die('Invalid kernel-info action row: ' + line)
        label, kind = cols[0], cols[1]
        if label in actions:
            die('Repeated action ' + label)
        if kind == 'loop':
            actions[label] = {'kind': 'loop', 'bound': int(cols[2])}
        elif kind == 'array':
            if len(cols) < 5 or len(cols[3:]) % 2:
                die('Invalid array dimensions: ' + line)
            dims = {int(cols[i]): int(cols[i+1]) for i in range(3, len(cols), 2)}
            actions[label] = {'kind': 'array', 'variable': cols[2], 'dims': dims}
        else:
            die('Unknown action kind: ' + kind)
    return lines[0], actions


def factors(n):
    return sorted({1, *[2**p for p in range(1, 20) if 2**p <= n],
                   *[i for i in range(2, n + 1) if n % i == 0]})


def validate(cfg, actions):
    if not isinstance(cfg, list) or len(cfg) != len(actions):
        die('Configuration must contain exactly one row per Lk')
    by = {}
    for row in cfg:
        lid = row.get('action_id')
        if lid in by or lid not in actions:
            die('Duplicated or unknown action: ' + str(lid))
        action = actions[lid]
        if row.get('kind') != action['kind']:
            die('Action kind mismatch: ' + lid)
        if action['kind'] == 'loop':
            p, u = row.get('pipeline_ii'), row.get('unroll_factor')
            if type(p) is not int or p not in (0, 1) or type(u) is not int or u < 0:
                die('Unsupported loop pragma tuple at ' + lid)
            # User-provided original values are preserved; new factors are chosen from factors(n).
            if u > action['bound']:
                die('Unroll exceeds recorded metadata bound at ' + lid)
        else:
            typ, fac, dim = (row.get('partition_type'), row.get('partition_factor'), row.get('partition_dim'))
            if row.get('variable') != action['variable'] or typ not in ('none', 'complete', 'block', 'cyclic'):
                die('Unsupported array tuple at ' + lid)
            if type(fac) is not int or type(dim) is not int:
                die('Invalid partition factor/dimension at ' + lid)
            if typ == 'none' and (fac != 0 or dim != 0):
                die('none partition must have factor=0, dim=0 at ' + lid)
            if typ != 'none' and dim not in action['dims']:
                die('Unknown partition dimension at ' + lid)
            if typ == 'complete' and fac != 0:
                die('complete partition factor must be zero at ' + lid)
            if typ in ('block', 'cyclic') and not (2 <= fac <= action['dims'][dim]):
                die('Illegal partition factor at ' + lid)
        by[lid] = dict(row)
    if set(by) != set(actions):
        die('Action coverage mismatch')
    return [by[k] for k in sorted(by, key=natural)]


def materialize(template, cfg, label_loops=True):
    """Keep source line count stable, for diagnostic line attribution."""
    by = {a['action_id']: a for a in cfg}
    out = []
    for line in template.splitlines(keepends=True):
        found = PH.findall(line)
        if not found:
            out.append(line)
            continue
        labs = {lid for _, lid in found}
        if len(labs) != 1:
            die('A template pragma line spans more than one action')
        lid = next(iter(labs))
        if lid not in by:
            die('Placeholder not present in configuration: ' + lid)
        obj = by[lid]
        if re.search(r'#pragma\s+HLS\s+pipeline\b', line, re.I):
            if obj['kind'] != 'loop':
                die('Loop/array placeholder mismatch')
            if obj['pipeline_ii'] == 0:
                out.append('\n')
            else:
                out.append(PH.sub(lambda m: str(obj['pipeline_ii']), line))
        elif re.search(r'#pragma\s+HLS\s+unroll\b', line, re.I):
            if obj['kind'] != 'loop':
                die('Loop/array placeholder mismatch')
            if obj['unroll_factor'] in (0, 1):
                out.append('\n')
            else:
                out.append(PH.sub(lambda m: str(obj['unroll_factor']), line))
        elif re.search(r'#pragma\s+HLS\s+array_partition\b', line, re.I):
            if obj['kind'] != 'array':
                die('Loop/array placeholder mismatch')
            if obj['partition_type'] == 'none':
                out.append('\n')
            else:
                replacements = {'ARRAY_T': obj['partition_type'],
                                'ARRAY_F': str(obj['partition_factor']),
                                'ARRAY_D': str(obj['partition_dim'])}
                fixed = PH.sub(lambda m: replacements[m.group(1)], line)
                if obj['partition_type'] == 'complete':
                    fixed = re.sub(r'\s+factor=0\b', '', fixed)
                out.append(fixed)
        else:
            die('Unknown pragma placeholder context: ' + line)
    text = ''.join(out)
    if PH.search(text):
        die('Unmaterialized pragma placeholder')
    if label_loops:
        # Only the current graph/source Lk loop anchors; no array declaration labels.
        def repl(m):
            label = 'L' + m.group(1)
            if label not in by or by[label]['kind'] != 'loop':
                return m.group(0)
            return label + ':'
        text = ANCHOR.sub(repl, text)
    if len(text.splitlines()) != len(template.splitlines()):
        die('Line-number invariance broken')
    return text


def graph_index(path, actions):
    """Verify Lk directive anchors and array-scope -> LLVM access linkage.

    Uses the actual flow=200 pragma edges and flow=7 array-scope edges from the
    original graph generator. A relationship is *supported*, not a proven alias,
    dependence or bank-conflict relation. Unknown evidence blocks an edit.
    """
    try:
        import networkx as nx
    except ImportError:
        die('Install networkx on liono for preparation (Coroni runner needs only stdlib)')
    g = nx.read_gexf(path)
    found = {x: [] for x in actions}
    functions = {x: set() for x in actions}
    scopes = {}
    for nid, d in g.nodes(data=True):
        typ = str(d.get('type', ''))
        text = str(d.get('full_text', d.get('features', '')))
        labels = set(re.findall(r'_L(\d+)\}', text))
        if typ == '100' or d.get('text') in ('PIPELINE', 'UNROLL', 'ARRAY_PARTITION'):
            for x in labels:
                lid = 'L' + x
                if lid in found:
                    found[lid].append(str(nid))
                    functions[lid].add(str(d.get('function', '')))
        if typ == '104' or d.get('text') == 'array_scope':
            var = d.get('array_var')
            if var:
                scopes.setdefault(str(var), []).append(str(nid))
    array_links = {}
    for lid, a in actions.items():
        if a['kind'] != 'array':
            continue
        var = a['variable']
        evidence = []
        for anchor in found[lid]:
            if anchor not in g:
                continue
            for _, scope, edge in g.out_edges(anchor, data=True):
                if str(edge.get('flow')) != '200' or str(g.nodes[scope].get('type')) != '104':
                    continue
                if g.nodes[scope].get('array_var') != var:
                    continue
                for _, operation, link in g.out_edges(scope, data=True):
                    if str(link.get('flow')) != '7':
                        continue
                    opnode = g.nodes[operation]
                    fulltext = str(opnode.get('full_text', opnode.get('features', '')))
                    if (re.search(r'\b(load|store|getelementptr|alloca)\b', fulltext) and
                        re.search(r'(?<![A-Za-z0-9_])%?'+re.escape(var)+r'(?![A-Za-z0-9_])', fulltext)):
                        evidence.append({'scope_node': str(scope), 'operation_node': str(operation),
                                         'function': str(opnode.get('function', '')),
                                         'ir': fulltext[:190]})
        array_links[lid] = evidence[:16]
    matched = [k for k, v in found.items() if v]
    return {'path_name': Path(path).name, 'sha256': digest(path), 'nodes': g.number_of_nodes(),
            'edges': g.number_of_edges(), 'anchored_actions': matched,
            'action_nodes': {k: v for k, v in found.items() if v},
            'action_functions': {k: sorted(v) for k, v in functions.items()},
            'array_scope_access_links': array_links,
            'array_scope_nodes': scopes,
            'missing_actions': sorted(set(actions) - set(matched), key=natural),
            'meaning': 'Real ProGraML pragma->array_scope->IR access links. This does not prove loop dependence, memory bank conflicts or alias freedom.'}


def loop_spans(template, actions):
    """Discover labeled for-loop braced spans. Conservative textual fallback.

    This is NOT LLVM LoopInfo. Label/brace correspondence is treated as source
    attribution and must be coupled to graph anchor existence.
    """
    lines = template.splitlines()
    spans = {}
    for idx, line in enumerate(lines):
        m = ANCHOR.search(line)
        if not m:
            continue
        lid = 'L' + m.group(1)
        if lid not in actions or actions[lid]['kind'] != 'loop':
            continue
        tail = '\n'.join(lines[idx:idx+4])
        head = re.search(r'for\s*\(\s*(?:int\s+)?([A-Za-z_]\w*)\s*=', tail)
        if not head:
            continue
        iterator = head.group(1)
        # IMPORTANT: directive placeholders have braces (auto{_PIPE_Lk}),
        # which must NEVER be mistaken for a C/C++ loop body. Skip preprocessor
        # lines, then require a real brace on the for line or a standalone
        # opening brace on the next nonblank code line. Unbraced loops abstain.
        brace_line = None
        for j in range(idx, min(idx + 12, len(lines))):
            candidate = lines[j].strip()
            if not candidate or candidate.startswith('#') or candidate.startswith('//'):
                continue
            if j == idx:
                for_pos = candidate.find('for')
                brace_pos = candidate.find('{', for_pos + 3) if for_pos >= 0 else -1
                if brace_pos >= 0:
                    brace_line = j
                    break
                continue
            if candidate.startswith('{'):
                brace_line = j
                break
            # Any other intervening statement means the loop is unbraced or
            # the source form is not supported by this conservative parser.
            break
        if brace_line is None:
            continue
        nesting, end = 0, None
        for j in range(brace_line, len(lines)):
            t = lines[j].strip()
            if t.startswith('#') or t.startswith('//'):
                continue  # pragma placeholders can contain balanced braces
            # Count only syntactic braces in ordinary code, not placeholders.
            nesting += t.count('{') - t.count('}')
            if nesting == 0:
                end = j + 1
                break
        if end is None:
            continue
        spans[lid] = {'start': idx + 1, 'end': end, 'iterator': iterator}
    return spans


def direct_accesses(template, actions, spans, graph):
    lines = template.splitlines()
    anchored = set(graph['anchored_actions'])
    functions = graph['action_functions']
    array_links = graph['array_scope_access_links']
    relations = []
    for lid, loop in spans.items():
        if lid not in anchored:
            continue
        for a_label, a in actions.items():
            if a['kind'] != 'array' or a_label not in anchored:
                continue
            # The graph must connect the array pragma through an explicit
            # array_scope node to a representative LLVM IR access/declaration,
            # and its action scopes must agree on the owning function.
            lf = set(functions.get(lid, [])) - {''}
            af = set(functions.get(a_label, [])) - {''}
            if not (lf and af and (lf & af)):
                continue
            ir_accesses = [x for x in array_links.get(a_label, []) if x['function'] in lf]
            if not ir_accesses:
                continue
            var = a['variable']
            regex = re.compile(r'\b' + re.escape(var) + r'\s*\[([^\]]+)\]')
            rows = []
            for lineno in range(loop['start']+1, loop['end']+1):
                if lineno > len(lines):
                    break
                for match in regex.finditer(lines[lineno-1]):
                    index = match.group(1).replace(' ', '')
                    it = loop['iterator']
                    consecutive = bool(re.fullmatch(re.escape(it) + r'(?:[+-]\d+)?', index))
                    rows.append({'line': lineno, 'index': index, 'consecutive_for_this_loop': consecutive})
            if rows:
                relations.append({'loop': lid, 'array': a_label, 'variable': var,
                                  'accesses': rows[:30], 'LLVM_access_links': ir_accesses[:5],
                                  'evidence': 'direct_source_access+LLVM_pragma_scope_access_link_same_function',
                                  'alias_or_bank_proof': False})
    return relations


def prepare(args):
    out = Path(args.bundle).expanduser().resolve()
    if out.exists():
        die('Bundle exists. Use a fresh path: ' + str(out))
    source = Path(args.template).resolve()
    info = Path(args.kernel_info).resolve()
    graph = Path(args.graph).resolve()
    initial = Path(args.initial).resolve()
    srcdir = Path(args.source_dir).resolve()
    for p in (source, info, graph, initial):
        if not p.is_file():
            die('Missing input: ' + str(p))
    if not (source == srcdir or srcdir in source.parents):
        die('Template must be inside source-dir')
    if source.suffix not in ('.cpp', '.c'):
        die('Template must be a .c or .cpp file')
    top, actions = parse_info(info.read_text())
    if args.top and args.top != top:
        die('--top differs from kernel_info top function')
    inp = jload(initial)
    if isinstance(inp, dict):
        if inp.get('kernel') and inp['kernel'].lower() != args.kernel.lower():
            die('Initial configuration names a different kernel')
        if inp.get('objective') and inp['objective'] != 'LATENCY':
            die('This refinement prototype optimizes LATENCY, not ADP/AREA')
        if inp.get('device') and inp['device'] != args.part:
            die('Initial configuration device conflicts with target part')
        if inp.get('clock_period_ns') and abs(float(inp['clock_period_ns'])-args.clock) > 1e-6:
            die('Initial configuration clock conflicts with requested clock')
        if inp.get('budget_percent') and abs(float(inp['budget_percent'])-args.budget_percent) > 1e-6:
            die('Initial configuration budget conflicts with requested budget')
    cfg = inp['directives'] if isinstance(inp, dict) else inp
    cfg = validate(cfg, actions)
    graph_data = graph_index(graph, actions)
    if graph_data['missing_actions'] and not args.allow_partial_graph:
        die('LLVM GEXF is missing Lk action anchors: ' + ', '.join(graph_data['missing_actions']) +
            '. Regenerate with the original llvm-ir pragma-attached graph pipeline;'
            ' or explicitly use --allow-partial-graph (unmapped sites never editable).')
    template = source.read_text()
    placeholders = {(kind, lid) for kind, lid in PH.findall(template)}
    for lid, action in actions.items():
        expected = {'PIPE', 'UNROLL'} if action['kind'] == 'loop' else {'ARRAY_T', 'ARRAY_F', 'ARRAY_D'}
        missing = {k for k in expected if (k, lid) not in placeholders}
        if missing:
            die('Missing placeholders for ' + lid + ': ' + repr(missing))
    if {lid for _, lid in placeholders} - set(actions):
        die('Extra template placeholders not in kernel_info')
    materialize(template, cfg)
    spans = loop_spans(template, actions)
    relations = direct_accesses(template, actions, spans, graph_data)
    out.mkdir(parents=True)
    (out/'inputs').mkdir()
    # Only snapshot source/header inputs; exclude analysis-only HLS vendor stubs.
    for p in sorted(srcdir.iterdir()):
        if p.is_file() and p.suffix.lower() in ('.h', '.hpp', '.inc') and p.name not in VENDOR_HEADERS:
            shutil.copy2(str(p), str(out/'inputs'/p.name))
    (out/'inputs'/'template.cpp').write_text(template)
    (out/'inputs'/'kernel_info.txt').write_text(info.read_text())
    shutil.copy2(str(graph), str(out/'inputs'/'llvm_graph.gexf'))
    manifest = {'schema': 'cg-bdr-v0.1-research', 'kernel': args.kernel,
                'top': top, 'template_suffix': source.suffix, 'part': args.part, 'clock_ns': args.clock,
                'budget_percent': args.budget_percent, 'max_evaluations': args.max_evaluations,
                'compiler_graph': graph_data, 'actions': actions,
                'loop_spans': spans, 'loop_array_relations': relations,
                'initial_directives': cfg,
                'input_hashes': {p.name: digest(p) for p in (out/'inputs').iterdir()},
                'controller_sha256': digest(__file__),
                'settings': SETTINGS, 'source_identity': str(source),
                'artifact_note': 'No model inference or HLS synthesis performed in prepare',
                'diagnostic_policy': 'No low-confidence Lk causal conclusions from graph proximity alone'}
    jdump(out/'manifest.json', manifest)
    jdump(out/'state.json', {'evaluations': [], 'stop': None})
    print(json.dumps({'bundle': str(out), 'actions': len(actions),
                      'graph_anchored': len(graph_data['anchored_actions']),
                      'source_loops_mapped': len(spans), 'loop_array_relations': len(relations),
                      'synthesis_calls': 0}, indent=2))


def verify_bundle(bundle):
    m = jload(bundle/'manifest.json')
    for name, expected in m['input_hashes'].items():
        if digest(bundle/'inputs'/name) != expected:
            die('Bundle snapshot hash mismatch: ' + name)
    if m['schema'] != 'cg-bdr-v0.1-research':
        die('Unknown bundle format')
    if m.get('controller_sha256') != digest(__file__):
        die('Controller code changed since prepare: use a new bundle/protocol')
    # JSON object keys are strings, but array dimension indices are integers.
    for action in m['actions'].values():
        if action['kind'] == 'array':
            action['dims'] = {int(k): int(v) for k, v in action['dims'].items()}
    return m


def xml_text(node, path):
    val = node.findtext(path)
    return val.strip() if val is not None else None


def read_report(path, m):
    root = ET.parse(path).getroot()
    version = xml_text(root, './/ReportVersion/Version')
    if version is not None and version != '2021.1':
        die('HLS report version differs from expected 2021.1')
    top = xml_text(root, './/UserAssignments/TopModelName')
    part = xml_text(root, './/UserAssignments/Part')
    clock = xml_text(root, './/UserAssignments/TargetClockPeriod')
    if top != m['top'] or part != m['part'] or clock is None or abs(float(clock)-m['clock_ns']) > 1e-4:
        die('HLS report context mismatch (top/part/clock)')
    raw = xml_text(root, './/SummaryOfOverallLatency/Worst-caseLatency')
    if not raw or not re.fullmatch(r'\d+', raw) or int(raw) <= 0:
        die('Missing positive worst-case latency')
    resource_tags = {'bram': 'BRAM_18K', 'dsp': 'DSP', 'ff': 'FF', 'lut': 'LUT'}
    resources = {}
    for r, tag in resource_tags.items():
        t = xml_text(root, './/AreaEstimates/Resources/'+tag)
        if t is None or not re.fullmatch(r'\d+', t):
            die('Missing resource ' + tag)
        resources[r] = int(t)
    pct = {k: 100*resources[k]/CAPACITIES[k] for k in CAPACITIES}
    budget = m['budget_percent']
    feasible = all(pct[k] <= budget + 1e-8 for k in pct)
    clk = xml_text(root, './/SummaryOfTimingAnalysis/EstimatedClockPeriod')
    estimated = float(clk) if clk and re.fullmatch(r'[\d.eE+-]+', clk) else None
    loops = []
    for x in root.findall('.//SummaryOfLoopLatency/*'):
        name = x.tag
        latency = xml_text(x, './Latency')
        interval = xml_text(x, './IterationInterval') or xml_text(x, './II')
        loops.append({'name': name,
                      'latency': int(latency) if latency and latency.isdigit() else None,
                      'achieved_ii': int(interval) if interval and interval.isdigit() else None})
    return {'latency_cycles': int(raw), 'resources': resources, 'utilization_percent': pct,
            'budget_feasible': feasible, 'estimated_clock_ns': estimated,
            'hls_estimated_clock_within_target': estimated <= m['clock_ns'] if estimated is not None else None,
            'loop_summary': loops}


def parse_diagnostics(log, template, m):
    """Parse actual Vitis 2021.1 log diagnostics; fail closed on ambiguous action scope."""
    source_lines = template.splitlines()
    spans = m['loop_spans']
    anchored = set(m['compiler_graph']['anchored_actions'])
    diagnostics = []
    entries = log.splitlines()
    for idx, line in enumerate(entries):
        lo = line.lower()
        if not ('warning:' in lo or 'error:' in lo):
            continue
        typ = None
        if ('limited memory ports' in lo or 'memory port' in lo or 'memory access conflict' in lo):
            typ = 'memory_ports'
        elif ('dependence' in lo or 'dependency' in lo) and ('ii' in lo or 'pipeline' in lo):
            typ = 'recurrence'
        elif 'unable to satisfy' in lo and 'ii' in lo:
            typ = 'unmet_ii_unknown_cause'
        if typ is None:
            continue
        am = re.search(r"\bon array\s+['\"]([^'\"]+)['\"]", line, re.I)
        if not am:
            am = re.search(r"\b(?:array|memory)\s+['\"]([^'\"]+)['\"]", line, re.I)
        array = am.group(1) if am else None
        loc = re.search(r'\b([A-Za-z0-9_.-]+\.(?:c|cpp|cc)):(\d+)\b', line)
        src_line = (int(loc.group(2)) if loc and loc.group(1) == 'candidate'+m.get('template_suffix', '.cpp') else None)
        lids = []
        if src_line and src_line <= len(source_lines):
            lids = [lid for lid, span in spans.items() if span['start'] <= src_line <= span['end'] and lid in anchored]
            lids.sort(key=lambda lid: spans[lid]['end']-spans[lid]['start'])
        if not lids:
            # A literal loop label in a Vitis message is stronger than guessing by proximity.
            ids = set(REF.findall(line))
            lids = sorted(ids & set(spans) & anchored, key=natural)
        named_loop = re.search(r"\(loop\s+['\"](L\d+)['\"]\)", line)
        reported_loop = named_loop.group(1) if named_loop and named_loop.group(1) in anchored else None
        diagnostics.append({'reported_loop': reported_loop, 'type': typ, 'array': array, 'source_line': src_line,
                            'affected_loops': lids, 'raw': line.strip(),
                            'evidence': 'explicit_vitis_log', 'attribution': 'source_line+LLVM_action_anchor' if lids else 'unmapped'})
    return diagnostics


def parse_schedule_results(log):
    """Parse Vitis HLS 2021.1 per-loop TargetII/Final II from scheduling log.

    Information-only, not an inferred LLVM dependence or a CSynth XML metric.
    The log is kept in full as the authoritative source.
    """
    rows=[]
    pat=re.compile(r'Pipelining result\s*:\s*Target\s*II\s*=\s*(\d+)\s*,\s*'
                   r'Final\s*II\s*=\s*(\d+)\s*,\s*Depth\s*=\s*(\d+)\s*,\s*'
                   r"loop\s*['\"]([^'\"]+)['\"]",re.I)
    for line in log.splitlines():
        match=pat.search(line)
        if match:
            t,f,d,name=match.groups()
            rows.append({'loop':name,'target_ii':int(t),'achieved_ii':int(f),
                         'pipeline_depth':int(d), 'source':'vitis_hls_scheduling_log'})
    return rows


def lower_factor(cur, bound):
    opts = [x for x in factors(bound) if 1 < x < cur]
    return max(opts) if opts else (1 if cur > 1 else None)


def higher_factor(cur, bound):
    opts = [x for x in factors(bound) if x > max(1, cur)]
    return min(opts) if opts else None


def vitis_array_matches_source(diagnostic, relation, template_lines):
    """Strict, provenance-tagged match for Vitis SSA/split-suffixed array names.

    A source/graph relation must ALREADY exist. The Vitis name alone never
    creates an array relation or establishes alias/dependence proof.
    """
    reported = diagnostic.get('array')
    variable = relation['variable']
    if not reported or not variable:
        return None
    if reported == variable:
        match_kind = 'exact_source_name'
    elif re.fullmatch(re.escape(variable) + r'_V_\d+(?:_\d+)*', reported):
        match_kind = 'vitis_generated_V_numeric_suffix'
    else:
        return None
    line_no = diagnostic.get('source_line')
    if not isinstance(line_no, int):
        return None
    if template_lines is not None:
        if not 1 <= line_no <= len(template_lines):
            return None
        source = template_lines[line_no - 1]
        if not re.search(r'\b' + re.escape(variable) + r'\s*\[', source):
            return None
    else:
        source = ''
    if not any(access['line'] == line_no for access in relation['accesses']):
        return None
    return {'source_array': variable, 'hls_array': reported, 'match_kind': match_kind,
            'source_line': line_no, 'source_text': source.strip(),
            'warning': 'Vitis-derived name matching is attribution, not bank/alias proof'}


def generate_repair(m, original, result, used, explore=False):
    """Rank diagnostic-grounded repairs. No general search or QoR surrogate."""
    acts = m['actions']
    config = {r['action_id']: dict(r) for r in original}
    rels = m['loop_array_relations']
    anchored = set(m['compiler_graph']['anchored_actions'])
    suggestions = []
    template_lines = m.get('_source_template_lines')

    def propose(changes, kind, why, evidence, priority):
        new = {k: dict(v) for k, v in config.items()}
        for lid, updates in changes.items():
            if lid not in anchored or lid not in new:
                return
            new[lid].update(updates)
        try:
            cfg = validate(list(new.values()), acts)
        except RuntimeError:
            return
        sig = signature(cfg)
        if sig not in used and sig != signature(original):
            suggestions.append({'directives': cfg, 'signature': sig, 'kind': kind,
                                'reason': why, 'evidence': evidence,
                                'changed_actions': sorted(changes, key=natural), 'priority': priority})

    res = result.get('metrics', {})
    util = res.get('utilization_percent', {})
    for d in result.get('diagnostics', []):
        if d['type'] != 'memory_ports':
            continue
        array = d.get('array')
        if not array:
            continue  # no invented array assignment
        for r in rels:
            if r['loop'] not in d['affected_loops']:
                continue
            array_match = vitis_array_matches_source(d, r, template_lines)
            if not array_match:
                continue
            lid, aid = r['loop'], r['array']
            la, aa = config[lid], config[aid]
            proof = {'diagnostic': d['raw'], 'source_array_match': array_match, 'source_accesses': r['accesses'],
                     'LLVM_action_anchors': [lid, aid], 'alias_proven': False}
            dim = aa['partition_dim']
            # A direct index expression with the loop iterator supports considering cyclic banking.
            consecutive = any(x['consecutive_for_this_loop'] for x in r['accesses'])
            # When consecutive accesses are unrolled beyond the current bank
            # count, first test cyclic banking with an appropriate legal bank
            # count (one changed Lk site, despite two changed pragma fields).
            if consecutive and aa['partition_type'] == 'block' and la['unroll_factor'] > aa['partition_factor']:
                bound = acts[aid]['dims'].get(dim)
                compatible = [n for n in factors(bound) if n >= la['unroll_factor']] if bound else []
                if compatible and max(util.values() or [0]) < 75:
                    propose({aid: {'partition_type': 'cyclic', 'partition_factor': min(compatible)}},
                            'banking_style_and_factor',
                            'Measured memory-port pressure; consecutive accesses with loop unroll higher than bank count: test cyclic mapping with sufficient nominal banks',
                            proof, 115)
            if consecutive and aa['partition_type'] == 'block' and aa['partition_factor'] > 1 and la['unroll_factor'] > 1:
                propose({aid: {'partition_type': 'cyclic'}}, 'banking_style',
                        'Vitis identified memory-port contention; consecutive loop-indexed accesses may benefit from cyclic rather than block banks',
                        proof, 100)
            if aa['partition_type'] in ('block', 'cyclic') and dim in acts[aid]['dims']:
                bound = acts[aid]['dims'][dim]
                larger = higher_factor(aa['partition_factor'], bound)
                if larger and max(util.values() or [0]) < 75:
                    propose({aid: {'partition_factor': larger}}, 'banking_factor',
                            'Vitis identified memory-port contention; test one legal increase in banks with resource headroom', proof, 85)
            if la['unroll_factor'] > 1:
                new_u = lower_factor(la['unroll_factor'], acts[lid]['bound'])
                if new_u is not None and consecutive and aa['partition_type'] == 'block':
                    # A two-action repair hypothesis: change the bank mapping and
                    # reduce simultaneous accesses. This is not a guaranteed fix.
                    propose({aid: {'partition_type': 'cyclic'}, lid: {'unroll_factor': new_u}},
                            'coupled_banking_and_concurrency',
                            'HLS memory-port violation with index-compatible cyclic banking: repair both access distribution and requested concurrency',
                            proof, 60)
                if new_u is not None:
                    propose({lid: {'unroll_factor': new_u}}, 'reduce_concurrency',
                            'Vitis identified memory-port contention; reduce simultaneous accesses when banking cannot serve the requested lanes',
                            proof, 65)
    # Loop-only action spaces can still respond to explicit memory-port pressure.
    # This is a bounded concurrency experiment; it does not claim a dependence
    # was removed. Prefer the HLS-reported loop over nested source spans.
    for d in result.get('diagnostics', []):
        if d['type'] != 'memory_ports':
            continue
        targets = ([d['reported_loop']] if d.get('reported_loop') else d.get('affected_loops', []))
        for lid in targets:
            a = config.get(lid)
            if not a or lid not in anchored or a['kind'] != 'loop' or a['unroll_factor'] <= 1:
                continue
            new_u = lower_factor(a['unroll_factor'], acts[lid]['bound'])
            if new_u is None:
                continue
            propose({lid: {'unroll_factor': new_u}}, 'reduce_memory_port_concurrency',
                    'Explicit Vitis memory-port conflict in this scheduled loop: test a bounded reduction in concurrent accesses',
                    {'diagnostic': d['raw'], 'reported_loop': d.get('reported_loop'),
                     'warning': 'This cannot be assumed to remove a true loop-carried recurrence'}, 45)
    # Rescue infeasible or strongly timing-stressed designs by reducing a replication point.
    pressure = not res.get('budget_feasible', True)
    clock_problem = (res.get('estimated_clock_ns') is not None and
                     res['estimated_clock_ns'] > m['clock_ns'])
    if pressure or clock_problem:
        for lid, a in config.items():
            if lid in anchored and a['kind'] == 'loop' and a['unroll_factor'] > 1:
                new_u = lower_factor(a['unroll_factor'], acts[lid]['bound'])
                if new_u is not None:
                    propose({lid: {'unroll_factor': new_u}}, 'reduce_replication',
                            'Reduce a legal unroll factor to address measured capacity/timing pressure',
                            {'current_utilization': util, 'estimated_clock_ns': res.get('estimated_clock_ns')},
                            55 + min(a['unroll_factor'], 32)/32)
    # When requested explicitly, use measured loop latency to explore one *bounded* unroll edit.
    # This is labelled exploratory; it is NOT a confirmed diagnosis.
    if explore and not suggestions:
        for loop in sorted(res.get('loop_summary', []), key=lambda a: a.get('latency') or 0, reverse=True):
            mt = re.match(r'^(L\d+)(?:_.*)?$', loop['name'])
            if not mt:
                continue
            lid = mt.group(1)
            if lid not in anchored or lid not in config or config[lid]['kind'] != 'loop':
                continue
            a = config[lid]
            new_u = higher_factor(a['unroll_factor'], acts[lid]['bound'])
            if new_u and new_u <= max(2, a['unroll_factor'] * 2):
                propose({lid: {'unroll_factor': new_u}}, 'exploratory_hot_loop',
                        'Measured loop is expensive; bounded unroll exploration, not proven causality',
                        {'loop_summary': loop, 'evidence_strength': 'exploratory'}, 10)
            break
    suggestions.sort(key=lambda x: (-x['priority'], len(x['changed_actions']), x['signature']))
    return suggestions


def tcl_quote(s):
    s = str(s)
    if any(c in s for c in '{}\n\r'):
        die('Unsafe Tcl string')
    return '{' + s + '}'


def synthesize(bundle, m, cfg, idx, executable, timeout, test_mock=False):
    work = bundle/'projects'/('eval-%02d' % idx)
    work.mkdir(parents=True, exist_ok=False)
    template = (bundle/'inputs'/'template.cpp').read_text()
    text = materialize(template, cfg)
    src = work/('candidate'+m.get('template_suffix', '.cpp'))
    src.write_text(text)
    for p in (bundle/'inputs').iterdir():
        if p.is_file() and p.suffix in ('.h', '.hpp', '.inc') and p.name not in VENDOR_HEADERS:
            shutil.copy2(str(p), str(work/p.name))
    vendor_flags = ''
    if not test_mock:
        hls_root = executable.resolve().parent.parent
        include = hls_root/'include'
        if not (include/'ap_fixed.h').is_file() or not (include/'ap_int.h').is_file():
            die('Genuine Vitis ap_fixed.h/ap_int.h not found under ' + str(include))
        vendor_flags = ' -cflags ' + tcl_quote('-I'+str(include))
    tcl = '\n'.join(['open_project -reset hls_project', 'set_top '+m['top'],
                     'add_files'+vendor_flags+' '+tcl_quote(src.name),
                     'open_solution -reset solution1 -flow_target vivado',
                     'set_part '+tcl_quote(m['part']),
                     'create_clock -period %.8g -name default' % m['clock_ns'],
                     *m['settings'], 'csynth_design', 'exit', ''])
    (work/'run_hls.tcl').write_text(tcl)
    logpath = work/'synthesis.log'
    start = time.monotonic()
    error = None
    with logpath.open('w') as log:
        proc = subprocess.Popen([str(executable), '-f', 'run_hls.tcl'], cwd=str(work),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            rc = -9
            error = 'timeout'
    xml = work/'hls_project'/'solution1'/'syn'/'report'/(m['top']+'_csynth.xml')
    metrics = None
    completed = ('Finished Command csynth_design' in logpath.read_text(errors='replace') and
                 'Exiting vitis_hls' in logpath.read_text(errors='replace'))
    if rc == 0 and xml.is_file() and completed:
        try:
            metrics = read_report(xml, m)
        except (RuntimeError, ValueError, ET.ParseError) as ex:
            error = 'untrusted_report: ' + str(ex)
    elif error is None:
        error = 'vitis_failed_or_no_complete_report, returncode=%s completed=%s' % (rc, completed)
    result = {'index': idx, 'synthesis_status': 'success' if metrics else 'failed',
              'error': error, 'duration_seconds': round(time.monotonic()-start, 3),
              'source_sha256': digest(src), 'tcl_sha256': digest(work/'run_hls.tcl'),
              'report_sha256': digest(xml) if xml.is_file() else None,
              'log_sha256': digest(logpath), 'metrics': metrics,
              'diagnostics': parse_diagnostics(logpath.read_text(errors='replace'), template, m),
              'work_dir': str(work.relative_to(bundle)), 'tool': str(executable),
              'tool_mode': 'SYNTHETIC_TEST_ONLY' if test_mock else 'VITIS_HLS'}
    jdump(work/'result.json', result)
    return result


def eligible(metrics):
    return bool(metrics and metrics['budget_feasible'] and
                metrics.get('hls_estimated_clock_within_target') is not False)


def run(args):
    bundle = Path(args.bundle).expanduser().resolve()
    m = verify_bundle(bundle)
    m['_source_template_lines'] = (bundle/'inputs'/'template.cpp').read_text().splitlines()
    state = jload(bundle/'state.json')
    if state['stop']:
        print('Stopped: ' + state['stop'])
        return
    exe = Path(args.vitis_hls).expanduser().resolve()
    if not exe.is_file() or not os.access(str(exe), os.X_OK):
        die('Not executable: ' + str(exe))
    if not args.test_mock:
        vendor_root = exe.parent.parent / 'include'
        if not (vendor_root/'ap_fixed.h').is_file() or not (vendor_root/'ap_int.h').is_file():
            die('Genuine vendor headers missing before budget reservation: ' + str(vendor_root))
        p = subprocess.run([str(exe), '-version'], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
        if '2021.1' not in p.stdout:
            die('This protocol expects Vitis HLS 2021.1. Actual: ' + p.stdout[:500])
    if len(state['evaluations']) >= m['max_evaluations']:
        die('Evaluation budget already exhausted')
    if (bundle/'pending.json').exists():
        die('Pending reserved synthesis exists: investigate before manual recovery. No free retry.')
    count = min(args.limit, m['max_evaluations']-len(state['evaluations']))
    for _ in range(count):
        # Disk exhaustion is a known source of irreproducible Vitis failures.
        if shutil.disk_usage(str(bundle)).free < args.min_free_gb * 1024**3:
            die('Insufficient free disk space before synthesis; no query charged')
        old = jload(bundle/'state.json')
        hist = old['evaluations']
        initial = m['initial_directives']
        if hist:
            scored = [row for row in hist if eligible(row['result']['metrics'])]
            measured = [row for row in hist if row['result']['metrics']]
            if not measured:
                old['stop'] = 'No synthesizable measured configuration; cannot identify safe repairs'
                jdump(bundle/'state.json', old)
                break
            if scored:
                best = min(scored, key=lambda row: row['result']['metrics']['latency_cycles'])
            else:
                # A synthesized but resource-infeasible baseline can be repaired.
                # No infeasible candidate is ever reported as a winning design.
                best = min(measured, key=lambda row: (
                    max(row['result']['metrics']['utilization_percent'].values()) - m['budget_percent'],
                    row['result']['metrics']['latency_cycles']))
            incumbent = best['directives']
            used = {row['signature'] for row in hist}
            # Always diagnose the measured incumbent, not the previous failed or
            # inferior candidate. No stale bottleneck attribution is transferred.
            proposals = generate_repair(m, incumbent, best['result'], used,
                                        explore=args.explore_on_no_diagnostic)
            if not proposals:
                old['stop'] = 'No evidence-supported untried repair'
                jdump(bundle/'state.json', old)
                break
            selection = proposals[0]
        else:
            selection = {'directives': initial, 'signature': signature(initial),
                         'kind': 'initial_mailohls', 'reason': 'Initial model configuration',
                         'evidence': {}, 'changed_actions': []}
        number = len(hist) + 1
        reservation = {'index': number, 'candidate_signature': selection['signature'],
                       'selection': {k: v for k, v in selection.items() if k != 'directives'},
                       'created_at_utc': time.time()}
        jdump(bundle/'pending.json', reservation)
        print('Reserved HLS evaluation %d/%d: %s' % (number, m['max_evaluations'], selection['kind']), flush=True)
        try:
            result = synthesize(bundle, m, selection['directives'], number, exe, args.timeout, args.test_mock)
            record = {'signature': selection['signature'], 'directives': selection['directives'],
                      'selection': reservation['selection'], 'result': result}
            old['evaluations'].append(record)
            jdump(bundle/'state.json', old)
            (bundle/'pending.json').unlink()
        except BaseException:
            print('Evaluation interrupted; pending.json remains charged. Investigate before attempting recovery.',
                  file=sys.stderr)
            raise
        print('  status=%s  latency=%s  feasible=%s  diagnostics=%d' % (
            result['synthesis_status'],
            result['metrics']['latency_cycles'] if result['metrics'] else '-',
            result['metrics']['budget_feasible'] if result['metrics'] else '-',
            len(result['diagnostics'])), flush=True)
    summary(bundle)


def summary(bundle):
    m = verify_bundle(bundle)
    state = jload(bundle/'state.json')
    hist = state['evaluations']
    good = [row for row in hist if eligible(row['result']['metrics'])]
    if good:
        best = min(good, key=lambda x: x['result']['metrics']['latency_cycles'])
        initial = hist[0]['result']['metrics']['latency_cycles'] if hist and hist[0]['result']['metrics'] else None
        speedup = (initial / best['result']['metrics']['latency_cycles']) if initial else None
        payload = {'best_directives': best['directives'],
                   'best_latency_cycles': best['result']['metrics']['latency_cycles'],
                   'improvement_percent': 100*(1-1/speedup) if speedup else None,
                   'initial_latency_cycles': initial, 'best_evaluation': best['result']['index'],
                   'best_resources': best['result']['metrics']['resources']}
    else:
        payload = {'best_directives': None, 'best_latency_cycles': None}
    base_bottlenecks = {(d['type'], d.get('array'), tuple(d.get('affected_loops', [])))
                        for d in (hist[0]['result']['diagnostics'] if hist else [])}
    migration = []
    for row in hist:
        current = {(d['type'], d.get('array'), tuple(d.get('affected_loops', [])))
                   for d in row['result']['diagnostics']}
        migration.append({'index': row['result']['index'],
                          'observed_bottlenecks': len(current),
                          'initial_bottleneck_signatures_not_observed': len(base_bottlenecks-current),
                          'new_bottleneck_signatures': len(current-base_bottlenecks),
                          'warning': 'A disappearing diagnostic is not automatically proof the hardware bottleneck was solved'})
    report = {'kernel': m['kernel'], 'objective': 'LATENCY', 'part': m['part'],
              'clock_ns': m['clock_ns'], 'budget_percent': m['budget_percent'],
              'charged_evaluations': len(hist), 'max_evaluations': m['max_evaluations'],
              'stop': state['stop'], 'best': payload, 'bottleneck_migration': migration,
              'timeline': [{'index': row['result']['index'], 'kind': row['selection']['kind'],
                            'why': row['selection']['reason'],
                            'changed_actions': row['selection']['changed_actions'],
                            'latency': row['result']['metrics']['latency_cycles'] if row['result']['metrics'] else None,
                            'feasible': row['result']['metrics']['budget_feasible'] if row['result']['metrics'] else False,
                            'diagnostics': row['result']['diagnostics']} for row in hist],
              'limitations': ['HLS estimate only, no place-and-route',
                              'LLVM action anchors do not prove alias or dependence',
                              'Loop source span parsing is intentionally restricted',
                              'No functional equivalence or cosimulation validated']}
    jdump(bundle/'summary.json', report)
    print(json.dumps({k: report[k] for k in ('kernel','charged_evaluations','max_evaluations','stop','best')}, indent=2))
    return report


def preview(bundle):
    """Read-only candidate inspection; never reserves, writes or invokes Vitis."""
    m = verify_bundle(bundle)
    m['_source_template_lines'] = (bundle/'inputs'/'template.cpp').read_text().splitlines()
    state = jload(bundle/'state.json')
    hist = state['evaluations']
    if not hist:
        print(json.dumps({'status':'no measured evaluation yet', 'proposals':[]},indent=2))
        return []
    if state.get('stop') or len(hist) >= m['max_evaluations']:
        print(json.dumps({'status':'stopped or exhausted', 'stop':state['stop'],
                          'charged_evaluations':len(hist), 'proposals':[]},indent=2))
        return []
    good = [x for x in hist if eligible(x['result']['metrics'])]
    measured = [x for x in hist if x['result']['metrics']]
    if not measured:
        print(json.dumps({'status':'no trustworthy measured design', 'proposals':[]},indent=2))
        return []
    if good:
        inc=min(good,key=lambda x:x['result']['metrics']['latency_cycles'])
    else:
        inc=min(measured,key=lambda x:(max(x['result']['metrics']['utilization_percent'].values()) - m['budget_percent'],
                                       x['result']['metrics']['latency_cycles']))
    proposals=generate_repair(m,inc['directives'],inc['result'],
                              {x['signature'] for x in hist})
    base={x['action_id']:x for x in inc['directives']}
    entries=[]
    for p in proposals:
        changed={x['action_id']:{'old':base[x['action_id']],'new':x} for x in p['directives'] if base[x['action_id']]!=x}
        entries.append({'kind':p['kind'],'priority':p['priority'],'reason':p['reason'],
                        'changed':changed,'evidence':p['evidence']})
    print(json.dumps({'status':'ok','charged_evaluations':len(hist),
                      'current_incumbent_latency':inc['result']['metrics']['latency_cycles'],
                      'diagnostics':inc['result']['diagnostics'],
                      'next_selection':entries[0] if entries else None,
                      'num_proposals':len(entries)},indent=2))
    return entries


def main():
    ap = argparse.ArgumentParser(description='Compiler-grounded bottleneck-directed HLS refinement')
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Liono: snapshot source, LLVM graph, and baseline; no HLS')
    p.add_argument('--kernel', required=True)
    p.add_argument('--template', required=True)
    p.add_argument('--source-dir', required=True)
    p.add_argument('--kernel-info', required=True)
    p.add_argument('--graph', required=True)
    p.add_argument('--initial', required=True)
    p.add_argument('--top', default='')
    p.add_argument('--bundle', required=True)
    p.add_argument('--part', default=PART)
    p.add_argument('--clock', type=float, default=10.0)
    p.add_argument('--budget-percent', type=float, default=100.0)
    p.add_argument('--max-evaluations', type=int, default=4)
    p.add_argument('--allow-partial-graph', action='store_true')
    q = sub.add_parser('run', help='Coroni: charge/run at most --limit HLS evaluations')
    q.add_argument('--bundle', required=True)
    q.add_argument('--vitis-hls', required=True)
    q.add_argument('--limit', type=int, default=1)
    q.add_argument('--timeout', type=int, default=7200)
    q.add_argument('--min-free-gb', type=float, default=20)
    q.add_argument('--explore-on-no-diagnostic', action='store_true')
    q.add_argument('--test-mock', action='store_true', help='TEST ONLY: skip Vitis version/vendor include checks')
    v = sub.add_parser('preview', help='Read-only repair preview; does NOT use HLS')
    v.add_argument('--bundle', required=True)
    s = sub.add_parser('summary', help='Inspect existing charged experiments')
    s.add_argument('--bundle', required=True)
    args = ap.parse_args()
    if args.command == 'prepare':
        if args.max_evaluations < 1 or args.max_evaluations > 4:
            die('Evaluation cap must be 1–4')
        if not 0 < args.budget_percent <= 100 or args.clock <= 0:
            die('Invalid synthesis context')
        if args.part != PART:
            die('This validated prototype only supports the ZCU104 part; do not reuse ZCU104 capacity tables on another part')
        prepare(args)
    elif args.command == 'run':
        if args.limit < 1 or args.timeout < 1 or args.min_free_gb < 0:
            die('limit/timeout must be positive')
        run(args)
    elif args.command == 'preview':
        preview(Path(args.bundle).expanduser().resolve())
    elif args.command == 'summary':
        summary(Path(args.bundle).expanduser().resolve())

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, ET.ParseError) as exc:
        print('CG-BDR ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(2)
