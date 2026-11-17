"""Observed schedule/diagnostic groups and conservative source attribution (3.8+).

No alias, physical-port, dependence, or exact loop critical-path claims.
"""
import re
import statistics


def schedule_groups(rows):
    groups = {}
    for position, row in enumerate(rows, 1):
        # Never assign generated/unnamed loops to a source Lk by proximity.
        name = row['loop']
        group = groups.setdefault(name, {'source_loop': name if re.fullmatch(r'L[1-9]\d*', name) else None,
                                         'rows': [], 'provenance': 'explicit_Vitis_scheduling_log'})
        group['rows'].append(dict(row, occurrence=position))
    for group in groups.values():
        group['count'] = len(group['rows'])
        for field in ('target_ii', 'achieved_ii', 'pipeline_depth'):
            values = [r[field] for r in group['rows']]
            group[field] = dict(values=values, min=min(values), max=max(values), median=statistics.median(values))
        group['stalled'] = all(r['achieved_ii'] > r['target_ii'] for r in group['rows'])
        group['exact_final_ii'] = (group['achieved_ii']['min'] if
                                   group['achieved_ii']['min'] == group['achieved_ii']['max'] else None)
    return groups


def canonical_memory(name):
    return re.sub(r'_V_\d+(?:_\d+)*$', '', name) if name else None


def warning_groups(rows):
    groups = {}
    for row in rows:
        key = (row['type'], canonical_memory(row.get('array')), row.get('scheduled_loop'),
               row.get('access_owning_loop'), row.get('source_line'), tuple(row.get('affected_loops', [])))
        if key not in groups:
            groups[key] = dict(row, canonical_memory=key[1], occurrence_count=0, representative_raw=[])
        group = groups[key]
        group['occurrence_count'] += 1
        if row['raw'] not in group['representative_raw'] and len(group['representative_raw']) < 2:
            group['representative_raw'].append(row['raw'])
    return list(groups.values())


def analyze(base, template, actions, spans, graph, feedback_only=False):
    # Source hierarchy in feedback-only must not depend on any graph filtering.
    source_graph = {'anchored_actions': list(actions)} if feedback_only else graph
    sem = base.semantic_v2.analyze(template, actions, spans, source_graph)
    sem['schema'] = 'action-semantics-v3-source-observations'
    if feedback_only:
        sem.pop('llvm_action_anchors', None)
        for edge in sem['loop_edges']:
            edge['provenance'] = 'labeled_C++_source_spans_only'
    sem['trip_counts'] = {}
    lines = template.splitlines()
    for lid, span in spans.items():
        # Only literal, unit-stride loops. Metadata upper bounds are NOT trips.
        text = lines[span['start']-1]
        it = re.escape(span['iterator'])
        pat = r'for\s*\(\s*(?:int\s+)?'+it+r'\s*=\s*(\d+)\s*;\s*'+it+r'\s*<\s*(\d+)\s*;\s*'+it+r'\s*\+\+\s*\)'
        match = re.search(pat, text)
        if match and int(match[2]) > int(match[1]):
            sem['trip_counts'][lid] = dict(count=int(match[2])-int(match[1]), source_line=span['start'],
                                          provenance='literal_source_unit_stride_loop_not_metadata_bound')
    return sem


def chain(sem, loop):
    seen = set()
    while loop in sem['loop_parent'] and loop not in seen:
        seen.add(loop)
        yield loop
        loop = sem['loop_parent'][loop]


def source_access(m, diagnostic, loop):
    """Exact warning line + array token + verified containment, never dependence."""
    sem = m['semantic_v3']
    owner = diagnostic.get('access_owning_loop')
    if owner not in sem['loop_parent'] or loop not in list(chain(sem, owner)):
        return None
    for row in sem['direct_statements'].get(owner, []):
        if row['line'] != diagnostic.get('source_line'):
            continue
        accesses = [a for a in row['accesses'] if a['name'] == diagnostic.get('canonical_memory')]
        if accesses:
            return dict(owning_loop=owner, controlling_loop=loop, line=row['line'], accesses=accesses,
                        parent_chain=list(chain(sem, owner)), dependence_proven=False)
    return None


def criticality(m, metrics, loop):
    # High-level XML often describes only an enclosing nest. Never add these.
    parent_chain = list(chain(m['semantic_v3'], loop))
    contextual = [r for r in metrics.get('loop_summary', []) if r['name'] in parent_chain]
    return dict(exact_loop_criticality='unknown', enclosing_report_context=contextual,
                source_parent_chain=parent_chain, nested_latencies_summed=False)
