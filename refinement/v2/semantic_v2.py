#!/usr/bin/env python3
"""CG-BDR v2 lightweight compiler-grounded ACTION/CONSTRAINT facts.

The existing LLVM/ProGraML GEXF is the authority for Lk anchor existence.
This sidecar extracts SOURCE-VERIFIED containment and direct memory accesses;
optional LLVM opt analysis is a separate corroborating artifact. None of the
facts produced here prove physical banking, alias freedom, or a recurrence.
Pure stdlib, compatible with liono Python 3.7 and Coroni Python 3.8.
"""
import re

LABEL = re.compile(r'\bL[1-9]\d*\b')
ARRAY = re.compile(r'(?<!\w)([A-Za-z_]\w*)\s*((?:\[[^\]]+\]\s*)+)')
SKIP = set('if while for return sizeof switch'.split())


def analyze(template, actions, spans, graph):
    lines = template.splitlines()
    anchor = set(graph.get('anchored_actions', []))
    loops = {lid: v for lid, v in spans.items() if lid in anchor and actions[lid]['kind']=='loop'}
    parent = {}
    for child, sp in loops.items():
        holders = [(other, o) for other, o in loops.items()
                   if other != child and o['start'] < sp['start'] and o['end'] >= sp['end']]
        parent[child] = min(holders, key=lambda x: x[1]['end']-x[1]['start'])[0] if holders else None
    children = {lid: [] for lid in loops}
    for lid, p in parent.items():
        if p: children[p].append(lid)
    for v in children.values(): v.sort(key=lambda lid: loops[lid]['start'])
    direct = {}
    for lid, sp in loops.items():
        # Direct operations of this loop, excluding regions owned by nested Lk loops.
        excluded = set()
        for child, csp in loops.items():
            if child != lid and sp['start'] < csp['start'] and csp['end'] <= sp['end']:
                excluded.update(range(csp['start'],csp['end']+1))
        entries = []
        for lineno in range(sp['start']+1, min(sp['end'],len(lines))+1):
            if lineno in excluded: continue
            line = lines[lineno-1]
            if line.lstrip().startswith(('#','//')): continue
            # Expressions are SOURCE-level observations. No alias/fp reassociation proof.
            reads_writes = []
            for match in ARRAY.finditer(line):
                var, bracket=match.groups()
                if var in SKIP: continue
                inds=re.findall(r'\[([^\]]+)\]',bracket)
                # Detect a clear lvalue assignment to this access; otherwise unknown.
                tail=line[match.end():]
                lvalue=bool(re.match(r'\s*(?:(?:\+|-|\*|/|%|&|\||\^)?)=(?!=)',tail))
                reads_writes.append({'name':var,'indices':[v.strip() for v in inds],
                                     'role':'write_or_update' if lvalue else 'read_or_unknown'})
            red=bool(re.search(r'(?<![=!<>])(?:\+=|-=|\*=|/=)',line))
            # Source-level self-update clue (e.g. C[i][j] = C[i][j] + x).
            # Remains 'possible' because indices/aliases need compiler proof.
            for acc in reads_writes:
                if acc['role']=='write_or_update':
                    t=acc['name']
                    if len(re.findall(r'\b'+re.escape(t)+r'\s*\[',line))>=2:
                        red=True
            entries.append({'line':lineno,'text':line.strip()[:240], 'accesses':reads_writes,
                            'possible_accumulation':red})
        direct[lid]=entries
    # The action graph is NOT traversed for speculative data dependencies.
    links=[]
    for lid, p in parent.items():
        if p:
            links.append({'from':p,'to':lid,'kind':'contains',
                          'provenance':'labeled_C++_source_spans+LLVM_pragma_action_anchors',
                          'llvm_loopinfo_correspondence':'unverified'})
    accessed={}
    for lid, rows in direct.items():
        seen=set()
        for r in rows:
            for a in r['accesses']:
                name=a['name']
                if name not in seen:
                    accessed.setdefault(name,[]).append(lid)
                    seen.add(name)
    # Shared-memory relation is useful for multi-phase stencils but is NOT
    # automatically a dependence: two loops could read disjoint indices.
    shared=[]
    seq=sorted(loops,key=lambda lid:loops[lid]['start'])
    for ii,left in enumerate(seq):
        for right in seq[ii+1:]:
            if left==parent.get(right) or right==parent.get(left):continue
            lmem={a['name'] for row in direct.get(left,[]) for a in row['accesses']}
            rmem={a['name'] for row in direct.get(right,[]) for a in row['accesses']}
            for name in sorted(lmem & rmem):
                shared.append({'a':left,'b':right,'array':name,
                               'provenance':'same_source_array_token_direct_access',
                               'dependence_proven':False,'overlapping_indices_proven':False})
    return {'schema':'action-semantics-v2-source-supported',
            'source_shared_memory_hints':shared,
            'loop_parent':parent,'loop_children':children,'loop_edges':links,
            'direct_statements':direct,'array_accessing_loops':accessed,
            'llvm_action_anchors':sorted(anchor),
            'caveats':['Source containment does not prove matching LLVM LoopInfo loop IDs',
                       'Array-access regex is a source hint, not alias/dependence analysis',
                       'No true recurrence is inferred from an accumulation operator alone']}


def affected_loops(diagnostic, sem):
    reported=diagnostic.get('reported_loop')
    if reported in sem.get('loop_parent',{}): return [reported]
    return [x for x in diagnostic.get('affected_loops',[]) if x in sem.get('loop_parent',{})]


def source_criticality(loop, metrics, actions):
    """Return provenance-labelled, deliberately noncalibrated criticality hints."""
    if loop not in actions or actions[loop]['kind']!='loop': return None
    for record in metrics.get('loop_summary',[]):
        name=record.get('name','')
        if name==loop or name.startswith(loop+'_'):
            latency=record.get('latency')
            if latency is not None:
                return {'tier':'measured_loop','value':latency, 'provenance':'Vitis_CSynth_SummaryOfLoopLatency'}
    return {'tier':'unmeasured_loop','value':None,'source_bound':actions[loop]['bound'],
            'provenance':'kernel_info_only_not_measured_criticality'}


def schedule_ii(result, loop):
    rows = [r for r in result.get('schedule_results',[]) if r.get('loop')==loop]
    return rows[-1] if rows else None


def scope_independent_hint(semantic, lid, parent_lid=None):
    """Only reject explicitly suspicious source reductions; never prove independence."""
    rows=semantic.get('direct_statements',{}).get(lid,[])
    suspicious=[x['line'] for x in rows if x['possible_accumulation']]
    return {'possible_accumulations':suspicious,
            'independence_proven':False,
            'note':'May be useful for conservative risk ranking, not a dependence proof'}


def action_constraint_graph(manifest, result):
    """Small, provenance-bearing post-HLS diagnostic/action constraint graph.

    Edges express *reported* constraints, *source-observed* scope/access, and
    *editable* directives, never physical causal/banking proof.
    """
    sem=manifest.get('semantic_v2',{})
    actions=manifest.get('actions',{})
    loop_rels=manifest.get('loop_array_relations',[])
    metric=result.get('metrics') or {}
    nodes={}
    edges=[]
    for lid, action in actions.items():
        nodes['action:'+lid]={'type':'editable_pragma','Lk':lid,'action_kind':action['kind'],
                              'verified_ProGraML_anchor': lid in manifest.get('compiler_graph',{}).get('anchored_actions',[])}
        if action['kind']=='loop':
            nodes['loop:'+lid]={'type':'source_loop','Lk':lid,
                                'criticality':source_criticality(lid,metric,actions)}
            edges.append({'from':'action:'+lid,'to':'loop:'+lid,'relation':'controls',
                          'evidence':'Lk_source_marker+ProGraML_action_anchor'})
        else:
            nodes['array:'+lid]={'type':'source_array','variable':action['variable']}
            edges.append({'from':'action:'+lid,'to':'array:'+lid,'relation':'controls',
                          'evidence':'array_scope_action_anchor'})
    for edge in sem.get('loop_edges',[]):
        edges.append({'from':'loop:'+edge['from'],'to':'loop:'+edge['to'],
                      'relation':'contains','evidence':edge['provenance'],
                      'alias_or_loopinfo_proof':False})
    for link in sem.get('source_shared_memory_hints',[]):
        edges.append({'from':'loop:'+link['a'],'to':'loop:'+link['b'],
                      'relation':'source_shared_memory_hint',
                      'array':link['array'],'evidence':link['provenance'],
                      'dependence_proven':False})
    for r in loop_rels:
        edges.append({'from':'loop:'+r['loop'],'to':'array:'+r['array'],
                      'relation':'source_accesses','evidence':r['evidence'],
                      'physical_bank_proven':False})
    for i,d in enumerate(result.get('diagnostics',[]),1):
        cid='constraint:C'+str(i)
        nodes[cid]={'type':d['type'],'message':d['raw'],
                    'source_line':d.get('source_line'),
                    'reported_array':d.get('array'),
                    'evidence':'explicit_Vitis_scheduling_warning'}
        owner=d.get('access_owning_loop')
        scheduled=d.get('scheduled_loop') or d.get('reported_loop')
        if scheduled and scheduled in actions and actions[scheduled]['kind']=='loop':
            edges.append({'from':cid,'to':'loop:'+scheduled,
                          'relation':'reported_scheduled_loop','evidence':'HLS_log_explicit_loop_name'})
        if owner and owner in actions and actions[owner]['kind']=='loop':
            edges.append({'from':cid,'to':'loop:'+owner,
                          'relation':'source_location_owner','evidence':'candidate_file_source_line+Lk_span'})
    return {'schema':'cg-bdr-action-constraint-graph-v2','nodes':nodes,'edges':edges,
            'disclaimer':'Graph encodes attributed evidence; a path is not a proof of effective QoR or alias freedom'}
