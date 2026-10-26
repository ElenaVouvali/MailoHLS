"""Compare corrected compiler relationships and interpretable proposal ordering.

The scoped adapter follows refinement_search.benchmark's integration pattern.
Search, validation, measurements, incumbents and budgets remain in the existing
controller. No graph or QoR predictor is constructed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess
import sys

from GNN_branch import mlir_graph_v2 as v2
from GNN_branch.action_point_relations import get_loop_semantics
from GNN_branch.generate_mlir_v2_corpus import checked_views, csv_rows, DEFAULT_OUTPUT as CORPUS
from LLM_branch.common import directive_domains as domains
from experiments.refinement import configurations as c, experiment as existing, relations as legacy
from experiments.refinement_search import controller as ctl, policies as p, benchmark

POLICIES = ('coordinate', 'structural', 'v21_relations', 'v21_priority')
DEFAULT_OUTPUT = c.ROOT / 'experiments/refinement/results/v21_refinement'
VERIFIED = ('direct_verified', 'indirect_verified')
RULES = {
    'P0': 'Unchanged balanced greedy coordinate controller',
    'P1': 'Unchanged legacy verified structural policy and value ordering',
    'P2': 'Replace eligibility with proven V2.1 access/hierarchy/dependence; preserve shared-pair order and endpoint value selection',
    'P3': 'Lexicographic structural action ordering; family round-robin; verified pair slots 4 and 8; singles otherwise',
    'action_priority': ['known repeated loop work first', 'loops without proven carrying dependence first (unknown is not independence)',
                        'larger compiler trip count first', 'more arithmetic operations first', 'shallower loop first', 'seeded identity tie'],
    'value_order': 'Existing nearest lower/upper/disable alternatives unchanged. No QoR-derived value priorities.',
    'budget': 'Baseline, failed attempts and authenticated cache reveals each cost one logical query; isolated histories',
    'physical_context': 'Unchanged R1 engine code hashes define physical evaluations; the full V2.1 policy code and graph hashes are separately sealed in shared inputs and protocol',
    'limitations': 'GAN/covariance/Kalman are previously studied development cases. This heuristic is exploratory, not held-out evidence.'}


def code_hashes(protocol=None):
    result = benchmark.core_hashes()
    for filename in ('experiments/v21_refinement/__init__.py', 'experiments/v21_refinement/experiment.py',
                     'experiments/refinement_search/benchmark.py', 'GNN_branch/mlir_graph_v2.py',
                     'GNN_branch/mlir_semantics.py', 'GNN_branch/action_point_relations.py',
                     'GNN_branch/generate_mlir_v2_corpus.py', 'GNN_branch/generate_mlir_dataset.py',
                     'experiments/structural_audit/compare_v2.py', 'experiments/structural_audit/audit.py'):
        result[filename] = c.sha256(c.ROOT / filename)
    if protocol:
        result.update(protocol['artifact_sha256'])
        result[protocol['protocol_path']] = c.sha256(c.repo_path(protocol['protocol_path']))
    return result


def verified_pairs(graph, old_relations, actions, artifact_hash):
    """Adapt typed queries, preserving all direction/context evidence per pair.

    Shared pairs retain their *ordering bucket* from P1. Their evidence class
    remains the actual V2.1 class; an indirect access is never called direct to
    preserve a queue position. Only eligibility is changed in P2.
    """
    if graph.graph['schema'] != v2.SCHEMA_V21:
        raise ValueError('Relationship replacement requires V2.1')
    _, facts = checked_views(graph)
    old = {frozenset((r['action_i'], r['action_j'])): r for r in old_relations
           if r['verified_relation_class'] in VERIFIED}
    grouped = defaultdict(list)
    for label, rows in facts.items():
        for row in rows:
            other = row.get('neighbor')
            if (row['evidence_status'] == 'proven' and row['relation_type'] in
                    ('memory_access', 'loop_hierarchy', 'memory_dependence')
                    and other in actions and label in actions and label != other):
                grouped[frozenset((label, other))].append(row)
    pairs = []
    for key, rows in grouped.items():
        inherited = old.get(key)
        left, right = (inherited['action_i'], inherited['action_j']) if inherited else sorted(key, key=lambda s:int(s[1:]))
        local = any(r.get('access_path') != 'through_call' for r in rows)
        kind = 'direct_verified' if local else 'indirect_verified'
        pairs.append({'action_i': left, 'action_j': right, 'verified_relation_class': kind,
            'verified_relation_types': sorted({r['relation_type'] for r in rows}),
            'ordering_bucket': inherited['verified_relation_class'] if inherited else kind,
            'compiler_evidence_source': v2.SCHEMA_V21, 'artifact_hash': artifact_hash,
            'typed_evidence': sorted(rows, key=v2.canonical_json)})
    return sorted(pairs, key=lambda r:(int(r['action_i'][1:]),int(r['action_j'][1:])))


def pair_proposals(problem, reference, seed):
    """Reuse the legacy endpoint proposals and round-robin; change eligibility only."""
    pairs = sorted(problem['v21_pairs'], key=lambda r:(r['ordering_bucket'] != 'direct_verified',
        {problem['actions'][r['action_i']].kind, problem['actions'][r['action_j']].kind} != {'loop','array'},
        p._tie(seed, [r['action_i'],r['action_j']])))
    queues = [p.structural_proposals({**problem, 'relations':[relation]}, reference, seed) for relation in pairs]
    result, seen = [], set()
    while any(queues):
        for queue in queues:
            if queue:
                row = queue.pop(0)
                if row['signature'] not in seen:
                    row = {**row, 'selection_reason':'P2: proven V2.1 eligibility; unchanged legacy value and shared-pair ordering'}
                    result.append(row); seen.add(row['signature'])
    return result


def action_features(graph):
    """Compiler facts only. Operation counts are context counts, not hardware cost."""
    features = {}
    for node, point in graph.nodes(data=True):
        if point['category'] != 'action_point':
            continue
        label = point['action_id']
        target = next(v for _,v,e in graph.out_edges(node,data=True) if e['relation_type']=='targets')
        record = {'kind':point['action_kind'], 'target':target,
                  'trip_count':point.get('compiler_proven_trip_count'), 'arithmetic_operations':0,
                  'loop_depth':len(graph.nodes[target].get('loop_context',[])), 'proven_carrying_dependences':0}
        if point['action_kind'] == 'loop':
            loop = get_loop_semantics(graph,label)
            operation_nodes = [a for _,a in graph.nodes(data=True) if a['category']=='operation'
                               and target in a.get('loop_context',[])]
            record['arithmetic_operations'] = sum(a.get('operation_family') in ('arithmetic','arith') or
                str(a.get('text','')).startswith(('arith.','math.')) for a in operation_nodes)
            record['arithmetic_context_scope'] = 'Local and nested body; callee arithmetic is not counted as caller-local work'
            record['proven_carrying_dependences'] = sum(r.get('evidence_status')=='proven' and
                r.get('carrying_loop')==target for r in loop['memory_dependences'])
            record['concrete_accesses'] = len(loop['accesses'])
            record['ssa_state_count'] = len(loop['ssa_state'])
        features[label] = record
    return features


def action_key(label, features, seed):
    row = features.get(label,{})
    trip = row.get('trip_count')
    repeated = isinstance(trip,int) and trip > 1
    return (not repeated, bool(row.get('proven_carrying_dependences')),
            -(trip or 0), -row.get('arithmetic_operations',0), row.get('loop_depth',0), p._tie(seed,label))


def prioritized(problem, reference, seed, joint=False):
    rows = pair_proposals(problem,reference,seed) if joint else p.single_proposals(problem,reference,seed)
    features = problem['v21_features']
    def key(row):
        labels = row['reference_changed_actions']
        return min(action_key(label,features,seed) for label in labels)
    if joint:
        # Stable sort: retain endpoint value/direction order within an action pair.
        rows.sort(key=key)
    else:
        groups = {family:[] for family in p.FAMILIES}
        for row in rows:
            groups[row['family']].append(row)
        for group in groups.values(): group.sort(key=key)
        rows = []
        while any(groups.values()):
            for family in p.FAMILIES:
                if groups[family]: rows.append(groups[family].pop(0))
    return [{**row, 'selection_reason':'P3: frozen lexicographic compiler-context priority; ' + row['selection_reason'],
        'structural_priority':{'key':list(key(row)), 'actions':{label:features.get(label,{}) for label in row['reference_changed_actions']}}} for row in rows]


def choose(policy, problem, history, incumbent, seed, original_choose):
    if not history or policy in ('coordinate','structural'):
        return original_choose('coordinate' if policy.startswith('v21_') else policy,problem,history,incumbent,seed)
    if policy not in POLICIES:
        raise ValueError('Unsupported V2.1 comparison policy')
    seen = {r['signature'] for r in history}
    if policy == 'v21_relations':
        queues = [pair_proposals(problem,incumbent,seed)]
    else:
        structural_slot = (len(history)+1) in (4,8)
        queues = [prioritized(problem,incumbent,seed,joint=structural_slot),
                  prioritized(problem,incumbent,seed,joint=not structural_slot)]
    return next((r for queue in queues for r in queue if r['signature'] not in seen),None)


@contextmanager
def runtime(protocol):
    """Scope extensions to this experiment; original engine/checkpoint bytes stay fixed."""
    old_hash, old_choose, old_names, old_problem, old_measure, old_candidate = (
        ctl.code_hashes,p.choose,p.POLICIES,ctl.problem_for,ctl.measure,ctl.physical_candidate)
    ctl.code_hashes = lambda:code_hashes(protocol)
    p.POLICIES = POLICIES
    def problem_for(manifest,shared):
        problem = old_problem(manifest,shared)
        record = protocol['kernels'][manifest['kernel']]
        return {**problem,'v21_pairs':record['pairs'],'v21_features':record['features']}
    ctl.problem_for = problem_for
    p.choose = lambda policy,problem,history,incumbent,seed=123: choose(policy,problem,history,incumbent,seed,old_choose)
    def physical_candidate(manifest,shared,proposal):
        candidate = old_candidate(manifest,shared,proposal)
        # The unchanged vendor runner, materializer, domains and source context
        # define a physical evaluation. Proposal-only code is authenticated by
        # the sealed policy manifest, rather than changing identical hardware
        # receipt identities. All original receipt/XML checks still execute.
        candidate['evaluation_context_hash'] = c.object_hash({
            'source_context':existing.hls.source_context(shared['kernels'][manifest['kernel']]),
            'signature':proposal['signature'],'search_code':benchmark.core_hashes()})
        return candidate
    ctl.physical_candidate = physical_candidate
    def measure(manifest,shared,proposal,query_id,root,executable,timeout):
        cached = benchmark._cached_measurement(protocol,manifest,shared,proposal,root)
        if cached is not None:
            return cached, {'physical_synthesis_status':'v21_authenticated_reuse','new_physical_synthesis_runs':0,
                'authenticated_measurement_reuses':1,'physical_synthesis_runtime_seconds':0.0}
        return old_measure(manifest,shared,proposal,query_id,root,executable,timeout)
    ctl.measure = measure
    try:
        yield
    finally:
        ctl.code_hashes,p.choose,p.POLICIES,ctl.problem_for,ctl.measure,ctl.physical_candidate = (
            old_hash,old_choose,old_names,old_problem,old_measure,old_candidate)


def source_problem(name,kernel,native):
    raw = domains.load_source_actions(name,c.repo_path(kernel['source_provenance']['snapshot_metadata']).parent)
    verified,inactive = legacy.audit_kernel(kernel,raw)
    actions = c.compiler_actions(kernel,raw)
    return {'baseline':kernel['initial_assignment'],'actions':actions,'native_domains':native,
            'relations':verified,'inactive_actions':inactive}


def graph_for(name,kernel,corpus):
    meta = kernel['relations'].get('compiler_active_metadata_path') or kernel['source_provenance']['snapshot_metadata']
    matches = [r for r in corpus['records'].values() if r.get('validation')=='pass' and
        r['source_sha256']==kernel['source_provenance']['source_sha256'] and r['metadata_sha256']==c.sha256(c.repo_path(meta))]
    if not matches:
        raise ValueError('No authenticated V2.1 source/active-metadata snapshot for '+name)
    # Local arithmetic analysis headers must match the selected corpus alias;
    # synthesis separately authenticates genuine Vitis headers in its receipts.
    for row in matches:
        graph = v2.read_graph(row['canonical'])
        if c.sha256(row['canonical']) != row['canonical_sha256'] or graph.graph['schema'] != v2.SCHEMA_V21:
            raise ValueError('Corpus artifact changed or wrong graph schema')
        source = c.repo_path(row['source'])
        from GNN_branch.generate_mlir_v2_corpus import authenticate
        authenticate(row['canonical'],row,corpus['toolchain'])
        if c.sha256(source)!=row['source_sha256'] or c.sha256(row['metadata'])!=row['metadata_sha256']:
            raise ValueError('Corpus source/metadata changed')
        return row,graph


def action_report(name,graph,problem):
    _, facts = checked_views(graph)
    records = []
    for label in sorted(facts,key=lambda s:int(s[1:])):
        point = dict(graph.nodes['action:'+label]); target = next(v for _,v,e in graph.out_edges('action:'+label,data=True) if e['relation_type']=='targets')
        fields = {k:v for k,v in problem['baseline'].items() if k.endswith('_'+label+'}')}
        action = problem['actions'][label]
        legal = domains.tcad.domains_from_actions({label:action})
        records.append({'kernel':name,'action_point':label,'compiler_point':point,'compiler_target':dict(graph.nodes[target]),
            'compiler_relationships':facts[label], 'loop_semantics':get_loop_semantics(graph,label) if action.kind=='loop' else None,
            'local_and_nested_computations':[{'node':n,'operation':a.get('text'),'types':a.get('result_types'),
                'source_line':a.get('source_line')} for n,a in graph.nodes(data=True)
                if a['category']=='operation' and target in a.get('loop_context',[])],
            'current_SALT_directives':fields,'source_derived_legal_values':legal,
            'legality_note':'Array fields form dimension-dependent tuples; individual vocabulary membership is insufficient.',
            'numerical_fidelity':graph.graph.get('numerical_fidelity')})
    return records


def prepare(args):
    output = c.repo_path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output exists; resume it or choose a fresh directory')
    if args.max_evaluations not in (1,2,4,8): raise ValueError('Pilot budget must be 1, 2, 4 or 8')
    original,_ = existing.load_output(c.repo_path(args.input_dir))
    corpus_path = c.repo_path(args.corpus)/'corpus_manifest.json'
    corpus = json.loads(corpus_path.read_text())
    if corpus['schema'] != v2.SCHEMA_V21: raise ValueError('V2.1 corpus is required')
    output.mkdir(parents=True)
    kernels,artifacts,proposals,comparisons,reports = {},{},[],[],[]
    for name in args.kernels:
        kernel = copy.deepcopy(original['kernels'][name])
        problem = source_problem(name,kernel,original['native_domains'][name])
        row,graph = graph_for(name,kernel,corpus)
        graph_path = output/'graphs'/(name+'.json'); v2.write_graph(graph,graph_path)
        # Compare the compiler-proven trip-count contract, never silently change P0/P1 domains.
        active = {a['action_id'] for _,a in graph.nodes(data=True) if a['category']=='action_point'}
        if active != set(problem['actions'])-set(problem['inactive_actions']):
            raise ValueError('V1/V2.1 active action-point contract differs for '+name)
        for label in active:
            proven = graph.nodes['action:'+label].get('compiler_proven_trip_count')
            if proven is not None and problem['actions'][label].trip_count != proven:
                raise ValueError('Different proven counts would confound P2 relationship replacement: '+name+'/'+label)
        pairs = verified_pairs(graph,problem['relations'],problem['actions'],c.sha256(graph_path))
        features = action_features(graph)
        problem.update(v21_pairs=pairs,v21_features=features)
        old = {tuple(sorted((r['action_i'],r['action_j']),key=lambda s:int(s[1:]))) for r in problem['relations'] if r['verified_relation_class'] in VERIFIED}
        new = {tuple(sorted((r['action_i'],r['action_j']),key=lambda s:int(s[1:]))) for r in pairs}
        kernels[name] = {'graph':c.display_path(graph_path),'graph_sha256':c.sha256(graph_path),
            'source_sha256':row['source_sha256'],'compiler_metadata_sha256':row['metadata_sha256'],
            'synthesis_metadata_sha256':kernel['source_provenance']['metadata_sha256'],
            'inactive_actions':problem['inactive_actions'],'pairs':pairs,'features':features,
            'relationship_coverage':{'shared':sorted(old&new),'v1_only':sorted(old-new),'v21_only':sorted(new-old)}}
        artifacts[c.display_path(graph_path)] = c.sha256(graph_path)
        reports += action_report(name,graph,problem)
        queues = {'coordinate':p.single_proposals(problem,problem['baseline'],args.seed),
                  'structural':p.structural_proposals(problem,problem['baseline'],args.seed),
                  'v21_relations':pair_proposals(problem,problem['baseline'],args.seed),
                  'v21_priority':prioritized(problem,problem['baseline'],args.seed)}
        for policy,queue in queues.items():
            proposals += [{'kernel':name,'policy':policy,'initial_queue_rank':i+1,**proposal} for i,proposal in enumerate(queue)]
            comparisons.append({'kernel':name,'policy':policy,'initial_proposals':len(queue),
                'action_points_covered':sorted({label for r in queue for label in r['reference_changed_actions']}),
                'modification_types':dict(Counter(r.get('family','joint') for r in queue))})
        # Record P3's separate joint slot pool rather than implying its singles queue is the whole search.
        proposals += [{'kernel':name,'policy':'v21_priority_joint_pool','initial_queue_rank':i+1,**proposal}
                      for i,proposal in enumerate(prioritized(problem,problem['baseline'],args.seed,True))]
        controls = benchmark.matched_proposals({**problem,'relations':pairs},problem['baseline'],args.seed)
        proposals += [{'kernel':name,'policy':'matched_control_diagnostic_pool','initial_queue_rank':i+1,**proposal}
                      for i,proposal in enumerate(controls)]
        comparisons.append({'kernel':name,'policy':'matched_control_diagnostic_pool',
            'initial_proposals':len(controls),'execution':'prepared only; outside the four-arm budget'})
    c.write_jsonl(output/'candidate_proposals.jsonl',proposals)
    c.write_jsonl(output/'action_point_reports.jsonl',reports)
    csv_rows(output/'candidate_comparison.csv',comparisons)
    for path in (output/'candidate_proposals.jsonl',output/'action_point_reports.jsonl',output/'candidate_comparison.csv'):
        artifacts[c.display_path(path)] = c.sha256(path)
    c.write_json(output/'corpus_manifest.json',corpus)
    for filename in ('graph_coverage.csv','relation_coverage.csv'):
        (output/filename).write_bytes((c.repo_path(args.corpus)/filename).read_bytes())
    # No measurement loader has run yet. Freeze this protocol and complete initial pools now.
    protocol = ctl._sealed({'experiment':'V2.1 structure-guided refinement','protocol_revision':1,
        'protocol_path':c.display_path(output/'protocol.json'),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
        'worktree_status':subprocess.check_output(['git','status','--porcelain'],cwd=c.ROOT,text=True),
        'code_sha256':code_hashes(),'artifact_sha256':artifacts,'kernels':kernels,
        'policies':list(POLICIES),'rules':RULES,'seed':args.seed,'max_evaluations':args.max_evaluations,
        'execution':args.execution,'context':c.CONTEXT,'corpus_manifest_sha256':c.sha256(corpus_path),
        'physical_cache_roots':list(dict.fromkeys(args.cache_search)),
        'synthesis_input_manifest_sha256':c.sha256(c.repo_path(args.input_dir)/'manifest.json')})
    c.write_json(output/'protocol.json',protocol)
    with runtime(protocol):
        ctl.prepare(argparse.Namespace(input_dir=args.input_dir,output_dir=str(output/'search'),
            measurements=args.measurements,kernels=args.kernels,policies=list(POLICIES),
            max_evaluations=args.max_evaluations,seed=args.seed,execution=args.execution,fresh_synthesis=False))
    summarize(output)


def load_protocol(output):
    protocol = ctl._read_sealed(Path(output)/'protocol.json')
    if protocol['code_sha256'] != code_hashes(): raise ValueError('Frozen implementation changed; prepare a new directory')
    for path,digest in protocol['artifact_sha256'].items():
        if c.sha256(c.repo_path(path))!=digest: raise ValueError('Frozen artifact changed: '+path)
    if protocol['context']!=c.CONTEXT or protocol['rules']!=RULES: raise ValueError('Protocol context/rules changed')
    return protocol


def execute(args):
    output = c.repo_path(args.output_dir); protocol = load_protocol(output)
    expected = 'replay' if args.mode=='replay' else 'synthesis'
    if protocol['execution']!=expected: raise ValueError('Execution differs from frozen mode')
    names = args.kernels or list(protocol['kernels']); policies = args.policies or list(POLICIES)
    if not set(names)<=set(protocol['kernels']) or not set(policies)<=set(POLICIES): raise ValueError('Unknown selected context')
    if args.mode=='run' and len(names)*len(policies)>1 and not args.allow_batch:
        raise ValueError('First run one kernel/policy smoke test; prospective batches require --allow-batch')
    with runtime(protocol):
        for name in names:
            for policy in policies:
                ctl.search(output/'search'/name/policy,executable=args.vitis_hls,timeout=args.timeout,limit=args.limit)
    summarize(output)


def summarize(output):
    output = Path(output); protocol = load_protocol(output)
    rows,traces,measurements = [],[],[]
    with runtime(protocol):
        for name in protocol['kernels']:
            for policy in POLICIES:
                path = output/'search'/name/policy
                rows += [r for r in ctl.summarize(path) if r['budget']<=protocol['max_evaluations']]
                traces += [{'kernel':name,'policy':policy,**r} for r in c.read_jsonl(path/'search_trace.jsonl')]
                measurements += [{'kernel':name,'policy':policy,'validation_status':'not_tested',**r} for r in c.read_jsonl(path/'measurements.jsonl')]
    csv_rows(output/'refinement_summary.csv',rows)
    c.write_jsonl(output/'search_traces.jsonl',traces); c.write_jsonl(output/'synthesis_results.jsonl',measurements)
    lines = ['# V2.1 corpus and refinement validation','',
        'Compiler evidence, legal proposal rationale and measured hardware evidence are separate. '
        'These three kernels are development diagnostics, not independent held-out confirmation.', '',
        '## Frozen comparison','', *[f'- {key}: {RULES[key]}' for key in ('P0','P1','P2','P3')], '',
        'P3 retains nearest balanced legal values. Its lexicographic rules are an exploratory hypothesis, not a QoR predictor. '
        'Pair eligibility correction is isolated in P2; P3 additionally changes ranking and allocation.', '',
        '| Kernel | Policy | Budget | Coverage | Best ADP | Improvement % | Logical | New HLS | Reuse |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+' | '.join(str(row[k]) for k in ('kernel','policy','budget','status','best_ADP','improvement_pct','logical_evaluations','new_physical_synthesis_runs','authenticated_measurement_reuses'))+' |')
    lines += ['', '## Relationship eligibility','']
    for name,kernel in protocol['kernels'].items():
        lines.append(f"- {name}: {json.dumps(kernel['relationship_coverage'])}; frozen inactive actions {kernel['inactive_actions']}.")
    # Only now, after protocol freezing, perform the requested known-edit ordering diagnostic.
    pool = c.read_jsonl(output/'candidate_proposals.jsonl')
    lines += ['', '## Retrospective ordering diagnostic','',
        'The following source edits were specified by prior experiments. They were not inputs to selection. '
        'A missing initial rank means the existing nearest-value neighborhood did not contain that edit; '
        'a feedback-reanchored search may reach it later.']
    for name,field,value in (('GAN',c.field_key('UNROLL','L10'),'4'),('covariance',c.field_key('PIPE','L6'),'1')):
        for policy in POLICIES:
            found = [r['initial_queue_rank'] for r in pool if r['kernel']==name and r['policy']==policy and
                r['reference_changed_fields']==[field] and str(r['assignment'].get(field))==value]
            lines.append(f'- {name} {field}={value}, {policy}: '+(str(found) if found else 'absent from initial pool'))
    corpus = json.loads((output/'corpus_manifest.json').read_text())
    counts = Counter(r['status'] for r in corpus['records'].values())
    completed = [r for r in corpus['records'].values() if r.get('validation')=='pass']
    lines += ['', '## Corpus coverage','',f"{len(corpus['inventory']['snapshots'])} inventoried snapshots; {dict(counts)}.",
        f"Registered action points: {sum(r['coverage']['registered'] for r in completed)}; "
        f"compiler-active: {sum(r['coverage']['compiler_active'] for r in completed)}; "
        f"unresolved registered targets: {sum(r['coverage']['compiler_unresolved'] for r in completed)}. Counts include distinct metadata/header snapshots.",
        'See graph_coverage.csv and relation_coverage.csv for exact snapshot identities, sparse action alignment, '
        'unresolved roots, call bindings and dependencies. Canonical/compact equivalence is checked for every successful graph.', '',
        'Blocked cases: '+json.dumps([{'kernel':r['kernel'],'error':r.get('error','')[-700:]} for r in corpus['records'].values() if r['status'] in ('blocked','invalid_artifact')]), '',
        '## Interpretation and limitations','',
        'Matched-control pools are prepared with the existing matcher (same types, field families/counts and domain novelty, nearest factor magnitude). '
        'Their hardware comparisons are pending; they are not free observations inside any main policy.', '',
        'Unavailable replay observations are not failures or numerical QoR. Incomplete curves cannot establish policy superiority. '
        'No functional simulation, RTL equivalence or post-route timing is established by these records. '
        'Analysis arithmetic stubs are structurally useful and not hardware-exact; HLS reuse separately authenticates vendor headers. '
        'Draft semantic annotations are not independently reviewed ground truth. '
        'Absent verified relations do not prove independence. Dependence intervals are not exact distances. '
        'A V2.1 encoder remains a separate controlled learning experiment; structural correctness alone does not establish predictive utility.', '',
        f"Logical observations: {len(measurements)}; new physical runs: {sum(r.get('new_physical_synthesis_runs',0) for r in measurements)}; "
        f"authenticated reuses: {sum(r.get('authenticated_measurement_reuses',0) for r in measurements)}."]
    (output/'V21_REFINEMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','replay','run','summarize'),default='prepare')
    parser.add_argument('--input-dir',default=str(ctl.DEFAULT_INPUT)); parser.add_argument('--corpus',default=str(CORPUS))
    parser.add_argument('--output-dir',default=str(DEFAULT_OUTPUT))
    parser.add_argument('--kernels',nargs='+'); parser.add_argument('--policies',nargs='+',choices=POLICIES)
    parser.add_argument('--measurements',action='append',default=[])
    parser.add_argument('--cache-search',action='append',default=[],help='Existing physical cache root; hidden until a paid query')
    parser.add_argument('--execution',choices=('replay','synthesis'),default='replay')
    parser.add_argument('--max-evaluations',type=int,default=8); parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--vitis-hls',default='vitis_hls'); parser.add_argument('--timeout',type=int,default=21600)
    parser.add_argument('--limit',type=int); parser.add_argument('--allow-batch',action='store_true')
    args = parser.parse_args(argv)
    if args.timeout<=0 or args.limit is not None and args.limit<1: parser.error('Positive timeout/query limit required')
    try:
        if args.mode=='prepare':
            args.kernels = args.kernels or ['GAN','covariance','Kalman']; prepare(args)
        elif args.mode=='summarize': summarize(c.repo_path(args.output_dir))
        else: execute(args)
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print('V2.1 refinement: '+str(exc),file=sys.stderr); return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
