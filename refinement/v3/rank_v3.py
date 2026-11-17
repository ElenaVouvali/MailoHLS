"""Small deterministic proposal contracts. Proxies are not latency predictions."""
import copy
import math
import re
import semantic_v3 as sem


def fingerprint(result):
    return dict(metrics=result.get('metrics'), schedule=result.get('schedule_groups'),
                constraints=[{k: d.get(k) for k in ('type','canonical_memory','scheduled_loop',
                             'access_owning_loop','source_line')} for d in result.get('bottleneck_families', [])])


def propose(base, m, incumbent, history, enable_scout=False):
    cfg = incumbent['directives']
    result = incumbent['result']
    metrics = result.get('metrics') or {}
    by = {a['action_id']: a for a in cfg}
    source = m['semantic_v3']
    graph_mode = m['policy'] == 'graph'
    anchors = set(m['compiler_graph']['anchored_actions']) if graph_mode else set(m['actions'])
    used = {r['signature'] for r in history}
    accepted, rejected = {}, []
    schedules = result.get('schedule_groups', {})

    def reject(family, action, reason, evidence=None):
        rejected.append(dict(family=family, action=action, reason=reason, evidence=evidence))

    def add(lid, values, family, evidence, tradeoff, risk='medium', exploratory=False):
        if lid not in anchors:
            return reject(family, lid, 'Missing editable action anchor')
        candidate = copy.deepcopy(cfg)
        for row in candidate:
            if row['action_id'] == lid:
                row.update(values)
        candidate = base.validate(candidate, m['actions'])
        sig = base.signature(candidate)
        if sig in used:
            return reject(family, lid, 'Duplicate full candidate signature already charged')
        # Do not repeat a local unsuccessful family in an unchanged evidence context.
        for old in history:
            selection = old.get('selection', {})
            if selection.get('family') == family and selection.get('changed_actions') == [lid]:
                prior_context = selection.get('context_fingerprint')
                unsuccessful = (not old['result'].get('metrics') or
                                old['result']['metrics']['latency_best_cycles'] >= metrics['latency_best_cycles'])
                if unsuccessful and (old.get('outcome') == 'hardware-insensitive / neutral' or prior_context == base.signature(fingerprint(result))):
                    return reject(family, lid, 'Unsuccessful local family; incumbent diagnostics/schedule unchanged')
        leverage = sem.criticality(m, metrics, evidence.get('scheduled_loop', lid))
        # Lexicographic, inspectable categories; no calibrated numeric QoR oracle.
        breakdown = dict(evidence='exploratory' if exploratory else 'exact_log+measured_II+source'+('+LLVM' if graph_mode else ''),
                         top_level_leverage=leverage, parallelism_tradeoff=tradeoff,
                         resource_risk=risk, modified_action_count=1)
        tier = (1 if exploratory else 0,
                0 if leverage['enclosing_report_context'] else 1,
                0 if tradeoff == 'lanes_preserved' else 1,
                1 if risk == 'high' else 0, 1, sig)
        proposal = dict(signature=sig, directives=candidate, family=family, kind=family,
                        changed_actions=[lid], changed={lid: dict(old=by[lid], new=next(a for a in candidate if a['action_id']==lid))},
                        evidence=evidence, score_breakdown=breakdown, rank_key=list(tier), risk=risk,
                        exploratory=exploratory, context_fingerprint=base.signature(fingerprint(result)),
                        claim='supported intervention hypothesis' if not exploratory else 'exploratory',
                        expected_effect='Uncertain; requires approved synthesis; no quantitative kernel latency prediction')
        if sig not in accepted or tier < tuple(accepted[sig]['rank_key']):
            accepted[sig] = proposal

    for d in result.get('bottleneck_families', []):
        loop = d.get('scheduled_loop')
        schedule = schedules.get(loop)
        if d['type'] != 'memory_ports':
            reject('pipeline_relocation', loop, 'Disabled: no positive compiler dependence-safety proof and end-to-end rationale')
            continue
        if not schedule or not schedule['source_loop'] or not schedule['stalled']:
            reject('memory_relief', loop, 'No explicitly named source loop with measured II > target')
            continue
        access = sem.source_access(m, d, loop)
        if not access:
            reject('memory_relief', loop, 'Missing exact source-line access/verified containment')
            continue
        evidence = dict(scheduled_loop=loop, diagnostic=d, schedule=schedule, source_access=access,
                        physical_cause_proven=False)
        # A1: existing partition dimension and affine unit-stride access only.
        for lid, action in by.items():
            if action['kind'] != 'array' or action['variable'] != d['canonical_memory'] or lid not in anchors:
                continue
            if graph_mode and not m['compiler_graph']['array_scope_access_links'].get(lid):
                reject('banking', lid, 'No graph array-scope/IR access link')
                continue
            dim = action['partition_dim']
            bound = m['actions'][lid]['dims'].get(dim)
            owner = d['access_owning_loop']
            iterator = m['loop_spans'][owner]['iterator']
            indices = [a['indices'][dim-1].replace(' ', '') for a in access['accesses'] if 0 < dim <= len(a['indices'])]
            consecutive = any(re.fullmatch(re.escape(iterator)+r'(?:[+-]\d+)?', index) for index in indices)
            lanes = max(1, by[owner]['unroll_factor'])
            proof = dict(evidence, partition_dimension=dim, source_indices=indices, documented_stride=1,
                         lanes=lanes, array_action=lid)
            if graph_mode:
                proof['LLVM_access_links'] = m['compiler_graph']['array_scope_access_links'][lid]
                # Function-scoped source/IR relation must agree too.
                if not any(r['loop']==owner and r['array']==lid for r in m['loop_array_relations']):
                    reject('banking', lid, 'Source/LLVM owning function relation missing')
                    continue
            if not consecutive or not bound or lanes <= 1 or action['partition_type'] not in ('block','cyclic'):
                reject('banking', lid, 'No documented unit-stride unrolled lanes in the existing partition dimension')
                continue
            if max(metrics.get('utilization_percent', {}).values() or [100]) >= 90:
                reject('banking', lid, 'Insufficient measured resource headroom for extra banks')
                continue
            if action['partition_type'] == 'block':
                opts = [f for f in base.factors(bound) if f >= max(lanes, action['partition_factor'])]
                if opts:
                    add(lid, dict(partition_type='cyclic', partition_factor=min(opts)), 'banking', proof, 'lanes_preserved')
            else:
                nxt = base.higher_factor(action['partition_factor'], bound)
                if nxt:
                    add(lid, dict(partition_factor=nxt), 'banking', proof, 'lanes_preserved')
        # A2/A3: nearest controlling unroll only; one action per hypothesis.
        if loop not in by or loop not in anchors:
            continue
        local = by[loop]['unroll_factor'] > 1
        target = loop if local else next((p for p in list(sem.chain(source, loop))[1:]
                                            if by[p]['unroll_factor'] > 1), None)
        if not target or target not in anchors:
            continue
        link = sem.source_access(m, d, target)
        if not link:
            continue
        iterator = m['loop_spans'][target]['iterator']
        if not any(re.search(r'\b'+re.escape(iterator)+r'\b', index)
                   for a in link['accesses'] for index in a['indices']):
            reject('ancestor_concurrency', target, 'Source access does not show controlling iterator')
            continue
        old = by[target]['unroll_factor']
        lower = base.lower_factor(old, m['actions'][target]['bound'])
        if not lower:
            continue
        family = 'local_unroll' if local else 'ancestor_concurrency'
        proxy = dict(kind='noncalibrated_work_proxy', kernel_latency_prediction=False,
                     old_unroll=old, new_unroll=lower, trip_count=source['trip_counts'].get(target),
                     caveat='Lost parallel lanes can outweigh reduced II; exact loop criticality unknown')
        trip = source['trip_counts'].get(target, {}).get('count')
        if trip:
            old_groups, new_groups = math.ceil(trip/old), math.ceil(trip/lower)
            optimistic = new_groups*schedule['target_ii']['min']
            old_range = [old_groups*schedule['achieved_ii'][k] for k in ('min','max')]
            proportional = [new_groups*max(schedule['target_ii']['min'], math.ceil(schedule['achieved_ii'][k]*lower/old))
                            for k in ('min','max')]
            proxy.update(old_work_range=old_range, optimistic_new_work=optimistic,
                         proportional_relief_work_range=proportional)
            if optimistic >= old_range[1]:
                reject(family, target, 'Even optimistic target-II work proxy cannot improve', proxy)
                continue
            if local and proportional[0] >= old_range[0]:
                reject(family, target, 'Proportional II relief trades away parallelism; marginal/no credible latency win', proxy)
                continue
        elif local:
            reject(family, target, 'No corresponding literal source trip count; end-to-end benefit unsupported', proxy)
            continue
        add(target, dict(unroll_factor=lower), family, dict(evidence, controlling_access=link, proxy=proxy),
            'lanes_lost_II_relief_uncertain', 'high')

    # A5: relief only in a measured ineligible context and with source replication.
    pressure = metrics and (not metrics.get('budget_feasible') or metrics.get('hls_estimated_clock_within_target') is False)
    if pressure:
        for lid, action in by.items():
            if action['kind'] != 'loop' or action['unroll_factor'] <= 1 or lid not in anchors:
                continue
            if not any(r['accesses'] for r in source['direct_statements'].get(lid, [])):
                continue
            replication = [r for r in result.get('compiler_replication', []) if r['loop']==lid and r['factor']==action['unroll_factor']]
            if not replication:
                reject('feasibility_relief',lid,'No observed compiler replication to reduce')
                continue
            lower = base.lower_factor(action['unroll_factor'], m['actions'][lid]['bound'])
            add(lid, dict(unroll_factor=lower), 'feasibility_relief',
                dict(measured_ineligible=metrics, source_replication_loop=lid, compiler_replication=replication), 'lanes_lost_feasibility_recovery', 'high')
    # A6: one opt-in reduction scout on an exact measured top-level source loop.
    if enable_scout and metrics.get('budget_feasible') and metrics.get('hls_estimated_clock_within_target') is True and not accepted and not any(r.get('selection', {}).get('exploratory') for r in history):
        for row in sorted(metrics.get('loop_summary', []), key=lambda r: -(r.get('latency') or 0)):
            lid = row['name']
            if (lid not in anchors or lid not in source['loop_parent'] or source['loop_parent'][lid] is not None or
                    not row.get('latency') or row['latency'] < metrics['latency_best_cycles']*.2 or
                    by[lid]['unroll_factor'] <= 1 or max(metrics['utilization_percent'].values()) >= 90):
                continue
            lower = base.lower_factor(by[lid]['unroll_factor'], m['actions'][lid]['bound'])
            add(lid, dict(unroll_factor=lower), 'scout_replication_reduction',
                dict(measured_high_impact_loop=row, scheduled_loop=lid), 'unknown_exploratory', 'high', True)
            break
    ordered = sorted(accepted.values(), key=lambda p: tuple(p['rank_key']))
    for rank, p in enumerate(ordered, 1):
        p['rank'] = rank
    return ordered, rejected
