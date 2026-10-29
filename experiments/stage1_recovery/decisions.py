"""Experimental legal action choices and value-only scoring; no new HLS rules."""
from __future__ import annotations

import collections
import re

import torch
import torch.nn.functional as F

from LLM_branch.common import mailohls_contract
from LLM_branch.train import train_SFT_xattn_new as production


def parse(text):
    return {lhs.upper(): rhs for lhs, rhs in re.findall(
        r'(auto\{_[A-Z_]+_L\d+\})\s*=\s*(\w+)', text, flags=re.IGNORECASE)}


def action_plan(source):
    groups = collections.OrderedDict()
    for label, lhs in production.extract_ordered_lhs_plan(source):
        groups.setdefault(label, []).append(lhs)
    return list(groups.items())


def legal_actions(kernel, fields, registry, prior=None):
    """Enumerate current source domains using the existing semantics filter.

    In particular PIPE and UNROLL may both be positive. Dataset exclusivity is
    not introduced as a compiler legality constraint.
    """
    rows = [dict(prior or {})]
    prior_keys = set(prior or {})
    for lhs in fields:
        new = []
        for chosen in rows:
            _, values = production.constrained_rhs_candidates(kernel, lhs, registry, chosen)
            for value in values:
                new.append(dict(chosen, **{lhs.upper(): value}))
        rows = new
    return [{k: v for k, v in row.items() if k not in prior_keys} for row in rows]


def encode_action(tok, fields, action):
    ids, positions = [], []
    for lhs in fields:
        ids.extend(tok(f'{lhs} = ', add_special_tokens=False)['input_ids'])
        value = tok(action[lhs.upper()], add_special_tokens=False)['input_ids']
        positions.extend(range(len(ids), len(ids) + len(value)))
        ids.extend(value)
        ids.extend(tok('\n', add_special_tokens=False)['input_ids'])
    return ids, positions


def action_score(model, prefix_ids, action_ids, positions):
    """Differentiable sum log probability of values; fixed syntax is context."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(prefix_ids) + action_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    indices = torch.tensor(positions, dtype=torch.long, device=device)
    chosen = logits[0, len(prefix_ids) + indices - 1].float()
    target = ids[0, len(prefix_ids) + indices]
    return F.log_softmax(chosen, dim=-1).gather(-1, target[:, None]).sum()


@torch.no_grad()
def score_actions_batch(model, tok, prefix, fields, candidates, batch_size=4):
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        encoded = [encode_action(tok, fields, c) for c in chunk]
        width = len(prefix) + max(len(ids) for ids, _ in encoded)
        inputs = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long, device=device)
        masks = torch.zeros_like(inputs)
        for b, (ids, _) in enumerate(encoded):
            values = prefix + ids
            inputs[b, :len(values)] = torch.tensor(values, device=device)
            masks[b, :len(values)] = 1
        logits = model(input_ids=inputs, attention_mask=masks).logits
        for b, (candidate, (ids, positions)) in enumerate(zip(chunk, encoded)):
            indices = torch.tensor(positions, device=device) + len(prefix)
            chosen = logits[b, indices - 1].float()
            score = F.log_softmax(chosen, dim=-1).gather(-1, inputs[b, indices, None]).sum()
            result.append((float(score), candidate, ids))
    return result


def training_actions(sample, row, tok, registry):
    pack = production.build_deterministic_rhs_pack(row['input'], row['target'], tok,
                directive_domain_registry=registry, kernel_name=row['kernel_name'])
    offset = len(sample['input_ids']) - len(pack.input_ids)
    gold = parse(row['target'])
    prior, result = {}, []
    for label, fields in action_plan(row['input']):
        choices = legal_actions(row['kernel_name'], fields, registry, prior)
        expected = {lhs.upper(): gold[lhs.upper()] for lhs in fields}
        if expected not in choices:
            raise ValueError(f'Gold action outside legal source domain: {label}')
        if len(choices) > 1:
            first_value = next(i for i, key in enumerate(pack.site_keys) if key == fields[0].upper())
            start = offset + first_value - len(tok(f'{fields[0]} = ', add_special_tokens=False)['input_ids'])
            result.append(dict(label=label, prefix_ids=sample['input_ids'][:start].tolist(),
                               fields=fields, gold=expected, choices=choices))
        prior.update(expected)
    return result


@torch.no_grad()
def joint_decode(model, tok, row, registry):
    """Greedy between complete actions, exhaustive within each action."""
    fields = mailohls_contract.target_prompt_fields(row, device_mode='known')
    sections = mailohls_contract.build_prompt_sections(row['input'], row['obj_mode'], fields)
    prefix = [i for s in sections for i in tok(s, add_special_tokens=False)['input_ids']]
    prefix += mailohls_contract.selected_clock_response_token_ids(row, tok)
    chosen, lines, trace = {}, [], []
    for label, names in action_plan(row['input']):
        prefix += tok(f'<{label}>\n', add_special_tokens=False)['input_ids']
        candidates = legal_actions(row['kernel_name'], names, registry, chosen)
        scored = score_actions_batch(model, tok, prefix, names, candidates)
        score, best, ids = max(scored, key=lambda item: item[0])
        prefix += ids
        chosen.update(best)
        lines.append(f'<{label}>')
        lines.extend(f'{lhs} = {best[lhs.upper()]}' for lhs in names)
        trace.append(dict(label=label, choices=len(candidates), chosen=best, value_sum_logprob=score))
    mailohls_contract.validate_directive_assignments(chosen)
    return dict(kernel_name=row['kernel_name'], jsonl_idx=row['_jsonl_idx'],
                reference_target=row['target'], prediction='\n'.join(lines), trace=trace)


def modes_from_train(rows):
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        for lhs, value in parse(row['target']).items():
            counts[production.lhs_kind(lhs)][value] += 1
    return {kind: max(c, key=lambda v: (c[v], v)) for kind, c in counts.items()}


def core_balanced(metrics):
    """The four collapsing fields, without the often trivial dimension field."""
    kinds = metrics['decision_by_kind']
    values = [kinds[k]['balanced_accuracy'] for k in ['PIPE', 'UNROLL', 'ARRAY_T', 'ARRAY_F'] if k in kinds]
    if not values:
        raise ValueError('No core decision fields to score')
    return sum(values) / len(values)


def constant_baseline(row, registry, modes):
    chosen = {}
    for label, lhs in production.extract_ordered_lhs_plan(row['input']):
        _, legal = production.constrained_rhs_candidates(row['kernel_name'], lhs, registry, chosen)
        mode = modes[production.lhs_kind(lhs)]
        chosen[lhs.upper()] = mode if mode in legal else legal[0]
    return '\n'.join(f'{lhs} = {value}' for lhs, value in chosen.items())


def metrics(rows, registry=None):
    """Macro recall over observed reference classes, plus full-action accuracy."""
    if not rows:
        raise ValueError('Metrics require at least one example')
    confusions = collections.defaultdict(collections.Counter)
    decision_confusions = collections.defaultdict(collections.Counter)
    forced = 0
    actions_correct = actions_total = exact = 0
    for row in rows:
        gold, pred = parse(row['reference_target']), parse(row['prediction'])
        if set(gold) != set(pred):
            raise ValueError('Prediction keys differ from reference keys')
        exact += gold == pred
        groups = collections.defaultdict(list)
        prior = {}
        for lhs, value in gold.items():
            kind = production.lhs_kind(lhs)
            confusions[kind][(value, pred[lhs])] += 1
            if registry is not None:
                _, legal = production.constrained_rhs_candidates(row['kernel_name'], lhs, registry, prior)
                if value not in legal:
                    raise ValueError(f'Reference outside its source-derived domain: {lhs}={value}')
                if len(legal) > 1:
                    decision_confusions[kind][(value, pred[lhs])] += 1
                else:
                    forced += 1
            prior[lhs] = value
            groups[re.search(r'_(L\d+)\}', lhs).group(1)].append(lhs)
        for keys in groups.values():
            actions_total += 1
            actions_correct += all(gold[k] == pred[k] for k in keys)
    def kind_metrics(kind, confusion):
        totals = collections.Counter()
        right = collections.Counter()
        predictions = collections.Counter()
        for (gold, pred), n in confusion.items():
            totals[gold] += n
            right[gold] += n * (gold == pred)
            predictions[pred] += n
        recalls = {v: right[v] / n for v, n in totals.items()}
        return dict(accuracy=sum(right.values()) / sum(totals.values()),
                             balanced_accuracy=sum(recalls.values()) / len(recalls),
                             per_value_recall=recalls, count=sum(totals.values()),
                             prediction_counts=dict(predictions))
    by_kind = {kind: kind_metrics(kind, confusion) for kind, confusion in confusions.items()}
    decision_kinds = {kind: kind_metrics(kind, confusion) for kind, confusion in decision_confusions.items()}
    return dict(examples=len(rows), exact_design_accuracy=exact / len(rows),
                joint_action_accuracy=actions_correct / actions_total,
                macro_kind_balanced_accuracy=sum(v['balanced_accuracy'] for v in by_kind.values()) / len(by_kind),
                by_kind=by_kind, decision_by_kind=decision_kinds, forced_sites=forced,
                decision_macro_kind_balanced_accuracy=(
                    sum(v['balanced_accuracy'] for v in decision_kinds.values()) / len(decision_kinds)
                    if decision_kinds else None))


def decision_report(rows, registry):
    """Generated decision metrics, strata and correctly paired target responses.

    Field forcedness uses the GOLD prefix; legality remains checked separately by
    the constrained decoder. Responsiveness alone never receives correctness credit.
    """
    from experiments.stage1_recovery.prepare import target_contrasts
    import hashlib
    import json
    result = metrics(rows, registry)
    result['case_identity_sha256'] = hashlib.sha256(json.dumps([
        (r['kernel_name'],r.get('device'),r.get('selected_clock_period'),r.get('resource_budget_id'),
         r.get('jsonl_idx',r.get('_jsonl_idx')),sorted(parse(r['reference_target']).items())) for r in rows
    ],sort_keys=True).encode()).hexdigest()
    action_counts = collections.Counter()
    forced_right = forced_total = decision_right = decision_total = 0
    positive = collections.defaultdict(lambda: [0, 0])
    binary_positive = collections.defaultdict(lambda: [0, 0])
    illegal = 0
    strata = collections.defaultdict(list)
    for row in rows:
        gold, pred = parse(row['reference_target']), parse(row['prediction'])
        groups, prior, predicted_prior = collections.defaultdict(list), {}, {}
        for key, value in gold.items():
            kind = production.lhs_kind(key)
            legal = production.constrained_rhs_candidates(row['kernel_name'], key, registry, prior)[1]
            predicted_legal = production.constrained_rhs_candidates(row['kernel_name'], key, registry, predicted_prior)[1]
            illegal += pred[key] not in predicted_legal
            predicted_prior[key] = pred[key]
            if len(legal) == 1:
                forced_total += 1
                forced_right += value == pred[key]
            else:
                decision_total += 1
                decision_right += value == pred[key]
                if kind in ('PIPE', 'UNROLL') and int(value) > 0:
                    positive[kind][1] += 1
                    positive[kind][0] += value == pred[key]
                    binary_positive[kind][1] += 1
                    binary_positive[kind][0] += str(pred[key]).isdigit() and int(pred[key]) > 0
            prior[key] = value
            groups[re.search(r'_(L\d+)\}', key).group(1)].append(key)
        for keys in groups.values():
            category = 'loop' if any(production.lhs_kind(k) == 'PIPE' for k in keys) else 'array'
            action_counts[category + '_total'] += 1
            action_counts[category + '_correct'] += all(gold[k] == pred[k] for k in keys)
        fractions = [row.get(k) for k in ('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut')]
        tightness = ('unknown' if any(v is None for v in fractions) else
                     'tight_min_fraction_below_0.25' if min(fractions)<.25 else
                     'full' if min(fractions)==1 else 'intermediate')
        for dimension, value in [('kernel',row['kernel_name']),('family',row.get('_family','unknown')),
                                 ('device',row.get('device','unknown')),('clock',row.get('selected_clock_period','unknown')),
                                 ('device_clock',(row.get('device','unknown'),row.get('selected_clock_period','unknown'))),
                                 ('budget',tightness)]:
            strata[f'{dimension}:{value}'].append(row)
    result.update(complete_action_accuracy=result['joint_action_accuracy'],
                  complete_loop_action_accuracy=(action_counts['loop_correct']/action_counts['loop_total'] if action_counts['loop_total'] else None),
                  complete_array_action_accuracy=(action_counts['array_correct']/action_counts['array_total'] if action_counts['array_total'] else None),
                  forced_accuracy=forced_right/forced_total if forced_total else None,
                  nonforced_decision_accuracy=decision_right/decision_total if decision_total else None,
                  positive_reference_value_recall={k: a/b for k,(a,b) in positive.items()},
                  positive_binary_recall={k: a/b for k,(a,b) in binary_positive.items()},
                  illegal_predicted_fields=illegal,
                  core_decision_balanced_accuracy=core_balanced(result))
    result['by_context'] = {k: metrics(v,registry) for k,v in sorted(strata.items())}
    requests = [dict(r, target=r['reference_target']) for r in rows]
    contrasts = target_contrasts(requests)
    transitions = collections.defaultdict(collections.Counter)
    for pair in contrasts:
        kind, left, right = pair['kind'], rows[pair['left']], rows[pair['right']]
        if not pair['comparable']:
            transitions[kind]['unsupported'] += 1
            continue
        ga,gb,pa,pb = [parse(t) for t in [left['reference_target'],right['reference_target'],left['prediction'],right['prediction']]]
        changed = pair['changed_fields']
        group = 'changed' if changed else 'invariant'
        transitions[kind][group+'_pairs'] += 1
        transitions[kind][group+'_complete_correct'] += ga==pa and gb==pb
        if changed:
            transitions[kind]['changed_only_groups_correct'] += all(pa[k]==ga[k] and pb[k]==gb[k] for k in changed)
            transitions[kind]['changed_fields'] += len(changed)
            transitions[kind]['changed_fields_both_endpoints_correct'] += sum(pa[k]==ga[k] and pb[k]==gb[k] for k in changed)
        else:
            transitions[kind]['invariant_prediction_pairs'] += pa==pb
            # Equal but wrong predictions are explicitly not correct invariance.
    result['target_transitions'] = {k: dict(v) for k,v in transitions.items()}
    return result


def promotion_assessment(candidate, historical, constant):
    """Report-only gate; never moves or overwrites a checkpoint."""
    if not (candidate['case_identity_sha256']==historical['case_identity_sha256']==constant['case_identity_sha256']):
        raise ValueError('Promotion requires identical validation cases')
    checks = dict(complete_action_better_than_reference=candidate['complete_action_accuracy']>historical['complete_action_accuracy'],
                  complete_action_better_than_constant=candidate['complete_action_accuracy']>constant['complete_action_accuracy'],
                  positive_unroll_recall=candidate['positive_reference_value_recall'].get('UNROLL',0)>0,
                  positive_pipeline_recall=candidate['positive_reference_value_recall'].get('PIPE',0)>0,
                  correct_budget_transition=candidate.get('target_transitions',{}).get('budget',{}).get('changed_complete_correct',0)>0,
                  legal_fields=candidate['illegal_predicted_fields']==0,
                  no_device_clock_default_collapse=all(
                      len(kind['prediction_counts'])>1
                      for key,context in candidate['by_context'].items() if key.startswith('device_clock:')
                      for name,kind in context['decision_by_kind'].items()
                      if name in ('PIPE','UNROLL','ARRAY_T','ARRAY_F') and len(kind['per_value_recall'])>1))
    return dict(eligible_on_measured_decisions=all(checks.values()), checks=checks,
                remaining_checks=['No automatic promotion; inspect per-kernel and joint device/clock behavior before Stage 2'],
                automatic_checkpoint_replacement=False)
