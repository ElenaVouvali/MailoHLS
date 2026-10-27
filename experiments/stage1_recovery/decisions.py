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
        r'(auto\{_[A-Z_]+_L\d+\})\s*=\s*(\w+)', text)}


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
