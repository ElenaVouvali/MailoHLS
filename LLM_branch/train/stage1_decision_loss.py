"""Versioned Stage-1 supervision arithmetic; no model or dataset replacement.

All inputs to frequency fitting must be the selected TRAIN partition. The caller
records its provenance. Directive domains and serialization remain authoritative.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import hashlib
import re

POLICIES = ('legacy', 'decision', 'decision_balanced')
VERSION = 'stage1-action-normalized-values-v1'


def _value_positions(tok, rhs, ids):
    """Locate values in the *existing* RHS+newline encoding, or fail closed."""
    try:
        encoded = tok(rhs + '\n', add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError):
        encoded = {}
    offsets = encoded.get('offset_mapping')
    if offsets is not None:
        if encoded['input_ids'] != ids:
            raise ValueError('Value-token alignment changed target encoding')
        if any(a < len(rhs) < b for a, b in offsets):
            raise ValueError('A tokenizer token merges directive value and delimiter')
        positions = [i for i, (a, b) in enumerate(offsets) if b > a and b <= len(rhs)]
        covered = {c for i in positions for c in range(*offsets[i])}
        if covered != set(range(len(rhs))):
            raise ValueError('Incomplete directive-value token alignment')
    else:
        value = tok(rhs, add_special_tokens=False)['input_ids']
        newline = tok('\n', add_special_tokens=False)['input_ids']
        if ids != value + newline:
            raise ValueError('Cannot exactly align directive value without token offsets')
        positions = list(range(len(value)))
    if not positions:
        raise ValueError('Empty directive-value supervision')
    return positions


def apply_decision_policy(pack, tok, rhs_map, policy='legacy', value_weights=None):
    """Mutate loss/routing masks only. Input IDs and context never change.

    Unbalanced: each action receives 1/A mass; each of its J decision fields
    receives 1/(A*J); each field's V tokens receive 1/(A*J*V).
    Balanced: multiply these masses by train-fitted within-kind value weights,
    then renormalize per example. Forced fields and EOS receive no decision loss.
    """
    if policy not in POLICIES:
        raise ValueError(f'Unknown Stage-1 supervision policy: {policy}')
    if policy == 'legacy':
        return pack
    if policy == 'decision_balanced' and value_weights is None:
        raise ValueError('Balanced decisions require training-only fitted weights')
    all_positions, fields, actions = defaultdict(list), {}, defaultdict(list)
    for i, key in enumerate(pack.site_keys):
        if key is not None:
            all_positions[key].append(i)
    gold = {k.upper(): v.strip() for k, v in rhs_map.items()}
    for key, positions in all_positions.items():
        if not any(pack.labels[i] != -100 for i in positions):
            continue  # semantically forced under the gold legal prefix
        selected = _value_positions(tok, gold[key], [pack.input_ids[i] for i in positions])
        fields[key] = [positions[i] for i in selected]
        action = re.search(r'_L(\d+)\}', key).group(1)
        actions[action].append(key)
    if not actions:
        raise ValueError('No nonforced directive decisions in example')
    pack.labels[:] = [-100] * len(pack.labels)
    pack.token_weights[:] = [0.] * len(pack.token_weights)
    pack.xattn_target_mask[:] = [0] * len(pack.xattn_target_mask)
    for keys in actions.values():
        for key in keys:
            kind = re.match(r'AUTO\{_(.+)_L\d+\}', key).group(1)
            factor = 1.
            if policy == 'decision_balanced':
                # Unseen values are neutral, not assigned an invented frequency.
                factor = value_weights.get(kind, {}).get(gold[key], 1.)
            weight = factor / (len(actions) * len(keys) * len(fields[key]))
            for i in fields[key]:
                pack.labels[i] = pack.input_ids[i]
                pack.token_weights[i] = weight
                pack.xattn_target_mask[i] = 1
    total = sum(pack.token_weights)
    for i in range(len(pack.token_weights)):
        pack.token_weights[i] /= total
    return pack


def decision_fields(row, registry, candidates, parse):
    """Use existing sequential legal domains, including array tuple constraints."""
    prior = {}
    for key, rhs in parse(row['target']).items():
        legal = candidates(row['kernel_name'], key, registry, prior)[1]
        if rhs not in legal:
            raise ValueError(f'Illegal gold value: {row["kernel_name"]}/{key}={rhs}')
        kind = re.match(r'AUTO\{_(.+)_L\d+\}', key).group(1)
        yield key, kind, rhs, len(legal) == 1
        prior[key] = rhs


def fit_value_weights(rows, registry, candidates, parse, minimum=.5, maximum=2.):
    """Bounded inverse sqrt frequency, counting a measured row once, not budgets.

    Reference frequency is the arithmetic mean of observed value counts in each
    kind. A monotone bounded rescaling makes the count-weighted mean exactly one.
    This is an explicit directive adaptation, not Cui et al.'s effective-number
    formula. No counts from validation/test or absent values are added.
    """
    if not 0 < minimum <= 1 <= maximum or not math.isfinite(maximum):
        raise ValueError('Balancing bounds must be finite and straddle one')
    counts, seen = defaultdict(Counter), set()
    for row in rows:
        identity = (row['kernel_name'], row.get('_jsonl_idx', row['target']))
        if identity in seen:
            continue
        seen.add(identity)
        for _, kind, rhs, forced in decision_fields(row, registry, candidates, parse):
            if not forced:
                counts[kind][rhs] += 1
    weights = {}
    for kind, values in counts.items():
        reference = sum(values.values()) / len(values)
        raw = {v: math.sqrt(reference / n) for v, n in values.items()}
        low, high = 0., max(1., maximum / min(raw.values())) * 2
        for _ in range(80):
            scale = (low + high) / 2
            mean = sum(n * min(maximum, max(minimum, scale * raw[v]))
                       for v, n in values.items()) / sum(values.values())
            if mean < 1:
                low = scale
            else:
                high = scale
        weights[kind] = {v: min(maximum, max(minimum, (low + high) / 2 * w))
                         for v, w in raw.items()}
    return dict(version=VERSION, counting_unit='distinct selected measured row; budget repeats excluded',
                distinct_measured_rows=len(seen), counts={k: dict(v) for k, v in counts.items()},
                minimum=minimum, maximum=maximum, weights=weights)


def emphasize_budget_contrasts(rows, factor=1.25):
    """Optional train-only emphasis; invariant requests retain positive weight.

    Emphasis requires measured cross-budget infeasibility, not just a changed winner.
    Near-tied feasible winners are not treated as statistically certain preferences.
    This explicitly counts contexts (not new physical measurements).
    """
    from experiments.stage1_recovery.prepare import target_contrasts
    if not 1 <= factor <= 2:
        raise ValueError('Contrast factor must be bounded in [1, 2]')
    from LLM_branch.train import train_SFT_xattn_new as production
    participants = set()
    boundaries = cross_feasible = missing = 0
    utilization = ('bram_util_%','dsp_util_%','ff_util_%','lut_util_%')
    fractions = ('budget_frac_bram_18k','budget_frac_dsp','budget_frac_ff','budget_frac_lut')
    for pair in target_contrasts(rows):
        if pair['kind']!='budget' or pair['classification']!='changed':
            continue
        left,right = rows[pair['left']],rows[pair['right']]
        if any(row.get(key) is None for row in (left,right) for key in utilization+fractions):
            missing += 1
            continue
        left_budget = production.SharedResourceBudget(*(float(left[k]) for k in fractions))
        right_budget = production.SharedResourceBudget(*(float(right[k]) for k in fractions))
        if not (production.design_fits_shared_budget(left,left_budget)
                and production.design_fits_shared_budget(right,right_budget)):
            raise ValueError('Selected contrast winner is infeasible in its own request')
        if (production.design_fits_shared_budget(left,right_budget)
                and production.design_fits_shared_budget(right,left_budget)):
            cross_feasible += 1
            continue  # changed tie-breaking is not evidence that a budget forces change
        boundaries += 1
        participants.update([pair['left'], pair['right']])
    values = [float(r.get('_sample_weight',1.)) * (factor if i in participants else 1.)
              for i,r in enumerate(rows)]
    mean = sum(values)/len(values)
    for row,value in zip(rows,values):
        row['_sample_weight'] = value/mean
    return dict(factor=factor, eligibility='measured_cross_budget_infeasibility_v1',
                feasibility_boundary_pairs=boundaries,both_winners_cross_feasible_pairs=cross_feasible,
                incomplete_feasibility_pairs=missing,changed_budget_participants=len(participants),
                invariant_or_other_contexts=len(rows)-len(participants), mean_weight=1.,
                distinct_measured_rows=len({r.get('_jsonl_idx') for r in rows}),
                weighting_unit='selected context; no independent-measurement claim',
                reduction='mean of training-mean-normalized weights; not batch-normalized')


# Training auditing is shared with the existing small-data runner. It does not
# change optimization or the scheduler and observes Accelerate's skipped updates.
from pathlib import Path
import json
import torch
from transformers import TrainerCallback


class UpdateAudit(TrainerCallback):
    def __init__(self, output):
        self.output = Path(output)
        self.records = []
        self.trainer = None
        self.completed = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.output.mkdir(parents=True,exist_ok=True)
        previous = self.output/'optimizer_audit.json'
        if state.global_step and previous.is_file():
            original = previous.read_bytes()
            history = json.loads(original)
            self.records = [r for r in history if r['step'] <= state.global_step]
            if len(self.records) != len(history):
                backup = self.output/f'optimizer_audit_before_resume_{state.global_step}_{hashlib.sha256(original).hexdigest()[:12]}.json'
                if not backup.exists():
                    backup.write_bytes(original)
            self.completed = sum(not r['optimizer_step_skipped'] for r in self.records
                                 if 'optimizer_step_skipped' in r and r['step'] <= state.global_step)
        dataloader = kwargs.get('train_dataloader')
        schedule = dict(transformers_version=__import__('transformers').__version__,
            trainer_max_steps=state.max_steps, epochs=args.num_train_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps, resume_step=state.global_step,
            dataloader_batches=len(dataloader) if dataloader is not None else None,
            note='Trainer 4.51 uses floor(batches/accumulation) for epoch-based max_steps; coverage guard uses ceil. Actual successful updates exclude AMP skips.')
        (self.output/'schedule_audit.json').write_text(json.dumps(schedule,indent=2)+'\n')

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step % 10 == 0:
            norms = defaultdict(float)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f'Nonfinite gradient: {name}')
                    kind = 'embedding' if 'trainable_tokens_delta' in name else 'lora'
                    norms[kind] += float(parameter.grad.float().square().sum())
            self.records.append(dict(step=state.global_step+1,gradient_norms={k:v**.5 for k,v in norms.items()}))

    def on_step_end(self, args, state, control, **kwargs):
        skipped = bool(self.trainer.accelerator.optimizer_step_was_skipped)
        self.completed += int(not skipped)
        scheduler = self.trainer.lr_scheduler
        self.records.append(dict(step=state.global_step, optimizer_step_skipped=skipped,
             actual_updates=self.completed, scheduler_last_epoch=getattr(scheduler,'last_epoch',None),
             learning_rates=[g['lr'] for g in self.trainer.optimizer.param_groups]))
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'optimizer_audit.json').write_text(json.dumps(self.records,indent=2)+'\n')
