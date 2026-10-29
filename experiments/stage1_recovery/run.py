"""Isolated Stage-1 memorization/decision-learning experiments; never overwrites a run.

Uses the production RHS pack, collator and Trainer, with an explicit experimental
loss policy. Training uses only authenticated train indices. Kalman is excluded.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import random
import re
import subprocess
import time
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import TrainerCallback, TrainingArguments, set_seed

from LLM_branch.common import frozen_stage1, mailohls_contract as contract_api
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from LLM_branch.train import train_SFT_xattn_new as production
from experiments.stage1_recovery import decisions
from experiments.stage1_recovery.prepare import DEFAULT_OUTPUT, prepare

DEFAULT_PARENT = 'mailohls_runs/stage1_final_final_latency_s123/best_custom_stage1'
AUDIT = DEFAULT_OUTPUT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def assignments(text):
    return {lhs.upper(): rhs for lhs, rhs in re.findall(
        r'(auto\{_[A-Z_]+_L\d+\})\s*=\s*(\w+)', text)}


def choose_rows(rows, sources, tok, count, max_length, per_kernel=2):
    """Cover diverse families and values using training labels only."""
    eligible = []
    for row in rows:
        source = sources[row['source_key']]
        if len(tok(source, add_special_tokens=False)['input_ids']) > max_length - 650:
            continue
        target = assignments(row['target'])
        if len(target) > 24:
            continue
        features = {('family', row['_family'])}
        features.update((re.search(r'AUTO\{_(.+)_L\d+\}', lhs).group(1), rhs)
                        for lhs, rhs in target.items())
        eligible.append((row, features))
    chosen, covered, configs = [], set(), set()
    counts = collections.Counter()
    while len(chosen) < count:
        pool = [(r, f) for r, f in eligible if counts[r['kernel_name']] < per_kernel
                and (r['kernel_name'], r['target']) not in configs]
        if not pool:
            raise ValueError(f'Only {len(chosen)} diverse examples fit the requested control')
        row, features = max(pool, key=lambda x: (len(x[1] - covered), -x[0]['_jsonl_idx']))
        row = dict(row, input=sources[row['source_key']])
        chosen.append(row)
        counts[row['kernel_name']] += 1
        configs.add((row['kernel_name'], row['target']))
        covered.update(features)
    return chosen


def remove_delimiter_supervision(data, rows, tok, registry, balance=False):
    """Retain all context tokens; supervise values only with equal site mass."""
    counts = collections.Counter()
    for row in rows:
        chosen = {}
        for lhs, rhs in assignments(row['target']).items():
            if len(production.constrained_rhs_candidates(row['kernel_name'], lhs, registry, chosen)[1]) > 1:
                counts[(production.lhs_kind(lhs), rhs)] += 1
            chosen[lhs] = rhs
    kind_totals = collections.Counter()
    kind_values = collections.Counter()
    for (kind, rhs), n in counts.items():
        kind_totals[kind] += n
        kind_values[kind] += 1
    for sample, row in zip(data.samples, rows):
        pack = production.build_deterministic_rhs_pack(row['input'], row['target'], tok,
                    directive_domain_registry=registry, kernel_name=row['kernel_name'])
        start = len(sample['input_ids']) - len(pack.input_ids)
        # This experiment never supervises EOS and all clocks are specified.
        assert sample['input_ids'][start:].tolist() == pack.input_ids
        groups = collections.defaultdict(list)
        for i, key in enumerate(pack.site_keys):
            if key is not None and pack.labels[i] != -100:
                groups[key].append(start + i)
        gold = assignments(row['target'])
        for key, positions in groups.items():
            ids = tok(gold[key], add_special_tokens=False)['input_ids']
            assert sample['input_ids'][positions[:len(ids)]].tolist() == ids
            assert sample['input_ids'][positions[len(ids):]].tolist() == tok('\n', add_special_tokens=False)['input_ids']
            factor = 1.
            if balance:
                kind = production.lhs_kind(key)
                factor = min(2., max(.5, (kind_totals[kind] / (kind_values[kind] * counts[(kind, gold[key])])) ** .5))
            sample['labels'][positions] = -100
            sample['token_weights'][positions] = 0.
            sample['labels'][positions[:len(ids)]] = sample['input_ids'][positions[:len(ids)]]
            sample['token_weights'][positions[:len(ids)]] = factor / len(ids)


@torch.no_grad()
def teacher_metrics(model, data, rows, tok, registry):
    """Full-vocabulary teacher-forced value accuracy, excluding delimiters."""
    model.eval()
    correct = total = exact_examples = 0
    by_kind = collections.defaultdict(lambda: [0, 0])
    for sample, row in zip(data.samples, rows):
        ids = sample['input_ids'].to(model.device).unsqueeze(0)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        pred = logits[0, :-1].argmax(-1).cpu()
        pack = production.build_deterministic_rhs_pack(row['input'], row['target'], tok,
                    directive_domain_registry=registry, kernel_name=row['kernel_name'])
        offset = len(sample['input_ids']) - len(pack.input_ids)
        sites = collections.defaultdict(list)
        for i, key in enumerate(pack.site_keys):
            if key is not None and pack.labels[i] != -100:
                sites[key].append(offset + i)
        all_right = True
        for key, positions in sites.items():
            n = len(tok(assignments(row['target'])[key], add_special_tokens=False)['input_ids'])
            positions = positions[:n]
            right = torch.equal(pred[[p - 1 for p in positions]], sample['input_ids'][positions])
            correct += right
            total += 1
            by_kind[production.lhs_kind(key)][0] += right
            by_kind[production.lhs_kind(key)][1] += 1
            all_right &= right
        exact_examples += all_right
    return dict(teacher_forced_value_site_accuracy=correct / total, correct=correct, sites=total,
                teacher_forced_exact_examples=exact_examples, examples=len(rows), by_kind=dict(by_kind))


# Shared audit implementation; the historical optimization path is unchanged.
from LLM_branch.train.stage1_decision_loss import UpdateAudit


class JointChoiceTrainer(production.LengthGroupedTrainer):
    """Sampled complete-action softmax plus value-only CE, experimental only."""
    def __init__(self, *args, tokenizer, **kwargs):
        self.decision_tokenizer = tokenizer
        self.choice_log = []
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        sites = inputs.pop('joint_actions')
        ce = super().compute_loss(model, dict(inputs), return_outputs=False,
                                  num_items_in_batch=num_items_in_batch)
        if not model.training:
            return ce
        # Batch size is explicitly one in this bounded experiment.
        actions = sites[0]
        rng = random.Random(123 + self.state.global_step)
        action = actions[rng.randrange(len(actions))]
        negatives = [c for c in action['choices'] if c != action['gold']]
        # Sample legal rivals from source domains, never measured held-out labels.
        negatives = rng.sample(negatives, min(3, len(negatives)))
        scores = []
        for candidate in [action['gold']] + negatives:
            ids, positions = decisions.encode_action(self.decision_tokenizer, action['fields'], candidate)
            scores.append(decisions.action_score(model, action['prefix_ids'], ids, positions))
        choice_loss = -torch.log_softmax(torch.stack(scores), dim=0)[0]
        total = ce + choice_loss
        if self.state.global_step % 10 == 0:
            self.choice_log.append(dict(step=self.state.global_step, label=action['label'],
                                       sampled_candidates=len(scores), ce=float(ce.detach()),
                                       choice_loss=float(choice_loss.detach())))
        return total


def evaluate_greedy(model, tok, rows, registry):
    model.eval()
    results = []
    for row in rows:
        case = inference.InferenceCase(row['kernel_name'], row['input'], row['obj_mode'], 1., 0., row, row['target'])
        result = inference.predict_case(model=model, tok=tok, case=case, directive_domain_registry=registry,
            stage='stage1', max_prompt_tokens=7168, mem_bank=None, mem_dim=0, max_slots=64,
            structural_routing='exact_slot', score_reduction='mean', debug_topk=0,
            candidate_max_prefix_tokens=0, candidate_keep_head_tokens=256, print_xattn_debug_flag=False,
            num_samples=1, no_greedy_first=False, sample_temperature=.8, sample_top_p=1.,
            sample_top_k=0, sample_seed=123)
        result['reference_target'] = row['target']
        result['jsonl_idx'] = row['_jsonl_idx']
        results.append(result)
        print('GREEDY', row['kernel_name'], flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', default=DEFAULT_PARENT)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['ce', 'decision-ce', 'balanced-decision-ce', 'joint-choice'], default='ce')
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--examples', type=int, default=20)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--lr-lora', type=float, default=1.5e-5)
    ap.add_argument('--lr-embed', type=float, default=5e-6)
    ap.add_argument('--eval-greedy', type=int, default=4)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if not 0 < args.steps <= 1000 or not 0 < args.examples <= 50:
        raise ValueError('This control is limited to 1..1000 steps and 1..50 train examples')
    out.mkdir(parents=True)
    start = time.perf_counter()
    parent = Path(args.parent)
    contract = json.loads((parent / 'training_contract.json').read_text())
    selection_manifest = prepare(parent)
    before = {name: sha(parent / name) for name in ['adapter_model.safetensors', 'training_contract.json', 'tokenizer.json']}
    set_seed(123)
    tok = inference.load_checkpoint_tokenizer(str(parent), contract)
    production.TARGET_CFG = production.TargetAwareConfig(device_mode='known', seed=123)
    production.extract_ordered_lhs_plan = functools.lru_cache(maxsize=256)(production.extract_ordered_lhs_plan)
    production.reorder_target_by_source_order = functools.lru_cache(maxsize=100000)(production.reorder_target_by_source_order)
    registry = production.load_directive_domain_registry(str(parent.parent / 'directive_domain_registry.json'))
    sources = json.loads(Path('artifacts/llm/mailohls_sft.sources.json').read_text())['templates']
    split = json.loads(Path('mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    assert sha('artifacts/llm/mailohls_sft.jsonl') == contract['dataset_sha256'] == split['dataset_sha256']
    rows = [json.loads(line) for line in (AUDIT / 'train_reconstructed_selected.jsonl').read_text().splitlines()]
    train_indices = set(split['train_jsonl_idx'])
    assert all(r['_jsonl_idx'] in train_indices for r in rows)
    selected = choose_rows(rows, sources, tok, args.examples, args.max_length)
    dump(out / 'selected_train.json', [{k: v for k, v in r.items() if k != 'input'} for r in selected])
    data = production.SFTDataset(selected, tok, args.max_length, kind_loss_weights=contract['directive_loss_weights'],
                                  directive_domain_registry=registry, input_only_special_ids=contract['special_token_ids'])
    assert max(data.lengths) < args.max_length
    if args.mode != 'ce':
        remove_delimiter_supervision(data, selected, tok, registry, args.mode == 'balanced-decision-ce')
    if args.mode == 'joint-choice':
        for sample, row in zip(data.samples, selected):
            sample['joint_actions'] = decisions.training_actions(sample, row, tok, registry)
    base = inference.load_base_model(contract['model'], revision=contract['model_revision'])
    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, parent, is_trainable=True)
    production.assert_stage1_trainable_contract(model)
    initial_hash = frozen_stage1.frozen_stage1_hashes(model)
    report = dict(schema='mailohls-stage1-recovery-experiment-v1', settings=vars(args),
                  parent_hashes=before, initial_trainable_hashes=initial_hash,
                  source_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  trainable_contract=production.assert_stage1_trainable_contract(model),
                  selected_examples=len(selected), max_sequence_length=max(data.lengths),
                  gpu=torch.cuda.get_device_name(), heldout_test_used=False)
    report['selection_manifest'] = selection_manifest
    report['implementation_hashes'] = {str(p): sha(p) for p in [Path(__file__), Path(decisions.__file__), Path(production.__file__)]}
    report['initial_train_teacher'] = teacher_metrics(model, data, selected, tok, registry)
    dump(out / 'report.json', report)
    print('INITIAL', report['initial_train_teacher'], flush=True)
    callback = UpdateAudit(out)
    training_args = TrainingArguments(output_dir=str(out / 'trainer'), max_steps=args.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, fp16=True,
        gradient_checkpointing=True, logging_steps=10, save_strategy='no', eval_strategy='no',
        report_to=[], remove_unused_columns=False, lr_scheduler_type='constant', warmup_ratio=0.,
        max_grad_norm=1., seed=123, dataloader_num_workers=0, disable_tqdm=True)
    def collate(samples):
        result = production.PadCollator(tok)(samples)
        if args.mode == 'joint-choice':
            result['joint_actions'] = [sample['joint_actions'] for sample in samples]
        return result

    trainer_class = JointChoiceTrainer if args.mode == 'joint-choice' else production.LengthGroupedTrainer
    extra = dict(tokenizer=tok) if args.mode == 'joint-choice' else {}
    trainer = trainer_class(model=model, args=training_args, train_dataset=data,
        data_collator=collate, lr_lora=args.lr_lora, lr_embed=args.lr_embed,
        lora_weight_decay=.01, family_sampling_power=0., loss_chunk_t=128, xattn_diagnostic_steps=0,
        callbacks=[callback], **extra)
    callback.trainer = trainer
    trainer.train()
    if args.mode == 'joint-choice':
        dump(out / 'choice_loss_audit.json', trainer.choice_log)
    report['final_train_teacher'] = teacher_metrics(model, data, selected, tok, registry)
    report['final_trainable_hashes'] = frozen_stage1.frozen_stage1_hashes(model)
    report['actual_updates'] = sum(not r['optimizer_step_skipped'] for r in callback.records if 'optimizer_step_skipped' in r)
    report['trainables_changed'] = initial_hash != report['final_trainable_hashes']
    # Separate schema prevents accidental promotion through the production CLI.
    keys = ['model', 'model_revision', 'tokenizer', 'tokenizer_size', 'special_tokens',
            'special_token_ids', 'embedding_weights_tied', 'objective', 'prompt_schema_version', 'device_mode']
    new_contract = dict({k: contract[k] for k in keys},
        schema='mailohls-stage1-recovery-adapter-v1', stage='stage1_experimental',
        special_token_to_id={s: tok.convert_tokens_to_ids(s) for s in contract['special_tokens']},
        experimental_recovery=dict(schema=report['schema'], mode=args.mode, parent=str(parent),
            parent_adapter_sha256=before['adapter_model.safetensors'], steps=args.steps,
            lr_lora=args.lr_lora, lr_embed=args.lr_embed, training_examples_sha256=sha(out / 'selected_train.json')))
    production.save_mailohls_adapter(model, tok, str(out / 'adapter'), new_contract)
    # A training-set generated-output control is not a generalization score.
    if args.eval_greedy:
        dump(out / 'train_greedy.json', evaluate_greedy(model, tok, selected[:args.eval_greedy], registry))
    report['parent_unchanged'] = before == {name: sha(parent / name) for name in before}
    assert report['parent_unchanged'] and report['trainables_changed']
    report['elapsed_seconds'] = time.perf_counter() - start
    dump(out / 'report.json', report)
    print('COMPLETE', report, flush=True)


if __name__ == '__main__':
    main()
