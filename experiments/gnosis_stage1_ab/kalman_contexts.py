"""Evaluate the common V4 parent and selected GNOSIS A/B checkpoints on Kalman.

No training, HLS synthesis, test measurements, or structural-memory replacement.
FiLM masks reuse the exact training serialization and nonforced-value policy.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.environ.get('MAILOHLS_REPO_ROOT', str(Path(__file__).resolve().parents[2])))
import torch
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from LLM_branch.common import directive_domains as domains, device_adaptation as da
from LLM_branch.common import mailohls_contract as mc
from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
from LLM_branch.train import train_SFT_xattn_new as prod
from LLM_branch.train.stage1_decision_loss import apply_decision_policy
from experiments.gnosis_stage1_ab import train_gnosis_ab as trained

KERNEL = 'serrano-kalman-filter'
PARENT_SHA = '28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164'


def sha(path):
    return trained.sha(path)


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def contexts(source):
    rows = []
    specifications = [
        ('zcu104_10ns_100pct', 'xczu7ev-ffvc1156-2-e', 10., 1.),
        ('u200_10ns_100pct', 'xcu200-fsgd2104-2-e', 10., 1.),
        ('zcu104_3p33ns_100pct', 'xczu7ev-ffvc1156-2-e', 3.33, 1.),
        ('zcu104_10ns_20pct', 'xczu7ev-ffvc1156-2-e', 10., .2),
    ]
    for name, device, clock, fraction in specifications:
        row = prod.attach_shared_budget(dict(
            kernel_name=KERNEL, input=source, device=device, clock_period=clock,
            selected_clock_period=clock, obj_mode='PARETO_LATENCY',
            objective='PARETO_LATENCY', frequency_mode='specified', context_id=name,
        ), prod.SharedResourceBudget(fraction, fraction, fraction, fraction))
        rows.append(row)
    return rows


def prompt_tokens(tok, row):
    sections = mc.build_prompt_sections(row['input'], row['obj_mode'],
                                      mc.target_prompt_fields(row, device_mode='known'))
    return ([i for s in sections for i in tok(s, add_special_tokens=False)['input_ids']]
            + mc.selected_clock_response_token_ids(row, tok))


def causal_kind_mask(tok, source, registry, full_ids, attention_mask, prompt_length):
    """Reconstruct partial candidate serialization exactly, including prior values.

    FiLM applies at logits predicting nonforced *value* tokens, excluding newline.
    A mismatch fails closed; prompt tokens and padding always have kind zero.
    """
    result = torch.zeros_like(full_ids, dtype=torch.long)
    for b in range(full_ids.shape[0]):
        length = int(attention_mask[b].sum())
        target_ids = full_ids[b, prompt_length:length].tolist()
        target = tok.decode(target_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
        pack = prod.build_deterministic_rhs_pack(
            source, target, tok, directive_domain_registry=registry, kernel_name=KERNEL)
        if pack.input_ids != target_ids:
            raise ValueError('Candidate FiLM mask cannot reproduce target tokenization')
        apply_decision_policy(pack, tok, prod.build_rhs_map_from_target(target), 'decision')
        for i, key in enumerate(pack.site_keys):
            if key is not None and pack.token_weights[i] > 0:
                position = prompt_length + i - 1
                if position < prompt_length:
                    raise ValueError('FiLM prediction position overlaps prompt')
                result[b, position] = trained.kind_id(key)
    return result


class CandidateModel:
    """Bind the trained FiLM hook around each existing production scoring call."""
    def __init__(self, model, tok, registry, hooks, film, max_length):
        self.model, self.tok, self.registry = model, tok, registry
        self.hooks, self.film, self.max_length = hooks, film, max_length
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def bind_case(self, row, prompt, name, enabled):
        self.row, self.prompt, self.name, self.enabled = row, prompt, name, enabled
        self.started = time.monotonic()
        self.calls = 0

    def __call__(self, **kwargs):
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        if ids.shape[1] > self.max_length:
            raise ValueError('Complete source + directive candidate exceeds context; no truncation')
        kinds = causal_kind_mask(self.tok, self.row['input'], self.registry,
                                 ids, mask, len(self.prompt))
        context = trained.context_features(self.row).to(ids.device)[None, None, :]
        self.film.last.clear()
        with self.hooks.bind(context, kinds, enabled=self.enabled):
            with torch.autocast('cuda', dtype=torch.float16):
                output = self.model(**kwargs, use_cache=False)
        self.calls += 1
        if self.calls == 1 or self.calls % 5 == 0:
            print(json.dumps(dict(event='candidate_batches', model=self.name,
                context=self.row['context_id'], batches=self.calls,
                sequence_tokens=ids.shape[1], batch=ids.shape[0],
                seconds=round(time.monotonic() - self.started, 2))), flush=True)
        return output


def verify_loaded_residual(model, checkpoint):
    path = checkpoint / 'device_adapt/adapter_model.safetensors'
    state = load_file(str(path))
    set_peft_model_state_dict(model, state, adapter_name='device_adapt')
    actual = get_peft_model_state_dict(model, adapter_name='device_adapt')
    if set(actual) != set(state):
        raise ValueError('Residual checkpoint keys differ')
    for key, value in state.items():
        torch.testing.assert_close(actual[key].detach().cpu(), value, rtol=0, atol=0,
                                   check_dtype=True)


def preparation(args):
    source_path = args.source / 'code_to_memory_outputs/meta/serrano-kalman-filter_placeholders.cpp'
    source = domains.prepare_source_template(KERNEL, source_path, args.source)
    actions = domains.load_source_actions(KERNEL, args.source)
    rows = contexts(source)
    contract = json.loads((args.parent / 'training_contract.json').read_text())
    prod.TARGET_CFG.device_mode = contract['device_mode']
    if sha(args.parent / 'adapter_model.safetensors') != PARENT_SHA:
        raise ValueError('Common V4 parent SHA mismatch')
    if contract['stage'] != 'stage1' or contract['objective'] != 'PARETO_LATENCY':
        raise ValueError('Wrong Stage-1 parent objective')
    registry_path = args.parent / 'directive_domain_registry.json'
    if sha(registry_path) != contract['directive_domain_registry_sha256']:
        raise ValueError('Parent registry SHA mismatch')
    registry = inf.load_directive_domain_registry(str(registry_path))
    inf.add_missing_source_directive_domains(registry, [type('Case', (), dict(
        kernel_name=KERNEL, source_text=source))()], str(args.source))
    tok = inf.load_checkpoint_tokenizer(str(args.parent), contract)
    checkpoints = {}
    for arm in ['A', 'B']:
        best = json.loads((args.training_run / arm / 'best.json').read_text())
        cp = Path(best['checkpoint'])
        receipt = json.loads((cp / 'receipt.json').read_text())
        if receipt['parent_sha256'] != PARENT_SHA or receipt['arm'] != arm:
            raise ValueError('Selected checkpoint lineage mismatch')
        if sha(cp / 'device_adapt/adapter_model.safetensors') != receipt['lora_sha256']:
            raise ValueError('Selected residual SHA mismatch')
        if arm == 'B' and sha(cp / 'film.pt') != receipt['film_sha256']:
            raise ValueError('Selected FiLM SHA mismatch')
        checkpoints[arm] = dict(path=str(cp), receipt=receipt)
    # Verify the inference causal masks against the actual training encoder.
    chosen, lines = {}, []
    for _, lhs in prod.extract_ordered_lhs_plan(source):
        _, candidates = prod.constrained_rhs_candidates(KERNEL, lhs, registry, chosen)
        rhs = candidates[0]
        lines.append(f'{lhs} = {rhs}')
        chosen[lhs.upper()] = rhs
    target = '\n'.join(lines)
    lengths = {}
    encoder = trained.Encoded(tok, registry, max_length=args.max_length,
                              special_ids=contract['special_token_ids'])
    for row in rows:
        prompt = prompt_tokens(tok, row)
        pack = prod.build_deterministic_rhs_pack(source, target, tok,
            directive_domain_registry=registry, kernel_name=KERNEL)
        ids = torch.tensor([prompt + pack.input_ids])
        sample = encoder.get(row, target)
        if not torch.equal(sample['ids'], ids[0]):
            raise ValueError('Inference prompt differs from the actual training serialization')
        mask = causal_kind_mask(tok, source, registry, ids, torch.ones_like(ids), len(prompt))
        torch.testing.assert_close(mask[0], sample['kinds'], rtol=0, atol=0)
        if ids.shape[1] > args.max_length:
            raise ValueError('Kalman source would be truncated')
        lengths[row['context_id']] = dict(prompt=len(prompt), fixture_complete=ids.shape[1])
    manifest = dict(schema='gnosis-selected-stage1-kalman-contexts-v1',
        status='CPU_PREFLIGHT_PASSED_NOT_INFERENCE_RESULTS', source_sha256=sha(source_path),
        action_metadata_sha256=sha(args.source / 'kernel_info.txt'),
        action_count=len(actions), directive_field_count=len(prod.extract_ordered_lhs_plan(source)),
        parent=dict(path=str(args.parent), sha256=PARENT_SHA, contract_sha256=sha(args.parent/'training_contract.json')),
        checkpoints=checkpoints, contexts=rows, token_lengths=lengths,
        directive_domains=registry[prod.normalize_kname(KERNEL)],
        max_length=args.max_length, batch_size=args.batch_size,
        mask_matches_training_encoder=True, objective='PARETO_LATENCY',
        physical_qor_improvement='NOT EVALUATED', source_code=source,
        fixed_prefix_policy='All models/contexts scored under the common parent baseline prediction; no measured Kalman labels',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    return manifest, tok, registry, rows, checkpoints, contract


def validate_assignment(row, source, registry):
    plan = prod.extract_ordered_lhs_plan(source)
    assignment = prod.parse_assignment_dict(row)
    expected = {lhs.upper() for _, lhs in plan}
    if set(assignment) != expected:
        raise ValueError('Incomplete Kalman assignment')
    prior = {}
    for _, lhs in plan:
        _, choices = prod.constrained_rhs_candidates(KERNEL, lhs, registry, prior)
        if assignment[lhs.upper()] not in choices:
            raise ValueError('Out-of-domain directive')
        prior[lhs.upper()] = assignment[lhs.upper()]
    return assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--training-run', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--max-length', type=int, default=6144)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory; existing results are protected')
    args.out.mkdir(parents=True)
    torch.manual_seed(123)
    manifest, tok, registry, rows, checkpoints, contract = preparation(args)
    write_json(args.out/'manifest.json', manifest)
    if args.preflight_only:
        print('CPU PREFLIGHT PASSED', args.out/'manifest.json', flush=True)
        return
    trained.validate_gpu()
    uid = os.environ['CUDA_VISIBLE_DEVICES']
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    if any(p[1].strip() == uid and int(p[0]) != os.getpid()
           for l in apps.splitlines() if len(p := l.split(',')) == 2):
        raise RuntimeError('Selected V100 has another compute process; refusing GPU contention')
    base = inf.load_base_model(contract['model'], revision=contract['model_revision'])
    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    with warnings.catch_warnings(record=True) as caught:
        model = PeftModel.from_pretrained(base, str(args.parent), adapter_name='mailohls_base', is_trainable=False)
    if any('missing adapter keys' in str(w.message).lower() for w in caught):
        raise RuntimeError('Parent adapter did not load completely')
    da.isolate_parent_token_serialization(model)
    config = LoraConfig.from_pretrained(str(Path(checkpoints['A']['path'])/'device_adapt'))
    if config.trainable_token_indices:
        raise ValueError('Residual adapter must not overwrite parent token rows')
    model.add_adapter('device_adapt', config)
    film = trained.ResidualFiLM(model.config.hidden_size, len(model.get_base_model().model.layers)).cuda()
    hooks = trained.InstalledFiLM(model, film)
    proxy = CandidateModel(model, tok, registry, hooks, film, args.max_length)
    reference = None
    outputs = []
    for name in ['parent', 'A_selected', 'B_selected_ON', 'B_selected_OFF']:
        if name == 'parent':
            model.base_model.set_adapter('mailohls_base')
            model.requires_grad_(False)
        else:
            arm = 'A' if name.startswith('A_') else 'B'
            cp = Path(checkpoints[arm]['path'])
            verify_loaded_residual(model, cp)
            da.activate(model, trainable=False)
            if arm == 'B':
                film.load_state_dict(torch.load(cp/'film.pt', map_location='cuda', weights_only=True), strict=True)
        model.eval(); film.eval()
        enabled = name == 'B_selected_ON'
        for row in rows:
            prompt = prompt_tokens(tok, row)
            proxy.bind_case(row, prompt, name, enabled)
            started = time.monotonic()
            print('BEGIN', name, row['context_id'], flush=True)
            with torch.inference_mode():
                prediction, trace = prod.constrained_decode_rhs_by_candidate_scoring(
                    model=proxy, tok=tok, prompt_ids=prompt, source_text=row['input'],
                    kernel_name=KERNEL, directive_domain_registry=registry,
                    candidate_batch_size=args.batch_size, return_score_trace=True)
                assignment = validate_assignment(prediction, row['input'], registry)
                if reference is None:
                    reference = dict(assignment)
                fixed = prod.score_teacher_forced_rhs_candidates(
                    model=proxy, tok=tok, prompt_ids=prompt, source_text=row['input'],
                    kernel_name=KERNEL, reference_assignments=reference,
                    directive_domain_registry=registry, candidate_batch_size=args.batch_size)
            result = dict(model=name, context_id=row['context_id'], platform={k:v for k,v in row.items() if k!='input'},
                directives=assignment, prediction=prediction, candidate_scores=trace,
                fixed_prefix_candidate_scores=fixed, fixed_prefix_reference=reference,
                source_domain_complete=True, seconds=time.monotonic()-started,
                prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                film_enabled=enabled, model_batches=proxy.calls,
                physical_qor_improvement='NOT EVALUATED')
            outputs.append(result)
            write_json(args.out/f'{name}__{row["context_id"]}.json', result)
            print('COMPLETE', name, row['context_id'], 'seconds',round(result['seconds'],1), flush=True)
    manifest.update(status='INFERENCE_COMPLETED', predictions=len(outputs),
                    gpu=torch.cuda.get_device_name(0), precision='FP16 autocast; NF4 frozen backbone',
                    implementation_sha256=sha(Path(__file__)))
    write_json(args.out/'manifest.json', manifest)
    write_json(args.out/'predicted_directives.json', [
        {k:v for k,v in r.items() if k not in ['candidate_scores','fixed_prefix_candidate_scores','fixed_prefix_reference']}
        for r in outputs])
    print('ALL INFERENCES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
