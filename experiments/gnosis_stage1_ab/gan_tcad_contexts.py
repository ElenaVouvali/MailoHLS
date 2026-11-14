"""Run the TCAD GAN grid with the selected Stage-1 residual LoRA and FiLM."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import warnings
import zipfile

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from experiments.gnosis_stage1_ab import kalman_contexts as k


def prepare(a):
    m = json.loads((a.prepared / 'manifest.json').read_text())
    with zipfile.ZipFile(a.draft) as z:
        section = z.read('sections/05__Detailed_Design.tex')
        figure = z.read('figures/02__Methodology/target_conditioning_heatmap.pdf')
    assert k.sha(a.draft) == m['draft_zip_sha256']
    assert hashlib.sha256(section).hexdigest() == m['section_sha256']
    assert hashlib.sha256(figure).hexdigest() == m['figure_sha256']
    assert k.sha(a.prepared / 'requests.jsonl') == m['requests_sha256']
    source_dir = a.prepared / 'sources'
    assert k.sha(source_dir / 'GAN/GAN_placeholders.cpp') == m['source_sha256']
    assert k.sha(source_dir / 'GAN/kernel_info.txt') == m['metadata_sha256']
    for name, digest in m['header_sha256'].items():
        assert k.sha(source_dir / 'GAN' / name) == digest
    rows = [json.loads(s) for s in (a.prepared / 'requests.jsonl').read_text().splitlines()]
    expected = {(d, c, b) for d in ('xczu7ev-ffvc1156-2-e', 'xcu200-fsgd2104-2-e')
                for c in (3.33, 5., 10.) for b in (.2, .5, 1.)}
    assert len(rows) == 18
    assert {(r['device'], r['clock_period'], r['resource_budget_frac']) for r in rows} == expected
    source = k.domains.prepare_source_template('GAN', source_dir/'GAN/GAN_placeholders.cpp', source_dir)
    contract = json.loads((a.parent / 'training_contract.json').read_text())
    assert k.sha(a.parent/'adapter_model.safetensors') == k.PARENT_SHA == m['checkpoint_sha256']
    assert k.sha(a.parent/'training_contract.json') == m['contract_sha256']
    assert contract['stage'] == 'stage1' and contract['objective'] == 'PARETO_LATENCY'
    registry_path = a.parent/'directive_domain_registry.json'
    assert k.sha(registry_path) == contract['directive_domain_registry_sha256'] == m['registry_sha256']
    selected = json.loads((a.training_run/'B/best.json').read_text())
    cp = a.training_run/'B'/Path(selected['checkpoint']).name
    receipt = json.loads((cp/'receipt.json').read_text())
    assert receipt['arm'] == 'B' and receipt['parent_sha256'] == k.PARENT_SHA
    assert receipt['step'] == selected['step']
    assert k.sha(cp/'film.pt') == receipt['film_sha256']
    assert k.sha(cp/'device_adapt/adapter_model.safetensors') == receipt['lora_sha256']
    k.prod.TARGET_CFG.device_mode = contract['device_mode']
    registry = k.inf.load_directive_domain_registry(str(registry_path))
    k.inf.add_missing_source_directive_domains(registry, [type('Case', (), dict(
        kernel_name='GAN', source_text=source))()], str(source_dir))
    tok = k.inf.load_checkpoint_tokenizer(str(a.parent), contract)
    chosen, lines = {}, []
    for _, lhs in k.prod.extract_ordered_lhs_plan(source):
        _, choices = k.prod.constrained_rhs_candidates('GAN', lhs, registry, chosen)
        chosen[lhs.upper()] = choices[0]
        lines.append(f'{lhs} = {choices[0]}')
    target = '\n'.join(lines)
    encoder = k.trained.Encoded(tok, registry, max_length=a.max_length,
                                special_ids=contract['special_token_ids'])
    lengths = {}
    for row in rows:
        assert row['kernel_name'] == 'GAN' and row['input'] == source
        assert row['obj_mode'] == 'PARETO_LATENCY'
        fraction = row['resource_budget_frac']
        # The sealed requests contain absolute prompt budgets. FiLM additionally
        # consumes the four fraction features used by its training encoder.
        row.update({key: fraction for key in k.trained.FEATURE_KEYS})
        for res, field in k.mc.AVAIL_FIELD_BY_RESOURCE.items():
            assert row[field] == round(k.mc.DEVICE_RESOURCES[row['device']][res] * fraction)
        prompt = k.prompt_tokens(tok, row)
        sample = encoder.get(row, target)
        pack = k.prod.build_deterministic_rhs_pack(source, target, tok,
            directive_domain_registry=registry, kernel_name='GAN')
        ids = torch.tensor([prompt + pack.input_ids])
        assert torch.equal(ids[0], sample['ids'])
        kinds = k.causal_kind_mask(tok, source, registry, ids, torch.ones_like(ids), len(prompt), 'GAN')
        torch.testing.assert_close(kinds[0], sample['kinds'], rtol=0, atol=0)
        assert ids.shape[1] <= a.max_length
        lengths[row['context_id']] = ids.shape[1]
    manifest = dict(status='CPU_PREFLIGHT_PASSED', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        kernel='GAN', objective='PARETO_LATENCY', requests=18, film_enabled=True,
        parent=str(a.parent), parent_sha256=k.PARENT_SHA, checkpoint=str(cp),
        checkpoint_step=receipt['step'], film_sha256=receipt['film_sha256'], lora_sha256=receipt['lora_sha256'],
        prepared_manifest=m, source_code=source, contexts=rows, token_lengths=lengths,
        directive_field_count=len(lines), mask_matches_training=True,
        candidate_policy='unchanged production checkpoint registry; source-derived GAN extension',
        decoder='greedy constrained full RHS/newline; mean continuation score',
        physical_qor='NOT MEASURED', implementation_sha256=k.sha(Path(__file__)),
        helper_sha256=k.sha(Path(k.__file__)))
    return manifest, rows, contract, registry, tok, cp


def report(out, results):
    fields = list(results[0]['directives'])
    with (out/'directives.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['request', 'device', 'clock_ns', 'budget_pct'] + fields)
        w.writeheader()
        for r in results:
            w.writerow(dict(request=r['context_id'], device=r['device'], clock_ns=r['clock_ns'],
                budget_pct=r['budget_pct'], **r['directives']))
    text = ['# GAN: 18 TCAD requests, selected Stage-1 B checkpoint 600 with FiLM ON', '',
        'Objective: PARETO_LATENCY. These are model predictions; no synthesis was run.', '']
    for i, r in enumerate(results, 1):
        text.extend([f'## Request {i}: {r["context_id"]}', '', '```text', r['prediction'], '```', ''])
    (out/'DIRECTIVES.md').write_text('\n'.join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared', 'draft', 'parent', 'training-run', 'out'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--max-length', type=int, default=6144)
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    manifest, rows, contract, registry, tok, cp = prepare(a)
    k.write_json(a.out/'manifest.json', manifest)
    if a.preflight_only:
        print('CPU_PREFLIGHT_PASSED', flush=True)
        return
    uid = os.environ['CUDA_VISIBLE_DEVICES']
    assert uid == 'GPU-cea3d9e4-d189-a5b9-cf2e-9bd77b8218fb'
    gpu = subprocess.check_output(['nvidia-smi', '-i', uid, '--query-gpu=uuid,name,memory.free',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    assert 'V100S' in gpu and int(gpu.split(',')[-1]) > 28000
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True)
    assert not any(l.split(',')[0].strip() == uid for l in apps.splitlines())
    torch.manual_seed(123)
    base = k.inf.load_base_model(contract['model'], revision=contract['model_revision'])
    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    with warnings.catch_warnings(record=True) as caught:
        model = PeftModel.from_pretrained(base, str(a.parent), adapter_name='mailohls_base', is_trainable=False)
    assert not any('missing adapter keys' in str(w.message).lower() for w in caught)
    k.da.isolate_parent_token_serialization(model)
    cfg = LoraConfig.from_pretrained(str(cp/'device_adapt'))
    assert not cfg.trainable_token_indices
    model.add_adapter('device_adapt', cfg)
    k.verify_loaded_residual(model, cp)
    k.da.activate(model, trainable=False)
    film = k.trained.ResidualFiLM(model.config.hidden_size, len(model.get_base_model().model.layers)).cuda()
    film.load_state_dict(torch.load(cp/'film.pt', map_location='cuda', weights_only=True), strict=True)
    hooks = k.trained.InstalledFiLM(model, film)
    proxy = k.CandidateModel(model, tok, registry, hooks, film, a.max_length)
    model.eval(); film.eval()
    results = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        prompt = k.prompt_tokens(tok, row)
        proxy.bind_case(row, prompt, 'B_selected_ON', True)
        tick = time.monotonic()
        print('BEGIN', index, row['context_id'], flush=True)
        with torch.inference_mode():
            prediction, trace = k.prod.constrained_decode_rhs_by_candidate_scoring(
                model=proxy, tok=tok, prompt_ids=prompt, source_text=row['input'], kernel_name='GAN',
                directive_domain_registry=registry, candidate_batch_size=a.batch_size, return_score_trace=True)
        assignment = k.validate_assignment(prediction, row['input'], registry, 'GAN')
        assert len(assignment) == manifest['directive_field_count']
        result = dict(context_id=row['context_id'], device=row['device'], clock_ns=row['clock_period'],
            budget_pct=round(100*row['resource_budget_frac']), directives=assignment, prediction=prediction,
            candidate_scores=trace, seconds=time.monotonic()-tick, film_enabled=True,
            source_domain_complete=True, physical_qor='NOT MEASURED')
        k.write_json(a.out/f'request_{index:02d}.json', result)
        results.append(result)
        k.write_json(a.out/'predicted_directives.json', results)
        report(a.out, results)
        print('COMPLETE', index, row['context_id'], 'seconds', round(result['seconds'], 1), flush=True)
    assert len(results) == 18
    assert k.sha(cp/'film.pt') == manifest['film_sha256']
    assert k.sha(cp/'device_adapt/adapter_model.safetensors') == manifest['lora_sha256']
    manifest.update(status='INFERENCE_COMPLETED', completed_requests=18, gpu=gpu,
                    precision='FP16 autocast; NF4 frozen backbone', seconds=time.monotonic()-started)
    k.write_json(a.out/'manifest.json', manifest)
    print('ALL 18 REQUESTS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
