#!/usr/bin/env python3
"""SALT-HLS FiLM-B + frozen GMT + Stage-2 structural-XAttn inference.

Run on the pinned Davinci V100 with the *same source revision* used for training.
Requires independent, source-only historical GNN/GMT memory for every requested
unseen evaluation kernel. Never substitutes zero, shuffled, or training memory.

Source layout: SOURCES_ROOT/<kernel>/kernel_info.txt and one labeled C/C++ file
(or SOURCES_ROOT/<kernel>/source.txt containing the template). If multiple source
files are present, name the intended one source.cpp, source.c, or source.hpp.

Output uses the SALT_experiments.jsonl directive schema. No HLS run is performed:
latency and hardware resource counts are always null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

DEPLOY = Path('/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113')
V100 = 'GPU-8c6e3ac2-5348-2b86-5514-52852fedd21f'
#    'GAN', 'Kalman', 'chstone-aes', 'chstone-jpeg', 'syr2k', 'trmm-opt',
KERNELS = (
    'covariance', 'jacobi-2d', 'rosetta-3d-rendering',
)
DEVICE_ALIASES = {
    'zcu104': 'xczu7ev-ffvc1156-2-e',
    'xczu7ev-ffvc1156-2-e': 'xczu7ev-ffvc1156-2-e',
    'u200': 'xcu200-fsgd2104-2-e',
    'xcu200-fsgd2104-2-e': 'xcu200-fsgd2104-2-e',
}
LABEL = re.compile(r'^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)', re.I)
VARIABLE = re.compile(r'\bvariable\s*=\s*([A-Za-z_]\w*)', re.I)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_path(directory):
    preferred = [directory / n for n in ('source.cpp', 'source.c', 'source.hpp', 'source.h', 'source.txt')]
    named = [p for p in preferred if p.is_file()]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        raise ValueError(f'Multiple canonical source files in {directory}: {named}')
    candidates = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in {'.c', '.cpp', '.cc', '.h', '.hpp'})
    if len(candidates) != 1:
        raise ValueError(f'Expected one labeled source file in {directory}; found: {candidates}')
    return candidates[0]


def array_variables(source_text):
    """Resolve Lk array variable names from *source* pragmas, never from labels."""
    result = {}
    label = None
    for line in source_text.splitlines():
        m = LABEL.match(line)
        if m:
            label = (m[1] or m[2]).upper()
        v = VARIABLE.search(line)
        if label and v and 'array_partition' in line.lower():
            old = result.setdefault(label, v[1])
            if old != v[1]:
                raise ValueError(f'Conflicting array names for {label}: {old} / {v[1]}')
    return result


def as_directives(assignments, actions, source_text):
    variables = array_variables(source_text)
    plan = {}
    for label, action in actions.items():
        if action.kind == 'loop':
            pipe = f'AUTO{{_PIPE_{label}}}'
            unroll = f'AUTO{{_UNROLL_{label}}}'
            plan[label] = {
                'action_id': label, 'kind': 'loop',
                'pipeline_ii': int(assignments[pipe]),
                'unroll_factor': int(assignments[unroll]),
            }
        elif action.kind == 'array':
            if label not in variables:
                raise ValueError(f'No source-verified array variable for {label}')
            t = assignments[f'AUTO{{_ARRAY_T_{label}}}'].lower()
            f = int(assignments[f'AUTO{{_ARRAY_F_{label}}}'])
            d = int(assignments[f'AUTO{{_ARRAY_D_{label}}}'])
            plan[label] = {
                'action_id': label, 'kind': 'array', 'variable': variables[label],
                'partition_type': t, 'partition_factor': f, 'partition_dim': d,
            }
        else:
            raise ValueError(f'Unsupported action kind: {action.kind}')
    return [plan[k] for k in sorted(plan, key=lambda x: int(x[1:]))]


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', default='all', help='One of the nine evaluation kernels, or all')
    p.add_argument('--device', default='zcu104', help='zcu104, u200, or exact FPGA part')
    p.add_argument('--budget', type=float, default=100, help='Percentage of each FPGA resource, e.g. 100 or 50')
    frequency = p.add_mutually_exclusive_group()
    frequency.add_argument('--clock-ns', type=float, default=None, help='Target clock period in nanoseconds (default 10)')
    frequency.add_argument('--frequency-mhz', type=float, help='Clock frequency in MHz; 100 MHz = 10 ns')
    p.add_argument('--objective', default='latency', choices=['latency'], help='Latency-only Stage-1 supervision')
    p.add_argument('--checkpoint-step', type=int, default=450, help='Saved checkpoint step: 0, 150, 300, or 450')
    p.add_argument('--deployment', type=Path, default=DEPLOY)
    p.add_argument('--sources-root', type=Path, required=True, help='Nine labeled source directories with kernel_info.txt')
    p.add_argument('--memory-dir', type=Path, required=True, help='Authentic, frozen-GMT inference memory bank for these kernels')
    p.add_argument('--output', type=Path, default=Path('stage2_experiments.jsonl'))
    p.add_argument('--dry-run', action='store_true', help='Validate data/checkpoint/memory without loading the GPU model')
    return p.parse_args()


def main():
    a = get_args()
    a.deployment = a.deployment.resolve()
    a.sources_root = a.sources_root.resolve()
    a.memory_dir = a.memory_dir.resolve()
    a.output = a.output.resolve()
    a.repo = a.deployment / 'code'
    a.parent = Path('/home/evouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run')
    a.film_checkpoint = a.deployment / 'assets/film/B/checkpoint-0600'
    a.max_length = 7168
    a.seed = 123
    a.lr_gate, a.lr_xattn, a.weight_decay = 2e-4, 1e-4, 0.01
    if a.frequency_mhz is not None and a.frequency_mhz <= 0:
        raise ValueError('--frequency-mhz must be positive')
    if a.clock_ns is not None and a.clock_ns <= 0:
        raise ValueError('--clock-ns must be positive')
    requested_clock = (1000.0 / a.frequency_mhz if a.frequency_mhz is not None
                       else (a.clock_ns if a.clock_ns is not None else 10.0))
    clocks = (3.33, 5.0, 10.0)
    a.clock_ns = min(clocks, key=lambda value: abs(value - requested_clock))
    if abs(a.clock_ns - requested_clock) >= 0.02:
        raise ValueError('Frozen FiLM context supports only 3.33, 5.0, 10.0 ns')
    if a.device.lower() not in DEVICE_ALIASES:
        raise ValueError(f'Unsupported device {a.device}; supported: {sorted(DEVICE_ALIASES)}')
    device_name = DEVICE_ALIASES[a.device.lower()]
    if not (0 < a.budget <= 100):
        raise ValueError('--budget must be in (0, 100]')
    if a.kernel != 'all' and a.kernel.lower() not in [k.lower() for k in KERNELS]:
        raise ValueError(f'Unknown evaluation kernel: {a.kernel}; supported: {KERNELS}')
    kernels = list(KERNELS if a.kernel == 'all' else [next(k for k in KERNELS if k.lower() == a.kernel.lower())])
    out = a.deployment / 'runs/campaign_s123_v1/final'
    checkpoint = out / f'checkpoint-{a.checkpoint_step:04d}'
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'Stage-2 checkpoint not saved yet: {checkpoint}')
    if a.checkpoint_step == 450:
        complete = out / 'COMPLETE.json'
        if not complete.exists() or json.loads(complete.read_text()).get('status') != 'SUCCESS_STAGE2_TRAINED':
            raise RuntimeError('Checkpoint 450 requested, but the campaign has not completed successfully')
    receipt = json.loads((checkpoint / 'receipt.json').read_text())
    assert int(receipt['step']) == a.checkpoint_step
    structural_path = checkpoint / 'structural_xattn.pt'
    if sha256(structural_path) != receipt['structural_state_sha256']:
        raise ValueError('Checkpoint weight SHA-256 mismatch')
    protocol = json.loads((out / 'protocol.json').read_text())
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()
    if protocol_hash != receipt['protocol_sha256']:
        raise ValueError('Checkpoint/run protocol mismatch')
    if protocol['model'] != 'FiLM Arm B checkpoint-0600 frozen + historical GMT B frozen + gated Xattn':
        raise ValueError('Not the selected FiLM+GMT Stage-2 protocol')
    entry = a.repo / 'experiments/film_gmt_stage2_final/train.py'
    if sha256(entry) != protocol['launcher_sha256']:
        raise RuntimeError('Training source differs from the authenticated Stage-2 protocol')
    deployment_paths = json.loads((a.deployment / 'adapter/paths.json').read_text())
    deployment_status = json.loads((a.deployment / 'receipts/deployment_status.json').read_text())
    pinned = '6ee1390611c9047a4b8e2b07200d223b93ee8d16'
    if deployment_paths['commit'] != pinned or deployment_status['commit'] != pinned:
        raise RuntimeError('Deployment is not the pinned Stage-2 source revision')
    if deployment_status.get('post_smoke_integrity', {}).get('status') != 'PASS':
        raise RuntimeError('Post-deployment authenticated integrity receipt not PASS')

    repo_revision = protocol['train_config'].get('repo')
    if str(repo_revision) not in (str(a.repo), '/home/elvouvali/MailoHLS/worktrees/film-gmt-stage2-final'):
        raise ValueError(f'Unexpected train source revision path: {repo_revision}')
    # Set visibility *before* importing any torch-backed repository modules.
    observed = os.environ.get('CUDA_VISIBLE_DEVICES')
    if observed not in (None, V100):
        raise RuntimeError(f'Refusing GPU override CUDA_VISIBLE_DEVICES={observed}')
    os.environ['CUDA_VISIBLE_DEVICES'] = V100
    sys.path.insert(0, str(a.repo))
    from LLM_branch.common import directive_domains as domains, mailohls_contract as mc
    from LLM_branch.common import structural_memory as sm
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
    from experiments.film_gmt_stage2_final import train
    from experiments.film_gmt_stage2_final import bank_contract as bc
    from experiments.gnosis_stage1_ab import train_gnosis_ab as gn

    # Authenticate frozen model identities before allocating GPU memory.
    train.require_hash(a.parent / 'adapter_model.safetensors', train.PARENT_SHA, 'V4 LoRA')
    train.require_hash(a.film_checkpoint / 'device_adapt/adapter_model.safetensors', train.RESIDUAL_SHA, 'FiLM B residual LoRA')
    train.require_hash(a.film_checkpoint / 'film.pt', train.FILM_SHA, 'FiLM B parameters')
    train.require_hash(a.parent / 'directive_domain_registry.json', bc.PARENT_REGISTRY_SHA, 'V4 directive registry')
    cp_receipt = json.loads((a.film_checkpoint / 'receipt.json').read_text())
    if cp_receipt['arm'] != 'B' or int(cp_receipt['step']) != 600:
        raise ValueError('Wrong FiLM checkpoint')
    original_memory = a.repo / 'artifacts/film_gmt_stage2_final_20261113_v1/fallback_memory_original_anchors/memory_B/memory_manifest.json'
    original = json.loads(original_memory.read_text())
    evaluation = json.loads((a.memory_dir / 'memory_manifest.json').read_text())
    for key in ('gnn_checkpoint_sha256', 'gmt_checkpoint_sha256'):
        if evaluation.get(key) != original.get(key):
            raise ValueError(f'Unseen-kernel memory uses a different frozen {key}')
    if evaluation.get('kind') != 'B_GMT':
        raise ValueError('Expected frozen historical GNN + GMT type-B memory, not an ablation')
    for name, expected in evaluation['files'].items():
        if sha256(a.memory_dir / name) != expected:
            raise ValueError(f'Unseen memory file altered: {name}')
    memory, dim = sm.load_memory_bank(str(a.memory_dir), expected_mem_dim=128,
                                      expected_max_slots=64, require_pragma_free_memory=True)
    if dim != 128:
        raise ValueError('Wrong structural embedding dimension')
    registry = inf.load_directive_domain_registry(str(a.parent / 'directive_domain_registry.json'))

    cases = []
    caps = mc.DEVICE_RESOURCES[device_name]
    available = {res: int(round(cap * a.budget / 100)) for res, cap in caps.items()}
    if any(x <= 0 for x in available.values()):
        raise ValueError('Budget too small for positive resource capacities')
    for kernel in kernels:
        folder = a.sources_root / kernel
        metadata = folder / 'kernel_info.txt'
        if not metadata.is_file():
            raise FileNotFoundError(f'Missing kernel action metadata: {metadata}')
        code = source_path(folder)
        prepared = domains.prepare_source_template(kernel, code, a.sources_root)
        actions = domains.load_source_actions(kernel, a.sources_root)
        if len(actions) > 64:
            raise ValueError(f'{kernel}: >64 action slots unsupported by this Stage-2 checkpoint')
        plan = inf.extract_ordered_lhs_plan(prepared)
        if len(plan) != sum(2 if x.kind == 'loop' else 3 for x in actions.values()):
            raise ValueError(f'{kernel}: incomplete action/directive plan')
        normalized = sm.normalize_kname(kernel)
        if normalized not in memory:
            raise RuntimeError(f'{kernel}: no authenticated inference GMT memory; no zero-memory fallback')
        rec = memory[normalized]
        if rec['memory_interface'] != 'rc3_action_evidence_c_v1' or not rec['evidence_metadata']:
            raise ValueError(f'{kernel}: no genuine compiler-role GMT evidence')
        for label in actions:
            slot = int(label[1:]) - 1
            if not bool(rec['action_slot_mask'][slot]):
                raise ValueError(f'{kernel}: {label} has no real structural anchor')
        packed = sm.get_structural_memory_pack_for_kernel(memory, kernel, 64, 128, 'compiler_relational')
        if packed[-1] is None:
            raise ValueError(f'{kernel}: missing compiler-relational attention mask')
        case = inf.InferenceCase(
            kernel_name=kernel, source_text=prepared, obj_mode='PARETO_LATENCY',
            w_lat=1.0, w_area=0.0,
            platform_row={'kernel_name':kernel,'device':device_name,
                          'clock_period':a.clock_ns,'selected_clock_period':a.clock_ns,
                          'frequency_mode':'specified',
                          'avail_bram':available['BRAM_18K'],'avail_dsp':available['DSP'],
                          'avail_ff':available['FF'],'avail_lut':available['LUT'],
                          **{f'budget_frac_{key.lower()}':a.budget / 100
                             for key in ('BRAM_18K','DSP','FF','LUT')}})
        # Source-derived domains are the ONLY legitimate extension for unseen kernels.
        inf.add_missing_source_directive_domains(registry, [case], str(a.sources_root))
        cases.append((case, actions, code, packed))
    print(f'[PREFLIGHT] {len(cases)}/{len(kernels)} kernels have labeled source, domains and real GMT memory')
    print('[TARGET]', device_name, f'{a.clock_ns:g} ns', f'{a.budget:g}%', 'LATENCY')
    print('[CHECKPOINT]', checkpoint, receipt['structural_state_sha256'])
    if a.dry_run:
        print('[DRY RUN PASS] No GPU model loaded; no predictions created')
        return

    # The V100 is exclusive: never compete with the 450-update training campaign.
    query = subprocess.check_output(['nvidia-smi', '-i', V100,
          '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'], text=True)
    fields = [x.strip() for x in query.strip().split(',')]
    if fields[0] != V100 or 'V100' not in fields[1] or int(fields[2]) < 24000:
        raise RuntimeError(f'V100 busy or wrong GPU; do not compete with training: {query}')

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError('Expected exactly one visible Davinci Volta V100')
    parent_contract = json.loads((a.parent / 'training_contract.json').read_text())
    model, film, hooks, tok, _, _, optimizer, scaler, dtype, _ = train.model_and_runtime(a, parent_contract)
    state = torch.load(structural_path, map_location='cpu', weights_only=True)
    expected = {name for name in model.state_dict() if 'gated_cross_attn_layer' in name}
    if set(state) != expected:
        raise RuntimeError(f'Structural checkpoint keys differ: missing={len(expected-set(state))}, extra={len(set(state)-expected)}')
    model.load_state_dict(state, strict=False)
    if train.tensors_sha(model, film) != receipt['frozen_composite_sha256']:
        raise RuntimeError('Frozen V4/FiLM tensors differ from training')
    model.requires_grad_(False)
    model.eval()
    film.eval()
    del optimizer, scaler
    assert dtype == torch.float16

    a.output.parent.mkdir(parents=True, exist_ok=True)
    already = set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            already.add((record.get('kernel'), record.get('device'), record.get('clock_period_ns'),
                         record.get('budget_percent'), record.get('checkpoint_sha256')))
    for case, actions, code, packed in cases:
        key = (case.kernel_name, device_name, a.clock_ns, a.budget, receipt['structural_state_sha256'])
        if key in already:
            print('[SKIP existing]', case.kernel_name)
            continue
        print(f'\n=== {case.kernel_name} | {a.device} | {a.clock_ns:g}ns | {a.budget:g}% | LATENCY ===', flush=True)
        assignment, model_score = decode(case, packed, actions, model, film, hooks, tok, registry, dtype)
        directives = as_directives(assignment, actions, case.source_text)
        for entry in directives:
            print(' ', json.dumps(entry, ensure_ascii=False), flush=True)
        output_row = {
            'kernel': case.kernel_name, 'stage': 'stage-2-gmt', 'objective': 'LATENCY',
            'latency': None, 'resources': None, 'directives': directives,
            'device': device_name, 'clock_period_ns': a.clock_ns,
            'budget_percent': a.budget, 'checkpoint_step': a.checkpoint_step,
            'checkpoint_sha256': receipt['structural_state_sha256'],
            'source_sha256': sha256(code), 'memory_manifest_sha256': sha256(a.memory_dir / 'memory_manifest.json'),
            'mean_nonforced_site_logprob': model_score, 'status': 'inference_only_not_synthesized',
        }
        with a.output.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(output_row, ensure_ascii=False, separators=(',', ':')) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print('[SAVED]', a.output, flush=True)
    print('\n[DONE] Inference only; run Vitis HLS separately for measured latency/resources.')


def decode(case, packed, actions, model, film, hooks, tok, registry, dtype):
    """Deterministic, legal RHS scoring with training-equivalent causal FiLM+XAttn masks."""
    import torch
    from torch.nn import functional as F
    from LLM_branch.common import mailohls_contract as mc
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
    from experiments.gnosis_stage1_ab import train_gnosis_ab as gn
    device = torch.device('cuda:0')
    fields = mc.target_prompt_fields(case.platform_row)
    sections = mc.build_prompt_sections(case.source_text, case.obj_mode, fields)
    base = [token for section in sections for token in tok(section, add_special_tokens=False)['input_ids']]
    prefix = base + mc.selected_clock_response_token_ids(case.platform_row, tok)
    if len(prefix) > 7168:
        raise ValueError(f'{case.kernel_name}: prompt length {len(prefix)} > 7168; no truncation permitted')
    routing_idx = torch.tensor([len(base)], device=device, dtype=torch.long)
    kv, mask, relations = (x.to(device) if x is not None else None for x in packed)
    ctx = gn.context_features(case.platform_row).to(device)[None, None, :]
    labels = inf.extract_ordered_lhs_plan(case.source_text)
    chosen, scores = {}, []
    current_label = None
    model.condition_structural_memory(kv, mask, relations)
    try:
        for label, lhs in labels:
            if label != current_label:
                prefix.extend(tok(f'<{label}>\n', add_special_tokens=False)['input_ids'])
                current_label = label
            prefix.extend(tok(f'{lhs} = ', add_special_tokens=False)['input_ids'])
            original = inf.get_rhs_candidates_for_lhs(case.kernel_name, lhs, registry)
            candidates = mc.filter_semantic_candidates(lhs, original, chosen,
                         registry[inf.normalize_kname(case.kernel_name)])
            if not candidates:
                raise ValueError(f'No valid candidate values for {case.kernel_name}/{lhs}')
            if len(candidates) == 1:
                winner, win_score = candidates[0], None
            else:
                alternatives = []
                for rhs in candidates:
                    rhs_ids = tok(rhs + '\n', add_special_tokens=False)['input_ids']
                    nbase, ncand = len(prefix), len(rhs_ids)
                    if not rhs_ids or nbase + ncand > 7168:
                        raise ValueError(f'{case.kernel_name}: empty or oversized RHS completion for {lhs}')
                    ids = torch.tensor([prefix + rhs_ids], dtype=torch.long, device=device)
                    xmask = inf.make_candidate_xattn_apply_mask(
                        full_length=nbase + ncand, base_len=nbase,
                        candidate_len=ncand, device=device)
                    kinds = torch.zeros_like(ids, dtype=torch.long)
                    kinds[:, nbase-1:nbase-1+ncand] = gn.kind_id(lhs)
                    with torch.inference_mode(), hooks.bind(ctx, kinds, enabled=True):
                        with torch.autocast('cuda', dtype=dtype):
                            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False, routing_start_idx=routing_idx,
                                xattn_apply_mask=xmask).logits
                        selected = logits[:, nbase-1:nbase-1+ncand, :].float()
                        gold = torch.tensor(rhs_ids, dtype=torch.long, device=device)[None, :]
                        token_scores = F.log_softmax(selected, -1).gather(-1, gold[..., None]).squeeze(-1)
                        score = float(token_scores.mean())
                    if not math.isfinite(score):
                        raise FloatingPointError(f'Nonfinite directive score: {case.kernel_name}/{lhs}')
                    alternatives.append((score, rhs))
                win_score, winner = max(alternatives, key=lambda item: (item[0], item[1]))
                scores.append(win_score)
            chosen[lhs.strip().upper()] = winner
            prefix.extend(tok(winner + '\n', add_special_tokens=False)['input_ids'])
        mc.validate_directive_assignments(chosen)
        return chosen, (sum(scores) / len(scores) if scores else None)
    finally:
        model.clear_structural_memory()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
