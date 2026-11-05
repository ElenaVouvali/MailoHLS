"""Authenticate checkpoint; freeze four requests and CPU-side provenance."""
from __future__ import annotations

import argparse
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import transformers
import peft

from LLM_branch.common import directive_domains, source_tool_policy
from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
from .common import (ADAPTER, DEFAULT_OUT, EXPECTED_ADAPTER, EXPECTED_CONTRACT, GPU, HISTORY,
                     PACKAGE, POLICY, ROOT, contract, digest_object, prompt_parts, read_json, sha, write_json)


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return dict(argv=args, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def requests(source):
    devices = ['xczu7ev-ffvc1156-2-e', 'xcu200-fsgd2104-2-e']
    specifications = [('R1', devices[0], 10.0, 1.0), ('R2', devices[0], 10.0, 0.2),
                      ('R3', devices[0], 3.33, 1.0), ('R4', devices[1], 10.0, 1.0)]
    rows = []
    for name, device, clock, fraction in specifications:
        row = dict(context_id=name, kernel_name='gemv', input=source, objective='PARETO_LATENCY', obj_mode='PARETO_LATENCY',
                   device=device, clock_period=clock, frequency_mode='specified',
                   available_clock_periods=[clock], resource_budget_id=f'uniform-{int(fraction * 100)}')
        row.update({field: int(round(contract.DEVICE_RESOURCES[device][resource] * fraction))
                    for resource, field in contract.AVAIL_FIELD_BY_RESOURCE.items()})
        rows.append(row)
    return rows


def authenticate():
    c = read_json(ADAPTER / 'training_contract.json')
    assert sha(ADAPTER / 'adapter_model.safetensors') == EXPECTED_ADAPTER
    assert sha(ADAPTER / 'training_contract.json') == EXPECTED_CONTRACT
    assert c['stage'] == 'stage1' and c['objective'] == 'PARETO_LATENCY'
    assert c['model'] == 'deepseek-ai/deepseek-coder-6.7b-base'
    assert c['model_revision'] == 'ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912'
    assert c['prompt_schema_version'] == contract.PROMPT_SCHEMA_VERSION == 2
    for name in ('response_prefix_policy', 'clock_supervision_policy', 'directive_supervision_policy', 'semantic_domain_policy'):
        assert c[name] == getattr(contract, name.upper())
    assert read_json(ADAPTER / 'adapter_config.json')['r'] == 8
    assert sha(ADAPTER / 'directive_domain_registry.json') == c['directive_domain_registry_sha256']
    tok = inference.load_checkpoint_tokenizer(str(ADAPTER), c)
    return c, tok


def prepare(out):
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise FileExistsError(f'Frozen package exists: {out}; use it or choose a fresh --out')
    out.mkdir(parents=True, exist_ok=True)
    c, tok = authenticate()
    history = read_json(HISTORY)
    # The Stage-1 branch lacks these original inputs. Copy their exact authenticated
    # bytes from the existing GNN worktree, without changing either source branch.
    input_root = PACKAGE / 'inputs'
    input_root.mkdir(exist_ok=True)
    original = Path('/tmp/MailoHLS-gnn-refinement/paper_figures/gemv_rc3_20261105')
    for name, expected in [('gemv.c', history['source']['sha256']),
                           ('kernel_info.txt', history['source']['metadata_sha256'])]:
        target = input_root / name
        source = original / 'inputs' / name
        if not target.exists():
            target.write_bytes(source.read_bytes())
        assert sha(target) == expected
    source = (input_root / 'gemv.c').read_text()
    rows = requests(source)
    actions = directive_domains.load_source_actions('gemv', input_root)
    registry = inference.load_directive_domain_registry(str(ADAPTER / 'directive_domain_registry.json'))
    added = inference.add_missing_source_directive_domains(
        registry, [inference.InferenceCase('gemv', source, 'PARETO_LATENCY', 1., 0., rows[0], None)], str(input_root))
    write_json(out / 'effective_gemv_domains.json', registry['gemv'])
    request_path = out / 'requests.jsonl'
    request_path.write_text(''.join(__import__('json').dumps(r, sort_keys=True) + '\n' for r in rows))
    manifest_rows = []
    for row in rows:
        text, ids = prompt_parts(row, tok)
        prompt_path = out / 'prompts' / (row['context_id'] + '.txt')
        prompt_path.parent.mkdir(exist_ok=True)
        prompt_path.write_text(text)
        markers = []
        for action in actions:
            token_id = tok.convert_tokens_to_ids(f'<SRC_{action}>')
            positions = [i for i, value in enumerate(ids) if value == token_id]
            assert len(positions) == 1
            markers.append(dict(action=action, token_id=token_id, positions=positions,
                                token_text=tok.decode([token_id])))
        write_json(prompt_path.with_suffix('.tokens.json'), dict(token_ids=ids, source_markers=markers))
        manifest_rows.append(dict(request_id=row['context_id'], request_sha256=digest_object(row),
                                  prompt_sha256=sha(prompt_path), prompt_token_ids_sha256=digest_object(ids),
                                  prompt_token_count=len(ids), capacities=contract.DEVICE_RESOURCES[row['device']],
                                  available_resources={k: row[v] for k, v in contract.AVAIL_FIELD_BY_RESOURCE.items()}))
    manifest = dict(schema='gemv-stage1-v4-four-requests-v1', requests=manifest_rows,
                    requests_file_sha256=sha(request_path), source_sha256=sha(input_root / 'gemv.c'),
                    metadata_sha256=sha(input_root / 'kernel_info.txt'),
                    source_origin_commit=history['source']['origin_commit'], source_origin_path=history['source']['path'],
                    snapshot_input_dir=str(input_root), candidate_policy='legacy', score_reduction='mean',
                    decode_mode='greedy', num_samples=1, sample_seed=123,
                    candidate_max_prefix_tokens=0, candidate_keep_head_tokens=256,
                    debug_topk=10000, effective_domains_sha256=sha(out / 'effective_gemv_domains.json'))
    write_json(out / 'manifest.json', manifest)
    files = ['adapter_model.safetensors', 'adapter_config.json', 'training_contract.json',
             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'directive_domain_registry.json']
    checkpoint = dict(adapter_dir=str(ADAPTER), files={name: sha(ADAPTER / name) for name in files},
                      training_contract=c, special_token_to_id={t: tok.convert_tokens_to_ids(t) for t in c['special_tokens']},
                      registry_extension=added, quantization=dict(load_in_4bit=True, quant_type='nf4',
                      double_quant=True, compute_dtype_on_v100s='float16'), weights_loaded=False)
    write_json(out / 'checkpoint.json', checkpoint)
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(c['model'], revision=c['model_revision'], local_files_only=True)
    (out / 'native_decoder_forward.py.txt').write_text(inspect.getsource(LlamaDecoderLayer.forward))
    graph = read_json(original / 'gemv.canonical.json')
    write_json(out / 'structure.json', dict(reference_worktree=str(original),
        reference_commit=command(['git', '-C', str(original.parents[1]), 'rev-parse', 'HEAD']),
        hashes={name: sha(original / name) for name in ('gemv.canonical.json', 'gemv.compact.json', 'gemv.mlir', 'manifest.json')},
        graph_top_level_keys=list(graph),
        architecture=dict(model_type=config.model_type, layers=config.num_hidden_layers,
                          hidden_size=config.hidden_size, attention_heads=config.num_attention_heads,
                          decoder_class='LlamaDecoderLayer', source_sha256=sha(out / 'native_decoder_forward.py.txt'))))
    # Authenticate historical evidence independently. These are never fresh predictions.
    hmodel = history['models']['multi_reference']
    assert checkpoint['files']['adapter_model.safetensors'] == hmodel['adapter_sha256']
    assert checkpoint['files']['training_contract.json'] == hmodel['training_contract_sha256']
    historical = {}
    policy = source_tool_policy.load_policy(POLICY)
    for request_id, historical_id in [('R1', 'gemv-zcu104-10ns-full'), ('R4', 'gemv-u200-10ns-full')]:
        record = hmodel['results'][historical_id]
        assignments = inference.parse_assignment_dict(record['prediction'])
        source_tool_policy.validate_source_assignment(assignments, actions)
        source_tool_policy.validate_complete_assignment(assignments, actions, policy)
        historical[request_id] = dict(record, evidence='historical; NOT rerun', source_valid=True, tool_v2_valid=True)
    for value in history['outputs']['multi_reference'].values():
        assert sha(ROOT / value['path']) == value['sha256']
    write_json(out / 'historical_reference.json', dict(history_report_sha256=sha(HISTORY),
               outputs_authenticated=True, policy_sha256=sha(POLICY), requests=historical))
    state = dict(hostname=platform.node(), python=sys.executable, torch=torch.__version__,
                 transformers=transformers.__version__, peft=peft.__version__,
                 expected_gpu_uuid=GPU, expected_gpu_name='Tesla V100S-PCIE-32GB',
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 cuda_available=torch.cuda.is_available(), cuda_count=torch.cuda.device_count(),
                 gpu_query=command(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']),
                 processes=command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader']),
                 branch=command(['git', 'branch', '--show-current']), commit=command(['git', 'rev-parse', 'HEAD']),
                 preexisting_worktree_status=command(['git', 'status', '--short']))
    try:
        torch.cuda.init()
    except RuntimeError as error:
        state['cuda_init_error'] = str(error)
    write_json(out / 'environment.json', state)
    write_json(out / 'status.json', dict(status='PREPARED' if state['cuda_available'] else 'BLOCKED_GPU',
               official_requests_completed=0, hidden_vectors_captured=0,
               blocker=state.get('cuda_init_error'), checkpoint_authenticated=True,
               tokenizer_authenticated=True, source_and_metadata_authenticated=True))
    print(f'Prepared {out}; CUDA available={state["cuda_available"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == '__main__':
    main()
