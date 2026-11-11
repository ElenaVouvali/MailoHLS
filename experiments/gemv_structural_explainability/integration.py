"""Scoped extension of the maintained trainer, rather than another LLM wrapper.

One alignment module lives under the first selected attention module so existing
optimizer, frozen-Stage1 checks, checkpoint and resume paths own its parameters.
All decoder layers read the same freshly computed memory for a forward/backward.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from experiments.gemv_structural_explainability.audit import sha, GNN_SHA
from experiments.gemv_structural_explainability.memory_alignment import MemoryAlignment, SCHEMA, to_device


@contextmanager
def aligned_runtime(production, raw_dir, mode):
    from LLM_branch.common import structural_memory as common
    manifest_path = Path(raw_dir)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['gnn_sha256'] != GNN_SHA:
        raise ValueError('Alignment bank uses a different GNN')
    raw = {}
    for kernel,record in manifest['packs'].items():
        path = Path(raw_dir)/record['file']
        if sha(path) != record['sha256']: raise ValueError('Raw memory hash mismatch')
        pack = torch.load(path,map_location='cpu',weights_only=False)
        if pack['gnn_sha256'] != GNN_SHA: raise ValueError('Raw encoder identity mismatch')
        raw[common.normalize_kname(kernel)] = pack
    old_move = production.move_structural_modules_to_model_device
    old_condition = production.LengthGroupedTrainer._condition_structural_memory_from_kernel_names
    old_get = common.get_structural_memory_pack_for_kernel
    old_dump = production.dump_json
    holder = {}

    def move(model):
        old_move(model)
        attention = next(m for n,m in model.named_modules() if n.endswith('gated_cross_attn_layer.attn'))
        # Fork RNG so extra parameters do not perturb matched fusion init.
        if not hasattr(attention,'alignment'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(123)
                attention.alignment = MemoryAlignment(manifest['edge_dim'],mode).to(
                    device=attention.to_q.weight.device,dtype=torch.float32)
        holder['alignment'] = attention.alignment
        holder['device'] = attention.to_q.weight.device

    def get(bank,kernel_name,max_slots,mem_dim,structural_routing='exact_slot'):
        if max_slots != 64 or mem_dim != 128 or structural_routing != 'compiler_relational':
            raise ValueError('Alignment requires absolute 64-slot, 128D routed memory')
        module = holder['alignment']
        device = next(module.parameters()).device if mode!='action' else holder['device']
        pack = to_device(raw[common.normalize_kname(kernel_name)],device)
        result = module(pack)
        return result['memory'],result['mask'],result['relation']

    def condition(trainer,model,kernel_names):
        device = next(model.parameters()).device
        holder['device'] = device
        items = [get(trainer.mem_bank,k,64,128,'compiler_relational') for k in kernel_names]
        memory,mask,relation = common.pad_memory_pack_batch(
            [x[0][0] for x in items],[x[1][0] for x in items],[x[2][0] for x in items])
        model.condition_structural_memory(memory,mask,action_relation_mask=relation)

    def dump(path,payload):
        if Path(path).name=='training_contract.json':
            payload=dict(payload,structural=dict(payload.get('structural',{}),
                embedding_mode='static_pre_npt::conv1_jkmax_raw_v1'))
            payload = dict(payload,structural_alignment=dict(schema=SCHEMA,mode=mode,
                raw_manifest_sha256=sha(manifest_path),gnn_sha256=GNN_SHA,
                summary_tokens_per_action=2 if mode=='gmt' else 0,
                raw_dir=str(Path(raw_dir).resolve()),normalization='raw first|JK states',
                trained_status='see trainer_state global_step and completion receipt'))
        return old_dump(path,payload)

    production.move_structural_modules_to_model_device = move
    production.LengthGroupedTrainer._condition_structural_memory_from_kernel_names = condition
    common.get_structural_memory_pack_for_kernel = get
    production.dump_json = dump
    try:
        yield holder
    finally:
        production.move_structural_modules_to_model_device = old_move
        production.LengthGroupedTrainer._condition_structural_memory_from_kernel_names = old_condition
        common.get_structural_memory_pack_for_kernel = old_get
        production.dump_json = old_dump
