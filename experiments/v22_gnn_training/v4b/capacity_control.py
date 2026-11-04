"""One predeclared parameter-count control for the optional FiLM arm.

Only the standard candidate ActionMLP hidden width changes. The RC3 encoder,
output width, loss, normalization, sampling trace and evaluator are inherited
from a completed corrected v3 full-cohort protocol. Width is selected solely
by parameter count, never by validation performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import torch
from torch import nn
from experiments.v22_gnn_training import train, checkpointing
from .train_probe import (_initial_models, _all_parameters, _pair_schedule, _json_sha,
                         _model_state, train_arm, evaluate_arm, require_isolated_gpu, sample_coverage)
from .cohorts import cache_identity
from .provenance import validate_cached_measurements
from .quality import apply_quality, POLICY


def capacity_model(seed, feature_dim, target):
    model=_initial_models(seed,feature_dim)['compiler']
    initial_hash=train.model_state_sha256(model)
    current=sum(p.numel() for p in model.parameters())
    first,last=model.local[0],model.local[2]
    per_hidden=first.in_features+1+last.out_features
    width=min(range(1,2049),key=lambda w:(abs(current+(w-first.out_features)*per_hidden-target),w))
    torch.manual_seed(seed)
    model.local=nn.Sequential(nn.Linear(first.in_features,width),nn.ReLU(),
                              nn.Linear(width,last.out_features),nn.ReLU())
    return model,width,initial_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--release-dir',required=True)
    p.add_argument('--authenticated-audit',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    parent=json.loads((a.reference_run/'protocol.json').read_text())
    receipt=json.loads((a.reference_run/'run_receipt.json').read_text())
    if parent['version']!='v4b-quality-corrected-v3' or parent['quality_policy']!=POLICY:
        raise ValueError('Capacity control requires the corrected frozen v3 protocol')
    root=Path(__file__).resolve().parents[3]
    for name,expected in parent['source_sha256'].items():
        if train.overlay.file_hash(root/name)!=expected:
            raise ValueError(f'Parent protocol source changed: {name}')
    uuid=parent['gpu_uuid']
    require_isolated_gpu('V100S',uuid,6000)
    torch.set_num_threads(parent['cpu_threads'])
    manifest=json.loads(a.manifest.read_text())
    if _json_sha(manifest)!=parent['manifest']: raise ValueError('Cohort changed')
    data=train.PreparedData(a.release_dir,'/tmp/v4b-cap-unused.json',
        selected_kernels=manifest['training_kernels']+manifest['validation_kernels'],
        authenticated_audit_path=a.authenticated_audit)
    cached=torch.load(a.cache,map_location='cpu',weights_only=False)
    if cached['identity']!=cache_identity(manifest): raise ValueError('Cache identity changed')
    prepared=cached['prepared']
    validate_cached_measurements(prepared,data,manifest)
    quality=apply_quality(prepared)
    schedule=_pair_schedule(prepared['groups'],parent['updates'],parent['seed'],True)
    schedule_hash=_json_sha([[prepared['groups'][gi]['key'],fraction,pi] for gi,fraction,pi in schedule])
    if schedule_hash!=receipt['sample_trace_sha256']: raise ValueError('Sample trace changed')
    target=receipt['training']['target_film']['parameter_count']
    model,width,initial_hash=capacity_model(parent['seed'],prepared['profile_feature_dim'],target)
    if initial_hash!=receipt['initial_encoder_hashes']['compiler']:
        raise ValueError('Initial compiler control state changed')
    count=sum(p.numel() for p in model.parameters())
    protocol={**parent,'version':'v4b-capacity-standard-head-v1',
        'parent_protocol_fingerprint':receipt['protocol_fingerprint'],
        'capacity_target_parameters':target,'capacity_actual_parameters':count,'local_hidden_width':width,
        'control_source_sha256':train.overlay.file_hash(Path(__file__)),
        'width_rule':'nearest parameter count; no validation tuning'}
    fingerprint=_json_sha(protocol)
    require_isolated_gpu('V100S',uuid,6000)
    a.output.mkdir(parents=True)
    checkpointing.atomic_json(a.output/'protocol.json',protocol)
    checkpointing.atomic_json(a.output/'quality_audit.json',quality)
    stopped=[]
    signal.signal(signal.SIGTERM,lambda *_:stopped.append(True))
    signal.signal(signal.SIGINT,lambda *_:stopped.append(True))
    device=torch.device('cuda');model.to(device)
    torch.cuda.reset_peak_memory_stats()
    steps=tuple(range(parent['validate_every'],parent['updates']+1,parent['validate_every']))
    result=train_arm('compiler',model,prepared,schedule,device,steps,
        state_path=a.output/'training_state.pt',fingerprint=fingerprint,stop_requested=lambda:bool(stopped))
    result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['ablations']={mode:evaluate_arm('compiler',model,prepared,device,mode)
                         for mode in ('zero','shuffled','global_mean','target_mismatched')}
    result['controlled_action_probe']={'status':'standard_head_control'}
    report={k:v for k,v in result.items() if k!='checkpoints'}
    checkpointing.atomic_torch_save(a.output/'compiler_capacity_selected.pt',{
        'state':_model_state(model),'protocol_fingerprint':fingerprint,
        'selected_step':result['selected_step'],'local_hidden_width':width})
    checkpointing.atomic_json(a.output/'compiler_capacity_report.json',report)
    output={'schema':'v4b-capacity-standard-head-v1-run','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_fingerprint':fingerprint,'manifest_sha256':parent['manifest'],
        'sample_trace_sha256':schedule_hash,'sample_coverage':sample_coverage(prepared['groups'],schedule),
        'training':{'compiler_capacity':report},'test_kernels_loaded':[],
        'matched_parent_protocol_fingerprint':receipt['protocol_fingerprint']}
    checkpointing.atomic_json(a.output/'run_receipt.json',output)
    checkpointing.atomic_json(a.output/'status.json',{'status':'complete','protocol_fingerprint':fingerprint})
    print({'width':width,'parameters':count,'target':target,'selected_step':result['selected_step']})


if __name__=='__main__':main()
