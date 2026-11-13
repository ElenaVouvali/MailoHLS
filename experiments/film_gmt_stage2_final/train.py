#!/usr/bin/env python3
"""SALT-HLS FiLM Arm B -> frozen historical GMT memory -> gated structural Stage-2.

Intentionally independent from the ordinary single-adapter Stage-2 launcher.
Requires the MailoHLS research repository on PYTHONPATH; no source files changed.

Modes:
  audit  CPU-only source/adapter/supervision/GMT memory checks; no GPU allocated.
  smoke One real optimizer update, after parity and causal/recompute tests; fresh output.
  train Full optimizer budget with periodic checkpoint and validation; fresh output.
  resume Continue an existing full run from a saved, authenticated optimizer checkpoint.

No fabricated structural evidence, no silent missing-kernel fallback, no FiLM
fine-tuning, no action-slot reindexing, no direct test-label access.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

PARENT_SHA = "28dd67b87e10852c2e25975c0330259ac2dd09c4c9dff9cf84c3fff1bc3bf164"
RESIDUAL_SHA = "6628fe1f268f93ce760428ca0c50b46dbcbfe55e00d45fba97a20142e89b2f80"
FILM_SHA = "d2697977c97c467c7f433b134dff3031c4e842f3b8f2cf9f0c57613c1ecf46ae"
HIST_GNN_SHA = "34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623"
KINDS = ("PIPE", "UNROLL", "ARRAY_T", "ARRAY_F", "ARRAY_D")

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(s) for s in f if s.strip()]

def require_hash(path, expected, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = file_sha(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}: {path}")
    return actual

def get_git(repo):
    def run(args):
        return subprocess.check_output(args, cwd=repo, text=True).strip()
    return {"head": run(["git", "rev-parse", "HEAD"]),
            "dirty_diff": run(["git", "diff", "--binary", "--", "LLM_branch", "experiments", "GNN_branch"]),
            "status": run(["git", "status", "--porcelain", "--untracked-files=no"])}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("audit", "smoke", "train", "resume"), required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True, help="Pinned V4 multi-reference Stage-1 adapter directory")
    p.add_argument("--film-checkpoint", type=Path, required=True, help="GNOSIS Arm B/checkpoint-0600 directory")
    p.add_argument("--gnosis-bank", type=Path, required=True, help="GNOSIS target-conditioning A/B bank (train_sft.jsonl etc.)")
    p.add_argument("--gmt-memory", type=Path, required=True, help="memory_B directory exported by prepare_banks.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-uuid", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--contexts", type=int, default=0, help="0 = every authenticated context; caps cannot drop coverage")
    p.add_argument("--max-length", type=int, default=7168)
    p.add_argument("--eval-interval", type=int, default=150)
    p.add_argument("--val-sft", type=int, default=24)
    p.add_argument("--val-pair-groups", type=int, default=8)
    p.add_argument("--lr-xattn", type=float, default=1e-4)
    p.add_argument("--lr-gate", type=float, default=2e-4)
    p.add_argument("--gate-open-steps", type=int, default=60,
                   help="Predeclared initial gate-unlocking phase; zero initialization is retained")
    p.add_argument("--gate-open-lr", type=float, default=8e-4,
                   help="Gate-only LR during unlocking; attention projections keep --lr-xattn")
    p.add_argument("--probe-sft", type=int, default=3,
                   help="Fixed held-out SFT contexts for paired xattn-on/off forward probes")
    p.add_argument("--rank-budget", type=float, default=.20)
    p.add_argument("--rank-clock", type=float, default=.05)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--min-train-families", type=int, default=10)
    p.add_argument("--min-val-families", type=int, default=2)
    p.add_argument("--allow-explicit-subset", action="store_true", help="Explicitly permit loss of GNOSIS bank families/kernels; excluded names saved")
    p.add_argument("--resume-from", type=Path, help="For --mode resume, exact saved checkpoint dir")
    p.add_argument("--cohort", choices=("expanded-recovery", "original-film-31+2"), required=True)
    return p.parse_args()

def validate_gpu(uuid, min_free_mib=0):
    if not uuid.startswith("GPU-"):
        raise ValueError("Supply --gpu-uuid GPU-... to isolate A2 or V100S")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != uuid:
        raise ValueError(f"CUDA_VISIBLE_DEVICES={visible!r} does not match requested {uuid!r}")
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"], text=True)
    entries = [[v.strip() for v in line.split(",")] for line in out.splitlines() if line.strip()]
    rows = [x for x in entries if x[0] == uuid]
    if len(rows) != 1:
        raise ValueError(f"GPU UUID not found on this machine: {uuid}")
    if int(rows[0][2]) < min_free_mib:
        raise RuntimeError(f"GPU insufficient free memory: {rows[0]}")
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    other = [line for line in occupied.splitlines() if line.split(",")[0].strip() == uuid and int(line.split(",")[-1]) != os.getpid()]
    if other:
        raise RuntimeError(f"GPU has another compute PID; refusing contention: {other}")
    print("[GPU]", rows[0], flush=True)

def prepare_inputs(a):
    """CPU-scope validation; never read final/test rows."""
    from LLM_branch.common import structural_memory as sm
    from LLM_branch.train import train_SFT_xattn_new as prod
    a.repo = a.repo.resolve(); a.parent=a.parent.resolve(); a.film_checkpoint=a.film_checkpoint.resolve()
    a.gnosis_bank=a.gnosis_bank.resolve(); a.gmt_memory=a.gmt_memory.resolve(); a.output=a.output.resolve()
    assert (a.repo/"LLM_branch/train/train_SFT_xattn_new.py").exists(), "Wrong repository/worktree"
    if not (a.repo/"experiments/gnosis_stage1_ab/train_gnosis_ab.py").is_file():
        raise RuntimeError("This script must run against research/gnosis-stage1-film-ab-20261112 (or a compatible worktree), NOT the historical GMT worktree lacking GNOSIS FiLM source")
    require_hash(a.parent/"adapter_model.safetensors", PARENT_SHA, "V4 parent")
    require_hash(a.film_checkpoint/"device_adapt/adapter_model.safetensors", RESIDUAL_SHA, "B residual LoRA")
    require_hash(a.film_checkpoint/"film.pt", FILM_SHA, "B trained FiLM")
    cp_receipt=json.loads((a.film_checkpoint/"receipt.json").read_text())
    assert cp_receipt["arm"]=="B" and int(cp_receipt["step"])==600
    assert cp_receipt["parent_sha256"]==PARENT_SHA
    assert cp_receipt['film_sha256']==FILM_SHA and cp_receipt['lora_sha256']==RESIDUAL_SHA
    from experiments.film_gmt_stage2_final import bank_contract as bc
    if cp_receipt['bank_manifest_sha256']!=bc.ORIGINAL_BANK_SHA:
        raise ValueError('Selected FiLM Stage-1 bank authentication changed')
    stage1_protocol=json.loads((a.film_checkpoint.parent/'protocol.json').read_text())
    require_hash(a.parent/'training_contract.json',stage1_protocol['parent_contract_sha256'],'original V4 training contract')
    require_hash(a.parent/'directive_domain_registry.json',bc.PARENT_REGISTRY_SHA,'original V4 legal domains')
    parent_contract=json.loads((a.parent/"training_contract.json").read_text())
    assert parent_contract["stage"]=="stage1" and parent_contract["objective"]=="PARETO_LATENCY"
    manifest=json.loads((a.gnosis_bank/"manifest.json").read_text())
    if a.cohort=='expanded-recovery':
        bc.verify_recovered_bank(a.gnosis_bank,a.parent/'directive_domain_registry.json',a.repo)
    elif manifest['schema']!='gnosis-target-conditioning-ab-v1' or file_sha(a.gnosis_bank/'manifest.json')!=bc.ORIGINAL_BANK_SHA:
        raise ValueError('Fallback must be the exact original selected 31+2 bank')
    for f,digest in manifest["outputs_sha256"].items():
        require_hash(a.gnosis_bank/f, digest, f"GNOSIS bank {f}")
    data={fold:{kind:jsonl(a.gnosis_bank/f"{fold}_{kind}.jsonl") for kind in ("sft","pairs")} for fold in ("train","val")}
    assert all(data[fold][kind] for fold in data for kind in data[fold]), "Missing SFT or measured pair supervision"
    if a.allow_explicit_subset:raise ValueError('Subset bypass is forbidden for final integration')
    # Each source assignment and every paired alternative must satisfy the frozen
    # registry. Existing Stage-1 authentication remains independent of recovery.
    registry=prod.load_directive_domain_registry(str(a.parent/'directive_domain_registry.json'))
    from experiments.gnosis_stage1_ab.prepare_gnosis_ab import validate_row
    canonical=json.loads((a.repo/'Data/manifest.json').read_text())
    eligible={e['kernel'] for e in canonical['kernels'] if e['complete_source_configuration_eligible'] and e['split']!='test'}
    from Preprocessing.create_jsonl import load_actions,source_with_placeholders
    sources={e['kernel']:e for e in canonical['kernels']}
    checked={}
    for fold in data:
        for kind in data[fold]:
            for entry in data[fold][kind]:
                row=entry['row'];k=row['kernel_name']
                if k not in checked:
                    source=a.repo/sources[k]['files']['source']['path']
                    checked[k]=source_with_placeholders(source,load_actions(k,source.read_text()))
                if row['input'].strip()!=checked[k].strip():raise ValueError('Bank source template differs: '+k)
                targets=[row['target']] if kind=='sft' else [entry['preferred_target'],entry['rejected_target']]
                for target in targets:
                    reason=validate_row(dict(row,target=target),eligible,registry)
                    if reason:raise ValueError(k+': incomplete or illegal supervised assignment: '+reason)
    # Audit all imported bank files against the authenticated SHA-manifest.
    mem_manifest=json.loads((a.gmt_memory/"memory_manifest.json").read_text())
    assert mem_manifest["schema"]=="mailohls-matched-historical-gmt-stage2-v1"
    assert mem_manifest["kind"]=="B_GMT" and mem_manifest["gnn_checkpoint_sha256"]==HIST_GNN_SHA
    assert mem_manifest["gmt_checkpoint_sha256"], "No trained GMT checkpoint receipt"
    # A 40-update GMT smoke is not an appropriate foundation for the final run.
    # Require the actual retained training receipt and selected checkpoint.
    gmt_selected=a.gmt_memory.parent/"gmt_selected.pt"
    require_hash(gmt_selected, mem_manifest["gmt_checkpoint_sha256"], "selected GMT encoder")
    import torch
    gmt_state=torch.load(gmt_selected,map_location="cpu",weights_only=True)
    if gmt_state.get("schema")!="mailohls-historical-gmt-train-only-v1" or int(gmt_state.get("steps",0))!=600:
        raise RuntimeError("Final GMT bank must have exactly 600 successful train-only auxiliary updates")
    from experiments.film_gmt_stage2_final.assets import verify_receipt
    assets=verify_receipt(a.gnosis_bank/'final_assets.json',a.gnosis_bank)
    if set(assets['kernels'])!={name.removesuffix('.memory.pt') for name in mem_manifest['files']}:
        raise ValueError('Final source/graph receipt and memory kernel identities differ')
    for name,h in mem_manifest["files"].items(): require_hash(a.gmt_memory/name,h,"GMT memory")
    bank,dim=sm.load_memory_bank(str(a.gmt_memory), expected_mem_dim=128, expected_max_slots=64,
                                  require_pragma_free_memory=True)
    assert dim==128
    kernels_by_fold={fold:{x["row"]["kernel_name"] for x in data[fold]["sft"]} for fold in data}
    for fold in data:
        kernels_by_fold[fold].update(x["row"]["kernel_name"] for x in data[fold]["pairs"])
    expected_counts=(49,4) if a.cohort=='expanded-recovery' else (31,2)
    if tuple(len(kernels_by_fold[n]) for n in ('train','val'))!=expected_counts:
        raise ValueError('Selected supervision cohort lost a required source kernel')
    missing={fold:sorted(k for k in kernels_by_fold[fold] if sm.normalize_kname(k) not in bank) for fold in data}
    if any(missing.values()) and not a.allow_explicit_subset:
        raise ValueError("GMT memory misses GNOSIS training/validation kernels. Rebuild authenticated bank or pass --allow-explicit-subset after reviewing: "+str(missing))
    excluded={fold:missing[fold] for fold in missing}
    for fold in data:
        for kind in data[fold]:
            data[fold][kind]=[x for x in data[fold][kind] if sm.normalize_kname(x["row"]["kernel_name"]) in bank]
        if not all(data[fold][kind] for kind in ("sft","pairs")):
            raise RuntimeError(f"Insufficient authenticated {fold} supervised examples in GMT cohort")
    families={fold:sorted({x["row"].get("_family",prod.family_id_from_kernel_name(x["row"]["kernel_name"])) for x in data[fold]["sft"]}) for fold in data}
    assert not(set(families["train"]) & set(families["val"])), "Family leakage between train and validation"
    if len(families["train"])<a.min_train_families or len(families["val"])<a.min_val_families:
        raise RuntimeError(f"Insufficient independent families after GMT intersection: {families}")
    # Action identity and routes are required. No action-only fallback is permitted.
    for fold in data:
        for k in kernels_by_fold[fold]-set(missing[fold]):
            rec=bank[sm.normalize_kname(k)]
            if rec["memory_interface"]!="rc3_action_evidence_c_v1" or not rec["evidence_metadata"]:
                raise RuntimeError(f"{k}: missing GMT evidence keys")
            if rec["kv"].size(0)<=64 or not rec["mask"][64:].any():
                raise RuntimeError(f"{k}: no active GMT evidence")
            rel=rec["relation_mask"]
            if rel is None or not rel[:64,64:].any():
                raise RuntimeError(f"{k}: evidence not reachable by action routes")
            for ev in rec["evidence_metadata"]:
                aidx=int(ev["action_slot"]); assert 0<=aidx<64
                assert ev["group"] in ("memory","compute_dependence","control")
    provenance={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "schema":"film-gmt-stage2-preflight-v1",
        "stage1":{"parent_sha256":PARENT_SHA,"residual_sha256":RESIDUAL_SHA,"film_sha256":FILM_SHA,"selected_step":600},
        "gnosis_bank_manifest_sha256":file_sha(a.gnosis_bank/"manifest.json"),
        "gmt_manifest_sha256":file_sha(a.gmt_memory/"memory_manifest.json"),
        "gmt_checkpoint_sha256":mem_manifest["gmt_checkpoint_sha256"],
        "historical_gnn_sha256":HIST_GNN_SHA,
        "coverage":{"train_families":families["train"],"val_families":families["val"],
                    "excluded_kernel_names":excluded,
                    "rows":{fold:{kind:len(data[fold][kind]) for kind in data[fold]} for fold in data},
                    "memory_files":len(mem_manifest["files"]),"sample_kernels":sorted(set(kernels_by_fold["train"])|set(kernels_by_fold["val"]))},
        "configuration":vars(a).copy(),"git":get_git(a.repo)}
    provenance['cohort']=a.cohort
    provenance['source_files_sha256']={str(p.relative_to(a.repo)):file_sha(p)
        for directory in ['experiments/film_gmt_stage2_final','experiments/historical_mlir_gmt_gemv','experiments/historical_gmt_stage2_ab']
        for p in sorted((a.repo/directory).glob('*.py'))}
    return data,bank,provenance,parent_contract

def fixed_validation(data, seed, n_sft, n_groups):
    from experiments.film_gmt_stage2_final.coverage import physical
    rng=random.Random(seed+119)
    sft=collections.defaultdict(list)
    for row in data["val"]["sft"]:
        sft[physical(row['row'])].append(row)
    selection=[]
    for fam in sorted(sft):
        values=sorted(sft[fam],key=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest())
        selection.append(values[0]); sft[fam]=values[1:]
    extra=[v for vs in sft.values() for v in vs]
    rng.shuffle(extra)
    selection=(selection+extra)[:max(n_sft,len(sft))]
    groups=collections.defaultdict(list)
    for x in data["val"]["pairs"]:
        r=x["row"];groups[(r.get("_family") or x.get("source_family"),x["kind"],x["group_id"])].append(x)
    byfam=collections.defaultdict(list)
    for key,values in groups.items():
        if len(values)==2:byfam[key[0]].append(key)
    for fam in byfam:
        byfam[fam].sort(key=lambda v:hashlib.sha256(repr(v).encode()).hexdigest())
    selected=[]
    while len(selected)<n_groups and any(byfam.values()):
        for fam in sorted(byfam):
            if byfam[fam]:selected.append(byfam[fam].pop(0))
            if len(selected)>=n_groups:break
    if not selected:raise RuntimeError("No complete held-out budget/clock pairs")
    return selection,[v for k in selected for v in groups[k]],selected

def build_schedule(data, steps, context_cap, seed):
    from LLM_branch.train import train_SFT_xattn_new as prod
    per_context=collections.defaultdict(list)
    for x in data["train"]["sft"]:per_context[x["context_id"]].append(x)
    strata=collections.defaultdict(list)
    for c,items in per_context.items():
        r=items[0]["row"]
        strata[(r["kernel_name"],r["device"],float(prod._clock_of(r)))].append(c)
    for k in strata:
        strata[k].sort(key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    selected=[]
    cap=len(per_context) if context_cap==0 else context_cap
    if cap<len(per_context):raise ValueError('Context cap loses authenticated budget/target coverage; use --contexts 0')
    while len(selected)<cap and any(strata.values()):
        for k in sorted(strata):
            if strata[k]:selected.append(strata[k].pop(0))
            if len(selected)>=cap:break
    if not selected:raise RuntimeError("No Stage-2 SFT contexts")
    random.Random(seed).shuffle(selected)
    pairs=data["train"]["pairs"].copy()
    random.Random(seed+20).shuffle(pairs)
    return per_context, selected, pairs

def native_autocast_dtype(torch):
    # PyTorch 2.6 can report software BF16 emulation on Volta. Match the frozen
    # parent's hardware policy: native BF16 on Ampere+, FP16 on V100/V100S.
    native_bf16=torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native_bf16 else torch.float16

def model_and_runtime(a, parent_contract):
    import torch
    from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inf
    from LLM_branch.common import device_adaptation as da, structural_xattn
    from LLM_branch.train import train_SFT_xattn_new as prod
    from experiments.gnosis_stage1_ab import train_gnosis_ab as gn
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    random.seed(a.seed)
    tok=inf.load_checkpoint_tokenizer(str(a.parent),parent_contract)
    registry_path=a.parent/"directive_domain_registry.json"
    if not registry_path.exists():registry_path=a.parent.parent/"directive_domain_registry.json"
    registry=inf.load_directive_domain_registry(str(registry_path))
    base=inf.load_base_model(parent_contract["model"],revision=parent_contract["model_revision"])
    base.resize_token_embeddings(len(tok))
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={"use_reentrant":False})
    model=PeftModel.from_pretrained(base,str(a.parent),adapter_name="mailohls_base",is_trainable=False)
    da.isolate_parent_token_serialization(model)
    cfg=LoraConfig.from_pretrained(str(a.film_checkpoint/"device_adapt"))
    if cfg.trainable_token_indices:
        raise RuntimeError("Residual LoRA must not override parent's special-token embeddings")
    model.load_adapter(str(a.film_checkpoint/"device_adapt"),adapter_name="device_adapt",is_trainable=False)
    da.activate(model,trainable=False)
    for name,module in model.named_modules():
        if module.__class__.__name__=="Dropout":module.p=0.0
    n_layers=len(model.get_base_model().model.layers)
    film=gn.ResidualFiLM(model.config.hidden_size,n_layers).cuda()
    state=torch.load(a.film_checkpoint/"film.pt",map_location="cpu",weights_only=True)
    film.load_state_dict(state,strict=True)
    film.eval();film.requires_grad_(False)
    hooks=gn.InstalledFiLM(model,film)
    # Must extend PEFT wrapper (not frozen backbone) so the forward receives Lk routing.
    structural_xattn.extend_instance(model,structural_xattn.StructuralCrossAttentionMixin)
    model.set_decoder_layers_attr_name(structural_xattn.infer_decoder_layers_attr_name(model))
    placeholder_ids=tok.convert_tokens_to_ids(prod.TARGET_PLACEHOLDER_TOKENS)
    if any(x is None or int(x)<0 for x in placeholder_ids) or len(set(map(int,placeholder_ids)))!=len(placeholder_ids):
        raise RuntimeError("Missing or duplicate <Lk> placeholder tokens from parent tokenizer")
    if any(tok.convert_ids_to_tokens(int(idx))!=token for idx,token in zip(placeholder_ids,prod.TARGET_PLACEHOLDER_TOKENS)):
        raise RuntimeError("Tokenizer placeholders are not bound to their exact Lk token IDs")
    model.init_structural_cross_attention(
        placeholder_token_ids=placeholder_ids,lang_hidden_size=model.config.hidden_size,
        mem_hidden_size=128,cross_attn_every_n_layers=8,xattn_heads=4,
        xattn_dim_head=64,only_attend_immediate_memory=True,mask_mode="segment",
        structural_fusion_placement="post_self_attention_residual")
    if list(model.structural_xattn_layer_indices)!=[8,16,24,32]:
        raise RuntimeError("Expected structural fusion layers 8/16/24/32")
    prod.move_structural_modules_to_model_device(model)
    # Only structural branch trainable; keep parent LoRAs, tokens, FiLM immutable.
    model.requires_grad_(False)
    from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
    for module in model.modules():
        if isinstance(module,GatedCrossAttentionBlock):
            for name,parameter in module.named_parameters():
                if name.startswith("attn.") or name=="attn_gate":parameter.requires_grad_(True)
    prod.install_structural_checkpoint_context(model)
    model.enable_input_require_grads()
    model.train();film.eval()
    params=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not params or any("gated_cross_attn_layer" not in n for n,p in params):
        raise RuntimeError(f"Trainable inventory incorrect: {[n for n,p in params][:20]}")
    if not any(n.endswith("attn_gate") for n,p in params):raise RuntimeError("Gates not trainable")
    opt=torch.optim.AdamW([
        {"params":[p for n,p in params if n.endswith("attn_gate")],"lr":a.lr_gate},
        {"params":[p for n,p in params if not n.endswith("attn_gate")],"lr":a.lr_xattn}],
        weight_decay=a.weight_decay)
    dtype=native_autocast_dtype(torch)
    use_bf16=dtype==torch.bfloat16
    scaler=torch.amp.GradScaler("cuda",enabled=not use_bf16,init_scale=256,growth_interval=200)
    enc=gn.Encoded(tok,registry,max_length=a.max_length,
                   special_ids=parent_contract["special_token_ids"])
    # Wrap Encoded.get non-invasively to preserve the exact completion and
    # registry needed for authoritative Lk alignment checks in the forward.
    old_get=enc.get
    def annotated_get(row,target):
        native=old_get(row,target)
        return dict(native,_target_text=target,_tokenizer=tok,_registry=registry)
    enc.get=annotated_get
    return model,film,hooks,tok,registry,enc,opt,scaler,dtype,params

def bind_sample(model, hooks, sample, row, memory, dtype, backprop):
    import torch
    from LLM_branch.common import structural_memory as sm
    from LLM_branch.common import structural_xattn
    from experiments.gnosis_stage1_ab.train_gnosis_ab import context_features
    from experiments.final_architecture_v3.preferences_objective import selected_logps
    from LLM_branch.train import train_SFT_xattn_new as prod
    device=torch.device("cuda:0")
    ids=sample["ids"].unsqueeze(0).to(device)
    attn=sample["attn"].unsqueeze(0).to(device)
    kinds=sample["kinds"].unsqueeze(0).to(device)
    labels=sample["labels"].to(device)
    weights=sample["weights"].to(device)
    pack=sm.get_structural_memory_pack_for_kernel(memory,row["kernel_name"],64,128,"compiler_relational")
    kv,mk,relation=[x.to(device) if x is not None else None for x in pack]
    slot_ids=structural_xattn.build_placeholder_slot_ids(ids,model.placeholder_token_ids)
    routed=structural_xattn.forward_fill_slot_ids(slot_ids)
    causal=kinds.gt(0)
    if not causal.any():raise ValueError("No supervised FiLM positions")
    if torch.any(routed[causal].eq(0)):
        raise ValueError("FiLM supervised RHS token has no active <Lk> structural route")
    if (routed[causal]>64).any():raise ValueError("RHS target action exceeds 64 original action slots")
    # Validate exact action binding, not just the existence of some reachable Lk.
    # Pairs may have distinct preferred/rejected targets; always use the
    # actual encoded completion rather than the row's default target.
    actual_target=sample.get("_target_text", row.get("target", ""))
    ordered=prod.reorder_target_by_source_order(row["input"],actual_target)
    plan=prod.build_deterministic_rhs_pack(row["input"],ordered,
                                             sample["_tokenizer"],
                                             directive_domain_registry=sample["_registry"],
                                             kernel_name=row["kernel_name"])
    offset=len(sample["ids"])-len(plan.input_ids)
    if offset<0:raise ValueError("Malformed source-to-response serialization")
    expected=torch.zeros_like(kinds)
    for j,key in enumerate(plan.site_keys):
        pos=offset+j
        if key is None or pos<=0 or not float(sample["weights"][pos]):
            continue
        match=re.search(r"_L([1-9][0-9]*)\}",key.upper())
        if not match:raise RuntimeError(f"Invalid supervised action identifier {key}")
        expected[0,pos-1]=int(match.group(1))
    if not torch.equal(expected.gt(0),causal):
        raise RuntimeError("FiLM-kind and structural action positions do not agree")
    if not torch.equal(expected[causal],routed[causal]):
        raise RuntimeError("STRUCTURAL ACTION MISBINDING: route Lk != supervised Lk")
    if not mk[0,routed[causal]-1].all():raise ValueError("RHS references inactive action anchor")
    # Require at least one routed key for every FiLM-conditioned RHS position.
    if not relation[0,routed[causal]-1].any(dim=-1).all():raise ValueError("RHS has no eligible structural memory keys")
    ctx=context_features(row).to(device)[None,None,:]
    model.condition_structural_memory(kv,mk,relation)
    try:
        with hooks.bind(ctx,kinds,enabled=True):
            with torch.autocast("cuda",dtype=dtype):
                result=model(input_ids=ids,attention_mask=attn,use_cache=False,
                             xattn_apply_mask=causal.float())
            # Token weights encode corrected action-normalized nonforced RHS supervision.
            _,nll,_=selected_logps(result.logits,labels,weights)
            score=-nll
            if backprop is not None:
                scaler,coefficient=backprop
                scaler.scale(float(coefficient)*score).backward()
            return float(score.detach())
    finally:
        model.clear_structural_memory()

def evaluate(model,film,hooks,enc,bank,dtype,validation,step):
    import torch
    sft, pairs, keys=validation
    model.eval();film.eval();film.last.clear()
    sft_values=[];pair_values=[]
    with torch.inference_mode():
        for x in sft:
            row=x["row"]
            s=bind_sample(model,hooks,enc.get(row,row["target"]),row,bank,dtype,None)
            sft_values.append({"family":row.get("_family",x.get("source_family")),"nll":-s})
        for x in pairs:
            row=x["row"]
            sp=bind_sample(model,hooks,enc.get(row,x["preferred_target"]),row,bank,dtype,None)
            sn=bind_sample(model,hooks,enc.get(row,x["rejected_target"]),row,bank,dtype,None)
            pair_values.append({"family":row.get("_family",x.get("source_family")),"kind":x["kind"],
                                "group_id":x["group_id"],"margin":sp-sn,"correct":sp>sn})
    fam=sorted({v["family"] for v in pair_values})
    macro=sum(sum(x["correct"] for x in pair_values if x["family"]==f)/sum(x["family"]==f for x in pair_values) for f in fam)/len(fam)
    groups=collections.defaultdict(list)
    for v in pair_values:groups[(v["family"],v["kind"],v["group_id"])].append(v)
    by_family={f:sum(len(v)==2 and all(x["correct"] for x in v) for (fa,_,_),v in groups.items() if fa==f)/max(1,sum(fa==f for fa,_,_ in groups)) for f in fam}
    group_macro=sum(by_family.values())/len(by_family)
    from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
    gates=[float(m.attn_gate.detach().tanh()) for m in model.modules() if isinstance(m,GatedCrossAttentionBlock)]
    film_numerics=film.last[-min(len(film.last),24):]
    signal=structural_effect_probe(model,film,hooks,enc,bank,dtype,validation, getattr(model,"_film_gmt_probe_sft",3))
    result={"step":step,"structural_gate_tanh":gates,
            "film_effective_fp16_fraction":(sum(x["effective"] for x in film_numerics)/max(1e-10,sum(x["intended"] for x in film_numerics))) if film_numerics else None,
            "structural_effect":signal,
            "sft_nll":sum(x["nll"] for x in sft_values)/len(sft_values),
            "macro_pair_accuracy":macro,"macro_two_context_group_accuracy":group_macro,
            "n_validation_families":len(fam),"n_pairs":len(pair_values),"n_complete_groups":len(groups),
            "per_family_group_accuracy":by_family,"pair_values":pair_values}
    model.train();film.eval()
    return result

def structural_effect_probe(model, film, hooks, enc, bank, dtype, validation, sample_cap=3):
    """Measure actual post-rounding structural residual on fixed held-out cases.

    The same frozen FiLM is enabled for both paths.  OFF bypasses only the
    four new gated structural blocks.  Neither score difference nor attention
    strength alone is a physical QoR improvement; report both explicitly.
    """
    import torch
    from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
    model.eval(); film.eval()
    blocks = [b for b in model.modules() if isinstance(b, GatedCrossAttentionBlock)]
    if len(blocks) != 4:
        raise RuntimeError(f"Expected four structural blocks for diagnostics, got {len(blocks)}")
    per_layer = [[] for _ in blocks]
    debug = [[] for _ in blocks]
    record_on = [True]
    old_flags = [(b.collect_diagnostics, b.attn.collect_diagnostics) for b in blocks]
    handles = []
    def capture(index):
        def fn(module, args, kwargs, output):
            if not record_on[0]: return
            apply = kwargs.get("xattn_apply_mask")
            if apply is None: return
            active = apply.bool()
            if active.ndim == 3: active = active.squeeze(-1)
            if not bool(active.any()): return
            before = args[0][active].float()
            after = output[active].float()
            # Crucially: actual addition after output dtype rounding, not the
            # projected attention norm or scalar gate magnitude alone.
            effect = (after - before).square().mean().sqrt()
            base = before.square().mean().sqrt().clamp_min(1e-12)
            ratio = float((effect/base).item())
            per_layer[index].append(ratio)
        return fn
    examples = validation[0][:max(1, sample_cap)]
    paired = []
    try:
        for idx, b in enumerate(blocks):
            b.collect_diagnostics = True
            b.attn.collect_diagnostics = True
            handles.append(b.register_forward_hook(capture(idx), with_kwargs=True))
        with torch.inference_mode():
            for entry in examples:
                row=entry["row"]; sample=enc.get(row,row["target"])
                on = bind_sample(model,hooks,sample,row,bank,dtype,None)
                for i,b in enumerate(blocks):
                    dbg=dict(b.last_debug)
                    debug[i].append({
                        "multi_key_token_fraction":dbg.get("multi_key_token_fraction"),
                        "legal_keys_per_query_mean":dbg.get("keys_per_routed_token_mean"),
                        "pre_add_gated_to_hidden_rms":dbg.get("gated_residual_to_hidden_rms_ratio"),
                        "gate_tanh":dbg.get("gate_tanh"),
                    })
                old_forwards=[b.forward for b in blocks]
                try:
                    record_on[0]=False
                    for b in blocks:
                        b.forward=lambda x,*args,**kwargs: x
                    off = bind_sample(model,hooks,sample,row,bank,dtype,None)
                finally:
                    for b,fn in zip(blocks,old_forwards): b.forward=fn
                    record_on[0]=True
                paired.append({"kernel":row["kernel_name"],
                               "film_plus_xattn_score":on,
                               "film_only_score":off,
                               "score_delta":on-off})
    finally:
        for h in handles: h.remove()
        for b,(flag,attn_flag) in zip(blocks,old_flags):
            b.collect_diagnostics=flag
            b.attn.collect_diagnostics=attn_flag
        model.train();film.eval()
    return {
        "n_samples":len(paired),
        "mean_abs_score_delta":sum(abs(e["score_delta"]) for e in paired)/len(paired),
        "per_sample":paired,
        "per_layer": [{
            "layer":(i+1)*8,
            "effective_post_rounding_rms_ratio_mean":sum(values)/max(1,len(values)),
            "effective_post_rounding_rms_ratio_max":max(values) if values else 0.,
            "gate_tanh":float(blocks[i].attn_gate.detach().tanh().item()),
            "diagnostic_samples":debug[i],
        } for i,values in enumerate(per_layer)],
    }

def tensors_sha(model, film):
    import torch
    h=hashlib.sha256()
    from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
    # Do not hash the entire quantized 6.7B backbone: it is frozen and would
    # waste many GiB of host memory. Audit the exact mutable Stage-1 inventories.
    included = 0
    for name,p in sorted(model.named_parameters()):
        if "gated_cross_attn_layer" not in name and (
            "lora_" in name or "trainable_tokens_delta" in name
        ):
            h.update(name.encode());h.update(p.detach().float().cpu().numpy().tobytes())
            included += 1
    if included < 2:
        raise RuntimeError("No frozen LoRA/token tensors were authenticated")
    for name,p in sorted(film.named_parameters()):
        h.update(name.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()

def save_checkpoint(a,model,film,opt,scaler,scheduler,step,metric,parent_frozen_sha,protocol_sha):
    import torch
    from LLM_branch.train import train_SFT_xattn_new as prod
    path=a.output/f"checkpoint-{step:04d}"
    if path.exists():raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    path.mkdir()
    # Does not save V4/residual/FiLM again: frozen references are pinned in protocol.
    torch.save(prod.get_structural_xattn_state_dict(model),path/"structural_xattn.pt")
    torch.save({"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                "rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                "python_rng":random.getstate()},path/"optimizer_state.pt")
    state={"step":step,"metric":metric,"protocol_sha256":protocol_sha,
           "frozen_composite_sha256":parent_frozen_sha,
           "structural_state_sha256":file_sha(path/"structural_xattn.pt")}
    write_json(path/"receipt.json",state)
    return path

def restore_checkpoint(a,model,opt,scaler,scheduler,protocol_sha,parent_sha):
    import torch
    folder=a.resume_from.resolve()
    rec=json.loads((folder/"receipt.json").read_text())
    if rec["protocol_sha256"]!=protocol_sha or rec["frozen_composite_sha256"]!=parent_sha:
        raise ValueError("Resume protocol/frozen checkpoint mismatch")
    state=torch.load(folder/"structural_xattn.pt",map_location="cpu",weights_only=True)
    expected={k for k in model.state_dict() if "gated_cross_attn_layer" in k}
    if set(state)!=expected:raise RuntimeError(f"Structural resume state keys mismatch: missing={expected-set(state)}, extra={set(state)-expected}")
    model.load_state_dict(state,strict=False)
    history=torch.load(folder/"optimizer_state.pt",map_location="cpu",weights_only=False)
    opt.load_state_dict(history["optimizer"]);scheduler.load_state_dict(history["scheduler"]);scaler.load_state_dict(history["scaler"])
    torch.set_rng_state(history["rng"]);torch.cuda.set_rng_state_all(history["cuda_rng"])
    random.setstate(history["python_rng"])
    return int(rec["step"])

def run(a):
    if a.mode=="audit" and a.output.exists():
        raise FileExistsError("Audit output must be fresh")
    if a.mode in ("smoke","train") and a.output.exists():
        raise FileExistsError("Training output must be fresh: "+str(a.output))
    if a.mode=="resume" and (not a.output.exists() or not a.resume_from):
        raise ValueError("Resume requires existing output and --resume-from")
    os.chdir(a.repo.resolve());sys.path.insert(0,str(a.repo.resolve()))
    data,bank,provenance,parent_contract=prepare_inputs(a)
    from experiments.film_gmt_stage2_final.coverage import exposure_report,token_audit,paired_groups
    schedule=build_schedule(data,a.steps,a.contexts,a.seed)
    val=fixed_validation(data,a.seed,a.val_sft,a.val_pair_groups)
    provenance['training_context_coverage']=exposure_report(data,schedule,val,a.steps,a.grad_accum)
    from LLM_branch.train import train_SFT_xattn_new as prod
    registry=prod.load_directive_domain_registry(str(a.parent/'directive_domain_registry.json'))
    provenance['token_audit']=token_audit(data,a.parent,parent_contract,registry,a.max_length)
    print("[AUDIT]",json.dumps({k:v for k,v in provenance.items() if k in ("coverage","stage1","gmt_checkpoint_sha256")},default=str),flush=True)
    if a.mode=="audit":
        a.output.mkdir(parents=True)
        write_json(a.output/"preflight.json",provenance)
        print("[AUDIT] PASS: CPU identities and memory coverage; no GPU training performed",flush=True)
        return
    devices=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name","--format=csv,noheader"],text=True)
    selected_device=[line for line in devices.splitlines() if line.split(",")[0].strip()==a.gpu_uuid]
    if len(selected_device)!=1:raise ValueError("GPU UUID not found")
    required_free=14000 if "A2" in selected_device[0] else 22000
    validate_gpu(a.gpu_uuid,min_free_mib=required_free)
    import torch
    from LLM_branch.train import train_SFT_xattn_new as prod
    model,film,hooks,tok,registry,encoder,opt,scaler,dtype,params=model_and_runtime(a,parent_contract)
    model._film_gmt_probe_sft=a.probe_sft
    contexts,selected,pairs=schedule
    rank_groups=list(paired_groups(pairs).values())
    # For A2, successful smoke does not prove every long-context input fits.
    print("[SCHEDULE]",json.dumps({"contexts":len(selected),"train_pairs":len(pairs),"grad_accum":a.grad_accum,
                                   "val_sft":len(val[0]),"val_pairs":len(val[1])}),flush=True)
    a.output.mkdir(parents=True,exist_ok=a.mode=="resume")
    if a.mode!="resume":
        write_json(a.output/"preflight.json",provenance)
    protocol={"schema":"salt-film-gmt-stage2-v1","mode":"full" if a.mode in ("train","resume") else "smoke",
              "train_config":{k:v for k,v in vars(a).items() if k not in ("mode","resume_from")},
              "launcher_sha256":file_sha(Path(__file__).resolve()),
              "dataset_sha256":provenance["gnosis_bank_manifest_sha256"],
              "memory_sha256":provenance["gmt_manifest_sha256"],"parent":provenance["stage1"],
              "model":"FiLM Arm B checkpoint-0600 frozen + historical GMT B frozen + gated Xattn",
              "loss":"action-normalized nonforced RHS SFT + measured target-switch logistic pair ranking",
              "causal_apply_mask":"FiLM kind>0 @ token immediately before supervised RHS value",
              "routing":"compiler_relational; original 64 action anchors + 3 typed GMT slots per action; no fallback",
              "selection":"family-macro correct-both-target-contexts, then pair accuracy, then -SFT NLL"}
    # Freeze the run contract for deterministic resume (output path included).
    serialized=json.dumps(protocol,sort_keys=True,default=str).encode()
    protocol_sha=hashlib.sha256(serialized).hexdigest()
    if a.mode=="resume":
        current=json.loads((a.output/"protocol.json").read_text())
        if current!=json.loads(serialized):raise ValueError("Run protocol changed before resume")
    else:write_json(a.output/"protocol.json",protocol)
    # Parity: with zero xattn gates, resulting loss and candidate scores must be
    # identical when structural branch is disabled. Use full model, not just layers.
    example=data['train']['sft'][provenance['token_audit']['longest_training_sft']['index']]['row']
    encoded=encoder.get(example,example["target"])
    frozen_before=tensors_sha(model,film)
    model.eval();film.eval()
    with torch.inference_mode():
        initial=bind_sample(model,hooks,encoded,example,bank,dtype,None)
        from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
        blocks=[m for m in model.modules() if isinstance(m,GatedCrossAttentionBlock)]
        if len(blocks)!=4:raise RuntimeError("Expected exactly four structural blocks")
        if any(float(m.attn_gate.detach().abs())!=0 for m in blocks):
            raise RuntimeError("Structural attention gates must initialize exactly at zero")
        original_forwards=[b.forward for b in blocks]
        try:
            for block in blocks:
                block.forward=lambda x,*args,**kwargs:x
            parity=bind_sample(model,hooks,encoded,example,bank,dtype,None)
        finally:
            for block,fn in zip(blocks,original_forwards):block.forward=fn
        if abs(initial-parity)>1e-4:
            raise RuntimeError(f"FiLM Stage-1/zero-gate parity failed: {initial} vs {parity}")
    model.train();film.eval()
    # First smoke: real forward/backward including checkpoint recomputation.
    opt.zero_grad(set_to_none=True)
    s=bind_sample(model,hooks,encoded,example,bank,dtype,(scaler,-1.))
    grads=[float(p.grad.detach().norm()) for n,p in params if n.endswith("attn_gate") and p.grad is not None]
    if not grads or not all(math.isfinite(x) for x in grads) or max(grads)<=0:
        raise RuntimeError(f"No finite nonzero gate gradients at step zero: {grads}")
    opt.zero_grad(set_to_none=True)
    if a.mode=="smoke":
        # A real gate update on TRAIN data only. The full 450-update run remains
        # a separate fresh process/output. No validation backward is performed.
        bind_sample(model,hooks,encoded,example,bank,dtype,(scaler,-1.))
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _,p in params],a.max_grad_norm)
        previous_scale=float(scaler.get_scale())
        scaler.step(opt);scaler.update()
        if float(scaler.get_scale())<previous_scale:raise FloatingPointError('Smoke optimizer update skipped by AMP')
        opt.zero_grad(set_to_none=True)
        # Probe a fixed, declared nonzero gate to test projection gradients and
        # actual FP16 post-rounding effects; restore gates afterwards. This is a
        # diagnostic perturbation, never a checkpoint or architecture search.
        saved_gates=[b.attn_gate.detach().clone() for b in blocks]
        try:
            with torch.no_grad():
                for block in blocks:block.attn_gate.fill_(math.atanh(.05))
            # Decoder checkpointing is active only in train mode. This is a
            # backward diagnostic, unlike the inference-only ON/OFF probe.
            model.train();film.eval()
            bind_sample(model,hooks,encoded,example,bank,dtype,(scaler,-1.))
            scaler.unscale_(opt)
            projection_grads={n:float(p.grad.detach().float().norm()) for n,p in params
                              if not n.endswith('attn_gate') and p.grad is not None}
            if not projection_grads or max(projection_grads.values())<=0 or not all(math.isfinite(g) for g in projection_grads.values()):
                raise RuntimeError('Opened structural branch has no finite projection gradients')
            effects=structural_effect_probe(model,film,hooks,encoder,bank,dtype,val,2)
            if effects['mean_abs_score_delta']<=0:raise RuntimeError('FP16 structural ON/OFF effect is absent')
            if not all(layer['effective_post_rounding_rms_ratio_mean']>0 for layer in effects['per_layer']):
                raise RuntimeError('A structural layer has no post-rounding effect')
        finally:
            with torch.no_grad():
                for block,gate in zip(blocks,saved_gates):block.attn_gate.copy_(gate)
            opt.zero_grad(set_to_none=True)
        if frozen_before!=tensors_sha(model,film):raise RuntimeError("Frozen model changed during smoke backward")
        write_json(a.output/"smoke.json",{"status":"SMOKE_PASS","score":s,"parity_self":parity,
            "zero_gate_film_parity_abs_delta":abs(initial-parity),'training_optimizer_updates':1,
            'projection_gradient_norms_at_probe_gate':projection_grads,'probe_gate_tanh':.05,
            'structural_effect_probe':effects,'dtype':str(dtype),'longest_training_example':provenance['token_audit']['longest_training_sft'],
            "gate_gradient_norms":grads,"frozen_composite_sha256":frozen_before})
        print("[SMOKE] Successful contextual forward/backward with nonzero structural gate gradients",flush=True)
        return
    # Warmup/cosine: keep the frozen FiLM parent intact while structural gates
    # learn to open, then reduce update magnitude for checkpoint stability.
    if not (0 < a.gate_open_steps < a.steps):
        raise ValueError("Gate-opening phase must be positive and shorter than total training")
    if a.gate_open_lr < a.lr_gate or a.gate_open_lr <= 0:
        raise ValueError("Gate-opening LR should be >= normal gate LR")
    warmup=max(1,math.ceil(.05*a.steps))
    def cosine_factor(i):
        if i<warmup:return (i+1)/warmup
        fraction=min(1.,(i-warmup)/max(1,a.steps-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*fraction))
    def gate_factor(i):
        # Predeclared differential gate curriculum; no manual gate forcing.
        gate_base=(a.gate_open_lr if i<a.gate_open_steps else a.lr_gate)
        return cosine_factor(i)*(gate_base/a.lr_gate)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=[gate_factor,cosine_factor])
    completed=0
    if a.mode=="resume":completed=restore_checkpoint(a,model,opt,scaler,scheduler,protocol_sha,frozen_before)
    best=(-1e20,-1e20,-1e20)
    if a.mode=="resume" and (a.output/"best.json").exists():
        prev=json.loads((a.output/"best.json").read_text())
        m=prev["metrics"];best=(m["macro_two_context_group_accuracy"],m["macro_pair_accuracy"],-m["sft_nll"])
    metrics_path=a.output/"training.jsonl"
    for step in range(completed,a.steps+1):
        if (a.mode!="resume" or step!=completed) and (step==0 or step%a.eval_interval==0 or step==a.steps):
            m=evaluate(model,film,hooks,encoder,bank,dtype,val,step)
            write_json(a.output/f"val_{step:04d}.json",m)
            key=(m["macro_two_context_group_accuracy"],m["macro_pair_accuracy"],-m["sft_nll"])
            cp=save_checkpoint(a,model,film,opt,scaler,scheduler,step,m,frozen_before,protocol_sha)
            if key>best:
                best=key
                write_json(a.output/"best.json",{"checkpoint":str(cp),"step":step,"metrics":m})
            effects=m["structural_effect"]
            per_layer={x["layer"]:round(x["effective_post_rounding_rms_ratio_mean"],7)
                       for x in effects["per_layer"]}
            print("[VAL]",step,"group",m["macro_two_context_group_accuracy"],
                  "pairs",m["macro_pair_accuracy"],"NLL",m["sft_nll"],
                  "xattn_mean_abs_score_delta",effects["mean_abs_score_delta"],
                  "xattn_post_rounding_residual_ratio",per_layer,flush=True)
            # Diagnostic warning only: never reward arbitrary perturbation or
            # tune a representation using the held-out outcomes.
            if step>=a.gate_open_steps and max(per_layer.values(),default=0.0)<5e-4:
                print("[WEAK-STRUCTURAL-FUSION] all four post-rounding structural"
                      " residual ratios < 0.05%; inspect actual gradients and"
                      " multi-key routing before continuing",flush=True)
        if step==a.steps:break
        model.train();film.eval()
        t0=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        sft_scores=[]
        for micro in range(a.grad_accum):
            idx=step*a.grad_accum+micro
            context=selected[idx%len(selected)]
            choices=contexts[context]
            exposure=choices[(idx//len(selected))%len(choices)]
            row=exposure["row"]
            sample=encoder.get(row,row["target"])
            sft_scores.append(bind_sample(model,hooks,sample,row,bank,dtype,(scaler,-1./a.grad_accum)))
        rank_info=None
        for pair in rank_groups[step%len(rank_groups)] if rank_groups else []:
            row=pair["row"]
            pos=encoder.get(row,pair["preferred_target"])
            neg=encoder.get(row,pair["rejected_target"])
            model.eval();film.eval()
            with torch.inference_mode():
                sp=bind_sample(model,hooks,pos,row,bank,dtype,None)
                sn=bind_sample(model,hooks,neg,row,bank,dtype,None)
            margin=sp-sn
            weight=a.rank_budget if pair["kind"]=="budget_switch" else a.rank_clock
            coeff=weight/(len(rank_groups[step%len(rank_groups)])*(1+math.exp(max(-60.,min(60.,margin)))))
            model.train();film.eval()
            bind_sample(model,hooks,pos,row,bank,dtype,(scaler,-coeff))
            bind_sample(model,hooks,neg,row,bank,dtype,(scaler,+coeff))
            rank_info={"kind":pair["kind"],"margin":margin,"weight":weight,"correct":margin>0}
        scaler.unscale_(opt)
        grad_norm=float(torch.nn.utils.clip_grad_norm_([p for n,p in params],a.max_grad_norm))
        if not math.isfinite(grad_norm):raise FloatingPointError("Nonfinite structural gradient")
        previous_scale=float(scaler.get_scale())
        scaler.step(opt);scaler.update()
        if float(scaler.get_scale())<previous_scale:
            raise FloatingPointError("AMP overflow: update skipped. Resume from last checkpoint after diagnosing; do not count as successful update")
        scheduler.step()
        gate_grads=[float(p.grad.detach().float().norm()) for n,p in params if n.endswith("attn_gate") and p.grad is not None]
        proj_grads=[float(p.grad.detach().float().norm()) for n,p in params if not n.endswith("attn_gate") and p.grad is not None]
        from LLM_branch.common.structural_xattn import GatedCrossAttentionBlock
        gate_values=[float(m.attn_gate.detach().tanh()) for m in model.modules() if isinstance(m,GatedCrossAttentionBlock)]
        row={"step":step+1,"sft_nll":-sum(sft_scores)/len(sft_scores),"rank":rank_info,
             "grad_norm":grad_norm,"gate_grad_l2":math.sqrt(sum(g*g for g in gate_grads)),
             "projection_grad_l2":math.sqrt(sum(g*g for g in proj_grads)),
             "gate_tanh":gate_values,
             "gate_lr":opt.param_groups[0]["lr"],"projection_lr":opt.param_groups[1]["lr"],
             "seconds":time.perf_counter()-t0}
        with metrics_path.open("a") as f:f.write(json.dumps(row)+"\n")
        if (step+1)%10==0:print("[TRAIN]",row,flush=True)
    if frozen_before!=tensors_sha(model,film):raise RuntimeError("Frozen V4/residual/FiLM tensors mutated")
    write_json(a.output/"COMPLETE.json",{"status":"SUCCESS_STAGE2_TRAINED","steps":a.steps,
              "best":json.loads((a.output/"best.json").read_text()),
              "frozen_composite_sha256":frozen_before,"protocol_sha256":protocol_sha,
              "dpo_status":"NOT_TRAINED; requires composite-reference-aware Stage3 implementation"})
    print("STAGE2 COMPLETE",a.output,flush=True)

if __name__=="__main__":
    args=parse_args()
    try:run(args)
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise
