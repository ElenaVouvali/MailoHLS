#!/usr/bin/env python3
"""Historical MailoHLS GEMV: typed MLIR evidence -> GMT-inspired action memory.

PRETRAINING/DIAGNOSTIC only. This is NOT a validated Stage-2 LLM/QoR result.
Import historical.py for authenticated frozen epoch-9 inference; never mix RC3 weights.

Usage (from MailoHLS repository root):
  python -m experiments.historical_mlir_gmt_gemv.historical_gmt_slots prepare --out artifacts/historical_gmt_slots_v1
  python -m experiments.historical_mlir_gmt_gemv.historical_gmt_slots train --out artifacts/historical_gmt_slots_v1 --steps 600
  python -m experiments.historical_mlir_gmt_gemv.historical_gmt_slots evaluate --out artifacts/historical_gmt_slots_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, deque
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = ('memory', 'compute_dependence', 'control')
GROUP = {s:i for i,s in enumerate(GROUPS)}
DIM = 128
NUM_FEATURES = 20
MAX_EVIDENCE_RECORDS = 16384  # strict overflow rejection; never truncate
FLOW_ALLOWED = {0,1,2,5,6,7,8,10,11,12}
FLOW_IDS = {v:i+1 for i,v in enumerate(sorted(FLOW_ALLOWED))} # 0 means missing
ROLE_NAMES = (
    'next_in_block','block_successor','calls','operand','result',
    'mlir_block_adjacency','loop_child','loop_parent',
    'array_root','array_root_reverse','array_read','array_read_reverse',
    'array_write','array_write_reverse','read','read_reverse','write','write_reverse',
    'region_entry','region_exit','iter_init','loop_backedge','loop_result',
    'RAW','WAR','WAW','memory_shape_feature',
)
ROLE_IDS = {s:i+1 for i,s in enumerate(ROLE_NAMES)} # 0 means unknown
CERT_IDS = {'proven':1,'no_dependence':2,'unknown':3} # 0 not provided
READ_ROLES = {'array_read','read','array_read_reverse','read_reverse'}
WRITE_ROLES = {'array_write','write','array_write_reverse','write_reverse'}


def canonical_bytes(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def write(path, data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def yn(v):
    return str(v).lower() in ('true','1','yes')


def numeric(attrs,key,default=0.):
    try: return float(attrs.get(key,default) or default)
    except (ValueError,TypeError):return float(default)


def edge_group(attrs):
    f=int(attrs.get('flow',-1))
    if f in (7,10): return GROUP['memory']
    if f in (1,2,11,12): return GROUP['compute_dependence']
    if f in (0,5,6,8): return GROUP['control']
    return None


def node_group(attrs):
    ty=int(attrs.get('type',-1))
    txt=str(attrs.get('text',''))
    if ty==104 or yn(attrs.get('is_memory',False)) or yn(attrs.get('is_memory_root',False)) or txt.startswith(('memref.','affine.load','affine.store')):
        return GROUP['memory']
    if txt.startswith(('affine.for','scf.for','affine.if','scf.if','pseudo_block')) or yn(attrs.get('is_loop',False)):
        return GROUP['control']
    if txt.startswith(('arith.','affine.apply','affine.yield','scf.yield')) or ty==1:
        return GROUP['compute_dependence']
    return None


def node_feats(attrs):
    v=np.zeros(NUM_FEATURES,dtype=np.float32)
    txt=str(attrs.get('text',''))
    v[0]=float('for' in txt and ('affine.' in txt or 'scf.' in txt))
    v[1]=float(txt.startswith(('affine.if','scf.if')))
    v[2]=float(txt.startswith(('affine.load','memref.load')))
    v[3]=float(txt.startswith(('affine.store','memref.store')))
    v[4]=float(txt.startswith('arith.'))
    v[5]=float(yn(attrs.get('is_memory',False)))
    v[6]=float(int(attrs.get('type',-1))==104)
    v[7]=float(int(attrs.get('type',-1))==1 and not yn(attrs.get('is_memory',False)))
    v[8]=min(numeric(attrs,'memory_rank'),8.)/8.
    v[9]=min(numeric(attrs,'loop_depth'),8.)/8.
    v[10]=math.log1p(max(0.,numeric(attrs,'trip_count')))/10.
    v[11]=min(numeric(attrs,'compiler_operation_effect_read'),1.)
    v[12]=min(numeric(attrs,'compiler_operation_effect_write'),1.)
    v[13]=float(yn(attrs.get('is_memory_root',False)))
    v[18]=float(yn(attrs.get('trip_count_static',False)))
    v[19]=float(yn(attrs.get('compiler_effect_analysis_unknown',False)))
    return v


def edge_feats(attrs):
    v=np.zeros(NUM_FEATURES,dtype=np.float32)
    flow=int(attrs.get('flow',-1));role=str(attrs.get('role',''))
    v[2]=float(role in READ_ROLES)
    v[3]=float(role in WRITE_ROLES)
    v[11]=float(role in READ_ROLES)
    v[12]=float(role in WRITE_ROLES)
    v[14]=float(flow==7)
    v[15]=float(flow==10)
    v[16]=float(flow==12 and attrs.get('certainty')=='proven')
    v[17]=float(flow in (1,11))
    v[18]=float(flow in (0,5,6,8))
    v[19]=float(yn(attrs.get('distance_known',False)))
    return v


def flow_edges(graph, source, allowed_flows):
    for _, t, key, attrs in graph.out_edges(source,keys=True,data=True):
        if int(attrs.get('flow',-1)) in allowed_flows:
            yield t,key,attrs


def loop_body(graph, root):
    """Match authenticated GNN_branch/build_structural_memory._loop_region_distances.

    Crucial: enter the loop body by *region_entry*, NOT by following the
    loop header's next_in_block edge, which may refer to code AFTER the loop.
    Exclude region_exit; include nested regions and compiler-visible calls.
    """
    root=str(root)
    seen=set();q=deque()
    for dst,key,attrs in flow_edges(graph,root,{8}):
        if attrs.get('role')=='region_entry':q.append(str(dst))
    while q:
        here=q.popleft()
        if here in seen:continue
        seen.add(here)
        for _,dst,key,attrs in graph.out_edges(here,keys=True,data=True):
            flow=int(attrs.get('flow',-1));role=str(attrs.get('role',''))
            follow=(flow==8 and role=='region_entry') or (flow==0 and role in ('next_in_block','block_successor')) or (flow==2 and role=='calls')
            if follow:q.append(str(dst))
    return seen


def scoped_nodes(graph, anchor, kind, loop_scopes):
    anchor=str(anchor)
    scope={anchor}
    if kind=='loop':
        scope |= loop_body(graph,anchor)
        base=set(scope)
        # Compiler-linked operands/SSA, roots, accesses, and dependencies.
        for n in base:
            for u,v,key,a in graph.in_edges(n,keys=True,data=True):
                f=int(a.get('flow',-1))
                if f in (1,7,10,11) or (f==12 and a.get('certainty')=='proven'):
                    scope.add(str(u))
            for u,v,key,a in graph.out_edges(n,keys=True,data=True):
                f=int(a.get('flow',-1))
                if f in (1,7,10,11) or (f==12 and a.get('certainty')=='proven'):
                    scope.add(str(v))
    elif kind=='array':
        # Source anchors are array_scope nodes (type 104); consult explicit
        # compiler array_root/array_read/array_write edges only.
        if int(graph.nodes[anchor].get('type',-1))!=104:
            raise ValueError('Array action is not bound to a compiler array_scope')
        for dest,key,attr in flow_edges(graph,anchor,{7}):
            if attr.get('role') in ('array_root','array_read','array_write'):
                scope.add(str(dest))
        base=set(scope)
        # Read/write operations and their directly linked SSA results/operands.
        for n in base:
            if n==anchor:continue
            for u,v,key,a in graph.in_edges(n,keys=True,data=True):
                f=int(a.get('flow',-1))
                if f in (1,10):scope.add(str(u))
            for u,v,key,a in graph.out_edges(n,keys=True,data=True):
                f=int(a.get('flow',-1))
                if f in (1,10):scope.add(str(v))
        # Loop context is proven by an access occurring INSIDE the loop closure.
        access=base-{anchor}
        for loop_node,loop_set in loop_scopes.items():
            if access & loop_set:scope.add(loop_node)
    else: raise ValueError(f'Unknown action type {kind}')
    return scope


def make_evidence(graph, pack, mapping, key, gemv_strict=False):
    nodes=pack['node_states'].detach().float().cpu()
    if nodes.shape!=(len(graph),DIM):
        raise ValueError(f'Historical frozen node rows mismatch: {tuple(nodes.shape)}, {len(graph)}')
    if set(graph.nodes)!=set(str(i) for i in range(len(graph))):
        raise ValueError('Cannot establish numeric node-row identity from historical graph')
    original=pack['node_embs'].float().cpu()
    active=pack['node_embs_mask'].bool().cpu()
    base_relation=torch.as_tensor(pack['action_relation_mask']).bool().cpu()
    if base_relation.shape!=(64,64):
        raise ValueError('Original compiler action-to-action relation mask is not 64x64')
    if original.shape!=(64,128) or int(active.sum())!=len(mapping):
        raise ValueError('Historical action mapping/memory mismatch')
    if gemv_strict and (len(mapping)!=5 or not bool(active[:5].all()) or int(active.sum())!=5):
        raise ValueError('Historical GEMV expected exactly five active slots L1..L5')
    loops={str(m['compiler_node_id']):loop_body(graph,str(m['compiler_node_id'])) for m in mapping if m['action_type']=='loop'}
    observations=[];stats=[]; action_scopes={}
    for idx,m in enumerate(mapping):
        if idx>=len(mapping) or not bool(active[m['absolute_memory_slot']]):raise ValueError('Inactive or inconsistent mapping')
        if gemv_strict and (idx!=m['absolute_memory_slot'] or m['action']!=f'L{idx+1}'):
            raise ValueError('Historical GEMV mapping differs')
        anchor=str(m['compiler_node_id'])
        a=graph.nodes[anchor]
        if str(a.get('action_id'))!=m['action'] or a.get('action_resolution')!='source_location':
            raise ValueError('Authenticated action anchor missing')
        scope=scoped_nodes(graph,anchor,m['action_type'],loops)
        action_scopes[m['action']]=sorted(scope,key=int)
        for n in sorted(scope,key=int):
            group=node_group(graph.nodes[n])
            if group is None:continue
            observations.append(dict(action=idx,group=group,kind='node',src=int(n),dst=int(n),
                 desc=f'node:{n}',facts=node_feats(graph.nodes[n]).tolist(),
                 role_idx=0,flow_idx=0,certainty_idx=0,
                 identity=dict(node_id=n,operation=graph.nodes[n].get('text'),
                      source_location=graph.nodes[n].get('source_location'),
                      memory_rank=graph.nodes[n].get('memory_rank'),
                      trip_count=graph.nodes[n].get('trip_count'))))
        for u,v,ek,attrs in sorted(graph.edges(keys=True,data=True),key=lambda e:(int(e[0]),int(e[1]),str(e[2]))):
            if u not in scope or v not in scope:continue
            group=edge_group(attrs)
            if group is None:continue
            if int(attrs.get('flow',-1))==12 and attrs.get('certainty')!='proven':continue
            observations.append(dict(action=idx,group=group,kind='edge',src=int(u),dst=int(v),
                 desc=f'edge:{u}>{v}:{ek}',facts=edge_feats(attrs).tolist(),
                 role_idx=ROLE_IDS.get(str(attrs.get('role','')),0),
                 flow_idx=FLOW_IDS.get(int(attrs.get('flow',-1)),0),
                 certainty_idx=CERT_IDS.get(str(attrs.get('certainty','')),0),
                 identity=dict(src=u,dst=v,edge_key=str(ek),flow=attrs.get('flow'),role=attrs.get('role'),
                               certainty=attrs.get('certainty'),analysis=attrs.get('analysis'),
                               distance_known=attrs.get('distance_known'))))
    # Every output derives from a concrete record; zero evidence -> masked zero slot.
    if not observations:raise RuntimeError('No compiler evidence discovered')
    if len(observations)>MAX_EVIDENCE_RECORDS:
        raise ValueError(f'Evidence overflow {len(observations)} > {MAX_EVIDENCE_RECORDS}; no truncation')
    count=Counter((r['action'],r['group']) for r in observations)
    if not all(sum(count[(i,g)] for g in range(3))>0 for i in range(len(mapping))):
        raise RuntimeError('Action missing compiler evidence')
    if gemv_strict and not any(r['kind']=='edge' and r['identity']['flow']==12 for r in observations):
        raise RuntimeError('Expected historical GEMV RAW compiler dependence is absent')
    route=torch.zeros(len(mapping),3,len(observations),dtype=torch.bool)
    for j,r in enumerate(observations):route[r['action'],r['group'],j]=True
    feats=torch.tensor([r['facts'] for r in observations],dtype=torch.float32)
    signature=torch.zeros(len(mapping),3,NUM_FEATURES+1)
    for i in range(len(mapping)):
        for g in range(3):
            rows=feats[route[i,g]]
            if len(rows):
                signature[i,g,:NUM_FEATURES]=rows.mean(0)
                signature[i,g,-1]=math.log1p(len(rows))/6
    data=dict(nodes=nodes,anchors=original,active=active,base_relation=base_relation,action_slots=torch.tensor([m['absolute_memory_slot'] for m in mapping],dtype=torch.long),src=torch.tensor([r['src'] for r in observations]),
              dst=torch.tensor([r['dst'] for r in observations]),
              edge=torch.tensor([r['kind']=='edge' for r in observations],dtype=torch.bool),
              group=torch.tensor([r['group'] for r in observations]),facts=feats,
              role_idx=torch.tensor([r['role_idx'] for r in observations]),
              flow_idx=torch.tensor([r['flow_idx'] for r in observations]),
              certainty_idx=torch.tensor([r['certainty_idx'] for r in observations]),
              route=route,signature=signature)
    meta=dict(actions=mapping,records=observations,scopes=action_scopes,
              counts={mapping[i]['action']:{GROUPS[g]:count[(i,g)] for g in range(3)} for i in range(len(mapping))},
              purpose='compiler-derived roles, numeric features and frozen GNN states only; no source strings/IDs as input features',
              source_graph_sha=sha(key))
    return data,meta


class HistoricalRoleGMT(nn.Module):
    """Three separately routed GMT-inspired queries; no arbitrary slot repulsion.

    Every slot has explicit provenance and a queryable compiler evidence subset.
    The original historical 64 x 128 action anchors remain unmodified.
    """
    def __init__(self,dim=128,heads=4):
        super().__init__()
        assert dim%heads==0
        self.dim=dim;self.heads=heads;self.hd=dim//heads
        self.node_norm=nn.LayerNorm(dim)
        self.node_enc=nn.Sequential(nn.Linear(dim+NUM_FEATURES,dim),nn.GELU(),nn.Linear(dim,dim))
        self.edge_enc=nn.Sequential(nn.Linear(2*dim+NUM_FEATURES,dim),nn.GELU(),nn.Linear(dim,dim))
        self.group_embedding=nn.Embedding(len(GROUPS),dim)
        self.role_embedding=nn.Embedding(len(ROLE_NAMES)+1,dim,padding_idx=0)
        self.flow_embedding=nn.Embedding(len(FLOW_IDS)+1,dim,padding_idx=0)
        self.certainty_embedding=nn.Embedding(len(CERT_IDS)+1,dim,padding_idx=0)
        self.evidence_norm=nn.LayerNorm(dim)
        self.queries=nn.Parameter(torch.randn(len(GROUPS),dim)*.02)
        self.action_query=nn.Linear(dim,dim,bias=False)
        self.q_proj=nn.Linear(dim,dim,bias=False)
        self.k_proj=nn.Linear(dim,dim,bias=False)
        self.v_proj=nn.Linear(dim,dim,bias=False)
        self.o_proj=nn.Linear(dim,dim,bias=False)
        self.memory_norm=nn.LayerNorm(dim)
        self.signature_head=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,NUM_FEATURES+1))
        self.log_scale=nn.Parameter(torch.tensor(math.log(4.0)))

    def forward(self,data, drop=0., return_attention=False):
        nodes=self.node_norm(data['nodes'])
        src=nodes[data['src']]; dst=nodes[data['dst']]
        facts=data['facts'];is_edge=data['edge'][:,None]
        evid=torch.where(is_edge,self.edge_enc(torch.cat((src,dst,facts),-1)),
                                   self.node_enc(torch.cat((src,facts),-1)))
        # Compiler flow, exact edge role and knownness are separate learned features.
        # This preserves operand/result, RAW, region-entry and loop-hierarchy
        # distinctions that a coarse read/write histogram would lose.
        relation=(self.role_embedding(data['role_idx'])+
                  self.flow_embedding(data['flow_idx'])+
                  self.certainty_embedding(data['certainty_idx']))
        evid=self.evidence_norm(evid+self.group_embedding(data['group'])+relation)
        action=self.node_norm(data['anchors'][data['action_slots']])
        nact=len(action)
        query=self.queries[None,:,:]+self.action_query(action)[:,None,:]
        q=F.normalize(self.q_proj(query).view(nact,3,self.heads,self.hd),dim=-1)
        k=F.normalize(self.k_proj(evid).view(-1,self.heads,self.hd),dim=-1)
        v=self.v_proj(evid).view(-1,self.heads,self.hd)
        logits=torch.einsum('aghd,rhd->aghr',q,k)*self.log_scale.exp().clamp(1.,20.)
        routes=data['route']
        if drop>0 and self.training:
            keep=(torch.rand(routes.shape[-1],device=routes.device)>drop)
            # Keep at least one record for every nonempty action-group route.
            for a in range(nact):
                for g in range(3):
                    ids=routes[a,g].nonzero().flatten()
                    if len(ids) and not keep[ids].any(): keep[ids[0]]=True
            routes=routes & keep[None,None,:]
        mask=routes.any(-1)
        logits=logits.masked_fill(~routes[:,:,None,:],-10000.)
        attn=torch.softmax(logits,-1)
        attn=torch.where(routes[:,:,None,:],attn,0.)
        values=torch.einsum('aghr,rhd->aghd',attn,v).reshape(nact,3,self.dim)
        summary=self.memory_norm(self.o_proj(values))
        summary=summary.masked_fill(~mask[:,:,None],0.)
        pred=self.signature_head(summary)
        return summary,mask,pred,(attn if return_attention else None)


def loss_fn(model,data,drop=.1):
    summary,mask,pred,_=model(data,drop)
    if not mask.any(): raise RuntimeError('All evidence masked')
    target=data['signature']
    reg=F.mse_loss(pred[mask],target[mask])
    # Similar compiler signatures are positives, NOT arbitrary negatives.
    # Discrimination is supervised on actual compiler-derived signatures.
    sg=target[mask];pr=pred[mask]
    dist=torch.cdist(sg,sg)
    positives=(dist<0.05)
    logits=F.normalize(pr,dim=-1)@F.normalize(sg,dim=-1).T / .15
    logden=torch.logsumexp(logits,dim=1)
    logpos=torch.logsumexp(logits.masked_fill(~positives,-10000.),dim=1)
    contrastive=(logden-logpos).mean()
    return reg+.07*contrastive, {'reconstruction':float(reg.detach()),'contrastive':float(contrastive.detach())}


def device_data(d,device):
    return {k:v.to(device) for k,v in d.items()}



def extract_kernel_records(value):
    """Read kernel identities nested under an explicitly named TRAIN section."""
    names=set()
    if isinstance(value,list):
        for entry in value:
            if isinstance(entry,str):names.add(entry)
            elif isinstance(entry,(list,dict)):names |= extract_kernel_records(entry)
    elif isinstance(value,dict):
        for k,v in value.items():
            if str(k).lower() in ('kernel','kernel_name','name') and isinstance(v,str):
                names.add(v)
            elif str(k).lower() in ('kernels','kernel_names','kernel_records','records','examples','items'):
                names |= extract_kernel_records(v)
    return names


def find_train_names(obj):
    """Extract a kernel allowlist ONLY from an authenticated training split."""
    if not isinstance(obj,(dict,list)):
        return set()
    found=set()
    if isinstance(obj,list):
        for x in obj:found |= find_train_names(x)
        return found
    for k,value in obj.items():
        name=str(k).lower()
        if name in ('train_kernels','training_kernels','train_kernel_names','training_kernel_names'):
            if isinstance(value,list):
                found |= extract_kernel_records(value)
        elif name in ('train','training'):
            found |= extract_kernel_records(value)
        elif isinstance(value,dict):
            found |= find_train_names(value)
    return found


def cohort_action_mapping(graph,pack):
    """Restore action anchors from the actual historical exporter, not labels."""
    entries=[]
    for slot in pack['node_embs_mask'].nonzero().flatten().tolist():
        node_id=pack['node_ids'][slot]
        if node_id<0: raise RuntimeError('Active action lacks historical GNN node ID')
        attrs=graph.nodes[str(node_id)]
        action=str(attrs.get('action_id',''))
        if not action or attrs.get('action_resolution')!='source_location':
            raise ValueError('Training graph has an unresolved action scope')
        if int(attrs.get('type',-1))==104:kind='array'
        elif yn(attrs.get('is_loop',False)):kind='loop'
        else:raise ValueError('Unknown compiler action node kind')
        entries.append(dict(action=action,absolute_memory_slot=slot,
                            action_type=kind,compiler_node_id=node_id))
    if not entries:raise ValueError('No active actions in historical kernel')
    return entries


def prepare_training_cohort(args,runtime,gemv_manifest):
    from experiments.historical_mlir_gmt_gemv.historical import export_graph
    graphdir=args.training_dir.expanduser()
    if not graphdir.is_dir():raise FileNotFoundError(f'Historical training graph dir not found: {graphdir}')
    split_info=gemv_manifest['authenticated_training_files'].get('experiment_split.json')
    if not split_info:raise RuntimeError('No authenticated training split provenance')
    split_file=Path(split_info['path'])
    if sha(split_file)!=split_info['sha256']:
        raise RuntimeError('Training split has changed from historical GNN provenance')
    allowlist=find_train_names(json.loads(split_file.read_text()))
    if not allowlist:
        raise RuntimeError('Cannot parse TRAIN kernels from authenticated split; refusing arbitrary graph training')
    allowlist={str(v).lower().replace('-','_') for v in allowlist}
    if 'gemv' in allowlist:raise RuntimeError('GEMV in historical training allowlist; refusing evaluation leakage')
    candidates=sorted(graphdir.rglob('*.gexf'))
    if not candidates:raise FileNotFoundError(f'No *.gexf training candidates in {graphdir}')
    cohort=args.out/'cohort';cohort.mkdir(parents=True,exist_ok=True)
    results={};skipped={}
    generator=gemv_manifest['graph_provenance']['generator_sha256']
    for gexf in candidates:
        stem = gexf.stem
        name = (stem[:-5] if stem.endswith('_mlir') else stem).lower().replace('-', '_')
        if name not in allowlist:continue
        if (cohort/(name+'.pt')).exists() and not args.overwrite:
            raise FileExistsError(f'Previously exported cohort graph: {name}. Use --overwrite.')
        try:
            graph,pack=export_graph(gexf,name,runtime)
            provenance=pack['static']['graph_provenance']
            if provenance.get('generator_sha256')!=generator:
                raise ValueError('Compiler graph generator identity mismatch')
            mapping=cohort_action_mapping(graph,pack)
            evidence,meta=make_evidence(graph,pack,mapping,gexf)
            torch.save(evidence,cohort/(name+'.pt'))
            results[name]=dict(graph_sha256=sha(gexf),
                               actions=len(mapping),evidence=len(meta['records']),
                               evidence_sha256=sha(cohort/(name+'.pt')),
                               original_gnn_sha256=gemv_manifest['checkpoint_sha256'])
        except Exception as exc:
            skipped[name]=type(exc).__name__+': '+str(exc)
    if len(results)<5:
        raise RuntimeError(f'Only {len(results)} authenticated train kernels exported; need >=5 for held-out structural validation. Issues: {list(skipped.items())[:6]}')
    write(args.out/'cohort_manifest.json',dict(schema='historical-role-gmt-cohort-v1',
          split_sha256=sha(split_file),training_allowlist_size=len(allowlist),
          exported_train_kernels=results,skipped=skipped,
          caution='Only historical GNN-training kernels; GEMV never trained on GMT'))
    print(f'Prepared {len(results)} training kernels; {len(skipped)} incompatible graphs skipped')

def prepare(args):
    # Verify exact original frozen checkpoint, training provenance and MLIR graph.
    from experiments.historical_mlir_gmt_gemv.historical import load,export_graph
    from experiments.historical_mlir_gmt_gemv.historical import HERE,SNAPSHOT
    runtime=load()
    graphpath=SNAPSHOT/'paper_examples/gemv/gemv_mlir.gexf'
    mapping=json.loads((HERE/'results/action_mapping.json').read_text())
    graph,pack=export_graph(graphpath,'gemv',runtime)
    if len(graph)!=63 or graph.number_of_edges()!=231:
        raise RuntimeError('Wrong historical MLIR graph size')
    provenance=pack['static']['graph_provenance']
    manifest=json.loads((HERE/'results/manifest.json').read_text())
    for key,value in [('gnn_checkpoint_sha256',manifest['checkpoint_sha256']),
                      ('source_gexf_sha256',manifest['graph_sha256'])]:
        assert (pack.get(key) if key in pack else sha(graphpath))==value,(key,value)
    if provenance!=manifest['graph_provenance']:
        raise RuntimeError('Historical graph provenance changed')
    data,meta=make_evidence(graph,pack,mapping,graphpath,gemv_strict=True)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.save(data,out/'evidence.pt')
    write(out/'evidence_meta.json',meta)
    write(out/'prepare_receipt.json',dict(schema='historical-gmt-mlir-slots-v1',
      graph_sha256=sha(graphpath),historical_gnn_sha256=manifest['checkpoint_sha256'],
      historical_source_revision=manifest['source_revision'],
      original_action_cosine=metrics(pack['node_embs'][:5])['pairwise_cosine'],
      action_slots=5,original_memory_slots=64,groups=GROUPS,
      evidence_record_count=len(meta['records']),counts=meta['counts'],
      evidence_sha256=sha(out/'evidence.pt'),evidence_metadata_sha256=sha(out/'evidence_meta.json'),
      warning='GEMV-only diagnostic, not cross-kernel generalization and not LLM alignment'))
    print('Prepared authentic historical frozen GNN memory and',len(meta['records']),'typed evidence records.')
    if args.training_dir is not None:
        prepare_training_cohort(args,runtime,manifest)


def metrics(v):
    x=F.normalize(v.float(),dim=-1)
    cos=(x@x.T).detach().cpu().tolist()
    cv=v.double()-v.double().mean(0)
    singular=torch.linalg.svdvals(cv)
    p=(singular/singular.sum().clamp_min(1.e-12))
    return dict(pairwise_cosine=cos,
      centered_singular_values=singular.detach().cpu().tolist(),
      effective_rank=float(torch.exp(-(p*p.clamp_min(1.e-12).log()).sum())),
      mean_cosine_off_diagonal=float((x@x.T)[~torch.eye(len(x),dtype=torch.bool,device=x.device)].mean()))


def train(args):
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if (args.out/'trained.pt').exists() and not args.overwrite:
        raise FileExistsError('Refusing overwrite: --overwrite to explicitly replace experimental result')
    gemv_data=torch.load(args.out/'evidence.pt',map_location='cpu',weights_only=True)
    cohort=sorted((args.out/'cohort').glob('*.pt'))
    if not cohort and not args.gemv_only_pilot:
        raise RuntimeError('Training cohort absent. Run prepare --training-dir PATH, or use --gemv-only-pilot for non-transferable technical diagnostic.')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    data=device_data(gemv_data,device)
    training=[device_data(torch.load(p,map_location='cpu',weights_only=True),device) for p in cohort] if cohort else [data]
    names=[p.stem for p in cohort] if cohort else ['gemv_pilot_only']
    rng=np.random.default_rng(args.seed)
    order=rng.permutation(len(training)).tolist()
    # Five or more kernels: reserve two as internal GMT validation. Never tune on GEMV.
    dev_count=max(1,round(len(order)*.1)) if len(order)>=5 else 0
    dev=[training[i] for i in order[:dev_count]]
    train_data=[training[i] for i in order[dev_count:]]
    training_names=[names[i] for i in order[dev_count:]]
    dev_names=[names[i] for i in order[:dev_count]]
    model=HistoricalRoleGMT().to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    history=[];model.train()
    best_val=float('inf');best_state=None;best_step=None
    for step in range(1,args.steps+1):
        opt.zero_grad(set_to_none=True)
        # Rotate across authenticated training kernels; no GEMV in transferable training.
        sample=train_data[(step-1)%len(train_data)]
        loss,parts=loss_fn(model,sample,drop=.10)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss at step {step}')
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step==1 or step%max(1,args.steps//20)==0 or step==args.steps:
            val=None
            if dev:
                model.eval()
                with torch.no_grad():
                    val=float(np.mean([loss_fn(model,x,drop=0)[1]['reconstruction'] for x in dev]))
                if val<best_val:
                    best_val=val;best_step=step
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.train()
            history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad,dev_reconstruction_mse=val,**parts))
            print(f'{step:4d}/{args.steps}: loss={float(loss.detach()):.5f} dev={val} grad={grad:.5f}',flush=True)
    # Export validation-selected model if a cohort exists; otherwise final pilot state.
    if best_state is not None:model.load_state_dict(best_state,strict=True)
    displacement={k:float((v.detach().cpu()-initial[k]).norm()) for k,v in model.state_dict().items()}
    if not any(z>1e-6 for z in displacement.values()):raise RuntimeError('No trained parameter update')
    payload=dict(schema='historical-role-gmt-v1',model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},steps=args.steps,seed=args.seed,
                 training_kernel_names=training_names,dev_kernel_names=dev_names,best_dev_step=best_step,
                 evidence_sha256=sha(args.out/'evidence.pt'),history=history,displacement=displacement,
                 train_objective='compiler-signature reconstruction + compiler-signature-based positives')
    torch.save(payload,args.out/'trained.pt')
    write(args.out/'training_receipt.json',dict(checkpoint_sha256=sha(args.out/'trained.pt'),
      steps=args.steps,seed=args.seed,device=device,loss_history=history,
      training_kernels=training_names,internal_dev_kernels=dev_names,best_dev_step=best_step,
      parameter_displacement=displacement,classification='GEMV-only structural warm-start; no LLM/QoR supervision'))


def evaluate(args):
    out=args.out
    saved=torch.load(out/'trained.pt',map_location='cpu',weights_only=False)
    if saved.get('steps',0)<=0 or not saved.get('displacement'):
        raise RuntimeError('Checkpoint lacks optimizer update receipt')
    if saved['evidence_sha256']!=sha(out/'evidence.pt'):
        raise RuntimeError('Evidence provenance checksum mismatch')
    data=torch.load(out/'evidence.pt',map_location='cpu',weights_only=True)
    meta=json.loads((out/'evidence_meta.json').read_text())
    model=HistoricalRoleGMT()
    model.load_state_dict(saved['model'],strict=True);model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(saved['seed'])
        initial_model=HistoricalRoleGMT().eval()
    with torch.no_grad():
        summary,active,pred,attention=model(data,return_attention=True)
        init_summary,init_mask,init_pred,init_attn=initial_model(data,return_attention=True)
    if not torch.isfinite(summary).all():raise ValueError('Nonfinite trained summaries')
    if not torch.all(summary[~active]==0):raise ValueError('Masked summaries must be exactly zero')
    # First 64 positions must ALWAYS be the original historical action embeddings.
    full=torch.zeros(64+64*3,128)
    full[:64]=data['anchors']
    valid=torch.zeros(64+64*3,dtype=torch.bool)
    valid[:64]=data['active']
    route=torch.zeros(64+64*3,64+64*3,dtype=torch.bool)
    # Retain the HISTORICAL GNN builder's compiler-derived inter-action routing.
    # A diagonal-only fallback would silently discard array-loop and dependence facts.
    route[:64,:64]=data['base_relation'] & data['active'][:,None] & data['active'][None,:]
    slot_records=[]
    for a in range(5):
        for g in range(3):
            slot=64+3*a+g
            if active[a,g]:
                full[slot]=summary[a,g];valid[slot]=True;route[a,slot]=True
            slot_records.append(dict(action=f'L{a+1}',absolute_memory_slot=slot,
                    group=GROUPS[g],active=bool(active[a,g]),records=int(data['route'][a,g].sum())))
    route &=valid[:,None] & valid[None,:]
    assert torch.equal(full[:64],data['anchors'])
    rows=[];sensitivity=[]
    for a in range(5):
        for g in range(3):
            ids=data['route'][a,g].nonzero().flatten()
            if not len(ids):continue
            w=attention[a,g].index_select(-1,ids)
            # each of four heads normalized over its compiler-authorized subset
            w=w / w.sum(-1,keepdim=True).clamp_min(1.e-12)
            top=int(ids[w.mean(0).argmax()])
            ent=-(w*w.clamp_min(1e-30).log()).sum(-1).mean()
            norm_ent=float(ent/math.log(len(ids))) if len(ids)>1 else 0.
            rows.append(dict(action=f'L{a+1}',group=GROUPS[g],count=len(ids),
                normalized_entropy=norm_ent,top1_attention=float(w.mean(0).max()),
                top_evidence=meta['records'][top]['desc']))
            # Causal encoder-stage sensitivity: remove the top-supported evidence.
            modified={k:v.clone() for k,v in data.items()}
            modified['route'][a,g,top]=False
            with torch.no_grad():changed,_,_,_=model(modified)
            delta=float((changed[a,g]-summary[a,g]).norm())
            sensitivity.append(dict(action=f'L{a+1}',group=GROUPS[g],removed_evidence=meta['records'][top]['desc'],
                summary_l2_delta=delta,relative_delta=delta/(float(summary[a,g].norm())+1e-10)))
    comparisons={GROUPS[g]:metrics(summary[active[:,g],g]) if int(active[:,g].sum())>1 else None for g in range(3)}
    init_comparisons={GROUPS[g]:metrics(init_summary[init_mask[:,g],g]) if int(init_mask[:,g].sum())>1 else None for g in range(3)}
    within_action={f'L{a+1}':metrics(summary[a,active[a]])['pairwise_cosine']
                   if int(active[a].sum())>1 else None for a in range(5)}
    reconstruction=float(F.mse_loss(pred[active],data['signature'][active]))
    initial_reconstruction=float(F.mse_loss(init_pred[init_mask],data['signature'][init_mask]))
    # Sparse attention audit: weights are tied to authenticated record sidecars.
    with (out/'memory_attention.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['action','group','head','evidence_id','source','destination','type','flow','role','certainty','attention'])
        for a in range(5):
            for g in range(3):
                for idx in data['route'][a,g].nonzero().flatten().tolist():
                    record=meta['records'][idx];identity=record['identity']
                    for head in range(4):
                        writer.writerow([f'L{a+1}',GROUPS[g],head,record['desc'],record.get('src',''),record.get('dst',''),record['kind'],
                                         identity.get('flow',''),identity.get('role',''),identity.get('certainty',''),
                                         float(attention[a,g,head,idx])])
    np.savez_compressed(out/'historical_gmt_slots.npz',
      original_action_embeddings=data['anchors'][:5].numpy(),
      summary_embeddings=summary.numpy(),summary_mask=active.numpy(),
      full_memory=full.numpy(),full_mask=valid.numpy(),slot_route=route.numpy(),
      compiler_signatures=data['signature'].numpy())
    torch.save(dict(node_embs=full,node_embs_mask=valid,action_relation_mask=route,
        slot_ids=torch.arange(1,len(valid)+1),gnn_dim=128,max_slots=64,memory_capacity=len(valid),
        base_action_slots=64,summary_groups=list(GROUPS),slot_records=slot_records,
        schema='historical-gmt-slots-v1',checkpoint_sha256=sha(out/'trained.pt')),
        out/'historical_gmt_memory.pt')
    report=dict(checkpoint_sha256=sha(out/'trained.pt'),train_steps=saved['steps'],
       evidence_sha256=saved['evidence_sha256'],original=metrics(data['anchors'][:5]),
       rolewise_cosine=comparisons,initial_rolewise_cosine=init_comparisons,
       within_action_summary_cosine=within_action,
       compiler_signature_reconstruction_mse=reconstruction,
       random_initialization_reconstruction_mse=initial_reconstruction,
       active_summary_tokens=int(active.sum()),memory_shape=list(full.shape),
       norm_entropy_and_top_record=rows,single_evidence_deletions=sensitivity,
       duplicate_same_action_slots_warning='Do not infer independence from role labels; check actual cosine/causal contribution',
       limitation='GEMV-only structural pretraining; no Stage-2 LLM alignment, downstream cross-attention or QoR measurement')
    write(out/'evaluation.json',report)
    write(out/'memory_manifest.json',dict(schema='historical-gmt-slots-v1',original_gnn_sha256=json.loads((out/'prepare_receipt.json').read_text())['historical_gnn_sha256'],
         original_action_slots=64,summary_slots_per_action=3,full_memory_shape=list(full.shape),
         slot_records=slot_records,training_steps=saved['steps'],
         trained_checkpoint_sha256=sha(out/'trained.pt'),
         memory_pt_sha256=sha(out/'historical_gmt_memory.pt'),
         memory_npz_sha256=sha(out/'historical_gmt_slots.npz'),
         memory_attention_csv_sha256=sha(out/'memory_attention.csv'),
         source_evidence_sha256=sha(out/'evidence.pt'),rolewise_attention=rows,
         warning='not validated for LLM fusion; memory must be trained/checked jointly with Stage-2'))
    try:
        plot_evaluation(out,data['anchors'][:5],summary,active)
    except ImportError:
        print('Matplotlib not installed: numerical evaluation remains complete')
    print(json.dumps({k:v for k,v in report.items() if k in ('train_steps','active_summary_tokens','compiler_signature_reconstruction_mse','random_initialization_reconstruction_mse','memory_shape')},indent=2))
    print('Artifacts:',out/'historical_gmt_memory.pt',out/'historical_gmt_slots.npz',out/'evaluation.json')



def plot_evaluation(out,original,summary,mask):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tensors=[original]+[summary[:,g] for g in range(3)]
    masks=[torch.ones(5,dtype=torch.bool)]+[mask[:,g] for g in range(3)]
    labels=['Original historical action anchors']+['Historical trained: '+x for x in GROUPS]
    fig,axes=plt.subplots(1,4,figsize=(18,4.5),layout='constrained')
    for ax,x,valid,title in zip(axes,tensors,masks,labels):
        norm=F.normalize(x,dim=-1)
        cos=(norm@norm.T).detach().cpu().numpy()
        keep=valid.detach().cpu().numpy()
        cos[~keep,:]=np.nan;cos[:,~keep]=np.nan
        ax.imshow(cos,cmap='coolwarm',vmin=-1,vmax=1)
        ax.set_xticks(range(5),[f'L{i+1}' for i in range(5)])
        ax.set_yticks(range(5),[f'L{i+1}' for i in range(5)])
        ax.set_title(title,fontsize=10)
        for i in range(5):
            for j in range(5):
                if np.isfinite(cos[i,j]):ax.text(j,i,f'{cos[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.suptitle('Historical GEMV: original GNN vs role-conditioned GMT memory (not QoR evidence)',fontsize=12)
    fig.savefig(out/'historical_role_cosines.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def selftest():
    torch.manual_seed(123)
    model=HistoricalRoleGMT()
    n=8;r=12
    data=dict(nodes=torch.randn(n,DIM),anchors=torch.randn(64,DIM),active=torch.zeros(64,dtype=torch.bool),
      src=torch.randint(0,n,(r,)),dst=torch.randint(0,n,(r,)),edge=torch.randint(0,2,(r,),dtype=torch.bool),
      group=torch.arange(r)%3,facts=torch.randn(r,NUM_FEATURES),
      role_idx=torch.zeros(r,dtype=torch.long),flow_idx=torch.zeros(r,dtype=torch.long),
      certainty_idx=torch.zeros(r,dtype=torch.long),route=torch.zeros(5,3,r,dtype=torch.bool),
      signature=torch.rand(5,3,NUM_FEATURES+1))
    data['active'][:5]=True
    data['action_slots']=torch.arange(5)
    for i in range(5):
        for g in range(3): data['route'][i,g,(i*3+g)%r]=True
    s,mask,p,attn=model(data,return_attention=True)
    assert s.shape==(5,3,128) and mask.all() and torch.isfinite(s).all()
    assert attn.shape==(5,3,4,r)
    assert torch.allclose(attn.sum(-1),torch.ones(5,3,4))
    loss,_=loss_fn(model,data)
    loss.backward()
    assert model.queries.grad is not None and torch.isfinite(model.queries.grad).all()
    data['route'][0,0]=False
    s2,m2,_,_=model(data)
    assert not m2[0,0] and (s2[0,0]==0).all()
    print('PASS: shapes, routing, masked zero, normalized attention, finite loss and gradients')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['selftest','prepare','train','evaluate','all'])
    parser.add_argument('--out',type=Path,default=Path('artifacts/historical_gmt_slots_v1'))
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--lr',type=float,default=5e-4)
    parser.add_argument('--device',default='auto',help='auto|cpu|cuda')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--training-dir',type=Path,help='Directory containing historical training-kernel *_mlir.gexf files')
    parser.add_argument('--gemv-only-pilot',action='store_true',help='Explicitly allow GEMV-only diagnostic; not transferable')
    args=parser.parse_args()
    if args.steps<=0:parser.error('--steps must be positive')
    if args.command=='selftest':selftest()
    else:
        if args.command in ('prepare','all'):prepare(args)
        if args.command in ('train','all'):train(args)
        if args.command in ('evaluate','all'):evaluate(args)


if __name__=='__main__':main()
