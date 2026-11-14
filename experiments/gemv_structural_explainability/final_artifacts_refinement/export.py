"""Strict FINAL_ARTIFACTS encoder export and authenticated native-IR bridge.

The V1 checkpoint is loaded only into its saved Net implementation/schema. RC3
records are a separate evidence adapter; no V1 weights enter PaperRC3Net.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tarfile

import networkx as nx
import torch
from torch_geometric.data import Data,Batch

ROOT=Path(__file__).resolve().parents[3]
HOST_ROOT=Path('/home/elvouvali/MailoHLS')
RUN=HOST_ROOT/'Checkpoints/gnn_final_absolute_direct_rank_s123/run1'
SHA='34b2b2c6405aae715001d6763e735e2163668a724008fbf13867aa575df61623'
BRANCH='bf82c648273a5cd5eb30363059a3809e7b3979db'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def load_encoder(out):
    final=json.loads((ROOT/'FINAL_ARTIFACTS.json').read_text())
    checkpoint=HOST_ROOT/final['gnn']['checkpoint']
    contract=json.loads((RUN/'gnn_checkpoint_contract.json').read_text())
    sidecar=json.loads((RUN/'val_structural_rank_model_state_dict.pth.json').read_text())
    if sha(checkpoint)!=SHA or sidecar['checkpoint_sha256']!=SHA:
        raise ValueError('Wrong FINAL_ARTIFACTS GNN')
    if sha(RUN/'gnn_checkpoint_contract.json')!=sidecar['contract_sha256']:
        raise ValueError('GNN contract changed')
    runtime=out/'runtime/GNN_branch';runtime.mkdir(parents=True,exist_ok=True)
    for name,rec in contract['training_artifacts']['files'].items():
        source=RUN/'provenance'/name
        if sha(source)!=rec['sha256']:raise ValueError('Frozen training artifact changed: '+name)
        shutil.copyfile(source,runtime/name)
    sys.path.insert(0,str(runtime))
    original=sys.argv
    sys.argv=['final_artifacts_export','--dataset','mlir','--num_layers','4','--D','64',
              '--target','perf','area','--experiment_name','final_artifacts_gemv_export']
    try:
        config=importlib.import_module('config')
        for name,value in contract['resolved_flags'].items():setattr(config.FLAGS,name,value)
        for name,value in contract['model_init_flags'].items():setattr(config.FLAGS,name,value)
        config.FLAGS.device='cpu';config.FLAGS.experiment_name='final_artifacts_gemv_export'
        model_code=importlib.import_module('model')
        data_code=importlib.import_module('mlir_data')
        model=model_code.Net(contract['model_init']['in_channels'],edge_dim=contract['model_init']['edge_dim'],
            target_stats=contract['target_stats'],resource_stats=contract['resource_stats'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
        model.load_state_dict(state,strict=True)
        model.eval().requires_grad_(False)
    finally:sys.argv=original
    with (runtime/'encoders.pkl').open('rb') as stream:encoders=pickle.load(stream)
    # Reuse the maintained source-action mapping/export helpers. Parsing only
    # these arguments has no export or training side effect.
    sys.argv=['memory_builder','--pt_path',str(out),'--ckpt',str(checkpoint),
        '--checkpoint_contract',str(RUN/'gnn_checkpoint_contract.json'),
        '--checkpoint_sidecar',str(RUN/'val_structural_rank_model_state_dict.pth.json'),
        '--feature_schema',str(runtime/'feature_schema.json'),'--out',str(out),
        '--embedding_mode','static_pre_npt']
    try:builder=importlib.import_module('GNN_branch.build_structural_memory')
    finally:sys.argv=original
    return model,data_code,encoders,builder,contract


def official_memory(first,jk,nodes,slots,norm):
    action=first.new_zeros(64,128)
    local=first[nodes]-first[nodes].mean(0,keepdim=True)
    action[slots]=torch.cat((jk[nodes]/norm['jkn_rms'],
                            local/norm['centered_conv1_rms']),-1)/math.sqrt(2)
    return action


def bridge(rc3,legacy,node_h,action,node_ids):
    if rc3.graph['provenance']['source_sha256']!=json.loads(legacy.graph['name'].split(':',1)[1])['source_sha256']:
        raise ValueError('Legacy/RC3 source identity differs')
    lookup={}
    for n,a in legacy.nodes(data=True):
        uid=a.get('op_uid') or a.get('ssa_id')
        if uid:
            if uid in lookup:raise ValueError('Ambiguous native compiler identity')
            lookup[uid]=n
    states={};proofs={}
    for n,a in rc3.nodes(data=True):
        uid=a.get('op_uid') or a.get('ssa_id')
        if not uid or uid not in lookup:continue
        old=legacy.nodes[lookup[uid]]
        fields=('text','full_text','source_line','source_column') if a['category']=='operation' else ('value_type',)
        if any(a.get(k)!=old.get(k) for k in fields):continue
        states[n]=node_h[int(lookup[uid])]
        proofs[n]=dict(rule='exact_native_identity_and_payload',native_uid=uid,
                       legacy_nodes=[lookup[uid]],compared_fields=list(fields))
    # Memory-object tokens derive from the compiler-established canonical root
    # value. Control tokens derive from their authenticated owner operation.
    # These are explicitly derived representations, not identity equivalences.
    for _ in range(3):
        for n,a in rc3.nodes(data=True):
            if n in states:continue
            basis=None;rule=None
            if a['category']=='memory_object':basis=a.get('root_value');rule='canonical_root_value_basis'
            elif a['category']=='control_region':
                basis=a.get('owner') or a.get('parent_region');rule='compiler_owner_context_basis'
            elif a['category']=='action_point':
                k=int(a['action_id'][1:])-1
                if node_ids[k]>=0:
                    states[n]=action[k];proofs[n]=dict(rule='authenticated_source_action_anchor',legacy_nodes=[str(node_ids[k])])
                continue
            if basis in states:
                states[n]=states[basis]
                proofs[n]=dict(rule=rule,canonical_basis=basis,legacy_nodes=proofs[basis]['legacy_nodes'])
    return states,proofs


def export(args):
    from experiments.gemv_structural_explainability.audit import authenticate,catalog
    from experiments.rc3_fusion_foundation.rc3_action_relations import build_action_relations
    from GNN_branch.mlir_graph_v2 import read_graph
    from experiments.gemv_structural_explainability.constants import TYPES,STATUS,KNOWN
    from experiments.gemv_structural_explainability.final_artifacts_refinement.facts import encode_facts,FACT_NAMES
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'manifest.json').exists():raise FileExistsError('Fresh export directory required')
    torch.set_num_threads(4)
    model,data_code,encoders,builder,contract=load_encoder(args.out)
    final=json.loads((ROOT/'FINAL_ARTIFACTS.json').read_text())
    bank=HOST_ROOT/final['gnn']['memory']
    norm=json.loads((bank/'normalization_stats.json').read_text())
    bank_manifest=json.loads((bank/'memory_manifest.json').read_text())
    if sha(bank/'normalization_stats.json')!=bank_manifest['multiscale']['normalization_stats_sha256']:
        raise ValueError('Frozen normalization differs')
    split=json.loads((RUN/'provenance/experiment_split.json').read_text())
    selected=split['train_kernels']+split['val_kernels'] if args.cohort else []
    graph_dir=args.out/'graphs';graph_dir.mkdir(exist_ok=True)
    packs=args.out/'packs';packs.mkdir(exist_ok=True)
    manifest=dict(schema='salt-final-artifacts-rc3-evidence-bridge-v1',checkpoint_sha256=SHA,
        checkpoint_path=str(HOST_ROOT/final['gnn']['checkpoint']),checkpoint_contract_sha256=sha(RUN/'gnn_checkpoint_contract.json'),
        final_artifacts_sha256=sha(ROOT/'FINAL_ARTIFACTS.json'),mlir_target_commit=BRANCH,
        normalization=norm,normalization_sha256=sha(bank/'normalization_stats.json'),
        feature_schema_sha256=contract['feature_schema']['sha256'],split=split,packs={},
        embedding_layout='[JK/jkn_rms | (conv1-action_mean)/centered_conv1_rms]/sqrt(2)',
        gnn_retrained=False,gemv_fitted=False,locked_test_deserialized=False,fact_names=FACT_NAMES)
    graph_inputs={'gemv':ROOT/'paper_examples/gemv/gemv_mlir.gexf'}
    for kernel in selected:
        path=graph_dir/(kernel+'.gexf')
        path.write_bytes(subprocess.check_output(['git','show',BRANCH+':GNN_branch/MLIR_graphs/'+kernel+'.gexf'],cwd=ROOT))
        graph_inputs[kernel]=path
    rc3_inputs={}
    if selected:
        release=json.loads((ROOT/'docs/graph-release-v22-rc3.json').read_text())
        archive=ROOT/release['archive']
        if not archive.is_file():archive=HOST_ROOT/release['archive']
        if sha(archive)!=release['archive_sha256']:raise ValueError('RC3 archive authentication failed')
        canonical=args.out/'rc3_graphs';canonical.mkdir(exist_ok=True)
        with tarfile.open(archive) as stream:
            members=[m for m in stream.getmembers() if m.name.startswith('canonical/') and m.name.endswith('.json')]
            for kernel in selected:
                legacy_md=json.loads(nx.read_gexf(graph_inputs[kernel]).graph['name'].split(':',1)[1])
                matches=[]
                for member in members:
                    if Path(member.name).name.startswith(kernel+'-'):
                        raw=stream.extractfile(member).read()
                        if hashlib.sha256(raw).hexdigest()!=release['files_sha256'][member.name]:raise ValueError('RC3 member hash changed')
                        obj=json.loads(raw)
                        if obj['metadata']['provenance']['source_sha256']==legacy_md['source_sha256']:matches.append(raw)
                if len(matches)!=1:raise ValueError('No unique source-identical RC3 graph for '+kernel)
                target=canonical/(kernel+'.json');target.write_bytes(matches[0]);rc3_inputs[kernel]=target
        cache_dir=HOST_ROOT/'GNN_branch/MLIR_dataset/all_kernels/graphs'
        digest=hashlib.sha256()
        paths=sorted(cache_dir.glob('*.pt'),key=lambda p:p.name)
        for path in paths:digest.update(path.name.encode()+b'\0'+sha(path).encode()+b'\n')
        tree=contract['training_artifacts']['static_graph_tensor_tree']
        if len(paths)!=tree['count'] or digest.hexdigest()!=tree['sha256']:raise ValueError('Frozen V1 tensor tree differs')
        manifest['static_tensor_tree_sha256']=digest.hexdigest()
    for kernel,path in graph_inputs.items():
        legacy=nx.read_gexf(path)
        if kernel=='gemv':payload=data_code.encode_static_graph(legacy,kernel,kernel,encoders)
        else:
            _,payload=builder._load_static_mlir_graph(cache_dir/(kernel+'.pt'))
            builder._load_and_verify_matching_gexf(path,payload,payload['x'].shape[0])
        if payload['x'].shape[1]!=403 or payload['edge_attr'].shape[1]!=82:
            raise ValueError('Frozen feature dimensions differ')
        data=Data(**{k:v.float() if v.is_floating_point() else v for k,v in payload.items() if torch.is_tensor(v)})
        data=builder._build_llm_slots_from_gexf(data,legacy,max_slots=64)
        batch=builder._disable_pragma_conditioning(Batch.from_data_list([data]))
        with torch.no_grad():layers=model.forward_static_node_embed_layers(batch)
        first,jk=layers['conv_1'],layers['jkn']
        action_nodes=data.X_llm_scopeids.nonzero().flatten()
        slots=data.X_llm_labelid[action_nodes].long()-1
        active=torch.zeros(64,dtype=torch.bool);active[slots]=True
        node_ids=[-1]*64
        for k,n in zip(slots.tolist(),action_nodes.tolist()):node_ids[k]=n
        action=official_memory(first,jk,action_nodes,slots,norm)
        # Reproduce the production export on in-cohort kernels. GEMV is new,
        # so it uses the same frozen transform without fitting new statistics.
        parity=None
        saved=bank/(kernel+'.memory.pt')
        if saved.is_file():
            reference=torch.load(saved,map_location='cpu',weights_only=False)
            parity=float((reference['node_embs']-action).abs().max())
            if parity>2e-5:raise ValueError('FINAL_ARTIFACTS export parity failed: '+kernel+':'+str(parity))
        if kernel=='gemv':rc3,_,_=authenticate()
        else:rc3=read_graph(rc3_inputs[kernel])
        h=torch.cat((jk/norm['jkn_rms'],(first-first[action_nodes].mean(0))/norm['centered_conv1_rms']),-1)/math.sqrt(2)
        states,proofs=bridge(rc3,legacy,h,action,node_ids)
        records=catalog(rc3,build_action_relations(rc3))
        n=len(records);a=torch.zeros(1,n,128);b=a.clone();facts=torch.zeros(1,n,len(FACT_NAMES))
        valid=torch.zeros(1,n,dtype=torch.bool);routes=torch.zeros(1,64,n,dtype=torch.bool)
        is_edge=torch.zeros(1,n,dtype=torch.bool)
        for i,r in enumerate(records):
            nodes=[r['node']] if r['kind']=='node' else [r['source'],r['target']]
            mapped=all(node in states for node in nodes)
            valid[0,i]=mapped
            if mapped:
                a[0,i]=states[nodes[0]];b[0,i]=states[nodes[-1]]
            is_edge[0,i]=r['kind']=='edge'
            facts[0,i]=torch.tensor(encode_facts(r,rc3))
            for label in r['eligible_actions']:routes[0,int(label[1:])-1,i]=mapped
            r['gnn_basis']={node:proofs.get(node,dict(rule='unresolved')) for node in nodes}
        pack=dict(action=action[None],active=active[None],a=a,b=b,facts=facts,
            valid=valid,routes=routes,is_edge=is_edge,records=records,
            types=torch.tensor([[TYPES.index(r['type']) for r in records]]),
            status=torch.tensor([[STATUS.index(r['status']) for r in records]]),
            known=torch.tensor([[[r['known'][k] for k in KNOWN] for r in records]],dtype=torch.float32),
            edge=torch.zeros(1,n,48),edge_dim=48,
            gnn_sha256=SHA,graph_identity=rc3.graph['graph_identity'],
            legacy_graph_sha256=sha(path),basis_proofs=proofs,node_ids=node_ids,
            schema='salt-final-artifacts-action-evidence-v1')
        for i,r in enumerate(records):
            if r['kind']=='edge':
                from experiments.paper_gnn.data import RELATIONS,EVIDENCE
                pack['edge'][0,i,RELATIONS.index(r['type'])]=1.
                if r['status'] in EVIDENCE:pack['edge'][0,i,2*len(RELATIONS)+EVIDENCE.index(r['status'])]=1.
        target=packs/(kernel+'.pt');torch.save(pack,target)
        manifest['packs'][kernel]=dict(file=str(target.relative_to(args.out)),sha256=sha(target),
            legacy_graph_sha256=sha(path),rc3_graph_identity=pack['graph_identity'],
            action_count=int(active.sum()),record_count=n,mapped_records=int(valid.sum()),
            eligible_counts=routes[0,active].sum(-1).tolist(),production_memory_max_abs_delta=parity)
        print(kernel,manifest['packs'][kernel],flush=True)
    dump(args.out/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cohort',action='store_true')
    parser.add_argument('--rc3-graphs',type=Path,default=Path('/home/elvouvali/MailoHLS-rc3-stage2-multikernel-c/artifacts/rc3_stage2_multikernel/v1/memory_c/graphs'))
    args=parser.parse_args();export(args)


if __name__=='__main__':main()
