"""Export-only correction: source actions without compiler nodes stay explicitly unbound.

No changes to the optimizer, encoder tensors, candidates, loss, schedule or weights.
"""
import math
import torch
from experiments.paper_gnn.data import sha,fingerprint
from experiments.paper_gnn.train import atomic_save,write_json

def export(model,data,out,checkpoint,objective,protocol):
    from GNN_branch.action_point_relations import action_point_views
    model.eval(); raw={}; device=next(model.parameters()).device
    with torch.no_grad():
        for kernel,graph in sorted(data.tensors.items()):
            raw[kernel]=model.memory(graph.to(device)).cpu()
            expected={int(label[1:])-1 for label in data.actions[kernel]}
            if not set(graph.action_slots.cpu().tolist())<=expected: raise ValueError('Source action identity mismatch')
            probe=graph.clone(); probe.target_condition=torch.randn(1,5,device=device); probe.candidates=torch.randn(64,8,device=device); probe.measured_qor=torch.randn(6,device=device); probe.resource_budget=torch.rand(4,device=device)
            if not torch.equal(raw[kernel],model.memory(probe).cpu()): raise ValueError('Static memory leakage: '+kernel)
    # Equal kernels when fitting normalization, exclusively training families.
    rms=[]
    for begin in (0,64):
        rms.append(math.sqrt(sum(float(raw[k][data.tensors[k].action_slots.cpu(),begin:begin+64].square().mean())
            for k in data.split['train_kernels'] if data.tensors[k].action_slots.numel()) /
            sum(bool(data.tensors[k].action_slots.numel()) for k in data.split['train_kernels'])))
    if min(rms)<=0 or not all(math.isfinite(x) for x in rms): raise ValueError('Invalid memory normalization')
    stats={'schema':'salt-paper-memory-normalization-v1','layout':['conv_1','jk_max'],
           'rms':rms,'fitted_kernels':data.split['train_kernels'],'split_sha256':data.identity['split_sha256'],
           'rule':'per-component train-kernel-balanced RMS then divide sqrt(2); no target or QoR statistics'}
    bank=out/'memory'; bank.mkdir(exist_ok=True)
    write_json(stats,bank/'normalization.json')
    manifest={'schema':'salt-paper-structural-memory-bank-v1','representation_schema':model.schema,
        'objective':objective,'gnn_checkpoint_sha256':sha(checkpoint),'feature_schema_sha256':fingerprint(data.schema),
        'protocol_sha256':fingerprint(protocol),'normalization_sha256':sha(bank/'normalization.json'),
        'layout':['conv_1:0:64','jk_max:64:128'],'gnn_dim':128,'files':{},
        'export_implementation_sha256':sha(__file__),
        'unbound_policy':'isolated source-only action keys encode authenticated source metadata with the same frozen GNN; no compiler binding or relation is invented; self visibility preserves routing for unbound actions',
        'fusion_checkpoint_policy':'fresh objective-specific adapter required; no historical fusion weights accepted'}
    for kernel,memory in raw.items():
        graph=data.tensors[kernel]; active=torch.zeros(64,dtype=torch.bool); active[graph.action_slots.cpu()]=True
        normalized=memory.clone(); normalized[:,:64]/=rms[0]*math.sqrt(2); normalized[:,64:]/=rms[1]*math.sqrt(2)
        relations=action_point_views(data.graphs[kernel],max_slots=64)
        # Full relation metadata is preserved even for relations without a legacy bit.
        relation_mask=torch.tensor(relations['connectivity'],dtype=torch.bool) if 'connectivity' in relations else None
        if relation_mask is None:
            relation_mask=torch.eye(64,dtype=torch.bool)
        relation_mask &= active[:,None]&active[None,:]
        # Identity visibility is valid for every registered source action.
        # Unbound source-only keys receive no invented compiler neighbors.
        relation_mask |= torch.eye(64,dtype=torch.bool)&active[:,None]&active[None,:]
        bound=torch.zeros(64,dtype=torch.bool)
        for node,a in data.graphs[kernel].nodes(data=True):
            if a['category']=='action_point' and any(e['relation_type']=='targets' for _,_,e in data.graphs[kernel].out_edges(node,data=True)):
                bound[int(a['action_id'][1:])-1]=True
        source_active=torch.zeros(64,dtype=torch.bool)
        for label in data.actions[kernel]:source_active[int(label[1:])-1]=True
        compiler_node_mask=torch.zeros(64,dtype=torch.bool)
        for _,attrs in data.graphs[kernel].nodes(data=True):
            if attrs['category']=='action_point':compiler_node_mask[int(attrs['action_id'][1:])-1]=True
        pack={'compiler_node_mask':compiler_node_mask,'source_only_unbound_mask':source_active & ~compiler_node_mask,
            'source_action_mask':source_active,'compiler_memory_mask':active,'source_action_ids':sorted(data.actions[kernel],key=lambda x:int(x[1:])),
            'action_bound_mask':bound,'action_unbound_mask':source_active & ~bound,'schema':'salt-paper-action-memory-v1','representation_schema':model.schema,
            'node_embs':normalized,'node_embs_mask':active,'gnn_dim':128,
            'max_slots':64,'disable_pragma_injection':True,
            'embedding_mode':'static_pre_npt::conv1_jkmax_rms_v1',
            'action_relation_schema':'salt-paper-rc3-action-relations-v1',
            'labels':torch.tensor([i+1 if active[i] else -1 for i in range(64)]),
            'slot_ids':torch.arange(1,65),'action_relation_mask':relation_mask,
            'compiler_relations':relations,'graph_sha256':data.graph_hashes[kernel],
            'feature_schema_sha256':fingerprint(data.schema),'gnn_checkpoint_sha256':sha(checkpoint),
            'normalization_sha256':sha(bank/'normalization.json'),'objective':objective}
        path=bank/(kernel+'.memory.pt'); atomic_save(pack,path); manifest['files'][path.name]=sha(path)
    write_json(manifest,bank/'memory_manifest.json')
    return manifest
