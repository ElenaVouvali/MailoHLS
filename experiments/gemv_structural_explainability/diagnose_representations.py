"""CPU-only frozen RC3 diagnostics. No encoder, source, or label updates.

Run as a module from final-architecture; see representation_diagnostics.md.
Every output directory must be new. Graph metadata masking is an observation
intervention, never a recompiled or authenticated alternative source program.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import tarfile

import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F

from experiments.gemv_structural_explainability.audit import (
    ROOT, GNN, GNN_SHA, authenticate, load_encoder, sha, write_json)
from experiments.paper_gnn.data import tensorize, NUMERIC, CATEGORIES, RELATIONS, EVIDENCE
from GNN_branch.mlir_graph_v2 import read_graph
from LLM_branch.common.mailohls_contract import family_id_from_kernel_name

STAGES = ('input', 'conv1', 'conv2', 'conv3', 'conv4', 'jk', 'memory', 'rms_memory')


@torch.no_grad()
def extract(model, tensor, rms):
    """Observe the maintained encode() call using removable forward hooks."""
    layers = []
    handles = [conv.register_forward_hook(
        lambda module, args, output: layers.append(F.elu(output).clone()))
        for conv in [model.conv_first, *model.conv_layers]]
    try:
        first, jk = model.encode(tensor)
    finally:
        for handle in handles:
            handle.remove()
    assert len(layers) == 4 and torch.equal(first, layers[0])
    assert torch.equal(jk, torch.stack(layers).max(0).values)
    memory = torch.cat((first, jk), -1)
    # Match train.export's scalar arithmetic order as well as its formula.
    scaled = memory.clone()
    scaled[:,:64] /= rms[0]*math.sqrt(2)
    scaled[:,64:] /= rms[1]*math.sqrt(2)
    return dict(zip(STAGES, [tensor.x, *layers, jk, memory, scaled]))


def metrics(x, kinds):
    x = x.double()
    norms = x.norm(dim=1)
    cosine = F.normalize(x, dim=1) @ F.normalize(x, dim=1).T
    distance = torch.cdist(x, x)
    centered = x-x.mean(0)
    energy = torch.linalg.svdvals(centered).square()
    total = float(energy.sum())
    probabilities = energy[energy > max(total*1e-12, 1e-24)] / max(total, 1e-24)
    groups = {}
    for name, predicate in (
        ('within_array', lambda a,b: a==b=='array'),
        ('within_loop', lambda a,b: a==b=='loop'),
        ('within_category', lambda a,b: a==b),
        ('between_category', lambda a,b: a!=b)):
        pairs = [(i,j) for i in range(len(kinds)) for j in range(i) if predicate(kinds[i],kinds[j])]
        groups[name] = dict(pairs=len(pairs), mean_distance=float(torch.tensor(
            [distance[i,j] for i,j in pairs]).mean()) if pairs else None,
            mean_cosine=float(torch.tensor([cosine[i,j] for i,j in pairs]).mean()) if pairs else None)
    return dict(norms=norms.tolist(), cosine=cosine.tolist(), euclidean=distance.tolist(), groups=groups,
        centered_total_variance=float(centered.square().sum()/max(len(x)-1,1)),
        centered_energy_fraction=total/max(float(x.square().sum()),1e-24),
        centered_effective_rank=float(torch.exp(-(probabilities*probabilities.log()).sum())) if total>1e-24 else 0.,
        effective_rank_definition='exp entropy of centered squared singular values; relative cutoff 1e-12',
        centered_max_rank=min(len(x)-1,x.shape[1]), centered_singular_values=energy.sqrt().tolist())


def action_info(graph):
    rows = []
    for n,a in graph.nodes(data=True):
        if a['category'] != 'action_point':
            continue
        targets = [v for _,v,_,e in graph.out_edges(n,keys=True,data=True)
                   if e['relation_type']=='targets' and e.get('evidence_status')=='proven']
        target = targets[0] if len(targets)==1 else None
        rows.append(dict(action=a['action_id'],node=n,kind=a['action_kind'],target=target,
                         slot=int(a['action_id'][1:])-1))
    return sorted(rows,key=lambda r:r['slot'])


def numeric_columns(schema, names):
    start = len(CATEGORIES)+len(schema['vocabulary'])
    return [i for name in names for i in (start+2*NUMERIC.index(name),start+2*NUMERIC.index(name)+1)]


def mask_numeric(tensor, schema, names, rows=None):
    result = tensor.clone()
    indices = torch.arange(len(result.x)) if rows is None else torch.tensor(rows,dtype=torch.long)
    columns = numeric_columns(schema,names)
    result.x[indices[:,None],torch.tensor(columns)[None,:]] = 0
    return result


def mask_relations(tensor, names):
    """Withhold both directions together; remaining edge types/statuses unchanged."""
    result = tensor.clone()
    cols = [RELATIONS.index(name)+rev*len(RELATIONS) for name in names for rev in (0,1)]
    keep = ~tensor.edge_attr[:,cols].bool().any(1)
    result.edge_index = tensor.edge_index[:,keep]
    result.edge_attr = tensor.edge_attr[keep]
    return result


def metadata_mask(graph, schema, name):
    """Remove only omitted payloads from an in-memory copy, then re-tensorize."""
    altered = copy.deepcopy(graph)
    node_fields = {
        'memory_shape': ('memory_shape',),
        'source_shape': ('source_array_dimensions','array_binding_evidence'),
    }.get(name,())
    edge_fields = {
        'affine_access': ('access_indices',),
        'access_role': ('access','source_analysis'),
        'recurrence_detail': ('carrying_loop','forwarding_role','state_kind','legacy_evidence'),
        'dependence_detail': ('dependence_kind','dependence_depth','distance_bounds','distance_components',
                              'distance_exact','loop_carried','possible_loop_carried','source_analysis'),
    }.get(name,())
    for _,a in altered.nodes(data=True):
        for key in node_fields:
            a.pop(key,None)
    for _,_,a in altered.edges(data=True):
        if name=='recurrence_detail' and a['relation_type']!='loop_carried': continue
        if name=='dependence_detail' and a['relation_type']!='depends_on': continue
        for key in edge_fields:
            a.pop(key,None)
    return tensorize(altered,schema['vocabulary'])


def feature_coverage(graph, tensor, schema, actions):
    nodes = sorted(graph.nodes)
    undirected = graph.to_undirected()
    records=[]
    for row in actions:
        for role,n in [('action',row['node']),('target',row['target'])]:
            if n is None: continue
            a=graph.nodes[n]; x=tensor.x[nodes.index(n)]
            numbers={name:dict(raw=a.get(name),value=float(x[numeric_columns(schema,[name])[0]]),
                              known=bool(x[numeric_columns(schema,[name])[1]])) for name in NUMERIC}
            records.append(dict(action=row['action'],role=role,node=n,features=x.tolist(),norm=float(x.norm()),
                numeric=numbers,numeric_value_norm=float(x[numeric_columns(schema,NUMERIC)[::2]].norm()),
                knownness_norm=float(x[numeric_columns(schema,NUMERIC)[1::2]].norm()),
                nonzero_columns=x.nonzero().flatten().tolist(),
                omitted_fields=sorted(set(a)-{'category',*NUMERIC}-({'text'} if a['category']=='operation' else set()))))
    paths=[]
    for row in actions:
        for relation in ('memory_access','operand','defines','loop_carried','depends_on','loop_child'):
            edges=[]
            for u,v,k,a in graph.edges(keys=True,data=True):
                if a['relation_type']!=relation: continue
                du=nx.shortest_path_length(undirected,row['node'],u)
                dv=nx.shortest_path_length(undirected,row['node'],v)
                # The endpoints receive each other's messages before reaching the action.
                edges.append(dict(id=str(k),source=u,target=v,status=a.get('evidence_status'),
                                  earliest_action_layer=1+min(du,dv),payload=a))
            paths.append(dict(action=row['action'],relation=relation,edges=edges))
    return dict(node_dim=tensor.x.shape[1],edge_dim=tensor.edge_attr.shape[1],schema=schema,
                action_and_target_features=records,relation_paths=paths,
                node_field_counts=dict(Counter(k for _,a in graph.nodes(data=True) for k in a)),
                edge_field_counts=dict(Counter(k for _,_,a in graph.edges(data=True) for k in a)),
                warning='Paths are earliest possible propagation, not proof of feature recovery. All reverse edges added.')


def ablations(model, graph, tensor, schema, rms, baseline, actions):
    nodes=sorted(graph.nodes); indices=tensor.action_nodes
    experiments=[]
    for name,names in [('memory_rank',['memory_rank']),('loop_depth',['loop_depth']),
                       ('trip_counts',['trip_count','compiler_proven_trip_count','tcad_trip_count']),
                       ('operand_result_use',['operand_count','result_count','use_count'])]:
        experiments.append(('mask_numeric_'+name,mask_numeric(tensor,schema,names),
                            'Withhold value and knownness together; no fabricated known zero'))
        gated=tensor.clone()
        gated.x[:,numeric_columns(schema,names)[::2]]=0
        experiments.append(('gate_value_channel_'+name,gated,
                            'Diagnostic channel gate only; knownness retained, zero is a masked signal, not a new compiler fact'))
    for row in actions:
        if row['target'] is None: continue
        name='memory_rank' if row['kind']=='array' else 'loop_depth'
        experiments.append(('mask_'+row['action']+'_'+name,mask_numeric(tensor,schema,[name],[nodes.index(row['target'])]),
                            'Withhold only this target observation; all source facts unchanged'))
    for name in ('memory_shape','source_shape','affine_access','access_role','recurrence_detail','dependence_detail'):
        experiments.append(('mask_payload_'+name,metadata_mask(graph,schema,name),
                            'Omitted metadata negative control; authenticated original never edited'))
    token=tensor.clone(); start=len(CATEGORIES); end=start+len(schema['vocabulary'])
    for n,a in graph.nodes(data=True):
        if a.get('operation_family')=='arithmetic':
            token.x[nodes.index(n),start:end]=0
            token.x[nodes.index(n),start+schema['vocabulary'].index('<unknown>')]=1
    experiments.append(('mask_arithmetic_tokens',token,'Valid unknown-operation token; operation category preserved'))
    for name,names in [('memory_access',['memory_access']),('ssa',['operand','defines']),
                       ('recurrence',['loop_carried']),('dependence',['depends_on']),('target_binding',['targets'])]:
        experiments.append(('withhold_edges_'+name,mask_relations(tensor,names),
                            'Observation subgraph: both directions withheld; no claim of program independence'))
    status=tensor.clone(); status.edge_attr[:,-len(EVIDENCE):]=0
    experiments.append(('mask_evidence_status',status,'All-zero status means withheld observation; no status relabeling'))
    records=[]
    for name,changed,semantics in experiments:
        states=extract(model,changed,rms)
        same=all(torch.equal(getattr(tensor,k),getattr(changed,k)) for k in ('x','edge_index','edge_attr'))
        record=dict(intervention=name,semantics=semantics,tensor_identical=same,
                    x_delta_norm=float((changed.x-tensor.x).norm()),
                    withheld_directed_edges=tensor.edge_index.shape[1]-changed.edge_index.shape[1],stages={})
        for stage in STAGES:
            before,after=baseline[stage][indices],states[stage][indices]
            diff=(after-before).norm(dim=1)
            before_cos=F.normalize(before,dim=1)@F.normalize(before,dim=1).T
            after_cos=F.normalize(after,dim=1)@F.normalize(after,dim=1).T
            record['stages'][stage]=dict(delta_norm=diff.tolist(),relative_delta=(diff/before.norm(dim=1).clamp_min(1e-12)).tolist(),
                cosine_to_original=F.cosine_similarity(before,after,dim=1).tolist(),
                pairwise_cosine=after_cos.tolist(),pairwise_cosine_delta=(after_cos-before_cos).tolist(),
                euclidean=torch.cdist(after.double(),after.double()).tolist())
        records.append(record)
    return dict(actions=[r['action'] for r in actions],interventions=records,
                scope='Frozen inference only; masks do not represent recompiled source changes')


@torch.no_grad()
def local_decomposition(model,tensor,states):
    raw,(ends,alpha)=model.conv_first(tensor.x,tensor.edge_index,tensor.edge_attr,
                                    return_attention_weights=True)
    assert torch.equal(F.elu(raw),states['conv1'])
    records=[]
    for slot,node in zip(tensor.action_slots.tolist(),tensor.action_nodes.tolist()):
        incoming=(ends[1]==node).nonzero().flatten()
        node_term=model.conv_first.lin_value(tensor.x[ends[0,incoming]])
        edge_term=model.conv_first.lin_edge(tensor.edge_attr[incoming])
        weights=alpha[incoming]
        skip=model.conv_first.lin_skip(tensor.x[node])
        recomposed=skip+((node_term+edge_term)*weights).sum(0)
        assert torch.allclose(recomposed,raw[node],atol=1e-7)
        records.append(dict(action='L'+str(slot+1),incoming_edges=incoming.tolist(),
            attention=weights.flatten().tolist(),skip_norm=float(skip.norm()),
            neighbor_term_norm=float((node_term*weights).sum(0).norm()),
            edge_term_norm=float((edge_term*weights).sum(0).norm()),
            incoming_node_indices=ends[0,incoming].tolist()))
    stack=torch.stack([states[s] for s in ('conv1','conv2','conv3','conv4')])
    winners=stack.argmax(0)
    return dict(first_layer=records,jk_max_layer_counts={
        'L'+str(slot+1):torch.bincount(winners[node],minlength=4).tolist()
        for slot,node in zip(tensor.action_slots.tolist(),tensor.action_nodes.tolist())},
        explanation='Term norms before ELU; affine biases included; each incoming alpha sums to one. '
                    'JK counts break exact ties in favor of earliest layer.')


@torch.no_grad()
def training_path_sensitivity(model,tensor,states):
    """Legal complete candidate controls; no measured labels or decoder involved."""
    from LLM_branch.common import directive_domains as domains
    from experiments.paper_gnn.data import candidate_vector
    from GNN_branch.target_graph_v22 import load_registry
    actions=domains.load_source_actions('gemv',ROOT/'paper_figures/gemv_rc3_20261105/inputs')
    choices={label:domains.action_tuples(action) for label,action in actions.items()}
    baseline={f'AUTO{{_{field}_{label}}}':value for label,action in actions.items()
              for field,value in zip(domains.KINDS[action.kind],choices[label][0])}
    rows=[]; candidates=[]
    for label,options in choices.items():
        for option in options:
            assignment=dict(baseline)
            for field,value in zip(domains.KINDS[actions[label].kind],option):
                assignment[f'AUTO{{_{field}_{label}}}']=value
            domains.validate_assignments(assignment,actions)
            candidates.append(candidate_vector(assignment,actions)); rows.append(dict(action=label,option=option))
    candidates=torch.stack(candidates)
    registry=load_registry()['profiles']['u200']['resources']
    context=torch.tensor([[math.log2(registry[r]['value'])/24 for r in ('BRAM_18K','DSP','FF','LUT')]+[0.]])
    static=states['jk']; pooled=[]
    for candidate in candidates:
        changed=static.clone(); nodes=tensor.action_nodes
        changed[nodes]=model.injection(torch.cat((static[nodes],candidate[tensor.action_slots]),-1))
        pooled.append(F.relu(model.post_directive(changed,tensor.edge_index,tensor.edge_attr)).mean(0))
    pooled=torch.stack(pooled); summary=model.candidate_summary(candidates.flatten(1))
    def predict(graph_part,summary_part):
        joined=torch.cat((graph_part+summary_part,context.expand(len(candidates),-1)),-1)
        return torch.cat([model.qor_heads[n](joined) for n in ('perf','area')]+
                         [model.resource_heads[n](joined) for n in ('bram','dsp','ff','lut')],-1)
    full=predict(pooled,summary)
    assert torch.equal(full,model.from_static(tensor,candidates,context,static))
    conditions={'full':full,'graph_path_only_varies':predict(pooled,summary[:1].expand_as(summary)),
                'summary_path_only_varies':predict(pooled[:1].expand_as(pooled),summary)}
    return dict(protocol='All legal per-action tuples against one complete source-derived reference; '
                'U200 10ns fixed head context. Clamp one candidate pathway to its reference output. '
                'Prediction sensitivity in standardized head units, not measured QoR or causal loss attribution.',
        head_order=['perf','area','bram','dsp','ff','lut'],candidates=rows,
        conditions={k:dict(predictions=v.tolist(),
            rms_delta_from_reference=(v-v[:1]).square().mean(0).sqrt().tolist(),
            range=(v.max(0).values-v.min(0).values).tolist()) for k,v in conditions.items()},
        graph_pooled_mean_norm=float(pooled.norm(dim=1).mean()),
        candidate_summary_mean_norm=float(summary.norm(dim=1).mean()),
        graph_pooled_rms_change=float((pooled-pooled[:1]).square().mean().sqrt()),
        candidate_summary_rms_change=float((summary-summary[:1]).square().mean().sqrt()),
        source_fields=domains.domains_from_actions(actions),
        downstream_decoder_status='NOT_EVALUATED; no trained compatible GMT checkpoint available')


def evidence_distinctions(graph,relations,actions):
    from experiments.gemv_structural_explainability.audit import catalog
    evidence=catalog(graph,relations)
    result=[]
    for row in actions:
        source=graph.nodes[row['node']]; target=graph.nodes[row['target']]
        selected=[r for r in evidence if row['action'] in r['eligible_actions']]
        # These are individually identified records, never a claim that a
        # learned projection already consumes each retained payload value.
        result.append(dict(action=row['action'],compiler_target=row['target'],
            graph_identity=graph.graph['graph_identity'],
            compiler_inputs=graph.graph['provenance'],
            source_shape=dict(value=source.get('source_array_dimensions'),
                known=source.get('source_array_dimensions') is not None,
                role='source-derived directive domain; separate from optimized storage'),
            compiler_shape=dict(value=target.get('memory_shape'),
                known=target.get('memory_shape') is not None),
            compiler_numeric={name:dict(value=target.get(name),known=isinstance(target.get(name),(int,float)))
                              for name in ('memory_rank','loop_depth','trip_count','operand_count','result_count')},
            individually_routed_records=[dict(id=r['id'],type=r['type'],status=r['status'],known=r['known'],
                provenance=r['provenance'],
                distinguishing_payload={k:v for k,v in r['payload'].items() if k in {
                    'memory_shape','memory_rank','loop_depth','trip_count','operand_count','result_count',
                    'access_indices','access','origin_loop','carrying_loop','forwarding_role','state_kind',
                    'dependence_kind','distance_components','distance_bounds','loop_carried','operation_family'}})
                for r in selected]))
    return dict(status='Availability and numerical payload coverage only; learned retrieval not measured',
                actions=result,
                required_adapter='Encode typed payload values and explicit known/status bits; sidecar presence alone is insufficient')


def authenticated_corpus(split, release, directory):
    """Read graph bytes only for train/development. Never instantiate PaperData/measurements."""
    archive=ROOT/release['archive']
    assert sha(archive)==release['archive_sha256']
    with tarfile.open(archive) as tar:
        members=[m for m in tar.getmembers() if m.name.startswith('canonical/') and m.name.endswith('.json')]
        for partition in ('train','val'):
            for kernel in sorted(split[partition+'_kernels']):
                matches=[m for m in members if Path(m.name).name.startswith(kernel+'-')]
                assert len(matches)==1, kernel
                member=matches[0]; raw=tar.extractfile(member).read()
                digest=hashlib.sha256(raw).hexdigest()
                assert digest==release['files_sha256'][member.name]
                path=directory/(kernel+'.json'); path.write_bytes(raw)
                yield kernel,partition,read_graph(path),dict(member=member.name,sha256=digest)


def labels_for(graph, row):
    if row['target'] is None: return {}
    a=graph.nodes[row['target']]; source=graph.nodes[row['node']]
    result={}
    for prop in ('memory_rank','loop_depth'):
        if isinstance(a.get(prop),(int,float)):
            result[prop]=float(a[prop])
    if a.get('is_loop') and a.get('trip_count_static')==1 and a.get('trip_count',0)>0:
        result['log2_trip_count']=math.log2(a['trip_count'])
    shape=a.get('memory_shape')
    if row['kind']=='array' and isinstance(shape,list) and all(isinstance(v,int) and v>0 for v in shape):
        result['log2_compiler_elements']=sum(math.log2(v) for v in shape)
    source_shape=source.get('source_array_dimensions')
    if row['kind']=='array' and isinstance(source_shape,list) and source_shape:
        result['source_rank']=float(len(source_shape))
    if row['kind']=='loop':
        # Presence probe, not a negative semantic proof when no edge is recorded.
        result['recorded_recurrence']=float(any(e['relation_type']=='loop_carried' and
            e.get('carrying_loop')==row['target'] and e.get('evidence_status')=='proven'
            for _,_,e in graph.edges(data=True)))
    return result


def fit_probe(train_x, train_y, train_weights, val_x, val_y, val_families):
    """Fixed-alpha ridge with train-only standardization; intercept unpenalized."""
    w=train_weights/train_weights.sum()
    mu=(train_x*w[:,None]).sum(0)
    sd=((train_x-mu).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-6)
    ym=(train_y*w).sum()
    x=(train_x-mu)/sd; z=(val_x-mu)/sd
    coefficient=torch.linalg.solve(x.T@(x*w[:,None])+torch.eye(x.shape[1],dtype=x.dtype),
                                   x.T@((train_y-ym)*w))
    prediction=z@coefficient+ym
    families={}
    for family in sorted(set(val_families)):
        keep=torch.tensor([f==family for f in val_families])
        truth=val_y[keep]; pred=prediction[keep]
        families[family]=dict(n=int(keep.sum()),mae=float((truth-pred).abs().mean()),
            baseline_mae=float((truth-ym).abs().mean()))
    return dict(alpha=1.,train_count=len(train_x),val_count=len(val_x),
        mae=float((prediction-val_y).abs().mean()),baseline_mae=float((val_y-ym).abs().mean()),
        family_macro_mae=float(np.mean([r['mae'] for r in families.values()])),
        family_macro_baseline_mae=float(np.mean([r['baseline_mae'] for r in families.values()])),
        per_family=families,predictions=prediction.tolist(),truth=val_y.tolist(),
        rounded_accuracy=float((prediction.round()==val_y).double().mean()),
        constant_baseline_rounded_accuracy=float((ym.round()==val_y).double().mean()))


def probes(samples):
    results=[]
    for prop in sorted({p for s in samples for p in s['labels']}):
        selected=[s for s in samples if prop in s['labels']]
        train=[s for s in selected if s['split']=='train']; val=[s for s in selected if s['split']=='val']
        if len(train)<10 or len(val)<3:
            results.append(dict(property=prop,status='insufficient independent train/development examples',
                                train_count=len(train),val_count=len(val))); continue
        counts=Counter(s['family'] for s in train)
        weights=torch.tensor([1/counts[s['family']] for s in train],dtype=torch.float64)
        for stage in STAGES:
            result=fit_probe(torch.stack([s['vectors'][stage] for s in train]).double(),
                torch.tensor([s['labels'][prop] for s in train],dtype=torch.float64), weights,
                torch.stack([s['vectors'][stage] for s in val]).double(),
                torch.tensor([s['labels'][prop] for s in val],dtype=torch.float64),[s['family'] for s in val])
            results.append(dict(property=prop,stage=stage,status='measured',**result,
                val_actions=[s['kernel']+':'+s['action'] for s in val]))
    return dict(protocol='Fixed alpha=1; equal total training family weight; train-only feature scaling; '
                'official disjoint development families; no tuning or GEMV fitting; property-specific category subsets',
                limitations='Only three development families; missing targets excluded; recurrence means recorded evidence presence. '
                'Predictability may reflect compiler/category correlations, not causal directive sufficiency.',results=results)


def figure(stage_metrics, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'salt-gemv-representation-diagnostics','font.size':9})
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for ax,stage in zip(axes.flat,STAGES):
        matrix=np.array(stage_metrics[stage]['cosine'])
        im=ax.imshow(matrix,vmin=.7,vmax=1,cmap='viridis')
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{matrix[i,j]:.6f}',ha='center',va='center',fontsize=6,
                        color='black' if matrix[i,j]>.88 else 'white')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=['L1','L2','L3','L4','L5'],
               yticklabels=['L1','L2','L3','L4','L5'],title=stage)
    fig.colorbar(im,ax=list(axes.flat),label='Cosine (shared scale; values below 0.7 clipped)')
    fig.suptitle('Frozen four-layer checkpoint: GEMV action representations')
    fig.savefig(out/'layerwise_similarity.png',dpi=180)
    fig.savefig(out/'layerwise_similarity.svg',metadata={'Date':None})
    svg=out/'layerwise_similarity.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=GNN)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    graph,relations,original=authenticate()
    model,schema,identity=load_encoder(args.checkpoint)
    assert next(model.parameters()).device.type=='cpu'
    before={k:v.clone() for k,v in model.state_dict().items()}
    bank=args.checkpoint.parent/'memory'
    manifest=json.loads((bank/'memory_manifest.json').read_text())
    assert manifest['gnn_checkpoint_sha256']==GNN_SHA
    assert sha(bank/'normalization.json')==manifest['normalization_sha256']
    norm=json.loads((bank/'normalization.json').read_text()); rms=norm['rms']
    split_path=ROOT/'mailohls_runs/mailohls_final_family_split_s123.json'
    split=json.loads(split_path.read_text())
    assert sha(split_path)==norm['split_sha256']
    assert norm['fitted_kernels']==split['train_kernels']
    assert not set(split['train_families'])&set(split['val_families'])
    actions=action_info(graph); tensor=tensorize(graph,schema['vocabulary'])
    states=extract(model,tensor,rms); indices=tensor.action_nodes
    saved=torch.load(Path(__file__).parent/'results/action_memory.pt',map_location='cpu',weights_only=False)
    assert torch.equal(states['memory'][indices],saved['action_memory'][tensor.action_slots])
    kinds=[r['kind'] for r in actions]
    stage_metrics={stage:metrics(x[indices],kinds) for stage,x in states.items()}
    write_json(args.out/'layerwise_metrics.json',dict(actions=actions,stages=stage_metrics))
    arrays={stage:x[indices].numpy() for stage,x in states.items()}
    arrays.update(action_nodes=indices.numpy(),slot_ids=tensor.action_slots.numpy(),
                  rms=np.array(rms),node_ids=np.array(sorted(graph.nodes)))
    for stage,x in states.items(): arrays['all_nodes_'+stage]=x.numpy()
    np.savez_compressed(args.out/'layerwise_embeddings.npz',**arrays)
    write_json(args.out/'compiler_feature_coverage.json',feature_coverage(graph,tensor,schema,actions))
    write_json(args.out/'action_feature_ablations.json',ablations(model,graph,tensor,schema,rms,states,actions))
    write_json(args.out/'first_layer_decomposition.json',local_decomposition(model,tensor,states))
    write_json(args.out/'training_path_sensitivity.json',training_path_sensitivity(model,tensor,states))
    write_json(args.out/'evidence_distinctions.json',evidence_distinctions(graph,relations,actions))
    release_path=ROOT/'docs/graph-release-v22-rc3.json'
    release=json.loads(release_path.read_text())
    corpus=[]; samples=[]; bank_checks=[]
    with tempfile.TemporaryDirectory(prefix='gemv-diagnostic-graphs-') as directory:
        for kernel,partition,g,auth in authenticated_corpus(split,release,Path(directory)):
            t=tensorize(g,schema['vocabulary']); info=action_info(g); h=extract(model,t,rms)
            kinds=[r['kind'] for r in info]
            layer_metrics={stage:metrics(x[t.action_nodes],kinds) for stage,x in h.items()}
            # Keep full pairwise matrices for named representatives; all kernels
            # retain norms, group distances and diversity in a small artifact.
            representative=kernel in {'machsuite-gemm-blocked','machsuite-md-knn','machsuite-stencil2d',
                'rodinia-hotspot-0-baseline','machsuite-sort-radix','machsuite-viterbi',
                'rodinia_pathfinder_0_baseline_0'}
            if not representative:
                for value in layer_metrics.values():
                    for key in ('cosine','euclidean','centered_singular_values'):
                        value.pop(key)
            corpus.append(dict(kernel=kernel,split=partition,family=family_id_from_kernel_name(kernel),
                graph=auth,actions=info,nodes=len(g),edges=g.number_of_edges(),
                full_pairwise_matrices=representative,stages=layer_metrics))
            for i,row in enumerate(info):
                samples.append(dict(kernel=kernel,split=partition,family=family_id_from_kernel_name(kernel),
                    action=row['action'],kind=row['kind'],labels=labels_for(g,row),
                    vectors={stage:x[t.action_nodes[i]] for stage,x in h.items()}))
            filename=kernel+'.memory.pt'
            assert sha(bank/filename)==manifest['files'][filename]
            pack=torch.load(bank/filename,map_location='cpu',weights_only=False)
            expected=pack['node_embs'][t.action_slots]
            actual=h['rms_memory'][t.action_nodes]
            error=float((expected-actual).abs().max())
            assert torch.allclose(expected,actual,atol=2e-6,rtol=2e-5),(kernel,error)
            bank_checks.append(dict(kernel=kernel,max_absolute_error=error))
    write_json(args.out/'corpus_layerwise_metrics.json',dict(locked_test_labels_read=False,kernels=corpus))
    write_json(args.out/'linear_probes.json',probes(samples))
    with (args.out/'layerwise_metrics.csv').open('w') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['kernel','split','family','stage','within_array_distance','within_loop_distance',
                         'between_category_distance','centered_variance','centered_fraction','centered_effective_rank'])
        for entry in [dict(kernel='gemv',split='descriptive_only',family='gemv',stages=stage_metrics),*corpus]:
            for stage,m in entry['stages'].items():
                writer.writerow([entry['kernel'],entry['split'],entry['family'],stage,
                    *[m['groups'][key]['mean_distance'] for key in ('within_array','within_loop','between_category')],
                    m['centered_total_variance'],m['centered_energy_fraction'],m['centered_effective_rank']])
    figure(stage_metrics,args.out)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    write_json(args.out/'diagnostic_manifest.json',dict(schema='salt-frozen-representation-diagnostics-v1',
        checkpoint=identity,normalization=norm,normalization_sha256=sha(bank/'normalization.json'),
        normalized_memory_bank_checks=bank_checks,split_sha256=sha(split_path),
        release_manifest_sha256=sha(release_path),gemv_exact_historical_memory=True,
        encoder_state_unchanged=True,device='cpu',threads=4,torch=torch.__version__,
        train_kernels=len(split['train_kernels']),development_kernels=len(split['val_kernels']),
        locked_test_labels_read=False,gemv_training=False,
        files={f.name:sha(f) for f in sorted(args.out.iterdir()) if f.is_file()},
        script_sha256=sha(Path(__file__)),expensive_alignment_started=False))
    print(json.dumps(dict(out=str(args.out),kernels=len(corpus),actions=len(samples),rms=rms)))


if __name__=='__main__':
    main()
