"""Numerical diagnostics. Vector coordinates have no assigned semantic labels."""
import torch


def embedding_metrics(pack,first,jk):
    active = pack['active'][0]
    action = pack['action'][0,active].float()
    norms = action.norm(dim=-1)
    unit = action/norms[:,None].clamp_min(1e-12)
    f,j = action[:,:64],action[:,64:]
    return dict(schema='salt-gemv-four-layer-embedding-metrics-v1',
        actions=['L'+str(i+1) for i in active.nonzero().flatten().tolist()],
        shape=list(pack['action'][0].shape),active_slots=active.nonzero().flatten().tolist(),
        norms=norms.tolist(),first_layer_norms=f.norm(dim=-1).tolist(),
        jk_norms=j.norm(dim=-1).tolist(),pairwise_cosine=(unit@unit.T).tolist(),
        first_vs_jk_cosine=torch.nn.functional.cosine_similarity(f,j,dim=-1).tolist(),
        first_layer_pairwise_cosine=(torch.nn.functional.normalize(f,dim=-1)@
                                   torch.nn.functional.normalize(f,dim=-1).T).tolist(),
        jk_pairwise_cosine=(torch.nn.functional.normalize(j,dim=-1)@
                           torch.nn.functional.normalize(j,dim=-1).T).tolist(),
        active_dimension_variance=action.var(dim=0,unbiased=False).tolist(),
        variance_mean=float(action.var(dim=0,unbiased=False).mean()),
        nonfinite_count=int((~torch.isfinite(action)).sum()),
        inactive_slots_exact_zero=bool((pack['action'][0,~active]==0).all()),
        raw_node_first_shape=list(first.shape),raw_node_jk_shape=list(jk.shape),
        normalization='raw export; figures use explicitly described row L2 normalization',
        learned_summary_metrics='UNAVAILABLE: alignment not trained',
        interpretation='Similarity/norm diagnostics do not establish semantic interpretation, retrieval or QoR')
