"""Data-driven publication figures; never synthesize missing learned results."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from experiments.gemv_structural_explainability.audit import write_json,sha


def save(fig,out,name):
    for suffix in ('svg','png','pdf'):
        fig.savefig(out/(name+'.'+suffix),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def generate(results,out):
    results,out = Path(results),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
                         'axes.spines.right':False})
    actions = json.loads((results/'action_provenance.json').read_text())
    metrics = json.loads((results/'embedding_metrics.json').read_text())
    manifest = json.loads((results/'manifest.json').read_text())
    data = np.load(results/'action_embeddings.npz')
    active = data['action'][data['active']]
    if active.shape!=(5,128) or not np.isfinite(active).all():
        raise ValueError('Invalid exported embeddings')
    if not np.allclose(np.linalg.norm(active,axis=-1),metrics['norms'],rtol=1e-6):
        raise ValueError('Figure/metrics inconsistency')
    unit = active/np.linalg.norm(active,axis=-1,keepdims=True)
    labels = [f"{r['action']}  {r['construct']}  ({r['source_file']}:{r['line']})" for r in actions]
    bound = float(abs(unit).max())
    fig,ax = plt.subplots(figsize=(13,3.7))
    im = ax.imshow(unit,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound,interpolation='nearest')
    ax.set_yticks(range(5),labels)
    ax.set_xticks([0,15,31,47,63,79,95,111,127],['1','16','32','48','64','80','96','112','128'])
    ax.axvline(63.5,color='#252525',lw=1.5)
    ax.set_xlabel('Embedding coordinate: first-layer state [1–64] | JK-max state [65–128]')
    ax.set_title('GEMV: trained four-layer structural action memory',loc='left',weight='bold',pad=12)
    fig.colorbar(im,ax=ax,shrink=.85,pad=.02,label='Coordinate / full-vector L2 norm')
    fig.text(.01,-.035,'Actual frozen checkpoint inference · per-row L2 normalization for display · coordinates have no assigned semantic meanings',fontsize=9,color='#454545')
    save(fig,out,'embedding_heatmap')
    fig,ax = plt.subplots(figsize=(5.2,4.6))
    cosine = unit@unit.T
    if not np.allclose(cosine,metrics['pairwise_cosine'],rtol=1e-5,atol=1e-6):
        raise ValueError('Cosine metrics differ from raw vectors')
    im = ax.imshow(cosine,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(5),[r['action'] for r in actions]);ax.set_yticks(range(5),[r['action'] for r in actions])
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{cosine[i,j]:.3f}',ha='center',va='center',color='white' if abs(cosine[i,j])>.6 else '#222222')
    ax.set_title('Action-memory cosine similarity',loc='left',weight='bold')
    fig.colorbar(im,ax=ax,shrink=.85,label='Cosine similarity')
    save(fig,out,'embedding_cosine')
    fig,axes = plt.subplots(1,2,figsize=(10.5,3.6))
    x = np.arange(5)
    for dx,key,label,color in [(-.17,'first_layer_norms','First layer','#31718e'),(.17,'jk_norms','JK max','#c18a42')]:
        axes[0].bar(x+dx,metrics[key],width=.32,label=label,color=color)
    axes[0].set_xticks(x,[r['action'] for r in actions]);axes[0].set_ylabel('Raw L2 norm');axes[0].legend(frameon=False)
    axes[0].set_title('Local and multihop components',loc='left',weight='bold')
    axes[1].bar(x,metrics['first_vs_jk_cosine'],color='#617b62')
    axes[1].set_ylim(-1,1);axes[1].set_xticks(x,[r['action'] for r in actions]);axes[1].set_ylabel('Cosine (first layer, JK max)')
    axes[1].set_title('Within-action component agreement',loc='left',weight='bold')
    fig.tight_layout();save(fig,out,'embedding_components')

    # Source snippets and graph anchors are taken from authenticated sidecars.
    from experiments.gemv_structural_explainability.audit import ROOT
    lines = (ROOT/actions[0]['source_path']).read_text().splitlines()
    fig,ax = plt.subplots(figsize=(13,5.0));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    for xpos,title in [(.015,'Authenticated source'),(.405,'RC3 compiler target'),(.785,'Frozen action memory')]:
        ax.text(xpos,.95,title,weight='bold',fontsize=12)
    for i,r in enumerate(actions):
        y = .82-i*.165
        snippet = lines[r['line']-1].strip().replace('/*'+r['action']+':*/','').strip()
        source = f"{r['action']} · {r['source_file']}:{r['line']}:{r['column']}\n{snippet}"
        a = r['compiler_metadata']
        detail = a.get('value_type') or f"{a.get('text')} · trip count {a.get('trip_count','unknown')}"
        if r['action']=='L3':detail += ' · scalarized; source y[32]'
        for xpos,width,text in [(.01,.32,source),(.40,.32,r['compiler_anchor']+'\n'+detail),
                                (.79,.20,f"slot {r['slot']}  ↔  {r['action']}\n[first layer | JK max] 128D")]:
            ax.add_patch(FancyBboxPatch((xpos,y-.055),width,.12,boxstyle='round,pad=.008',
                facecolor='#f0f4f6',edgecolor='#b5c2c9',lw=.8))
            ax.text(xpos+.008,y+.005,text,fontsize=8.6,va='center')
        for start,end in [(.335,.395),(.735,.785)]:
            ax.add_patch(FancyArrowPatch((start,y+.005),(end,y+.005),arrowstyle='-|>',mutation_scale=12,color='#526777'))
    ax.text(.01,.025,'66 nodes · 106 edges · 5 authenticated bindings · slots 5–63 inactive (zero) · source and compiler shapes kept distinct',fontsize=9)
    ax.set_title('Source → compiler anchor → absolute memory slot',loc='left',weight='bold',pad=18)
    save(fig,out,'source_compiler_memory')
    catalog = json.loads((results/'evidence_catalog.json').read_text())
    types = sorted({r['type'] for r in catalog['records'] if r['eligible_actions']})
    count = np.array([[catalog['coverage'][r['action']]['by_type'].get(t,0) for t in types] for r in actions])
    fig,ax=plt.subplots(figsize=(12,4.7))
    im=ax.imshow(count,aspect='auto',cmap='Blues',vmin=0)
    ax.set_yticks(range(5),[r['action']+' '+r['construct'] for r in actions])
    ax.set_xticks(range(len(types)),[t.replace('_','\n') for t in types],fontsize=8)
    for i in range(5):
        for j in range(len(types)):ax.text(j,i,str(count[i,j]),ha='center',va='center',fontsize=9,
                                         color='white' if count[i,j]>count.max()*.6 else '#222222')
    ax.set_title('Compiler-authorized evidence accessibility (record counts)',loc='left',weight='bold')
    fig.colorbar(im,ax=ax,shrink=.8,label='Records');fig.text(.01,-.02,'Evidence availability only; learned retrieval and causal decision influence have not been evaluated.',fontsize=9)
    save(fig,out,'action_evidence_coverage')
    write_json(out/'figure_manifest.json',dict(schema='salt-gemv-figure-provenance-v1',
        checkpoint_sha256=manifest['gnn_sha256'],graph_identity=actions[0]['graph_identity'],
        inputs={n:sha(results/n) for n in ('action_embeddings.npz','action_provenance.json','embedding_metrics.json','evidence_catalog.json')},
        display_normalization='embedding heatmap: each full 128D row divided by its L2 norm; shared symmetric color scale',
        files={f.name:sha(f) for f in sorted(out.iterdir()) if f.suffix in ('.svg','.png','.pdf')},
        learned_attention_figures_generated=False))
