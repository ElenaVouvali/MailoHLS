"""Deterministic CPU-only analysis and high-resolution figures from saved inference evidence."""
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.hashsalt'] = 'salt-hls-gemv-stage1-full-depth-v1'
from .common import DEFAULT_OUT,read_json,write_json,sha

ACTIONS=[f'L{i}' for i in range(1,6)]
FOCUS=['auto{_ARRAY_T_L1}','auto{_ARRAY_F_L1}','auto{_ARRAY_D_L1}','auto{_ARRAY_F_L2}','auto{_PIPE_L4}','auto{_UNROLL_L4}','auto{_PIPE_L5}','auto{_UNROLL_L5}']

def safe_cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else float('nan')
def sim(a,b):
 a=np.asarray(a,np.float32);b=np.asarray(b,np.float32);n1=float(np.linalg.norm(a));n2=float(np.linalg.norm(b))
 return dict(cosine=safe_cos(a,b),relative_l2=float(np.linalg.norm(b-a)/(n1+1e-12)),norm_ref=n1,norm_cmp=n2,norm_relative_change=float((n2-n1)/(n1+1e-12)))
def read_results(out):
 rows=read_json(out/'candidate_scores.json');return {x['request_id']:x for x in rows}
def req_ids():
 return [f'GRID_{d}_{c}NS_{b}PCT' for d in ['ZCU104','XCU200'] for c in ['10P0','5P0','3P33'] for b in [100,50,25]]
def contrasts():
 out=[]
 out.append(dict(kind='device',name='XCU200 vs ZCU104 / 10ns / 100%',a='GRID_ZCU104_10P0NS_100PCT',b='GRID_XCU200_10P0NS_100PCT'))
 for d in ['ZCU104','XCU200']:
  for c in ['5P0','3P33']:
   out.append(dict(kind='clock',name=f'{d} / {c}ns vs 10ns / 100%',a=f'GRID_{d}_10P0NS_100PCT',b=f'GRID_{d}_{c}NS_100PCT'))
  for budget in [50,25]:
   out.append(dict(kind='budget',name=f'{d} / {budget}% vs 100% / 10ns',a=f'GRID_{d}_10P0NS_100PCT',b=f'GRID_{d}_10P0NS_{budget}PCT'))
 return out

def fields_of(row):return {x['lhs']:x for x in row['decisions']}
def control_scores(row,field):
 return next((x for x in row['trace_records'] if x['prefix_mode']=='controlled' and x['field']==field),None)
def margin(scores,r1=None,r2=None):
 d={x['rhs']:float(x.get('score',x.get('mean_logprob'))) for x in scores}
 if r1 is not None:
  return d[r1]-d[r2] if r1 in d and r2 in d else None
 if len(d)<2:return None
 vals=sorted(d.values(),reverse=True);return vals[0]-vals[1]
def main(out):
 out=Path(out); figdir=out/'figures';figdir.mkdir(exist_ok=True)
 results=read_results(out); ids=req_ids(); assert all(x in results for x in ids)
 tensors={}; indices={}
 for rid in results:
  p=out/'hidden_states'/f'{rid}.npz'
  if p.exists():tensors[rid]=np.load(p)['vectors'];indices[rid]=read_json(out/'hidden_states'/f'{rid}.index.json')
 def state(rid,mode,field,layer,boundary='post_self_attention_residual'):
  rows=indices[rid]; hit=[x for x in rows if x['prefix_mode']==mode and x['field']==field and x['layer']==layer and x['boundary']==boundary]
  if len(hit)!=1:return None
  return tensors[rid][hit[0]['vector_index']]
 layer_rows=[]; score_rows=[]
 for cmp in contrasts():
  for mode in ['natural','controlled']:
   for field in [x['lhs'] for x in results[cmp['a']]['decisions']]:
    for layer in range(1,33):
     a=state(cmp['a'],mode,field,layer);b=state(cmp['b'],mode,field,layer)
     if a is None or b is None:continue
     for boundary in ['post_self_attention_residual','post_decoder_residual','final_rms_normalized_lm_head_input']:
      if boundary!='post_self_attention_residual':a=state(cmp['a'],mode,field,layer,boundary);b=state(cmp['b'],mode,field,layer,boundary)
      if a is None or b is None:continue
      layer_rows.append(dict(kind=cmp['kind'],contrast=cmp['name'],request_a=cmp['a'],request_b=cmp['b'],prefix_mode=mode,field=field,action=next((f'L{i}' for i in range(1,6) if f'_L{i}' in field.upper()),'source'),layer=layer,boundary=boundary,**sim(a,b)))
  da,db=fields_of(results[cmp['a']]),fields_of(results[cmp['b']])
  for field in da:
   ca,cb=da[field],db[field];same=set(ca['active_candidate_domain'])==set(cb['active_candidate_domain'])
   sa,sb=ca['top_candidates'],cb['top_candidates'];common=sorted({x['rhs'] for x in sa}&{x['rhs'] for x in sb})
   ma=margin(sa);mb=margin(sb)
   ra=[x['rhs'] for x in sa];rb=[x['rhs'] for x in sb]
   score_rows.append(dict(kind=cmp['kind'],contrast=cmp['name'],request_a=cmp['a'],request_b=cmp['b'],field=field,action=ca['label'],selected_a=ca['chosen_rhs'],selected_b=cb['chosen_rhs'],winner_changed=ca['chosen_rhs']!=cb['chosen_rhs'],candidate_domain_identical=same,common_candidates=len(common),margin_a=ma,margin_b=mb,delta_margin=(mb-ma) if same and ma is not None and mb is not None else None,top_two_ranking_changed=(ra[:2]!=rb[:2]) if same else None,top_candidates_a=sa[:5],top_candidates_b=sb[:5]))
 # Full layer/action hardware movement matrices: median over controlled target-pair vectors, natural prefixes shown separately.
 hardware_heat={}
 for mode in ['natural','controlled']:
  for kind in ['device','clock','budget']:
   mat=np.full((32,5),np.nan)
   for li in range(1,33):
    for ai,action in enumerate(ACTIONS):
     vals=[r['relative_l2'] for r in layer_rows if r['prefix_mode']==mode and r['kind']==kind and r['boundary']=='post_self_attention_residual' and r['layer']==li and r['action']==action]
     if vals:mat[li-1,ai]=float(np.median(vals))
   hardware_heat[f'{mode}_{kind}']=mat
 # Structure/source contrasts under matched controlled prefixes.
 subcontexts=['SUB_ZCU104_10P0NS_100PCT','SUB_ZCU104_10P0NS_25PCT','SUB_ZCU104_3P33NS_100PCT','SUB_ZCU104_3P33NS_25PCT','SUB_XCU200_10P0NS_100PCT','SUB_XCU200_10P0NS_25PCT','SUB_XCU200_3P33NS_100PCT','SUB_XCU200_3P33NS_25PCT']
 structures=[]; interactions=[]
 for ctx in subcontexts:
  for mode in ['natural','controlled']:
   for field in FOCUS:
    if mode=='controlled' and field not in FOCUS:continue
    for layer in range(1,33):
     ids2=[f'{src}__{ctx}' for src in ['reduce32','independent32','reduce32_rename']]
     vec={src:state(rid,mode,field,layer) for src,rid in zip(['reduce32','independent32','reduce32_rename'],ids2)}
     if all(v is not None for v in vec.values()):
      structures.append(dict(context=ctx,prefix_mode=mode,field=field,action=next((f'L{i}' for i in range(1,6) if f'_L{i}' in field.upper()),''),layer=layer,independent_vs_reduce=sim(vec['reduce32'],vec['independent32'])['relative_l2'],rename_vs_reduce=sim(vec['reduce32'],vec['reduce32_rename'])['relative_l2'],syntax_control_cosine=sim(vec['reduce32'],vec['reduce32_rename'])['cosine']))
  # margins for hardware interactions: 100 vs 25 budget, 10 vs 3.33 clock, device at 100.
  prefix=ctx.split('SUB_',1)[1]
  if prefix=='ZCU104_10P0NS_100PCT':continue
  # find same-stratum comparison pair from the subset.
  if '_25PCT' in prefix:
   a=f'reduce32__SUB_{prefix.replace("25PCT","100PCT")}';b=f'reduce32__SUB_{prefix}'
   kind='budget'; other_a=f'independent32__SUB_{prefix.replace("25PCT","100PCT")}';other_b=f'independent32__SUB_{prefix}'
  elif '3P33NS' in prefix:
   a=f'reduce32__SUB_{prefix.replace("3P33NS","10P0NS")}';b=f'reduce32__SUB_{prefix}'
   kind='clock';other_a=f'independent32__SUB_{prefix.replace("3P33NS","10P0NS")}';other_b=f'independent32__SUB_{prefix}'
  elif prefix.startswith('XCU200_10P0NS_100PCT'):
   a='reduce32__SUB_ZCU104_10P0NS_100PCT';b=f'reduce32__SUB_{prefix}';kind='device';other_a='independent32__SUB_ZCU104_10P0NS_100PCT';other_b=f'independent32__SUB_{prefix}'
  else:continue
  if a not in results or b not in results or other_a not in results or other_b not in results:continue
  for field in FOCUS:
   for layer in range(1,33):
    v={'rT1':state(a,'controlled',field,layer),'rT2':state(b,'controlled',field,layer),'iT1':state(other_a,'controlled',field,layer),'iT2':state(other_b,'controlled',field,layer)}
    if all(x is not None for x in v.values()):
     inter=(v['iT2']-v['rT2'])-(v['iT1']-v['rT1'])
     interactions.append(dict(kind=kind,target_a=a,target_b=b,field=field,layer=layer,interaction_l2=float(np.linalg.norm(inter)),interaction_relative=float(np.linalg.norm(inter)/(np.linalg.norm(v['rT1'])+1e-12))))
 # Build score margin change rows over source-target factorial, only where all 4 exact candidate RHS scores exist.
 for field in FOCUS:
  for kind,a,b,ia,ib in [
   ('budget','reduce32__SUB_ZCU104_10P0NS_100PCT','reduce32__SUB_ZCU104_10P0NS_25PCT','independent32__SUB_ZCU104_10P0NS_100PCT','independent32__SUB_ZCU104_10P0NS_25PCT'),
   ('clock','reduce32__SUB_ZCU104_10P0NS_100PCT','reduce32__SUB_ZCU104_3P33NS_100PCT','independent32__SUB_ZCU104_10P0NS_100PCT','independent32__SUB_ZCU104_3P33NS_100PCT'),
   ('device','reduce32__SUB_ZCU104_10P0NS_100PCT','reduce32__SUB_XCU200_10P0NS_100PCT','independent32__SUB_ZCU104_10P0NS_100PCT','independent32__SUB_XCU200_10P0NS_100PCT')]:
   rows=[control_scores(results[x],field) if x in results else None for x in [a,b,ia,ib]]
   if any(x is None for x in rows):continue
   maps=[{v['rhs']:float(v.get('score',v.get('mean_logprob'))) for v in z['scores']} for z in rows]
   common=set.intersection(*(set(m) for m in maps))
   if len(common)<2:continue
   ra=rows[0]['scores'];ranked=sorted(ra,key=lambda v:float(v.get('score',v.get('mean_logprob'))),reverse=True);c1,c2=ranked[0]['rhs'],ranked[1]['rhs']
   if c1 not in common or c2 not in common:continue
   ms=[m[c1]-m[c2] for m in maps];inter=(ms[3]-ms[2])-(ms[1]-ms[0])
   interactions.append(dict(kind=kind,field=field,score_rhs_pair=[c1,c2],score_margin_interaction=inter,source_margin_T1=ms[0],source_margin_T2=ms[1],other_margin_T1=ms[2],other_margin_T2=ms[3],candidate_set_size=len(common)))
 # Save numerical evidence tables.
 write_json(out/'layer_pairwise_metrics.json',layer_rows);write_json(out/'candidate_preference_contrasts.json',score_rows);write_json(out/'structure_target_interactions.json',interactions);write_json(out/'structure_pairwise_metrics.json',structures)
 for name,rows in [('layer_pairwise_metrics.csv',layer_rows),('candidate_preference_contrasts.csv',score_rows),('structure_target_interactions.csv',interactions),('structure_pairwise_metrics.csv',structures)]:
  if rows:
   keys=list(rows[0]);
   with (out/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 # Figure 1: 32-layer by action hardware sensitivity, natural trajectory.
 fig,axs=plt.subplots(1,3,figsize=(14,8),constrained_layout=True)
 for ax,kind in zip(axs,['device','clock','budget']):
  mat=hardware_heat[f'natural_{kind}'];im=ax.imshow(mat,aspect='auto',origin='lower',cmap='magma',vmin=0,vmax=np.nanpercentile(mat,95) if np.isfinite(mat).any() else 1)
  ax.set_title(f'{kind.capitalize()} contrasts');ax.set_xticks(range(5),ACTIONS);ax.set_yticks([0,7,15,23,31],[1,8,16,24,32]);ax.set_ylabel('Decoder layer')
 fig.colorbar(im,ax=axs,label='Median relative L2 movement (natural prefix)',shrink=.75);fig.suptitle('Full-depth hardware sensitivity by action (same source)')
 savefig(fig,figdir,'fig1_layer_action_hardware')
 # Figure 2: structural source and syntax controls.
 fig,axs=plt.subplots(1,2,figsize=(11,8),constrained_layout=True)
 for ax,key,title in [(axs[0],'independent_vs_reduce','Independent iterations vs reduction'),(axs[1],'rename_vs_reduce','Semantics-preserving rename vs reduction')]:
  mat=np.full((32,5),np.nan)
  for li in range(1,33):
   for ai,a in enumerate(ACTIONS):
    vals=[r[key] for r in structures if r['prefix_mode']=='controlled' and r['layer']==li and r['action']==a]
    if vals:mat[li-1,ai]=np.median(vals)
  im=ax.imshow(mat,aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=max(.05,float(np.nanpercentile(mat,95))) if np.isfinite(mat).any() else .1);ax.set_title(title);ax.set_xticks(range(5),ACTIONS);ax.set_yticks([0,7,15,23,31],[1,8,16,24,32]);ax.set_ylabel('Decoder layer')
 fig.colorbar(im,ax=axs,label='Median relative L2 movement (controlled prefix)',shrink=.75);fig.suptitle('Structure contrasts after source and prefix controls')
 savefig(fig,figdir,'fig2_layer_action_structure')
 # Figure 3: candidate margin deltas for each target contrast and field.
 piv=[r for r in score_rows if r['delta_margin'] is not None]
 kinds=['device','clock','budget']; fields=sorted({r['field'] for r in piv});mat=np.full((len(fields),len(kinds)),np.nan)
 for i,f in enumerate(fields):
  for j,k in enumerate(kinds):
   vals=[r['delta_margin'] for r in piv if r['field']==f and r['kind']==k]
   if vals:mat[i,j]=np.mean(vals)
 fig,ax=plt.subplots(figsize=(8,max(5,len(fields)*.3)),constrained_layout=True);im=ax.imshow(mat,aspect='auto',cmap='coolwarm',vmin=-max(.05,np.nanmax(np.abs(mat))),vmax=max(.05,np.nanmax(np.abs(mat))))
 ax.set_xticks(range(3),kinds);ax.set_yticks(range(len(fields)),[f.replace('auto{_','').replace('}','') for f in fields]);ax.set_title('Change in production top-two mean-logprob margin (matched candidate sets)');fig.colorbar(im,ax=ax,label='Δ margin (log-probability/token)');savefig(fig,figdir,'fig3_candidate_margin_changes')
 # Figure 4: natural/controlled movement at the selected focus fields.
 fig,axs=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
 for ax,kind in zip(axs,kinds):
  for mode,style in [('natural','-'),('controlled','--')]:
   means=[]
   for li in range(1,33):
    vals=[r['relative_l2'] for r in layer_rows if r['kind']==kind and r['prefix_mode']==mode and r['boundary']=='post_self_attention_residual' and r['layer']==li and r['field'] in FOCUS]
    means.append(np.median(vals) if vals else np.nan)
   ax.plot(range(1,33),means,style,label=mode)
  ax.set_title(kind.capitalize());ax.set_xlabel('Layer');ax.set_ylabel('Median relative L2');ax.legend(frameon=False);ax.grid(alpha=.2)
 fig.suptitle('Autoregressive prefix contribution: selected decisions');savefig(fig,figdir,'fig4_natural_controlled')
 # Figure 5: layer × field source-structure/target interaction matrices.
 vector_inter=[x for x in interactions if 'interaction_relative' in x]
 focus_labels=[x.replace('auto{_','').replace('}','') for x in FOCUS]
 fig,axs=plt.subplots(1,3,figsize=(15,8),constrained_layout=True)
 for ax,kind in zip(axs,kinds):
  mat=np.full((32,len(FOCUS)),np.nan)
  for li in range(1,33):
   for fi,field in enumerate(FOCUS):
    vals=[r['interaction_relative'] for r in vector_inter if r['kind']==kind and r['field']==field and r['layer']==li]
    if vals:mat[li-1,fi]=float(np.median(vals))
  im=ax.imshow(mat,aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=max(.02,float(np.nanpercentile(mat,95))) if np.isfinite(mat).any() else .1)
  ax.set_title(kind.capitalize());ax.set_xticks(range(len(FOCUS)),focus_labels,rotation=60,ha='right');ax.set_yticks([0,7,15,23,31],[1,8,16,24,32]);ax.set_ylabel('Layer')
 fig.colorbar(im,ax=axs,label='Relative L2 of source×target interaction',shrink=.75);fig.suptitle('Compiler-verified source contrast × hardware context (controlled prefix)')
 savefig(fig,figdir,'fig5_structure_target_interaction')
 # Figure 6: learned hardware tokens and initial-vs-trained embedding geometry.
 embjson=out/'special_token_embeddings.json';embedding=read_json(embjson) if embjson.exists() else {'tokens':[]}
 wanted=[r for r in embedding['tokens'] if r['token'] in ['<DEV=XCZU7EV_FFVC1156_2E>','<DEV=XCU200_FSGD2104_2E>','<CLK=10NS>','<CLK=5NS>','<CLK=3P33NS>','<OBJ=PARETO_LATENCY>']]
 z=np.load(out/'special_token_embeddings.npz') if (out/'special_token_embeddings.npz').exists() else None
 fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
 if wanted:
  axs[0].bar(range(len(wanted)),[x['relative_update_l2'] for x in wanted],color='#4c78a8');axs[0].set_xticks(range(len(wanted)),[x['token'].replace('<','').replace('>','') for x in wanted],rotation=30,ha='right');axs[0].set_ylabel('||trained−initial||₂ / ||initial||₂');axs[0].set_title('Hardware/objective token row updates')
  ids=[x['token_id'] for x in wanted]; loc=[list(z['token_ids']).index(i) for i in ids]; m=z['trained'][loc]@z['trained'][loc].T; norms=np.linalg.norm(z['trained'][loc],axis=1);m=m/(norms[:,None]*norms[None,:]);im=axs[1].imshow(m,vmin=-1,vmax=1,cmap='coolwarm');axs[1].set_xticks(range(len(ids)),[x['token'].replace('<','').replace('>','') for x in wanted],rotation=45,ha='right');axs[1].set_yticks(range(len(ids)),[x['token'].replace('<','').replace('>','') for x in wanted]);axs[1].set_title('Cosine of trained token rows');fig.colorbar(im,ax=axs[1],label='cosine')
 else: axs[0].text(.5,.5,'embedding receipt unavailable',ha='center');axs[1].axis('off')
 fig.suptitle('Learned input-token vectors; geometric similarity is not physical device similarity');savefig(fig,figdir,'fig6_hardware_token_embeddings')
 if z is not None and wanted:
  idlist=[x['token_id'] for x in wanted];loc=[list(z['token_ids']).index(i) for i in idlist]
  mats={}
  for key in ['initial','trained','updates']:
   arr=z[key][loc].astype(np.float64);norm=np.linalg.norm(arr,axis=1);mats[key]=(arr@arr.T)/(norm[:,None]*norm[None,:]+1e-12)
  write_json(out/'hardware_token_cosines.json',dict(tokens=[x['token'] for x in wanted],initial=mats['initial'].tolist(),trained=mats['trained'].tolist(),update=mats['updates'].tolist(),interpretation='token vector geometry only; no FPGA physical similarity semantics'))
 # Figure 7: evidence table/curve for candidate layer placement.
 patch_data=read_json(out/'activation_patching.json') if (out/'activation_patching.json').exists() else {'records':[]}
 layer_evidence=[];targ=[];rec=[];rename=[];patch=[];flip=[]
 for li in range(1,33):
  tv=[r['relative_l2'] for r in layer_rows if r['prefix_mode']=='controlled' and r['boundary']=='post_self_attention_residual' and r['layer']==li and r['field'] in FOCUS]
  sv=[r['independent_vs_reduce'] for r in structures if r['prefix_mode']=='controlled' and r['layer']==li and r['field'] in FOCUS]
  rv=[r['rename_vs_reduce'] for r in structures if r['prefix_mode']=='controlled' and r['layer']==li and r['field'] in FOCUS]
  pr=[r for r in patch_data['records'] if r.get('status')=='PATCHED' and r.get('layer')==li]
  t=float(np.median(tv)) if tv else float('nan');sval=float(np.median(sv)) if sv else float('nan');rval=float(np.median(rv)) if rv else float('nan')
  pval=float(np.median([abs(x['delta_margin']) for x in pr])) if pr else float('nan');fval=int(sum(bool(np.sign(x['receiver_margin'])!=np.sign(x['patched_margin'])) for x in pr))
  layer_evidence.append(dict(layer=li,controlled_target_median_relative_l2=t,recurrence_source_median_relative_l2=sval,rename_control_median_relative_l2=rval,activation_patch_median_abs_margin_change=pval,activation_patch_winner_flips=fval,patch_samples=len(pr),interpretation='descriptive; per-layer patch subset only' if pr else 'no patch at this layer'))
  targ.append(t);rec.append(sval);rename.append(rval);patch.append(pval);flip.append(fval)
 write_json(out/'layer_evidence.json',layer_evidence)
 with (out/'layer_evidence.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(layer_evidence[0]));w.writeheader();w.writerows(layer_evidence)
 fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True);ax.plot(range(1,33),targ,label='Target movement (controlled prefix)');ax.plot(range(1,33),rec,label='Reduction vs independent source movement');ax.plot(range(1,33),rename,label='Rename control movement');ax.set_xlabel('Layer');ax.set_ylabel('Median relative L2');ax.set_xticks(range(1,33,2));ax.grid(alpha=.2)
 ax2=ax.twinx();ax2.plot(range(1,33),patch,color='#b23a48',marker='o',label='Median |patched Δ margin|');ax2.set_ylabel('Candidate-score margin change (mean log-prob/token)');lines=ax.get_lines()+ax2.get_lines();ax.legend(lines,[x.get_label() for x in lines],frameon=False,loc='upper left');ax.set_title('All-layer representation evidence plus read-only patching at sampled layers');savefig(fig,figdir,'fig7_layer_shortlist_evidence')
 # Figure 9: layer-specific donor patch effect and placebo.
 patch_rows=[x for x in patch_data['records'] if x.get('status')=='PATCHED'];fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
 if patch_rows:
  for kind in kinds:
   xx=[];yy=[]
   for li in patch_data['layers']:
    vals=[abs(x['delta_margin']) for x in patch_rows if x['kind']==kind and x['layer']==li]
    xx.append(li);yy.append(float(np.median(vals)) if vals else np.nan)
   ax.plot(xx,yy,marker='o',label=kind)
  placebo=max((abs(x.get('placebo_delta',0.)) for x in patch_rows),default=0.);ax.axhline(placebo,color='black',linestyle=':',label=f'max placebo |Δ|={placebo:.1e}')
 ax.set_xlabel('Patched decoder layer');ax.set_ylabel('Absolute change in matched candidate margin');ax.set_xticks(patch_data.get('layers',[]));ax.grid(alpha=.2);ax.legend(frameon=False);ax.set_title('Donor R1 post-attention state patched into target request; same-condition placebo')
 savefig(fig,figdir,'fig9_activation_patch_margin')
 # Figure 8: architecture information flow, implemented Stage-1 vs future fusion.
 fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
 boxes=[(.03,.60,.22,.25,'C source + target\nconstraints'),(.34,.60,.25,.25,'Stage-1 LLM\n32 decoder layers'),(.72,.60,.24,.25,'Constrained RHS scoring\n13 directive fields'),(.03,.12,.22,.25,'C → MLIR V2.2 RC3\ncompiler graph'),(.34,.12,.25,.25,'V4b structural encoder\n64 slots × 128D + masks'),(.72,.12,.24,.25,'Future gated cross-attention\npost-SA residual tests')]
 for x,y,w,h,t in boxes:ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#eef3f8',edgecolor='#315a7d',lw=1.5));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
 for p1,p2 in [((.25,.725),(.34,.725)),((.59,.725),(.72,.725)),((.14,.60),(.14,.37)),((.25,.245),(.34,.245)),((.59,.245),(.72,.245)),((.465,.60),(.465,.37))]:ax.annotate('',xy=p2,xytext=p1,arrowprops=dict(arrowstyle='->',lw=1.6,color='#385d7a'))
 ax.text(.48,.48,'current experiment: Stage-1 only; no memory injected',ha='center',fontsize=9,color='#8b2d2d');ax.text(.5,.02,'Future Stage-2 path is a proposal, not executed here.',ha='center',fontsize=9);savefig(fig,figdir,'fig8_information_flow')
 # Counts and summary receipts.
 captions={
  'fig1_layer_action_hardware':'Natural-prefix target contrasts; rows are layers 1–32, columns L1–L5; cells are median relative L2 over matching fields/contrasts. No QoR interpretation.',
  'fig2_layer_action_structure':'Controlled-prefix vectors for initialized reduction vs independent diagnostic source and a meaning-preserving rename control. The reduction/independent programs are not equivalent.',
  'fig3_candidate_margin_changes':'Production mean token log-probability top-two margins; only identical candidate sets are compared.',
  'fig4_natural_controlled':'Natural and common R1 directive prefixes; movement remains representational sensitivity.',
  'fig5_structure_target_interaction':'Difference-in-differences of hidden vectors under source×target conditions; descriptive and not physical correctness.',
  'fig6_hardware_token_embeddings':'Trained rows relative to PEFT-preserved resize-time initialization; numerical resource strings are ordinary tokenizer pieces.',
  'fig7_layer_shortlist_evidence':'All-layer descriptive state measures and patch measurements only at predeclared sampled layers.',
  'fig8_information_flow':'Implemented Stage-1 and proposed future GNN-memory path, which was not fused during this experiment.',
  'fig9_activation_patch_margin':'R1 controlled post-attention donor vector patched at the decision position; placebo donor is same-condition state.'}
 write_json(out/'figure_captions.json',captions)
 summary=dict(request_count=len(results),grid_request_count=18,historical_request_count=1,structure_target_request_count=24,layer_pairwise_rows=len(layer_rows),candidate_contrast_rows=len(score_rows),structure_rows=len(structures),interaction_rows=len(interactions),capture_layers=list(range(1,33)),boundaries=['post_self_attention_residual','post_decoder_residual','final_rms_normalized_lm_head_input'],vector_shape=[4096],hidden_state_files=len(tensors),figure_files=sorted(str(p.name) for p in figdir.iterdir()),source_validation_all=all(x['validation']['source'] for x in results.values()),v2_validation_all=all(x['validation']['source_tool_v2'] for x in results.values()))
 write_json(out/'analysis_summary.json',summary)
 print(json.dumps(summary,indent=2))

def savefig(fig,path,name):
 fig.savefig(path/f'{name}.png',dpi=220,bbox_inches='tight');fig.savefig(path/f'{name}.svg',bbox_inches='tight',metadata={'Date': None});plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT);main(p.parse_args().out)
