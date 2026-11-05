"""Render actual RC3 GEMV; overview SSA projections are labelled explicitly."""
import json,subprocess,hashlib
from pathlib import Path
from collections import Counter
import networkx as nx
from GNN_branch import mlir_graph_v2 as v2

OUT=Path(__file__).resolve().parent

def q(s):return json.dumps(str(s),ensure_ascii=False)

def label(n,a):
 kind=a['category']
 if kind=='action_point':return a['action_id']+' | '+a['action_kind']+(' '+a['array_variable'] if a.get('array_variable') else '')
 if kind=='operation':
  s=a.get('text',a.get('full_text',n))
  if a.get('is_loop'):s+=chr(10)+a.get('action_id','')+' : 0..31 (trip count 32)'
  return s
 if kind=='memory_object':return a.get('value_type',n)
 if kind=='ssa_value':return n.split(':',1)[-1]+chr(10)+a.get('value_type','SSA')
 return a.get('text',a.get('name',n))

def render(g,name,overview=False):
 keep={n for n,a in g.nodes(data=True) if not overview or a['category'] in ('operation','memory_object','action_point')}
 lines=['digraph gemv {','graph [rankdir=LR, bgcolor="white", pad="0.3", nodesep="0.25", ranksep="0.6", fontname="Helvetica", labelloc=t, label='+q('GEMV 32 × 32 | structural V2.2 RC3'+(' | operation overview' if overview else ' | complete compact graph'))+'];','node [fontname="Helvetica", fontsize=10, style="rounded,filled", margin="0.12,0.08"];','edge [fontname="Helvetica", fontsize=8, arrowsize=0.6];']
 colors={'operation':'#e5eefb','memory_object':'#dcf3e7','action_point':'#ede1fa','ssa_value':'#f4f6f8','control_region':'#fff0cf','constant_or_feature':'#f4f4f4'}
 for n,a in g.nodes(data=True):
  if n in keep:lines.append(f'{q(n)} [label={q(label(n,a))}, fillcolor={q(colors[a["category"]])}, shape={q("box" if a["category"]!="action_point" else "hexagon")}];')
 seen=set()
 for u,v,a in g.edges(data=True):
  r=a['relation_type']
  if u not in keep or v not in keep:continue
  if overview and r not in ('targets','memory_access','depends_on','next_operation','loop_child'):continue
  key=(u,v,r)
  if key in seen:continue
  seen.add(key)
  color={'targets':'#7851a9','memory_access':'#238858','depends_on':'#d66b18','operand':'#3071ae','defines':'#3071ae'}.get(r,'#a2a5a9')
  suffix=' / '+a['evidence_status'] if r in ('depends_on','alias') else ''
  style='dashed' if r=='targets' or a.get('evidence_status') in ('possible','not_established') else 'solid'
  lines.append(f'{q(u)} -> {q(v)} [label={q(r+suffix)},color={q(color)},style={q(style)}];')
 if overview:
  for value,a in g.nodes(data=True):
   if a['category']!='ssa_value':continue
   producers=[u for u,_,e in g.in_edges(value,data=True) if e['relation_type']=='defines' and u in keep]
   consumers=[v for _,v,e in g.out_edges(value,data=True) if e['relation_type']=='operand' and v in keep]
   for u in producers:
    for v in consumers:
     if (u,v,'SSA') not in seen:
      seen.add((u,v,'SSA'));lines.append(f'{q(u)} -> {q(v)} [label="SSA projection",color="#3071ae"];')
 lines.append('}')
 dot=OUT/(name+'.dot');dot.write_text('\n'.join(lines))
 for kind in ('svg','png','pdf'):
  subprocess.run(['dot','-T'+kind,str(dot),'-o',str(OUT/(name+'.'+kind))],check=True)

def main():
 g=v2.read_graph(OUT/'gemv.canonical.json');receipt=v2.validate(g)
 compact=v2.compact_view(g);v2.write_graph(compact,OUT/'gemv.compact.json');v2.validate(compact)
 export=nx.MultiDiGraph()
 for n,a in g.nodes(data=True):export.add_node(n,**{k:v if isinstance(v,(str,bool,int,float)) else json.dumps(v,sort_keys=True) for k,v in a.items()})
 for u,v,k,a in g.edges(keys=True,data=True):export.add_edge(u,v,key=k,**{k:v if isinstance(v,(str,bool,int,float)) else json.dumps(v,sort_keys=True) for k,v in a.items()})
 nx.write_gexf(export,OUT/'gemv.rc3.gexf')
 render(compact,'gemv.rc3.full');render(compact,'gemv.rc3.overview',True)
 (OUT/'validation.json').write_text(json.dumps({'canonical':receipt,'compact':v2.validate(compact),'graph_identity':g.graph['graph_identity'],'actions':g.graph['action_point_coverage'],'visualization':'Full view includes all compact nodes; parallel identical visual edges are grouped. Overview omits SSA/scope nodes and uses explicitly labelled derived SSA projection edges. Canonical JSON remains authoritative.'},sort_keys=True,indent=2)+'\n')

if __name__=='__main__':main()
