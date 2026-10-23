#!/usr/bin/env python3
"""Deterministic ASPLOS visualization for the production GEMV MLIR graph.

The script reads the already-validated GEXF and emits styled GEXF, SVG, PDF,
PNG, zooms, a composite, and a provenance manifest.  It never creates graph
relations: every retained edge is copied from the production GEXF.
"""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
plt.rcParams["svg.hashsalt"] = "mailohls-gemv-mlir-paper-v1"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper_examples/gemv/gemv_mlir.gexf"
META = ROOT / "gemv/code_to_memory_outputs/meta/kernel_info.txt"
OUT = ROOT / "paper_figures"

FLOW_NAME = {0:"control", 1:"ssa/data", 2:"call", 4:"block",
             5:"block-connect", 6:"loop-hierarchy", 7:"array-scope",
             8:"region", 9:"memory-view/alias", 10:"memory-access",
             11:"loop-carried", 12:"proven-dependence", 13:"memory-uncertainty",
             200:"Stage-2 action routing"}
KEEP_FLOWS = {1, 6, 7, 8, 9, 10, 11, 12, 13, 200}
EDGE_COLOR = {1:"#6B879A", 6:"#777777", 7:"#59A14F", 8:"#777777",
              9:"#59A14F", 10:"#59A14F", 11:"#E07B39", 12:"#E07B39",
              13:"#E07B39", 200:"#222222"}
NODE_COLOR = {"loop":"#2878B5", "array":"#59A14F", "pragma":"#D62728",
              "memory":"#59A14F", "arith":"#B0B4B8", "context":"#D9DDE1"}

def flow(d):
    try: return int(d.get("flow", -1))
    except (TypeError, ValueError): return -1

def action_groups(G):
    groups = defaultdict(list)
    for n, d in G.nodes(data=True):
        a = str(d.get("action_id", "")).strip()
        if a: groups[a].append(str(n))
    return dict(groups)

def anchor_for(G, action, nodes):
    candidates = []
    for n in nodes:
        d = G.nodes[n]
        t, text = d.get("type"), str(d.get("text", ""))
        # Source-resolved affine/scf loop or array_scope is the semantic anchor.
        if d.get("action_resolution") == "source_location" and t == 0:
            candidates.append((0, n))
        elif d.get("action_resolution") == "source_location" and text == "array_scope":
            candidates.append((1, n))
    if candidates: return sorted(candidates)[0][1]
    return sorted(nodes)[0]

def parse_actions():
    actions = {}
    lines = META.read_text().splitlines()
    for line in lines[1:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            actions[p[0]] = {"kind": p[1], "metadata": p[2:]}
    return actions

def classify_node(d, action_anchor=False):
    text = str(d.get("text", "")).lower()
    if action_anchor:
        return "loop" if "for" in text or "loop" in text else "array"
    if int(d.get("type", -1)) == 100: return "pragma"
    if "array" in text or "memref" in text or "memory" in text: return "memory"
    if any(x in text for x in ("arith", "affine.apply", "constant")): return "arith"
    return "context"

def build_reduced(G, groups, anchors):
    # Two semantic hops from the selected action satellites.  Only semantic
    # production relations are traversed; bookkeeping block/control edges are
    # deliberately excluded from expansion.
    selected_actions = {"L1", "L4", "L5"}
    seeds = set(sum((groups[a] for a in selected_actions), []))
    selected = set(seeds); frontier = set(seeds)
    for _ in range(2):
        nxt = set()
        for u, v, d in G.edges(data=True):
            if flow(d) not in KEEP_FLOWS: continue
            if str(u) in frontier: nxt.add(str(v))
            if str(v) in frontier: nxt.add(str(u))
        nxt -= selected; selected |= nxt; frontier = nxt
    # Preserve the complete endpoint pair of any production-proven
    # dependence touching this semantic neighborhood.
    for u, v, d in G.edges(data=True):
        if flow(d) == 12 and (str(u) in selected or str(v) in selected):
            selected.update((str(u), str(v)))
    H = G.subgraph(selected).copy()
    # Keep a compact context: remove isolated non-action bookkeeping nodes,
    # while retaining every node needed by an included semantic edge.
    H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0 and n not in seeds])
    return H, selected_actions

def layout(H, groups, anchors):
    pos = {}
    centers = {"L1": (-3.8, 2.4), "L4": (-1.0, -2.3), "L5": (2.8, -2.3)}
    for a, ns in groups.items():
        if a not in centers: continue
        cx, cy = centers[a]; ns = sorted(set(ns), key=str)
        for i, n in enumerate(ns):
            if n == anchors[a]: pos[n] = (cx, cy); continue
            if len(ns) == 2:
                angle = math.pi / 2
            else:
                angle = 2 * math.pi * i / max(1, len(ns)-1)
            pos[n] = (cx + .72 * math.cos(angle), cy + .58 * math.sin(angle))
    context = sorted((n for n in H if n not in pos), key=str)
    C = nx.Graph()
    C.add_nodes_from(context)
    for u, v in sorted({tuple(sorted((u, v), key=str)) for u, v in H.edges() if u in context and v in context}, key=lambda e:(str(e[0]),str(e[1]))):
        C.add_edge(u, v)
    if context:
        cp = nx.spring_layout(C, seed=1729, iterations=80, scale=2.5, center=(0.1, .15))
        for n in sorted(context, key=str): pos[n] = tuple(float(x) for x in cp[n])
    return pos

def style_graph(H, groups, anchors, pos):
    action_by_node = {n:a for a,ns in groups.items() for n in ns}
    for n, d in H.nodes(data=True):
        a = action_by_node.get(n)
        role = classify_node(d, a in anchors and n == anchors[a])
        d["paper_role"] = role; d["paper_action"] = a or ""
        if a and n == anchors[a]:
            if d.get("text") == "array_scope":
                anchor_label = f"{a}\n{d.get('full_text','').split(' action=')[0]}"
            else:
                anchor_label = f"{a}\n{d.get('text','')}"
        else:
            anchor_label = None
        d["label"] = (anchor_label if anchor_label is not None
                       else (str(d.get('text','')).replace('ARRAY_PARTITION','ARRAY\nPARTITION') if a and int(d.get('type',-1)) == 100
                             else (str(d.get('text','')) if d.get('text') in
                                   {"affine.for","scf.for","memref.load","memref.store","arith.addf","arith.mulf"}
                                   else "")))
        d["viz"] = {"position": {"x": float(pos[n][0]*120), "y": float(pos[n][1]*120), "z": 0.0},
                    "color": {"r": int(NODE_COLOR[role][1:3],16), "g": int(NODE_COLOR[role][3:5],16), "b": int(NODE_COLOR[role][5:7],16)},
                    "size": 24.0 if n in anchors.values() else (15.0 if a else 8.0)}
        d["paper_label"] = d["label"]
    for u,v,d in H.edges(data=True):
        f=flow(d); d["paper_flow_name"]=FLOW_NAME.get(f,str(f)); d["viz"]={"color": {"r":int(EDGE_COLOR.get(f,"#AAB0B5")[1:3],16),"g":int(EDGE_COLOR.get(f,"#AAB0B5")[3:5],16),"b":int(EDGE_COLOR.get(f,"#AAB0B5")[5:7],16)},"thickness": 2.6 if f==200 else 1.0}

def write_manifest(G, H, groups, anchors, selected_actions, pos):
    nodes=[]
    for n,d in H.nodes(data=True):
        reason = "selected action satellite" if d.get("paper_action") else "semantic context within two production relation hops"
        nodes.append({"id":str(n),"type":d.get("type"),"text":d.get("text"),"full_text":d.get("full_text"),
                      "action_id":d.get("paper_action",""),"role":d.get("paper_role"),"label":d.get("paper_label",""),"reason":reason})
    edges=[]
    for u,v,d in H.edges(data=True):
        edges.append({"source":str(u),"target":str(v),"flow":flow(d),"relation":FLOW_NAME.get(flow(d),str(flow(d))),"production_edge_id":d.get("id")})
    all_actions={}
    for a,ns in groups.items():
        an=anchors[a]; ad=G.nodes[an]
        all_actions[a]={"kind":parse_actions().get(a,{}).get("kind"),"source_line":ad.get("action_source_line"),
                       "anchor_node":an,"anchor_type":ad.get("type"),"anchor_text":ad.get("text"),"anchor_full_text":ad.get("full_text")}
    payload={"source_gexf":str(SRC),"source_sha256":hashlib.sha256(SRC.read_bytes()).hexdigest(),
             "selected_actions":sorted(selected_actions),"all_actions":all_actions,
             "production_flow_names":FLOW_NAME,"nodes":nodes,"edges":edges,
             "layout":{"algorithm":"fixed semantic centers + seeded local spring context","seed":1729}}
    (OUT/"gemv_mlir_reduced.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")

def save_gexf(H, name): nx.write_gexf(H, OUT/name, prettyprint=True, version="1.2draft")

def canonical_graph(H):
    """Make node/edge and attribute insertion order independent of set order."""
    C = nx.MultiDiGraph()
    C.graph.update(H.graph)
    for n in sorted(H.nodes(), key=str):
        d = H.nodes[n]
        C.add_node(n, **{k:d[k] for k in sorted(d)})
    for u,v,k,d in sorted(H.edges(keys=True, data=True), key=lambda x:(str(x[0]),str(x[1]),str(x[2]),str(x[3].get('id','')))):
        C.add_edge(u,v,key=k,**{a:d[a] for a in sorted(d)})
    return C

def draw(H, pos, path, title=None, highlight=None):
    fig, ax = plt.subplots(figsize=(11,7), dpi=220); ax.set_axis_off()
    for u,v,d in H.edges(data=True):
        f=flow(d); col=EDGE_COLOR.get(f,"#AAB0B5"); style="dashed" if f==200 else "solid"; width=2.0 if f==200 else (1.2 if f in {7,10,12,13} else .75)
        nx.draw_networkx_edges(H,pos,edgelist=[(u,v)],ax=ax,edge_color=col,width=width,style=style,arrows=False,alpha=.82)
    node_groups=defaultdict(list)
    for n,d in H.nodes(data=True): node_groups[d.get("paper_role","context")].append(n)
    for role,ns in node_groups.items():
        nx.draw_networkx_nodes(H,pos,nodelist=ns,node_color=NODE_COLOR[role],node_size=[(650 if n in (highlight or set()) else (390 if d_is_anchor(H,n) else 110)) for n in ns],edgecolors="#263238",linewidths=.55,ax=ax)
    labels={n:d.get("paper_label","") for n,d in H.nodes(data=True) if d.get("paper_label")}
    nx.draw_networkx_labels(H,pos,labels=labels,font_size=6,font_weight="bold",ax=ax,
                            bbox=dict(facecolor="white",edgecolor="none",alpha=.72,pad=.2))
    handles=[Line2D([0],[0],marker='o',color='w',label='action / semantic anchor',markerfacecolor='#D62728',markeredgecolor='#263238',markersize=8),Line2D([0],[0],marker='o',color='w',label='loop structure',markerfacecolor='#2878B5',markersize=8),Line2D([0],[0],marker='o',color='w',label='memory / array',markerfacecolor='#59A14F',markersize=8),Line2D([0],[0],color='#6B879A',label='SSA / dataflow'),Line2D([0],[0],color='#E07B39',label='dependence'),Line2D([0],[0],color='#222222',linestyle='--',label='Stage-2 action routing')]
    ax.legend(handles=handles,loc='upper right',fontsize=7,frameon=True)
    if title: ax.set_title(title,fontsize=12,weight='bold')
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)

def d_is_anchor(H,n): return bool(H.nodes[n].get("paper_label","") and H.nodes[n].get("paper_action") and int(H.nodes[n].get("type",-1)) in {0,104})

def main():
    OUT.mkdir(exist_ok=True); G=nx.read_gexf(SRC); groups=action_groups(G); anchors={a:anchor_for(G,a,ns) for a,ns in groups.items()}
    H,selected=build_reduced(G,groups,anchors); pos=layout(H,groups,anchors); style_graph(H,groups,anchors,pos); H=canonical_graph(H); write_manifest(G,H,groups,anchors,selected,pos)
    save_gexf(H,"gemv_mlir_reduced.gexf"); save_gexf(H,"gemv_mlir_paper.gexf")
    draw(H,pos,OUT/"gemv_mlir_paper.svg","Production SALT-HLS structured-MLIR GEMV graph",{anchors[a] for a in selected})
    draw(H,pos,OUT/"gemv_mlir_paper.pdf","Production SALT-HLS structured-MLIR GEMV graph",{anchors[a] for a in selected})
    draw(H,pos,OUT/"gemv_mlir_paper.png","Production SALT-HLS structured-MLIR GEMV graph",{anchors[a] for a in selected})
    # Action zooms use the action satellite plus its immediate production neighbors.
    for a, fn in (("L4","gemv_mlir_zoom_L4"),("L1","gemv_mlir_zoom_L1")):
        ns=set(groups[a]);
        for u,v,d in H.edges(data=True):
            if flow(d) in KEEP_FLOWS and (u in ns or v in ns): ns.update((u,v))
        Z=H.subgraph(ns).copy(); zp={n:((pos[n][0]-pos[anchors[a]][0])*1.5,(pos[n][1]-pos[anchors[a]][1])*1.5) for n in Z}
        draw(Z,zp,OUT/(fn+".svg"),f"GEMV {a} action neighborhood",{anchors[a]}); draw(Z,zp,OUT/(fn+".pdf"),f"GEMV {a} action neighborhood",{anchors[a]})
    # Composite: overview left, two action zooms right.
    fig=plt.figure(figsize=(14,8),dpi=220); ax0=fig.add_axes([0.02,0.08,0.60,0.84]); ax1=fig.add_axes([0.66,0.53,0.32,0.39]); ax2=fig.add_axes([0.66,0.08,0.32,0.39])
    for ax,Z,Zp,title,hi in [(ax0,H,pos,"Overview",{anchors[a] for a in selected}), (ax1,H.subgraph(set(groups['L4'])|set(n for n in H if any(flow(d) in KEEP_FLOWS and (u in groups['L4'] or v in groups['L4']) for u,v,d in H.edges(data=True)))).copy(),pos,"L4 outer loop",{anchors['L4']}), (ax2,H.subgraph(set(groups['L1'])|set(n for n in H if any(flow(d) in KEEP_FLOWS and (u in groups['L1'] or v in groups['L1']) for u,v,d in H.edges(data=True)))).copy(),pos,"L1 array partition",{anchors['L1']})]:
        ax.set_axis_off()
        for u,v,d in Z.edges(data=True): nx.draw_networkx_edges(Z,Zp,edgelist=[(u,v)],ax=ax,edge_color=EDGE_COLOR.get(flow(d),'#AAB0B5'),width=2 if flow(d)==200 else 1,style='dashed' if flow(d)==200 else 'solid',arrows=False)
        for role in NODE_COLOR:
            ns=[n for n,d in Z.nodes(data=True) if d.get('paper_role')==role]
            if ns: nx.draw_networkx_nodes(Z,Zp,nodelist=ns,node_color=NODE_COLOR[role],node_size=[500 if n in hi else 120 for n in ns],edgecolors='#263238',linewidths=.5,ax=ax)
        nx.draw_networkx_labels(Z,Zp,labels={n:d.get('paper_label','') for n,d in Z.nodes(data=True) if d.get('paper_label')},font_size=7,ax=ax); ax.set_title(title,fontsize=10,weight='bold')
    fig.savefig(OUT/"gemv_mlir_composite.svg",bbox_inches='tight'); fig.savefig(OUT/"gemv_mlir_composite.pdf",bbox_inches='tight'); plt.close(fig)
    print("Wrote deterministic GEMV paper visualization to",OUT)

if __name__ == '__main__': main()
