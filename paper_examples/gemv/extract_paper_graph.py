import networkx as nx
from collections import Counter

SRC = "paper_examples/gemv/gemv_mlir.gexf"
OUT = "paper_examples/gemv/gemv_paper_subgraph.gexf"

G = nx.read_gexf(SRC)
FLOW = {0:"control",1:"ssa/data",2:"call",4:"block",5:"block-connect",
        6:"loop-hierarchy",7:"array-scope",8:"region",9:"memory-view/alias",
        10:"memory-access",11:"loop-carried",12:"proven-dependence",
        13:"memory-uncertainty",200:"pragma/action"}
print("Graph:", type(G).__name__)
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

actions = {}
for n,d in G.nodes(data=True):
    action = str(d.get("action_id", "")).strip()
    if action:
        actions.setdefault(action, []).append((n,d.get("type"),d.get("text"),
            d.get("full_text"),d.get("action_resolution"),d.get("action_source_line")))
print("\n=== ACTION NODES ===")
for action in sorted(actions, key=lambda s:int(s[1:])):
    print("\n", action)
    for row in actions[action]: print("   ", row)

counts = Counter()
for u,v,d in G.edges(data=True):
    try: flow=int(d.get("flow",-1))
    except Exception: flow=-1
    counts[FLOW.get(flow,str(flow))] += 1
print("\n=== EDGE TYPES ===")
for k,v in counts.most_common(): print(f"{k:24s} {v}")

action_nodes={n for n,d in G.nodes(data=True) if str(d.get("action_id","")).strip()}
KEEP_FLOWS={1,6,7,8,9,10,11,12,13,200}
selected=set(action_nodes); frontier=set(action_nodes)
for _ in range(2):
    new=set()
    for u,v,d in G.edges(data=True):
        try: flow=int(d.get("flow",-1))
        except Exception: continue
        if flow not in KEEP_FLOWS: continue
        if u in frontier: new.add(v)
        if v in frontier: new.add(u)
    new -= selected; selected |= new; frontier = new
H=G.subgraph(selected).copy()
print("\nPaper subgraph:")
print("Nodes:", H.number_of_nodes())
print("Edges:", H.number_of_edges())
nx.write_gexf(H, OUT)
print("Wrote:", OUT)
