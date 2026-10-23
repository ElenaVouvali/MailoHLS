# graph_gen_deterministic_memory.py
#
# Wrapper around graph_gen_deterministic that ALSO emits memory metadata sidecars,
# without changing the generated GEXF graphs (HARP GNN encoder input stays identical).
#
# Run:
#   PYTHONHASHSEED=0 python -m graph_gen_deterministic_memory --all
# Or metadata only (if hierarchy graphs already exist):
#   PYTHONHASHSEED=0 python -m graph_gen_deterministic_memory --meta-only
#
# Outputs:
#   <root>/harp/processed/memory_meta_v1/<kernel>_memory_meta.json
#   <root>/harp/processed/memory_meta_v1/manifest.json

import os
import re
import json
import hashlib
import argparse
from collections import defaultdict, OrderedDict, Counter 
from typing import Any, Dict, List, Tuple, Optional, Set, DefaultDict

import networkx as nx

# Import your deterministic pipeline as-is.
import graph_gen_deterministic as ggd

# -----------------------------
# Determinism guard
# -----------------------------

def require_pythonhashseed() -> None:
    seed = os.environ.get("PYTHONHASHSEED", "")
    if seed == "":
        raise RuntimeError(
            "Determinism requires PYTHONHASHSEED to be set before Python starts.\n"
            "Run like:\n"
            "  PYTHONHASHSEED=0 python -m graph_gen_deterministic_memory --all\n"
        )

# -----------------------------
# Paths (match your project)
# -----------------------------

TYPE_GRAPH = getattr(ggd, "type_graph", "harp")  # should be 'harp'
ROOT = ggd.get_root_path()

BASE_DATASET_DIR = "/home/elvouvali/Data4LLMPrompting/ApplicationDataset_2"

# The hierarchy output folder produced by graph_gen_deterministic.py
HIER_GEXF_DIR = os.path.join(
    ROOT,
    f"{TYPE_GRAPH}/processed/extended-pseudo-block-connected-hierarchy-2/"
)

# Sidecar metadata output folder
META_DIR = os.path.join(
    ROOT,
    f"{TYPE_GRAPH}/processed/structural_memory_meta/"
)

# -----------------------------
# Stable helpers
# -----------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def as_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def get_full_text(ndata: Dict[str, Any]) -> str:
    # Post-processed graphs usually have 'full_text'.
    if "full_text" in ndata and ndata["full_text"] is not None:
        return str(ndata["full_text"])
    # Some graphs store stringified dict in 'features'
    feat = ndata.get("features")
    if isinstance(feat, dict):
        ft = feat.get("full_text")
        if isinstance(ft, list) and ft:
            return str(ft[0])
    if isinstance(feat, str):
        # Conservative parse: attempt literal_eval-like behavior without importing ast.
        # If it fails, ignore.
        try:
            obj = eval(feat, {"__builtins__": {}})
            if isinstance(obj, dict):
                ft = obj.get("full_text")
                if isinstance(ft, list) and ft:
                    return str(ft[0])
        except Exception:
            pass
    return ""

def node_sort_key(G: nx.Graph, n: Any) -> Tuple:
    d = G.nodes[n]
    # Cast to stable comparable types; node id can be str or int.
    nid = as_int(n, default=10**18) if str(n).isdigit() else str(n)
    return (
        as_int(d.get("function", -1)),
        as_int(d.get("block", -1)),
        as_int(d.get("type", -1)),
        str(d.get("text", "")),
        get_full_text(d),
        nid,
    )

def iter_edges_any(G: nx.Graph):
    """
    Yield edges as (u, v, k, d) for MultiDiGraph, or (u, v, 0, d) for DiGraph.
    """
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, k, d in G.edges(keys=True, data=True):
            yield u, v, k, d
    else:
        for u, v, d in G.edges(data=True):
            yield u, v, 0, d

def iter_in_edges_any(G: nx.Graph, n: Any):
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, k, d in G.in_edges(n, keys=True, data=True):
            yield u, v, k, d
    else:
        for u, v, d in G.in_edges(n, data=True):
            yield u, v, 0, d

def iter_out_edges_any(G: nx.Graph, n: Any):
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, k, d in G.out_edges(n, keys=True, data=True):
            yield u, v, k, d
    else:
        for u, v, d in G.out_edges(n, data=True):
            yield u, v, 0, d

def edge_flow(d: Dict[str, Any]) -> int:
    # In GEXF, attributes may be strings.
    return as_int(d.get("flow", -1))

def edge_pos(d: Dict[str, Any]) -> int:
    return as_int(d.get("position", -1))

# -----------------------------
# Placeholder + pragma parsing
# -----------------------------

RE_AUTO = re.compile(r"auto\{([^}]+)\}")
RE_LABEL = re.compile(r"(L\d+)", re.IGNORECASE)
RE_AP_VAR = re.compile(r"\bvariable\s*=\s*([A-Za-z_]\w*)\b")
LLVM_DEF_RE = re.compile(r'^\s*(%[\w.]+)\s*=\s*(.+)$')
LLVM_GEP_DEF_RE = re.compile(r'^\s*(%[\w.]+)\s*=\s*getelementptr\b')
LLVM_COND_BR_RE = re.compile(r'\bbr\s+i1\b|\bswitch\b')


def build_ptr_root_map(ll_lines, segs):
    """
    Map SSA pointer values -> their root alloca/arg pointer, for simple cases:
      %p = load double*, double** %m2.addr   => root(%p) = %m2.addr
    """
    root = {}

    for (s, e) in segs:
        for i in range(s, e+1):
            line = ll_lines[i].strip()
            m = LLVM_DEF_RE.match(line)
            if not m:
                continue
            dst, rhs = m.group(1), m.group(2)

            # match: %dst = load <ty*>, <ty**> %src, ...
            lm = re.search(r'^\s*load\s+[^,]*\*,\s*[^,]*\*\*\s*(%[\w.]+)\b', rhs)
            if lm:
                src = lm.group(1)
                root[dst] = src

    # path-compress
    def find(x):
        seen = set()
        while x in root and x not in seen:
            seen.add(x)
            x = root[x]
        return x
    for k in list(root.keys()):
        root[k] = find(root[k])
    return root


def build_gep_result_to_array_map(
    ll_lines: List[str],
    segments: List[Tuple[int,int]],
    ptr_root_map: Dict[str,str],
) -> Dict[str,str]:
    """
    Map GEP result SSA (%arrayidx25) -> array name ('prod') or root ptr ('%prod.addr')
    Deterministic, purely local to [segments].
    """
    out: Dict[str,str] = {}
    for (s,e) in segments:
        for i in range(s, e+1):
            line = ll_lines[i].strip()
            m = LLVM_GEP_DEF_RE.match(line)
            if not m:
                continue
            gep_dst = m.group(1)  # e.g. %arrayidx25
            base = _llvm_extract_gep_base(line)  # e.g. %20
            if not base:
                continue
            root = ptr_root_map.get(base, base)  # e.g. %prod.addr
            arr = array_name_from_root(root)     # e.g. 'prod'
            out[gep_dst] = arr if arr else root
    return out


def detect_rmw(load_ptrs: Set[str], store_ptrs: Set[str]) -> List[str]:
    return sorted(set(load_ptrs) & set(store_ptrs))

def parse_placeholder_key(pragma_line: str) -> Optional[str]:
    m = RE_AUTO.search(pragma_line)
    if not m:
        return None
    return m.group(1).strip()


def parse_loop_label_from_placeholder(ph: str) -> Optional[str]:
    if not ph:
        return None
    m = RE_LABEL.search(ph)
    return m.group(1).upper() if m else None

def parse_array_var(pragma_line: str) -> Optional[str]:
    m = RE_AP_VAR.search(pragma_line)
    return m.group(1) if m else None

def pragma_kind_from_node(ndata: Dict[str, Any]) -> str:
    # Your pragma nodes typically have text = "PIPELINE" / "UNROLL" / "ARRAY_PARTITION"
    t = str(ndata.get("text", "")).upper()
    if t in {"PIPELINE", "UNROLL", "ARRAY_PARTITION"}:
        return t
    # Fallback: infer from full_text
    ft = get_full_text(ndata).upper()
    if "PIPELINE" in ft:
        return "PIPELINE"
    if "UNROLL" in ft:
        return "UNROLL"
    if "ARRAY_PARTITION" in ft:
        return "ARRAY_PARTITION"
    return "UNKNOWN"

# -----------------------------
# LLVM opcode + memory-ish features (heuristics)
# -----------------------------

LLVM_LABEL_RE = re.compile(r'^\s*([A-Za-z0-9_.]+):')  # e.g. "for.cond14:"
LLVM_GEP_RE   = re.compile(r'\bgetelementptr\b')
LLVM_LOAD_RE  = re.compile(r'^\s*%[\w.]+\s*=\s*load\b|\bload\b')
LLVM_STORE_RE = re.compile(r'\bstore\b')
LLVM_BR_TO_RE = re.compile(r'\bbr\s+label\s+%([A-Za-z0-9_.]+)\b')

def _prev_label_line(ll_lines: List[str], start_idx: int) -> int:
    i = start_idx
    while i >= 0:
        if LLVM_LABEL_RE.match(ll_lines[i]):
            return i
        i -= 1
    return -1

def _extend_child_start_to_preheader(ll_lines: List[str], child_start: int, child_block: str) -> int:
    """
    If the immediately preceding label-block contains 'br label %child_block',
    treat it as the child's preheader and include it in the hole.
    """
    prev_lbl = _prev_label_line(ll_lines, child_start - 1)
    if prev_lbl < 0:
        return child_start

    for i in range(prev_lbl, child_start):
        m = LLVM_BR_TO_RE.search(ll_lines[i])
        if m and m.group(1) == child_block:
            return prev_lbl
    return child_start


def _llvm_extract_ptr_from_load(line: str) -> Optional[str]:
    # matches: load float, float* %p   OR load float, ptr %p
    m = re.search(r'\bload\b[^,]*,\s*(?:ptr|[^,]*\*+)\s*([^,\s]+)', line)
    return m.group(1) if m else None

def _llvm_extract_ptr_from_store(line: str) -> Optional[str]:
    # matches: store float %v, float* %p  OR store float %v, ptr %p
    m = re.search(r'\bstore\b[^,]*,\s*(?:ptr|[^,]*\*+)\s*([^,\s]+)', line)
    return m.group(1) if m else None

def _llvm_extract_gep_base(line: str) -> Optional[str]:
    # matches: ..., float* %base, ...  OR ..., ptr %base, ...
    m = re.search(r'\bgetelementptr\b.*?,\s*(?:ptr|[^,]*\*+)\s*([^,\s]+)\s*,', line)
    return m.group(1) if m else None




_FIRST_LOAD_RE  = re.compile(r"^\s*%[\w\.\-]+\s*=\s*load\b.*?\*\s*(%[\w\.\-]+)\b")
_FIRST_STORE_RE = re.compile(r"^\s*store\b.*?\*\s*(%[\w\.\-]+)\b")


def _iter_region_lines(ll_lines: List[str], segments: List[Tuple[int, int]]) -> List[str]:
    """
    Flatten (inclusive) segments into a list of lines in order.
    Assumes segments are 0-based inclusive indices (your code already uses that).
    """
    out: List[str] = []
    n = len(ll_lines)
    for (s, e) in segments or []:
        s = max(0, int(s))
        e = min(n - 1, int(e))
        if s <= e:
            out.extend(ll_lines[s:e + 1])
    return out


def classify_scalar_rmw_first_access(
    ll_lines: List[str],
    segments: List[Tuple[int, int]],
    cand_scalars: List[str],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    For each candidate scalar pointer p:
      - if first occurrence in region is a load  -> carried scalar
      - if first occurrence is a store          -> temp scalar
    """
    cand: Set[str] = set(cand_scalars or [])
    first_op: Dict[str, str] = {}  # p -> "load"|"store"

    if not cand or not segments:
        return [], [], first_op

    for line in _iter_region_lines(ll_lines, segments):
        if ("load" not in line) and ("store" not in line):
            continue

        m = _FIRST_LOAD_RE.match(line)
        if m:
            p = m.group(1)
            if p in cand and p not in first_op:
                first_op[p] = "load"
            continue

        m = _FIRST_STORE_RE.match(line)
        if m:
            p = m.group(1)
            if p in cand and p not in first_op:
                first_op[p] = "store"
            continue

    carried = sorted([p for p, op in first_op.items() if op == "load"])
    temps   = sorted([p for p, op in first_op.items() if op == "store"])
    return carried, temps, first_op


def _llvm_opcode(line: str):
    # Quick opcode bucket for counts (your current summary has br/icmp/load)
    s = line.strip()
    if not s or s.endswith(':') or s.startswith(';'):
        return None
    # handle "  %x = opcode ..." and "  opcode ..."
    if '=' in s:
        rhs = s.split('=', 1)[1].strip()
    else:
        rhs = s
    op = rhs.split(None, 1)[0]
    # normalize common punctuation
    return op


def _loop_children_spans(ll_lines: List[str], for_l: str, for_blocks_info: dict):
    spans = []
    for child in for_blocks_info.get(for_l, {}).get("children", []):
        cinfo = for_blocks_info.get(child)
        if not cinfo:
            continue

        s = int(cinfo.get("line_num", -1))
        e = int(cinfo.get("end", (None, None))[0] or -1)
        if s < 0 or e < 0 or e < s:
            continue

        child_block = str(child).split(":", 1)[0]   # "for.cond14:110" -> "for.cond14"
        s2 = _extend_child_start_to_preheader(ll_lines, s, child_block)

        spans.append((s2, e))

    spans.sort()
    return spans



def _subtract_spans(base_span: Tuple[int,int], holes: List[Tuple[int,int]]):
    """
    Return a list of disjoint segments within base_span excluding holes.
    base_span and holes are inclusive [s,e].
    """
    s0, e0 = base_span
    segs = [(s0, e0)]
    for hs, he in holes:
        new_segs = []
        for s, e in segs:
            if he < s or hs > e:
                new_segs.append((s, e))
                continue
            if hs > s:
                new_segs.append((s, hs - 1))
            if he < e:
                new_segs.append((he + 1, e))
        segs = new_segs
        if not segs:
            break
    return segs


SSA_RE = re.compile(r'(%[\w.]+)')


def find_induction_ptr(ll_lines: List[str], for_l: str, for_blocks_info: Dict, window: int = 30) -> Optional[str]:
    info = for_blocks_info.get(for_l, {})
    start = int(info.get("line_num", -1))
    if start < 0:
        return None
    end = min(len(ll_lines)-1, start + window)
    for i in range(start, end+1):
        line = ll_lines[i].strip()
        m = re.search(r'^\s*(%[\w.\-]+)\s*=\s*load\s+i(32|64),\s+i\2\*\s+(%[\w.\-]+)\b', line)
        if m:
            return m.group(3)  
    return None


def classify_pattern_stride(idxs: List[str], iv_ssa: Optional[str], defs: Dict[str,str], induction_ptr: Optional[str]) -> Tuple[str, Optional[int]]:
    if not idxs or not iv_ssa:
        return ("unknown", None)

    memo: Dict[str, Optional[int]] = {}
    ks_raw = [coeff_in_iv(x, iv_ssa, defs, memo, induction_ptr) for x in idxs]

    if any(k is None for k in ks_raw):
        return ("unknown", None)

    ks = [int(k) for k in ks_raw]
    if all(k == 0 for k in ks):
        return ("invariant", 0)

    uniq = sorted(set(ks))
    if len(uniq) == 1:
        k = uniq[0]
        return ("linear", abs(k)) if abs(k) == 1 else ("strided", abs(k))
    return ("mixed", None)



def space_of_array(name: str) -> str:
    if not name:
        return "unknown"
    return "local" if name.startswith("local_") else "global"


def detect_scalar_recurrences(
    ll_lines: List[str],
    segments: List[Tuple[int,int]],
    scalar_ptrs: List[str],
) -> List[Dict[str, Any]]:
    """
    Finds stores where the stored SSA depends on a load from the same pointer.
    Classifies reduction if store value is produced by add/fadd that uses the loaded SSA.
    """
    if not segments or not scalar_ptrs:
        return []

    scalar_set = set(scalar_ptrs)
    defs = build_defs_map(ll_lines, segments)
    deps = build_value_deps(ll_lines, segments)

    loads_by_ptr: Dict[str, Set[str]] = defaultdict(set)
    stores: List[Tuple[str, Optional[str]]] = []

    for line in _iter_region_lines(ll_lines, segments):
        s = line.strip()
        m = LLVM_DEF_RE.match(s)
        if m and m.group(2).strip().startswith("load"):
            dst = m.group(1)
            ptr = _llvm_extract_ptr_from_load(s)
            if ptr in scalar_set:
                loads_by_ptr[ptr].add(dst)
            continue

        if " store " in f" {s} ":
            ptr = _llvm_extract_ptr_from_store(s)
            val = _llvm_extract_store_value(s)
            if ptr in scalar_set:
                stores.append((ptr, val))
            continue

    recs: List[Dict[str, Any]] = []
    for ptr, val in stores:
        if not val or not val.startswith("%"):
            continue
        load_ssas = loads_by_ptr.get(ptr, set())
        if not load_ssas:
            continue
        dset = deps.get(val, set())
        if not any(l in dset for l in load_ssas):
            continue

        rhs = defs.get(val, "")
        op = rhs.split(None, 1)[0] if rhs else "unknown"

        # reduction if (f)add uses the loaded SSA directly
        is_reduction = False
        if op in {"fadd", "add"}:
            ops = set(SSA_RE.findall(rhs))
            if ops & load_ssas:
                is_reduction = True

        recs.append({
            "var": ptr.lstrip("%"),
            "ptr": ptr,
            "kind": "reduction" if is_reduction else "recurrence",
            "op": op,
        })

    # deterministic ordering
    recs.sort(key=lambda r: (r["kind"], r["var"], r["op"]))
    return recs



# Add near your helpers (next to other LLVM extractors)
def _llvm_extract_store_value(line: str) -> Optional[str]:
    # Example:
    #   store i32 %mul10, i32* %i_row, align 4
    #   store i32 0, i32* %i, align 4   (no SSA value)
    m = re.search(r'\bstore\b\s+[^ ]+\s+([^,]+)\s*,\s*[^,]*\*+\s*[^,\s]+', line)
    if not m:
        return None
    v = m.group(1).strip()
    return v if v.startswith("%") else None


def build_value_deps(ll_lines: List[str], segments: List[Tuple[int,int]]) -> Dict[str, Set[str]]:
    """
    deps[%x] = set of source pointers (allocas like %k,%j,%i,%kk,%jj, and any SSA derived from them)
    Fix: propagate deps through allocas by modeling "last stored value deps" for each alloca pointer.
    Deterministic, local to [segments].
    """
    deps: Dict[str, Set[str]] = defaultdict(set)

    # Key idea: for allocas (like %i_row, %k_row, %temp_x, %mul, and loop vars %i/%k/%j/...),
    # remember the deps of the *last SSA value stored* into that alloca pointer.
    alloca_last_deps: Dict[str, Set[str]] = {}

    for (s, e) in segments:
        for i in range(s, e + 1):
            line = ll_lines[i].strip()
            if not line or line.startswith(";") or LLVM_LABEL_RE.match(line):
                continue

            # ---- Handle STORE (critical for fixing L3) ----
            if " store " in f" {line} ":
                ptr = _llvm_extract_ptr_from_store(line)    # destination pointer (e.g. %i_row)
                val = _llvm_extract_store_value(line)       # SSA value stored (e.g. %mul10) or None for constants
                if ptr and ptr.startswith("%"):
                    if val and val.startswith("%"):
                        # deps of the stored value + the value SSA itself (optional but helps)
                        alloca_last_deps[ptr] = set(deps.get(val, set())) | {val}
                    else:
                        # constant store: kill value-deps (address still depends on ptr itself)
                        alloca_last_deps[ptr] = set()
                continue

            # ---- Handle SSA definitions ----
            m = LLVM_DEF_RE.match(line)
            if not m:
                continue

            dst, rhs = m.group(1), m.group(2).strip()

            # LOAD: %dst = load ..., ...* %ptr
            if rhs.startswith("load"):
                ptr = _llvm_extract_ptr_from_load(line)
                if ptr and ptr.startswith("%"):
                    # If this is a scalar alloca load (e.g. %21 = load i32, i32* %i_row),
                    # use the deps of the last stored value into that alloca.
                    if ptr in alloca_last_deps:
                        deps[dst] |= alloca_last_deps[ptr]
                        deps[dst].add(ptr)  # keep pointer identity if you want
                    else:
                        # fallback: pointer identity + whatever deps it has
                        deps[dst].add(ptr)
                        deps[dst] |= deps.get(ptr, set())
                continue

            # GEP: %dst = getelementptr ..., <ty>* %base, i64 %idx ...
            if rhs.startswith("getelementptr"):
                ops = SSA_RE.findall(rhs)
                # union deps of operands + operands themselves
                for op in ops:
                    deps[dst] |= deps.get(op, set())
                    deps[dst].add(op)
                continue

            # Generic SSA-propagation: deps[dst] = union deps[operands] + operands
            ops = SSA_RE.findall(rhs)
            for op in ops:
                deps[dst] |= deps.get(op, set())
                deps[dst].add(op)

    return deps



def _llvm_extract_gep_index_tok(line: str) -> Optional[str]:
    pairs = re.findall(r'\bi(32|64)\s+([^,\s]+)', line)
    if not pairs:
        return None
    return pairs[-1][1].strip()   # may be %idx or 0 or 5


def _llvm_extract_gep_index_ssa(line: str) -> Optional[str]: 
    pairs = re.findall(r'\bi(32|64)\s+([^,\s]+)', line) 
    if not pairs: 
        return None 
    x = pairs[-1][1].strip() 
    return x if x.startswith("%") else None


def array_name_from_root(root_ptr: str) -> Optional[str]:
    """
    Only convert nice roots like %searchSpace.addr -> searchSpace.
    For anonymous SSA like %1 keep it as None to avoid garbage names.
    """
    if not root_ptr:
        return None
    x = root_ptr.lstrip("%")
    if x.endswith(".addr"):
        x = x[:-5]
        return x
    # If it's an SSA temp (%1, %20, %idxprom) don't pretend it's a source array name.
    if x.isdigit():
        return None
    # Heuristic: keep names that look like identifiers (searchSpace, local_distance, etc.)
    if re.match(r'^[A-Za-z_]\w*$', x):
        return x
    return None



def build_gep_maps(
    ll_lines: List[str],
    segments: List[Tuple[int, int]],
    ptr_root_map: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      gep_result_to_array:     %gep -> 'm1'/'m2'/'prod' or '%root.addr'
      gep_result_to_index_ssa: %gep -> %idxprom-like SSA (or None if not found)
    """
    gep_result_to_array: Dict[str, str] = {}
    gep_result_to_index: Dict[str, str] = {}

    for (s, e) in segments:
        for i in range(s, e + 1):
            line = ll_lines[i].strip()
            m = LLVM_GEP_DEF_RE.match(line)
            if not m:
                continue
            gep_dst = m.group(1)
            base = _llvm_extract_gep_base(line)
            if not base:
                continue

            root = ptr_root_map.get(base, base)
            arr = array_name_from_root(root)
            gep_result_to_array[gep_dst] = arr if arr else root

            idx_tok = _llvm_extract_gep_index_tok(line)   # <- token, may be "%x" OR "0"
            if idx_tok is not None:
                gep_result_to_index[gep_dst] = idx_tok

    return dict(sorted(gep_result_to_array.items())), dict(sorted(gep_result_to_index.items()))


LLVM_DEFINE_RE = re.compile(r'^\s*define\b.*@([A-Za-z0-9_.$]+)\s*\((.*)\)\s*(?:#[0-9]+\s*)?\{')
LLVM_ALLOCA_RE = re.compile(r'^\s*(%[\w\.\-]+)\s*=\s*alloca\s+(.+?)(?:,|\s+align|\s*$)')

def llvm_elt_bits(ty: str) -> Optional[int]:
    t = ty.strip()
    if t == "float":
        return 32
    if t == "double":
        return 64
    if t == "half":
        return 16
    m = re.match(r'^i(\d+)$', t)
    if m:
        return int(m.group(1))
    return None

def parse_llvm_array_type(ty: str) -> Tuple[List[int], str]:
    """
    Parse LLVM array type like: [1024 x float] or [2 x [512 x float]]
    Returns (dims, elem_ty).
    """
    s = ty.strip()
    dims: List[int] = []

    while s.startswith('['):
        # find matching outermost ']'
        depth = 0
        end = -1
        for i, ch in enumerate(s):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            break

        inner = s[1:end].strip()   # "1024 x float" or "2 x [512 x float]"
        rest = s[end+1:].strip()

        m = re.match(r'^(\d+)\s+x\s+(.+)$', inner)
        if not m:
            break

        dims.append(int(m.group(1)))
        s = m.group(2).strip()     # continue parsing nested arrays

        # if there is trailing stuff after ']' (shouldn't be for pure type), ignore it
        if rest:
            # sometimes ty includes extra tokens; we stop here safely
            pass

    elem_ty = s.strip()
    # drop pointer stars if alloca captured weirdly
    elem_ty = elem_ty.rstrip('*').strip()
    return dims, elem_ty


def parse_function_args(ll_lines: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Extract pointer args as global arrays from the *entry* function.
    Heuristic: prefer @workload if present, otherwise use the last `define` in the file.
    """
    # 1) collect all function signatures in-order
    func_args_blob = OrderedDict()  # fname -> arg_blob
    for line in ll_lines:
        m = LLVM_DEFINE_RE.match(line)
        if not m:
            continue
        fname = m.group(1)
        arg_blob = m.group(2)
        func_args_blob[fname] = arg_blob

    if not func_args_blob:
        return {}

    # 2) choose entry function
    entry = "workload" if "workload" in func_args_blob else next(reversed(func_args_blob.keys()))
    arg_blob = func_args_blob[entry]

    out: Dict[str, Dict[str, Any]] = {}

    # 3) parse args of that function only
    args = [a.strip() for a in arg_blob.split(",") if a.strip()]
    for a in args:
        names = SSA_RE.findall(a)
        if not names:
            continue
        name = names[-1].lstrip("%")

        pre = a[:a.rfind(names[-1])].strip()
        toks = [
            t for t in pre.split()
            if not re.match(r'^(noalias|nocapture|readonly|writeonly|nonnull|noundef|align|dereferenceable|dereferenceable_or_null)$', t)
        ]
        toks = [t for t in toks if not t.isdigit()]

        ty = toks[-1] if toks else ""
        if ty.endswith("*"):
            base = ty[:-1].strip()
            bits = llvm_elt_bits(base)
            out[name] = {"storage": "global_ptr", "elem_type": base or None, "elem_bits": bits, "dims": None, "bytes_est": None}
        elif ty == "ptr":
            out[name] = {"storage": "global_ptr", "elem_type": None, "elem_bits": None, "dims": None, "bytes_est": None}

    return out


def parse_allocas(ll_lines: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Extract allocas as local arrays/buffers.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for line in ll_lines:
        m = LLVM_ALLOCA_RE.match(line.strip())
        if not m:
            continue
        dst = m.group(1).lstrip("%")
        ty = m.group(2).strip()

        # array alloca
        if ty.startswith('['):
            dims, elem_ty = parse_llvm_array_type(ty)
            bits = llvm_elt_bits(elem_ty)
            total_elems = 1
            for d in dims:
                total_elems *= int(d)
            bytes_est = (total_elems * bits // 8) if (bits is not None) else None

            out[dst] = {
                "storage": "local_alloca",
                "elem_type": elem_ty,
                "elem_bits": bits,
                "dims": dims,
                "bytes_est": bytes_est,
            }
        else:
            # scalar alloca (optional)
            pass

    return out

def build_array_facts(ll_lines: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Merge arg pointers + allocas into one map keyed by array-ish name.
    """
    facts = {}
    facts.update(parse_function_args(ll_lines))
    allocas = parse_allocas(ll_lines)
    # allocas should override if name collides
    for k, v in allocas.items():
        facts[k] = v
    return dict(sorted(facts.items()))


# ----------------------------------------
# Step B: build per-base access lists
# ----------------------------------------

PTR_CAST_RE = re.compile(r'^\s*(%[\w.\-]+)\s*=\s*(bitcast|addrspacecast)\b.*\s+(%[\w.\-]+)\s+to\b')

def build_ptr_alias_map(ll_lines, segments):
    alias = {}
    for line in _iter_region_lines(ll_lines, segments):
        m = PTR_CAST_RE.match(line.strip())
        if m:
            alias[m.group(1)] = m.group(3)
    # path compress
    def find(x):
        seen=set()
        while x in alias and x not in seen:
            seen.add(x); x = alias[x]
        return x
    for k in list(alias.keys()):
        alias[k]=find(alias[k])
    return alias

def resolve_ptr(p: str, alias: Dict[str,str]) -> str:
    while p in alias:
        p = alias[p]
    return p


def build_access_lists(ll_lines, segments, gep_result_to_array, gep_result_to_index):
    loads_by_base = defaultdict(list)
    stores_by_base = defaultdict(list)

    alias = build_ptr_alias_map(ll_lines, segments)  # build ONCE

    for line in _iter_region_lines(ll_lines, segments):
        line = line.strip()
        if not line or line.startswith(";") or LLVM_LABEL_RE.match(line):
            continue

        # LOAD
        if (" load " in f" {line} ") or (line.startswith("%") and " = load " in line):
            p = _llvm_extract_ptr_from_load(line)
            if p:
                p = resolve_ptr(p, alias)  # resolve BEFORE membership check
                base = gep_result_to_array.get(p)
                idx  = gep_result_to_index.get(p)
                if base and idx is not None:
                    loads_by_base[base].append(idx)
            continue

        # STORE
        if " store " in f" {line} ":
            p = _llvm_extract_ptr_from_store(line)
            if p:
                p = resolve_ptr(p, alias)
                base = gep_result_to_array.get(p)
                idx  = gep_result_to_index.get(p)
                if base and idx is not None:
                    stores_by_base[base].append(idx)
            continue

    return {k: sorted(v) for k, v in sorted(loads_by_base.items())}, \
           {k: sorted(v) for k, v in sorted(stores_by_base.items())}


# ----------------------------------------
# Step C: affine evaluator w.r.t. IV
# ----------------------------------------

_CONST_INT_RE = re.compile(r'^-?\d+$')

def _is_int_token(tok: str) -> bool:
    return bool(tok) and bool(_CONST_INT_RE.match(tok))

def _parse_int(tok: str) -> Optional[int]:
    if _is_int_token(tok):
        try:
            return int(tok)
        except Exception:
            return None
    return None

def build_defs_map(ll_lines: List[str], segments: List[Tuple[int, int]]) -> Dict[str, str]:
    """
    defs[%dst] = rhs string (no leading/trailing spaces).
    Only within given segments (deterministic).
    """
    defs: Dict[str, str] = {}
    for (s, e) in segments:
        for i in range(s, e + 1):
            line = ll_lines[i].strip()
            if not line or line.startswith(";") or LLVM_LABEL_RE.match(line):
                continue
            m = LLVM_DEF_RE.match(line)
            if not m:
                continue
            dst, rhs = m.group(1), m.group(2).strip()
            defs[dst] = rhs
    return defs


def find_iv_value_ssa_in_region(
    ll_lines: List[str],
    segments: List[Tuple[int, int]],
    induction_ptr: str,
) -> Optional[str]:
    """
    Find SSA that loads the loop induction alloca pointer within region:
      %ivssa = load i32, i32* %i, ...
    Returns %ivssa if found.
    """
    if not induction_ptr:
        return None

    pat = re.compile(r'^\s*(%[\w\.\-]+)\s*=\s*load\s+i(32|64),\s+i\2\*\s+(' + re.escape(induction_ptr) + r')\b')
    for line in _iter_region_lines(ll_lines, segments):
        m = pat.match(line.strip())
        if m:
            return m.group(1)
    return None


_BINOP_FLAGS = {"nsw", "nuw", "exact"}

def _strip_type_and_flags(rhs: str) -> Tuple[str, str]:
    # returns (op, rest_after_type)
    toks = rhs.split()
    if not toks:
        return "", ""
    op = toks[0]
    toks = toks[1:]
    while toks and toks[0] in _BINOP_FLAGS:
        toks = toks[1:]
    if toks:
        toks = toks[1:]  # drop type token (i32/i64/...)
    return op, " ".join(toks)

def _parse_2ops(rest: str) -> Optional[Tuple[str,str]]:
    # expects "a, b" possibly with extra spaces
    if "," not in rest:
        return None
    a, b = rest.split(",", 1)
    return a.strip(), b.strip()


def depends_on_iv(tok: str, iv_ssa: str, defs: Dict[str,str], induction_ptr: Optional[str]) -> bool:
    if not tok or _is_int_token(tok):
        return False
    if tok == iv_ssa:
        return True
    rhs = defs.get(tok)
    if rhs is None:
        return False  # conservative-in-practice for -O0; if you prefer, return True here
    if rhs.startswith("load") and induction_ptr:
        m = re.search(r'\bload\b[^,]*,\s*(?:ptr|[^,]*\*+)\s*(%[\w.\-]+)\b', rhs)
        if m and m.group(1) == induction_ptr:
            return True
    ops = SSA_RE.findall(rhs)
    return any(depends_on_iv(op, iv_ssa, defs, induction_ptr) for op in ops)


def coeff_in_iv(tok: str, iv_ssa: str, defs: Dict[str,str], memo: Dict[str,Optional[int]], induction_ptr: Optional[str]=None) -> Optional[int]:
    # constants -> coeff 0
    if tok and _is_int_token(tok):
        return 0
    if not tok or not tok.startswith("%") or not iv_ssa:
        return None
    if tok == iv_ssa:
        return 1
    if tok in memo:
        return memo[tok]

    rhs = defs.get(tok)
    if rhs is None:
        memo[tok] = None
        return None

    # Treat loads from the induction alloca as IV-equivalent (this is HUGE)
    if rhs.startswith("load") and induction_ptr:
        m = re.search(r'\bload\b[^,]*,\s*(?:ptr|[^,]*\*+)\s*(%[\w.\-]+)\b', rhs)
        if m and m.group(1) == induction_ptr:
            memo[tok] = 1
            return 1
        memo[tok] = None
        return None

    # casts: identity
    m = re.search(r'^(sext|zext|trunc)\b.*\s+(%[\w.\-]+)\s+to\b', rhs)
    if m:
        memo[tok] = coeff_in_iv(m.group(2), iv_ssa, defs, memo, induction_ptr)
        return memo[tok]

    op, rest = _strip_type_and_flags(rhs)

    if op in {"add","sub"}:
        a, b = _parse_2ops(rest)
        ka = coeff_in_iv(a, iv_ssa, defs, memo, induction_ptr)
        kb = coeff_in_iv(b, iv_ssa, defs, memo, induction_ptr)

        # if one side is "unknown" but does NOT depend on IV, treat it as 0
        if ka is None and not depends_on_iv(a, iv_ssa, defs, induction_ptr):
            ka = 0
        if kb is None and not depends_on_iv(b, iv_ssa, defs, induction_ptr):
            kb = 0

        memo[tok] = None if (ka is None or kb is None) else (ka + kb if op=="add" else ka - kb)
        return memo[tok]

    if op == "mul":
        ab = _parse_2ops(rest)
        if not ab:
            memo[tok] = None
            return None
        a, b = ab
        # mul by constant only
        if _is_int_token(a):
            kb = coeff_in_iv(b, iv_ssa, defs, memo, induction_ptr)
            memo[tok] = None if kb is None else kb * int(a)
            return memo[tok]
        if _is_int_token(b):
            ka = coeff_in_iv(a, iv_ssa, defs, memo, induction_ptr)
            memo[tok] = None if ka is None else ka * int(b)
            return memo[tok]
        memo[tok] = None
        return None

    # Support shift-left as mul by power-of-two (common in address calc)
    if op == "shl":
        ab = _parse_2ops(rest)
        if not ab:
            memo[tok] = None
            return None
        a, b = ab
        if _is_int_token(b):
            ka = coeff_in_iv(a, iv_ssa, defs, memo, induction_ptr)
            memo[tok] = None if ka is None else ka * (1 << int(b))
            return memo[tok]

    memo[tok] = None
    return None



def _empty_loop_region_summary(
    start: int = -1,
    end: int = -1,
    segments: Optional[List[Tuple[int, int]]] = None,
    reason: str = "invalid_span",
) -> Dict[str, Any]:
    # Unified schema (matches the "normal" return)
    return {
        "num_inst": 0,
        "compute": {
            "loads": 0,
            "stores": 0,
            "geps": 0,
            "fadd": 0,
            "fmul": 0,
            "fsub": 0,
        },
        "arrays": {"reads": [], "writes": []},
        "ports": {},

        "loads_by_base": {},
        "stores_by_base": {},

        "rmw_ptrs_scalar": [],

        "segments": segments or [],
        "gep_result_to_index_ssa": {},
        "gep_result_to_array": {},

        # optional but very useful for debugging/QA
        "span": {"start": int(start), "end": int(end)},
        "span_valid": False,
        "reason": reason,
    }


BR_I1_RE = re.compile(r'\bbr\s+i1\b.*\blabel\s+%([A-Za-z0-9_.]+)\b.*\blabel\s+%([A-Za-z0-9_.]+)\b')
LABEL_RE = LLVM_LABEL_RE

def build_blocks_in_segments(ll_lines, segments):
    lines = _iter_region_lines(ll_lines, segments)
    blocks = {}  # label -> list of lines
    cur = None
    for ln in lines:
        m = LABEL_RE.match(ln)
        if m:
            cur = m.group(1)
            blocks[cur] = []
            continue
        if cur is not None:
            blocks[cur].append(ln)
    return blocks

def block_has_store(block_lines):
    return any(" store " in f" {x.strip()} " for x in block_lines)


def detect_conditional_store(ll_lines, segments, header_block: Optional[str] = None) -> bool:
    blocks = build_blocks_in_segments(ll_lines, segments)
    block_labels = set(blocks.keys())

    cur_label = None
    for ln in _iter_region_lines(ll_lines, segments):
        m_lbl = LABEL_RE.match(ln)
        if m_lbl:
            cur_label = m_lbl.group(1)
            continue

        # Skip loop-control conditional in the header block
        if header_block and cur_label == header_block:
            continue

        m = BR_I1_RE.search(ln)
        if not m:
            continue
        t, f = m.group(1), m.group(2)
        if t in block_labels and f in block_labels:
            st = block_has_store(blocks.get(t, []))
            sf = block_has_store(blocks.get(f, []))
            if st != sf and (st or sf):
                return True
    return False



def infer_dependence_hints(
    loop_region_summary: Dict[str, Any],
    iv_ssa: Optional[str],
    defs: Dict[str, str],
    ll_lines: List[str],
    segments: List[Tuple[int,int]],
    induction_ptr = None,   
) -> Dict[str, Any]:
    loads_by_base = loop_region_summary.get("loads_by_base", {}) or {}
    stores_by_base = loop_region_summary.get("stores_by_base", {}) or {}

    rmw_arrays_potential: Set[str] = set()
    rmw_arrays_definite: Set[str] = set()




    for base in sorted(set(loads_by_base.keys()) | set(stores_by_base.keys())):
        lset = set(loads_by_base.get(base, []))
        sset = set(stores_by_base.get(base, []))
        if lset and sset:
            rmw_arrays_potential.add(base)
            if lset & sset:
                rmw_arrays_definite.add(base)

    carried_definite: Set[str] = set()
    carried_potential: Set[str] = set()

    if not iv_ssa:
        carried_potential |= set(stores_by_base.keys())
    else:
        for base, idxs in stores_by_base.items():
            if not iv_ssa:
                carried_potential.add(base)
                continue
            if any(_is_int_token(x) or (not depends_on_iv(x, iv_ssa, defs, induction_ptr)) for x in idxs):
                carried_definite.add(base)

    header = None
    conditional_store = detect_conditional_store(ll_lines, segments, header_block=header)

    return {
        "carried_arrays_definite": sorted(carried_definite),
        "carried_arrays_potential": sorted(carried_potential),
        "rmw_arrays_definite": sorted(rmw_arrays_definite),
        "rmw_arrays_potential": sorted(rmw_arrays_potential),
        "conditional_store": conditional_store,
    }



def summarize_loop_region_from_llvm(
    ll_lines: List[str],
    for_l: str,
    for_blocks_info: Dict,
    exclude_children: bool = True,
) -> dict:
    info = for_blocks_info.get(for_l, {})
    start = int(info.get("line_num", -1))
    end   = int(info.get("end", (None, None))[0] or -1)

    # --- NEW: clamp to file bounds to avoid out-of-range + silent partials ---
    n = len(ll_lines)
    if n <= 0:
        return _empty_loop_region_summary(start, end, [], reason="empty_ll")

    if start >= 0:
        start = max(0, min(start, n - 1))
    if end >= 0:
        end = max(0, min(end, n - 1))

    if start < 0 or end < 0 or end < start:
        return _empty_loop_region_summary(start, end, [], reason="invalid_span")

    base_span = (start, end)
    segments = [base_span]
    if exclude_children:
        holes = _loop_children_spans(ll_lines, for_l, for_blocks_info)
        if holes:
            segments = _subtract_spans(base_span, holes)

    # If subtraction eliminates everything, still return unified schema
    if not segments:
        return _empty_loop_region_summary(start, end, [], reason="empty_segments_after_holes")

    ptr_root_map = build_ptr_root_map(ll_lines, segments)
    gep_result_to_array, gep_result_to_index_ssa = build_gep_maps(ll_lines, segments, ptr_root_map)

    opcode_counts: Dict[str,int] = {}
    gep_bases: Set[str] = set()
    load_ptrs: Set[str] = set()
    store_ptrs: Set[str] = set()

    has_cond_br = False
    num_inst = 0

    for (s, e) in segments:
        for idx in range(s, e + 1):
            line = ll_lines[idx].strip()
            if not line or line.startswith(';'):
                continue
            if LLVM_LABEL_RE.match(line):
                continue

            op = _llvm_opcode(line)
            if op is None:
                continue
            
            if LLVM_COND_BR_RE.search(line):
                has_cond_br = True

            # count instruction-like lines
            num_inst += 1
            opcode_counts[op] = opcode_counts.get(op, 0) + 1

            if LLVM_GEP_RE.search(line):
                b = _llvm_extract_gep_base(line)
                if b:
                    root = ptr_root_map.get(b,b)
                    arr = array_name_from_root(root)
                    gep_bases.add(arr if arr else root)

            if " load " in f" {line} " or line.startswith("%") and " = load " in line:
                p = _llvm_extract_ptr_from_load(line)
                if p:
                    load_ptrs.add(p)

            if " store " in f" {line} ":
                p = _llvm_extract_ptr_from_store(line)
                if p:
                    store_ptrs.add(p)

    rmw_ptrs_ssa = detect_rmw(load_ptrs, store_ptrs)

    # lift RMW pointers to array names when the pointer is a GEP result
    rmw_gep_ptrs = sorted(set(rmw_ptrs_ssa) & set(gep_result_to_array.keys()))

    rmw_ptrs_all = sorted(set(load_ptrs) & set(store_ptrs))
    rmw_ptrs_gep = sorted(set(rmw_ptrs_all) & set(gep_result_to_array.keys()))
    rmw_ptrs_scalar = sorted(set(rmw_ptrs_all) - set(rmw_ptrs_gep))

    rmw_arrays_set: Set[str] = set()
    for p in rmw_gep_ptrs:
        a = gep_result_to_array.get(p)
        if a:
            rmw_arrays_set.add(a)

    # --- NEW: per-array access lists (for pattern/stride and port pressure) ---
    loads_by_base, stores_by_base = build_access_lists(
        ll_lines, segments, gep_result_to_array, gep_result_to_index_ssa
    )

    # --- NEW: per-array port pressure (counts of loads/stores that hit each base) ---
    read_cnt = Counter()
    write_cnt = Counter()
    for (s, e) in segments:
        for idx in range(s, e + 1):
            line = ll_lines[idx].strip()
            if not line or line.startswith(";") or LLVM_LABEL_RE.match(line):
                continue
            if (" load " in f" {line} ") or (line.startswith("%") and " = load " in line):
                p = _llvm_extract_ptr_from_load(line)
                if p and p in gep_result_to_array:
                    base = gep_result_to_array.get(p)
                    if base:
                        read_cnt[base] += 1
            if " store " in f" {line} ":
                p = _llvm_extract_ptr_from_store(line)
                if p and p in gep_result_to_array:
                    base = gep_result_to_array.get(p)
                    if base:
                        write_cnt[base] += 1

    # iv SSA (needed for stride)
    ind_ptr = None  # don't recompute here; pass it in if you prefer

    return {
        "num_inst": int(num_inst),
        # keep a tiny compute summary if you want
        "compute": {
            "loads": int(opcode_counts.get("load", 0)),
            "stores": int(opcode_counts.get("store", 0)),
            "geps": int(opcode_counts.get("getelementptr", 0)),
            "fadd": int(opcode_counts.get("fadd", 0)),
            "fmul": int(opcode_counts.get("fmul", 0)),
            "fsub": int(opcode_counts.get("fsub", 0)),
        },
        "arrays": {
            "reads":  sorted(read_cnt.keys()),
            "writes": sorted(write_cnt.keys()),
        },
        "ports": {
            k: {"reads_per_iter": int(read_cnt.get(k, 0)), "writes_per_iter": int(write_cnt.get(k, 0))}
            for k in sorted(set(read_cnt.keys()) | set(write_cnt.keys()))
        },

        # keep these ONLY if you want to compute stride later; otherwise drop them too
        "loads_by_base": loads_by_base,
        "stores_by_base": stores_by_base,

        "has_cond_br": bool(has_cond_br),
        "rmw_ptrs_gep": rmw_ptrs_gep,
        "rmw_ptrs_scalar": rmw_ptrs_scalar,

        # keep segments only for internal debugging (or drop entirely)
        "segments": segments,
        "gep_result_to_index_ssa": dict(sorted(gep_result_to_index_ssa.items())),
        "gep_result_to_array": dict(sorted(gep_result_to_array.items())),
    }




# -----------------------------
# Mapping: for_blocks_info -> pseudo_block node ids (deterministic)
# -----------------------------

def successors_sorted(G: nx.DiGraph, n: Any) -> List[Any]:
    try:
        succ = list(G.successors(n))
    except Exception:
        succ = list(G.neighbors(n))
    return sorted(succ, key=lambda x: node_sort_key(G, x))

def map_for_blocks_to_pseudo_nodes(G: nx.DiGraph, for_blocks_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct the same mapping augment_graph_hierarchy uses:
      for_l key -> (block,function) -> pseudo_block node id
    """
    # 1) locate each loop's block/function using next_instr chain matching
    block_ids: Dict[str, Tuple[Any, Any]] = {}
    for for_l in sorted(for_blocks_info.keys()):
        info = for_blocks_info[for_l]
        next_instr = info.get("next_instr", [])
        if len(next_instr) < 3:
            continue

        found = False
        for n in sorted(G.nodes(), key=lambda x: node_sort_key(G, x)):
            ft = get_full_text(G.nodes[n])
            if not ft:
                continue
            if next_instr[0] in ft:
                correct = 1
                # check neighbor chain deterministically
                for nb in successors_sorted(G, n):
                    if next_instr[1] in get_full_text(G.nodes[nb]):
                        correct += 1
                        if correct == 2:
                            for nb2 in successors_sorted(G, nb):
                                if next_instr[2] in get_full_text(G.nodes[nb2]):
                                    correct += 1
                                    break
                    if correct == 3:
                        break
                if correct == 3:
                    d = G.nodes[n]
                    block_ids[for_l] = (d.get("block"), d.get("function"))
                    found = True
                    break

        if not found:
            # Keep deterministic behavior: skip but record as missing
            block_ids[for_l] = (None, None)

    # 2) map (block,function) -> pseudo_block node id
    pseudo_map: Dict[str, Any] = {}
    for for_l in sorted(for_blocks_info.keys()):
        b, f = block_ids.get(for_l, (None, None))
        if b is None or f is None:
            continue

        candidates = []
        for n in G.nodes():
            d = G.nodes[n]
            if str(d.get("text", "")) != "pseudo_block":
                continue
            if str(d.get("block")) == str(b) and str(d.get("function")) == str(f):
                candidates.append(n)

        if candidates:
            candidates = sorted(candidates, key=lambda x: node_sort_key(G, x))
            pseudo_map[for_l] = candidates[0]

    return pseudo_map

def compute_loop_tree(pseudo_map: Dict[str, Any], for_blocks_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build loop tree keyed by for_l (LLVM for.cond key) using for_blocks_info['children'].
    Returns:
      {
        "for_l": {
          "pseudo_id": ...,
          "children": [... for_l ...],
          "parent": <for_l or None>,
          "depth": int,
        },
        ...
      }
    """
    children_of = {k: list(for_blocks_info[k].get("children", [])) for k in for_blocks_info.keys()}
    # parent mapping
    parent_of: Dict[str, Optional[str]] = {k: None for k in for_blocks_info.keys()}
    for k, kids in children_of.items():
        for c in kids:
            if c in parent_of and parent_of[c] is None:
                parent_of[c] = k

    roots = sorted([k for k, p in parent_of.items() if p is None])
    depth_of: Dict[str, int] = {}

    def dfs(u: str, depth: int):
        depth_of[u] = depth
        for v in children_of.get(u, []):
            if v in for_blocks_info:
                dfs(v, depth + 1)

    for r in roots:
        dfs(r, 0)

    tree = {}
    for k in sorted(for_blocks_info.keys()):
        tree[k] = {
            "pseudo_id": pseudo_map.get(k),
            "children": sorted(children_of.get(k, [])),
            "parent": parent_of.get(k),
            "depth": int(depth_of.get(k, 0)),
        }
    return tree

# -----------------------------
# Extract pragma <-> loop linkage
# -----------------------------

def pragma_neighbors_flow200(G: nx.DiGraph, p: Any) -> List[Any]:
    neigh = set()
    for _, v, _, d in iter_out_edges_any(G, p):
        if edge_flow(d) == 200:
            neigh.add(v)
    for u, _, _, d in iter_in_edges_any(G, p):
        if edge_flow(d) == 200:
            neigh.add(u)
    return sorted(neigh, key=lambda x: node_sort_key(G, x))

def find_icmp_node_among(G: nx.DiGraph, nodes: List[Any]) -> Optional[Any]:
    # Prefer instruction nodes whose full_text contains 'icmp'
    icmp = []
    for n in nodes:
        d = G.nodes[n]
        if as_int(d.get("type", -1)) == 0 and "icmp" in get_full_text(d):
            icmp.append(n)
    if icmp:
        icmp = sorted(icmp, key=lambda x: node_sort_key(G, x))
        return icmp[0]
    return nodes[0] if nodes else None

def find_pseudo_block_for_node(G: nx.DiGraph, n: Any) -> Optional[Any]:
    # Look for neighbor with flow=4 and text='pseudo_block'
    candidates = []
    for _, v, _, d in iter_out_edges_any(G, n):
        if edge_flow(d) == 4 and str(G.nodes[v].get("text", "")) == "pseudo_block":
            candidates.append(v)
    for u, _, _, d in iter_in_edges_any(G, n):
        if edge_flow(d) == 4 and str(G.nodes[u].get("text", "")) == "pseudo_block":
            candidates.append(u)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda x: node_sort_key(G, x))
    return candidates[0]

def scope_nodes_from_pseudo_block(G: nx.DiGraph, pseudo_id: Any) -> List[Any]:
    # All nodes connected via flow=4 between the pseudo block node and instructions in the block.
    scope = set([pseudo_id])
    for _, v, _, d in iter_out_edges_any(G, pseudo_id):
        if edge_flow(d) == 4:
            scope.add(v)
    for u, _, _, d in iter_in_edges_any(G, pseudo_id):
        if edge_flow(d) == 4:
            scope.add(u)
    return sorted(scope, key=lambda x: node_sort_key(G, x))

# -----------------------------
# Main metadata extraction per kernel
# -----------------------------

def extract_kernel_memory_meta(kernel: str, log: bool = False) -> Optional[Dict[str, Any]]:
    gexf_path = os.path.join(HIER_GEXF_DIR, f"{kernel}_processed_result.gexf")
    if not os.path.isfile(gexf_path):
        if log:
            print(f"[SKIP] Missing hierarchy GEXF: {gexf_path}")
        return None

    # Load graph
    G = nx.readwrite.gexf.read_gexf(gexf_path, node_type=str)
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)

    # Load LLVM loop structure
    llvm_kernel_dir = os.path.join(ROOT, f"{TYPE_GRAPH}/{kernel}")
    ll_path = os.path.join(llvm_kernel_dir, f"{kernel}.ll")
    if not os.path.isfile(ll_path):
        if log:
            print(f"[SKIP] Missing LLVM: {ll_path}")
        return None

    for_blocks_info = ggd.get_for_blocks_info(kernel, llvm_kernel_dir)
    pseudo_map = map_for_blocks_to_pseudo_nodes(G, for_blocks_info)
    loop_tree = compute_loop_tree(pseudo_map, for_blocks_info)

    # Tripcounts from kernel_info.txt (if present)
    kernel_info_path = os.path.join(llvm_kernel_dir, "kernel_info.txt")
    trip_by_label = {}
    if os.path.isfile(kernel_info_path):
        trip_by_label = ggd.load_tripcounts_by_label(kernel_info_path)

    # Index: pseudo_id -> for_l key
    pseudo_to_forl = {}
    for forl, rec in loop_tree.items():
        pid = rec.get("pseudo_id")
        if pid is not None:
            pseudo_to_forl[pid] = forl

    # Collect pragma nodes and build placeholder records
    pragma_nodes = []
    for n in G.nodes():
        d = G.nodes[n]
        if as_int(d.get("type", -1)) == 100:
            pragma_nodes.append(n)
    pragma_nodes = sorted(pragma_nodes, key=lambda x: node_sort_key(G, x))

    placeholder_records = []
    loop_labels_by_pseudo = defaultdict(set)

    for p in pragma_nodes:
        pd = G.nodes[p]
        p_ft = get_full_text(pd)
        p_kind = pragma_kind_from_node(pd)
        ph = parse_placeholder_key(p_ft)

        # Always record pragma nodes; placeholders may be None for fixed pragmas.
        neigh = pragma_neighbors_flow200(G, p)
        icmp_or_target = find_icmp_node_among(G, neigh) if neigh else None

        pseudo_id = None
        forl = None
        loop_depth = None
        loop_parent = None
        loop_children = None

        if p_kind in {"PIPELINE", "UNROLL"} and icmp_or_target is not None:
            pseudo_id = find_pseudo_block_for_node(G, icmp_or_target)
            if pseudo_id is not None:
                forl = pseudo_to_forl.get(pseudo_id)
                if forl is not None:
                    lt = loop_tree.get(forl, {})
                    loop_depth = lt.get("depth")
                    loop_parent = lt.get("parent")
                    loop_children = lt.get("children")

        label = parse_loop_label_from_placeholder(ph) if ph else None
        if label and pseudo_id is not None:
            loop_labels_by_pseudo[pseudo_id].add(label)

        ap_var = parse_array_var(p_ft) if p_kind == "ARRAY_PARTITION" else None

        placeholder_records.append({
            "pragma_node": p,
            "pragma_kind": p_kind,
            "pragma_full_text": p_ft,
            "placeholder_key": ph,
            "label": label,
            "array_var": ap_var,
            "targets_flow200": neigh,
            "icmp_or_target_node": icmp_or_target,
            "pseudo_block_node": pseudo_id,
            "for_l_key": forl,
            "loop_depth": loop_depth,
            "loop_parent_for_l": loop_parent,
            "loop_children_for_l": loop_children,
        })

    # Loop-level facts (keyed by for_l)
    with open(ll_path, "r") as f:
        ll_lines = f.readlines()

    array_facts = build_array_facts(ll_lines)

    loop_facts = OrderedDict()
    for forl in sorted(loop_tree.keys()):
        pid = loop_tree[forl].get("pseudo_id")
        # derive a preferred label if available via placeholders
        labels = sorted(loop_labels_by_pseudo.get(pid, set()))
        label = labels[0] if labels else None
        trip = trip_by_label.get(label) if label else None

        loop_region_summary = summarize_loop_region_from_llvm(
            ll_lines,
            forl,
            for_blocks_info,
            exclude_children=True,   
        )

        full_summary  = summarize_loop_region_from_llvm(ll_lines, forl, for_blocks_info, exclude_children=False)

        # inject ll_lines for analysis only (do NOT serialize this)
        full_summary["_ll_lines"] = ll_lines

        ind_ptr = find_induction_ptr(ll_lines, forl, for_blocks_info)
        descendants = []
        for child_forl in loop_tree[forl].get("children", []):
            c_ind = find_induction_ptr(ll_lines, child_forl, for_blocks_info)
            if c_ind:
                descendants.append(c_ind)

        # Collect ALL descendant loop keys (for_l), then sort inner->outer by depth
        desc_forls: List[str] = []

        def collect_desc_forls(u: str) -> None:
            for v in loop_tree[u].get("children", []):
                if v in loop_tree:                 # defensive
                    desc_forls.append(v)
                    collect_desc_forls(v)

        collect_desc_forls(forl)

        # Sort by depth descending: innermost loops first
        desc_forls.sort(key=lambda v: (int(loop_tree[v].get("depth", 0)), v), reverse=True)

        # Map to induction pointers in that (inner->outer) order, de-dup while preserving order
        descendants: List[str] = []
        seen: Set[str] = set()
        for v in desc_forls:
            ind = find_induction_ptr(ll_lines, v, for_blocks_info)
            if ind and ind not in seen:
                descendants.append(ind)
                seen.add(ind)

        # filter noisy scalar RMW pointers
        rmw_scalars_raw = full_summary.get("rmw_ptrs_scalar", [])
        rmw_scalars_raw_local = loop_region_summary.get("rmw_ptrs_scalar", [])

        # drop anything equal to this loop's induction ptr
        banned = set()
        if ind_ptr:
            banned.add(ind_ptr)

        # drop induction pointers of descendants
        banned.update(d for d in descendants if d)

        rmw_scalars_filtered = sorted([p for p in rmw_scalars_raw if p not in banned])
        rmw_scalars_filtered_local = sorted([p for p in rmw_scalars_raw_local if p not in banned])

        segs = loop_region_summary.get("segments", []) or []

        carried_scalars, temp_scalars, first_access_map = classify_scalar_rmw_first_access(ll_lines, segs, rmw_scalars_filtered_local)

        # iv SSA for this loop (if available)
        iv_ssa = find_iv_value_ssa_in_region(ll_lines, segs, ind_ptr) if ind_ptr else None
        defs = build_defs_map(ll_lines, segs)

        # Build mem signature from loads_by_base/stores_by_base
        reads = []
        writes = []

        loads_by_base = loop_region_summary.get("loads_by_base", {}) or {}
        stores_by_base = loop_region_summary.get("stores_by_base", {}) or {}

        for base in sorted(set(loads_by_base.keys()) | set(stores_by_base.keys())):
            sp = space_of_array(base)
            if base in loads_by_base:
                pat, stride = classify_pattern_stride(loads_by_base[base], iv_ssa, defs, ind_ptr)
                reads.append({"array": base, "space": sp, "pattern": pat, "stride": stride})
            if base in stores_by_base:
                pat, stride = classify_pattern_stride(stores_by_base[base], iv_ssa, defs, ind_ptr)
                writes.append({"array": base, "space": sp, "pattern": pat, "stride": stride})

        mem_sig = {
            "reads": reads,
            "writes": writes,
            "global_reads": sorted([x["array"] for x in reads if x["space"] == "global"]),
            "global_writes": sorted([x["array"] for x in writes if x["space"] == "global"]),
        }

        # Recurrences: run on carried scalars only (NOT iv, NOT temps)
        recs = detect_scalar_recurrences(ll_lines, segs, carried_scalars)

        header = forl.split(":", 1)[0]  
        dep_hints = infer_dependence_hints(loop_region_summary, iv_ssa, defs, ll_lines, segs, induction_ptr=ind_ptr)
        dep_hints["conditional_store"] = detect_conditional_store(ll_lines, segs, header_block=header)


        loop_facts[forl] = {
            "pseudo_block_node": pid,
            "depth": loop_tree[forl].get("depth"),
            "parent": loop_tree[forl].get("parent"),
            "children": loop_tree[forl].get("children", []),
            "labels_from_placeholders": labels,
            "tripcount_from_kernel_info": trip,

            "iv": {"ptr": ind_ptr, "name": ind_ptr.lstrip("%") if ind_ptr else None, "step": 1, "tripcount": trip},
            "mem": mem_sig,
            "ports": loop_region_summary.get("ports", {}),
            "recurrences": recs,
            "dependence_hints": dep_hints,

            "compute": loop_region_summary.get("compute", {}),
            "temps": [p.lstrip("%") for p in temp_scalars],
        }

    array_used_in = defaultdict(list)
    label_by_forl = {}

    for forl, lf in loop_facts.items():
        labs = lf.get("labels_from_placeholders", [])
        if labs:
            label_by_forl[forl] = labs[0]  # prefer Lx
        loop_label = label_by_forl.get(forl)
        if not loop_label:
            continue

        arrays = set()
        for r in lf.get("mem", {}).get("reads", []):
            arrays.add(r["array"])
        for w in lf.get("mem", {}).get("writes", []):
            arrays.add(w["array"])

        for a in sorted(arrays):
            array_used_in[a].append(loop_label)

    for a in list(array_used_in.keys()):
        array_used_in[a] = sorted(set(array_used_in[a]))


    # Kernel-level summary
    meta = {
        "kernel": kernel,
        "graph": {
            "num_nodes": int(G.number_of_nodes()),
            "num_edges": int(G.number_of_edges()),
        },
        "array": array_facts,
        "placeholders": placeholder_records,
        "loops": loop_facts,
    }

    # Deterministic JSON ordering:
    # - sort placeholder records by (kind, placeholder_key, pragma_node)
    meta["placeholders"] = sorted(
        meta["placeholders"],
        key=lambda r: (
            str(r.get("pragma_kind", "")),
            str(r.get("placeholder_key") or ""),
            str(r.get("pragma_node")),
        ),
    )

    for ph in meta["placeholders"]:
        if ph.get("pragma_kind") == "ARRAY_PARTITION":
            a = ph.get("array_var")
            ph["used_in_loops"] = array_used_in.get(a, [])
            ph["num_targets_flow200"] = len(ph.get("targets_flow200", []))
            ph.pop("targets_flow200", None)  # remove noise


    return meta

# -----------------------------
# Write outputs
# -----------------------------

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)

def kernels_list_from_dataset(limit: Optional[int] = None) -> List[str]:
    ks = sorted(os.listdir(BASE_DATASET_DIR))
    if limit is not None:
        ks = ks[:limit]
    return ks

def run_meta_export(kernels: List[str], log: bool = False) -> Dict[str, Any]:
    os.makedirs(META_DIR, exist_ok=True)

    manifest = {
        "type_graph": TYPE_GRAPH,
        "hierarchy_gexf_dir": HIER_GEXF_DIR,
        "meta_dir": META_DIR,
        "kernels": [],
        "skipped": [],
    }

    for k in kernels:
        m = extract_kernel_memory_meta(k, log=log)
        if m is None:
            manifest["skipped"].append(k)
            continue

        out_path = os.path.join(META_DIR, f"{k}_memory_meta.json")
        write_json(out_path, m)

        manifest["kernels"].append({
            "kernel": k,
            "meta_path": out_path,
            "num_placeholders": len(m["placeholders"]),
            "num_loops": len(m["loops"]),
        })

        if log:
            print(f"[OK] {k}: meta -> {out_path}")

    # deterministic sort
    manifest["kernels"] = sorted(manifest["kernels"], key=lambda r: r["kernel"])
    manifest["skipped"] = sorted(manifest["skipped"])

    write_json(os.path.join(META_DIR, "manifest.json"), manifest)
    return manifest

# -----------------------------
# CLI
# -----------------------------

def main():
    require_pythonhashseed()

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run full graph pipeline + metadata export.")
    ap.add_argument("--meta-only", action="store_true", help="Only export metadata (expects hierarchy graphs exist).")
    ap.add_argument("--kernels", nargs="*", default=None, help="Optional list of kernel directory names to process.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit number of kernels (dataset order).")
    ap.add_argument("--log", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    if not args.all and not args.meta_only:
        # Safe default: do meta-only if graphs already exist; otherwise user can pass --all.
        args.meta_only = True

    # Decide kernel list
    if args.kernels:
        kernels = sorted(args.kernels)
    else:
        kernels = kernels_list_from_dataset(limit=args.limit)

    if args.all:
        # Run your pipeline exactly as in graph_gen_deterministic __main__
        # (initial -> auxiliary base -> auxiliary connected -> hierarchy).
        ggd.run_graph_gen(mode="initial", connected=True)
        ggd.run_graph_gen(mode="auxiliary", connected=False)
        ggd.run_graph_gen(mode="auxiliary", connected=True)
        ggd.run_graph_gen(mode="hierarchy", connected=True)

    # Export metadata sidecars
    run_meta_export(kernels, log=args.log)

if __name__ == "__main__":
    main()

