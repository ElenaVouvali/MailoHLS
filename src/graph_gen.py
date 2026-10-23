import os
import networkx as nx
import json
import shutil
from os.path import join, abspath, basename, exists, dirname, isfile
from subprocess import Popen, PIPE
from collections import OrderedDict
from copy import deepcopy
import ast
from pprint import pprint
from glob import iglob
import glob
import csv
import re
import google.protobuf
import programl

from utils import create_dir_if_not_exists, get_root_path, natural_keys
from insert_placeholders import insert_placeholders

PRAGMA_POSITION = {
    'PIPELINE': 0,
    'UNROLL': 1,
    'ARRAY_PARTITION': 2
}

type_graph = 'harp'
processed_gexf_folder = join(get_root_path(), f'{type_graph}/processed')


class Node():
    def __init__(self, block, function, text, type_n, features = None):
        self.block : int = block
        self.function : int = function
        self.text : str = text
        self.type_n : int = type_n ## 0: instruction, 1: variable, 2: immediate, 100: pragma, 4: pseudo node for block
        self.features : str = features ## contains full text

    def get_attr(self, after_process = True):
        '''
            args:
                after_process : True if nodes are added to existing GNN-DSE graphs
                                False for initial graph generation in GNN-DSE
        '''
        n_dict = {}
        n_dict['block'] = self.block
        n_dict['function'] = self.function
        n_dict['text'] = self.text
        n_dict['type'] = self.type_n
        if after_process:
            n_dict['full_text'] = self.features
        else:
            n_dict['features'] = {'full_text': [self.features]}


        return n_dict

class Edge():
    def __init__(self, src, dst, flow, position):
        self.src : int = src
        self.dst : int = dst
        self.flow : int = flow ## 0: control, 1: data, 2: call, 200: pragma, 4: pseudo node for block, 5: connections between pseudo nodes, 6: for loop hierarchyhierarchy
        self.position : int = position

    def get_attr(self):
        e_dict = {}
        e_dict['flow'] = self.flow
        e_dict['position'] = self.position

        return e_dict




def _node_sort_key(node, data):
    # Use only fields that exist in your graphs and are stable.
    # Adjust this to your schema.
    return (
        int(data.get("function", -1)),
        int(data.get("block", -1)),
        int(data.get("type", -1)),
        str(data.get("text", "")),
        str(data.get("full_text", "")),
        str(node),  # tie-breaker
    )

def canonicalize_for_gexf(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Rebuild a graph with deterministic insertion order for nodes and edges.
    Does NOT change node IDs; it only stabilizes serialization order and
    any subsequent iteration-dependent numbering you do.
    """
    H = nx.MultiDiGraph()

    # Deterministic node insertion order
    nodes_sorted = sorted(G.nodes(data=True), key=lambda nd: _node_sort_key(nd[0], nd[1]))
    for n, d in nodes_sorted:
        H.add_node(n, **deepcopy(d))

    node_rank = {n: i for i, (n, _) in enumerate(nodes_sorted)}

    # Deterministic edge insertion order
    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        edges.append((u, v, k, deepcopy(d)))

    def edge_sort_key(e):
        u, v, k, d = e
        return (
            node_rank.get(u, 10**9),
            node_rank.get(v, 10**9),
            int(d.get("flow", -1)),
            int(d.get("position", -1)),
            str(u), str(v), str(k),
        )

    for u, v, k, d in sorted(edges, key=edge_sort_key):
        H.add_edge(u, v, key=k, **d)

    return H



def assign_deterministic_edge_ids(G: nx.MultiDiGraph) -> None:
    edges = list(G.edges(keys=True, data=True))

    def ek(e):
        u, v, k, d = e
        return (
            str(u), str(v), str(k),
            int(d.get("flow", -1)),
            int(d.get("position", -1)),
        )

    for eid, (u, v, k, d) in enumerate(sorted(edges, key=ek)):
        d["id"] = eid



def create_pseudo_node_block(block, function):
    return Node(block, function, text = 'pseudo_block', type_n = 4, features = 'auxiliary node for each block')

def add_to_graph(g_nx, nodes, edges):
    if len(nodes) > 0:
        g_nx.add_nodes_from(nodes)
    if len(edges) > 0:
        g_nx.add_edges_from(edges)


def read_json_graph(name, readable=True):
    '''
        reads a graph in json format as a netwrokx graph

        args:
            name: name of the json file/ kernel's name
            reaable: whether to store a readable format of the json file

        returns:
            g_nx: graph in networkx format
    '''
    filename = name + '.json'
    with open(filename) as f:
        js_graph=json.load(f)
    g_nx=nx.readwrite.json_graph.node_link_graph(js_graph, edges="links")
    if readable:
        make_json_readable(name, js_graph)

    return g_nx


def llvm_to_nx(name):
    '''
        reads a LLVM IR and converts it to a netwrokx graph

        args:
            name: name of the LLVM file/ kernel's name

        returns:
            g_nx: graph in networkx format
    '''
    filename = name + '.ll'
    with open(filename) as f:
        ll_file = f.read()
        G=programl.from_llvm_ir(ll_file)
        g_nx=programl.to_networkx(G)

    return g_nx

def make_json_readable(name, js_graph):
    '''
        gets a json file and beautifies it to make it readable

        args:
            name: kernel name
            js_graph: the graph in networkx format read from the json file

        writes:
            a readable json file with name {name}_pretty.json
    '''
    filename = name + '_pretty.json'
    f_json=open((filename), "w+")
    json.dump(js_graph, f_json, indent=4, sort_keys=True)
    f_json.close()



def extract_function_names(c_code):
    """
    Return list of (function_name, line_number) for top-level function definitions.
    We strip comments and ignore matches where the 'function name' is actually
    a control keyword such as 'if', 'for', 'while', etc.
    """

    # Strip C/C++ style comments but preserve line breaks where possible
    def _strip_comments(text):
        # Replace block comments with the same number of newlines so that line numbers remain roughly correct.
        def repl_block(m):
            return "\n" * m.group(0).count("\n")

        # Remove block comments: /* ... */
        text = re.sub(r"/\*.*?\*/", repl_block, text, flags=re.DOTALL)
        # Remove single-line comments: // ...
        text = re.sub(r"//.*", "", text)
        return text

    c_code_nocomments = _strip_comments(c_code)

    # Remove `extern "C" {` but keep newlines
    c_code_clean = re.sub(r'extern\s+"C"\s*{', '', c_code_nocomments)

    # pattern: <ret> <name>( ... ) {
    pattern = r'\b\w[\w\s\*]*\b([A-Za-z_]\w*)\s*\([^;]*\)\s*\{'

    # Do not treat control keywords as function names
    CONTROL_KEYWORDS = {"if", "for", "while", "switch", "else", "do", "case"}

    function_names = []
    for match in re.finditer(pattern, c_code_clean, flags=re.DOTALL):
        func_name = match.group(1)

        # Skip control constructs like "else if (...) {" being misread as a function
        if func_name in CONTROL_KEYWORDS:
            continue

        line_number = c_code_clean.count('\n', 0, match.start()) + 1
        function_names.append((func_name, line_number))

    return function_names



def get_icmp(path, name, log=False):
    llvm_file = join(path, f"{name}.ll")
    with open(llvm_file, "r") as f_llvm:
        lines_llvm = f_llvm.readlines()

    for_dict_llvm = OrderedDict()
    for_count_llvm = 0
    current_func = None
    local_for_count = 0

    for idx, line in enumerate(lines_llvm):
        s = line.strip()

        # Start of function
        if s.startswith("define"):
            current_func = s
            for_dict_llvm[current_func] = OrderedDict()
            local_for_count = 0
            continue

        # for-loop header
        if s.startswith("for.cond") and current_func is not None:
            local_for_count += 1
            for_count_llvm += 1
            loop_id = local_for_count

            # Look ahead for first icmp before next basic block label
            for idx2, line2 in enumerate(lines_llvm[idx+1:], start=idx+1):
                t = line2.strip()
                if re.match(r'^[A-Za-z0-9_.]+:\s*$', t):
                    break
                if "icmp" in t:
                    for_dict_llvm[current_func][loop_id] = [t, idx, idx2]
                    break

    if log:
        print(json.dumps(for_dict_llvm, indent=2))

    return for_dict_llvm, for_count_llvm



def get_pragmas_loops(path, name, EXT='cpp', top_func=None, log=False):
    '''
        gets a c/cpp kernel and returns the pragmas of each for loop

        args:
            path: parent directory of the kernel file
            name: kernel name (WITHOUT extension)

        returns:
            a dictionary with each entry showing the for loop and its pragmas
                {function_name: {for_loop_id: [for loop line, [list of pragmas]]}}
            number of for loops (total over all functions)
    '''

    src_file = join(path, f'{name}.{EXT}')

    with open(src_file, 'r') as f_source:
        code = f_source.read()

    lines_source = code.splitlines(True)

    function_names_list = extract_function_names(code)

    if not function_names_list:
        # treat the whole file as a single "GLOBAL" function
        print(
            f"[WARN] extract_function_names() did not find any functions in {src_file}; "
            "treating whole file as GLOBAL."
        )
        function_names_list = [("GLOBAL", 1)]

    for_dict_source = OrderedDict()
    for_count_source = 0

    #  Structured scan (per function)
    for f_id, (f_name, start_line_1based) in enumerate(function_names_list):
        # convert to 0-based indices for lines_source
        start_idx = max(0, start_line_1based - 1)

        if f_id + 1 < len(function_names_list):
            next_start_line_1based = function_names_list[f_id + 1][1]
            end_idx = max(0, next_start_line_1based - 1)  # exclusive
        else:
            end_idx = len(lines_source)

        for_dict_source[f_name] = OrderedDict()
        local_for_count_source = 0

        idx = start_idx
        while idx < end_idx:
            line = lines_source[idx].strip()

            if not line or 'scop' in line:
                idx += 1
                continue

            # detect for loop
            if re.search(r'\bfor\s*\(', line):
                for_count_source += 1
                local_for_count_source += 1
                loop_line = line.strip('{')

                # collect pragmas after this for loop
                pragma_list = []
                j = idx + 1
                while j < end_idx:
                    next_line = lines_source[j].strip()
                    # skip labels like "L1:"
                    if next_line.endswith(":") and not next_line.startswith("#pragma"):
                        j += 1
                        continue
                    if next_line.startswith("#pragma") and "KERNEL" not in next_line.upper():
                        pragma_list.append(next_line)
                        j += 1
                    else:
                        break

                for_dict_source[f_name][local_for_count_source] = [loop_line, pragma_list]
                idx = j
                continue

            idx += 1

    # if we found 0 for-loops, do a flat scan
    if for_count_source == 0:
        print(f"[WARN] No for-loops found in structured scan of {src_file}; "
              f"applying flat fallback scan.")

        fallback_loops = []
        for idx, line in enumerate(lines_source):
            if re.search(r'\bfor\s*\(', line):
                fallback_loops.append((idx, line.strip()))

        if fallback_loops:
            if top_func and top_func.strip():
                fname = top_func.strip()
            elif function_names_list:
                fname = function_names_list[0][0]
            else:
                fname = "FALLBACK"

            for_dict_source = OrderedDict()
            for_dict_source[fname] = OrderedDict()

            for local_id, (lineno, loop_line) in enumerate(fallback_loops, start=1):
                for_dict_source[fname][local_id] = [loop_line, []]  # no pragmas
            for_count_source = len(fallback_loops)

            if log:
                print(f"[INFO] Fallback loop detection in {src_file}: "
                      f"found {for_count_source} loops.")
        else:
            if log:
                print(f"[INFO] Fallback scan also found no for-loops in {src_file}.")

    if log:
        print(json.dumps(for_dict_source, indent=4))

    return for_dict_source, for_count_source



def get_pragmas_arrays(placeholder_cpp_file, log=False):
    """
    Parse *_placeholders.cpp and collect all
    `#pragma HLS array_partition` pragmas.

    Returns a list of dicts:
        { "function": <function name or None>,
          "var": <array variable name>,
          "pragma": <full pragma line> }
    """
    with open(placeholder_cpp_file, 'r') as f:
        lines = f.readlines()

    # Build function spans: (name, start_line_idx, end_line_idx)
    function_names_list = extract_function_names(''.join(lines))
    func_spans = []
    for i, (fname, start_line) in enumerate(function_names_list):
        start_idx = start_line - 1          # 0-based line index
        if i + 1 < len(function_names_list):
            end_idx = function_names_list[i + 1][1] - 2
        else:
            end_idx = len(lines) - 1
        func_spans.append((fname, start_idx, end_idx))

    # Map line index --> function name
    line_to_func = {}
    for fname, s, e in func_spans:
        for li in range(s, e + 1):
            line_to_func[li] = fname

    array_pragmas = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("#pragma"):
            continue
        if "array_partition" not in s:
            continue

        m = re.search(r'variable\s*=\s*([A-Za-z_]\w*)', s)
        if not m:
            continue
        varname = m.group(1)
        fname = line_to_func.get(idx, None)
        array_pragmas.append(
            {"function": fname, "var": varname, "pragma": s}
        )

    if log:
        pprint(array_pragmas)
    return array_pragmas


LABEL_IN_COMMENT = re.compile(r'/\*\s*(L\d+)\s*:\s*\*/', re.IGNORECASE)
LABEL_AT_START   = re.compile(r'^\s*(L\d+)\s*:', re.IGNORECASE)

def extract_loop_label(loop_line: str):
    if not isinstance(loop_line, str):
        return None
    m = LABEL_IN_COMMENT.search(loop_line)
    if m:
        return m.group(1).upper()
    m = LABEL_AT_START.match(loop_line)
    if m:
        return m.group(1).upper()
    return None


def load_tripcounts_by_label(kernel_info_file):
    by_label = {}  # 'L6' -> 2048
    with open(kernel_info_file) as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = [p.strip() for p in line.split(',')]
            # Expect: <Label>,loop,<bound>,<LoopName>
            if len(parts) >= 3 and parts[1].lower() == 'loop':
                label = parts[0].upper()
                bound = int(parts[2])
                by_label[label] = bound
    return by_label


def check_tripcount_consistency(for_loop_text, icmp_inst, tripcounts_by_label):
    """
    Compare loop info from kernel_info (by label) with the info from the LLVM icmp.
    Returns the label (or None) so that the caller can still use it for attaching pragmas.
    """
    label = extract_loop_label(for_loop_text)
    if not label or label not in tripcounts_by_label:
        return label  # nothing to check or no entry in kernel_info

    TC_for = tripcounts_by_label[label]  # value from kernel_info.txt

    # Strip debug info
    icmp_inst_ = icmp_inst.split('!dbg')[0]

    # Try to parse predicate and bound from the icmp
    m_pred = re.search(r'icmp\s+(\w+)\s+i\d+\s+[^,]+,\s*([-0-9]+)', icmp_inst_)
    if not m_pred:
        return label

    pred = m_pred.group(1)
    try:
        bound_icmp = int(m_pred.group(2))  # the constant in the compare
    except ValueError:
        # Non-integer bound --> skip check
        return label

    m_init = re.search(r'for\s*\(\s*[^=]+=\s*([0-9]+)', for_loop_text)
    start = int(m_init.group(1)) if m_init else None

    if start is None or pred in {"sge", "sgt"}:
        return label

    # We consider two possible interpretations for the kernel_info value:
    #  1) It may store the **bound** (the constant in the compare).
    #  2) It may store the **tripcount** (number of iterations).
    # If either fits (or is off by 1), we treat it as consistent.

    # 'bound semantics' – kernel_info value == bound in icmp
    bound_ok = (TC_for == bound_icmp)

    # 'tripcount semantics' – derive tripcount from bound + start
    tripcount_icmp = None
    if pred in {"slt", "ult"}:
        tripcount_icmp = max(0, bound_icmp - start)
    elif pred in {"sle", "ule"}:
        tripcount_icmp = max(0, bound_icmp - start + 1)

    trip_ok = (tripcount_icmp is not None and TC_for == tripcount_icmp)

    # Soft tolerance: ignore off-by-one between the two interpretations
    if not bound_ok and not trip_ok:
        if tripcount_icmp is not None:
            if (abs(TC_for - tripcount_icmp) <= 1) or (abs(TC_for - bound_icmp) <= 1):
                return label  # close enough, do not warn

        # Otherwise keep the warning
        print(
            f"[WARN] Tripcount mismatch for loop {for_loop_text} "
            f"label={label}: TC_for={TC_for}, "
            f"icmp_bound={bound_icmp}, start={start}, "
            f"trip_icmp={tripcount_icmp}"
        )

    return label



def create_pragma_nodes(g_nx, g_nx_nodes, kernel_info_file, for_dict_source, for_dict_llvm, log=False):
    """
    Create pragma nodes (loop pragmas + array pragmas) and edges.

    - Loop pragmas are attached to the LLVM icmp node of the corresponding loop.
    - Array pragmas are attached to the LLVM node(s) corresponding to the array variable.
    """
    new_nodes, new_edges = [], []
    next_node_id = g_nx_nodes

    # Load loop tripcounts from kernel_info.txt
    tripcounts_by_label = load_tripcounts_by_label(kernel_info_file)
    eligible_labels = set(tripcounts_by_label.keys())

    # Helper functions
    def resolve_llvm_key(src_func_name: str):
        """
        Find the LLVM function definition that corresponds to src_func_name.
        """
        # Collect candidate matches with different confidence levels
        exact_matches = []
        suffix_matches = []
        substring_matches = []

        for key in for_dict_llvm.keys():
            m = re.search(r'@([^(]+)\s*\(', key)
            if not m:
                continue
            mangled = m.group(1)

            demangled = None
            m2 = re.match(r'_Z(\d+)([A-Za-z_]\w*)', mangled)
            if m2:
                try:
                    name_len = int(m2.group(1))
                    candidate = m2.group(2)
                    # If the length prefix matches the name length, accept it
                    if len(candidate) == name_len:
                        demangled = candidate
                except ValueError:
                    pass

            # Fallback: if we can't demangle, just treat mangled as the name
            if demangled is None:
                demangled = mangled

            # Strongest: exact match on demangled or mangled name
            if demangled == src_func_name or mangled == src_func_name:
                exact_matches.append(key)
                continue

            # Next: demangled or mangled ends with the source name
            if demangled.endswith(src_func_name) or mangled.endswith(src_func_name):
                suffix_matches.append(key)
                continue

            # Weakest: source name appears as a substring
            if src_func_name in mangled or src_func_name in demangled:
                substring_matches.append(key)

        # Choose the best category that gives a unique match
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            print(
                f"[WARN] Multiple LLVM functions matched source function '{src_func_name}' "
                f"(exact): {exact_matches}. Skipping pragmas for this function."
            )
            return None

        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            print(
                f"[WARN] Multiple LLVM functions matched source function '{src_func_name}' "
                f"(suffix): {suffix_matches}. Skipping pragmas for this function."
            )
            return None

        if len(substring_matches) == 1:
            return substring_matches[0]
        if len(substring_matches) > 1:
            print(
                f"[WARN] Multiple LLVM functions matched source function '{src_func_name}' "
                f"(substring): {substring_matches}. Skipping pragmas for this function."
            )
            return None

        # If still nothing, preserve your old fallback behavior
        if len(for_dict_llvm) == 1:
            only_key = next(iter(for_dict_llvm.keys()))
            print(
                f"[WARN] No LLVM function matched source function '{src_func_name}'. "
                f"Falling back to the sole LLVM function '{only_key}'."
            )
            return only_key

        print(
            f"[WARN] Could not match source function '{src_func_name}' "
            f"to any LLVM function. Available: {list(for_dict_llvm.keys())}. "
            f"Skipping pragmas for this function."
        )
        return None


    def find_icmp_node(icmp_inst: str):
        """
        Find the node id (and its block/function) whose full_text matches icmp_inst.
        Returns (node_id, block_id, function_id) or (None, None, None).
        """
        for node, ndata in g_nx.nodes(data=True):
            full_text = None
            if "features" in ndata:
                try:
                    feat = ast.literal_eval(str(ndata["features"]))
                    full_text = feat.get("full_text", [None])[0]
                except Exception:
                    pass
            elif "full_text" in ndata:
                full_text = ndata["full_text"]

            if full_text == icmp_inst:
                block_id = int(ndata.get("block", -1))
                func_id = int(ndata.get("function", -1))
                return int(node), block_id, func_id

        return None, None, None

    # LOOP PRAGMAS
    for f_name, f_content in for_dict_source.items():
        if not f_content:  # no loops/pragmas in this function
            continue

        llvm_key = resolve_llvm_key(f_name)
        if llvm_key is None:
            # We could not map this source function to an LLVM function; skip it.
            continue

        llvm_content = for_dict_llvm[llvm_key]

        for for_loop_id, payload in f_content.items():
            # Unify payload format
            if isinstance(payload, dict):
                for_loop_text = payload["loop_line"]
                pragmas = payload["pragmas"]
                local_id = payload.get("local_id", for_loop_id)
            else:
                # Original format: {local_id: [loop_line, pragmas]}
                for_loop_text, pragmas = payload
                local_id = for_loop_id

            # Ensure we have a matching LLVM loop entry
            if local_id not in llvm_content:
                print(
                    f"[WARN] LLVM loop id {local_id} not found for function '{f_name}' "
                    f"(available: {list(llvm_content.keys())}). Skipping this loop."
                )
                continue

            icmp_inst = llvm_content[local_id][0]

            # Tripcount consistency check + label extraction
            label = check_tripcount_consistency(for_loop_text, icmp_inst, tripcounts_by_label)

            # Skip labeled loops that do not appear in kernel_info.txt
            if label and label not in eligible_labels:
                continue

            # Find the LLVM node corresponding to icmp_inst
            node_id, block_id, function_id = find_icmp_node(icmp_inst)
            if node_id is None:
                print(f"[WARN] icmp instruction not found in graph: {icmp_inst}")
                continue

            # Create pragma nodes for each pragma line attached to this loop
            for pragma in pragmas:
                tokens = pragma.split()
                if len(tokens) < 3:
                    print(f"[WARN] Unexpected pragma format (too few tokens): {pragma}")
                    continue

                pragma_kind = tokens[2].upper()  # "PIPELINE", "UNROLL", etc.
                if pragma_kind not in PRAGMA_POSITION:
                    print(f"[WARN] Skipping unknown pragma kind: {pragma_kind}")
                    continue

                p_dict = {
                    "type": 100,
                    "block": block_id,
                    "function": function_id,
                    "features": {"full_text": [pragma]},
                    "text": pragma_kind,
                }

                new_nodes.append((next_node_id, p_dict))

                e_attr = {"flow": 200, "position": PRAGMA_POSITION[pragma_kind]}
                # bidirectional edges between loop icmp node and pragma node
                new_edges.append((node_id, next_node_id, e_attr))
                new_edges.append((next_node_id, node_id, e_attr))

                next_node_id += 1

    # ARRAY PRAGMAS
    src_dir = os.path.dirname(kernel_info_file)
    kernel_name = os.path.basename(src_dir)

    # Support both C and C++ placeholder files: <kernel>_placeholders.c/.cpp
    placeholder_src = None
    for ext in (".cpp", ".c"):
        candidate = os.path.join(src_dir, f"{kernel_name}_placeholders{ext}")
        if os.path.isfile(candidate):
            placeholder_src = candidate
            break

    if placeholder_src:
        array_pragmas = get_pragmas_arrays(placeholder_src, log=log)

        for ap in array_pragmas:
            varname = ap["var"]
            pragma_line = ap["pragma"]

            matched_nodes = []
            decl_candidates = []  # prefer nodes that look like the declaration

            for node, ndata in g_nx.nodes(data=True):
                full_text = None
                if "features" in ndata:
                    try:
                        feat = ast.literal_eval(str(ndata["features"]))
                        full_text = feat.get("full_text", [None])[0]
                    except Exception:
                        pass
                elif "full_text" in ndata:
                    full_text = ndata["full_text"]

                if not full_text:
                    continue

                ft_str = str(full_text)
                if varname not in ft_str:
                    continue

                matched_nodes.append((node, ndata))

                # array declaration in LLVM looks like %var = alloca [N x <type>], ...
                if "alloca" in ft_str and "[" in ft_str and "]" in ft_str:
                    decl_candidates.append((node, ndata))

            if not matched_nodes:
                if log:
                    print(f"[WARN] No variable node found for array '{varname}' "
                          f"(pragma: {pragma_line})")
                continue

            # Prefer the declaration node, otherwise first match
            if decl_candidates:
                node0, data0 = decl_candidates[0]
            else:
                node0, data0 = matched_nodes[0]

            block_id = int(data0.get("block", -1))
            function_id = int(data0.get("function", -1))

            p_dict = {
                "type": 100,
                "block": block_id,
                "function": function_id,
                "features": {"full_text": [pragma_line]},
                "text": "ARRAY_PARTITION",
            }
            new_nodes.append((next_node_id, p_dict))

            e_attr = {"flow": 200, "position": PRAGMA_POSITION["ARRAY_PARTITION"]}

            # Attach pragma node to all uses/declaration nodes of that array
            for node, _ in matched_nodes:
                new_edges.append((node, next_node_id, e_attr))
                new_edges.append((next_node_id, node, e_attr))

            next_node_id += 1

    if log:
        pprint(new_nodes)
        pprint(new_edges)

    return new_nodes, new_edges



def prune_redundant_nodes(g_new):
    while True:
        remove_nodes = set()
        for node in g_new.nodes():
            if len(list(g_new.neighbors(node))) == 0 or node is None:
                print(node)
                remove_nodes.add(node)
        for node in remove_nodes:
            g_new.remove_node(node)
        if not remove_nodes:
            break


def process_graph(name, g, csv_dict=None):
    '''
        adjusts the node/edge attributes, removes redundant nodes,
            and writes the final graph to be used by GNN-DSE

        args:
            name: kernel name
            dest: where to store the graph
    '''
    g_new = nx.MultiDiGraph()
    for node, ndata in g.nodes(data=True):
        attrs = deepcopy(ndata)
        if 'features' in ndata:
            feat = ndata['features']
            attrs['full_text'] = feat['full_text'][0]
            del attrs['features']

        g_new.add_node(node)
        nx.set_node_attributes(g_new, {node: attrs})

    edge_list = []
    id = 0
    for nid1, nid2, edata in g.edges(data=True):
        edata['id'] = id
        edge_list.append((nid1, nid2, edata))
        id += 1
    g_new.add_edges_from(edge_list)

    prune_redundant_nodes(g_new)

    original_gexf_folder = join(processed_gexf_folder, 'original')
    create_dir_if_not_exists(original_gexf_folder)
    new_gexf_file = join(original_gexf_folder, f'{name}_processed_result.gexf')
    # if len(g_new.nodes) != len(g.nodes):
        # print('#nodes:', len(g_new.nodes), 'before processing was:', len(g.nodes))
    # if len(g_new.edges) != len(g.edges):
        # print('#edges:', len(g_new.edges), 'before processing was:', len(g.edges))
    os.makedirs(processed_gexf_folder, exist_ok=True)
    nx.write_gexf(g_new, new_gexf_file)
    current_g_value = {}
    current_g_value['num_node'] = len(g_new.nodes)
    current_g_value['num_edge'] = len(g_new.edges)
    current_g_value['name'] = name
    if csv_dict: csv_dict[name] = current_g_value



def graph_generator(name, path, benchmark, src_ext="cpp", generate_programl=False, csv_dict=None, top_func=None):
    """
        runs ProGraML [ICML'21] to generate the graph, adds the pragma nodes,
            processes the final graph to be accepted by GNN-DSE

        args:
            name: kernel name
            path: path to parent directory of the kernel file
            benchmark: [machsuite|poly] None: simple program
            src_ext: "c" or "cpp" (extension *without* dot) for the placeholder file
    """
    ## generate PrograML graph
    if generate_programl:
        cmd = ["/bin/bash", f"{get_root_path()}/src/clang_script.sh", str(name), str(path), str(type_graph)]
        p = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
        out, err = p.communicate()
        print("returncode:", p.returncode)
        print("stdout:\n", out)
        print("stderr:\n", err)

    ## convert it to networkx format
    g_nx = llvm_to_nx(join(path, name))
    g_nx_nodes, g_nx_edges = g_nx.number_of_nodes(), len(g_nx.edges)

    ## find for loops and icmp instructions in llvm code
    for_dict_llvm, for_count_llvm = get_icmp(path, name)

    ## find for loops and their pragmas in the C/C++ code (placeholder file)
    for_dict_source, for_count_source = get_pragmas_loops(
        path,
        f'{name}_placeholders',
        EXT=src_ext,          # works with both .c and .cpp
    )
    assert for_count_llvm == for_count_source, (
        f'the number of for loops from the LLVM code and source code do not match '
        f'{for_count_llvm} in llvm vs {for_count_source} in the code'
    )

    print(f'number of nodes: {g_nx_nodes} and number of edges: {g_nx_edges}')
    graph_path = join(path, name+'.gexf')
    nx.write_gexf(g_nx, graph_path)

    augment_graph = True
    if augment_graph:
        kernel_info_file = os.path.join(path, 'kernel_info.txt')
        new_nodes, new_edges = create_pragma_nodes(
            g_nx, g_nx_nodes, kernel_info_file, for_dict_source, for_dict_llvm
        )

        add_to_graph(g_nx, new_nodes, new_edges)
        print(f'number of new nodes: {g_nx.number_of_nodes()} and number of new edges: {len(g_nx.edges)}')
        process = True
        if process:
            process_graph(name, g_nx, csv_dict)

    copy_files_ = True
    if generate_programl:
        copy_files = True
    local = True  # True: programl is running in the directories inside this project
    if copy_files_:
        if not local:
            dest = join(os.getcwd(), f'{type_graph}', benchmark, name)
            create_dir_if_not_exists(dest)
            copy_files(name, path, dest)
        else:
            dest = path



def get_for_blocks_info(name, path):
    with open(join(path, f'{name}.ll'), 'r') as f_llvm:
        lines_llvm = f_llvm.readlines()

    for_blocks_info = OrderedDict() # label: {ind: loop number, preds:, next_instr:, line_num:, end: [(for.end line num, for.end label)], possible_children: children:}
    # possible_children are all children, children is only first level children
    # check up to 3 next instr to make sure the block is correct
    for_stack = [] # push for.cond pop for.end
    for_start = []
    for_end = []
    for_label = []
    i = 0
    for idx, line in enumerate(lines_llvm):
        if line.startswith('for.'):
            content = line.strip().split(';')
            line = content[0].strip()
            if 'for.cond' in line:
                key = f'{line}{idx}'
                assert key not in for_blocks_info
                for_blocks_info[key] = {'ind': i, 'preds': content[1], 'next_instr': [lines_llvm[idx+1].strip(), lines_llvm[idx+2].strip(),lines_llvm[idx+3].strip()], 'line_num': idx}
                for_stack.append(key)
                i += 1
            elif 'for.end' in line:
                res_cond = for_stack.pop()
                assert res_cond in for_blocks_info
                for_blocks_info[res_cond]['end'] = (idx, line)

    for for_l, for_l_value in for_blocks_info.items():
        if 'cond' in for_l:
            for_start.append(for_l_value['line_num'])
            for_end.append(for_l_value['end'][0])
            for_label.append(for_l)
    for idx, start_num in enumerate(for_start):
        child_idx = idx + 1
        possible_children = []
        for s, e in zip(for_start[idx+1:], for_end[idx+1:]):
            if s > start_num and e < for_end[idx]:
                possible_children.append(for_label[child_idx])
                child_idx += 1
            else:
                break
        for_blocks_info[for_label[idx]]['possible_children'] = possible_children

    for for_l, for_l_value in for_blocks_info.items():
        possible_children = for_l_value['possible_children']
        children = []
        i = 0
        while i < len(possible_children):
            children.append(possible_children[i])
            i += len(for_blocks_info[possible_children[i]]['possible_children']) + 1
        for_l_value['children'] = children

    return for_blocks_info


def augment_graph_hierarchy(name, for_blocks_info, src_path, dst_path, csv_dict=None, node_type = 'block'):
    if node_type == 'block':
        gexf_file = join(src_path, f'{name}_processed_result.gexf')
        new_gexf_file = join(dst_path, f'{name}_processed_result.gexf')
        print(f'processing {gexf_file}')
        g = nx.readwrite.gexf.read_gexf(gexf_file)
        g_nx_nodes, g_nx_edges = g.number_of_nodes(), len(g.edges)
        print(f'started with {g_nx_nodes} nodes and {g_nx_edges} edges')
        new_edges = [(nid1, nid2, edata) for nid1, nid2, edata in g.edges(data=True)]
        new_nodes = [(node, ndata) for node, ndata in g.nodes(data=True)]
        g_new = nx.MultiDiGraph()
        id = g_nx_edges
        current_g_value = {}
        current_g_value['name'] = name
        current_g_value['prev_node'] = g_nx_nodes
        current_g_value['prev_edge'] = g_nx_edges

        block_ids = {}
        for for_l, for_l_value in for_blocks_info.items():
            ## get block_id of each for loop
            found = False
            # print(for_l_value)
            for node, ndata in g.nodes(data = True):
                if 'full_text' not in ndata:
                    continue
                if for_l_value['next_instr'][0] in ndata['full_text']:
                    block, function = (ndata['block'], ndata['function'])
                    correct_node = 1
                    for neighbor in g.neighbors(node):
                        if for_l_value['next_instr'][1] in g.nodes[neighbor]['full_text']:
                            correct_node += 1
                        if correct_node == 2:
                            for nn in g.neighbors(neighbor):
                                if for_l_value['next_instr'][2] in g.nodes[nn]['full_text']:
                                    correct_node += 1
                                    break
                        if correct_node == 3:
                            break
                    if correct_node == 3: ### FIXME: should match the number of next instrs that were saved:
                        found = True
                        block_ids[for_l] = (block, function)
                        break
            if not found:
                print(f'could not find the respective block for label {for_l}')
                raise RuntimeError()


        node_ids_block = {}
        for for_l in for_blocks_info:
            ## find pseudo node of each for loop
            for node, ndata in g.nodes(data = True):
                if 'pseudo_block' not in ndata['text']:
                    continue
                if ndata['block'] == block_ids[for_l][0] and ndata['function'] == block_ids[for_l][1]:
                    node_ids_block[for_l] = node
                    break

        for for_l, for_l_value in for_blocks_info.items():
            ## connect hierarchy edges
            if len(for_l_value['children']) == 0:
                continue
            id1 = node_ids_block[for_l]
            position = 0
            for child in for_l_value['children']:
                id2 = node_ids_block[child]
                e_dict = {'id': id, 'flow': 6, 'position': position} ## assign a new flow to it
                new_edges.append((id1, id2, e_dict))
                id += 1
                e_dict = {'id': id, 'flow': 6, 'position': position}
                new_edges.append((id2, id1, e_dict))
                id += 1
                position += 1
                print(f'edge between {for_l} with id {id1} and {child} and id {id2}')


        add_to_graph(g_new, nodes = new_nodes, edges = new_edges)
        prune_redundant_nodes(g_new)
        g_nx_nodes, g_nx_edges = g_new.number_of_nodes(), len(g_new.edges)
        print(f'ending with {g_nx_nodes} nodes and {g_nx_edges} edges')
        current_g_value['new_node'] = g_nx_nodes
        current_g_value['new_edge'] = g_nx_edges
        if csv_dict: csv_dict[name] = current_g_value
        nx.write_gexf(g_new, new_gexf_file)

    else:
        raise NotImplementedError()



def add_auxiliary_nodes(name, path, processed_path, csv_dict, node_type = 'block', connected = False):
    if node_type == 'block':
        gexf_file = join(path, 'original', f'{name}_processed_result.gexf')
        new_gexf_file = join(processed_path, f'{name}_processed_result.gexf')
        if not isfile(gexf_file):
            print(f'Processed graph not found for kernel "{name}": {gexf_file} — skipping')
            return None
        print(f'processing {gexf_file}')
        g = nx.readwrite.gexf.read_gexf(gexf_file)
        g_nx_nodes, g_nx_edges = g.number_of_nodes(), len(g.edges)
        print(f'started with {g_nx_nodes} nodes and {g_nx_edges} edges')
        current_g_value = {}
        current_g_value['name'] = name
        current_g_value['prev_node'] = g_nx_nodes
        current_g_value['prev_edge'] = g_nx_edges
        orig_nodes = g_nx_nodes
        block_nodes = {}
        new_edges = [(nid1, nid2, edata) for nid1, nid2, edata in g.edges(data=True)]
        new_nodes = [(node, ndata) for node, ndata in g.nodes(data=True)]
        block_func = {}
        block_func = {}
        max_block = 0
        g_new = nx.MultiDiGraph()
        id = g_nx_edges

        for node, ndata in g.nodes(data = True):
            if f"function-{ndata['function']}-block-{ndata['block']}" not in block_nodes:
                new_node = create_pseudo_node_block(ndata['block'], ndata['function'])
                block_nodes[f"function-{ndata['function']}-block-{ndata['block']}"] = {'id': g_nx_nodes, 'node': new_node, 'last_position': 0}
                new_nodes.append((g_nx_nodes, new_node.get_attr(after_process = True)))
                g_nx_nodes += 1

            if ndata['function'] not in block_func:
                block_func[ndata['function']] = {}
                block_func[ndata['function']]['count'] = 1
                block_func[ndata['function']]['blocks'] = [ndata['block']]
            else:
                if ndata['block'] not in block_func[ndata['function']]['blocks']:
                    block_func[ndata['function']]['count'] += 1
                    block_func[ndata['function']]['blocks'].append(ndata['block'])

            key = f"function-{ndata['function']}-block-{ndata['block']}"
            pseudo_node = block_nodes[key]['node']
            pseudo_id = block_nodes[key]['id']
            pseudo_position = block_nodes[key]['last_position']
            assert pseudo_node.function == ndata['function']
            e_dict = {'id': id, 'flow': 4, 'position': pseudo_position}
            new_edges.append((node, pseudo_id, e_dict))
            id += 1
            e_dict = {'id': id, 'flow': 4, 'position': pseudo_position}
            new_edges.append((pseudo_id, node, e_dict))
            id += 1
            block_nodes[key]['last_position'] = pseudo_position + 1

        if connected:
            ## add edge between the new nodes
            sorted_nodes = sorted(block_nodes.keys(), key=natural_keys)
            for idx, node in enumerate(sorted_nodes[:-1]):
                id1 = block_nodes[node]['id']
                id2 = block_nodes[sorted_nodes[idx+1]]['id']
                e_dict = {'id': id, 'flow': 5, 'position': 0} ## assign a new flow to it
                new_edges.append((id1, id2, e_dict))
                id += 1
                e_dict = {'id': id, 'flow': 5, 'position': 0}
                new_edges.append((id2, id1, e_dict))
                id += 1


        add_to_graph(g_new, nodes = new_nodes, edges = new_edges)
        prune_redundant_nodes(g_new)
        g_nx_nodes, g_nx_edges = g_new.number_of_nodes(), len(g_new.edges)
        for f, b in block_func.items():
            # print(f, b)
            max_block += b['count']
        assert g_nx_nodes == orig_nodes + max_block
        print(f'ending with {g_nx_nodes} nodes and {g_nx_edges} edges, max block: {max_block}')
        current_g_value['new_node'] = g_nx_nodes
        current_g_value['new_edge'] = g_nx_edges
        current_g_value['block'] = max_block
        if csv_dict: csv_dict[name] = current_g_value
        nx.write_gexf(g_new, new_gexf_file)

    else:
        raise NotImplementedError()


def load_kernel_source_map(csv_path):
    """
    Load a mapping from kernel directory name (app_name) -> metadata.

    Returns:
        mapping[app_name] = {
            "file_name": "<source file inside ApplicationDataset>",
            "ext": ".c" or ".cpp",
            "top": "<top_level_function>",
        }
    """
    mapping = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        required = {"app_name", "top_level_function", "file_name", "file_name_extension"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        for row in reader:
            app = row["app_name"].strip()
            fname = row["file_name"].strip()
            ext_col = row["file_name_extension"].strip()
            top = row["top_level_function"].strip()

            if not app or not fname:
                continue

            # Derive extension robustly
            base, ext = os.path.splitext(fname)
            if not ext and ext_col:
                ext = "." + ext_col
            elif ext and ext_col and ext.lstrip(".") != ext_col:
                # Optional: warn if CSV extension disagrees with file_name suffix
                # print(f"[WARN] Mismatch in extension for {app}: {fname} vs {ext_col}")
                pass

            if not ext:
                raise ValueError(f"Could not determine extension for app '{app}', file '{fname}'")

            mapping[app] = {
                "file_name": fname,
                "ext": ext,       # includes the dot, e.g. ".c" or ".cpp"
                "top": top,
            }

    return mapping


def remove_extra_header(src_dir, kernel_name):
    from tempfile import mkstemp
    orig_file = join(src_dir, f'{kernel_name}.c')
    fnew, abs_path = mkstemp()
    with open(fnew, 'w') as fpnew:
        with open(orig_file) as fp:
            for line in fp:
                if line.startswith('#include') and 'merlin_type_define' in line:
                    continue
                fpnew.write(line)

    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)

def write_csv_file(csv_dict, csv_header, file_path):
    with open(join(get_root_path(), file_path), mode = 'w') as f:
        f_writer = csv.DictWriter(f, fieldnames=csv_header)
        f_writer.writeheader()
        for d, value in csv_dict.items():
            if d == 'header':
                continue
            f_writer.writerow(value)

def normalize_part(p):
    # Strip trailing digits and letters after digits, e.g. 'stencil2d' -> 'stencil'
    return re.sub(r'\d.*$', '', p)


def run_graph_gen(mode='initial', connected=True):
    test = 'original'
    global processed_gexf_folder
    base_dataset_dir = "/home/elvouvali/Data4LLMPrompting/ApplicationDataset"
    csvs_dir = "/home/elvouvali/Data4LLMPrompting/preprocessed_CSVS"
    source_map_csv = "/home/elvouvali/Data4LLMPrompting/ApplicationInformation.csv"

    if mode == 'initial':
        source_map = load_kernel_source_map(source_map_csv)

    if mode == 'initial':
        csv_header = ['name', 'num_node', 'num_edge']
    else:
        csv_header = ['name', 'prev_node', 'prev_edge', 'new_node', 'new_edge']
    if mode == 'auxiliary':
        csv_header.append('block')
    csv_dict = {'header': csv_header}

    if mode == 'initial':
        for kernel in sorted(os.listdir(base_dataset_dir)):
            kernel_path = os.path.join(base_dataset_dir, kernel)

            # get the source file info from the CSV mapping
            if kernel not in source_map:
                raise RuntimeError(
                    f"No source mapping found in CSV for kernel '{kernel}'"
                )
            info = source_map[kernel]
            orig_src_name = info["file_name"]
            ext = info["ext"]
            src_ext = ext.lstrip('.')
            top_func = info["top"]

            src_path = os.path.join(kernel_path, orig_src_name)
            if not os.path.isfile(src_path):
                raise FileNotFoundError(
                    f"Mapped source file '{src_path}' does not exist "
                    f"for kernel '{kernel}'"
                )

            header_files = list(iglob(os.path.join(kernel_path, "*.h"), recursive=False))
            kernel_info_file = glob.glob(os.path.join(kernel_path, "kernel_info.txt"))[0]

            print('####################')
            print('Now processing', kernel)
            harp_kernel_dir = join(get_root_path(), f'{type_graph}/{kernel}')

            if not exists(harp_kernel_dir):
                create_dir_if_not_exists(harp_kernel_dir)

                # Copy the source as <kernel>.c or <kernel>.cpp into harp/...
                new_src_path = os.path.join(harp_kernel_dir, f"{kernel}{ext}")
                shutil.copyfile(src_path, new_src_path)

                # copy headers as before
                for header_file in header_files:
                    new_header_path = os.path.join(harp_kernel_dir, os.path.basename(header_file))
                    shutil.copyfile(header_file, new_header_path)

                new_kernel_info_path = os.path.join(harp_kernel_dir, 'kernel_info.txt')
                shutil.copyfile(kernel_info_file, new_kernel_info_path)

                # insert placeholders on the *copied* file (kernel.c or kernel.cpp)
                placeholder_lines = insert_placeholders(new_src_path)
                placeholders_src_path = os.path.join(harp_kernel_dir, f"{kernel}_placeholders{ext}")
                with open(placeholders_src_path, "w") as f:
                    f.writelines(placeholder_lines)

            if not exists(processed_gexf_folder):
                create_dir_if_not_exists(processed_gexf_folder)

            graph_generator(
                kernel,
                harp_kernel_dir,
                kernel,
                src_ext=src_ext,
                generate_programl=True,
                csv_dict=csv_dict,
                top_func=top_func,
            )
            write_csv_file(csv_dict, csv_header, f'{type_graph}/{mode}.csv')

    elif mode == 'auxiliary':
        if connected:
            auxiliary_node_gexf_folder = join(
                get_root_path(),
                f'{type_graph}/processed/extended-pseudo-block-connected/'
            )
        else:
            auxiliary_node_gexf_folder = join(
                get_root_path(),
                f'{type_graph}/processed/extended-pseudo-block-base/'
            )
        create_dir_if_not_exists(auxiliary_node_gexf_folder)

        for kernel in sorted(os.listdir(base_dataset_dir)):
            print('####################')
            print('Now processing', kernel)
            # We *don't* need the source file anymore here; graphs already exist.
            add_auxiliary_nodes(
                kernel,
                processed_gexf_folder,
                auxiliary_node_gexf_folder,
                csv_dict=csv_dict,
                node_type='block',
                connected=connected
            )
            print()

        write_csv_file(csv_dict, csv_header, f'{type_graph}/{mode}_{connected}.csv')

    elif mode == 'hierarchy':
        auxiliary_node_gexf_folder = join(
            get_root_path(),
            f'{type_graph}/processed/extended-pseudo-block-connected/'
        )
        dest_path = join(
            get_root_path(),
            f'{type_graph}/processed/extended-pseudo-block-connected-hierarchy/'
        )
        create_dir_if_not_exists(dest_path)
        assert exists(auxiliary_node_gexf_folder)

        for kernel in sorted(os.listdir(base_dataset_dir)):
            llvm_kernel_dir = join(get_root_path(), f'{type_graph}/{kernel}')
            print('####################')
            print('now processing', kernel)
            for_blocks_info = get_for_blocks_info(kernel, llvm_kernel_dir)
            augment_graph_hierarchy(
                kernel,
                for_blocks_info,
                src_path=auxiliary_node_gexf_folder,
                dst_path=dest_path,
                csv_dict=csv_dict
            )
            print()

        write_csv_file(csv_dict, csv_header, f'{type_graph}/{mode}.csv')

    else:
        raise NotImplementedError()



##### Usage ######
run_graph_gen(mode='initial', connected=True)
run_graph_gen(mode='auxiliary', connected=False)
run_graph_gen(mode='auxiliary', connected=True)
run_graph_gen(mode='hierarchy', connected=True)

