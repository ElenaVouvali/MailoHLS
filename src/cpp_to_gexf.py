#!/usr/bin/env python3
import os
import shutil
from os.path import join
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Optional

import networkx as nx

import graph_gen_deterministic as gg
from insert_placeholders import insert_placeholders


def create_pragma_nodes_no_kernel_info(
    g_nx: nx.MultiDiGraph,
    g_nx_nodes: int,
    for_dict_source,
    for_dict_llvm,
    placeholder_src_file: str,
    log: bool = False,
):
    """
    Minimal clone of graph_gen.create_pragma_nodes(), but WITHOUT kernel_info.txt.

    Behavior differences vs original:
      - No tripcount consistency check
      - No eligible-label filtering (all loops are eligible)
      - Array pragmas are still supported, parsed from placeholder_src_file
    """
    import re
    import ast

    new_nodes, new_edges = [], []
    next_node_id = g_nx_nodes

    def resolve_llvm_key(src_func_name: str):
        # Same logic as graph_gen.py (kept compact)
        exact, suffix, substr = [], [], []
        for key in sorted(for_dict_llvm.keys()):
            m = re.search(r'@([^(]+)\s*\(', key)
            if not m:
                continue
            mangled = m.group(1)
            demangled = mangled
            m2 = re.match(r'_Z(\d+)([A-Za-z_]\w*)', mangled)
            if m2:
                try:
                    nlen = int(m2.group(1))
                    cand = m2.group(2)
                    if len(cand) == nlen:
                        demangled = cand
                except ValueError:
                    pass

            if demangled == src_func_name or mangled == src_func_name:
                exact.append(key)
            elif demangled.endswith(src_func_name) or mangled.endswith(src_func_name):
                suffix.append(key)
            elif src_func_name in demangled or src_func_name in mangled:
                substr.append(key)

        if len(exact) == 1:
            return sorted(exact)[0]
        if len(suffix) == 1:
            return sorted(suffix)[0]
        if len(substr) == 1:
            return sorted(substr)[0]
        if len(for_dict_llvm) == 1:
            return sorted(for_dict_llvm.keys())[0]
        return None

    def find_icmp_node(icmp_inst: str):
        # Same matching strategy as graph_gen.py
        sorted_nodes = sorted(g_nx.nodes(data=True), key=lambda x: gg._node_sort_key(x[0], x[1]))
        for node, ndata in sorted_nodes:
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
                return int(node), int(ndata.get("block", -1)), int(ndata.get("function", -1))
        return None, None, None

    # LOOP PRAGMAS: attach each pragma line to the icmp node for that loop
    for f_name in sorted(for_dict_source.keys()):
        f_content = for_dict_source[f_name]
        if not f_content:
            continue

        llvm_key = resolve_llvm_key(f_name)
        if llvm_key is None:
            continue
        llvm_content = for_dict_llvm[llvm_key]

        for for_loop_id, payload in f_content.items():
            if isinstance(payload, dict):
                pragmas = payload["pragmas"]
                local_id = payload.get("local_id", for_loop_id)
            else:
                _, pragmas = payload
                local_id = for_loop_id

            if local_id not in llvm_content:
                continue

            icmp_inst = llvm_content[local_id][0]
            node_id, block_id, function_id = find_icmp_node(icmp_inst)
            if node_id is None:
                continue

            for pragma in pragmas:
    
                tokens = pragma.split()
                if len(tokens) < 3:
                    continue
                pragma_kind = tokens[2].upper()
                if pragma_kind not in gg.PRAGMA_POSITION:
                    continue

                p_dict = {
                    "type": 100,
                    "block": block_id,
                    "function": function_id,
                    "features": {"full_text": [pragma]},
                    "text": pragma_kind,
                }
                new_nodes.append((next_node_id, p_dict))

                e_attr = {"flow": 200, "position": gg.PRAGMA_POSITION[pragma_kind]}
                new_edges.append((node_id, next_node_id, e_attr))
                new_edges.append((next_node_id, node_id, e_attr))
                next_node_id += 1

    # ARRAY PRAGMAS: reuse the existing parser in graph_gen.py
    if os.path.isfile(placeholder_src_file):
        array_pragmas = gg.get_pragmas_arrays(placeholder_src_file, log=log)
        for ap in array_pragmas:
            varname = ap["var"]
            pragma_line = ap["pragma"]

            matched_nodes = []
            decl_candidates = []

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

                ft = str(full_text)
                if varname not in ft:
                    continue
                matched_nodes.append((node, ndata))
                if "alloca" in ft and "[" in ft and "]" in ft:
                    decl_candidates.append((node, ndata))

            if not matched_nodes:
                continue

#            node0, data0 = decl_candidates[0] if decl_candidates else matched_nodes[0]
            if decl_candidates:
                decl_candidates.sort(key=lambda x: gg._node_sort_key(x[0], x[1]))
                node0, data0 = decl_candidates[0]
            elif matched_nodes:
                matched_nodes.sort(key=lambda x: gg._node_sort_key(x[0], x[1]))
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
            e_attr = {"flow": 200, "position": gg.PRAGMA_POSITION["ARRAY_PARTITION"]}
            
            matched_nodes.sort(key=lambda x: gg._node_sort_key(x[0], x[1]))

            for node, _ in matched_nodes:
                new_edges.append((node, next_node_id, e_attr))
                new_edges.append((next_node_id, node, e_attr))
            next_node_id += 1

    return new_nodes, new_edges


def cpp_to_gexf(
    name: str,
    path: str,
    src_ext: str,
    out_gexf: str,
):
    """
    Close to graph_gen.graph_generator(), but does NOT require kernel_info.txt.
    Writes the final connected-hierarchy gexf (optionally copied to out_gexf).
    """
    path = str(Path(path).resolve())

    # 1) Compile -> .ll
    cmd = ["/bin/bash", f"{gg.get_root_path()}/src/clang_script.sh", str(name), str(path), str(gg.type_graph)]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"clang_script.sh failed\nstdout:\n{out}\nstderr:\n{err}")

    # 2) Ensure placeholders exist (graph_gen expects <name>_placeholders.<ext>)
    src_file = join(path, f"{name}.{src_ext}")
    if not os.path.isfile(src_file):
        raise FileNotFoundError(f"Missing source file: {src_file}")

    placeholders_file = join(path, f"{name}_placeholders.{src_ext}")
    if not os.path.isfile(placeholders_file):
        placeholder_lines = insert_placeholders(src_file)
        with open(placeholders_file, "w", encoding="utf-8") as f:
            f.writelines(placeholder_lines)

    # 3) LLVM -> NetworkX
    g_nx = gg.llvm_to_nx(join(path, name))
    g_nx = gg.canonicalize_for_gexf(g_nx)
    g_nx_nodes = g_nx.number_of_nodes()

    # 4) Loop mapping (LLVM icmp vs source loops)
    for_dict_llvm, for_count_llvm = gg.get_icmp(path, name)
    for_dict_source, for_count_source = gg.get_pragmas_loops(
        path, f"{name}_placeholders", EXT=src_ext
    )

    if for_count_llvm != for_count_source:
        raise RuntimeError(
            f"for-loop count mismatch: llvm={for_count_llvm} vs source={for_count_source}. "
            "This must match for stable pragma attachment."
        )

    # 5) Add pragma nodes WITHOUT kernel_info.txt
    new_nodes, new_edges = create_pragma_nodes_no_kernel_info(
        g_nx, g_nx_nodes, for_dict_source, for_dict_llvm, placeholders_file
        )
    gg.add_to_graph(g_nx, new_nodes, new_edges)
    gg.assign_deterministic_edge_ids(g_nx)

    # 6) Process graph to processed/original/<name>_processed_result.gexf
    gg.process_graph(name, g_nx, csv_dict=None)

    # 7) Pseudo-block nodes (connected) -> processed/extended-pseudo-block-connected/
    aux_dir = Path(gg.get_root_path()) / f"{gg.type_graph}/processed/extended-pseudo-block-connected"
    aux_dir.mkdir(parents=True, exist_ok=True)
    gg.add_auxiliary_nodes(
        name,
        gg.processed_gexf_folder,
        str(aux_dir),
        csv_dict=None,
        node_type="block",
        connected=True,
    )

    # 8) Hierarchy edges -> processed/extended-pseudo-block-connected-hierarchy/
    hier_dir = Path(gg.get_root_path()) / f"{gg.type_graph}/processed/extended-pseudo-block-connected-hierarchy"
    hier_dir.mkdir(parents=True, exist_ok=True)
    for_blocks_info = gg.get_for_blocks_info(name, path)
    gg.augment_graph_hierarchy(
        name,
        for_blocks_info,
        src_path=str(aux_dir),
        dst_path=str(hier_dir),
        csv_dict=None,
        node_type="block",
    )

    final_gexf = hier_dir / f"{name}_processed_result.gexf"
    if out_gexf:
        out_path = Path(out_gexf).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final_gexf, out_path)
        final_gexf = out_path

    # NetworkX read_gexf returns MultiDiGraph if parallel edges exist. :contentReference[oaicite:1]{index=1}
    return nx.readwrite.gexf.read_gexf(str(final_gexf))



g = cpp_to_gexf(
    name='gemm',
    path='LLM_predictions',
    src_ext='c',
    out_gexf='LLM_predictions/machsuite-gemm-blocked_connected_hierarchy.gexf'
    )

print(f"OK: nodes={g.number_of_nodes()} edges={g.number_of_edges()}")


