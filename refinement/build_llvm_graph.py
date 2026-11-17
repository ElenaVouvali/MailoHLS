#!/usr/bin/env python3
"""Generate a HARP/LLVM GEXF via the ORIGINAL llvm-ir graph generator on liono.

Requires original LLVM10-compatible ProGraML env (not the Vitis synthesis env).
Does not modify the archival repo; cloned analysis inputs are isolated.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--llvm-repo', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--original', required=True, help='Pragma-free C/C++ source')
    p.add_argument('--placeholder', required=True, help='Lk-annotated template')
    p.add_argument('--kernel-info', required=True)
    p.add_argument('--headers-dir', required=True)
    p.add_argument('--work-root', required=True)
    p.add_argument('--out-graph', required=True)
    a=p.parse_args()
    repo=Path(a.llvm_repo).expanduser().resolve()
    root=Path(a.work_root).expanduser().resolve()
    work=root/'harp'/a.name
    out=Path(a.out_graph).expanduser().resolve()
    if root.exists() or out.exists():
        raise RuntimeError('Use fresh work-root and out-graph; no implicit overwrites')
    if not (repo/'GNN_branch'/'graph_gen_deterministic.py').is_file():
        raise RuntimeError('Missing llvm-ir graph generator')
    work.mkdir(parents=True)
    (root/'src').mkdir()
    (root/'src'/'clang_script.sh').symlink_to(repo/'GNN_branch'/'clang_script.sh')
    suffix=Path(a.original).suffix
    if suffix not in ('.cpp', '.c') or Path(a.placeholder).suffix != suffix:
        raise RuntimeError('Mismatched .cpp or .c sources')
    shutil.copy2(a.original, work/(a.name+suffix))
    shutil.copy2(a.placeholder, work/(a.name+'_placeholders'+suffix))
    shutil.copy2(a.kernel_info, work/'kernel_info.txt')
    for header in sorted(Path(a.headers_dir).iterdir()):
        if header.is_file() and header.suffix.lower() in ('.h','.hpp','.inc'):
            shutil.copy2(header,work/header.name)
    sys.path.insert(0,str(repo/'GNN_branch'))
    import graph_gen_deterministic as gg
    gg.get_root_path=lambda: str(root)
    gg.processed_gexf_folder=str(root/'harp'/'processed')
    gg.graph_generator(a.name,str(work),a.name,src_ext=suffix.lstrip('.'),
                       generate_programl=True)
    graph=root/'harp'/'processed'/'original'/(a.name+'_processed_result.gexf')
    if not graph.is_file():
        raise RuntimeError('Original graph builder did not generate '+str(graph))
    out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(graph,out)
    print('LLVM/ProGraML graph:',out)
    print('Original source, placeholder source, headers and LLVM IR:',work)
    print('NOTE: copied analysis-only ap_fixed.h/ap_int.h are ONLY for graph generation;'
          ' CG-BDR excludes them from actual Vitis synthesis.')

if __name__=='__main__':
    main()
