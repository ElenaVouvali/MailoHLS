#!/usr/bin/env python3
"""Evaluate two isolated, *non-historical* compiler compatibility routes.

This script is an inference-only compatibility experiment, NOT the untouched
historical compiler pipeline. The pinned graph generator is imported as bytes,
but its Kalman frontend and Rosetta action-root resolution are extended at
runtime. A separate signed-by-hash receipt describes the transformations.

No source, kernel_info, historical generator, checkpoint, or completed GEXF is
modified. If either original graph validator or post-hoc authenticity check
fails, execution stops; no dummy memory or action mappings are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

KERNELS = ('GAN', 'Kalman', 'chstone-aes', 'chstone-jpeg', 'covariance',
           'jacobi-2d', 'rosetta-3d-rendering', 'syr2k', 'trmm-opt')
SEVEN = tuple(x for x in KERNELS if x not in ('Kalman', 'rosetta-3d-rendering'))
EXPECTED_GENERATOR_SHA = '5e4f93c89ce4c275c04ff6ed4dd81a17858647a55b0abec0846b8da240da3b05'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_path(folder: Path) -> Path:
    for name in ('source.cpp', 'source.c'):
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(f'No source.cpp/source.c in {folder}')


def compiler_compat_module(path: Path):
    spec = importlib.util.spec_from_file_location('mailohls_pinned_eval_compat', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_compatibility_hooks(mod, kernel: str):
    if kernel == 'Kalman':
        baseline_compile = mod.compile_source_to_mlir

        def kalman_compile(*, source, kernel, mlir_output, action_manifest, cgeist, cflags):
            if kernel != 'krnl_KALMAN':
                return baseline_compile(source=source, kernel=kernel,
                                        mlir_output=mlir_output, action_manifest=action_manifest,
                                        cgeist=cgeist, cflags=cflags)
            executable = mod.resolve_cgeist(cgeist)
            cmd = mod.build_cgeist_command(executable=executable, source=source,
                                            kernel=kernel, mlir_output=mlir_output,
                                            action_manifest=action_manifest, cflags=cflags)
            # Explicitly test the source-preserving pre-optimization IR. The
            # historical generator prohibits this flag for training graphs.
            cmd.insert(cmd.index('-o'), '-immediate')
            proc = subprocess.run(cmd, cwd=source.parent, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError('Kalman immediate-mode cgeist failed:\n' + proc.stderr[-4000:])
            if not mlir_output.is_file():
                if 'module' not in proc.stdout:
                    raise RuntimeError('Kalman immediate-mode cgeist produced no valid file/stdout')
                mlir_output.write_text(proc.stdout)
            text = mlir_output.read_text()
            all_64 = [line for line in text.splitlines()
                      if 'memref.alloca' in line and 'memref<64xf32>' in line]
            if len(all_64) != 4:
                raise RuntimeError(f'Expected four *physical* 64-element allocations; got {len(all_64)}')
            if not re.search(r'(?m)^\s*(func\.func|llvm\.func)\b[^\n]*\{', text):
                raise RuntimeError('Immediate mode did not emit a function body')
            print(f'[KALMAN IMMEDIATE] All {len(all_64)} authentic source arrays retained', flush=True)
            return cmd, text

        mod.compile_source_to_mlir = kalman_compile

    elif kernel == 'rosetta-3d-rendering':
        original_resolver = mod.MLIRGraphBuilder._array_argument_node

        def static_global_resolver(self, action):
            if action.action_id != 'L1' or action.variable != 'z_buffer':
                return original_resolver(self, action)
            if action.function != 'zculling' or tuple(action.array_dimensions) != (256, 256):
                raise RuntimeError('Rosetta static global source contract changed')
            matches = []
            for record in self.operation_records:
                if record.op_name != 'memref.get_global' or 'zculling' not in record.function_name:
                    continue
                # The MLIR global symbol is generated from the actual source
                # static variable; do not guess from shape or declaration order.
                symbol = mod.operation_first_line(record.operation)
                if 'z_buffer' not in symbol:
                    continue
                for key in record.result_keys:
                    node = self.value_nodes[key]
                    shape = self._node_memory_shape(node)
                    if shape[:2] != (256, 256) or shape[2:] != (1,):
                        continue
                    lines = self._root_access_source_lines(node, action.source_file)
                    if not (set(action.source_use_lines) & lines):
                        # If alias tracing cannot connect compiler memory
                        # accesses to the physical global root, do not claim
                        # a meaningful structural action scope.
                        continue
                    matches.append((record, node, symbol, sorted(lines)))
            if len(matches) != 1:
                raise RuntimeError(f'Expected one source-grounded static z_buffer global root, '
                                   f'found {len(matches)}: {[(x[2], x[3]) for x in matches]}')
            rec, node, symbol, lines = matches[0]
            print(f'[ROSETTA STATIC GLOBAL] z_buffer root={node}, source accesses={lines}', flush=True)
            return rec.function_id, node, rec.block_id, 'global_symbol_and_access'

        mod.MLIRGraphBuilder._array_argument_node = static_global_resolver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--successful-graphs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cgeist', type=Path, required=True)
    args = ap.parse_args()
    repo, sources, graphs, out, cgeist = (
        args.repo.resolve(), args.sources.resolve(), args.successful_graphs.resolve(),
        args.output.resolve(), args.cgeist.resolve()
    )
    if out.exists():
        raise FileExistsError(f'No overwrite: {out}')
    base = repo / 'Checkpoints/gnn_final_absolute_direct_rank_s123/run1/provenance/mlir_graph_gen.py'
    if sha(base) != EXPECTED_GENERATOR_SHA:
        raise RuntimeError('Pinned historical graph generator hash mismatch')
    if not cgeist.is_file():
        raise FileNotFoundError(cgeist)
    src_manifest = json.loads((sources / 'SOURCES_MANIFEST.json').read_text())
    if set(src_manifest) != set(KERNELS):
        raise RuntimeError('Expected locked nine-source cohort')
    for k in KERNELS:
        folder = sources / k
        if sha(source_path(folder)) != src_manifest[k]['source_sha256']:
            raise RuntimeError(f'{k}: source is not authenticated')
        if sha(folder / 'kernel_info.txt') != src_manifest[k]['metadata_sha256']:
            raise RuntimeError(f'{k}: metadata is not authenticated')
    for k in SEVEN:
        if not (graphs / f'{k}.gexf').is_file():
            raise FileNotFoundError(f'{k}: existing successful GEXF missing')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Set PYTHONHASHSEED=0 before invoking')
    out.mkdir(parents=True)
    for k in SEVEN:
        shutil.copy2(graphs / f'{k}.gexf', out / f'{k}.gexf')
    for k in ('Kalman', 'rosetta-3d-rendering'):
        mod = compiler_compat_module(base)
        install_compatibility_hooks(mod, k)
        src = source_path(sources / k)
        parser = mod.build_arg_parser()
        flags = [str(src), '--cgeist', str(cgeist),
                 '--cflag=-I' + str(src.parent), '--output', str(out / f'{k}.gexf'),
                 '--analysis-output', str(out / f'{k}.analysis.json'),
                 '--mlir-output', str(out / f'{k}.mlir')]
        print(f'\n=== EVALUATION-ONLY COMPATIBILITY: {k} ===', flush=True)
        mod.run(parser.parse_args(flags))
        if not (out / f'{k}.gexf').is_file():
            raise RuntimeError(f'{k}: graph generator did not validate/write a graph')
    artifacts = {k: sha(out / f'{k}.gexf') for k in KERNELS}
    receipt = dict(
        schema='mailohls-stage2-eval-compat9-v1',
        status='COMPATIBILITY_EXPERIMENT_NOT_HISTORICAL_IDENTICAL',
        all_source_and_metadata_hashes_verified=True,
        source_action_values_preserved=True,
        base_generator_sha256=sha(base),
        compatibility_wrapper_sha256=sha(Path(__file__)),
        cgeist_sha256=sha(cgeist),
        variants={'Kalman': 'immediate_pre_optimization_mlir_retains_L7_L8',
                  'rosetta-3d-rendering': 'compiler_global_symbol_and_access_L1'},
        caveat='New graph representation for two kernels; requires explicit GNN/GMT/SoTA comparability checks',
        graph_sha256=artifacts,
    )
    (out / 'EVAL_COMPAT9_RECEIPT.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('\n[GRAPH 9/9 CANDIDATE PASS] All GEXF files present and sha256 recorded')
    print('[COMPATIBILITY, NOT IDENTICAL HISTORICAL PREPROCESSING]')


if __name__ == '__main__':
    main()
