#!/usr/bin/env python3
"""Seal frontend evidence and exact measured-source graphs in a new receipt.

Consumes the frontend-only comparison and existing strict graph generator outputs.
Does not modify the parent recovery, original inputs, measurements, or graphs.
"""
import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Preprocessing import gnosis_compilation_recovery as phase
from Preprocessing import gnosis_recovery as recovery

IDS = {phase.KNN:'knn', 'rodinia_dilate_2_pipeline_0':'dilate2',
       'rodinia_dilate_3_pipeline_0':'dilate3', 'spcl_example_03':'spcl03',
       'spcl_example_05':'spcl05'}


def seal(runtime, root):
    import networkx as nx
    target = root/phase.PROOF
    if (root/phase.REGISTRY).exists():
        raise FileExistsError('Preserve the existing compiler correspondence receipt')
    target.mkdir(parents=True, exist_ok=True)
    parent = recovery.load_registry(root)
    frontend = json.loads((runtime/'results_final/receipt.json').read_text())
    if any(c['returncode'] for c in frontend['commands']):
        raise ValueError('A frontend comparison failed')
    blobs = {r['path']:r for r in json.loads((root/recovery.PROOF/'upstream_blobs.json').read_text())}
    def relative(path): return str(path.relative_to(root))
    def copy(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return relative(dst)
    def compressed(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(gzip.compress(src if isinstance(src, bytes) else src.read_bytes(), mtime=0))
        return relative(dst)
    shared = [copy(runtime/'results_final/receipt.json', target/'frontend_receipt.json')]
    for name in ['existing_frontend_log.sha256', 'existing_frontend_command.txt']:
        shared.append(copy(runtime/name, target/name))
    for name in ['experiments/gnosis_recovery/compare_five_compilation.py',
                 'GNN_branch/mlir_graph_gen.py', 'toolchain/lock.env',
                 'Data/provenance/gnosis_17_recovery_20261113/upstream_blobs.json']:
        shared.append(name)
    receipt = dict(schema='gnosis-five-source-correspondence-v1',
        parent_commit='3053f674f88a3562796e96d9b9cecd6be3ec09fd',
        parent_registry_sha256=recovery.sha(root/recovery.REGISTRY),
        upstream_commit=recovery.PIN, canonical_contract=phase.CONTRACT,
        measured_source_contract=phase.ORIGINAL_CONTRACT,
        historical_per_measurement_flags_available=False,
        flags_limit='Documented generator has no add_files -cflags; installed Vitis 2021.1 flags corroborate, not an original run receipt.',
        shared_files={name:recovery.sha(root/name) for name in shared}, kernels={})
    compiler = Path('/home/elvouvali/tools/Polygeist/build-mailohls-assertions/bin/clang++')
    for kernel, short in IDS.items():
        entry = parent[kernel]
        original = runtime/'inputs/original'/kernel
        canonical = runtime/'inputs/canonical'/kernel
        folder = target/'kernels'/kernel
        bundle = folder/'measured_source'
        files = []
        original_blobs = {}
        for path in sorted(original.iterdir()):
            raw = path.read_bytes()
            upstream = 'data/ApplicationDataset/'+kernel+'/'+path.name
            blob = hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
            if blob != blobs[upstream]['blob']:
                raise ValueError('Not the pinned measured-source blob: '+upstream)
            dest = copy(path, bundle/path.name)
            files.append(dest)
            original_blobs[dest] = dict(upstream_path=upstream, git_blob=blob,
                                        sha256=recovery.sha(path))
        evidence = {}
        for kind in ['original', 'canonical']:
            src = runtime/'results_final'/kind/kernel
            pp = (src/'preprocessed.txt').read_text()
            # Retain only repository source, not installed proprietary headers.
            pp = pp[pp.rindex('extern "C" {'):] if kernel not in phase.SYMBOLS else pp[pp.index('constexpr int N'):]
            evidence[kind] = {mode:compressed(value, folder/kind/(mode+'.gz')) for mode,value in [
                ('preprocessed', pp.encode()), ('llvm', src/'llvm.ll')]}
            files.extend(evidence[kind].values())
            macros = (src/'macros.txt').read_text().splitlines()
            selected = [line for line in macros if re.match(r'#define (?:__VIVADO_HLS__|__VITIS_HLS__|__SYNTHESIS__|__HLS_SYN__|AESL_SYN|COALESCING_5_512bit|GET_RANGE|SET_RANGE|INTERFACE_WIDTH|MARS_WIDE_BUS_TYPE|LARGE_BUS|SIZE_[123])(?:\b|\()',line)]
            p = folder/kind/'relevant_macros.json'
            recovery.write_json(p, selected); files.append(relative(p))
        a = phase.normalize_ir(phase.read_text(root,evidence['original']['llvm']),phase.SYMBOLS.get(kernel))
        b = phase.normalize_ir(phase.read_text(root,evidence['canonical']['llvm']))
        eligible = kernel != phase.KNN
        if eligible and (a != b or phase.preprocessed_body(kernel,phase.read_text(root,evidence['original']['preprocessed'])) != phase.preprocessed_body(kernel,phase.read_text(root,evidence['canonical']['preprocessed']))):
            raise ValueError('Effective compiler input differs: '+kernel)
        if not eligible and a == b:
            raise ValueError('KNN mismatch evidence unexpectedly absent')
        differences = []
        for path in sorted(original.iterdir()):
            if path.suffix not in ('.cpp','.h'): continue
            differences.extend(difflib.unified_diff(path.read_text().splitlines(True),
                (canonical/path.name).read_text().splitlines(True),
                fromfile='measured_source/'+path.name,tofile='canonical/'+path.name))
        diff = folder/'source_changes.patch'
        diff.write_text(''.join(differences));files.append(relative(diff))
        graph = nx.read_gexf(runtime/(short+'.gexf'))
        metadata = json.loads(graph.graph['name'].removeprefix('mailohls-meta-v1:'))
        source, = original.glob('*.cpp')
        if metadata['source_sha256'] != recovery.sha(source) or metadata['action_resolutions'] != {'source_location':len(entry['source_actions'])}:
            raise ValueError('Graph is not an exact, complete measured-source binding')
        outputs = {name:compressed(runtime/(short+suffix),folder/('graph'+suffix+'.gz'))
                   for name,suffix in [('gexf','.gexf'),('mlir','.mlir'),('analysis','.analysis.json')]}
        files.extend(outputs.values())
        if (runtime/(short+'.graph.log')).exists():
            files.append(copy(runtime/(short+'.graph.log'),folder/'graph.log'))
        cflags = ['-I'+str((runtime/'include').resolve())] if kernel not in phase.SYMBOLS else []
        # Same Clang build/include search as cgeist; archive every transitive
        # dependency hash, not installed vendor header contents.
        deps = subprocess.run([str(compiler), *cflags, '-M', str(source.resolve())],
                              check=True, capture_output=True, text=True).stdout
        paths = shlex.split(deps.replace('\\\n','').split(':',1)[1])
        closure = {p:recovery.sha(Path(p)) for p in paths}
        dependency_file = folder/'graph_header_closure.json'
        recovery.write_json(dependency_file,closure);files.append(relative(dependency_file))
        measured_id = 'GNWSIS-'+recovery.PIN[:8]+'/'+kernel
        item = dict(canonical_eligible=eligible, measured_source_eligible=True,
            source_version=measured_id, original_bundle=relative(bundle), original_blobs=original_blobs,
            top=(original/'kernel_info.txt').read_text().splitlines()[0],
            basis='Full Vitis HLS LLVM module identical modulo top linkage and file paths' if eligible else 'Different effective widths: historical i512 versus canonical i32; no canonical correspondence',
            normalized_llvm_sha256_original=hashlib.sha256(a.encode()).hexdigest(),
            normalized_llvm_sha256_canonical=hashlib.sha256(b.encode()).hexdigest(),
            evidence=evidence, canonical_files=entry['canonical_files'], states=entry['states'],
            matching_measured_rows=entry['states']['rows'], recovered_actions=entry['missing_actions'],
            graph=dict(**outputs,source_sha256=metadata['source_sha256'],
                metadata_sha256=recovery.sha(original/'kernel_info.txt'),
                action_count=len(entry['source_actions']),nodes=len(graph),edges=graph.number_of_edges(),
                command=['GNN_branch/mlir_graph_gen.py', str(source.resolve()),
                    '--cgeist=/home/elvouvali/tools/Polygeist/build-mailohls-assertions/bin/cgeist',
                    *(['--compiler-symbol='+phase.SYMBOLS[kernel][0]] if kernel in phase.SYMBOLS else []),
                    *['--cflag='+flag for flag in cflags], '--output=<graph.gexf>',
                    '--mlir-output=<graph.mlir>', '--analysis-output=<graph.analysis.json>'],
                environment={'PYTHONHASHSEED':'0','python':'/home/elvouvali/.mlir-python311/bin/python',
                    'PYTHONPATH':'/home/elvouvali/tools/Polygeist/build-mailohls-assertions/tools/mlir/python_packages/mlir_core'},
                metadata=metadata,cflags=cflags,compiler_symbol=phase.SYMBOLS.get(kernel,[None])[0],
                representation='Polygeist v7; native compiler analysis; exact source-location actions; KNN uses genuine AP simulation headers, not native HLS i512 IR',
                historical_gnn_gmt_memory_qualified=False),
            files={name:recovery.sha(root/name) for name in files})
        receipt['kernels'][kernel] = item
    recovery.write_json(root/phase.REGISTRY, receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path, required=True)
    parser.add_argument('--root',type=Path,default=recovery.ROOT)
    args = parser.parse_args()
    seal(args.runtime.resolve(),args.root.resolve())
