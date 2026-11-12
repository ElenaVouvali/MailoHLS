"""Fresh Hotspot MLIR verification and authenticated RC3 hardware materialization.

Run from the MailoHLS repository root. The historical release is never modified.
This materializer requires compiler facts equal to the authenticated RC3 facts;
different source facts must be published as a separately reviewed snapshot.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from GNN_branch import mlir_graph_v2 as program
from GNN_branch import target_graph_v22 as hardware
from GNN_branch.action_point_relations import action_point_views, get_loop_semantics
from experiments.structural_audit.compare_v2 import _relations
from experiments.structural_audit.finalize_v22 import queries_equal

KERNELS = ('rodinia-hotspot-2-pipeline', 'rodinia-hotspot-3-unroll')
ARCHIVE = Path('GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz')
TARGETS = ('zcu104', 'u200', 'u50')
CLOCKS = (3.33, 5., 6.67, 10.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def query_facts(value):
    """Cross-build comparison only: keep facts, omit two context fingerprints.

    Actual query results and graph payloads retain both provenance fields.
    Canonical/compact equivalence below still uses the unchanged strict check.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'graph_identity':
                continue
            if key == 'source_provenance':
                item = {k: v for k, v in item.items() if k != 'mlir_sha256'}
            result[key] = query_facts(item)
        return result
    if isinstance(value, list):
        return [query_facts(v) for v in value]
    return value


def equal_queries(a, b):
    if query_facts(_relations(a)) != query_facts(_relations(b)) or query_facts(action_point_views(a)) != query_facts(action_point_views(b)):
        raise ValueError('Public action-point queries differ')
    for _, attrs in a.nodes(data=True):
        if attrs.get('category') == 'action_point' and attrs.get('action_kind') == 'loop':
            if query_facts(get_loop_semantics(a, attrs['action_id'])) != query_facts(get_loop_semantics(b, attrs['action_id'])):
                raise ValueError('Loop semantics differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--native-python', required=True)
    parser.add_argument('--cgeist', required=True)
    parser.add_argument('--finish-existing', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.finish_existing:
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name in ['reproduce.py', 'package.py', 'README.md']:
        origin = Path(__file__).resolve().parent / name
        destination = output / name
        if origin.exists() and origin.resolve() != destination.resolve():
            shutil.copy2(origin, destination)
    started = time.perf_counter()
    build = Path(args.cgeist).resolve().parents[1]
    libs = build / 'tools/mlir/python_packages/mlir_core'
    native_env = dict(os.environ, PYTHONPATH=str(libs), PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    report = dict(schema='salt-hls-hotspot-materialization-v1', graph_schema=program.SCHEMA_V22,
                  overlay_schema=hardware.SCHEMA, semantic_revision='rc3',
                  status='development materialization; independent RC3 semantic adjudication remains pending',
                  source_change_detected=False, kernels=[], target_graphs=[],
                  targets=list(TARGETS), clock_periods_ns=list(CLOCKS), resource_availability='full physical capacity',
                  native_python=args.native_python, cgeist=str(Path(args.cgeist).resolve()),
                  cgeist_sha256=sha(args.cgeist), implementation=hardware.implementation_identity(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  hls_synthesis_performed=False, neural_training_performed=False)
    protected = [ARCHIVE, hardware.RELEASE, Path('docs/graph-release-v22-rc3.sha256')]
    before = {str(p): sha(p) for p in protected}
    input_paths = [Path('GNN_branch/targets/fpga_profiles.json'), Path('toolchain/lock.env')]
    for kernel in KERNELS:
        input_paths.extend(Path('Data/ApplicationDataset') / kernel / name
                           for name in ['hotspot.cpp', 'hotspot.h', 'kernel_info.txt'])
    input_hashes = {str(p): sha(p) for p in input_paths}
    for sub in ['raw_mlir', 'release_raw_mlir', 'native_canonical', 'native_compact', 'canonical', 'compact', 'logs', 'sources', 'reference']:
        (output / sub).mkdir(exist_ok=True)
    with hardware.authenticated_release(ARCHIVE) as (baseline, release):
        shutil.copy2(hardware.RELEASE, output / 'reference' / hardware.RELEASE.name)
        shutil.copy2(hardware.PROFILES, output / 'reference' / hardware.PROFILES.name)
        report['release_authentication'] = dict(archive_sha256=sha(ARCHIVE),
            manifest_sha256=sha(hardware.RELEASE), graph_count=release['graph_count'],
            authenticated_members=len(release['files_sha256']))
        for kernel in KERNELS:
            source = Path('Data/ApplicationDataset') / kernel / 'hotspot.cpp'
            metadata = source.with_name('kernel_info.txt')
            source_copy = output / 'sources' / kernel
            source_copy.mkdir(exist_ok=True)
            for name in ['hotspot.cpp', 'hotspot.h', 'kernel_info.txt']:
                shutil.copy2(source.with_name(name), source_copy / name)
            raw = output / 'raw_mlir' / (kernel + '.mlir')
            fresh_path = output / 'native_canonical' / (kernel + '.json')
            command = [args.native_python, '-B', '-m', 'GNN_branch.mlir_graph_v2', '--revision', '2.2',
                '--semantic-revision', 'rc3', '--source', str(source), '--metadata', str(metadata),
                '--cgeist', args.cgeist, '--mlir-output', str(raw), '--output', str(fresh_path)]
            if not fresh_path.exists():
                with (output / 'logs' / (kernel + '.log')).open('w') as log:
                    subprocess.run(command, env=native_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            fresh = program.read_graph(fresh_path)
            if fresh.graph['native_correspondence']['status'] != 'pass':
                raise ValueError('Fresh compiler/native correspondence failed')
            if fresh.graph['provenance']['mlir_sha256'] != sha(raw):
                raise ValueError('Raw MLIR and graph provenance differ')
            for key, expected in [('source_sha256', sha(source)), ('metadata_sha256', sha(metadata)),
                                  ('cgeist_sha256', sha(args.cgeist))]:
                if fresh.graph['provenance'][key] != expected:
                    raise ValueError('Fresh compiler context differs: ' + key)
            member = next(n for n in release['files_sha256'] if n.startswith('canonical/' + kernel + '-'))
            base = program.read_graph(baseline / member)
            raw_member = 'raw_mlir/' + Path(member).name.replace('.json', '.mlir')
            if sha(baseline / raw_member) != base.graph['provenance']['mlir_sha256']:
                raise ValueError('Archived graph and raw module differ')
            shutil.copy2(baseline / raw_member, output / 'release_raw_mlir' / Path(raw_member).name)
            a, b = program.graph_payload(base), program.graph_payload(fresh)
            if a['nodes'] != b['nodes'] or a['edges'] != b['edges']:
                raise ValueError('Compiler facts differ from RC3; create a separate reviewed snapshot instead of claiming reuse')
            if base.graph['provenance']['source_sha256'] != sha(source) or base.graph['provenance']['metadata_sha256'] != sha(metadata):
                raise ValueError('Current inputs do not match the selected authenticated snapshot')
            for key in ['local_header_sha256', 'analysis_extension_sha256', 'cgeist_sha256']:
                if base.graph['provenance'][key] != fresh.graph['provenance'][key]:
                    raise ValueError('Pinned source/compiler context differs: ' + key)
            changed_metadata = {key for key in set(a['metadata']) | set(b['metadata'])
                                if a['metadata'].get(key) != b['metadata'].get(key)}
            if not changed_metadata <= {'provenance', 'native_correspondence', 'graph_identity'}:
                raise ValueError('Unexpected compiler semantic metadata difference')
            equal_queries(base, fresh)
            fresh_compact = program.compact_view(fresh)
            queries_equal(fresh, fresh_compact)
            program.write_graph(fresh_compact, output / 'native_compact' / (kernel + '.json'))
            # Repeat the fresh native interpretation independently, from the sealed raw module.
            with tempfile.TemporaryDirectory(prefix='hotspot-native-repeat-') as directory:
                repeat = Path(directory) / 'repeat.json'
                code = """import json,sys
from GNN_branch import mlir_graph_v2 as v
g,_=v.generate(sys.argv[1],sys.argv[2],sys.argv[3],provenance=json.loads(sys.argv[5]),revision='2.2',semantic_revision='rc3')
v.write_graph(g,sys.argv[4])
"""
                subprocess.run([args.native_python, '-B', '-c', code, str(source), str(metadata), str(raw),
                    str(repeat), json.dumps(fresh.graph['provenance'])], env=native_env, check=True)
                if program.graph_payload(program.read_graph(repeat)) != program.graph_payload(fresh):
                    raise ValueError('Repeated native interpretation is nondeterministic')
            views = {}
            for view in ['canonical', 'compact']:
                path = output / view / Path(member).name
                shutil.copy2(baseline / (view + '/' + Path(member).name), path)
                views[view], auth = hardware.authenticated_input(path)
                if view == 'canonical':
                    primary_auth = auth
            queries_equal(views['canonical'], views['compact'])
            coverage = base.graph['action_point_coverage']['records']
            record = dict(kernel=kernel, canonical=str(Path('canonical') / Path(member).name),
                compact=str(Path('compact') / Path(member).name), fresh_native_canonical=str(fresh_path.relative_to(output)),
                raw_mlir=str(raw.relative_to(output)), native_command=command,
                released_raw_mlir=str(Path('release_raw_mlir') / Path(raw_member).name),
                nodes=len(base), edges=base.number_of_edges(), compact_nodes=len(views['compact']),
                compact_edges=views['compact'].number_of_edges(),
                registered_actions=sum(bool(r['metadata_registered']) for r in coverage),
                matched_actions=sum(bool(r['compiler_matched']) for r in coverage),
                unresolved_actions=[r['action_id'] for r in coverage if r['metadata_registered'] and not r['compiler_matched']],
                checks=dict(fresh_native_valid=True, compiler_nodes_and_edges_equal_to_release=True,
                    fresh_and_archived_query_facts_equal=True, native_interpretation_repeat_identical=True,
                    canonical_compact_reconstruction_and_queries_equal=True), source_authentication=primary_auth,
                cross_build_query_comparison_ignored_context_fields=['graph_identity', 'source_provenance.mlir_sha256'],
                metadata_differences=[key for key in sorted(set(a['metadata']) | set(b['metadata']))
                                      if a['metadata'].get(key) != b['metadata'].get(key)])
            report['kernels'].append(record)
            for target in TARGETS:
                for clock in CLOCKS:
                    signatures = []
                    for view in ['canonical', 'compact']:
                        path = output / view / Path(member).name
                        graph, auth = hardware.authenticated_input(path)
                        original = program.graph_payload(graph)
                        enriched = hardware.enrich(graph, target, clock, authentication=auth)
                        status = hardware.validate(enriched)
                        if program.graph_payload(hardware.strip_overlay(enriched)) != original or program.graph_payload(graph) != original:
                            raise ValueError('Enrichment changed compiler facts')
                        repeat = hardware.enrich(graph, target, clock, authentication=auth)
                        if hardware.serialized(repeat) != hardware.serialized(enriched):
                            raise ValueError('Hardware enrichment is nondeterministic')
                        signatures.append(hardware.overlay_signature(enriched))
                        filename = Path('target_aware') / target / (f'{clock:g}'.replace('.', 'p') + 'ns_full') / view / Path(member).name
                        hardware.write_graph(enriched, output / filename)
                        report['target_graphs'].append(dict(kernel=kernel, target=target, clock_period_ns=clock,
                            view=view, path=str(filename), sha256=sha(output / filename), bytes=(output / filename).stat().st_size,
                            validation=status, exact_program_restoration=True, repeat_serialization_identical=True))
                    if signatures[0] != signatures[1]:
                        raise ValueError('Canonical/compact hardware semantics differ')
            print('Validated', kernel, 'across', len(TARGETS) * len(CLOCKS), 'target contexts', flush=True)
    if before != {str(p): sha(p) for p in protected} or input_hashes != {str(p): sha(p) for p in input_paths}:
        raise ValueError('Input/release files changed during materialization')
    report.update(protected_release_hashes=before, input_hashes=input_hashes, frozen_release_unchanged=True,
                  failed_graphs=0, target_graph_count=len(report['target_graphs']), runtime_seconds=time.perf_counter() - started)
    report['artifact_hashes'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.json'))
                               if p.name not in ['manifest.json', 'inputs.json']}
    save(output / 'manifest.json', report)
    print(json.dumps(dict(kernels=len(report['kernels']), target_graphs=report['target_graph_count'],
                         failed_graphs=0, runtime_seconds=report['runtime_seconds']), sort_keys=True))


if __name__ == '__main__':
    main()
