"""Finish the checksummed development export without changing the RC3 release."""
import argparse
import datetime
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
import shutil
import sys
import tarfile

import networkx
from GNN_branch import mlir_graph_v2 as program
from GNN_branch import target_graph_v22 as hardware


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--archive', required=True)
    args = ap.parse_args()
    root, archive = Path(args.directory), Path(args.archive)
    if archive.exists():
        raise FileExistsError('Refusing to overwrite an existing bundle')
    manifest = json.loads((root / 'manifest.json').read_text())
    test_output = (root / 'logs/tests.log').read_text()
    passed = re.search(r'(\d+) passed', test_output)
    if passed is None or re.search(r'\d+ (?:failed|error)', test_output):
        raise ValueError('A successful actual test receipt is required before packaging')
    (root / 'reference').mkdir(exist_ok=True)
    (root / 'release_raw_mlir').mkdir(exist_ok=True)
    # Preserve both contexts: release-authenticated compiler graph + matching
    # archived IR, and independently regenerated native graph + fresh IR.
    with hardware.authenticated_release(hardware.ROOT / 'GNN_branch/MLIR_graphs/salt-hls-graphs-v22-20261027-rc3.tar.gz') as (base, release):
        shutil.copy2(hardware.RELEASE, root / 'reference' / hardware.RELEASE.name)
        shutil.copy2(hardware.PROFILES, root / 'reference' / hardware.PROFILES.name)
        for row in manifest['kernels']:
            member = 'raw_mlir/' + Path(row['canonical']).name.replace('.json', '.mlir')
            path = root / 'release_raw_mlir' / Path(member).name
            shutil.copy2(base / member, path)
            graph = program.read_graph(root / row['canonical'])
            if sha(path) != graph.graph['provenance']['mlir_sha256']:
                raise ValueError('Primary compiler graph/IR hash mismatch')
            row['released_raw_mlir'] = str(path.relative_to(root))
    for index, row in enumerate(manifest['target_graphs'], start=1):
        path = root / row['path']
        if sha(path) != row['sha256']:
            raise ValueError('Export changed: ' + row['path'])
        graph = hardware.read_graph(path)
        if program.graph_payload(hardware.strip_overlay(graph)) != program.graph_payload(program.read_graph(root / graph.graph['program_metadata']['view'] / Path(row['path']).name)):
            raise ValueError('Read-back program restoration differs')
        row['serialized_readback_validated'] = True
        if index % 12 == 0:
            print('Read-back validated', index, 'of', len(manifest['target_graphs']), flush=True)
    for path, expected in manifest['protected_release_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Historical release changed')
    manifest.update(generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        environment={'python': sys.version.split()[0], 'networkx': networkx.__version__},
        tests={'command': 'python -B -m pytest -q -p no:cacheprovider tests/test_mlir_graph_v2.py tests/test_mlir_graph_v22.py tests/test_target_graph_v22.py',
               'passed': int(passed.group(1)), 'failed': 0}, all_serialized_target_graphs_readback_validated=True,
        cross_build_provenance_note='Fresh and release graphs retain different graph_identity and raw-MLIR hashes; their compiler facts are equal. No stored provenance was rewritten.')
    manifest['artifact_hashes'] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                                   if p.is_file() and p.name not in ['manifest.json', 'SHA256SUMS', 'package.log']}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS', 'package.log'])
    (root / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + str(p.relative_to(root)) + '\n' for p in files))
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'package.log')
    with archive.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as bundle:
                for path in files:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(root.name + '/' + str(path.relative_to(root)))
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    bundle.addfile(info, io.BytesIO(data))
    archive.with_name(archive.name + '.sha256').write_text(sha(archive) + '  ' + str(archive) + '\n')
    with tarfile.open(archive) as bundle:
        for path in files:
            data = bundle.extractfile(root.name + '/' + str(path.relative_to(root))).read()
            if hashlib.sha256(data).hexdigest() != sha(path):
                raise ValueError('Bundle verification failed')
    print(json.dumps({'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
                      'target_graphs': len(manifest['target_graphs']), 'readback_validated': True}))


if __name__ == '__main__':
    main()
