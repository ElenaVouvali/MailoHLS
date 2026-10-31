"""List, verify and restore explicitly registered immutable research artifacts."""
import argparse
import hashlib
import json
import os
import gzip
import io
import subprocess
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/artifacts.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def destination(root, name):
    path = Path(name)
    root = Path(root).resolve()
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Artifact path escapes destination through symlink: ' + name)
    result = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or not result.is_relative_to(root) or result == root:
        raise ValueError('Artifact path escapes destination root: ' + name)
    return result


def verify(path, record):
    if not Path(path).is_file():
        raise ValueError('Missing artifact: ' + str(path) + '; restore the registered Git payload or provide the original offline archive')
    if Path(path).stat().st_size != record['size'] or sha256(path) != record['sha256']:
        raise ValueError('Artifact size/SHA256 mismatch: ' + str(path))


def extract(archive, target):
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise ValueError('Extraction destination must not exist: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        stage = Path(temp) / 'content'
        stage.mkdir()
        with tarfile.open(archive) as tar:
            # Validate every member before writing any data; links and devices are forbidden.
            members = tar.getmembers()
            for member in members:
                if not (member.isfile() or member.isdir()) or '\\' in member.name:
                    raise ValueError('Unsupported archive member: ' + member.name)
                destination(stage, member.name)
            for member in members:
                path = destination(stage, member.name)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, path.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                    path.chmod(member.mode & 0o777)
        # Reserve the destination atomically: rename could replace an empty directory
        # created concurrently. File publication uses exclusive hard links instead.
        target.mkdir()
        publish_tree(stage, target)


def publish_tree(stage, root):
    """Publish regular files without replacing existing content, including races."""
    root = Path(root)
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    for source in files:
        target = destination(root, str(source.relative_to(stage)))
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or (sha256(target) != sha256(source) or (target.stat().st_mode & 0o111) != (source.stat().st_mode & 0o111)):
                raise ValueError('Restore conflict: ' + str(target))
    for source in files:
        target = destination(root, str(source.relative_to(stage)))
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-check parents after mkdir; an existing symlink must never redirect writes.
        target = destination(root, str(source.relative_to(stage)))
        try:
            os.link(source, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or (sha256(target) != sha256(source) or (target.stat().st_mode & 0o111) != (source.stat().st_mode & 0o111)):
                raise ValueError('Restore conflict: ' + str(target))


def restore(archive, record, root):
    if not record.get('restore_roots'):
        raise ValueError('This archive requires verify --extract-to a fresh directory')
    verify(archive, record)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        stage = Path(temp) / 'extracted'
        extract(archive, stage)
        manifest = json.loads((stage / 'MAILOHLS_ARTIFACT.json').read_text())
        if (manifest.get('schema') != 'mailohls-sealed-tree-v1' or manifest.get('artifact_id') != record['id'] or manifest['source_commit'] != record['source_commit']):
            raise ValueError('Restoration source identity mismatch')
        expected = {row['path']: row for row in manifest['files']}
        actual = {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
        if actual != set(expected) | {'MAILOHLS_ARTIFACT.json'}:
            raise ValueError('Restoration member set mismatch')
        for name, row in expected.items():
            if not any(name == prefix or name.startswith(prefix.rstrip('/') + '/') for prefix in record['restore_roots']):
                raise ValueError('Unexpected restoration path: ' + name)
            verify(destination(stage, name), row)
        (stage / 'MAILOHLS_ARTIFACT.json').unlink()
        publish_tree(stage, root)
        return len(expected)


def package(record, output):
    """Seal only committed public blobs, never local ignored weights or outputs."""
    output = Path(output)
    if output.exists():
        raise ValueError('Archive already exists: ' + str(output))
    ref = record['source_commit']
    entries = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-rz', ref, '--', *record['restore_roots']]).split(b'\0')
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / 'archive.tar.gz'
        with staged.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w|') as tar:
            for entry in entries:
                if not entry:
                    continue
                meta, name = entry.split(b'\t', 1)
                mode, kind, blob = meta.split()
                if mode not in (b'100644', b'100755') or kind != b'blob':
                    raise ValueError('Package requires regular committed files: ' + name.decode())
                if name.decode().endswith(('.pth', '.pt', '.safetensors', '.bin', '.ckpt')):
                    raise ValueError('Weights need a separate redistribution review')
                data = subprocess.check_output(['git', '-C', str(ROOT), 'cat-file', 'blob', blob.decode()])
                info = tarfile.TarInfo(name.decode())
                info.size, info.mode = len(data), int(mode, 8) & 0o777
                tar.addfile(info, io.BytesIO(data))
                rows.append(dict(path=name.decode(), size=len(data), sha256=hashlib.sha256(data).hexdigest(), mode=info.mode, git_blob=blob.decode()))
            manifest = json.dumps(dict(schema='mailohls-sealed-tree-v1', source_commit=ref, artifact_id=record['id'], files=rows), sort_keys=True, indent=2).encode()
            info = tarfile.TarInfo('MAILOHLS_ARTIFACT.json')
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest))
        os.link(staged, output)
    return rows


def restore_git(record, root):
    """Recover sealed scientific payloads without a Release or local runtime cache."""
    roots = record.get('git_roots')
    if not roots:
        raise ValueError('This artifact has no registered Git recovery contract')
    ref = record['source_commit']
    try:
        for name in roots:
            destination(Path(root), name)
            actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', ref + ':' + name], stderr=subprocess.DEVNULL).decode().strip()
            if actual != record['git_objects'][name]:
                raise ValueError('Git source identity mismatch: ' + name)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Original Git objects missing. Fetch origin and preservation tags before restoring.') from exc
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        if record.get('restore_roots'):
            archive = Path(temp) / 'recovered.tar.gz'
            package(record, archive)
            with tarfile.open(archive) as tar:
                sealed = tar.extractfile('MAILOHLS_ARTIFACT.json').read()
            if hashlib.sha256(sealed).hexdigest() != record['member_manifest_sha256']:
                raise ValueError('Original sealed member manifest mismatch')
            # Explicit payload recovery: gzip implementation differences do not
            # change original file hashes, modes, source or manifest identity.
            local = dict(record, size=archive.stat().st_size, sha256=sha256(archive))
            return restore(archive, local, root)
        archive = Path(temp) / 'catalog.tar'
        with archive.open('xb') as stream:
            subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', ref, '--', *roots], stdout=stream, check=True)
        stage = Path(temp) / 'catalog'
        extract(archive, stage)
        catalog = stage / 'graphs/latest'
        checksum = catalog / 'SHA256SUMS'
        if sha256(checksum) != record['catalog_checksums_sha256']:
            raise ValueError('Original catalog checksum manifest mismatch')
        names = set()
        for line in checksum.read_text().splitlines():
            digest, name = line.split(None, 1)
            name = name.lstrip('*')
            path = destination(catalog, name)
            if sha256(path) != digest:
                raise ValueError('Catalog payload mismatch: ' + name)
            names.add(name)
        actual = {str(p.relative_to(catalog)) for p in catalog.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
        if len(names) != record['catalog_file_count'] or names != actual:
            raise ValueError('Catalog member set mismatch')
        publish_tree(stage, root)
        return len(names)


def fetch(record, root):
    path = destination(root, record['path'])
    if path.exists() or path.is_symlink():
        raise ValueError('Refusing to overwrite ' + str(path) + '; verify the existing artifact first')
    url = record.get('download_url')
    if not url or not url.startswith('https://'):
        raise ValueError('No download needed for Git-backed evidence. Use restore-git --artifact ' + record.get('id', '<id>'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, stream)
            stream.flush()
            os.fsync(stream.fileno())
            verify(temp, record)
            # Hard-link publication is atomic and cannot replace an existing destination.
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('list', 'verify', 'fetch', 'restore', 'package', 'restore-git'))
    ap.add_argument('--artifact')
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--extract-to', type=Path)
    ap.add_argument('--output', type=Path, help='new archive path for package')
    args = ap.parse_args()
    records = json.loads(MANIFEST.read_text())['artifacts']
    if args.command == 'list':
        for row in records:
            print(row['id'], row['size'], row['path'], row.get('publication_status', 'unknown'))
        return
    selected = [r for r in records if r['id'] == args.artifact]
    if len(selected) != 1:
        ap.error('Select exactly one registered --artifact ID (see list)')
    row = selected[0]
    try:
        if args.command == 'restore-git':
            print('Restored original Git payload:', restore_git(row, args.root), 'verified members')
            return
        if args.command == 'package':
            if args.output is None:
                ap.error('package requires --output')
            rows = package(row, args.output)
            print('Packaged', len(rows), 'files', sha256(args.output))
            return
        path = fetch(row, args.root) if args.command == 'fetch' else destination(args.root, row['path'])
        verify(path, row)
        if args.command == 'restore':
            print('Restored', restore(path, row, args.root), 'verified original paths')
        if args.extract_to:
            extract(path, args.extract_to)
        print('Verified', row['id'], row['sha256'])
    except (ValueError, OSError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        ap.exit(1, str(exc)+'\n')

if __name__ == '__main__':
    main()
