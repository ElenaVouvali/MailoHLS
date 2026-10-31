"""List, verify and restore explicitly registered immutable research artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
    result = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or not result.is_relative_to(root) or result == root:
        raise ValueError('Artifact path escapes destination root: ' + name)
    return result


def verify(path, record):
    if not Path(path).is_file():
        raise ValueError('Missing artifact: ' + str(path) + '; use fetch only if a verified public URL is registered')
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
        # rename is atomic on the same filesystem; refuse a destination created meanwhile.
        if target.exists() or target.is_symlink():
            raise ValueError('Extraction destination appeared during restore')
        stage.rename(target)


def fetch(record, root):
    path = destination(root, record['path'])
    if path.exists() or path.is_symlink():
        raise ValueError('Refusing to overwrite ' + str(path) + '; verify the existing artifact first')
    url = record.get('download_url')
    if not url or not url.startswith('https://'):
        raise ValueError('No public HTTPS download registered; preserve the original artifact')
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
    ap.add_argument('command', choices=('list', 'verify', 'fetch'))
    ap.add_argument('--artifact')
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--extract-to', type=Path)
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
        path = fetch(row, args.root) if args.command == 'fetch' else destination(args.root, row['path'])
        verify(path, row)
        if args.extract_to:
            extract(path, args.extract_to)
        print('Verified', row['id'], row['sha256'])
    except (ValueError, OSError, tarfile.TarError) as exc:
        ap.exit(1, str(exc)+'\n')

if __name__ == '__main__':
    main()
