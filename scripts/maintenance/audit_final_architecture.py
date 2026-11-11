"""Read-only evidence, link and tracked-tree audit; never remove or publish files."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = (
    'old_GNN_checkpoint', 'old_harp_graphs', 'HARP-rep', 'Checkpoints',
    'LIFT_perf', 'LIFT_pareto', 'Histogram', 'mailohls_runs',
    'mailohls_u50_adaptation', 'asplos_results', 'asplos_paper',
    'paper_plots', 'Notebooks', 'artifacts',
)
MAINTAINED_DOCS = (
    'README.md', 'docs/ARCHITECTURE_EVIDENCE.md', 'docs/REPRODUCIBILITY.md',
    'docs/LIMITATIONS.md', 'docs/CONSOLIDATION_STATUS.md',
    'docs/CONSOLIDATION_INVENTORY.md',
    'docs/FILM_GMT_AND_HISTORICAL_ENCODER.md', 'docs/evidence/README.md',
    'examples/README.md',
)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def broken_links(relative):
    """Check local inline Markdown paths, excluding code blocks and URL anchors."""
    path = ROOT / relative
    body = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
    missing = []
    for raw in re.findall(r'\[[^\]\n]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', body):
        url = urlsplit(raw.strip('<>'))
        if url.scheme or url.netloc or not url.path:
            continue
        target = (path.parent / unquote(url.path)).resolve()
        if not target.exists():
            missing.append(raw)
    return missing


def audit():
    errors = []
    sources = []
    with (ROOT / 'docs/evidence/SOURCE_MAP.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            copy = ROOT / row['destination']
            original = git('show', row['commit_sha'] + ':' + row['source_path'])
            valid = copy.is_file() and sha(copy.read_bytes()) == sha(original) == row['sha256_of_copy']
            # Check that the source commit remains reachable, not merely present as a blob.
            refs = git('for-each-ref', '--contains=' + row['commit_sha'], '--format=%(refname)').decode().splitlines()
            sources.append(dict(row, verified=valid, containing_refs=refs,
                                source_url='https://github.com/ElenaVouvali/MailoHLS/blob/'
                                + row['commit_sha'] + '/' + row['source_path']))
            if not valid or not refs:
                errors.append('Evidence identity/reachability: ' + row['destination'])
    local_sources = []
    manifest = ROOT / 'docs/evidence/LOCAL_SOURCE_MAP.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text())['files']:
            path = ROOT / row['destination']
            valid = path.is_file() and sha(path.read_bytes()) == row['sha256']
            local_sources.append(dict(row, verified=valid))
            if not valid:
                errors.append('Local receipt identity: ' + row['destination'])
    links = {}
    for name in MAINTAINED_DOCS:
        links[name] = broken_links(name) if (ROOT / name).is_file() else ['DOCUMENT MISSING']
        if links[name]:
            errors.append('Maintained document links: ' + name)
    # Copied reports retain original relative paths; resolve those in the source commit.
    historical_links = {row['destination']: broken_links(row['destination'])
                        for row in sources if row['destination'].endswith('.md')}
    tracked = [p.decode('utf-8', 'surrogateescape') for p in git('ls-files', '-z').split(b'\0') if p]
    sizes = [(ROOT / p).lstat().st_size for p in tracked if (ROOT / p).exists()]
    largest = sorted((dict(path=p, bytes=(ROOT / p).lstat().st_size)
                      for p in tracked if (ROOT / p).is_file()),
                     key=lambda row: (-row['bytes'], row['path']))[:50]
    candidates = {}
    for name in CANDIDATES:
        paths = [p for p in tracked if p == name or p.startswith(name + '/')]
        result = subprocess.run(['git', '-C', str(ROOT), 'grep', '-n', '-I', '-F', name,
                                 '--', '*.py', '*.sh', '*.json', '*.yaml', '*.yml', '*.toml'],
                                capture_output=True, text=True)
        if result.returncode not in (0, 1):
            errors.append('Dependency search failed: ' + name)
        candidates[name] = dict(tracked_files=len(paths),
                                tracked_bytes=sum((ROOT / p).lstat().st_size for p in paths if (ROOT / p).exists()),
                                reference_lines=result.stdout.splitlines(),
                                disposition='RETAIN: no signed-off removal inventory')
    branches = []
    for ref in git('for-each-ref', '--format=%(refname)', 'refs/remotes/origin/').decode().splitlines():
        if ref.endswith('/HEAD'):
            continue
        branches.append(dict(ref=ref, sha=git('rev-parse', ref).decode().strip(),
                             commits_not_in_foundation=int(git('rev-list', '--count', 'HEAD..' + ref)),
                             changed_paths=git('diff', '--no-renames', '--name-status', 'HEAD', ref).decode().splitlines(),
                             disposition='RETAIN: content inventory, not retirement approval'))
    return dict(schema='salt-hls-consolidation-audit-v1',
                generated_utc=datetime.now(timezone.utc).isoformat(),
                head=git('rev-parse', 'HEAD').decode().strip(),
                branch=git('branch', '--show-current').decode().strip(),
                evidence=sources, local_receipts=local_sources,
                maintained_missing_links=links, historical_missing_links=historical_links,
                tracked_files=len(tracked), tracked_bytes=sum(sizes),
                tracked_bytes_scope='Current worktree bytes at tracked index paths; new untracked files excluded',
                top_level_counts=dict(Counter(p.split('/')[0] for p in tracked)),
                largest_files=largest, cleanup_candidates=candidates,
                local_remote_tracking_inventory=branches,
                worktrees=git('worktree', 'list', '--porcelain').decode(),
                errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON report path; existing files are refused')
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps({key: report[key] for key in ('branch', 'head', 'tracked_files', 'tracked_bytes', 'errors')}, indent=2))
    print('Verified Git evidence:', sum(row['verified'] for row in report['evidence']))
    print('Verified local receipts:', sum(row['verified'] for row in report['local_receipts']))
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
