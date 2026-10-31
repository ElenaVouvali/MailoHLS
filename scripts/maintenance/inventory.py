"""Inventory committed trees without importing research code or reading runtime weights."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def git(*args):
    return subprocess.check_output(['git', *args])


def tree(ref):
    rows = []
    for record in git('ls-tree', '-r', '-l', '-z', ref).split(b'\0'):
        if not record:
            continue
        meta, path = record.split(b'\t', 1)
        mode, kind, blob, size = meta.split()
        rows.append(dict(path=path.decode(), mode=mode.decode(), blob=blob.decode(),
                         size=int(size) if size != b'-' else 0))
    return rows


def classify(path):
    if path.startswith(('Data/', 'evaluation_set/', 'toolchain/', 'mailohls_runs/')) or path == 'FINAL_ARTIFACTS.json':
        return 'PROTECTED'
    if path.startswith(('Checkpoints/', 'old_', 'src/')):
        return 'REQUIRED_PROVENANCE'
    if path.startswith(('plots/ta_synthesis/', 'artifacts/', 'GNN_branch/MLIR_graphs/')):
        return 'EXTERNAL_ARTIFACT_CANDIDATE'
    return 'UNIQUE_ACTIVE_IMPLEMENTATION'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = tree(args.ref)
    groups = defaultdict(list)
    dependencies = []
    directories = defaultdict(int)
    # One content read per distinct blob, including exact source hashes.
    cache = {}
    for row in rows:
        blob = row['blob']
        if blob not in cache:
            cache[blob] = git('cat-file', 'blob', blob)
        data = cache[blob]
        row.update(sha256=hashlib.sha256(data).hexdigest(),
                   owner=row['path'].split('/')[0],
                   classification=classify(row['path']),
                   file_type=Path(row['path']).suffix or 'extensionless',
                   dependency_status='AST inventory plus manual candidate review; not proof of dead code',
                   reproducibility_required='preserved unless independently reviewed',
                   deterministic_regeneration=None, independent_archive_verified=False)
        groups[blob].append(row['path'])
        directories[row['owner']] += row['size']
        if row['path'].endswith('.py'):
            try:
                parsed = ast.parse(data)
                imports = [ast.unparse(n) for n in ast.walk(parsed) if isinstance(n, (ast.Import, ast.ImportFrom))]
                dynamic = [ast.unparse(n) for n in ast.walk(parsed) if isinstance(n, ast.Call) and any(s in ast.unparse(n.func) for s in ('import_module', '__import__', 'load', 'run_path'))]
                dependencies.append(dict(path=row['path'], imports=imports, dynamic_review=dynamic))
            except (SyntaxError, UnicodeDecodeError) as exc:
                dependencies.append(dict(path=row['path'], parse_error=str(exc)))
    branches = []
    for ref in git('for-each-ref', '--format=%(refname)', 'refs/remotes/origin/').decode().splitlines():
        if ref.endswith('/HEAD'):
            continue
        other = tree(ref)
        base = {r['path']: r['blob'] for r in rows}
        unique = [r for r in other if base.get(r['path']) != r['blob']]
        branches.append(dict(branch=ref.removeprefix('refs/remotes/origin/'), head=git('rev-parse', ref).decode().strip(),
            merge_base=git('merge-base', 'origin/main', ref).decode().strip(),
            unique_commits=git('rev-list', ref, '--not', 'origin/main').decode().splitlines(),
            differing_files=unique, preservation='Published branch; no retirement in this task',
            disposition='Retain pending content review and authenticated artifact publication'))
    duplicates = [dict(blob=b, paths=p, redundant_bytes=(len(p)-1)*len(cache[b])) for b,p in groups.items() if len(p)>1]
    summary = dict(ref=args.ref, commit=git('rev-parse', args.ref).decode().strip(), files=len(rows), bytes=sum(r['size'] for r in rows),
        python_files=sum(r['path'].endswith('.py') for r in rows), duplicate_groups=len(duplicates),
        redundant_paths=sum(len(d['paths'])-1 for d in duplicates), directories=dict(sorted(directories.items())))
    for name, value in [('repository', dict(summary=summary, files=rows)), ('branch', branches), ('duplicate', duplicates), ('dependency', dependencies), ('artifact', [r for r in rows if r['classification'] in ('PROTECTED','REQUIRED_PROVENANCE','EXTERNAL_ARTIFACT_CANDIDATE')])]:
        (out / (name+'-inventory.json')).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
