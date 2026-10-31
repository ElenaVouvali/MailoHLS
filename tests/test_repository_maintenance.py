"""Contract tests for offline artifacts and historical source entry points."""
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('artifact_tool', ROOT / 'scripts/artifacts.py')
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


def record(data):
    return dict(path='data/item.tar', size=len(data), sha256=hashlib.sha256(data).hexdigest(), download_url='https://example.invalid/item')


def test_fetch_verify_and_overwrite(tmp_path):
    data = b'authenticated payload'
    row = record(data)
    with patch.object(artifacts, 'urlopen', return_value=io.BytesIO(data)):
        path = artifacts.fetch(row, tmp_path)
    artifacts.verify(path, row)
    with pytest.raises(ValueError, match='overwrite'):
        artifacts.fetch(row, tmp_path)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='mismatch'):
        artifacts.verify(path, row)
    with pytest.raises(ValueError, match='Missing'):
        artifacts.verify(tmp_path / 'absent', row)


def test_failed_download_never_published(tmp_path):
    with patch.object(artifacts, 'urlopen', return_value=io.BytesIO(b'corrupt')):
        with pytest.raises(ValueError, match='mismatch'):
            artifacts.fetch(record(b'expected'), tmp_path)
    assert not (tmp_path / 'data/item.tar').exists()


@pytest.mark.parametrize('name,kind', [('../escape', 'file'), ('/absolute', 'file'), ('link', 'symlink')])
def test_archive_rejects_unsafe_members(tmp_path, name, kind):
    archive = tmp_path / 'input.tar'
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo(name)
        if kind == 'symlink':
            member.type = tarfile.SYMTYPE
            member.linkname = '/tmp/escape'
        tar.addfile(member)
    with pytest.raises(ValueError):
        artifacts.extract(archive, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_restore_fresh_and_refuse_existing(tmp_path):
    archive = tmp_path / 'input.tar'
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo('nested/source.txt')
        member.size = 4
        tar.addfile(member, io.BytesIO(b'data'))
    artifacts.extract(archive, tmp_path / 'output')
    assert (tmp_path / 'output/nested/source.txt').read_bytes() == b'data'
    with pytest.raises(ValueError, match='must not exist'):
        artifacts.extract(archive, tmp_path / 'output')


@pytest.mark.parametrize('legacy,canonical', [('data.py','GNN_branch/data.py'), ('cpp_to_gnn_emb.py','GNN_branch/cpp_to_gnn_emb.py'), ('filter_jsonl.py','Preprocessing/filter_jsonl.py'), ('inference_qLoRA.py','LLM_branch/inference/inference_qLoRA.py'), ('cpp_to_gexf_deterministic.py','GNN_branch/cpp_to_gexf_deterministic.py'), ('filter_pt.py','Preprocessing/filter_pt.py'), ('gexf_to_pt_zero.py','GNN_branch/gexf_to_pt_zero.py'), ('parameter.py','GNN_branch/parameter.py'), ('pragma_free_pts.py','GNN_branch/pragma_free_pts.py'), ('pt_to_gnn_emb.py','GNN_branch/pt_to_gnn_emb.py'), ('result.py','GNN_branch/result.py'), ('server.py','GNN_branch/server.py')])
def test_legacy_namespace_and_script_contract(tmp_path, legacy, canonical):
    src = tmp_path / 'src'
    src.mkdir()
    (src / legacy).write_bytes((ROOT / 'src' / legacy).read_bytes())
    target = tmp_path / canonical
    target.parent.mkdir(parents=True)
    target.write_text('class Payload: pass\nif __name__ == "__main__":\n    print("legacy CLI")\n')
    module = legacy[:-3]
    result = subprocess.check_output([sys.executable, '-c', f'import {module}, pickle; obj = {module}.Payload(); assert type(pickle.loads(pickle.dumps(obj))) is {module}.Payload; print({module}.Payload.__module__)'], cwd=src, text=True)
    assert result.strip() == module
    assert subprocess.check_output([sys.executable, str(src / legacy)], text=True).strip() == 'legacy CLI'


def test_artifact_path_escape_and_symlink(tmp_path):
    outside = tmp_path.parent / 'outside-test-artifact'
    (tmp_path / 'link').symlink_to(outside, target_is_directory=True)
    for name in ('../escape', '/absolute', 'link/item'):
        with pytest.raises(ValueError, match='escapes'):
            artifacts.destination(tmp_path, name)


def test_internal_symlink_also_rejected(tmp_path):
    (tmp_path/'real').mkdir()
    (tmp_path/'alias').symlink_to(tmp_path/'real', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        artifacts.destination(tmp_path, 'alias/file')


def test_extraction_destination_race_does_not_replace(tmp_path, monkeypatch):
    archive = tmp_path/'data.tar'
    with tarfile.open(archive, 'w') as tar:
        item = tarfile.TarInfo('file'); item.size = 1
        tar.addfile(item, io.BytesIO(b'x'))
    target = tmp_path/'output'
    original_open = artifacts.tarfile.open
    def concurrent_open(*args, **kwargs):
        target.mkdir()
        (target/'sentinel').write_text('concurrent work')
        return original_open(*args, **kwargs)
    monkeypatch.setattr(artifacts.tarfile, 'open', concurrent_open)
    with pytest.raises(FileExistsError):
        artifacts.extract(archive, target)
    assert (target/'sentinel').read_text() == 'concurrent work'


def test_sealed_package_determinism_and_restore(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    (repo/'inputs').mkdir()
    script=repo/'inputs/run.sh'; script.write_text('#!/bin/sh\necho original\n');script.chmod(0o755)
    subprocess.run(['git','-C',str(repo),'add','inputs'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    script.write_text('uncommitted local state')
    monkeypatch.chdir(repo)
    monkeypatch.setattr(artifacts, "ROOT", repo)
    row=dict(id='fixture',source_commit=commit,restore_roots=['inputs'])
    a,b=tmp_path/'a.tar.gz',tmp_path/'b.tar.gz'
    artifacts.package(row,a);artifacts.package(row,b)
    assert a.read_bytes()==b.read_bytes()
    row.update(size=a.stat().st_size,sha256=artifacts.sha256(a))
    root=tmp_path/'restored'
    assert artifacts.restore(a,row,root)==1
    assert (root/'inputs/run.sh').read_text()=='#!/bin/sh\necho original\n'
    assert (root/'inputs/run.sh').stat().st_mode & 0o111
    assert artifacts.restore(a,row,root)==1  # identical files are safe no-ops
    row['git_roots']=['inputs']
    row['git_objects']={'inputs':subprocess.check_output(['git','rev-parse',commit+':inputs'],text=True).strip()}
    with tarfile.open(a) as tar:
        row['member_manifest_sha256']=hashlib.sha256(tar.extractfile('MAILOHLS_ARTIFACT.json').read()).hexdigest()
    assert artifacts.restore_git(row,tmp_path/'git-restored')==1
    bad=dict(row,member_manifest_sha256='0'*64)
    with pytest.raises(ValueError,match='manifest mismatch'):
        artifacts.restore_git(bad,tmp_path/'bad-git-restore')
    (root/'inputs/run.sh').write_text('user work')
    with pytest.raises(ValueError,match='Restore conflict'):
        artifacts.restore(a,row,root)
    assert (root/'inputs/run.sh').read_text()=='user work'


def test_externalization_manifest_preserves_selected_contracts():
    import json
    manifest=json.loads((ROOT/'configs/artifact-migration.json').read_text())
    paths=[p for row in manifest['artifacts'] for p in row['externalized_paths']]
    assert len(paths)==len(set(paths))
    forbidden=('Data/', 'mailohls_runs/', 'GNN_branch/MLIR_graphs/', 'GNN_branch/targets/', 'Checkpoints/gnn_final_absolute_direct_rank_s123/')
    assert not any(p.startswith(forbidden) or p=='FINAL_ARTIFACTS.json' for p in paths)
    assert all(row['removed_paths']==len(row['externalized_paths']) for row in manifest['artifacts'])


def test_git_catalog_recovery_and_checksum_rejection(tmp_path, monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    catalog=repo/'graphs/latest';catalog.mkdir(parents=True)
    payload=b'{"compiler":"evidence"}'
    (catalog/'graph.json').write_bytes(payload)
    checks=hashlib.sha256(payload).hexdigest()+'  graph.json\n'
    (catalog/'SHA256SUMS').write_text(checks)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
    ref=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    obj=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD:graphs/latest'],text=True).strip()
    monkeypatch.setattr(artifacts,'ROOT',repo)
    row=dict(id='catalog',source_commit=ref,git_roots=['graphs/latest'],git_objects={'graphs/latest':obj},catalog_checksums_sha256=hashlib.sha256(checks.encode()).hexdigest(),catalog_file_count=1)
    assert artifacts.restore_git(row,tmp_path/'restore')==1
    assert (tmp_path/'restore/graphs/latest/graph.json').read_bytes()==payload
    bad=dict(row,catalog_file_count=2)
    with pytest.raises(ValueError,match='member set'):
        artifacts.restore_git(bad,tmp_path/'bad')
    assert not (tmp_path/'bad/graphs').exists()
