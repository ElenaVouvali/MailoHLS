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


@pytest.mark.parametrize('legacy,canonical', [('data.py','GNN_branch/data.py'), ('cpp_to_gnn_emb.py','GNN_branch/cpp_to_gnn_emb.py'), ('filter_jsonl.py','Preprocessing/filter_jsonl.py'), ('inference_qLoRA.py','LLM_branch/inference/inference_qLoRA.py')])
def test_legacy_namespace_and_script_contract(tmp_path, legacy, canonical):
    src = tmp_path / 'src'
    src.mkdir()
    (src / legacy).write_bytes((ROOT / 'src' / legacy).read_bytes())
    target = tmp_path / canonical
    target.parent.mkdir(parents=True)
    target.write_text('class Payload: pass\nif __name__ == "__main__":\n    print("legacy CLI")\n')
    module = legacy[:-3]
    result = subprocess.check_output([sys.executable, '-c', f'import {module}; print({module}.Payload.__module__)'], cwd=src, text=True)
    assert result.strip() == module
    assert subprocess.check_output([sys.executable, str(src / legacy)], text=True).strip() == 'legacy CLI'


def test_artifact_path_escape_and_symlink(tmp_path):
    outside = tmp_path.parent / 'outside-test-artifact'
    (tmp_path / 'link').symlink_to(outside, target_is_directory=True)
    for name in ('../escape', '/absolute', 'link/item'):
        with pytest.raises(ValueError, match='escapes'):
            artifacts.destination(tmp_path, name)
