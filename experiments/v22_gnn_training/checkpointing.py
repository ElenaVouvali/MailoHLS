"""Atomic persistence and deterministic RNG helpers for V2.2 experiments."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_torch_save(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def capture_rng_state() -> dict:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    else:
        state['torch_cuda_all'] = None
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    cuda_states = state.get('torch_cuda_all')
    if cuda_states is not None:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG state does not match the visible CUDA device set')
        torch.cuda.set_rng_state_all(cuda_states)


def verify_fingerprint(checkpoint: dict, expected: str) -> None:
    actual = checkpoint.get('protocol_fingerprint')
    if actual != expected:
        raise ValueError(f'Resume refused: protocol fingerprint mismatch ({actual} != {expected})')


def active_pid(pid: int) -> bool:
    if type(pid) is not int or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
