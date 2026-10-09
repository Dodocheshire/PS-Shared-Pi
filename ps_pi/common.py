"""Pinned official imports, atomic artifacts, and a shared local GPU lease."""
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    partial.replace(path)

def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    torch.save(value, partial); partial.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def verify_official():
    source = json.loads((ROOT / 'UPSTREAM.json').read_text())
    hashes = source['official_file_sha256']
    # Manifest spelling is pinned at repository creation.
    if not hashes:
        hashes = next(v for v in source.values() if isinstance(v, dict))
    for relative, expected in hashes.items():
        if digest(ROOT / 'official' / relative) != expected:
            raise RuntimeError(f'Official source changed: {relative}')
    return source['commit']

def official_module(name):
    verify_official()
    official = str(ROOT / 'official')
    if official not in sys.path: sys.path.insert(0, official)
    return importlib.import_module(name)

@contextlib.contextmanager
def gpu_lease(label):
    # Matches the completed D-Pi stage's flock, but requires no D-Pi import.
    directory = ROOT.parent / 'D-Pi/artifacts/resources'
    if not directory.parent.parent.exists(): directory = ROOT / 'artifacts/resources'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'gpu_work.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        status = {'pid': os.getpid(), 'label': label, 'acquired_at': time.time(), 'held': True}
        status_path = ROOT / 'artifacts/gpu_owner.json'
        atomic_json(status, status_path)
        try: yield
        finally:
            status.update(held=False, released_at=time.time()); atomic_json(status, status_path)
            fcntl.flock(stream, fcntl.LOCK_UN)
