#!/usr/bin/env python3
"""Stage locked noisy inputs into two inference arms using symlinks only."""
import csv
import hashlib
from pathlib import Path

ROOT = Path('/data/coding/sgmse')
EXP = ROOT / 'experiments/speaker_headroom_replication_64_20261017'
LOG = ROOT / 'experiment_logs/speaker_headroom_replication_64_20261017'
MANIFEST = EXP / 'manifest.csv'
LOCKED_HASH = 'c04657fa21a33c7f2dc749514cec899767101ba53f46222e9f4ab8577e2f4492'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if digest(MANIFEST) != LOCKED_HASH:
        raise RuntimeError('manifest changed after lock')
    rows = list(csv.DictReader(MANIFEST.open(newline='', encoding='utf-8')))
    if len(rows) != 128:
        raise RuntimeError(f'expected 128 manifest rows, found {len(rows)}')
    for arm in ('off', 'router'):
        stage = LOG / 'inputs' / arm
        stage.mkdir(parents=True, exist_ok=True)
        for row in rows:
            src = Path(row['noisy_path'])
            if digest(src) != row['noisy_sha256']:
                raise RuntimeError(f'noisy source hash changed: {src}')
            dst = stage / row['filename']
            if dst.is_symlink():
                if dst.resolve() != src.resolve():
                    raise RuntimeError(f'wrong existing symlink: {dst}')
            elif dst.exists():
                raise RuntimeError(f'not overwriting staged path: {dst}')
            else:
                dst.symlink_to(src)
    print(f'staged {len(rows)} symlinks per arm; manifest sha256={LOCKED_HASH}')

if __name__ == '__main__':
    main()
