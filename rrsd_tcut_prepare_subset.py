#!/usr/bin/env python3
"""Prepare deterministic DNS scp validation subset with paired clean references.

Matches the default WavePairsScp clean template clean_fileid_{fileid}.wav.
"""
import argparse
import csv
from pathlib import Path
import random
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scp', required=True)
    p.add_argument('--clean-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--clean-template', default='clean_fileid_{fileid}.wav')
    a = p.parse_args()
    if a.count < 1:
        p.error('--count must be >=1')
    scp = Path(a.scp).resolve()
    clean_dir = Path(a.clean_dir).resolve()
    out = Path(a.out).resolve()
    if not scp.is_file() or not clean_dir.is_dir():
        raise SystemExit('SCP or clean directory does not exist.')
    records = []
    n_bad = 0
    for line in scp.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[1].strip().endswith('|'):
            n_bad += 1
            continue
        utt, path = parts
        noisy = Path(path)
        if not noisy.is_absolute():
            noisy = scp.parent / noisy
        noisy = noisy.resolve()
        match = re.search(r'fileid_(\d+)', noisy.name)
        if match:
            fileid = int(match.group(1))
        else:
            match = re.search(r'(\d{3,})', utt)
            if not match:
                n_bad += 1
                continue
            fileid = int(match.group(1))
        clean = clean_dir / a.clean_template.format(fileid=fileid)
        if not noisy.is_file() or not clean.is_file():
            n_bad += 1
            continue
        records.append((utt, noisy, clean))
    if not records:
        raise SystemExit('No valid noisy/clean pairs. Verify wav.scp and clean_template.')
    selected = random.Random(a.seed).sample(records, min(a.count, len(records)))
    in_dir = out / 'noisy'
    clean_subdir = out / 'clean'
    for d in (in_dir, clean_subdir):
        d.mkdir(parents=True, exist_ok=True)
        # Only remove previous symlinks created for this subset; preserve any real files.
        for f in d.glob('*.wav'):
            if f.is_symlink():
                f.unlink()
            else:
                raise SystemExit(f'Refusing to overwrite regular wav file: {f}')
    manifest = out / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['filename','uttid','noisy','clean'])
        wr.writeheader()
        for i, (utt, noisy, clean) in enumerate(selected):
            # Synthetic alias guarantees unique basenames and deterministic per-file RNG.
            clean_match = re.search(r'fileid_(\d+)', clean.name)
            clean_id = clean_match.group(1) if clean_match else str(i)
            name = f'{i:03d}_fileid_{clean_id}.wav'
            (in_dir / name).symlink_to(noisy)
            (clean_subdir / name).symlink_to(clean)
            wr.writerow({'filename': name, 'uttid': utt, 'noisy': str(noisy), 'clean': str(clean)})
    print(f'[SUBSET] n={len(selected)} out={out} missing_or_unsupported={n_bad}')
    print(f'[SUBSET] manifest={manifest}')

if __name__ == '__main__':
    main()
