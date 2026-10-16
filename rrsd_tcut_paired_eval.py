#!/usr/bin/env python3
"""Paired PESQ/STOI/SI-SDR subset screen for 4 RRSD inference arms.
Not a replacement for DNSMOS-SIG/OVRL evaluation.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf


def si_sdr(ref, est, eps=1e-10):
    ref = ref - ref.mean()
    est = est - est.mean()
    proj = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    residual = est - proj
    return float(10.0 * np.log10((np.dot(proj, proj) + eps) / (np.dot(residual, residual) + eps)))


def metrics(ref, est, sr):
    from pesq import pesq
    from pystoi import stoi
    L = min(len(ref), len(est))
    ref, est = ref[:L].astype(np.float64), est[:L].astype(np.float64)
    if sr != 16000:
        raise ValueError(f'Expected 16kHz, got {sr}')
    return {'pesq': float(pesq(sr, ref.astype('float32'), est.astype('float32'), 'wb')),
            'stoi': float(stoi(ref, est, sr, extended=False)),
            'si_sdr': si_sdr(ref, est)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Output root with manifest.csv + 4 arm dirs')
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    root = Path(a.root)
    arms = ['off','router','cut005','cut010']
    with (root/'manifest.csv').open(newline='',encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    data = []
    for i, r in enumerate(manifest):
        ref,sr = sf.read(r['clean'], dtype='float32')
        if ref.ndim != 1:
            ref = ref.mean(axis=-1)
        row = {'filename': r['filename'], 'uttid': r['uttid']}
        for arm in arms:
            path = root/arm/r['filename']
            if not path.is_file():
                raise FileNotFoundError(path)
            est, esr = sf.read(path, dtype='float32')
            if est.ndim != 1:
                est = est.mean(axis=-1)
            if sr != esr:
                raise ValueError(f'Sample rate differs for {path}')
            got = metrics(ref, est, int(sr))
            for k, v in got.items():
                row[f'{arm}_{k}'] = v
        data.append(row)
        if (i+1)%8 == 0:
            print(f'[EVAL] {i+1}/{len(manifest)}', flush=True)
    if not data:
        raise SystemExit('No samples')
    with (root/'paired_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)
    rng=np.random.default_rng(2026)
    print(f'Paired evaluation: n={len(data)}, baseline=router; positive Δ means improvement')
    for metric in ['pesq','stoi','si_sdr']:
        base=np.array([r[f'router_{metric}'] for r in data])
        print(f'{metric}: Router mean={base.mean():.6f} (Off={np.mean([r[f"off_{metric}"] for r in data]):.6f})')
        for arm in ['cut005','cut010']:
            diff=np.array([r[f'{arm}_{metric}'] for r in data]) - base
            draws = rng.choice(diff, size=(a.bootstrap,len(diff)),replace=True).mean(axis=1)
            ci=np.percentile(draws,[2.5,97.5])
            print(f'  {arm}: Δ={diff.mean():+.6f}, 95% paired bootstrap CI=[{ci[0]:+.6f}, {ci[1]:+.6f}],'
                  f' improved={int((diff>0).sum())}/{len(diff)}')
    print(f'CSV: {root/"paired_metrics.csv"}')
    print('NOTE: DNSMOS-SIG/OVRL still require the existing DNSMOS evaluator.')

if __name__=='__main__':
    main()
