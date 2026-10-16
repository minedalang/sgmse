#!/usr/bin/env python3
"""Summarize [RRSD-PG] read-only gradient-sign probe logs by diffusion-time bin."""
import argparse
import math
import re
from pathlib import Path
import numpy as np


def parse(path):
    rows=[]
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.startswith('[RRSD-PG] '):
            continue
        pairs=dict(re.findall(r'([a-zA-Z_][\w]*)=([^ ]+)',line))
        try:
            row={k:(v if k=='anchor' else float(v)) for k,v in pairs.items()}
        except ValueError:
            continue
        rows.append(row)
    return rows


def ci(x, n=2000, seed=2026):
    a=np.asarray(x,dtype=float)
    rng=np.random.default_rng(seed)
    if len(a)==0:
        return (float('nan'), float('nan'))
    means=np.asarray([rng.choice(a,size=len(a),replace=True).mean() for _ in range(n)])
    return tuple(np.quantile(means, [.025,.975]))


def group(label, rows):
    if not rows:
        print(f'\n{label}: n=0');return
    print(f'\n{label}: n={len(rows)}, mean t={np.mean([r["t"] for r in rows]):.4f}')
    fields={
      'alpha_mean': 'Mean initial Router alpha',
      'active':'Fraction of active tiles (alpha>0.001)',
      'fgood':'Future-beneficial tiles',
      'pgood':'SSL-beneficial tiles',
      'conflict':'Conflict: future benefits, SSL harms',
      'agree':'Agreement: both benefit',
      'bothbad':'Both consider guidance harmful',
      'alpha_conflict_budget':'Fraction of alpha budget on conflict tiles',
      'gf_abs':'Mean abs future gradient per tile',
      'gp_abs':'Mean abs SSL gradient per tile',
    }
    for k,v in fields.items():
        x=np.asarray([r[k] for r in rows], dtype=float)
        lo,hi=ci(x)
        print(f'  {v}: {x.mean():.6g}  95% sample bootstrap CI [{lo:.6g},{hi:.6g}]')
    active_rows=[r for r in rows if r.get('active_tiles',0)>0]
    print(f'  Samples with active tiles: {len(active_rows)}/{len(rows)}')
    if active_rows:
        x=np.asarray([r['conflict_on_active'] for r in active_rows])
        print(f'  Conflict among active tiles, conditional on any active: mean={x.mean():.5f}, median={np.median(x):.5f}')
    fd=[r for r in rows if int(r.get('fd',0))==1]
    for tag in ('f','p'):
        valid=[r for r in fd if int(r.get(f'fd_{tag}_valid',0))==1]
        ok=sum(int(r.get(f'fd_{tag}_sign_ok',0)) for r in valid)
        print(f'  FD {tag} directional sign: {ok}/{len(valid)} valid, {len(fd)-len(valid)} below numerical floor')
        if valid:
            print(f'  FD {tag} mean prediction={np.mean([r[f"fd_{tag}_pred"] for r in valid]):+.5e}, '
                  f'observation={np.mean([r[f"fd_{tag}_obs"] for r in valid]):+.5e}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('log',type=Path)
    args=p.parse_args()
    rows=parse(args.log)
    print(f'Parsed valid rows: {len(rows)}; anchors={sorted({r.get("anchor","unknown") for r in rows})}')
    if not rows:
        raise SystemExit('No [RRSD-PG] results; check time filter, script launch, or earlier error.')
    group('ALL selected',rows)
    group('LOW [0.05,0.2)',[r for r in rows if .05<=r['t']<.2])
    group('MID [0.2,0.5)',[r for r in rows if .2<=r['t']<.5])
    print('\nImportant: normalized local derivatives at the reported anchor (zero or router), NOT DNSMOS gradients. '
          'Alpha gradients from frozen WavLM are diagnostic only. '
          'Signed finite differences should agree before treating a conflict mask as meaningful.')

if __name__=='__main__':main()
