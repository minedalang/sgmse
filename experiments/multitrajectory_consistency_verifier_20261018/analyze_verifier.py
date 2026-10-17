#!/usr/bin/env python3
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'experiments/multitrajectory_oracle_heldout16/consistency_probe.json'
OUT = Path(__file__).resolve().parent
BOOT, SEED = 20000, 20261018

def z(v):
    v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-12)

def grouped(rows, lam=0.5):
    groups={}
    for r in rows: groups.setdefault(r['filename'],[]).append(r)
    out=[]
    for fn, rs in sorted(groups.items()):
        base=next(r for r in rs if r['seed']==2026)
        ov=np.array([r['dnsmos_ovrl'] for r in rs]); co=np.array([r['consistency'] for r in rs])
        j=int(np.argmax(z(ov)-lam*z(co))); sel=rs[j]
        out.append({
            'filename':fn, 'selected_seed':sel['seed'],
            'ovrl_delta':sel['dnsmos_ovrl']-base['dnsmos_ovrl'],
            'sisdr_delta':sel['sisdr']-base['sisdr'],
            'pesq_delta':sel['pesq']-base['pesq'],
            'stoi_delta':sel['stoi']-base['stoi'],
            'sig_delta':sel['dnsmos_sig']-base['dnsmos_sig'],
            'bak_delta':sel['dnsmos_bak']-base['dnsmos_bak'],
            'selected_consistency':sel['consistency'],
        })
    return out

def summary(rows):
    keys=['ovrl_delta','sisdr_delta','pesq_delta','stoi_delta','sig_delta','bak_delta']
    out={'n':len(rows),'means':{},'negative_sisdr_files':int(sum(r['sisdr_delta']<0 for r in rows))}
    for k in keys:
        x=np.array([r[k] for r in rows],float); out['means'][k]=float(x.mean())
        rng=random.Random(SEED); vals=[]
        for _ in range(BOOT): vals.append(float(np.mean([x[rng.randrange(len(x))] for _ in x])))
        vals.sort(); out.setdefault('ci95',{})[k]=[vals[int(.025*BOOT)],vals[int(.975*BOOT)]]
    return out

def main():
    data=json.load(open(SRC)); result={'experiment_id':'multitrajectory_consistency_verifier_20261018','lambda':0.5,'source':str(SRC),'bootstrap':{'replicates':BOOT,'seed':SEED},'splits':{}}
    all_rows={}
    for split in ('dev','heldout'):
        rows=grouped(data[split],0.5); all_rows[split]=rows; result['splits'][split]=summary(rows)
        with (OUT/f'{split}_selected.json').open('w') as f: json.dump(rows,f,indent=2)
    with (OUT/'metrics.json').open('w') as f: json.dump(result,f,indent=2)

if __name__=='__main__': main()
