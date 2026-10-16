#!/usr/bin/env python3
"""Paired metrics for an RRSD selective-window inference arm."""
import argparse, csv
from pathlib import Path
import numpy as np
import soundfile as sf

def si_sdr(ref, est, eps=1e-10):
    ref = ref - ref.mean(); est = est - est.mean()
    proj = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    residual = est - proj
    return float(10*np.log10((np.dot(proj, proj)+eps)/(np.dot(residual,residual)+eps)))

def metrics(ref, est, sr):
    from pesq import pesq
    from pystoi import stoi
    L=min(len(ref),len(est)); ref=ref[:L].astype('float64'); est=est[:L].astype('float64')
    if sr != 16000: raise ValueError(sr)
    return {'pesq':float(pesq(sr,ref.astype('float32'),est.astype('float32'),'wb')),
            'stoi':float(stoi(ref,est,sr,extended=False)), 'si_sdr':si_sdr(ref,est)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--off-dir',required=True); p.add_argument('--router-dir',required=True); p.add_argument('--selective-dir',required=True); p.add_argument('--out',required=True); p.add_argument('--bootstrap',type=int,default=20000)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader(open(a.manifest,newline='',encoding='utf-8'))); data=[]
    for r in rows:
        ref,sr=sf.read(r['clean'],dtype='float32'); ref=ref.mean(axis=-1) if ref.ndim>1 else ref
        row={'filename':r['filename'],'uttid':r.get('uttid','')}
        for arm,d in [('off',a.off_dir),('router',a.router_dir),('selective',a.selective_dir)]:
            est,esr=sf.read(str(Path(d)/r['filename']),dtype='float32'); est=est.mean(axis=-1) if est.ndim>1 else est
            if esr!=sr: raise ValueError((r['filename'],sr,esr))
            for k,v in metrics(ref,est,int(sr)).items(): row[f'{arm}_{k}']=v
        data.append(row)
    with (out/'paired_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    rng=np.random.default_rng(2026); summary=[]
    for metric in ('pesq','stoi','si_sdr'):
        off=np.array([r[f'off_{metric}'] for r in data]); router=np.array([r[f'router_{metric}'] for r in data]); sel=np.array([r[f'selective_{metric}'] for r in data])
        summary.append(f'{metric}: off={off.mean():.6f}, router={router.mean():.6f}, selective={sel.mean():.6f}')
        for name,base in [('router',off),('selective',off),('selective_vs_router',router)]:
            vals={'router':router,'selective':sel,'selective_vs_router':sel}[name]; diff=vals-base
            draws=rng.choice(diff,size=(a.bootstrap,len(diff)),replace=True).mean(axis=1); ci=np.percentile(draws,[2.5,97.5])
            summary.append(f'  {name} delta={diff.mean():+.6f}, CI=[{ci[0]:+.6f},{ci[1]:+.6f}], improved={int((diff>0).sum())}/{len(diff)}')
    (out/'summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8'); print('\n'.join(summary)); print(f'CSV: {out/"paired_metrics.csv"}')
if __name__=='__main__': main()
