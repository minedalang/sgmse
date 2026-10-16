#!/usr/bin/env python3
"""Frozen SSL multi-layer, framewise-MSE comparison for RRSD (no training).

Input is exactly existing rrsd_tcut_screen/{manifest.csv,paired_metrics.csv,
clean/,off/,router/}. Same files/seed are reused across encoders.
Paper 2024 uses last-layer SSL-feature MSE; extra layers/cosine are diagnostics.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

VARIANTS = ('clean', 'off', 'router', 'noise25', 'noise15', 'lowpass3200', 'dropout50ms')


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/coding/sgmse/experiment_logs/rrsd_tcut_screen'))
    p.add_argument('--model-type', choices=['wavlm','unispeech_sv','unispeech'], required=True)
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--max-length', type=int, default=32000)
    p.add_argument('--layers', type=int, nargs='+', default=[3,6,9,12])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    return p.parse_args()


def get_model(a):
    try:
        from transformers import WavLMModel, UniSpeechSatModel, UniSpeechSatForXVector
    except ImportError as e:
        raise RuntimeError('Missing transformers; install in your existing SGMSE environment.') from e
    cls={'wavlm':WavLMModel,'unispeech':UniSpeechSatModel,'unispeech_sv':UniSpeechSatForXVector}[a.model_type]
    try:
        m=cls.from_pretrained(a.model_path,local_files_only=a.local_only)
    except Exception as e:
        raise RuntimeError('Official SSL weights unavailable. Your server previously failed to reach Hugging Face. '
                           'Download once on a networked machine and set --model-path to the copied local model directory. '
                           f'Model={a.model_path}; error={e}') from e
    m.to(a.device).eval().requires_grad_(False)
    return m


def read(path):
    wav,sr=sf.read(path,dtype='float32')
    if sr!=16000 or wav.ndim!=1 or not np.isfinite(wav).all():
        raise ValueError(f'Expected finite 16kHz mono WAV: {path}; sr={sr}, shape={wav.shape}')
    return wav


def preprocess(wav,device):
    a=torch.from_numpy(np.array(wav,dtype=np.float32,copy=True)).to(device=device).view(1,-1)
    a=a-a.mean(dim=-1,keepdim=True)
    return a/(a.abs().amax(dim=-1,keepdim=True)+1e-6)


def perturb(clean,name,noise):
    if name.startswith('noise'):
        snr=int(name[5:]); scale=np.sqrt(np.mean(clean.astype(np.float64)**2)+1e-12) / (10**(snr/20))
        return (clean+noise*scale).astype(np.float32)
    if name=='lowpass3200':
        return sosfiltfilt(butter(6,3200,btype='lowpass',fs=16000,output='sos'),clean).astype(np.float32)
    if name=='dropout50ms':
        x=clean.copy(); span=min(800,max(1,len(x)//8)); start=(len(x)-span)//2
        x[start:start+span]=0; return x
    raise ValueError(name)


def layer_features(m,wav,layers,device):
    x=preprocess(wav,device)
    with torch.inference_mode():
        out=m(input_values=x, output_hidden_states=True, return_dict=True)
    hs=out.hidden_states
    if hs is None or max(layers)>=len(hs):
        raise RuntimeError(f'Requested layers {layers}; model has {0 if hs is None else len(hs)} hidden states.')
    # 0=conv/projected features, 1..N=transformer outputs.
    return {i:hs[i][0].float() for i in layers}


def distance(x,y,kind):
    if x.shape!=y.shape:
        raise RuntimeError(f'Frame alignment mismatch x={x.shape} y={y.shape}')
    if kind=='mse': return float((x-y).square().mean().item())
    if kind=='framecos': return float((1-F.cosine_similarity(x,y,dim=-1,eps=1e-8)).mean().item())
    raise ValueError(kind)


def get_ci(vals,n,seed):
    a=np.asarray(vals,float); rng=np.random.default_rng(seed)
    return tuple(np.quantile(rng.choice(a,size=(n,len(a)),replace=True).mean(axis=1),[.025,.975]))


def report(rows,boot,seed):
    s=[]
    def log(x=''): s.append(x)
    log(f'Paired utterances={len(set(r["filename"] for r in rows))}. Lower D(SSL(X), SSL(Clean)) is closer to Clean.')
    log('Layer id 0=features before transformer, layers 1-12 are transformer blocks.')
    log('MSE=paper-inspired sequence-feature MSE; framecos=extra sensitivity diagnostic; both are independent of DNSMOS.')
    for metric in ('mse','framecos'):
        for layer in sorted(set(int(r['layer']) for r in rows)):
            subset=sorted((r for r in rows if r['metric']==metric and int(r['layer'])==layer),key=lambda a:a['filename'])
            if not subset:continue
            off=np.array([float(r['d_off']) for r in subset]); router=np.array([float(r['d_router']) for r in subset]);delta=router-off
            ci=get_ci(delta,boot,seed+layer)
            dose=sum(float(r['d_noise15'])>float(r['d_noise25'])+1e-7 for r in subset)
            # Reference noise spacing may not be monotonic for every sample.
            log(f'[{metric}] layer={layer} n={len(delta)}: Off={off.mean():.6g} Router={router.mean():.6g} '
                f'Delta={delta.mean():+.6g} CI95=[{ci[0]:+.6g},{ci[1]:+.6g}] '
                f'closer={sum(delta < -1e-7)}/{len(delta)} noise_dose={dose}/{len(delta)} '
                f'noise25={np.mean([float(r["d_noise25"]) for r in subset]):.5g} '
                f'noise15={np.mean([float(r["d_noise15"]) for r in subset]):.5g} '
                f'lowpass={np.mean([float(r["d_lowpass3200"]) for r in subset]):.5g} '
                f'dropout={np.mean([float(r["d_dropout50ms"]) for r in subset]):.5g}')
            for name in ('pesq','stoi','si_sdr'):
                f1='router_'+name; f2='off_'+name
                if f1 in subset[0] and f2 in subset[0]:
                    gains=np.array([float(r[f1])-float(r[f2]) for r in subset])
                    rho=float(spearmanr(delta,gains).statistic) if np.std(gains)>1e-12 and np.std(delta)>1e-12 else float('nan')
                    log(f'    Spearman(DeltaDistance, Delta{name})={rho:+.4f} (negative is expected, not assured)')
    log('CAUTION: this is a screening experiment; compare within-encoder changes, not absolute MSE between distinct encoders.')
    return '\n'.join(s)+'\n'


def main():
    a=args_parser()
    if a.count<1 or a.max_length<1600 or a.bootstrap<100 or min(a.layers)<1:
        raise ValueError('count>0, max-length>=1600, bootstrap>=100, layers>=1 required')
    root=a.root.expanduser().resolve(); outdir=a.out.expanduser().resolve();outdir.mkdir(parents=True,exist_ok=True)
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable: {a.device}')
    with (root/'manifest.csv').open(newline='',encoding='utf8') as f:manifest=list(csv.DictReader(f))[:a.count]
    if not manifest:raise RuntimeError('manifest.csv empty')
    score={}
    if (root/'paired_metrics.csv').is_file():
        with (root/'paired_metrics.csv').open(newline='',encoding='utf8') as f:
            score={r['filename']:r for r in csv.DictReader(f)}
    model=get_model(a)
    print(f'[RRSD-SSL] model={a.model_type} path={a.model_path} n={len(manifest)} layers={a.layers}',flush=True)
    rows=[]
    for j,m in enumerate(manifest):
        name=m['filename']; paths={k:root/k/name for k in ['clean','off','router']}
        raw={k:read(v) for k,v in paths.items()}
        size=min(a.max_length,*map(len,raw.values()))
        if size<1600:raise ValueError(f'Short file {name}')
        raw={k:v[:size] for k,v in raw.items()}
        deterministic=int.from_bytes(hashlib.sha256(f'{a.seed}|{name}'.encode()).digest()[:8],'little')
        noise=np.random.default_rng(deterministic).standard_normal(size).astype('float32')
        noise/=np.sqrt(np.mean(noise.astype(np.float64)**2)+1e-12)
        for perturb_name in ['noise25','noise15','lowpass3200','dropout50ms']:
            raw[perturb_name]=perturb(raw['clean'],perturb_name,noise)
        data={k:layer_features(model,raw[k],a.layers,a.device) for k in VARIANTS}
        for l in a.layers:
            for kind in ('mse','framecos'):
                r={'filename':name,'layer':l,'metric':kind,'clip_samples':size}
                ref=data['clean'][l]
                for k in VARIANTS:
                    if k=='clean':continue
                    r['d_'+k]=distance(data[k][l],ref,kind)
                r['delta_router_off']=r['d_router']-r['d_off']
                if name in score:
                    for k in ('off_pesq','router_pesq','off_stoi','router_stoi','off_si_sdr','router_si_sdr'):
                        if k in score[name]:r[k]=score[name][k]
                rows.append(r)
        if (j+1)%4==0 or j+1==len(manifest):print(f'[RRSD-SSL] {j+1}/{len(manifest)} completed ({name})',flush=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (outdir/'frame_features.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    text=report(rows,a.bootstrap,a.seed)
    (outdir/'report.txt').write_text(text,encoding='utf8')
    print(text,flush=True)
    print(f'[DONE] {outdir}',flush=True)


if __name__=='__main__':main()
