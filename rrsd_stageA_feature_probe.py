#!/usr/bin/env python3
"""Stage-A RRSD frozen-encoder sensitivity probe. No SGMSE inference/training.

Input: rrsd_tcut_screen/{manifest.csv,clean/,off/,router/,paired_metrics.csv}.
Feature extraction faithfully reproduces original training's 16-kHz first-32000-sample,
DC-centering + independent peak-normalization, with no gradient and an eval encoder.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr


FEATURES = ('off', 'router', 'noise25', 'noise15', 'lowpass3200', 'dropout50ms')


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/coding/sgmse/experiment_logs/rrsd_tcut_screen'))
    p.add_argument('--model-type', choices=['unispeech', 'unispeech_sv', 'campplus', 'xvector'], default='unispeech_sv',
                   help='Select EXACT encoder type used by the contrastive experiment.')
    p.add_argument('--model-path', type=str, default='microsoft/unispeech-sat-base-plus-sv',
                   help='Hugging Face repo or local pretrained model path; no random init fallback.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-plm-length', type=int, default=32000)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--bootstraps', type=int, default=2000)
    p.add_argument('--local-only', action='store_true', help='For UniSpeech-SAT, do not access Hugging Face network.')
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()
    if args.max_plm_length < 1600:
        p.error('--max-plm-length must be >=1600')
    if args.count < 1:
        p.error('--count must be >=1')
    if args.model_type in ('unispeech','unispeech_sv') and args.model_path == '':
        p.error('model-path missing')
    return args


def load_encoder(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable; use --device cpu or fix GPU environment (requested {args.device}).')
    if args.model_type in ('unispeech','unispeech_sv'):
        from transformers import UniSpeechSatModel, UniSpeechSatForXVector
        cls = UniSpeechSatForXVector if args.model_type == 'unispeech_sv' else UniSpeechSatModel
        model = cls.from_pretrained(args.model_path, local_files_only=args.local_only)
    elif args.model_type == 'campplus':
        from sgmse.speaker_encoders import WeSpeakerCAMPPlusEncoder
        model = WeSpeakerCAMPPlusEncoder(model_name_or_path=args.model_path)
    elif args.model_type == 'xvector':
        from sgmse.xvector_encoder import SpeechBrainXVectorEncoder
        model = SpeechBrainXVectorEncoder(model_name_or_path=args.model_path)
    else:
        raise ValueError(args.model_type)
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()
    return model


def clean_audio(path):
    wav, rate = sf.read(path, dtype='float32', always_2d=False)
    if rate != 16000:
        raise ValueError(f'Expected 16k audio, got {rate} in {path}')
    if wav.ndim != 1:
        raise ValueError(f'Requires mono audio for a controlled paired evaluation: {path} shape={wav.shape}')
    if not np.all(np.isfinite(wav)):
        raise ValueError(f'Nonfinite input audio in {path}')
    return wav.astype(np.float32, copy=False)


def perturb(clean, name, noise):
    if name.startswith('noise'):
        snr_db = int(name.replace('noise',''))
        scale = float(np.sqrt(np.mean(clean.astype(np.float64)**2) + 1e-12)) / (10**(snr_db/20))
        return (clean + scale * noise).astype(np.float32)
    if name == 'lowpass3200':
        sos = butter(6, 3200, btype='lowpass', fs=16000, output='sos')
        return sosfiltfilt(sos, clean).astype(np.float32)
    if name == 'dropout50ms':
        result = clean.copy()
        width = min(800, max(1, len(clean)//8))
        start = (len(clean)-width)//2
        result[start:start+width] = 0.0
        return result
    raise ValueError(name)


def normalize_like_training(audio, device):
    a = torch.as_tensor(np.array(audio, copy=True), device=device, dtype=torch.float32).unsqueeze(0)
    a = a-a.mean(dim=1, keepdim=True)
    a = a/(a.abs().amax(dim=1, keepdim=True)+1e-6)
    return a


def embedding_dict(model, wav, kind):
    if kind == 'unispeech_sv':
        out = model(input_values=wav, output_hidden_states=True, return_dict=True)
        if out.hidden_states is None or out.embeddings is None:
            raise RuntimeError('UniSpeech-SAT-SV missing hidden_states or embeddings; checkpoint/transformers mismatch.')
        # EXACT feature selection used in user's dual-representation model.py
        return {'hidden_mean': out.hidden_states[-1].mean(dim=1), 'speaker_embedding': out.embeddings}
    if kind == 'unispeech':
        out = model(input_values=wav, return_dict=True)
        return {'hidden_mean': out.last_hidden_state.mean(dim=1)}
    return {'speaker_embedding': model(wav)}


def similarity_distance(vec_a, vec_b):
    # Project embeddings to a unit hypersphere; cosine distance in [0,2].
    a = F.normalize(vec_a.float().flatten(1), p=2, dim=-1)
    b = F.normalize(vec_b.float().flatten(1), p=2, dim=-1)
    return float((1-F.cosine_similarity(a,b,dim=-1)).item())


def summarize(rows, bootstraps, seed):
    lines = []
    def put(s=''):
        lines.append(s)
    put(f'Samples: {len(set(x["filename"] for x in rows))}. Paired cosine distance (lower=closer to clean).')
    put('NOTE: 25dB/15dB noise is relative to clean RMS before peak-normalization; comparisons are diagnostic, not PESQ or DNSMOS.')
    types = list(dict.fromkeys(x['feature'] for x in rows))
    for branch in types:
        rr = [r for r in rows if r['feature']==branch]
        rr.sort(key=lambda r: r['filename'])
        d_off = np.array([float(x['d_off']) for x in rr]); d_router=np.array([float(x['d_router']) for x in rr]); delta=d_router-d_off
        rng=np.random.default_rng(seed)
        draws=rng.choice(delta, size=(bootstraps,len(delta)), replace=True).mean(axis=1)
        lo, hi=np.percentile(draws,[2.5,97.5])
        put('')
        put(f'--- {branch}: n={len(rr)} ---')
        put(f'Mean d(Off,Clean)={d_off.mean():.7g} ; d(Router,Clean)={d_router.mean():.7g}')
        put(f'Delta d=Router-Off: mean={delta.mean():+.8g}, median={np.median(delta):+.8g}, '
            f'paired bootstrap 95% CI=[{lo:+.8g}, {hi:+.8g}], closer={int((delta<-1e-6).sum())}/{len(delta)}, '
            f'farther={int((delta>1e-6).sum())}/{len(delta)} (dead-zone=1e-6)')
        for key in ['noise25','noise15','lowpass3200','dropout50ms']:
            a=np.array([float(x['d_'+key]) for x in rr])
            put(f'  Clean + {key:13s}  d_to_clean mean={a.mean():.7g} median={np.median(a):.7g}')
        n25=np.array([float(x['d_noise25']) for x in rr]); n15=np.array([float(x['d_noise15']) for x in rr])
        put(f'Noise-dose sanity: d(noise15)>d(noise25) for {int((n15>n25+1e-6).sum())}/{len(rr)} samples')
        for metric in ['pesq','stoi','si_sdr']:
            a=f'router_{metric}'; b=f'off_{metric}'
            if a in rr[0] and b in rr[0]:
                gain=np.array([float(x[a])-float(x[b]) for x in rr])
                if np.std(gain)<1e-12 or np.std(delta)<1e-12:
                    rho=np.nan
                else:
                    rho=float(spearmanr(delta, gain).statistic)
                put(f'Spearman(Delta feature distance, Delta {metric}): rho={rho:+.4f} (more negative ideally), n={len(rr)}')
        put('Interpretation reminder: speaker verification embeddings may intentionally suppress noise/quality cues.')
    return '\n'.join(lines)+'\n'


def main():
    args=cli()
    root=args.root.expanduser().resolve()
    out=(args.out_dir or (root/'feature_stageA'/args.model_type)).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (root/'manifest.csv').open(newline='',encoding='utf-8') as f:
        manifest=list(csv.DictReader(f))[:args.count]
    if not manifest:
        raise RuntimeError('manifest.csv is empty')
    metrics={}
    if (root/'paired_metrics.csv').is_file():
        with (root/'paired_metrics.csv').open(newline='',encoding='utf-8') as f:
            metrics={r['filename']:r for r in csv.DictReader(f)}
    print(f'[STAGE-A] encoder={args.model_type}, path={args.model_path}, device={args.device}',flush=True)
    print(f'[STAGE-A] n={len(manifest)}, max_plm_length={args.max_plm_length}, root={root}',flush=True)
    model=load_encoder(args)
    rows=[]
    for i,m in enumerate(manifest):
        name=m['filename']
        files={key:root/key/name for key in ('clean','off','router')}
        for key,path in files.items():
            if not path.is_file():
                raise FileNotFoundError(f'Missing {key} file for {name}: {path}')
        raw={key:clean_audio(path) for key,path in files.items()}
        length=min(args.max_plm_length, *[len(v) for v in raw.values()])
        if length<1600:
            raise RuntimeError(f'Audio too short (<100ms) {name}: {length}')
        clean=raw['clean'][:length]
        uid=int.from_bytes(hashlib.sha256((str(args.seed)+'|'+name).encode('utf-8')).digest()[:8],byteorder='little')
        noise=np.random.default_rng(uid).standard_normal(length).astype(np.float32)
        noise/=float(np.sqrt(np.mean(noise.astype(np.float64)**2))+1e-12)
        variants={k:v[:length] for k,v in raw.items()}
        for key in ('noise25','noise15','lowpass3200','dropout50ms'):
            variants[key]=perturb(clean,key,noise)
        with torch.no_grad():
            c=embedding_dict(model,normalize_like_training(variants['clean'],args.device),args.model_type)
            found={}
            for key in FEATURES:
                feats=embedding_dict(model,normalize_like_training(variants[key],args.device),args.model_type)
                if set(feats)!=set(c):
                    raise RuntimeError('Embedding branch mismatch')
                found[key]={b:similarity_distance(feats[b],c[b]) for b in c}
        for branch in c:
            row={'filename':name,'uttid':m.get('uttid',''),'feature':branch,'clip_samples':length}
            for key in FEATURES:
                row['d_'+key]=found[key][branch]
            row['delta_router_off']=row['d_router']-row['d_off']
            if name in metrics:
                for k in ('off_pesq','router_pesq','off_stoi','router_stoi','off_si_sdr','router_si_sdr'):
                    if k in metrics[name]:row[k]=metrics[name][k]
            rows.append(row)
        if (i+1)%4==0 or i+1==len(manifest):
            print(f'[STAGE-A] {i+1}/{len(manifest)} completed ({name})', flush=True)
    csv_path=out/'feature_distances.csv'
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    report=summarize(rows,args.bootstraps,args.seed)
    report_path=out/'feature_report.txt'
    report_path.write_text(report,encoding='utf-8')
    print(report,flush=True)
    print(f'[DONE] CSV={csv_path}\n[DONE] REPORT={report_path}',flush=True)


if __name__=='__main__':
    main()
