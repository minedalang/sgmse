#!/usr/bin/env python3
"""Paired PESQ/STOI/SI-SDR/DNSMOS guardrails with speaker-cluster intervals."""
import csv
import hashlib
import importlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

ROOT=Path('/data/coding/sgmse')
EXP=ROOT/'experiments/speaker_headroom_replication_64_20261017'
LOG=ROOT/'experiment_logs/speaker_headroom_replication_64_20261017'
MANIFEST=EXP/'manifest.csv'
LOCKED='c04657fa21a33c7f2dc749514cec899767101ba53f46222e9f4ab8577e2f4492'
SR=16000
BOOT=5000
SEED=20261017
DEVICE_INFO={}
SESSION_CACHE={}


def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def load(p):
 x,sr=sf.read(p,dtype='float32',always_2d=False)
 if x.ndim==2: x=x.mean(axis=-1)
 if sr!=SR or x.ndim!=1 or not np.isfinite(x).all(): raise ValueError(f'invalid waveform {p}: {sr} {x.shape}')
 return np.ascontiguousarray(x)

def si_sdr(ref,est):
 ref=np.asarray(ref,dtype=np.float64); est=np.asarray(est,dtype=np.float64)
 if ref.shape!=est.shape: raise ValueError((ref.shape,est.shape))
 ref=ref-ref.mean(); est=est-est.mean()
 alpha=np.dot(est,ref)/(np.dot(ref,ref)+1e-12)
 target=alpha*ref; err=est-target
 return float(10*np.log10((np.dot(target,target)+1e-12)/(np.dot(err,err)+1e-12)))

def dns_sessions(device):
 module=importlib.import_module('torchmetrics.functional.audio.dnsmos')
 loader=getattr(module,'_cached_load_session',None)
 if loader is None: raise RuntimeError('TorchMetrics lacks cached DNSMOS session loader')
 paths={'sig_bak_ovr':f'{module.DNSMOS_DIR}/DNSMOS/sig_bak_ovr.onnx',
        'model_v8':f'{module.DNSMOS_DIR}/DNSMOS/model_v8.onnx'}
 return {name:loader(path,device,1).get_providers() for name,path in paths.items()}

def choose_dns_device():
 requested='auto'; target=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 try:
  if target.type=='cuda':
   import onnxruntime as ort
   if 'CUDAExecutionProvider' not in ort.get_available_providers(): raise RuntimeError('CUDA EP absent')
   providers=dns_sessions(target)
   if any('CUDAExecutionProvider' not in v for v in providers.values()): raise RuntimeError(f'DNS sessions not CUDA: {providers}')
  else: providers=dns_sessions(target)
 except Exception as exc:
  warnings.warn(f'DNSMOS CUDA unavailable; CPU fallback: {exc}',RuntimeWarning)
  target=torch.device('cpu'); providers=dns_sessions(target)
 DEVICE_INFO.update({'requested_device':requested,'device':str(target),
   'execution_provider':'CUDAExecutionProvider' if all('CUDAExecutionProvider' in x for x in providers.values()) else 'CPUExecutionProvider',
   'onnx_session_providers':providers,'torch':torch.__version__})
 return str(target)

def dnsmos(x,device):
 a=np.asarray(x,dtype=np.float32)
 peak=float(np.max(np.abs(a)))+1e-12
 if peak>1: a=a/peak
 val=deep_noise_suppression_mean_opinion_score(torch.from_numpy(a),fs=SR,personalized=False,
     device=str(device),cache_session=True,num_threads=1)
 return val.detach().cpu().numpy().reshape(4).astype(float)

def main():
 started=time.time()
 if sha256(MANIFEST)!=LOCKED: raise RuntimeError('manifest hash changed')
 rows=list(csv.DictReader(MANIFEST.open(newline='',encoding='utf-8')))
 if len(rows)!=128 or len({r['speaker'] for r in rows})!=64: raise RuntimeError('manifest cardinality mismatch')
 try:
  device=choose_dns_device()
  dns_available=True
 except Exception as exc:
  device='unavailable'; dns_available=False
  DEVICE_INFO.update({'requested_device':'auto','device':'unavailable','execution_provider':'unavailable','error':repr(exc)})
 records=[]
 for k,r in enumerate(rows,1):
  ref=load(Path(r['clean_path']))
  clean_hash=sha256(Path(r['clean_path']))
  if clean_hash!=r['clean_sha256']: raise RuntimeError(f'clean hash changed {r["filename"]}')
  entry={'filename':r['filename'],'uttid':r['uttid'],'speaker':r['speaker']}
  for arm in ('off','router'):
   p=LOG/arm/r['filename']; enhanced=load(p)
   if len(ref)!=len(enhanced): raise RuntimeError(f'alignment mismatch {arm} {r["filename"]}')
   entry[f'{arm}_pesq_wb']=float(pesq(SR,ref,enhanced,'wb'))
   entry[f'{arm}_stoi']=float(stoi(ref,enhanced,SR,extended=False))
   entry[f'{arm}_si_sdr_db']=si_sdr(ref,enhanced)
   if dns_available:
    try: d=dnsmos(enhanced,device)
    except Exception as exc:
     dns_available=False; DEVICE_INFO['error']=repr(exc); d=[float('nan')]*4
   else: d=[float('nan')]*4
   for label,value in zip(('p808','sig','bak','ovrl'),d): entry[f'{arm}_dnsmos_{label}']=float(value)
  records.append(entry)
  if k%16==0: print(f'evaluated {k}/128 utterances',flush=True)
 labels=[f'{a}_{m}' for a in ('off','router') for m in ('pesq_wb','stoi','si_sdr_db','dnsmos_p808','dnsmos_sig','dnsmos_bak','dnsmos_ovrl')]
 speakers=sorted({r['speaker'] for r in rows})
 bysp={s:[i for i,r in enumerate(records) if r['speaker']==s] for s in speakers}
 if any(len(v)!=2 for v in bysp.values()): raise RuntimeError('expected exactly two utterances per speaker')
 rng=np.random.default_rng(SEED); draws=rng.integers(0,len(speakers),size=(BOOT,len(speakers)))
 counts=np.zeros((BOOT,len(speakers)),dtype=np.int16)
 for b in range(BOOT): counts[b]=np.bincount(draws[b],minlength=len(speakers))
 metric_results={}
 for metric in ('pesq_wb','stoi','si_sdr_db','dnsmos_p808','dnsmos_sig','dnsmos_bak','dnsmos_ovrl'):
  arm_stats={}; bootmeans={}
  for arm in ('off','router'):
   vals=np.asarray([r[f'{arm}_{metric}'] for r in records],dtype=np.float64)
   spvals=np.asarray([vals[bysp[s]].mean() for s in speakers])
   boot=(counts*spvals[None,:]).sum(axis=1)/counts.sum(axis=1)
   bootmeans[arm]=boot
   arm_stats[arm]={'mean':float(vals.mean()),'std':float(vals.std(ddof=1)),
                   'speaker_cluster_bootstrap_95ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
  delta=np.asarray([r[f'router_{metric}']-r[f'off_{metric}'] for r in records])
  speaker_delta=np.asarray([delta[bysp[s]].mean() for s in speakers])
  boot=(counts*speaker_delta[None,:]).sum(axis=1)/counts.sum(axis=1)
  metric_results[metric]={'per_arm':arm_stats,'router_minus_off':{
    'mean_delta':float(delta.mean()),'speaker_cluster_bootstrap_95ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}}
 with (EXP/'quality_per_utterance.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
 result={'experiment_id':'speaker_headroom_replication_64_20261017_quality','n':len(records),
  'n_speakers':len(speakers),'sample_rate_hz':SR,'metrics':metric_results,
  'dnsmos_runtime':DEVICE_INFO,'bootstrap':{'replicates':BOOT,'seed':SEED,'unit':'speaker cluster'},
  'runtime_seconds':time.time()-started,'manifest_sha256':LOCKED,'training':False}
 (EXP/'quality_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
