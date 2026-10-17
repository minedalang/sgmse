#!/usr/bin/env python3
"""In-memory noise-residual scaling probe on locked fresh-CV; never writes audio."""
from __future__ import annotations
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

ROOT=Path('/data/coding/sgmse')
EXP=Path(__file__).resolve().parent
MANIFEST=ROOT/'experiments/tr0_fresh_cv_manifest_20261018/manifest.csv'
CKPT=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
MS=[0.5,0.75,1.0,1.25,1.5]
SEED=2026
sys.path.insert(0,str(ROOT))

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def sisdr(ref,est):
 ref=np.asarray(ref,dtype=np.float64);est=np.asarray(est,dtype=np.float64)
 n=min(len(ref),len(est));ref=ref[:n];est=est[:n]
 ref=ref-ref.mean();est=est-est.mean();a=np.dot(est,ref)/(np.dot(ref,ref)+1e-12);t=a*ref
 return float(10*np.log10((np.dot(t,t)+1e-12)/(np.dot(est-t,est-t)+1e-12)))

def align(a,b):
 n=min(len(a),len(b)); return a[:n],b[:n]

def quality(ref,out):
 ref,out=align(ref,out)
 out=np.clip(out,-1,1)
 return {'pesq_wb':float(pesq(16000,ref.astype(np.float32),out.astype(np.float32),'wb')),
         'stoi':float(stoi(ref.astype(np.float32),out.astype(np.float32),16000,extended=False)),
         'si_sdr_db':sisdr(ref,out)}

def load_model(device):
 from sgmse.joint_data_module import JointWaveDataModule
 from sgmse.joint_model import JointSemambaSgmseModel
 m=JointSemambaSgmseModel.load_from_checkpoint(str(CKPT),map_location='cpu',strict=True,
     data_module_cls=JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,
     weights_only=False)
 m.eval(no_ema=False).to(device)
 m.frontend.eval()
 for p in m.parameters():p.requires_grad_(False)
 return m

def main():
 torch.set_num_threads(1); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 with MANIFEST.open(newline='') as f: allrows=list(csv.DictReader(f))
 rows=[]; speakers=set()
 for r in allrows:
  if r['speaker'] not in speakers:
   rows.append(r); speakers.add(r['speaker'])
  if len(rows)==4:break
 if len(rows)!=4:raise RuntimeError('Could not select four distinct speakers')
 # Hash source files before inference and verify all inputs are aligned 16 kHz.
 inputs=[]
 for r in rows:
  c,s1=sf.read(r['clean'],dtype='float32'); y,s2=sf.read(r['noisy'],dtype='float32')
  if c.ndim>1:c=c.mean(axis=1)
  if y.ndim>1:y=y.mean(axis=1)
  if s1!=16000 or s2!=16000:raise RuntimeError(f"sample rate mismatch for {r['uttid']}: {s1}/{s2}")
  c,y=align(c,y)
  if len(c)<16000:raise RuntimeError(f"too short {r['uttid']}: {len(c)}")
  inputs.append((r,c,y))
 m=load_model(device)
 records=[]; start=time.perf_counter()
 for i,(r,c,y) in enumerate(inputs):
  clean=torch.from_numpy(c.copy()).to(device)[None,None,:]
  noisy=torch.from_numpy(y.copy()).to(device)[None,None,:]
  residual=noisy-clean
  for mult in MS:
   ym=clean+float(mult)*residual
   # Match the production inference frontend path: peak-normalize the mixture,
   # run frozen StoRM, restore amplitude, then align frontend output to mixture.
   norm=ym.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)
   with torch.inference_mode():
    pre=m._frontend_forward(ym/norm)*norm
   cur,ref_len=pre.shape[-1],ym.shape[-1]
   if cur>ref_len:
    st=(cur-ref_len)//2;pre=pre[...,st:st+ref_len]
   elif cur<ref_len:
    d=ref_len-cur;pre=torch.nn.functional.pad(pre,(d//2,d-d//2))
   pre_np=pre[0,0].detach().float().cpu().numpy()
   frontend_q=quality(c,pre_np)
   # Router-off: default rrsd=False; preserve the StoRM frontend and OUVE PC sampler.
   torch.manual_seed(SEED)
   if device.type=='cuda':torch.cuda.manual_seed_all(SEED)
   with torch.inference_mode():
    enhanced=m.enhance(pre.squeeze(0),y_cond=ym.squeeze(0),sampler_type='pc',predictor='reverse_diffusion',
                       corrector='none',N=30,snr=0.5,rrsd=False)
   out=np.asarray(enhanced.detach().float().cpu() if torch.is_tensor(enhanced) else enhanced,dtype=np.float32).reshape(-1)
   q=quality(c,out)
   records.append({'filename':r['filename'],'uttid':r['uttid'],'speaker':r['speaker'],'fileid':r['fileid'],
      'M':float(mult),'input_snr_db':float(10*np.log10((np.mean(c.astype(np.float64)**2)+1e-20)/(np.mean((float(mult)*(y-c)).astype(np.float64)**2)+1e-20))),
      'frontend_only':frontend_q,'router_off_pc':q,'n_samples':int(len(c))})
  print(f"[{i+1}/4] {r['uttid']} speaker={r['speaker']}",flush=True)
 # M-axis second finite differences (spacing 0.25); SI-SDR kink=max adjacent slope change.
 for r in rows:
  rr=[x for x in records if x['uttid']==r['uttid']]
  rr.sort(key=lambda x:x['M'])
  for arm in ['frontend_only','router_off_pc']:
   vals=np.array([x[arm]['si_sdr_db'] for x in rr],float)
   slopes=np.diff(vals)/0.25
   second=np.diff(vals,2)/(0.25**2)
   summary={'si_sdr_values_db':vals.tolist(),'first_slopes_db_per_M':slopes.tolist(),
      'second_differences_db_per_M2':second.tolist(),'max_abs_slope_change_db_per_M':float(np.max(np.abs(np.diff(slopes)))) if len(slopes)>1 else 0.,
      'max_abs_second_difference_db_per_M2':float(np.max(np.abs(second))) if len(second) else 0.}
   for x in rr:x.setdefault('curvature',{})[arm]=summary
 # CSV: one row per utterance x M, flatten metrics and curvature stats.
 fields=['uttid','speaker','fileid','filename','M','input_snr_db','n_samples']
 for arm in ['frontend_only','router_off_pc']:
  fields += [f'{arm}_{k}' for k in ['pesq_wb','stoi','si_sdr_db','max_abs_slope_change_db_per_M','max_abs_second_difference_db_per_M2']]
 with (EXP/'per_utterance_per_M.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for x in records:
   row={k:x.get(k) for k in ['uttid','speaker','fileid','filename','M','input_snr_db','n_samples']}
   for arm in ['frontend_only','router_off_pc']:
    row.update({f'{arm}_{k}':x[arm][k] for k in ['pesq_wb','stoi','si_sdr_db']})
    row[f'{arm}_max_abs_slope_change_db_per_M']=x['curvature'][arm]['max_abs_slope_change_db_per_M']
    row[f'{arm}_max_abs_second_difference_db_per_M2']=x['curvature'][arm]['max_abs_second_difference_db_per_M2']
   w.writerow(row)
 with (EXP/'detailed_metrics.json').open('w') as f:json.dump(records,f,indent=2)
 # Per-M aggregate and summaries
 agg=[]
 for mult in MS:
  subset=[x for x in records if x['M']==mult]
  e={'M':float(mult),'mean_input_snr_db':float(np.mean([x['input_snr_db'] for x in subset])),'n':len(subset)}
  for arm in ['frontend_only','router_off_pc']:
   for key in ['pesq_wb','stoi','si_sdr_db']:
    e[f'{arm}_{key}_mean']=float(np.mean([x[arm][key] for x in subset]))
  agg.append(e)
 kink={}
 for arm in ['frontend_only','router_off_pc']:
  kink[arm]={'per_utterance_max_abs_slope_change_db_per_M':[x['curvature'][arm]['max_abs_slope_change_db_per_M'] for x in records if x['M']==1.0],
   'mean_max_abs_slope_change_db_per_M':float(np.mean([x['curvature'][arm]['max_abs_slope_change_db_per_M'] for x in records if x['M']==1.0])),
   'per_utterance_max_abs_second_difference_db_per_M2':[x['curvature'][arm]['max_abs_second_difference_db_per_M2'] for x in records if x['M']==1.0],
   'mean_max_abs_second_difference_db_per_M2':float(np.mean([x['curvature'][arm]['max_abs_second_difference_db_per_M2'] for x in records if x['M']==1.0]))}
 metrics={'experiment_id':'storm_noise_power_mismatch_probe_20261020','status':'completed','n_utterances':len(rows),'n_speakers':len(speakers),'M_values':MS,'checkpoint_sha256':sha256(CKPT),'manifest_sha256':sha256(MANIFEST),'inputs':[{'uttid':r['uttid'],'speaker':r['speaker'],'clean':r['clean'],'noisy':r['noisy'],'clean_sha256':sha256(Path(r['clean'])),'noisy_sha256':sha256(Path(r['noisy']))} for r,_,_ in inputs],
   'device':str(device),'seed':SEED,'sampler':'PC reverse_diffusion, N=30, snr=0.5, corrector=none','router':'off (rrsd=False)','frontend':'checkpoint StoRM, frozen; per-M peak normalization and amplitude restoration','normalization_note':'M scales only the noisy-clean residual; global peak normalization cannot cancel the changed clean/noise ratio. Measured input SNR is recorded per utterance/M.','aggregates_by_M':agg,'si_sdr_kink_summary':kink,'runtime_seconds':time.perf_counter()-start,'wav_files_written':False,'dataset_modified':False,'checkpoint_modified':False}
 with (EXP/'metrics.json').open('w') as f:json.dump(metrics,f,indent=2)
 print(json.dumps({'aggregates_by_M':agg,'kink':kink,'runtime_seconds':metrics['runtime_seconds']},indent=2))
if __name__=='__main__':main()
