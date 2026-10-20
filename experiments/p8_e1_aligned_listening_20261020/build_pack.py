from pathlib import Path
import csv, json, hashlib
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent
P8=ROOT.parent/'p8_frontend_backend_audit_20261020'
PROBE=ROOT.parent/'p8_e1_candidate_ranking_20261018'
OUT=ROOT/'clips'; OUT.mkdir(exist_ok=True)
# New windows: none overlap the four windows in p8_counterexample_listening_pack_20261020.
CANDS=[
 {'fileid':'4812','index':2,'start_s':17.0,'end_s':17.5},
 {'fileid':'4812','index':2,'start_s':14.5,'end_s':15.0},
 {'fileid':'10720','index':4,'start_s':9.0,'end_s':9.5},
 {'fileid':'6339','index':1,'start_s':10.5,'end_s':11.0},
]
ARMS=['noisy','clean','y_pre','off','router']

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def read(p):
 x,fs=sf.read(str(p),always_2d=False)
 if x.ndim>1:x=x.mean(1)
 return x.astype(np.float64),fs
def lag_corr(a,b,maxlag=1600):
 # same-length arrays; maximize normalized correlation around zero
 a=a-a.mean(); b=b-b.mean(); den=np.linalg.norm(a)*np.linalg.norm(b)
 vals=[]
 for lag in range(-maxlag,maxlag+1):
  if lag<0: x,y=a[-lag:],b[:len(a)+lag]
  elif lag>0: x,y=a[:-lag],b[lag:]
  else:x,y=a,b
  vals.append((float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-15)),lag))
 best=max(vals)
 z=vals[maxlag][0]
 return z,best[0],best[1]
manifest=list(csv.DictReader((P8/'manifest.csv').open()))
rank=list(csv.DictReader((PROBE/'candidate_windows.csv').open()))
records=[]; scan=[]
for j,c in enumerate(CANDS):
 rec=manifest[c['index']]; base=Path(rec['noisy_basename']).name
 paths={'noisy':Path(rec['noisy_path']),'clean':Path(rec['clean_path'])}
 for arm,sub in [('y_pre','y_pre'),('off','off'),('router','router')]: paths[arm]=P8/'outputs'/sub/f"{c['index']:02d}_{base}"
 # Read all, verify dimensions and derive crop with 0.75 s context.
 waves={}; srset=set(); lengths={}
 for arm,p in paths.items():
  x,fs=read(p); waves[arm]=x; srset.add(fs); lengths[arm]=len(x)
 if srset != {16000} or len(set(lengths.values()))!=1: raise RuntimeError((c,lengths,srset))
 sr=16000; target_a=round(c['start_s']*sr); target_b=round(c['end_s']*sr)
 ctx_a=max(0,target_a-round(.75*sr)); ctx_b=min(len(waves['clean']),target_b+round(.75*sr))
 stem=f"{j:02d}_fileid_{c['fileid']}_{c['start_s']:.2f}-{c['end_s']:.2f}s_context"
 for arm in ARMS: sf.write(str(OUT/f'{stem}_{arm}.wav'),waves[arm][ctx_a:ctx_b],sr,subtype='PCM_16')
 zc,mc,lag=lag_corr(waves['noisy'][ctx_a:ctx_b],waves['clean'][ctx_a:ctx_b])
 # alignment scan for every arm against clean on exact context
 arm_scan={}
 for arm in ARMS:
  z,m,l=lag_corr(waves[arm][ctx_a:ctx_b],waves['clean'][ctx_a:ctx_b])
  arm_scan[arm]={'zero_lag_corr':z,'max_corr':m,'best_lag_samples':l,'best_lag_ms':1000*l/sr}
 # candidate metrics from existing read-only table
 rr=[r for r in rank if r['fileid']==c['fileid'] and abs(float(r['start_s'])-c['start_s'])<.01]
 records.append({**c,'context_start_s':ctx_a/sr,'context_end_s':ctx_b/sr,'target_offset_in_clip_s':(target_a-ctx_a)/sr,'duration_s':(ctx_b-ctx_a)/sr,'sample_rate_hz':sr,'samples':ctx_b-ctx_a,'source_paths':{k:str(v) for k,v in paths.items()},'source_sha256':{k:sha(v) for k,v in paths.items()},'lengths_samples':lengths,'alignment_scan':arm_scan,'candidate_metrics':rr[0] if rr else {}})
 scan.append({'index':c['index'],'fileid':c['fileid'],'start_s':c['start_s'],'end_s':c['end_s'],'context_start_s':ctx_a/sr,'context_end_s':ctx_b/sr,'sample_rate_hz':sr,'lengths_samples':lengths,'alignment':arm_scan})
(ROOT/'metrics.json').write_text(json.dumps({'n_windows':4,'n_arms':5,'all_source_30s':all(v==480000 for r in records for v in r['lengths_samples'].values()),'records':records},indent=2)+'\n')
(ROOT/'alignment_scan.json').write_text(json.dumps(scan,indent=2)+'\n')
(ROOT/'config.yaml').write_text('''experiment_id: p8_e1_aligned_listening_20261020\npurpose: alignment_repair_listening_pack\nsource_manifest: ../p8_frontend_backend_audit_20261020/manifest.csv\nsource_outputs: ../p8_frontend_backend_audit_20261020/outputs\nsource_dataset: /data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5\nsample_rate_hz: 16000\nsource_duration_s: 30.0\ncentral_usable_range_s: [9.0, 21.0]\ntarget_window_duration_s: 0.5\ncontext_each_side_s: 0.75\nwindows: 4\narms: [noisy, clean, y_pre, off, router]\nselection_rule: >-10 dB local oracle SNR when available; otherwise highest noisy-clean local correlation among unused central-12s candidates; exclude all four prior counterexample-pack windows.\nprocessing: read-only source scan; crop only into new clips; no inference, training, DNSMOS, or source overwrite\n''')
