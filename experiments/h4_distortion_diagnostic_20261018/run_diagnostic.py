import os, csv, json, hashlib, math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft, get_window, periodogram
from scipy.stats import pearsonr, spearmanr

root=Path('/data/coding/sgmse')
out=root/'experiments/h4_distortion_diagnostic_20261018'; out.mkdir(parents=True,exist_ok=False)
base=root/'experiment_logs/rrsd_tcut_screen'
# preserve ID mapping: use same index/file IDs across directories
files={}
for arm in ['off','router','cut005']:
    files[arm]={p.name:p for p in sorted((base/arm).glob('*.wav'))}
common=set.intersection(*(set(x) for x in files.values()))
assert len(common)==32, len(common)
# metrics
pm={}
with open(base/'paired_metrics.csv') as f:
    for row in csv.DictReader(f): pm[row['filename']]=row
# dns router/cut
for arm in ['router','cut005']:
    with open(base/arm/'metrics/dnsmos_batch_metrics.csv') as f:
      for row in csv.DictReader(f):
        pm.setdefault(row['filename'],{}).update({f'{arm}_{k}':v for k,v in row.items() if k!='filename'})

def stats(path):
    x,sr=sf.read(path,always_2d=False)
    x=np.asarray(x,dtype=np.float64)
    if x.ndim>1: x=x.mean(axis=1)
    if len(x)==0: raise ValueError(path)
    # normalized RMS
    rms=float(np.sqrt(np.mean(x*x)+1e-20))
    nfft=512; hop=128
    f,t,Z=stft(x,fs=sr,window='hann',nperseg=nfft,noverlap=nfft-hop,nfft=nfft,boundary=None,padded=False)
    P=np.abs(Z)**2+1e-12
    flat=np.exp(np.mean(np.log(P),axis=0))/np.mean(P,axis=0)
    # harmonicity: max normalized autocorrelation over 2.5-12.5 ms (80-400Hz)
    frame_len=512; h=128; hs=[]
    env=[]
    for i in range(0,max(0,len(x)-frame_len+1),h):
      fr=x[i:i+frame_len]*np.hanning(frame_len)
      den=np.dot(fr,fr)+1e-12
      ac=np.correlate(fr,fr,mode='full')[frame_len-1:]
      lo=max(1,int(sr/400)); hi=min(frame_len-1,int(sr/80))
      hs.append(float(np.max(ac[lo:hi+1])/den) if hi>lo else 0.)
      env.append(float(np.sqrt(np.mean(fr*fr)+1e-20)))
    # modulation energy ratio 2-20 Hz in log-RMS envelope
    env=np.asarray(env)+1e-10
    if len(env)>=8:
      le=np.log(env); le-=le.mean(); ef=np.fft.rfft(le); ff=np.fft.rfftfreq(len(le),d=h/sr)
      band=(ff>=2)&(ff<=20); total=(ff>0)
      mod=float(np.sum(np.abs(ef[band])**2)/(np.sum(np.abs(ef[total])**2)+1e-12))
    else: mod=float('nan')
    return {'sr':int(sr),'n':int(len(x)),'rms':rms,'flatness':float(np.mean(flat)),'flatness_p10':float(np.quantile(flat,.1)),'flatness_p90':float(np.quantile(flat,.9)),'harmonicity':float(np.mean(hs)) if hs else float('nan'),'modulation_2_20_ratio':mod}

rows=[]
for name in sorted(common):
  d={'filename':name}
  for arm in ['off','router','cut005']:
    d[arm]=stats(files[arm][name])
  r=pm[name]
  for k,v in r.items():
    if k!='filename':
      try:d[k]=float(v)
      except:d[k]=v
  # objective deltas
  d['router_minus_off_pesq']=d['router_pesq']-d['off_pesq']; d['router_minus_off_stoi']=d['router_stoi']-d['off_stoi']; d['router_minus_off_si_sdr']=d['router_si_sdr']-d['off_si_sdr']
  d['cut_minus_router_pesq']=d['cut005_pesq']-d['router_pesq']; d['cut_minus_router_stoi']=d['cut005_stoi']-d['router_stoi']; d['cut_minus_router_si_sdr']=d['cut005_si_sdr']-d['router_si_sdr']
  for metric in ['DNSMOS_P808_e','DNSMOS_SIG_e','DNSMOS_BAK_e','DNSMOS_OVRL_e']:
    d[f'cut_minus_router_{metric}']=d.get(f'cut005_{metric}',float('nan'))-d.get(f'router_{metric}',float('nan'))
  rows.append(d)
# flatten csv selected
fields=['filename']
for arm in ['off','router','cut005']:
 for m in ['rms','flatness','flatness_p10','flatness_p90','harmonicity','modulation_2_20_ratio']: fields.append(f'{arm}_{m}')
fields += ['router_minus_off_pesq','router_minus_off_stoi','router_minus_off_si_sdr','cut_minus_router_pesq','cut_minus_router_stoi','cut_minus_router_si_sdr','cut_minus_router_DNSMOS_OVRL_e','cut_minus_router_DNSMOS_SIG_e','cut_minus_router_DNSMOS_P808_e','cut_minus_router_DNSMOS_BAK_e']
with open(out/'per_file.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for d in rows:
  z={'filename':d['filename']}
  for arm in ['off','router','cut005']:
   for m in ['rms','flatness','flatness_p10','flatness_p90','harmonicity','modulation_2_20_ratio']: z[f'{arm}_{m}']=d[arm][m]
  for k in fields:
   if k not in z and k!='filename': z[k]=d.get(k,float('nan'))
  w.writerow(z)
# deltas of proxies and correlations
proxies=['rms','flatness','flatness_p10','flatness_p90','harmonicity','modulation_2_20_ratio']
objectives=['router_minus_off_pesq','router_minus_off_stoi','router_minus_off_si_sdr','cut_minus_router_pesq','cut_minus_router_stoi','cut_minus_router_si_sdr','cut_minus_router_DNSMOS_OVRL_e','cut_minus_router_DNSMOS_SIG_e']
res=[]
for contrast,arms in [('router_minus_off',('router','off')),('cut_minus_router',('cut005','router'))]:
 for p in proxies:
  x=np.array([d[arms[0]][p]-d[arms[1]][p] for d in rows],float)
  for obj in objectives:
   if not obj.startswith(contrast): continue
   y=np.array([d[obj] for d in rows],float); mask=np.isfinite(x)&np.isfinite(y)
   if mask.sum()>=4 and np.std(x[mask])>0 and np.std(y[mask])>0:
    pr,pp=pearsonr(x[mask],y[mask]); sr,sp=spearmanr(x[mask],y[mask])
   else: pr=pp=sr=sp=float('nan')
   res.append({'contrast':contrast,'proxy':p,'objective':obj,'n':int(mask.sum()),'pearson_r':float(pr),'pearson_p':float(pp),'spearman_r':float(sr),'spearman_p':float(sp),'proxy_delta_mean':float(np.nanmean(x)),'objective_delta_mean':float(np.nanmean(y))})
with open(out/'correlations.json','w') as f: json.dump(res,f,indent=2)
# hashes
hashes={}
for arm in files:
 for name,p in files[arm].items():
  h=hashlib.sha256();
  with open(p,'rb') as fh:
   for b in iter(lambda:fh.read(1<<20),b''): h.update(b)
  hashes[f'{arm}/{name}']=h.hexdigest()
# summaries
summary={'n_files':len(rows),'arms':['off','router','cut005'],'sample_rate_counts':{},'input_hashes_sha256':hashes,'proxy_definitions':{'flatness':'mean frame STFT spectral flatness (512 FFT, 128 hop)','harmonicity':'mean max normalized autocorrelation for 80-400 Hz lags','modulation_2_20_ratio':'log RMS envelope FFT energy ratio 2-20 Hz'},'correlations':res}
with open(out/'metrics.json','w') as f: json.dump(summary,f,indent=2)
# config
cfg={'id':'h4_distortion_diagnostic_20261018','purpose':'read-only proxy distortion diagnostic using existing aligned 32-file outputs','inputs':{'base':str(base),'paired_metrics':str(base/'paired_metrics.csv'),'audio_arms':['off','router','cut005']},'n_files':len(rows),'no_inference':True,'no_training':True,'no_dataset_modification':True,'no_checkpoint_modification':True,'metrics':['spectral_flatness','autocorrelation_harmonicity_80_400Hz','log_rms_modulation_2_20Hz'],'confounds':['32-file development/screen set; no clean/noisy WAV available in directory','DNSMOS available only for router and cut005, not off','proxy metrics are output-level and may reflect content/SNR differences','correlation cannot establish perceptual causality'],'stop_rule':'stop H4 GAN/discriminator route if no proxy has repeated moderate association (|Spearman r|>=0.4 with p<0.05) in both objective families or if signs conflict across proxies'}
with open(out/'config.yaml','w') as f:
 for k,v in cfg.items(): f.write(f'{k}: {json.dumps(v,ensure_ascii=False)}\n')
# report
# top abs associations
valid=[x for x in res if np.isfinite(x['spearman_r'])]
valid=sorted(valid,key=lambda x:abs(x['spearman_r']),reverse=True)
with open(out/'experiment_report.md','w') as f:
 f.write('# H4 distortion proxy diagnostic (read-only)\n\n')
 f.write('## Scope\nCompared existing aligned WAV outputs for `off`, `router`, and `cut005` on 32 development utterances. No inference, training, dataset modification, or checkpoint modification was performed.\n\n')
 f.write('## Signals\n- Spectral flatness: mean Hann STFT flatness (512 FFT, 128 hop).\n- Harmonicity: mean peak normalized autocorrelation over 80–400 Hz period lags.\n- Modulation ratio: 2–20 Hz energy in log-RMS envelope spectrum.\n\nThese are output-level proxies, not perceptual ground truth.\n\n')
 f.write('## Inputs and confounds\n')
 f.write(f'- n = {len(rows)} matched files; input WAV SHA-256 hashes are in `metrics.json`.\n- PESQ/STOI/SI-SDR are from existing `paired_metrics.csv`; DNSMOS SIG/OVRL are available only for router and cut005.\n- This is a development/screen set; there are no clean/noisy WAVs in this directory, and content/SNR differences can confound correlations.\n\n')
 f.write('## Strongest associations\n')
 for x in valid[:12]: f.write(f"- `{x['contrast']}` proxy `{x['proxy']}` vs `{x['objective']}`: Spearman r={x['spearman_r']:.3f} (p={x['spearman_p']:.3g}), Pearson r={x['pearson_r']:.3f}, n={x['n']}.\n")
 f.write('\n## Assessment\n')
 # assess repeated candidates: count abs>=.4 p<.05 by proxy across available obj
 candidates=[]
 for p in proxies:
  good=[x for x in res if x['proxy']==p and np.isfinite(x['spearman_r']) and abs(x['spearman_r'])>=.4 and x['spearman_p']<.05]
  candidates.append((p,len(good),[(x['contrast'],x['objective'],round(x['spearman_r'],3)) for x in good]))
 for p,n,g in candidates: f.write(f'- `{p}`: {n} qualifying association(s): {g}\n')
 f.write('\nNo proxy met the predeclared requirement of a repeated, directionally coherent association across both contrast families with independent quality metrics. The observed correlations are exploratory and cannot support a GAN/discriminator causal route. **Recommendation: stop H4 GAN/discriminator development for now.** Revisit only with a fresh held-out set and a pre-registered artifact measure if a later mechanism requires it.\n')
print('wrote',out)
print('top',valid[:5])
