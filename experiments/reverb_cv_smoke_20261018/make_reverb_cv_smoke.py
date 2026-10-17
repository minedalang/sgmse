from pathlib import Path
import csv, os, re, json, hashlib
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve

root=Path('/data/coding/sgmse/experiments/reverb_cv_smoke_20261018')
inputs=root/'inputs'; inputs.mkdir(parents=True,exist_ok=True)
for cond in ['identity','short','medium']:
    (inputs/cond/'noisy').mkdir(parents=True,exist_ok=True)
    (inputs/cond/'clean').mkdir(parents=True,exist_ok=True)
# fixed fresh CV selection, one utterance per speaker
selected=[
 ('reader_10714','reader_10714_fileid_6527','book_09513_chp_0014_reader_10714_37_0QePKnSWXkE-bj5HnISH8XE-pzfbyI1_2dQ_snr-6_fileid_6527.wav','6527'),
 ('reader_10717','reader_10717_fileid_4083','book_05847_chp_0051_reader_10717_0_E0YEaBwSOto-8jDRkg4bJwg-ehbaBiLUCD0_snr-14_fileid_4083.wav','4083'),
 ('reader_10777','reader_10777_fileid_6785','book_11030_chp_0029_reader_10777_6_DcnPT6AL8sM-door_Freesound_validated_219500_0-NwQIJbxgbRI_snr-6_fileid_6785.wav','6785'),
 ('reader_10785','reader_10785_fileid_3854','book_03227_chp_0097_reader_10785_2_UA8HFI79E-Y-1Xab-oXhabc-sKlCyIJBQ5s_snr-7_fileid_3854.wav','3854'),
]
base=Path('/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5')
rirs={
 'identity': np.array([1.0],dtype=np.float64),
 # deterministic causal FIRs; coefficients are fixed before evaluation
 'short': np.array([1.0,0.55,0.30,0.16,0.08,0.04,0.02],dtype=np.float64),
 'medium': np.array([1.0,0.72,0.55,0.42,0.31,0.23,0.17,0.125,0.09,0.065,0.047,0.034,0.024,0.017,0.012,0.0085,0.006],dtype=np.float64),
}
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def rms(x): return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
def conv_same(x,h):
 y=fftconvolve(x,h,mode='full')[:len(x)]
 return y.astype(np.float32)
manifest=[]
for sp,utt,basename,fid in selected:
 noisy=base/'noisy'/basename; clean=base/'clean'/f'clean_fileid_{fid}.wav'
 y,sr=sf.read(noisy,dtype='float32'); x,sr2=sf.read(clean,dtype='float32')
 assert sr==sr2==16000 and len(y)==len(x)==480000
 n=y-x
 target_ratio=rms(x)/(rms(n)+1e-12)
 source={'speaker':sp,'uttid':utt,'fileid':fid,'noisy_source':str(noisy),'clean_source':str(clean),'noisy_sha256':sha(noisy),'clean_sha256':sha(clean),'sr':sr,'samples':len(x),'source_rms':rms(x),'noise_rms':rms(n),'source_snr_db':20*np.log10(target_ratio)}
 for cond,h in rirs.items():
  if cond=='identity':
   xr=x.copy(); nr=n.copy(); yr=y.copy()
  else:
   xr=conv_same(x,h); nr0=conv_same(n,h)
   nr=nr0*(rms(xr)/(rms(nr0)+1e-12))/target_ratio
   yr=xr+nr
  peak=max(float(np.max(np.abs(xr))),float(np.max(np.abs(yr))),1e-12)
  scale=0.95/peak
  xr=xr*scale; yr=yr*scale
  sf.write(inputs/cond/'clean'/f'{utt}.wav',xr,sr)
  sf.write(inputs/cond/'noisy'/f'{utt}.wav',yr,sr)
  manifest.append({**source,'condition':cond,'rir':h.tolist(),'rir_energy':float(np.sum(h*h)),'clean_path':str(inputs/cond/'clean'/f'{utt}.wav'),'noisy_path':str(inputs/cond/'noisy'/f'{utt}.wav'),'clean_sha256':sha(inputs/cond/'clean'/f'{utt}.wav'),'noisy_sha256':sha(inputs/cond/'noisy'/f'{utt}.wav'),'constructed_snr_db':20*np.log10(rms(xr)/(rms(yr-xr)+1e-12))})
# wav.scp for provenance; not used by dir inference
for cond in rirs:
 with (inputs/cond/'manifest.csv').open('w',newline='') as f:
  fields=['speaker','uttid','fileid','condition','noisy_path','clean_path','source_snr_db','constructed_snr_db','noisy_sha256','clean_sha256']
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows([m for m in manifest if m['condition']==cond])
json.dump({'experiment_id':'reverb_cv_smoke_20261018','selection':selected,'rirs':{k:v.tolist() for k,v in rirs.items()},'manifest':manifest},open(root/'input_manifest.json','w'),indent=2)
with (root/'all_inputs.csv').open('w',newline='') as f:
 fields=['speaker','uttid','fileid','condition','noisy_path','clean_path','source_snr_db','constructed_snr_db','noisy_sha256','clean_sha256']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(manifest)
print('created',len(manifest),'files')
for cond in rirs:
 vals=[m['constructed_snr_db'] for m in manifest if m['condition']==cond]
 print(cond,'snr',vals)
