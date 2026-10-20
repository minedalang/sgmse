import csv, json, hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.stats import pearsonr, spearmanr
ROOT=Path('/data/coding/sgmse'); EXP=ROOT/'experiments/speaker_headroom_replication_64_20261017'
M=EXP/'manifest.csv'; Q=EXP/'quality_per_utterance.csv'; OUT=EXP/'quality_strata_metrics.json'
rows=list(csv.DictReader(Q.open())); manifest={r['filename']:r for r in csv.DictReader(M.open())}
for r in rows:
 m=manifest[r['filename']]; clean,_=sf.read(m['clean_path'],dtype='float64'); noisy,_=sf.read(m['noisy_path'],dtype='float64'); n=min(len(clean),len(noisy)); clean=clean[:n]; noisy=noisy[:n]; noise=noisy-clean
 r['snr_db']=10*np.log10((np.mean(clean**2)+1e-12)/(np.mean(noise**2)+1e-12))
 for metric in ('pesq_wb','stoi','si_sdr_db','dnsmos_sig','dnsmos_ovrl'):
  r['delta_'+metric]=float(r['router_'+metric])-float(r['off_'+metric])

def arr(k): return np.array([float(r[k]) for r in rows])
def corr(x,y):
 return {'pearson_r':float(pearsonr(x,y).statistic),'pearson_p':float(pearsonr(x,y).pvalue),'spearman_r':float(spearmanr(x,y).statistic),'spearman_p':float(spearmanr(x,y).pvalue)}
def summary(sub):
 if not sub: return {'n':0,'speakers':0}
 return {'n':len(sub),'speakers':len(set(r['speaker'] for r in sub)),**{k:float(np.mean([r[k] for r in sub])) for k in ['snr_db','delta_pesq_wb','delta_stoi','delta_si_sdr_db','delta_dnsmos_sig','delta_dnsmos_ovrl']},'ovrl_negative_fraction':float(np.mean([r['delta_dnsmos_ovrl']<0 for r in sub]))}
bins=[('<=-15',lambda x:x<=-15),('(-15,-10]',lambda x:-15<x<=-10),('(-10,-5]',lambda x:-10<x<=-5),('>-5',lambda x:x>-5)]
out={'experiment_id':'speaker_headroom_replication_64_20261017_quality_strata','source_quality_csv':str(Q),'analysis':'offline per-utterance deltas; no new inference','overall':summary(rows),'snr_strata':{name:summary([r for r in rows if pred(r['snr_db'])]) for name,pred in bins}}
for target in ['delta_dnsmos_ovrl','delta_dnsmos_sig','delta_pesq_wb','delta_stoi','delta_si_sdr_db']:
 x=arr('snr_db'); out.setdefault('correlations_snr',{})[target]=corr(x,arr(target))
for a,b in [('delta_dnsmos_ovrl','delta_si_sdr_db'),('delta_dnsmos_ovrl','delta_pesq_wb'),('delta_dnsmos_ovrl','delta_stoi'),('delta_dnsmos_sig','delta_si_sdr_db')]: out.setdefault('correlations_metric_deltas',{})[f'{a}_vs_{b}']=corr(arr(a),arr(b))
# speaker-level paired means, to avoid treating two utterances as independent
sp=sorted(set(r['speaker'] for r in rows)); sp_rows=[]
for s in sp:
 sub=[r for r in rows if r['speaker']==s]; z={'speaker':s,'snr_db':float(np.mean([r['snr_db'] for r in sub]))}
 for k in ['delta_pesq_wb','delta_stoi','delta_si_sdr_db','delta_dnsmos_sig','delta_dnsmos_ovrl']: z[k]=float(np.mean([r[k] for r in sub]))
 sp_rows.append(z)
out['speaker_level_summary']={'n':len(sp_rows),'speakers':len(sp_rows),**{k:float(np.mean([z[k] for z in sp_rows])) for k in ['snr_db','delta_pesq_wb','delta_stoi','delta_si_sdr_db','delta_dnsmos_sig','delta_dnsmos_ovrl']},'ovrl_negative_fraction':float(np.mean([z['delta_dnsmos_ovrl']<0 for z in sp_rows]))}
out['speaker_level_correlations']={k: corr(np.array([z['snr_db'] for z in sp_rows]),np.array([z[k] for z in sp_rows])) for k in ['delta_pesq_wb','delta_stoi','delta_si_sdr_db','delta_dnsmos_sig','delta_dnsmos_ovrl']}
OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
