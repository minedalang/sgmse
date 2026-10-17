#!/usr/bin/env python3
"""Fixed 8-utterance pseudo-ASR stability smoke; never writes source audio."""
import csv, hashlib, json, re, time, platform, subprocess
from pathlib import Path
import torch, whisper

ROOT=Path('/data/coding/sgmse'); OUT=ROOT/'experiments/content_headroom_asr_smoke_20261018'
MAN=ROOT/'experiments/storm_error_attribution_p7_20261017/baseline_manifest.csv'
BASE=ROOT/'experiment_logs/rrsd_tcut_screen'
CACHE=Path('/data/.rootcache/whisper'); MODEL_NAME='tiny.en'; DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
TARGET_SNRS=[-20,-18,-16,-14,-11,-10,-8,-6]

def sha256(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def norm(s): return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip().split()
def lev(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
def snr(row):
 m=re.search(r'_snr(-?\d+)_fileid',row['noisy_path']); return int(m.group(1))
rows=list(csv.DictReader(open(MAN)))
# Fixed selection: for each target SNR, select row minimizing |actual-target|, then filename lexical.
selected=[]; used=set()
for target in TARGET_SNRS:
 cand=sorted((abs(snr(r)-target),r['filename'],r) for r in rows if r['filename'] not in used)
 _,_,r=cand[0]; selected.append(r); used.add(r['filename'])
selected=sorted(selected,key=lambda r:(snr(r),r['filename']))
# Conditions are existing outputs; clean/noisy from manifest, others indexed output directories.
conditions={}
for r in selected:
 idx=r['filename']
 conditions[idx]={'clean':r['clean_path'],'noisy':r['noisy_path']}
 for c,d in [('router_off','off'),('router','router'),('cut005','cut005')]: conditions[idx][c]=str(BASE/d/idx)
# load model and report hash
model_path=CACHE/'tiny.en.pt'
if not model_path.exists(): raise FileNotFoundError(model_path)
model_sha=sha256(model_path)
torch.manual_seed(0)
model=whisper.load_model(MODEL_NAME,device=DEVICE,download_root=str(CACHE))
rows_out=[]; start_all=time.time()
for idx,paths in conditions.items():
 trans={}; runt={}
 for cond,path in paths.items():
  p=Path(path)
  if not p.exists(): raise FileNotFoundError(path)
  st=time.time()
  result=model.transcribe(str(p), language='en', task='transcribe', fp16=(DEVICE=='cuda'), temperature=0, beam_size=1, best_of=1, condition_on_previous_text=False, verbose=False)
  trans[cond]=result.get('text','').strip(); runt[cond]=time.time()-st
 clean_words=norm(trans['clean'])
 rec={'filename':idx,'uttid':next(r['uttid'] for r in selected if r['filename']==idx),'snr_db':snr(next(r for r in selected if r['filename']==idx)),'transcripts':trans,'runtime_seconds':runt,'normalized_edit_distance':{},'word_counts':{'clean':len(clean_words)}}
 for c,t in trans.items():
  nw=norm(t); d=lev(nw,clean_words); rec['normalized_edit_distance'][c]=d/max(1,len(clean_words)); rec['word_counts'][c]=len(nw)
 rows_out.append(rec); print(json.dumps({'filename':idx,'snr_db':rec['snr_db'],'ned':rec['normalized_edit_distance']},ensure_ascii=False),flush=True)
metrics={'schema_version':1,'experiment_id':'content_headroom_asr_smoke_20261018','status':'completed','interpretation':'Pseudo-WER/ASR stability diagnostic only; clean transcript is a reference for this smoke, not dataset ground-truth WER.','manifest_path':str(MAN),'manifest_sha256':sha256(MAN),'selection_rule':{'targets_snr_db':TARGET_SNRS,'rule':'For each target SNR, choose unused P7 manifest row minimizing absolute SNR difference, tie-break by filename; selected rows sorted by actual SNR then filename.'},'model':{'name':MODEL_NAME,'path':str(model_path),'sha256':model_sha,'device':DEVICE,'torch':torch.__version__,'whisper_module':whisper.__file__},'seed':0,'decode':{'language':'en','task':'transcribe','temperature':0,'beam_size':1,'best_of':1,'condition_on_previous_text':False,'fp16':DEVICE=='cuda'},'n_utterances':len(rows_out),'conditions':['clean','noisy','router_off','router','cut005'],'total_runtime_seconds':time.time()-start_all,'rows':rows_out,'input_sha256':{r['filename']:{c:sha256(Path(p)) for c,p in conditions[r['filename']].items()} for r in selected}}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n')
