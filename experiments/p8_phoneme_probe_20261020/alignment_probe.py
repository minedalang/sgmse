import json,re
from pathlib import Path
import numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
ROOT=Path('/data/coding/sgmse/experiments/p8_phoneme_probe_20261020'); CLIPS=Path('/data/coding/sgmse/experiments/p8_e1_candidate_listening_aligned_20261020/clips'); M=ROOT/'local_model'
proc=Wav2Vec2Processor.from_pretrained(str(M),local_files_only=True); model=Wav2Vec2ForCTC.from_pretrained(str(M),local_files_only=True).eval().to('cuda' if torch.cuda.is_available() else 'cpu'); dev=next(model.parameters()).device; blank=proc.tokenizer.pad_token_id
D={}
for p in sorted(CLIPS.glob('*_context_*.wav')):
 m=re.match(r'(\d+_fileid_\d+_[^_]+-[^_]+s)_context_(clean|noisy|y_pre|off|router)\.wav$',p.name)
 if not m: continue
 a,sr=sf.read(p,dtype='float32');
 with torch.inference_mode(): ids=model(proc(a,sampling_rate=sr,return_tensors='pt').input_values.to(dev)).logits.argmax(-1)[0].cpu().numpy()
 D[(m.group(1),m.group(2))]=ids
rows=[]
for g in sorted({g for g,c in D}):
 c=D[(g,'clean')]; active=c!=blank
 for cond in ['clean','noisy','y_pre','off','router']:
  x=D[(g,cond)]; n=min(len(c),len(x)); cc=c[:n]; xx=x[:n]; aa=active[:n]
  row={'group':g,'condition':cond,'all_frame_agreement':float((xx==cc).mean()),'clean_active_frames':int(aa.sum()),'clean_active_frame_agreement':float((xx[aa]==cc[aa]).mean()) if aa.any() else None,'both_nonblank_frames':int(((xx!=blank)&aa).sum()),'both_nonblank_agreement':float((xx[(xx!=blank)&aa]==cc[(xx!=blank)&aa]).mean()) if ((xx!=blank)&aa).any() else None}
  rows.append(row)
json.dump({'experiment_id':'p8_phoneme_probe_20261020','model':'facebook/wav2vec2-lv-60-espeak-cv-ft','device':str(dev),'blank_id':int(blank),'rows':rows},open(ROOT/'alignment_results.json','w'),indent=2)
for cond in ['clean','noisy','y_pre','off','router']:
 z=[r for r in rows if r['condition']==cond]; vals=[r['clean_active_frame_agreement'] for r in z if r['clean_active_frame_agreement'] is not None]; vals2=[r['both_nonblank_agreement'] for r in z if r['both_nonblank_agreement'] is not None]
 print(cond,'clean-active agree',sum(vals)/len(vals),'both-nonblank',sum(vals2)/len(vals2) if vals2 else None)
