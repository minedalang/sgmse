import json,re
from pathlib import Path
import numpy as np,soundfile as sf,torch
from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
ROOT=Path('/data/coding/sgmse/experiments/p8_phoneme_probe_20261020'); CLIPS=Path('/data/coding/sgmse/experiments/p8_e1_candidate_listening_aligned_20261020/clips'); M=ROOT/'local_model'; dev='cuda' if torch.cuda.is_available() else 'cpu'
proc=Wav2Vec2Processor.from_pretrained(str(M),local_files_only=True); model=Wav2Vec2ForCTC.from_pretrained(str(M),local_files_only=True).eval().to(dev); blank=proc.tokenizer.pad_token_id
P={}; I={}
for p in sorted(CLIPS.glob('*_context_*.wav')):
 m=re.match(r'(\d+_fileid_\d+_[^_]+-[^_]+s)_context_(clean|noisy|y_pre|off|router)\.wav$',p.name)
 if not m: continue
 a,sr=sf.read(p,dtype='float32')
 with torch.inference_mode(): logits=model(proc(a,sampling_rate=sr,return_tensors='pt').input_values.to(dev)).logits[0].float(); probs=logits.softmax(-1).cpu().numpy()
 P[(m.group(1),m.group(2))]=probs; I[(m.group(1),m.group(2))]=probs.argmax(-1)
rows=[]
for g in sorted({x for x,c in P}):
 cp=P[(g,'clean')]; ci=I[(g,'clean')]; active=ci!=blank
 for cond in ['noisy','y_pre','off','router']:
  p=P[(g,cond)]; n=min(len(cp),len(p)); a=active[:n]; target=ci[:n]
  # Probability assigned to clean argmax token at each clean-active frame.
  tp=p[np.arange(n),target]
  # Exclude blank clean frames, which otherwise dominate.
  z=tp[a]
  rows.append({'group':g,'condition':cond,'clean_active_frames':int(a.sum()),'mean_clean_target_posterior':float(z.mean()),'median_clean_target_posterior':float(np.median(z)),'target_top1_fraction':float((p.argmax(-1)[:n][a]==target[a]).mean()),'mean_nonblank_clean_target_posterior':float(p[np.arange(n),target][a & (p.argmax(-1)[:n]!=blank)].mean()) if (a & (p.argmax(-1)[:n]!=blank)).any() else None})
# differences each group and aggregate
for g in sorted({r['group'] for r in rows}):
 d={r['condition']:r for r in rows if r['group']==g}
 for c in ['noisy','y_pre','off','router']:
  d[c]['delta_vs_y_pre']=d[c]['mean_clean_target_posterior']-d['y_pre']['mean_clean_target_posterior']
  d[c]['delta_vs_off']=d[c]['mean_clean_target_posterior']-d['off']['mean_clean_target_posterior']
json.dump({'experiment_id':'p8_phoneme_probe_20261020','model':'facebook/wav2vec2-lv-60-espeak-cv-ft','device':dev,'target_definition':'clean waveform CTC argmax per same frame; offline pseudo-target only','rows':rows},open(ROOT/'target_posterior_results.json','w'),indent=2)
for c in ['noisy','y_pre','off','router']:
 z=[r for r in rows if r['condition']==c]; print(c,'target posterior',np.mean([r['mean_clean_target_posterior'] for r in z]),'target top1',np.mean([r['target_top1_fraction'] for r in z]))
