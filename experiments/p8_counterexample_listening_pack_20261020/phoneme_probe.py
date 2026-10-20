import json,re
from pathlib import Path
import numpy as np,soundfile as sf,torch
from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
ROOT=Path('/data/coding/sgmse/experiments/p8_counterexample_listening_pack_20261020'); CLIPS=ROOT/'clips'; M=Path('/data/coding/sgmse/experiments/p8_phoneme_probe_20261020/local_model'); dev='cuda' if torch.cuda.is_available() else 'cpu'
proc=Wav2Vec2Processor.from_pretrained(str(M),local_files_only=True); model=Wav2Vec2ForCTC.from_pretrained(str(M),local_files_only=True).eval().to(dev); blank=proc.tokenizer.pad_token_id
V={}
for p in sorted(CLIPS.glob('*.wav')):
 m=re.match(r'(\d+_fileid_\d+_[0-9.]+-[0-9.]+s)_(clean|noisy|y_pre|off|router)\.wav$',p.name)
 if not m: continue
 a,sr=sf.read(p,dtype='float32')
 with torch.inference_mode(): logits=model(proc(a,sampling_rate=sr,return_tensors='pt').input_values.to(dev)).logits[0].float(); probs=logits.softmax(-1).cpu().numpy()
 V[(m.group(1),m.group(2))]=probs

def collapse(probs):
 ids=probs.argmax(-1); out=[]; prev=-1
 for i in ids:
  if i!=prev and i!=blank: out.append(int(i))
  prev=i
 return out
rows=[]
for g in sorted({g for g,c in V}):
 clean=V[(g,'clean')]; ci=clean.argmax(-1); active=ci!=blank; n=len(ci)
 labels=[proc.tokenizer.convert_ids_to_tokens(i) for i in range(clean.shape[-1])]
 def consonant(t):
  vowels=set('aeiouəɛɪʊɐɔɑɒʌæɝɚøœyɤɵɞɨ')
  return t not in {'<s>','<pad>','</s>','<unk>','|','▁','ː','̃','ʲ','ʷ','ˈ','ˌ'} and t not in vowels
 cids=[i for i,t in enumerate(labels) if consonant(t)]
 for cond in ['clean','noisy','y_pre','off','router']:
  p=V[(g,cond)]; ids=p.argmax(-1); z=active[:len(ids)]; target=ci[:len(ids)]
  tp=p[np.arange(len(ids)),target]
  cm=p[:,cids].sum(-1)
  out={'group':g,'condition':cond,'frames':len(ids),'decoded':' '.join(proc.tokenizer.convert_ids_to_tokens(collapse(p))),'mean_top1_conf':float(p.max(-1).mean()),'clean_active_target_posterior':float(tp[z].mean()),'clean_active_target_top1':float((ids[z]==target[z]).mean()),'consonant_mass':float(cm.mean()),'consonant_runs':sum(1 for i in collapse(p) if consonant(proc.tokenizer.convert_ids_to_tokens(i)))}
  rows.append(out)
json.dump({'experiment_id':'p8_counterexample_phoneme_probe_20261020','model':'facebook/wav2vec2-lv-60-espeak-cv-ft','device':dev,'rows':rows},open(ROOT/'metrics.json','w'),indent=2,ensure_ascii=False)
for g in sorted({r['group'] for r in rows}):
 print('\n',g)
 for r in rows:
  if r['group']==g: print(r['condition'],round(r['clean_active_target_posterior'],3),round(r['clean_active_target_top1'],3),round(r['consonant_mass'],3),r['decoded'])
