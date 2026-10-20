#!/usr/bin/env python3
import json,time,hashlib
from pathlib import Path
import torch, torchaudio
from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
ROOT=Path('/data/coding/sgmse'); IN=ROOT/'experiments/p8_e1_candidate_listening_aligned_20261020/clips'; OUT=ROOT/'experiments/p8_phoneme_probe_20261020'; SNAP=Path('/data/.rootcache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft/snapshots/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4')
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(0)
proc=Wav2Vec2Processor.from_pretrained(str(SNAP),local_files_only=True); model=Wav2Vec2ForCTC.from_pretrained(str(SNAP),local_files_only=True).to(DEVICE).eval()
vocab=proc.tokenizer.get_vocab(); id2tok={v:k for k,v in vocab.items()}; blank=proc.tokenizer.pad_token_id
vowels=set('aeiouəɛɪʊɐɔɑɒʌæɝɚøœyɤɵɞɨ')
def is_cons(tok):
 return tok not in {'<s>','<pad>','</s>','<unk>'} and tok not in vowels and tok not in {'|','▁','ː','̃','ʲ','ʷ','ˈ','ˌ'}
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
rows=[]; t0=time.time()
for p in sorted(IN.glob('*.wav')):
 wav,sr=torchaudio.load(str(p)); wav=wav.mean(0)
 if sr!=16000: wav=torchaudio.functional.resample(wav,sr,16000)
 inp=proc(wav.numpy(),sampling_rate=16000,return_tensors='pt',padding=True)
 with torch.inference_mode(): logits=model(inp.input_values.to(DEVICE)).logits[0]; probs=logits.softmax(-1); ids=probs.argmax(-1).cpu().tolist(); conf=probs.max(-1).values.cpu().tolist()
 # collapse CTC, retaining run-level confidence
 emit=[]; i=0
 while i<len(ids):
  j=i+1
  while j<len(ids) and ids[j]==ids[i]: j+=1
  if ids[i]!=blank:
   tok=id2tok.get(ids[i],'<unk>'); emit.append({'token':tok,'posterior_mean':sum(conf[i:j])/(j-i),'posterior_max':max(conf[i:j]),'frames':j-i})
  i=j
 decoded=' '.join(x['token'] for x in emit)
 cons=[x for x in emit if is_cons(x['token'])]
 cmass=sum(float(probs[:,k].mean().item()) for k,t in id2tok.items() if is_cons(t))
 rows.append({'file':p.name,'group':p.name.split('_')[0],'condition':p.stem.split('_context_')[-1],'sha256':sha(p),'sample_rate':int(sr),'num_frames':len(ids),'decoded_phonemes':decoded,'token_runs':emit,'mean_argmax_posterior':sum(conf)/len(conf),'mean_consonant_posterior_mass':cmass,'emitted_consonant_runs':len(cons),'emitted_consonant_posterior_mean':sum(x['posterior_mean'] for x in cons)/len(cons) if cons else None})
 print(p.name,decoded,flush=True)
out={'schema_version':1,'experiment_id':'p8_phoneme_probe_20261020','status':'completed','model':'facebook/wav2vec2-lv-60-espeak-cv-ft','model_snapshot':str(SNAP),'model_weight_sha256':sha(SNAP/'pytorch_model.bin'),'device':DEVICE,'seed':0,'decoder':'greedy CTC argmax, tokenizer grouped runs','phoneme_consonant_proxy':'mean frame posterior mass over IPA consonant tokens; supplementary proxy, not phoneme ground truth','n_clips':len(rows),'runtime_seconds':time.time()-t0,'rows':rows}
(OUT/'metrics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
