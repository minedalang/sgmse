import json, re, math, os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

ROOT=Path('/data/coding/sgmse/experiments/p8_phoneme_probe_20261020')
CLIPS=Path('/data/coding/sgmse/experiments/p8_e1_candidate_listening_aligned_20261020/clips')
MODEL=ROOT/'local_model'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
# Approximate IPA consonant inventory. Labels not classified as vowels/blank/special are not
# automatically treated as content truth; this is only a posterior proxy.
VOWEL_CHARS=set('aeiouɑɐɒæɛɪɔʊəʌɜɞɤøœyɨɯɵɘɝɚɒɐ')
VOWEL_TOKENS={'a','e','i','o','u','ɑ','ɐ','ɒ','æ','ɛ','ɪ','ɔ','ʊ','ə','ʌ','ɜː','ɜ','ɞ','ɤ','ø','œ','y','ɨ','ɯ','ɵ','ɘ','ɝ','ɚ','ː'}

def is_vowel(tok):
    t=str(tok).strip()
    if not t or t.startswith('<') or t in {'|','▁','-'}: return False
    if t in VOWEL_TOKENS: return True
    # IPA diphthongs/syllabic vowel-like tokens
    return all((c in VOWEL_CHARS or c in 'ː̯̩') for c in t)

def edit_distance(a,b):
    n,m=len(a),len(b); prev=list(range(m+1))
    for i,x in enumerate(a,1):
        cur=[i]+[0]*m
        for j,y in enumerate(b,1): cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(x!=y))
        prev=cur
    return prev[m]

print('loading', MODEL, 'device', DEVICE, flush=True)
proc=Wav2Vec2Processor.from_pretrained(str(MODEL),local_files_only=True)
model=Wav2Vec2ForCTC.from_pretrained(str(MODEL),local_files_only=True)
model.eval().to(DEVICE)
# Keep inference deterministic and avoid unnecessary autocast differences.
files=sorted(CLIPS.glob('*_context_*.wav'))
items=[]
for p in files:
    m=re.match(r'(?P<group>\d+_fileid_\d+_[^_]+-[^_]+s)_context_(?P<cond>clean|noisy|y_pre|off|router)\.wav$',p.name)
    if not m: continue
    wav,sr=sf.read(p,dtype='float32')
    if wav.ndim>1: wav=wav.mean(1)
    items.append((m.group('group'),m.group('cond'),p,wav,sr))
assert len(items)==20, len(items)
# Process one at a time to keep memory stable and retain frame posterior.
raw={}
for idx,(group,cond,p,wav,sr) in enumerate(items,1):
    if sr!=16000: raise RuntimeError((p,sr))
    inp=proc(wav,sampling_rate=sr,return_tensors='pt')
    x=inp.input_values.to(DEVICE)
    with torch.inference_mode():
        logits=model(x).logits[0].float().cpu()
    probs=torch.softmax(logits,dim=-1).numpy()
    ids=probs.argmax(-1)
    conf=probs.max(-1)
    # Greedy CTC collapse for a frame-level sequence.
    collapsed=[]; prev=-1
    for i in ids.tolist():
        if i!=prev and i!=proc.tokenizer.pad_token_id:
            collapsed.append(int(i))
        prev=i
    toks=[proc.tokenizer.convert_ids_to_tokens(i) for i in collapsed]
    # Determine proxy consonant vocab from all tokenizer labels.
    labels=[proc.tokenizer.convert_ids_to_tokens(i) for i in range(probs.shape[-1])]
    consonant_ids=[]
    for i,t in enumerate(labels):
        t=str(t)
        if i==proc.tokenizer.pad_token_id or t.startswith('<') or t in {'|','▁','-'}: continue
        if not is_vowel(t): consonant_ids.append(i)
    voiced=ids!=proc.tokenizer.pad_token_id
    cmass=probs[:,consonant_ids].sum(1) if consonant_ids else np.zeros(len(ids))
    # Specific consonants are useful for the user's k/d observation.
    def mass_for(sym):
        ii=[i for i,t in enumerate(labels) if str(t)==sym]
        return float(probs[:,ii].sum())/len(ids) if ii else None
    raw[(group,cond)]={
      'file':p.name,'frames':int(len(ids)),'decoded_tokens':toks,
      'decoded':' '.join(map(str,toks)),'mean_top1_confidence':float(conf.mean()),
      'nonblank_frame_fraction':float(voiced.mean()),
      'consonant_proxy_vocab_size':len(consonant_ids),
      'consonant_proxy_posterior_mass':float(cmass.mean()),
      'consonant_proxy_mass_nonblank':float(cmass[voiced].mean()) if voiced.any() else 0.0,
      'top1_consonant_proxy_fraction':float(np.isin(ids,consonant_ids).mean()),
      'specific_phone_posterior_mass':{s:mass_for(s) for s in ['k','d','t','g','p','b','f','s','z','m','n','l','r']},
      '_ids':ids.tolist(), '_probs':probs.tolist(),
    }
    print(idx,group,cond,raw[(group,cond)]['decoded'],flush=True)
# Build comparisons to clean in each group.
records=[]
for group in sorted({g for g,c in raw}):
    clean=raw[(group,'clean')]
    clean_ids=np.array(clean['_ids'])
    clean_seq=clean['decoded_tokens']
    for cond in ['clean','noisy','y_pre','off','router']:
        d=raw[(group,cond)]
        ids=np.array(d['_ids'])
        seq=d['decoded_tokens']
        n=min(len(ids),len(clean_ids))
        d['frame_top1_agreement_to_clean']=float((ids[:n]==clean_ids[:n]).mean())
        d['token_edit_distance_to_clean']=int(edit_distance(seq,clean_seq))
        d['token_edit_similarity_to_clean']=float(1-edit_distance(seq,clean_seq)/max(1,len(clean_seq)))
        out={k:v for k,v in d.items() if not k.startswith('_')}
        out.update({'group':group,'condition':cond})
        records.append(out)
# Remove internal arrays from persisted JSON; save full per-condition table and compact summaries.
for d in raw.values(): d.pop('_ids',None); d.pop('_probs',None)
by_group={}
for r in records: by_group.setdefault(r['group'],{})[r['condition']]=r
summary={}
for group,dd in by_group.items():
    summary[group]={}
    for cond in ['clean','noisy','y_pre','off','router']:
        summary[group][cond]={k:v for k,v in dd[cond].items() if k not in {'file','decoded_tokens','decoded'}}
    # Differences vs Y_pre and noisy for interpretable primary proxies.
    for base in ['noisy','y_pre']:
      for target in ['off','router']:
        summary[group][f'{target}_minus_{base}']={
          k: dd[target][k]-dd[base][k] for k in ['consonant_proxy_posterior_mass','consonant_proxy_mass_nonblank','top1_consonant_proxy_fraction','mean_top1_confidence','nonblank_frame_fraction','frame_top1_agreement_to_clean','token_edit_similarity_to_clean']
        }
json.dump({'experiment_id':'p8_phoneme_probe_20261020','model':'facebook/wav2vec2-lv-60-espeak-cv-ft','model_local_dir':str(MODEL),'device':DEVICE,'sample_count':20,'records':records,'summary_by_group':summary},open(ROOT/'phoneme_results.json','w'),ensure_ascii=False,indent=2)
# Aggregate concise metrics.
def avg(cond,key):
  vals=[r[key] for r in records if r['condition']==cond and r[key] is not None]
  return float(np.mean(vals)) if vals else None
metrics={'schema_version':2,'experiment_id':'p8_phoneme_probe_20261020','status':'completed','model':'facebook/wav2vec2-lv-60-espeak-cv-ft','model_size_bytes':1263535127,'cache_dir':'/data/.rootcache/huggingface','device':DEVICE,'groups':4,'conditions_per_group':5,'decoded_segments':20,'records_file':'phoneme_results.json','aggregate':{}}
for cond in ['clean','noisy','y_pre','off','router']:
  metrics['aggregate'][cond]={k:avg(cond,k) for k in ['consonant_proxy_posterior_mass','consonant_proxy_mass_nonblank','top1_consonant_proxy_fraction','mean_top1_confidence','nonblank_frame_fraction','frame_top1_agreement_to_clean','token_edit_similarity_to_clean']}
json.dump(metrics,open(ROOT/'metrics.json','w'),ensure_ascii=False,indent=2)
print('wrote',ROOT/'phoneme_results.json',ROOT/'metrics.json',flush=True)
