import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
base=Path(__file__).resolve().parent.parent
out=base/'calibration_outputs'
metrics=json.loads((out/'metrics.json').read_text())
rows=json.loads((out/'per_sample.json').read_text())
manifest=json.loads((base/'retrieval/selected_manifest_frozen.json').read_text())
frozen=json.loads((out/'frozen_inputs.json').read_text())
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert metrics['status']=='completed' and (out/'COMPLETED.json').exists()
assert not (out/'FAILED.json').exists() and not (out/'RUNNING.json').exists()
assert sha(base/'retrieval/selected_manifest_frozen.json')==frozen['manifest_sha256']
assert sha(base/'calibrate_clean.py')==frozen['script_sha256']
assert sha(base/'calibration_protocol.yaml')==frozen['protocol_sha256']
assert len(rows)==len(manifest['files'])==28
vocab=json.loads((Path(frozen['model_path'])/'vocab.json').read_text())
max_ctc_error=0.
for r,m in zip(rows,manifest['files']):
 assert r['sample_id']==m['sample_id'] and r['consonant']==m['consonant'] and r['is_primary']==m['is_primary']
 assert sha(r['wav_path'])==r['source_sha256']==m['sha256']
 arrpath=base/r['array_path']
 assert sha(arrpath)==r['array_sha256']
 with np.load(arrpath,allow_pickle=False) as arr:
  scores=arr['candidate_loglikelihoods']; labels=arr['candidate_labels'].tolist(); logits=arr['logits']; probs=arr['probabilities']; ids=arr['candidate_token_ids']
  assert len(labels)==24 and set(labels)==set(r['all24_loglikelihoods']) and ids.shape==(24,3)
  assert np.array_equal(scores,np.asarray([r['all24_loglikelihoods'][c] for c in labels],dtype=np.float32))
  assert np.array_equal(ids,np.asarray([[vocab[t] for t in r['all24_candidate_ipa_tokens'][c]] for c in labels]))
  logprobs=torch.from_numpy(logits).log_softmax(-1)
  assert np.allclose(logprobs.exp().numpy(),probs,rtol=1e-6,atol=1e-7)
  recalculated=-F.ctc_loss(logprobs[:,None,:].expand(-1,24,-1).contiguous(),torch.from_numpy(ids),torch.full((24,),len(logits),dtype=torch.long),torch.full((24,),3,dtype=torch.long),blank=0,reduction='none',zero_infinity=False).numpy()
  max_ctc_error=max(max_ctc_error,float(np.max(np.abs(recalculated-scores))))
  assert np.allclose(recalculated,scores,rtol=1e-6,atol=1e-6)
  winners=[c for c,s in zip(labels,scores) if s==scores.max()]
  pred=winners[0] if len(winners)==1 else None
  assert pred==r['top1_prediction'] and (pred==r['consonant'])==r['top1_correct']
  margin=float(scores[labels.index(r['consonant'])]-scores[labels.index(r['binary_competitor'])])
  assert margin==r['binary_loglikelihood_margin'] and (margin>0)==r['binary_correct'] and (margin==0)==r['binary_tie']
primary=[r for r in rows if r['is_primary']]
supp=[r for r in rows if not r['is_primary']]
assert len(primary)==16 and len(supp)==12 and len({r['speaker'] for r in primary})==4
assert Counter(r['consonant'] for r in primary)==Counter(k=4,g=4,t=4,d=4)
def summary(group):
 return {'n':len(group),'top1_correct':sum(r['top1_correct'] for r in group),'top1_accuracy':sum(r['top1_correct'] for r in group)/len(group),'binary_correct':sum(r['binary_correct'] for r in group),'binary_accuracy':sum(r['binary_correct'] for r in group)/len(group),'mean_binary_loglikelihood_margin':sum(r['binary_loglikelihood_margin'] for r in group)/len(group)}
assert summary(primary)==metrics['primary'] and summary(supp)==metrics['supplementary_descriptive_only']
for key,reported in metrics['primary_groups'].items():
 group=primary
 for item in key.split('/'):
  if item.startswith('speaker='): group=[r for r in group if r['speaker']==item[8:]]
 # contrast labels contain slash; match full suffix explicitly.
 if 'contrast=' in key:
  contrast=key.split('contrast=')[1]
  group=[r for r in group if r['consonant'] in contrast.split('/')]
 assert summary(group)==reported
conf=metrics['primary_confusion_matrix']
for truth,line in conf.items():
 actual=Counter((r['top1_prediction'] or 'tie') for r in primary if r['consonant']==truth)
 assert all(count==actual[pred] for pred,count in line.items())
gate=metrics['primary']['top1_correct']>=12 and metrics['primary']['binary_correct']>=14 and all(metrics['primary_groups']['contrast='+c]['top1_correct']>=6 for c in ['k/g','t/d'])
assert gate==metrics['gate_passed']
errors=[{'sample_id':r['sample_id'],'nominal':r['consonant'],'prediction':r['top1_prediction'],'binary_margin':r['binary_loglikelihood_margin']} for r in primary if not r['top1_correct']]
report={'status':'passed','saved_arrays':28,'all24_scores_recomputed_from_saved_logits':True,'max_recomputed_ctc_absolute_error':max_ctc_error,'source_wav_hashes_unchanged':True,'primary':metrics['primary'],'supplementary_descriptive_only':metrics['supplementary_descriptive_only'],'primary_groups':metrics['primary_groups'],'gate_passed':gate,'next_stage':metrics['next_stage'],'primary_top1_errors':errors,'process_wall_seconds':8.99,'script_runtime_seconds':metrics['runtime_seconds'],'cuda_forward_duration_seconds':None,'timing_note':'CUDA-only forward duration was not instrumented; complete script runtime includes hashing, model load, forward, scoring and writing. No rerun performed. Initial external /usr/bin/time launcher failed before Python execution; Bash builtin time then launched calibration once.'}
assert sum(p.stat().st_size for p in base.rglob('*') if p.is_file())<25000000
with (out/'worker_validation.json').open('x') as f: json.dump(report,f,indent=2); f.write('\n')
print(json.dumps(report,indent=2))
