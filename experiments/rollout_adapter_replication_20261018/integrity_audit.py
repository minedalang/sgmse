#!/usr/bin/env python3
import csv, hashlib, json
from pathlib import Path
ROOT=Path('/data/coding/sgmse')
EXP=ROOT/'experiments/rollout_adapter_replication_20261018'
MAN=ROOT/'experiments/tr0_fresh_cv_manifest_20261018/manifest.csv'
CKPT=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
rows=list(csv.DictReader(open(MAN,newline='')))
rows=rows[:8]
checks=[]
for r in rows:
 for key in ('noisy','clean'):
  p=Path(r[key]); actual=sha(p); expected=r.get(key+'_sha256')
  checks.append({'path':str(p),'kind':key,'expected':expected,'actual':actual,'match':(expected is not None and actual==expected), 'historical_expected_available': expected is not None})
result={
 'experiment_id':'rollout_adapter_replication_integrity_audit_20261018',
 'checkpoint':{'path':str(CKPT),'sha256':sha(CKPT),'expected':'ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9'},
 'manifest':{'path':str(MAN),'sha256':sha(MAN),'expected':'7143b210c35ec93eb15bd21766e2f6619e0a8a342410941c27bdb4ffa5b5b68c','rows_checked':len(rows)},
 'input_checks':checks,
 'checkpoint_and_manifest_match':sha(CKPT)=='ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9' and sha(MAN)=='7143b210c35ec93eb15bd21766e2f6619e0a8a342410941c27bdb4ffa5b5b68c',
 'input_hashes_recorded_current_only':True,
 'historical_input_hashes_available':False,
 'dataset_modified':None,
 'dataset_modified_note':'No pre-run input hashes were stored in the locked manifest; current hashes are recorded but cannot prove historical immutability.',
 'checkpoint_overwrite':False,
}
(EXP/'integrity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'checkpoint_and_manifest_match':result['checkpoint_and_manifest_match'],'rows_checked':len(rows),'input_hashes_recorded':len(checks),'historical_input_hashes_available':False},indent=2))
if not result['checkpoint_and_manifest_match']: raise SystemExit(1)
