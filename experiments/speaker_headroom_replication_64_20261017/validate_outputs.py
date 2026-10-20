#!/usr/bin/env python3
"""Validate paired output directories against the immutable 128-row manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path('/data/coding/sgmse')
EXP=ROOT/'experiments/speaker_headroom_replication_64_20261017'
LOG=ROOT/'experiment_logs/speaker_headroom_replication_64_20261017'
LOCKED='c04657fa21a33c7f2dc749514cec899767101ba53f46222e9f4ab8577e2f4492'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--arms',nargs='+',choices=('off','router'),default=('off','router'))
 args=parser.parse_args()
 manifest=EXP/'manifest.csv'
 if digest(manifest)!=LOCKED: raise RuntimeError('manifest hash changed')
 rows=list(csv.DictReader(manifest.open(newline='',encoding='utf-8')))
 expected={r['filename'] for r in rows}
 records=[]; subtype_counts={}
 for arm in args.arms:
  out=LOG/arm
  actual={p.name for p in out.glob('*.wav')}
  if actual != expected:
   raise RuntimeError(f'{arm}: expected {len(expected)} exact WAVs, got {len(actual)}, missing={len(expected-actual)}, extra={len(actual-expected)}')
  for r in rows:
   p=out/r['filename']; info=sf.info(p)
   x,sr=sf.read(p,dtype='float32',always_2d=False)
   if sr!=int(r['sample_rate_hz']) or x.ndim!=1 or len(x)!=int(r['samples']) or not np.isfinite(x).all():
    raise ValueError(f'{arm} invalid finite/aligned output: {p} sr={sr} shape={x.shape}')
   subtype_counts[info.subtype]=subtype_counts.get(info.subtype,0)+1
  print(f'{arm}: {len(actual)} finite, mono, sample-aligned outputs validated')
 for r in rows:
  noisy=Path(r['noisy_path']); clean=Path(r['clean_path'])
  if digest(noisy)!=r['noisy_sha256'] or digest(clean)!=r['clean_sha256']:
   raise RuntimeError(f'source hash changed for {r["filename"]}')
  row={'filename':r['filename'],'uttid':r['uttid'],'speaker':r['speaker'],
       'noisy_sha256':r['noisy_sha256'],'clean_sha256':r['clean_sha256']}
  for arm in args.arms:
   p=LOG/arm/r['filename']; row[f'{arm}_sha256']=digest(p)
   row[f'{arm}_subtype']=sf.info(p).subtype
   row[f'{arm}_samples']=sf.info(p).frames
  records.append(row)
 with (EXP/f'output_manifest_{"_".join(args.arms)}.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
 summary={'manifest_sha256':digest(manifest),'arms':args.arms,'n_outputs_per_arm':len(rows),
          'source_hashes_unchanged':True,'all_outputs_finite_aligned_mono_16k':True,
          'subtype_counts_across_arms':subtype_counts}
 (EXP/f'output_validation_{"_".join(args.arms)}.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
