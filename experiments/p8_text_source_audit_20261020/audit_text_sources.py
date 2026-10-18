#!/usr/bin/env python3
import csv,hashlib,json,glob,re
from pathlib import Path
ROOT=Path('/data/datasets/dns_ins20'); EXP=Path('/data/coding/sgmse/experiments/p8_text_source_audit_20261020')
MAN=Path('/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020/manifest.csv')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read_lines(p,keys):
 out=[]
 with open(p,errors='replace') as f:
  for n,line in enumerate(f,1):
   k=line.split(maxsplit=1)[0] if line.split() else ''
   if k in keys: out.append({'line':n,'raw':line.rstrip('\n'),'value':line.split(maxsplit=1)[1] if len(line.split(maxsplit=1))==2 else ''})
 return out
rows=list(csv.DictReader(open(MAN))); keys={r['utt_key'] for r in rows}
# Restrict to DNS metadata and SCP/config files; no broad/full-disk search.
files=[Path(x) for x in glob.glob(str(ROOT/'dns_snr_-20_-5'/'synthetic_snr-20_-5'/'scp'/'*'/'*')) if Path(x).is_file()]
files += [Path(x) for x in glob.glob(str(ROOT/'dns_snr_-20_-5'/'synthetic_snr-20_-5'/'scp_new'/'*'/'*')) if Path(x).is_file()]
files += [Path(x) for x in glob.glob(str(ROOT/'dns_snr_-20_-5'/'synthetic_snr-20_-5'/'*.cfg')) if Path(x).is_file()]
files=sorted(set(files))
text_matches=[]
for p in files:
 if p.name in {'text','utt2spk','spk2utt','wav.scp','spk1.scp','noise1.scp'}:
  ms=read_lines(p,keys)
  if ms: text_matches.append({'path':str(p),'sha256':sha(p),'matches':ms})
# exact clean and noisy identities/hashes from manifest; source basename metadata is tokenized but not transcript.
meta={'candidate_files_scanned':len(files),'text_or_keyed_metadata_matches':text_matches,'all_text_match_values_dummy':all(m['value'].strip()=='dummy' for x in text_matches if Path(x['path']).name=='text' for m in x['matches'])}
source_cfg=[]
for p in sorted(ROOT.glob('dns_snr_-20_-5/synthetic_snr-20_-5/*.cfg')): source_cfg.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
result={'schema_version':1,'experiment_id':'p8_text_source_audit_20261020','status':'completed','question':'Whether local DNS metadata provides non-dummy original speech text/books mapping for 8 P8 clean files.','manifest_path':str(MAN),'manifest_sha256':sha(MAN),'scope':'Read-only targeted audit of DNS Ins20 synthetic_snr-20_-5 metadata, SCPs and configs; no network and no model inference.','dataset_modified':False,'checkpoint_modified':False,'source_files':source_cfg,'metadata':meta,'utterances':[]}
for r in rows:
 clean=Path(r['clean_path']); utt=r['utt_key'];
 result['utterances'].append({'utt_key':utt,'speaker':r['speaker'],'fileid':r['fileid'],'clean_path':str(clean),'clean_exists':clean.exists(),'clean_sha256_actual':sha(clean) if clean.exists() else None,'clean_sha256_manifest':r['clean_sha256'],'clean_hash_matches_manifest':clean.exists() and sha(clean)==r['clean_sha256'],'book_chapter_filename_tokens':{'book':re.search(r'(book_\d+)',r['noisy_basename']).group(1),'chapter':re.search(r'(chp_\d+)',r['noisy_basename']).group(1)},'text_source_status':'no_non_dummy_local_transcript_found'})
(EXP/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
