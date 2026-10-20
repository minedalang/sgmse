"""Independent CPU archive-to-manifest audit; no network/audio opens/models."""
from pathlib import Path
import json,hashlib,zipfile,zlib,gzip,csv,collections
H=Path(__file__).resolve().parent;P=H.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
D=read(H/'metrics.json');C=read(H/'config.yaml');assert D['status']=='completed' and not D['failures'];zipfilep=H/'trainset_28spk_txt.zip';assert zipfilep.stat().st_size==6489312 and sha(zipfilep)==D['archive_SHA256'];assert hashlib.md5(zipfilep.read_bytes()).hexdigest()==D['archive_MD5']==C['expected_official_MD5']
assert all(sha(p)==v for p,v in read(H/'parent_snapshot_hashes.json').items())
local=read(P/'preparation_metrics.json');assert all(sha(p)==v for p,v in local['source_locks_before_after'].items());ids=read(P/'scp_id_inventory.json');trainids=set(ids['train_ids']);testids=set(ids['test_ids']);inv=read(P/'filename_inventory.json');sets={s:{a:set(inv[s][a]['ids']) for a in ['clean','noisy']} for s in ['train','valid']};headers={(r['local_split'],r['arm'],r['id']):r for r in csv.DictReader(gzip.open(P/'train_source_headers.csv.gz','rt'))}
rows=[json.loads(r) for r in (H/'train_reference_manifest.jsonl').read_text().splitlines()];assert len(rows)==len(trainids)==11572 and len({r['id'] for r in rows})==11572;assert {r['id'] for r in rows}==trainids and not {r['id'] for r in rows}&testids
assert sha(H/'train_reference_manifest.jsonl')==D['manifest_SHA256'];split=collections.Counter();sp=collections.Counter();empty=0
with zipfile.ZipFile(zipfilep) as z:
 for r in rows:
  sid=r['id'];s=r['local_partition'];assert sid in sets[s]['clean'] and sid in sets[s]['noisy'];assert r['speaker']==sid.split('_')[0];assert Path(r['clean_path'])==Path(inv[s]['clean']['directory'])/(sid+'.wav') and Path(r['noisy_path'])==Path(inv[s]['noisy']['directory'])/(sid+'.wav')
  for arm in ['clean','noisy']:
   h=headers[s,arm,sid];assert r[arm+'_header']=={'sample_rate':int(h['samplerate']),'channels':int(h['channels']),'frames':int(h['frames']),'subtype':h['subtype']}
  t=r['transcript'];member=z.getinfo(t['archive_entry']);raw=z.read(member);assert zlib.crc32(raw)&0xffffffff==member.CRC;assert hashlib.sha256(raw).hexdigest()==t['raw_text_SHA256'];assert t['raw_text_bytes']==len(raw);assert t['original_text']==raw.decode('utf-8-sig');norm=' '.join(t['original_text'].split());assert t['normalized_text']==norm and hashlib.sha256(norm.encode('utf8')).hexdigest()==t['normalized_text_SHA256'];assert norm
  split[s]+=1;sp[r['speaker']]+=1
assert dict(split)==D['manifest_local_split_counts'] and len(sp)==28
R={'status':'passed','manifest_rows_verified':len(rows),'unique_train_ids':len(trainids),'train_partition_rows':split['train'],'valid_partition_rows':split['valid'],'speakers':len(sp),'test_content_opened':False,'audio_or_header_opens':0,'network_requests':0,'model_calls':0,'archive_MD5_SHA_verified':True,'expected_train_member_CRC_recomputed':len(rows),'all_manifest_raw_text_hashes_and_normalized_text_verified':True,'all_existing_header_metadata_joins_verified':True,'parent_snapshot_files_unchanged':True,'original_local_source_hashes_unchanged':True,'all_train_ids_covered_once':True,'spoken_alignment_verified':False};(H/'mapping_validation.json').write_text(json.dumps(R,indent=2)+'\n');print(json.dumps(R,indent=2))
