# Independent standard-library reconstruction; does not import registration implementation.
import json,hashlib,zipfile,time,signal
from pathlib import Path
B=Path(__file__).resolve().parent
signal.alarm(60);t=time.monotonic()
r=json.loads((B/'prospective_registry.json').read_text());p=Path(r['source_manifest']);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==r['source_manifest_sha256']=='7f14a16a20880031932ab4ce7f7d3682c1b16cc0cc2eb8cd141e49ad7de087bc'
allrows=list(map(json.loads,data.decode().splitlines()));eligible=[]
for x in allrows:
 h=x['clean_header'];n=int(x['id'].rsplit('_',1)[1]);seconds=h['frames']/h['sample_rate']
 if x['local_partition']=='train' and x['speaker'] not in {'p226','p287'} and n>=26 and h['sample_rate']==16000 and h['channels']==1 and 3<=seconds<=5:eligible.append(x)
counts={s:sum(y['speaker']==s for y in eligible) for s in {x['speaker'] for x in eligible}}
sp=sorted(s for s in counts if counts[s]>=2)[:4];expected=[]
for s in sp:
 candidates=sorted((x for x in eligible if x['speaker']==s),key=lambda x:int(x['id'].rsplit('_',1)[1]));expected.extend(candidates[:2])
assert [x['id'] for x in expected]==r['selected_ids'];assert len(expected)==8
checks=[]
with zipfile.ZipFile(p.parent/'trainset_28spk_txt.zip') as z:
 for old,new in zip(expected,r['records']):
  for key,value in old.items():assert new[key]==value,(old['id'],key)
  q=Path(old['clean_path']);b=q.read_bytes();st=q.stat();assert len(b)==new['local_clean_file']['bytes'];assert hashlib.sha256(b).hexdigest()==new['local_clean_file']['sha256'];assert st.st_mtime_ns==new['local_clean_file']['mtime_ns'];assert new['duration_seconds']==old['clean_header']['frames']/16000
  txt=z.read(old['transcript']['archive_entry']);assert len(txt)==old['transcript']['raw_text_bytes'];assert hashlib.sha256(txt).hexdigest()==old['transcript']['raw_text_SHA256'];assert txt.decode('utf-8-sig')==old['transcript']['original_text'];assert ' '.join(txt.decode('utf-8-sig').split())==old['transcript']['normalized_text'];assert hashlib.sha256(old['transcript']['normalized_text'].encode()).hexdigest()==old['transcript']['normalized_text_SHA256'];checks.append(old['id'])
m=json.loads((B/'metrics.json').read_text())
for path,h in m['source_locks_pre'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h==m['source_locks_post'][path]
result={'status':'passed','scope':'independent metadata selection/source field/raw-file hash/publisher raw and normalized reference checks; NOT audio alignment, actual phone identity or use-history completeness','ids':checks,'elapsed_seconds':time.monotonic()-t,'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'registry_sha256':hashlib.sha256((B/'prospective_registry.json').read_bytes()).hexdigest(),'audio_decode':0,'models':0,'network':0}
(B/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');signal.alarm(0);print(json.dumps(result))
