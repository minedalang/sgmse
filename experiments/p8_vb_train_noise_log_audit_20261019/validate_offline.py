from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,zlib,re,time,urllib.parse
D=Path(__file__).resolve().parent

def main():
 t=time.monotonic();read=lambda p:json.loads(Path(p).read_text());m=read(D/'metrics.json');cfg=read(D/'config.yaml');reg=read(cfg['registry']);archive=(D/'logfiles.zip').read_bytes();assert hashlib.md5(archive).hexdigest()==cfg['expected_archive_md5_from_publisher_page'];assert hashlib.sha256(archive).hexdigest()==m['archive_and_training']['archive_sha256']
 with zipfile.ZipFile(D/'logfiles.zip') as z:
  found=[x for x in z.infolist() if PurePosixPath(x.filename).name==cfg['only_decoded_zip_basename']];assert len(found)==1;x=found[0];raw=z.read(x);assert zlib.crc32(raw)&0xffffffff==x.CRC and raw==(D/'log_trainset_28spk.txt').read_bytes()
 source={}
 for number,line in enumerate(raw.decode('ascii').splitlines(),1):
  match=re.fullmatch(r'(p[0-9]+_[0-9]+)\s+([A-Za-z0-9_\-]+)\s+([-+]?\d+(?:\.\d+)?)',line);assert match and match[1] not in source;source[match[1]]=(match[2],match[3],float(match[3]),number,line)
 assert len(source)==m['train_line_count']==m['unique_IDs']==11572;assert m['column3_unique_numeric_values']==sorted({r[2] for r in source.values()})
 assert len(m['selected'])==12 and [x['id'] for x in m['selected']]==[x['id'] for x in reg['records']]
 for row,src in zip(m['selected'],reg['records']):
  token,nraw,num,line,txt=source[row['id']];assert (row['literal_noise_token'],row['column3_raw'],row['column3_numeric'],row['source_line_number'],row['source_line_raw'])==(token,nraw,num,line,txt);assert row['speaker']==src['speaker'];assert row['noise_recording_id']=='unknown' and row['nominal_SNR_dB'] is None
  assert all(row[k] is None for k in ['offset','channel','gain','segment_length','noise_source_sha256']);assert not any(row[k] for k in ['noise_disjoint_ready','scientific_gate_ready','split_frozen'])
 # Independent union-find instead of closure-by-feature expansion.
 parent={x['id']:x['id'] for x in m['selected']}
 def root(x):
  while parent[x]!=x:x=parent[x]
  return x
 for a in m['selected']:
  for b in m['selected']:
   if a['speaker']==b['speaker'] or a['literal_noise_token']==b['literal_noise_token']:parent[root(a['id'])]=root(b['id'])
 groups={}
 for x in parent:groups.setdefault(root(x),[]).append(x)
 expected=sorted(sorted(v) for v in groups.values());assert expected==sorted(m['components_speaker_or_same_literal_token'])
 requests=read(D/'http_requests.json');assert len(requests)==3 and sum(x['body_bytes'] for x in requests)==m['network_body_bytes']<=524288
 for req in requests:assert req['status']==200 and urllib.parse.urlsplit(req['url']).netloc=='datashare.ed.ac.uk' and hashlib.sha256((D/req['artifact']).read_bytes()).hexdigest()==req['body_sha256']
 for p,x in cfg['source_locks'].items():assert Path(p).stat().st_size==x['bytes'] and hashlib.sha256(Path(p).read_bytes()).hexdigest()==x['sha256']
 assert m['opened_payloads']==['log_trainset_28spk.txt'] and m['test_56_payload_open_count']==0
 result={'status':'passed','seconds':time.monotonic()-t,'network_requests':0,'audio_decodes':0,'models':0,'ASR':0,'training':0,'opened_zip_payloads':['log_trainset_28spk.txt'],'train_rows':11572,'join_rows':12,'component_sizes':sorted(map(len,expected)),'archive_MD5_verified':True,'training_CRC_verified':True,'metrics_sha256':hashlib.sha256((D/'metrics.json').read_bytes()).hexdigest(),'source_locks_post_equal':True}
 with (D/'independent_validation.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
