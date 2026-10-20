"""Bounded official TRAIN transcript retrieval and filename mapping; no models/test content."""
from pathlib import Path,PurePosixPath
import urllib.request,urllib.error,time,datetime,json,hashlib,zipfile,zlib,re,gzip,csv,collections,subprocess
H=Path(__file__).resolve().parent;R=H.parent.parent;V=Path('/data/datasets/vb/voicebank_demand');T=time.monotonic()-(time.time()-1792376277.2217903);CAP=9000000;records=[];transferred=0
assert not (H/'metrics.json').exists(),'Refuse existing completed mapping'
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,o):(H/n).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def budget():
 assert time.monotonic()-T<180,'Networkstage180s cap'
 assert sum(p.stat().st_size for p in H.rglob('*') if p.is_file())<20000000,'20MBartifactcap'
def fetch(name,url,proxy=False):
 global transferred
 budget();r={'name':name,'url':url,'retrieval_utc':now(),'proxy':proxy,'status':'failed','bytes':0,'sha256':None,'http_status':None,'final_url':None}
 try:
  op=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'} if proxy else {}));request=urllib.request.Request(url,headers={'User-Agent':'SpeechEnhancementPrimarySourceAudit/1.0'})
  with op.open(request,timeout=22) as response:
   r.update(http_status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'),content_length=response.headers.get('Content-Length'))
   chunks=[];n=0
   while True:
    budget();b=response.read(min(262144,CAP-transferred+1))
    if not b:break
    n+=len(b);transferred+=len(b)
    if transferred>CAP:raise RuntimeError('9MBcumulativedownloadcap')
    chunks.append(b)
  data=b''.join(chunks);(H/name).parent.mkdir(parents=True,exist_ok=True);(H/name).write_bytes(data);r.update(status='saved',bytes=len(data),sha256=sha(data),md5=hashlib.md5(data).hexdigest())
 except Exception as exc:r.update(error=repr(exc),http_status=getattr(exc,'code',r['http_status']))
 records.append(r);dump('retrieval_progress.json',{'records':records,'cumulative_network_bytes':transferred,'wall_seconds':time.monotonic()-T});return r
sources=[('publisher/item.html','https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757'),('publisher/item_metadata.json','https://datashare.ed.ac.uk/server/api/core/items/6ed35425-bf14-4d2b-93a1-0a4984952757'),('publisher/license.txt','https://datashare.ed.ac.uk/server/api/core/bitstreams/7e7bc32f-e94a-436a-8e97-416a972c7e1a/content'),('publisher/train_bitstream_metadata.json','https://datashare.ed.ac.uk/server/api/core/bitstreams/fb55b5de-c2db-4451-ac0a-134e98011ffa'),('publisher/trainset_28spk_txt.zip','https://datashare.ed.ac.uk/bitstreams/fb55b5de-c2db-4451-ac0a-134e98011ffa/download')]
result={'status':'running','failures':[],'model_calls':0,'training_calls':0,'waveform_decodes':0,'test_transcript_fetches':0,'test_content_opened':False}
try:
 for name,url in sources:
  r=fetch(name,url,True)
  if r['status']!='saved':raise RuntimeError('Official source unavailable after one authorized fallback:'+name)
 md=json.loads((H/'publisher/train_bitstream_metadata.json').read_text());zpath=H/'publisher/trainset_28spk_txt.zip';blob=zpath.read_bytes();require_name=md.get('name')
 assert require_name=='trainset_28spk_txt.zip',('Unexpected officialbitstreamname',require_name)
 metadata_size=md.get('sizeBytes',md.get('size'));assert int(metadata_size)==len(blob),('Metadatafilesize mismatch',metadata_size,len(blob))
 checksum=md.get('checkSum',md.get('checksum',{}));checksum_algorithm=str(checksum.get('checkSumAlgorithm',checksum.get('algorithm',''))).upper();checksum_value=checksum.get('value');checksum_verified=None
 if checksum_algorithm=='MD5':checksum_verified=hashlib.md5(blob).hexdigest().lower()==checksum_value.lower();assert checksum_verified,'OfficialMD5 mismatch'
 lic=(H/'publisher/license.txt').read_text(errors='replace');license_ccby=('creativecommons.org/licenses/by/4.0' in lic.lower() or 'creative commons attribution 4.0' in lic.lower());assert license_ccby,'OfficiallicenseCCBY4notverified'
 ids=json.loads((H/'scp_id_inventory.json').read_text());trainset=set(ids['train_ids']);testset=set(ids['test_ids']);inventory=json.loads((H/'filename_inventory.json').read_text());headers=list(csv.DictReader(gzip.open(H/'train_source_headers.csv.gz','rt')));header_map={(r['local_split'],r['arm'],r['id']):r for r in headers}
 texts={};duplicates=[];ignored=[];unexpected=[];entries=[];empty=[];crc_verified=0
 with zipfile.ZipFile(zpath) as z:
  members=z.infolist();total_uncompressed=sum(i.file_size for i in members);assert total_uncompressed<=60000000 and all(i.file_size<=1000000 for i in members),'Unsafeuncompressedsize'
  for member in members:
   name=member.filename;pure=PurePosixPath(name);assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in name and not re.match(r'^[A-Za-z]:',name),'Unsafearchivepath'
   assert not (member.external_attr>>16 & 0o170000)==0o120000,'Archivesymlink'
   entry={'member':name,'uncompressed_bytes':member.file_size,'compressed_bytes':member.compress_size,'crc32':format(member.CRC,'08x')};entries.append(entry)
   if member.is_dir():entry['status']='directory';continue
   resource='__MACOSX' in pure.parts or pure.name.startswith('._') or pure.name=='.DS_Store'
   sid=pure.stem
   if not resource and pure.suffix.lower()=='.txt' and sid in testset:raise RuntimeError('ArchiveunexpectedlycontainstestID;stopwithoutreadingit')
   # ZipFile.read checks eachmemberCRC; non-content metadata is never used as labels.
   data=z.read(member);crc_verified+=1;assert (zlib.crc32(data)&0xffffffff)==member.CRC
   entry['bytes_sha256']=sha(data)
   if resource:entry['status']='ignored_resource_metadata';ignored.append(name);continue
   if pure.suffix.lower()!='.txt':entry['status']='ignored_nontext';ignored.append(name);continue
   if sid not in trainset:entry['status']='unexpected_train_archive_textID';unexpected.append(sid);continue
   original=data.decode('utf-8-sig');normalized=' '.join(original.split());transcript={'member':name,'raw_bytes':len(data),'raw_sha256':sha(data),'raw_encoding':'UTF-8 (optionalBOM)','original_text':original,'normalized_text':normalized,'normalization':'Unicode whitespace collapse/strip only;case andpunctuationpreserved','normalized_sha256':sha(normalized.encode('utf8')),'crc32':format(member.CRC,'08x')}
   if sid in texts:
    same=texts[sid]['raw_sha256']==transcript['raw_sha256'];duplicates.append({'id':sid,'first':texts[sid]['member'],'duplicate':name,'raw_bytes_identical':same,'normalized_identical':texts[sid]['normalized_text']==normalized});assert same,'Conflictingduplicatetranscriptbasename'
   else:texts[sid]=transcript
   if not normalized:empty.append(sid)
   entry['status']='mapped_train_text'
 missing=sorted(trainset-set(texts));assert not missing,'Missingofficialtraintranscripts';assert not empty,'Emptyofficialtraintranscripts'
 manifest=H/'train_reference_manifest.jsonl';localmissing=[];localids=set();mapped=0
 with manifest.open('w') as f:
  for sid in ids['train_ids']:
   splits=[s for s in ['train','valid'] if sid in inventory[s]['clean']['ids'] and sid in inventory[s]['noisy']['ids']];assert len(splits)==1;split=splits[0];text=texts[sid];record={'id':sid,'speaker':sid.split('_')[0],'publisher_partition':'trainset_28spk','local_partition':split,'clean_path':str(V/'16k'/split/'clean'/(sid+'.wav')),'noisy_path':str(V/'16k'/split/'noisy'/(sid+'.wav')),'clean_header':{k:int(header_map[split,'clean',sid][k]) for k in ['samplerate','channels','frames']},'noisy_header':{k:int(header_map[split,'noisy',sid][k]) for k in ['samplerate','channels','frames']},'clean_subtype':header_map[split,'clean',sid]['subtype'],'noisy_subtype':header_map[split,'noisy',sid]['subtype'],'transcript':text,'reference_provenance':'PublishertraintextmappedbyexactfilenameID;notASR;spokenalignmentandlocalconversionunverified','pretraining_exposure':'unknown'}
   f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');mapped+=1
 dump('archive_inventory.json',entries)
 result.update(status='completed',publisher_bitstream={'name':require_name,'metadata_size_bytes':metadata_size,'download_size_bytes':len(blob),'sha256':sha(blob),'metadata_checksum_algorithm':checksum_algorithm,'metadata_checksum_value':checksum_value,'official_checksum_verified':checksum_verified},license_cc_by_4_verified=True,zip_members=len(entries),zip_files_crc_verified=crc_verified,zip_uncompressed_bytes=total_uncompressed,ignored_metadata_count=len(ignored),ignored_metadata_members=ignored,unexpected_text_ids=unexpected,duplicate_basename_records=duplicates,empty_transcript_ids=empty,missing_train_ids=missing,mapped_train_reference_rows=mapped,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),reference_coverage={'train_scp_ids':len(trainset),'local_train_rows':sum(sid in inventory['train']['clean']['ids'] for sid in trainset),'local_valid_rows':sum(sid in inventory['valid']['clean']['ids'] for sid in trainset)},no_test_text_or_audio_content_opened=True)
except BaseException as exc:
 import traceback
 result.update(status='failed',failures=[{'error':repr(exc),'traceback':traceback.format_exc()}]);traceback.print_exc()
result.update(retrieval_finished_utc=now(),retrieval_records=records,cumulative_download_bytes=transferred,network_and_mapping_wall_seconds=time.monotonic()-T,local_inventory_metrics_sha256=sha((H/'preparation_metrics.json').read_bytes()),source_locks_before_after=read(H/'preparation_metrics.json')['source_locks_before_after'],driver_sha256=sha(Path(__file__).read_bytes()))
assert all(sha(Path(p).read_bytes())==h for p,h in result['source_locks_before_after'].items());result['all_original_local_source_locks_unchanged']=True
result['artifact_bytes_before_final_report']=sum(p.stat().st_size for p in H.rglob('*') if p.is_file());dump('metrics.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['retrieval_records','source_locks_before_after','ignored_metadata_members']},indent=2))
if result['status']!='completed':raise SystemExit(1)
