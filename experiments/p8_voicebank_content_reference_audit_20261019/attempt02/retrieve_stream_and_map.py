"""One durable official TRAIN ZIP transaction, verified archive and CPU ID mapping."""
from pathlib import Path,PurePosixPath
import os,sys,json,time,datetime,hashlib,subprocess,signal,zipfile,zlib,re,gzip,csv,collections,traceback
H=Path(__file__).resolve().parent;P=H.parent;R=P.parent.parent
URL='https://datashare.ed.ac.uk/server/api/core/bitstreams/fb55b5de-c2db-4451-ac0a-134e98011ffa/content';EXPECTED=6489312;MD5='48bfcd5945165c60ab9d2f312f4d6e1c';CAP=30000000
assert not (H/'STARTED.json').exists() and not (H/'metrics.json').exists(),'Refuse existing transaction/result'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def hashfile(p,algo='sha256'):
 h=hashlib.new(algo)
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,d):(H/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def size():return sum(p.stat().st_size for p in H.rglob('*') if p.is_file())
def artifactcap():assert size()<CAP,'30MBadditionalartifactcap'
T=time.monotonic();before={str(p):hashfile(p) for p in P.rglob('*') if p.is_file() and H not in p.parents};local=read(P/'preparation_metrics.json');locks=local['source_locks_before_after'];assert all(hashfile(p)==h for p,h in locks.items())
metadata=read(P/'publisher/train_bitstream_metadata.json');assert metadata['name']=='trainset_28spk_txt.zip' and metadata['sizeBytes']==EXPECTED and metadata['checkSum']['value']==MD5
assert 'creativecommons.org/licenses/by/4.0' in (P/'publisher/license.txt').read_text().lower()
cfg={'experiment_id':P.name+'_attempt02','status':'single_authorized_zip_only_transaction','url':URL,'proxy':'http://127.0.0.1:7890','download_is_resume':False,'previous_partial_recoverable':False,'no_refetch_official_documents':True,'expected_size_bytes':EXPECTED,'expected_official_MD5':MD5,'new_transfer_cap_bytes':7000000,'known_combined_network_cap_bytes':15000000,'prior_known_recorded_bytes':read(P/'metrics.json')['cumulative_download_bytes'],'prior_interrupted_direct_bytes_unknown':True,'network_hard_wall_seconds':420,'additional_artifact_cap_bytes':CAP,'archive_max_total_uncompressed_bytes':60000000,'archive_max_single_member_bytes':1000000,'test_policy':'SCPIDs only;no testtext/archive/audio/content retrieval','audio_decode':False,'models':False,'training':False};dump('config.yaml',cfg);dump('parent_snapshot_hashes.json',before);dump('STARTED.json',{'utc':now(),'parent_failure_status':read(P/'metrics.json')['status']})
partial=H/'trainset_28spk_txt.zip.partial';final=H/'trainset_28spk_txt.zip'
cmd=['/usr/bin/curl','--fail','--location','--connect-timeout','15','--max-time','420','--max-filesize','7000000','--retry','0','--proxy',cfg['proxy'],'--silent','--show-error','--dump-header',str(H/'http_headers.txt'),'--output',str(partial),'--write-out','%{json}',URL]
result={'status':'running','failures':[],'source_url':URL,'started_utc':now(),'curl_argv':cmd,'network_transactions':1,'automatic_retries':0,'models':0,'training_calls':0,'audio_decodes':0,'audio_outputs':0,'test_content_opened':False,'resume_claim':False}
try:
 nstart=time.monotonic();deadline=nstart+420
 with (H/'curl_metadata.json').open('wb') as out,(H/'curl_stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,cwd=R,start_new_session=True);last=-20;supervisor_stop=False
  while child.poll() is None:
   elapsed=time.monotonic()-nstart;n=partial.stat().st_size if partial.exists() else 0
   if elapsed>=420 or n>7000000 or size()>=CAP:
    os.killpg(child.pid,signal.SIGKILL);child.wait();supervisor_stop=True;break
   if elapsed-last>=15:
    dump('transfer_progress.json',{'utc':now(),'elapsed_seconds':elapsed,'durable_partial_bytes':n,'expected_bytes':EXPECTED,'curl_pid':child.pid,'curl_running':True});last=elapsed
   time.sleep(.2)
 result.update(network_wall_seconds=time.monotonic()-nstart,curl_returncode=child.returncode,supervisor_stopped=supervisor_stop,partial_bytes=partial.stat().st_size if partial.exists() else 0)
 dump('transfer_progress.json',{'utc':now(),'elapsed_seconds':result['network_wall_seconds'],'durable_partial_bytes':result['partial_bytes'],'expected_bytes':EXPECTED,'curl_pid':child.pid,'curl_running':False,'returncode':child.returncode})
 if (H/'curl_metadata.json').stat().st_size:
  try:result['curl_metadata']=read(H/'curl_metadata.json')
  except Exception as exc:result['curl_metadata_error']=repr(exc)
 assert child.returncode==0 and not supervisor_stop,'SingleZIPtransactionfailed;partialpreserved,no retry'
 assert partial.stat().st_size==EXPECTED,'UnexpectedcompleteZIPsize'
 actual_md5=hashfile(partial,'md5');actual_sha=hashfile(partial);assert actual_md5==MD5,'OfficialMD5 mismatch'
 ids=read(P/'scp_id_inventory.json');trainset=set(ids['train_ids']);testset=set(ids['test_ids']);entries=[];crcfiles=0;resourcecount=0
 with zipfile.ZipFile(partial) as z:
  members=z.infolist();uncompressed=sum(m.file_size for m in members);assert uncompressed<=60000000,'Unsafeuncompressedtotal'
  for m in members:
   pure=PurePosixPath(m.filename);assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in m.filename and not re.match(r'^[A-Za-z]:',m.filename),'Pathtraversal';assert m.file_size<=1000000,'Oversizedentry';assert not ((m.external_attr>>16)&0o170000)==0o120000,'Symlinkentry'
   resource='__MACOSX' in pure.parts or pure.name.startswith('._') or pure.name=='.DS_Store'
   if not resource and pure.suffix.lower()=='.txt':assert pure.stem not in testset,'UnexpectedtesttextID;refuseread'
   entry={'member':m.filename,'bytes':m.file_size,'compressed_bytes':m.compress_size,'crc32':format(m.CRC,'08x'),'is_directory':m.is_dir(),'resource_metadata':resource}
   if not m.is_dir():
    data=z.read(m);assert zlib.crc32(data)&0xffffffff==m.CRC,'CRCfailed';entry['raw_sha256']=sha(data);crcfiles+=1
   if resource:resourcecount+=1
   entries.append(entry)
 artifactcap();os.replace(partial,final)
 result.update(archive_bytes=EXPECTED,archive_MD5=actual_md5,archive_SHA256=actual_sha,official_MD5_verified=True,all_file_CRCs_verified=True,archive_members=len(entries),archive_file_CRC_count=crcfiles,archive_declared_uncompressed_bytes=uncompressed,resource_metadata_entry_count=resourcecount,ZIP_atomic_rename_after_validation=True)
 dump('archive_entry_inventory.json',entries)
 # No waveform opens. Reuse only prior localfilename/header metadata.
 inv=read(P/'filename_inventory.json');hrows=list(csv.DictReader(gzip.open(P/'train_source_headers.csv.gz','rt')));headers={(r['local_split'],r['arm'],r['id']):r for r in hrows};localsets={s:{arm:set(inv[s][arm]['ids']) for arm in ['clean','noisy']} for s in ['train','valid']}
 textmap={};duplicates=[];decoding=[];unexpected=[];empties=[];ignored=[];textentries=0
 with zipfile.ZipFile(final) as z:
  for entry in entries:
   pure=PurePosixPath(entry['member'])
   if entry['is_directory']:continue
   if entry['resource_metadata'] or pure.suffix.lower()!='.txt':ignored.append(entry['member']);continue
   sid=pure.stem;textentries+=1
   if sid not in trainset:unexpected.append({'id':sid,'member':entry['member']});continue
   data=z.read(entry['member']);assert sha(data)==entry['raw_sha256']
   try:original=data.decode('utf-8-sig')
   except UnicodeDecodeError as exc:decoding.append({'id':sid,'member':entry['member'],'error':str(exc)});continue
   normalized=' '.join(original.split());row={'archive_entry':entry['member'],'raw_text_bytes':len(data),'raw_text_SHA256':sha(data),'raw_encoding':'UTF-8withoptionalBOM','original_text':original,'normalized_text':normalized,'normalized_text_SHA256':sha(normalized.encode('utf8')),'normalization':'Whitespacecollapse/striponly;caseandpunctuationpreserved','CRC32':entry['crc32']}
   if sid in textmap:
    first=textmap[sid];identical=first['raw_text_SHA256']==row['raw_text_SHA256'];duplicates.append({'id':sid,'first_member':first['archive_entry'],'duplicate_member':row['archive_entry'],'raw_bytes_identical':identical,'normalized_identical':first['normalized_text']==normalized})
   else:textmap[sid]=row
   if not normalized:empties.append(sid)
 missing=sorted(trainset-set(textmap));conflicts=[r for r in duplicates if not r['raw_bytes_identical']]
 result.update(archive_nonmetadata_text_entries=textentries,unique_expected_train_text_ids=len(textmap),duplicate_basenames=duplicates,conflicting_duplicates=conflicts,decoding_issues=decoding,empty_transcript_ids=empties,missing_train_ids=missing,unexpected_text_ids=unexpected,ignored_nonlabel_members=ignored)
 assert not missing and not conflicts and not decoding and not empties,'Transcriptmappingeligibilityfailed'
 manifest=H/'train_reference_manifest.jsonl';count=0;splitcounts=collections.Counter();speakers=collections.Counter()
 with manifest.open('w') as f:
  for sid in ids['train_ids']:
   splits=[s for s in ['train','valid'] if sid in localsets[s]['clean'] and sid in localsets[s]['noisy']];assert len(splits)==1,'LocalaudioIDpartitionambiguity';s=splits[0];c=headers[s,'clean',sid];n=headers[s,'noisy',sid];record={'id':sid,'speaker':sid.split('_')[0],'publisher_partition':'trainset_28spk','local_partition':s,'clean_path':str(Path(inv[s]['clean']['directory'])/(sid+'.wav')),'noisy_path':str(Path(inv[s]['noisy']['directory'])/(sid+'.wav')),'clean_header':{'sample_rate':int(c['samplerate']),'channels':int(c['channels']),'frames':int(c['frames']),'subtype':c['subtype']},'noisy_header':{'sample_rate':int(n['samplerate']),'channels':int(n['channels']),'frames':int(n['frames']),'subtype':n['subtype']},'transcript':textmap[sid],'reference_status':'independentpublishertext,filenameIDjoinonly;spokenalignmentandlocalconversionunverified','pretraining_exposure':'unknown'};f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');count+=1;splitcounts[s]+=1;speakers[record['speaker']]+=1
 artifactcap();assert count==11572
 result.update(status='completed',manifest_rows=count,manifest_local_split_counts=dict(splitcounts),manifest_speaker_count=len(speakers),manifest_SHA256=hashfile(manifest),no_ASR_labels=True,spoken_alignment_verified=False,strict_pretraining_holdout=False)
except BaseException as exc:
 result.update(status='failed',failures=[{'error':repr(exc),'traceback':traceback.format_exc()}]);traceback.print_exc()
finally:
 result.update(finished_utc=now(),overall_wall_seconds=time.monotonic()-T,durable_partial_exists=partial.exists(),durable_partial_bytes=partial.stat().st_size if partial.exists() else 0,final_archive_exists=final.exists(),driver_SHA256=hashfile(__file__))
 result['parent_snapshot_files_unchanged']=all(hashfile(p)==h for p,h in before.items());result['original_local_sources_unchanged']=all(hashfile(p)==h for p,h in locks.items());result['prior_known_plus_attempt02_network_bytes']=cfg['prior_known_recorded_bytes']+result.get('curl_metadata',{}).get('size_download',result.get('partial_bytes',0));assert result['prior_known_plus_attempt02_network_bytes']<=15000000
 result['artifact_bytes_before_final_report']=size();dump('metrics.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['curl_metadata','curl_argv','ignored_nonlabel_members']},indent=2))
if result['status']!='completed':raise SystemExit(1)
