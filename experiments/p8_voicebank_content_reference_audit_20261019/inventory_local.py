"""Local VoiceBank metadata/header inventory; no test audio opens or network."""
from pathlib import Path
import os,json,hashlib,time,collections,csv,gzip,subprocess,re
import soundfile as sf
H=Path(__file__).resolve().parent;R=H.parent.parent;V=Path('/data/datasets/vb/voicebank_demand');T=time.monotonic();CAP=2000000
assert not (H/'preparation_metrics.json').exists(),'Refuse repeated completed inventory'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,d):(H/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def budget():
 assert time.monotonic()-T<120,'CPU wall cap'
 assert sum(p.stat().st_size for p in H.rglob('*') if p.is_file())<CAP,'Artifact cap'
def scp(path):
 raw=path.read_text().splitlines();ids=[r.strip().split()[0] for r in raw if r.strip()];counts=collections.Counter(ids)
 return {'path':str(path),'sha256':sha(path),'line_count':len(raw),'nonempty_ids':len(ids),'unique_ids':len(counts),'duplicate_ids':{i:n for i,n in counts.items() if n>1},'speaker_counts':dict(sorted(collections.Counter(i.split('_')[0] for i in ids).items())),'speakers':sorted({i.split('_')[0] for i in ids}),'malformed_id_count':sum(re.fullmatch(r'p\d+_\d+',i) is None for i in ids)},ids
cfg={'experiment_id':H.name,'status':'local_inventory_only','dataset_root':str(V),'training':False,'model_calls':False,'network':False,'waveform_decode':False,'test_access':'SCP IDs and directory filenames/count only; no test audio header/decode or transcript content','train_access':'all train/valid-source WAV headers only','budget_seconds':120,'artifact_cap_bytes':CAP,'external_metadata_status':'Root separately checking publisher license/download URL; no network or download authorized here','external_source_parent_reported':'https://datashare.ed.ac.uk/handle/10283/2791 -> item6ed35425-bf14-4d2b-93a1-0a4984952757; trainset_28spk_txt.zip6.19MB,testset_txt.zip429.96KB; not independently retrieved by this script','pretraining_exposure':'unknown for enhancement and ASR/SSL backbones; potential project-independent pilot is not strict pretraining holdout','speaker_selection':'none; no model-based split decisions'};dump('config.yaml',cfg)
gitbefore=subprocess.check_output(['git','status','--short'],cwd=R,text=True);(H/'git_status_before.txt').write_text(gitbefore)
locks={str(V/n):sha(V/n) for n in ['train.scp','test.scp','voicebank_prepare.py']};tr,trainids=scp(V/'train.scp');te,testids=scp(V/'test.scp');ts=set(trainids);es=set(testids)
layout=[];metadata=[];filelists={};header_counts={};errors=[];header_mismatch=[];pair_mismatch=[];duration={};train_header_rows=0
for root,dirs,files in os.walk(V):
 budget();root=Path(root);rel=str(root.relative_to(V));layout.append({'directory':rel,'immediate_subdirectories':sorted(dirs),'immediate_file_count':len(files),'wav_filename_count':sum(n.lower().endswith('.wav') for n in files)})
 for name in files:
  if Path(name).suffix.lower() in ['.txt','.json','.jsonl','.tsv','.csv','.zip','.scp','.yaml','.yml']:
   path=root/name;metadata.append({'path':str(path),'suffix':path.suffix.lower(),'bytes':path.stat().st_size,'contents_opened':path in [V/'train.scp',V/'test.scp']})
for split in ['train','valid','test']:
 filelists[split]={}
 for arm in ['clean','noisy']:
  folder=V/'16k'/split/arm;files=sorted(folder.glob('*.wav'));ids=[p.stem for p in files];filelists[split][arm]={'directory':str(folder),'wav_count':len(files),'unique_ids':len(set(ids)),'ids':ids,'speakers':sorted({i.split('_')[0] for i in ids}),'speaker_counts':dict(sorted(collections.Counter(i.split('_')[0] for i in ids).items()))}
# Test filenames are listed above only. No sf.info or other file open on test audio.
with gzip.open(H/'train_source_headers.csv.gz','wt',newline='') as gz:
 writer=csv.writer(gz);writer.writerow(['local_split','arm','id','samplerate','channels','frames','duration_seconds','subtype'])
 headers={}
 for split in ['train','valid']:
  headers[split]={}
  for arm in ['clean','noisy']:
   rows=[];counter=collections.Counter();headers[split][arm]={}
   for sid in filelists[split][arm]['ids']:
    if train_header_rows%200==0:budget()
    path=V/'16k'/split/arm/(sid+'.wav')
    try:
     info=sf.info(path);record=(info.samplerate,info.channels,info.frames,info.subtype);headers[split][arm][sid]=record;counter[str(record[0])+'Hz/'+str(record[1])+'ch/'+record[3]]+=1;rows.append(info.duration);writer.writerow([split,arm,sid,info.samplerate,info.channels,info.frames,format(info.duration,'.9g'),info.subtype]);train_header_rows+=1
     if info.samplerate!=16000 or info.channels!=1:header_mismatch.append({'path':str(path),'rate':info.samplerate,'channels':info.channels})
    except Exception as exc:errors.append({'path':str(path),'error':str(exc)})
   header_counts[split+'/'+arm]=dict(counter);duration[split+'/'+arm]={'files':len(rows),'sum_seconds':sum(rows),'min_seconds':min(rows) if rows else None,'max_seconds':max(rows) if rows else None,'mean_seconds':sum(rows)/len(rows) if rows else None}
  common=set(headers[split]['clean'])&set(headers[split]['noisy'])
  for sid in sorted(common):
   c=headers[split]['clean'][sid];n=headers[split]['noisy'][sid]
   if c[:3]!=n[:3]:pair_mismatch.append({'split':split,'id':sid,'clean_rate_channels_frames':c[:3],'noisy_rate_channels_frames':n[:3]})
coverage={}
for split in filelists:
 c=set(filelists[split]['clean']['ids']);n=set(filelists[split]['noisy']['ids']);coverage[split]={'clean_noisy_id_intersection':len(c&n),'clean_only_ids':sorted(c-n),'noisy_only_ids':sorted(n-c),'IDs_in_train_scp':len(c&ts),'IDs_in_test_scp':len(c&es)}
uniontrain=set(filelists['train']['clean']['ids'])|set(filelists['valid']['clean']['ids']);unionnoisy=set(filelists['train']['noisy']['ids'])|set(filelists['valid']['noisy']['ids']);actualtest=set(filelists['test']['clean']['ids']);localoverlap=set(filelists['train']['clean']['ids'])&set(filelists['valid']['clean']['ids']);trainsp=set(filelists['train']['clean']['speakers']);validsp=set(filelists['valid']['clean']['speakers'])
code=(V/'voicebank_prepare.py').read_text();matches=[]
for lineno,line in enumerate(code.splitlines(),1):
 if any(w.lower() in line.lower() for w in ['Resample','48000','16000','signal.view','torchaudio.save','remove_punctuation','strip().upper','Remove duplicate phones','phones =','valid_speakers =','wav_lst_valid','crop','trim','normalize','filename[-12:]','read_audio']):matches.append({'line':lineno,'text':line})
dump('prepare_code_matches.json',matches);dump('filename_inventory.json',filelists);dump('scp_id_inventory.json',{'train_ids':trainids,'test_ids':testids})
assert all(sha(p)==h for p,h in locks.items());after=subprocess.check_output(['git','status','--short'],cwd=R,text=True);(H/'git_status_after.txt').write_text(after)
result={'status':'local_inventory_completed','train_scp':tr,'test_scp':te,'scp_intersections':{'utterance_ids':sorted(ts&es),'speaker_ids':sorted(set(tr['speakers'])&set(te['speakers']))},'layout':layout,'coverage':coverage,'train_source_union_vs_scp':{'union_clean_count':len(uniontrain),'union_noisy_count':len(unionnoisy),'missing_clean_train_scp_ids':sorted(ts-uniontrain),'extra_clean_ids_outside_train_scp':sorted(uniontrain-ts),'missing_noisy_train_scp_ids':sorted(ts-unionnoisy),'extra_noisy_ids_outside_train_scp':sorted(unionnoisy-ts)},'test_filename_only_vs_scp':{'clean_count':len(actualtest),'clean_missing_scp_ids':sorted(es-actualtest),'clean_extra_ids':sorted(actualtest-es),'noisy_missing_scp_ids':sorted(es-set(filelists['test']['noisy']['ids'])),'noisy_extra_ids':sorted(set(filelists['test']['noisy']['ids'])-es)},'local_train_valid_id_intersection':sorted(localoverlap),'local_train_valid_speaker_intersection':sorted(trainsp&validsp),'local_train_speakers':sorted(trainsp),'local_valid_speakers':sorted(validsp),'header_files_read':train_header_rows,'test_audio_header_reads':0,'waveform_decodes':0,'header_counts':header_counts,'durations':duration,'header_errors':errors,'unexpected_rate_channels':header_mismatch,'paired_header_mismatches':pair_mismatch,'metadata_filename_candidates':metadata,'transcript_candidates_exist':any(Path(r['path']).suffix.lower() not in ['.scp'] for r in metadata),'source_locks_before_after':locks,'source_files_unchanged':True,'prep_script_execution_provenance':'unknown; presence is not evidence it produced local audio; local layout differs from expected original preparation layout','model_calls':0,'network_requests':0,'training_calls':0,'audio_outputs':0,'git_status_unchanged':gitbefore==after,'wall_seconds':time.monotonic()-T,'driver_sha256':sha(__file__)};dump('preparation_metrics.json',result);budget()
print(json.dumps({'train_scp':tr,'test_scp':te,'coverage':coverage,'local_train_valid_speaker_intersection':sorted(trainsp&validsp),'header_reads':train_header_rows,'header_errors':len(errors),'header_pair_mismatches':len(pair_mismatch),'metadata_files':metadata,'duration':duration,'seconds':result['wall_seconds']},indent=2))
