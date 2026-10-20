import os,sys,json,time,hashlib,collections,subprocess,signal,zipfile
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent; R=B.parent.parent
S=R/'experiments/p8_voicebank_content_reference_audit_20261019/attempt02/train_reference_manifest.jsonl'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(n,x): (B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def git(): return subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=R).decode()
def total():return sum(p.stat().st_size for p in B.rglob('*') if p.is_file())
if (B/'METADATA_STARTED').exists() or (B/'metrics.json').exists():raise SystemExit('existing marker/results: stop')
start=time.monotonic();t=datetime.now(timezone.utc).isoformat();fd=os.open(B/'METADATA_STARTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644);os.write(fd,(t+'\n').encode());os.close(fd)
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60s budget')));signal.alarm(60)
protected=[S,B/'config.yaml',B/'metadata_plan.md',R/'experiments/p8_voicebank_native_content_pilot_20261019/attempt02/metrics.json',R/'experiments/p8_vb_caseB_opening_existing_audit_20261019/metrics.json']
pre={str(p):sha(p) for p in protected};gb=git();write('git_pre.json',{'porcelain':gb})
try:
 assert pre[str(S)]=='7f14a16a20880031932ab4ce7f7d3682c1b16cc0cc2eb8cd141e49ad7de087bc'
 rows=[json.loads(l) for l in S.read_text().splitlines()];groups=collections.defaultdict(list)
 for r in rows:
  h=r['clean_header'];n=int(r['id'].split('_')[1])
  if r['local_partition']=='train' and r['speaker'] not in ('p226','p287') and n>=26 and h['sample_rate']==16000 and h['channels']==1 and 48000<=h['frames']<=80000:groups[r['speaker']].append(r)
 speakers=sorted(s for s,v in groups.items() if len(v)>=2)[:4];assert len(speakers)==4
 selected=[r for s in speakers for r in sorted(groups[s],key=lambda r:int(r['id'].split('_')[1]))[:2]];ids=[r['id'] for r in selected]
 # Enumerate file paths for all artifacts; inspect only bounded textual records, never audio samples.
 suffixes={'.json','.jsonl','.yaml','.yml','.md','.txt','.csv','.tsv','.log','.scp','.py','.html','.js'}
 hits=[];skips=[];scanned=0;readbytes=0;pathhits=[]
 for root in [R/'experiments',R/'research_memos',R/'experiment_logs']:
  for d,dirs,files in os.walk(root):
   if Path(d)==B:dirs[:]=[];continue
   dirs[:]=[x for x in dirs if x not in ('__pycache__','.git')]
   for n in files:
    p=Path(d)/n;rel=str(p.relative_to(R));matched=[i for i in ids if i in rel]
    if matched:pathhits.append({'path':rel,'ids':matched})
    if p.is_symlink():skips.append({'path':rel,'reason':'symlink; path inspected, not followed'});continue
    if p.suffix.lower() not in suffixes:continue
    size=p.stat().st_size
    if size>8*1024*1024:skips.append({'path':rel,'reason':'text exceeds 8MiB','bytes':size});continue
    b=p.read_bytes();readbytes+=len(b);scanned+=1
    if any(i.encode() in b for i in ids):
     lines=b.decode('utf-8',errors='replace').splitlines();matches=[]
     for j,l in enumerate(lines,1):
      ii=[i for i in ids if i in l]
      if ii:matches.append({'line':j,'ids':ii,'excerpt':l[:600]})
     hits.append({'path':rel,'sha256':hashlib.sha256(b).hexdigest(),'bytes':size,'matches':matches[:24],'matching_lines':len(matches)})
 write('use_history_scan.json',{'scope':['experiments','research_memos','experiment_logs'],'policy':'all paths; text suffix allowlist <=8MiB; no symlink traversal; audio/binary contents never read by history scan','text_files_scanned':scanned,'text_bytes_scanned':readbytes,'path_hits':pathhits,'text_hits':hits,'skips':skips,'limitations':'Search is current project scoped, not proof of global/model-pretraining nonexposure. Root must review hit classifications before any scientific stage.'})
 # History findings are classified separately after evidence inspection; do not claim no prior use merely from grep.
 with zipfile.ZipFile(S.parent/'trainset_28spk_txt.zip') as z:
  registry=[]
  for r in selected:
   r=dict(r);p=Path(r['clean_path']);assert p.is_file();st=p.stat();raw=z.read(r['transcript']['archive_entry']);assert hashlib.sha256(raw).hexdigest()==r['transcript']['raw_text_SHA256'];assert raw.decode('utf-8-sig')==r['transcript']['original_text']
   r['duration_seconds']=r['clean_header']['frames']/16000;r['local_clean_file']={'exists':True,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha(p),'inspection':'stat and raw file bytes only; no audio decode/header reparsing'};r['project_exposure_review']='pending explicit classification of current project scan';registry.append(r)
 write('prospective_registry.json',{'phase':'metadata_only','source_manifest':str(S),'source_manifest_sha256':sha(S),'selected_ids':ids,'records':registry,'model_stage_authorized':False,'claim_limit':'train-split exploratory metadata selection; no spoken-alignment or pretraining-independence claim'})
 post={str(p):sha(p) for p in protected};assert pre==post
 ga=git();write('git_post.json',{'porcelain':ga})
 # Own newly-created paths excluded; all other git entries must remain identical.
 prefix='experiments/'+B.name+'/'
 filt=lambda x:[l for l in x.splitlines() if prefix not in l]
 assert filt(gb)==filt(ga)
 write('metrics.json',{'status':'metadata_registered_pending_history_review','elapsed_seconds':time.monotonic()-start,'selected_ids':ids,'speaker_count':4,'utterance_count':8,'source_locks_pre':pre,'source_locks_post':post,'git_other_entries_unchanged':True,'network_calls':0,'audio_decode':0,'resample':0,'mixing':0,'models':0,'ASR':0,'training':0,'retries':0,'model_stage_authorized':False})
 write('execution_provenance.json',{'command':sys.executable+' -B '+str(Path(__file__).resolve()),'pid':os.getpid(),'started_utc':t,'ended_utc':datetime.now(timezone.utc).isoformat(),'thread_id':os.environ.get('CODEX_THREAD_ID'),'provider_verified':'codex_tools_compat via readonly SQLite before execution','budget_seconds':60,'artifact_limit_bytes':2097152,'artifact_bytes_at_execution_end':total(),'script_sha256':sha(Path(__file__))})
 assert total()<2097152
 print(json.dumps({'status':'metadata registered pending history review','ids':ids,'elapsed':time.monotonic()-start,'bytes':total(),'hits':len(hits),'pathhits':len(pathhits),'skips':len(skips)}))
except BaseException as e:
 write('failure.json',{'type':type(e).__name__,'message':str(e),'elapsed':time.monotonic()-start});raise
finally:signal.alarm(0)
