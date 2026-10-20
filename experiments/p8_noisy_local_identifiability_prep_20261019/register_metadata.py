"""I0 metadata registration. Import is inert; real Worker execution requires Root review.
Never reads audio frames. Header inspection is wave.getparams only."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,os,re,signal,sys,time,traceback,wave
OUT=Path(__file__).resolve().parent

def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def bytes_used(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def check_budget(start,cfg,root):
    require(time.monotonic()-start<cfg['budgets']['seconds']-2,'I0 execution deadline')
    require(bytes_used(root)<cfg['budgets']['artifact_bytes']-65536,'I0 artifact reserve')
def verify_lock(p,expected):require(Path(p).is_file() and sha(p)==expected,'Missing source or SHA mismatch '+str(p))

def derive(manifest,cfg):
    """Metadata-only eligibility; no filesystem audio access."""
    speakers={};seen=set()
    with Path(manifest).open('rb') as f:
        for number,line in enumerate(f,1):
            r=json.loads(line);sid=r.get('id');require(isinstance(sid,str) and sid not in seen,'Duplicate/malformed source ID');seen.add(sid)
            require(isinstance(r.get('local_partition'),str),'Missing partition')
            if r['local_partition']!=cfg['local_partition']:continue
            match=re.fullmatch(r'(p[0-9]+)_([0-9]+)',sid);require(match is not None and r.get('speaker')==match[1],'Invalid ID/speaker')
            speaker=match[1]
            if speaker in cfg['excluded_speakers'] or int(match[2])<cfg['minimum_utterance_number']:continue
            headers=[]
            for key in ['clean_header','noisy_header']:
                h=r.get(key);require(isinstance(h,dict),'Missing header schema')
                require(type(h.get('sample_rate')) is int and h['sample_rate']>0 and type(h.get('frames')) is int and h['frames']>0,'Bad rate/frame schema')
                headers.append(h)
            h,k=headers
            lo,hi=cfg['duration_seconds_inclusive']
            if not lo<=h['frames']/h['sample_rate']<=hi:continue
            require(h==k and h['sample_rate']==16000 and h.get('channels')==1 and h.get('subtype')=='PCM_16','Eligible metadata header mismatch')
            for role in ['clean','noisy']:
                p=Path(r.get(role+'_path',''));require(p.is_absolute() and p.name==sid+'.wav' and p.parent.name==role and p.parent.parent.name=='train','Bad local train source path')
            require(isinstance(r.get('transcript'),dict) and isinstance(r['transcript'].get('original_text'),str),'Missing reference schema')
            speakers.setdefault(speaker,[]).append(dict(source_line_number=number,source_line_sha256=hashlib.sha256(line).hexdigest(),source_row=r))
    eligible=sorted(s for s,rs in speakers.items() if len(rs)>=cfg['utterances_per_speaker'])
    require(len(eligible)>=cfg['speaker_count'],'Insufficient eligible speakers; no relaxation')
    chosen=[]
    for s in eligible[:cfg['speaker_count']]:
        chosen.extend(sorted(speakers[s],key=lambda x:(int(x['source_row']['id'].rsplit('_',1)[1]),x['source_row']['id']))[:cfg['utterances_per_speaker']])
    require(len(chosen)==cfg['expected_pairs'],'Insufficient expected pairs')
    return chosen

def approved_audio_path(raw,sid,role):
    require(role in ['clean','noisy'],'Unsupported audio role')
    p=Path(raw);anchor=Path('/data/datasets/vb/voicebank_demand/16k/train')/role
    require(p.name==sid+'.wav' and p.parent==anchor,'Unexpected frozen train audio path')
    require(not p.is_symlink() and p.resolve().parent==anchor and p.resolve().name==sid+'.wav','Resolved audio path escape')
    return p.resolve()

def header(p):
    with wave.open(str(p),'rb') as w:
        require(w.getcomptype()=='NONE' and (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000),'Actual non PCM16 mono16k header')
        require(Path(p).stat().st_size>=44+w.getnframes()*2,'Truncated declared PCM payload')
        return dict(sample_rate=w.getframerate(),channels=w.getnchannels(),frames=w.getnframes(),subtype='PCM_16')

def scan(cfg,allow):
    """Bounded provenance clues only; no automatic noise-identity inference."""
    policy=cfg['noise_metadata_scan'];rows=[];total=0;count=0;skips=[]
    forbidden=set(policy['exclude_path_components'])
    for value in cfg['noise_metadata_roots']:
        root=Path(value)
        if not root.exists():skips.append(dict(path=str(root),reason='root_missing'));continue
        require(not root.is_symlink() and root.is_dir(),'Unsafe metadata root')
        def walk(folder,depth):
            nonlocal total,count
            for p in sorted(folder.iterdir(),key=lambda p:p.name):
                if p.is_symlink():skips.append(dict(path=str(p),reason='symlink_not_followed'));continue
                require(p.resolve().is_relative_to(root.resolve()),'Metadata path escape')
                if any(c.lower() in forbidden or c.lower().startswith(('test','valid')) for c in p.relative_to(root).parts):continue
                if p.is_dir():
                    if depth<policy['max_depth']:walk(p,depth+1)
                    continue
                if not p.is_file() or p.suffix.lower() not in policy['allowed_suffixes']:continue
                if count>=policy['max_candidate_files']:skips.append(dict(path=str(p),reason='candidate_limit'));return
                count+=1;n=p.stat().st_size
                if total+n>policy['max_total_read_bytes']:skips.append(dict(path=str(p),reason='read_byte_limit'));continue
                allow.add(str(p.resolve()));b=p.read_bytes();require(len(b)==n,'Metadata changed during read');total+=len(b)
                try:text=b.decode('utf-8-sig')
                except UnicodeError:text=None
                rows.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),utf8_excerpt=text[:2000] if text is not None else None,interpretation='unreviewed provenance clue; never auto-assigned recording identity'))
        walk(root,0)
    return dict(files=rows,skips=skips,total_read_bytes=total,candidate_files_seen=count,noise_identity_resolved=False)

def guard(out,allowed,events):
    class NoModels(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] in {'torch','numpy','whisper','torchaudio','soundfile','transformers','pystoi','pesq'}:raise RuntimeError('I0 forbidden model/audio import '+name)
    sys.meta_path.insert(0,NoModels())
    def audit(event,args):
        if event.startswith('socket.') or event in {'subprocess.Popen','os.system','os.posix_spawn','os.fork'}:
            events.append(event);raise RuntimeError('I0 forbidden network/process')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:require(p.is_relative_to(out.resolve()),'I0 write outside output forbidden')
            elif str(p).startswith('/data/datasets/') or p.suffix.lower()=='.wav':require(str(p) in allowed,'I0 unapproved source read')
    sys.addaudithook(audit)

def run():
    cfg=read(OUT/'config.yaml');prep=read(OUT/'preparation_manifest.json')
    for n,h in prep['files'].items():verify_lock(OUT/n,h)
    for n in ['STARTED.json','registry.json','metrics.json','experiment_report.md']:require(not (OUT/n).exists(),'Existing registration artifact '+n)
    start=time.monotonic();events=[];allowed={str(Path(cfg['source_manifest']).resolve())}
    # One process, alarm finalizes failure before hard kill. No subprocess or retries.
    def alarm(*args):raise TimeoutError('I0 58s execution budget')
    def hard_alarm(*args):
        try:
            with (OUT/'hard_timeout_failure.json').open('w') as f:json.dump(dict(status='failed',reason='absolute60sdeadline',pid=os.getpid()),f)
        finally:os._exit(124)
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,cfg['budgets']['seconds']-2)
    with (OUT/'STARTED.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),preparation_manifest_sha256=sha(OUT/'preparation_manifest.json')),f)
    d=dict(status='running',failures=[],network_attempts=events,network_bytes=0,audio_sample_decodes=0,model_imports=0,training_calls=0,noise_disjoint_ready=False,split_frozen=False,wav_header_checks=0,full_wav_hash_bytes=0)
    try:
        guard(OUT,allowed,events);verify_lock(cfg['source_manifest'],cfg['source_manifest_sha256']);selected=derive(cfg['source_manifest'],cfg);rows=[];hashed=0
        for item in selected:
            check_budget(start,cfg,OUT);r=item['source_row'];files={}
            for role in ['clean','noisy']:
                p=approved_audio_path(r[role+'_path'],r['id'],role);allowed.add(str(p));actual=header(p);d['wav_header_checks']+=1;require(actual==r[role+'_header'],'Current header differs frozen metadata');filebytes=p.stat().st_size;files[role]=dict(path=str(p),bytes=filebytes,sha256=sha(p),header=actual);hashed+=1;d['full_wav_hash_bytes']+=filebytes
            require(files['clean']['header']==files['noisy']['header'],'Pair length/header mismatch')
            rows.append(dict(id=r['id'],speaker=r['speaker'],source_line_number=item['source_line_number'],source_line_sha256=item['source_line_sha256'],reference=r['transcript'],files=files,noise_recording_id='unknown',noise_provenance_status='unresolved_no_identity_claim',alignment='unverified',gain='unverified',residual_noise_purity='unknown',project_use_history='six explicitly excluded speakers only; external/anonymized/pretraining unknown'))
        require(hashed==cfg['expected_pairs']*2<=cfg['max_current_audio_files_to_hash'],'Audio hash count limit')
        clues=scan(cfg,allowed);check_budget(start,cfg,OUT)
        registry=dict(phase='I0_metadata_only',source_manifest_sha256=cfg['source_manifest_sha256'],records=rows,noise_metadata_scan=clues,noise_disjoint_ready=False,split_frozen=False,scientific_gate_ready=False,interpretation='Metadata registration is not waveform alignment/additivity/noise-purity or local-identifiability validation')
        write(OUT/'registry.json',registry);d.update(status='completed',pairs=len(rows),speakers=len({r['speaker'] for r in rows}),audio_files_hashed=hashed,registry_sha256=sha(OUT/'registry.json'))
    except BaseException as e:d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    finally:
        signal.signal(signal.SIGALRM,hard_alarm);signal.setitimer(signal.ITIMER_REAL,max(.000001,start+cfg['budgets']['seconds']-time.monotonic()))
    try:
        for n,h in prep['files'].items():verify_lock(OUT/n,h)
        verify_lock(cfg['source_manifest'],cfg['source_manifest_sha256']);require(not events,'Forbidden attempt recorded');require(bytes_used(OUT)<cfg['budgets']['artifact_bytes']-32768,'Final artifact reserve')
    except BaseException as e:d['status']='failed';d['failures'].append(dict(final_check=repr(e)))
    d.update(seconds=time.monotonic()-start,pid=os.getpid(),time_cap=60,byte_cap=cfg['budgets']['artifact_bytes'],actual_audio_samples_read=0)
    if d['seconds']>60:d['status']='failed';d['failures'].append(dict(error='final60s exceeded'))
    write(OUT/'metrics.json',d)
    (OUT/'experiment_report.md').write_text('# I0 metadata registration: '+d['status']+'\n\nPCM sample decodes=0. WAV headers inspected; complete file SHA reads payload bytes (full_wav_hash_bytes), not sample decoding. No models/network/training. Registry preserves unknown noise recording identity, false noise_disjoint_ready, unfrozen split and unverified alignment/gain/noise purity. Header match is not I1 success. Missing/insufficient inputs fail without fallback or retry.\n')
    require(bytes_used(OUT)<=cfg['budgets']['artifact_bytes'],'Final2MiB cap');require(time.monotonic()-start<=60,'Absolute60s final deadline');signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(dict(status=d['status'],seconds=d['seconds'],bytes=bytes_used(OUT))));return 0 if d['status']=='completed' else 1
if __name__=='__main__':
    require(sys.argv[1:]==['--execute-i0-after-root-review'],'No run by default; only approved Worker registration')
    sys.exit(run())
