"""CPU lifecycle and source IO only; no selection/mask/statistical business functions."""
from pathlib import Path
import hashlib,importlib.abc,io,json,os,signal,sys,time,wave
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def size():return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
def verify(locks):
    for p,r in locks.items():
        if Path(p).stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise RuntimeError('Changed source '+str(p))
def start(kind,cfg,events):
    marker=ROOT/('STARTED.json' if kind=='main' else 'VALIDATION_STARTED.json')
    if marker.exists():raise RuntimeError('Existing marker; no retry')
    begin=time.monotonic()
    def hard(*args):
        try:write(ROOT/(kind+'_hard_timeout.json'),dict(status='failed',reason='absolute120sdeadline',pid=os.getpid()))
        finally:os._exit(124)
    def soft(*args):raise TimeoutError('118s work deadline')
    signal.signal(signal.SIGALRM,soft);signal.setitimer(signal.ITIMER_REAL,118)
    with marker.open('x') as f:json.dump(dict(pid=os.getpid(),kind=kind),f)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] in {'torch','whisper','soundfile','torchaudio','transformers','pystoi','pesq','librosa'}:
                events.append(dict(kind='forbidden_import',name=name));raise RuntimeError('Forbidden model/audio import')
    sys.meta_path.insert(0,Block());allowed=set(cfg['inherited_audio_locks'])
    def audit(event,args):
        if event.startswith('socket.') or event in ['subprocess.Popen','os.system','os.posix_spawn','os.fork']:
            events.append(dict(kind='forbidden_network_process',event=event));raise RuntimeError('Forbidden network/process')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                if not p.is_relative_to(ROOT.resolve()) or p.suffix.lower() in ['.wav','.npy','.pcm','.f32le','.bin']:events.append(dict(kind='forbidden_write',path=str(p)));raise RuntimeError('Output source/audio/payload write forbidden')
            elif p.suffix.lower()=='.wav' or str(p).startswith('/data/datasets/'):
                if str(p) not in allowed:events.append(dict(kind='forbidden_read',path=str(p)));raise RuntimeError('Unapproved waveform')
    sys.addaudithook(audit)
    def check():
        if time.monotonic()-begin>=118 or size()>=cfg['budgets']['artifact_bytes']-262144:raise RuntimeError('Work time/byte reserve')
        if events:raise RuntimeError('Forbidden attempted operation, even if caught')
    def finalizing():signal.signal(signal.SIGALRM,hard);signal.setitimer(signal.ITIMER_REAL,max(.000001,begin+120-time.monotonic()))
    def finish():
        if time.monotonic()-begin>120 or size()>cfg['budgets']['artifact_bytes']:raise RuntimeError('Final120s/8MiB exceeded')
        signal.setitimer(signal.ITIMER_REAL,0)
    return begin,check,finalizing,finish

def decode(p,lock,counters,independent=False):
    import numpy as np,struct
    p=Path(p);role=p.parent.name;anchor=Path('/data/datasets/vb/voicebank_demand/16k/train')/role
    if role not in ['clean','noisy'] or p.parent!=anchor or p.resolve().parent!=anchor or p.is_symlink():raise RuntimeError('Source path escape')
    if p.stat().st_size!=lock['bytes'] or sha(p)!=lock['sha256']:raise RuntimeError('Audio source lock mismatch')
    b=p.read_bytes() # full original WAV retained only in memory for identity/sham proof
    array,payload=decode_container(b,lock['header']['frames'],independent)
    counters['pcm_file_decodes']+=1;counters['pcm_samples_decoded']+=len(payload)//2;counters['pcm_payload_bytes_decoded']+=len(payload);counters['decode_precheck_wav_hash_bytes']+=len(b);counters['identity_original_wav_bytes_read']+=len(b)
    if counters['pcm_file_decodes']>24:raise RuntimeError('24 decode cap')
    return array,b

def decode_container(b,frames,independent=False):
    import numpy as np,struct
    with wave.open(io.BytesIO(b),'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype(),w.getnframes())!=(16000,1,2,'NONE',frames):raise RuntimeError('Header mismatch')
        payload=w.readframes(w.getnframes())
    if len(payload)!=frames*2:raise RuntimeError('Truncated PCM')
    array=np.asarray([x[0]/32768 for x in struct.iter_unpack('<h',payload)],dtype=np.float64) if independent else np.frombuffer(payload,dtype='<i2').astype(np.float64)/32768
    if not np.array_equal(np.rint(array*32768).astype('<i2'),np.frombuffer(payload,dtype='<i2')):raise RuntimeError('Original PCM roundtrip mismatch')
    return array,payload

def execute(kind,compute,compare=None):
    cfg=read(ROOT/'config.yaml');prep=read(ROOT/'preparation_manifest.json');verify(prep['preparation_locks']);events=[];begin,check,finalizing,finish=start(kind,cfg,events)
    d=dict(status='running',failures=[],pairs=[],pcm_file_decodes=0,pcm_samples_decoded=0,pcm_payload_bytes_decoded=0,decode_precheck_wav_hash_bytes=0,identity_original_wav_bytes_read=0,forbidden_attempts=events,network_bytes=0,network_accounting='all sockets refused, not packet capture',model_calls=0,new_audio_files=0,gates=cfg['gates'])
    original_metrics_sha=sha(ROOT/'metrics.json') if kind!='main' else None
    try:
        verify(cfg['source_record_locks']);reg=read(cfg['registry'])
        if sha(cfg['registry'])!=cfg['registry_sha256'] or len(reg['records'])!=12 or [r['id'] for r in reg['records']]!=cfg['ids']:raise RuntimeError('Registry scope mismatch')
        stored=read(ROOT/'metrics.json') if kind!='main' else None
        if stored is not None and (stored['status']!='completed' or len(stored['pairs'])!=12):raise RuntimeError('Main incomplete')
        if any((ROOT/n).exists() for n in ['main_hard_timeout.json','validation_hard_timeout.json']):raise RuntimeError('Prior hard timeout')
        for index,r in enumerate(reg['records']):
            check();clean,cb=decode(r['files']['clean']['path'],cfg['inherited_audio_locks'][r['files']['clean']['path']],d,kind!='main');noisy,xb=decode(r['files']['noisy']['path'],cfg['inherited_audio_locks'][r['files']['noisy']['path']],d,kind!='main');result=compute(clean,noisy,cb,xb);result['id']=r['id']
            if len(result['events'])!=2 or len(result['operations'])!=8:raise RuntimeError('Missing event/operation slot')
            if stored is not None:compare(result,stored['pairs'][index])
            d['pairs'].append(result) if kind=='main' else None
        check()
        if d['pcm_file_decodes']!=24:raise RuntimeError('Incomplete24decode')
        d.update(status='completed' if kind=='main' else 'passed',target_slots=24,operation_slots=96)
        if kind!='main':d.pop('pairs');d['protected_metrics_sha256']=original_metrics_sha
    except BaseException as e:
        import traceback
        d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    finally:finalizing()
    try:
        if events:raise RuntimeError('Forbidden attempted operation')
        verify(cfg['source_record_locks']);verify(cfg['inherited_audio_locks']);verify(prep['preparation_locks'])
        if kind!='main' and sha(ROOT/'metrics.json')!=original_metrics_sha:raise RuntimeError('Original metrics changed')
    except BaseException as e:d['status']='failed';d['failures'].append(dict(final=repr(e)))
    d.update(seconds=time.monotonic()-begin,source_postcheck_hash_bytes=sum(r['bytes'] for r in cfg['inherited_audio_locks'].values()),pid=os.getpid())
    write(ROOT/('metrics.json' if kind=='main' else 'independent_validation.json'),d)
    if kind=='main':(ROOT/'experiment_report.md').write_text('# I1b technical operation integrity: '+d['status']+'\n\nAll24eventslots/96operations retained including unavailable/unsafe. Automatic spectral candidates are not phoneme truth; constant Hann spectral leakage is not evidence of a true event. Shared X-S residual may contain speech. Positioncontrol can contain speech, twofamilies/sham do not prove naturalness or remove detectable edit artifacts. No waveform output/classifier/model/training; all scientific/noise/split gates remainfalse. Main and independent exact rules, full candidates/rankings/float&PCM checks in metrics.\n')
    finish();print(json.dumps(dict(status=d['status'],seconds=d['seconds'])));return 0 if d['status'] in ['completed','passed'] else 1
