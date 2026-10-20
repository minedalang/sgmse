"""Bounded cache IO/lifecycle only. No feature or ridge business functions."""
from pathlib import Path
import hashlib,importlib.abc,io,json,os,signal,sys,time,wave
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def size():return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
def sources(cfg):
    if digest(cfg['input_registry'])!=cfg['input_registry_sha256'] or digest(cfg['model_metrics'])!=cfg['model_metrics_sha256']:raise RuntimeError('Frozen cache JSON mismatch')
    reg=read(cfg['input_registry']);met=read(cfg['model_metrics']);inputroot=Path(cfg['input_registry']).parent;modelroot=Path(cfg['model_metrics']).parent
    if len(reg['cases'])!=32 or len(reg['groups'])!=8 or met['status']!='completed' or len(met['native_cases'])!=32:raise RuntimeError('Cached scope incomplete')
    native={r['basename']:r for r in met['native_cases']};controls={r['id']:r['scaledclean'] for r in reg['groups']};result=[];allowed={}
    for r in reg['cases']:
        n=native[r['basename']];control=controls[r['id']]
        if n['input_sha256']!=r['wav_sha256'] or n['id']!=r['id']:raise RuntimeError('Basename/ID/SHA join mismatch')
        row={k:r[k] for k in ['basename','id','speaker','noise','snr_db']};row['samples']=r['samples'];row['sources']={}
        for role,path,h in [('noisy',inputroot/r['wav'],r['wav_sha256']),('frontend',modelroot/n['frontend_pcm']['path'],n['frontend_pcm']['sha256']),('clean',inputroot/control['wav'],control['wav_sha256'])]:
            expectedroot=(modelroot/'outputs/frontend').resolve() if role=='frontend' else (inputroot/'inputs').resolve()
            if path.resolve().parent!=expectedroot or path.is_symlink() or '/data/datasets/' in str(path):raise RuntimeError('Unapproved cached source path')
            row['sources'][role]=str(path);allowed[str(path)]=h
        result.append(row)
    if len(allowed)!=72 or len({r['basename'] for r in result})!=32 or sorted({r['speaker'] for r in result})!=cfg['fixed_speakers']:raise RuntimeError('Not72unique PCM or32cases/4speakers')
    return result,allowed

def decode(p,h,counts,independent=False):
    import numpy as np,struct
    b=Path(p).read_bytes()
    if hashlib.sha256(b).hexdigest()!=h:raise RuntimeError('Cached PCM SHA mismatch')
    with wave.open(io.BytesIO(b),'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(16000,1,2,'NONE'):raise RuntimeError('Cached PCM header mismatch')
        n=w.getnframes();pcm=w.readframes(n)
    if len(pcm)!=2*n or n<320:raise RuntimeError('Truncated/short PCM')
    counts['pcm_file_decodes']+=1;counts['pcm_samples_decoded']+=n;counts['pcm_payload_bytes_decoded']+=len(pcm);counts['full_cached_WAV_bytes_read']+=len(b)
    return np.asarray([t[0]/32768 for t in struct.iter_unpack('<h',pcm)],dtype=np.float64) if independent else np.frombuffer(pcm,dtype='<i2').astype(np.float64)/32768

def lifecycle(kind,cfg,allowed,events):
    marker=ROOT/('STARTED.json' if kind=='main' else 'VALIDATION_STARTED.json')
    if marker.exists():raise RuntimeError('Existing exclusive marker: no retries')
    begin=time.monotonic()
    def hard(*args):
        try:write(ROOT/(kind+'_timeout.json'),dict(status='failed',reason='absolute120s'))
        finally:os._exit(124)
    def soft(*args):raise TimeoutError('118s work deadline')
    signal.signal(signal.SIGALRM,soft);signal.setitimer(signal.ITIMER_REAL,118)
    with marker.open('x') as f:json.dump(dict(pid=os.getpid(),stage=kind),f)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] in {'torch','sklearn','whisper','soundfile','librosa','torchaudio','transformers','pystoi','pesq'}:events.append(name);raise RuntimeError('Forbidden model/audio/ML library')
    sys.meta_path.insert(0,Block())
    def audit(event,args):
        if event.startswith('socket.') or event in ['subprocess.Popen','os.system','os.posix_spawn','os.fork']:events.append(event);raise RuntimeError('Forbidden network/process')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and (not p.is_relative_to(ROOT.resolve()) or p.suffix.lower() in ['.wav','.npy','.pcm']):events.append('outside/audio write');raise RuntimeError('Forbidden source/audio write')
            if not writing and (p.suffix.lower() in ['.wav','.ckpt'] or str(p).startswith('/data/datasets/')) and str(p) not in allowed:events.append('unapprovedsource');raise RuntimeError('Forbidden OFF/dataset/checkpoint read')
    sys.addaudithook(audit)
    def check():
        if time.monotonic()-begin>=118 or size()>=cfg['max_directory_bytes']-262144 or events:raise RuntimeError('Work deadline/bytes/forbidden attempt')
    def finalizing():signal.signal(signal.SIGALRM,hard);signal.setitimer(signal.ITIMER_REAL,max(.000001,begin+120-time.monotonic()))
    def finish():
        if time.monotonic()-begin>120 or size()>cfg['max_directory_bytes']:raise RuntimeError('Final cap')
        signal.setitimer(signal.ITIMER_REAL,0)
    return begin,check,finalizing,finish

def load_cases(rows,allowed,feature,counts,independent=False):
    arrays={p:decode(p,h,counts,independent) for p,h in allowed.items()};cases=[]
    for r in rows:
        aa={k:arrays[p] for k,p in r['sources'].items()}
        if any(len(v)!=r['samples'] for v in aa.values()):raise RuntimeError('Cached lengths differ')
        noisy=feature(aa['noisy']);frontend=feature(aa['frontend']);clean=feature(aa['clean']);cases.append(dict(**{k:r[k] for k in ['basename','id','speaker','noise','snr_db']},target=clean['identity'],noisy=noisy,frontend=frontend,tail=clean['unused_tail_samples']))
    return cases

def execute(kind,feature,evaluate,compare=None):
    import numpy as np
    cfg=read(ROOT/'config.yaml');codepre={p.name:digest(p) for p in ROOT.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ['config.yaml','analysis_plan.md'])};rows,allowed=sources(cfg);events=[];begin,check,finalizing,finish=lifecycle(kind,cfg,allowed,events);d=dict(status='running',failures=[],forbidden_attempts=events,pcm_file_decodes=0,pcm_samples_decoded=0,pcm_payload_bytes_decoded=0,full_cached_WAV_bytes_read=0,enhancement_training_calls=0,enhancement_model_calls=0,ASR_calls=0,network_bytes=0,linear_probe_learning=True)
    oldsha=digest(ROOT/'metrics.json') if kind!='main' else None
    try:
        cases=load_cases(rows,allowed,feature,d,kind!='main');check();result,pred=evaluate(cases,check)
        if result['linear_fit_count']!=16 or d['pcm_file_decodes']!=72:raise RuntimeError('Incomplete16fits/72PCM')
        if kind=='main':
            np.savez_compressed(ROOT/'oof_predictions.npz',**pred);d['result']=result
        else:
            saved=read(ROOT/'metrics.json');archive=np.load(ROOT/'oof_predictions.npz',allow_pickle=False)
            if saved['status']!='completed' or set(archive.files)!=set(pred):raise RuntimeError('Main incomplete or predictionscope mismatch')
            compare(result,saved['result'])
            for key in pred:
                if not np.allclose(archive[key],pred[key],rtol=1e-8,atol=1e-9):raise AssertionError('OOF prediction mismatch '+key)
            d.update(protected_metrics_sha256=oldsha,verified_predictions=len(pred),independent_linear_fits=16)
        d.update(status='completed' if kind=='main' else 'passed',linear_fit_count=16,source_pre_hashes=allowed)
    except BaseException as e:
        import traceback
        d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    finally:finalizing()
    try:
        codepost={p.name:digest(p) for p in ROOT.iterdir() if p.name in codepre}
        if codepre!=codepost:raise RuntimeError('Code/config changed')
        d['code_config_sha_pre']=codepre;d['code_config_sha_post']=codepost
        if events or any(digest(p)!=h for p,h in allowed.items()):raise RuntimeError('Forbidden attempt or sourcechanged')
        if digest(cfg['input_registry'])!=cfg['input_registry_sha256'] or digest(cfg['model_metrics'])!=cfg['model_metrics_sha256']:raise RuntimeError('Metadatachanged')
        if kind!='main' and digest(ROOT/'metrics.json')!=oldsha:raise RuntimeError('Originalmetricschanged')
    except BaseException as e:d['status']='failed';d['failures'].append(dict(final=repr(e)))
    d.update(seconds=time.monotonic()-begin,source_post_sha_match=not d['failures'],source_post_hash_files=len(allowed),numpy_version=np.__version__,pid=os.getpid())
    write(ROOT/('metrics.json' if kind=='main' else 'independent_validation.json'),d)
    if kind=='main':(ROOT/'experiment_report.md').write_text('# Cached continuous-structure readability: '+d['status']+'\n\nThis run fits16fixed dualoutputlinear probes (learning), not zero training; StoRM/Score/ASR not trained or called.119dim full vs6dim simple plusidentity/trainmean controls,32cases allframes retained.4LOSO controls only thisprobe, not globalunseen/noise-disjoint. Frontend incorporates fullsentencecontext; differences are context+representation readability, not pureinformationloss. Two cleancontinuous spectraltargets are not phonemes/H1H0/CAEG or enhancementgain. Predictionsunclipped and targetwiseMSE/pairedspeaker/SNR directions retained. No newaudio/rawfeaturematrix/controllerexport.\n')
    finish();print(json.dumps(dict(status=d['status'],seconds=d['seconds'])));return 0 if d['status'] in ['completed','passed'] else 1
