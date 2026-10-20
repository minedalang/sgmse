"""Prepared clean-reference measurement calibration. Safe standard-library-only import."""
from pathlib import Path
import datetime, hashlib, json, os, random, re, signal, socket, subprocess, sys, time, traceback, wave, zipfile
OUT=Path(__file__).resolve().parent
ROOT=Path('/data/coding/sgmse')
TIME_CAP=180
ART_CAP=5_000_000
IDS=['p226_027','p226_028','p287_037','p287_054']

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def artifact_bytes():return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
def verify_locks(locks):
    values={str(p):sha(p) for p in locks}
    require(values==locks,'Source/preparation hash changed')
    return values

def offline_guard(attempts):
    """Refuse network creation/resolution before importing any heavy library."""
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:os.environ[k]='1'
    def refuse(*args,**kwargs):
        attempts.append({'event':'network_refused','argument_types':[type(x).__name__ for x in args]})
        raise RuntimeError('Network forbidden in clean calibration')
    original=socket.socket
    class OfflineSocket(original):
        def __init__(self,family=socket.AF_INET,type=socket.SOCK_STREAM,proto=0,fileno=None):
            if family in (socket.AF_INET,socket.AF_INET6):refuse(family,type,proto)
            super().__init__(family,type,proto,fileno)
        def connect(self,*a,**k):return refuse(*a,**k)
        def connect_ex(self,*a,**k):return refuse(*a,**k)
        def sendto(self,*a,**k):return refuse(*a,**k)
    socket.socket=OfflineSocket
    socket.create_connection=refuse;socket.getaddrinfo=refuse
    import urllib.request
    urllib.request.urlopen=refuse
    return refuse

def header(path):
    with wave.open(str(path),'rb') as w:
        return {'sample_rate':w.getframerate(),'channels':w.getnchannels(),'sample_width':w.getsampwidth(),'frames':w.getnframes(),'compression':w.getcomptype()}

def references(cfg):
    found={}
    with Path(cfg['text_manifest']).open() as f:
        for line in f:
            row=json.loads(line)
            if row['id'] in IDS:
                require(row['id'] not in found,'Duplicate reference ID');found[row['id']]=row
    require(list(sorted(found))==sorted(IDS),'Missing publisher reference')
    values=[]
    with zipfile.ZipFile(cfg['text_zip']) as z:
        for sid in IDS:
            t=found[sid]['transcript'];raw=z.read(t['archive_entry'])
            require(hashlib.sha256(raw).hexdigest()==t['raw_text_SHA256'],'Publisher raw text hash mismatch')
            require(len(raw)==t['raw_text_bytes'],'Publisher raw text byte count mismatch')
            text=raw.decode('utf-8-sig');require(text==t['original_text'],'Publisher original text mismatch')
            values.append({'id':sid,'archive_entry':t['archive_entry'],'raw_text_SHA256':t['raw_text_SHA256'],'raw_text_hex':raw.hex(),'original_text':text})
    return values

def scoring_tokens(text):
    text=text.replace("\u2019", "'").replace("\u2018", "'").lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*",text),re.findall('[a-z0-9]',text)

def alignment(reference,hypothesis):
    n=len(reference);m=len(hypothesis);dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):dp[i][0]=i
    for j in range(m+1):dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=min(dp[i-1][j-1]+(reference[i-1]!=hypothesis[j-1]),dp[i-1][j]+1,dp[i][j-1]+1)
    i=n;j=m;steps=[];counts={k:0 for k in ['H','S','D','I']}
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(reference[i-1]!=hypothesis[j-1]):
            op='H' if reference[i-1]==hypothesis[j-1] else 'S';i-=1;j-=1;ri=i;hj=j
        elif i and dp[i][j]==dp[i-1][j]+1:op='D';i-=1;ri=i;hj=None
        else:require(j>0 and dp[i][j]==dp[i][j-1]+1,'Invalid DP backtrace');op='I';j-=1;ri=None;hj=j
        counts[op]+=1;steps.append({'op':op,'reference_index':ri,'hypothesis_index':hj,'reference':reference[ri] if ri is not None else None,'hypothesis':hypothesis[hj] if hj is not None else None})
    steps.reverse();errors=counts['S']+counts['D']+counts['I']
    require(errors==dp[n][m] and counts['H']+counts['S']+counts['D']==n and counts['H']+counts['S']+counts['I']==m,'Alignment invariant')
    return dict(counts,N=n,N_hyp=m,edit_distance=errors,rate=errors/n if n else None,rate_status='defined' if n else 'undefined_empty_reference',alignment=steps)

def score(reference,hypothesis):
    rw,rc=scoring_tokens(reference);hw,hc=scoring_tokens(hypothesis)
    return {'original_reference':reference,'original_hypothesis':hypothesis,'reference_word_tokens':rw,'hypothesis_word_tokens':hw,'reference_scoring_words':' '.join(rw),'hypothesis_scoring_words':' '.join(hw),'reference_scoring_chars':''.join(rc),'hypothesis_scoring_chars':''.join(hc),'word':alignment(rw,hw),'character':alignment(rc,hc)}

def aggregate(scores):
    require(len(scores)>0,'No scores')
    out={}
    for kind in ['word','character']:
        sums={key:sum(s[kind][key] for s in scores) for key in ['S','D','I','H','N','N_hyp','edit_distance']}
        require(sums['N']>0 and all(s[kind]['rate'] is not None for s in scores),'Empty calibration reference')
        out[kind]=dict(sums,micro_rate=sums['edit_distance']/sums['N'],sentence_mean_rate=sum(s[kind]['rate'] for s in scores)/len(scores))
    return out


def supervise():
    manifest=read(OUT/'frozen_manifest.json');verify_locks({str(OUT/n):h for n,h in manifest.items()})
    for n in ['STARTED.json','outputs','metrics.json','experiment_report.md','run.log','progress.json','child_result.json']:
        require(not (OUT/n).exists() and not (OUT/n).is_symlink(),'Existing execution artifact '+n)
    start=time.monotonic()
    write_new=OUT/'STARTED.json'
    with write_new.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':[sys.executable,str(Path(__file__).resolve())],'frozen_manifest_SHA256':sha(OUT/'frozen_manifest.json')},f,indent=2)
    env=os.environ.copy();env.update(CALIBRATION_DEADLINE=str(start+TIME_CAP-2),CALIBRATION_SUPERVISED=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',NUMBA_CACHE_DIR=str(OUT/'outputs/numba_cache'),XDG_CACHE_HOME=str(OUT/'outputs/runtime_cache'),TOKENIZERS_PARALLELISM='false')
    reason=None
    with (OUT/'run.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            if time.monotonic()-start>=TIME_CAP-2 or artifact_bytes()>=ART_CAP-500_000:
                reason='180s budget:178s execution +2s finalization reserve' if time.monotonic()-start>=TIME_CAP-2 else '5MB artifact cap reserve';os.killpg(child.pid,signal.SIGKILL);child.wait();break
            time.sleep(.05)
    d=read(OUT/'child_result.json') if (OUT/'child_result.json').exists() else read(OUT/'progress.json') if (OUT/'progress.json').exists() else {'status':'failed','failures':[],'transcribe_api_calls':0}
    if reason or child.returncode!=0:d['status']='failed';d.setdefault('failures',[]).append({'supervisor_error':reason or 'child exit '+str(child.returncode)})
    try:
        d['supervisor_post_source_hashes']=verify_locks(read(OUT/'config.yaml')['source_locks'])
        d['supervisor_post_prepared_hashes']=verify_locks({str(OUT/n):h for n,h in manifest.items()})
    except BaseException as exc:d['status']='failed';d.setdefault('failures',[]).append({'supervisor_postcheck_error':repr(exc)})
    d.update(supervisor_seconds=time.monotonic()-start,child_exit_code=child.returncode,execution_deadline_seconds=178,finalization_reserve_seconds=2,artifact_bytes_before_final=artifact_bytes())
    d['artifact_manifest']={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in ['metrics.json','experiment_report.md']}
    write(OUT/'metrics.json',d)
    (OUT/'experiment_report.md').write_text('# Clean-text measurement calibration: '+d['status']+'\n\nFour fixed clean source-defined utterances only; publisher reference, not clean-ASR pseudo-reference. See metrics.json and outputs for raw decoding, all word/character alignments, S/D/I/N and micro versus sentence-mean scores. API-call counts are not autoregressive forward counts. Previous local-conversion gate remains failed. This single-ASR clean calibration is not human truth, independent holdout, enhancement performance or validation on noisy speech; all discrepancies remain unresolved unless separately audited. No accuracy pass threshold or post-result reselection. No network, noisy/test content, enhancement/Router/training.\n')
    require(artifact_bytes()<=ART_CAP,'Final artifacts exceed frozen5MB')
    print(json.dumps({'status':d['status'],'seconds':d['supervisor_seconds'],'transcribe_api_calls':d.get('transcribe_api_calls'),'aggregate':d.get('aggregate'),'failures':d.get('failures')}))
    return 0 if d['status']=='completed' else 1

def runtime():
    require(os.environ.get('CALIBRATION_SUPERVISED')==str(os.getppid()),'Supervisor required')
    cfg=read(OUT/'config.yaml');d={'status':'running','failures':[],'network_attempts':[],'network_bytes':0,'transcribe_api_calls':0,'internal_decode_api_calls':0,'neural_forward_count':'not_measured_not_equal_to_transcribe_calls','audio_decodes':0,'enhancement_calls':0,'training_calls':0,'test_content_accessed':False,'model_loads':0,'samples':{}}
    def checkpoint():
        write(OUT/'progress.json',d)
        require(time.monotonic()<float(os.environ['CALIBRATION_DEADLINE']),'180s deadline')
        require(artifact_bytes()<ART_CAP-500_000,'5MB artifact cap reserve')
    try:
        refuse=offline_guard(d['network_attempts'])
        (OUT/'outputs').mkdir();checkpoint()
        d['source_hashes_pre']=verify_locks(cfg['source_locks']);d['prepared_hashes_pre']=verify_locks({str(OUT/n):h for n,h in read(OUT/'frozen_manifest.json').items()})
        refs=references(cfg);require([r['id'] for r in refs]==IDS,'Reference order')
        require([len(scoring_tokens(r['original_text'])[0]) for r in refs]==[12,9,9,9],'Frozen reference counts')
        d['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();d['git_status_before']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        import dataclasses, inspect
        import numpy as np
        import soundfile as sf
        import torch
        import whisper
        import whisper.decoding as decoding
        import importlib.metadata
        from whisper.transcribe import transcribe
        require(Path(whisper.__file__).resolve()==Path(cfg['whisper_package'])/'__init__.py','Whisper package source mismatch')
        whisper._download=refuse
        require(torch.cuda.is_available(),'Frozen cuda:0 unavailable; no fallback')
        torch.cuda.set_device(0);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        d['environment']={'python':sys.version,'executable':sys.executable,'whisper':whisper.__version__,'torch':torch.__version__,'numpy':np.__version__,'soundfile':sf.__version__,'cuda':torch.version.cuda,'device':'cuda:0','gpu_name':torch.cuda.get_device_name(0),'seed':cfg['seed'],'seeded':['python','numpy','torch_cpu','torch_cuda_all'],'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'offline_env':{k:os.environ.get(k) for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']}}
        for package in ['openai-whisper','tiktoken','numba','llvmlite']:
            d['environment'][package]=importlib.metadata.version(package)
        options=dict(cfg['transcribe_options']);options['temperature']=tuple(options['temperature'])
        signature=inspect.signature(transcribe);explicit=set(signature.parameters)
        require(all(k in explicit or k in decoding.DecodingOptions.__dataclass_fields__ for k in options),'Unsupported frozen option')
        defaults={k:v.default for k,v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
        d['transcribe_installed_defaults']=defaults;d['transcribe_passed_options']=dict(options)
        effective={k:v for k,v in options.items() if k in decoding.DecodingOptions.__dataclass_fields__}
        effective.pop('temperature',None);effective.pop('best_of',None);effective['prompt']=[]
        # Installed transcribe drops best_of for t==0; preserve supplied1 and effectiveNone separately.
        effective_options=decoding.DecodingOptions(**effective,temperature=0.0)
        d['effective_decoding_options_before_inference']=dataclasses.asdict(effective_options)
        d['temperature_attempts']=1;d['best_of_zero_temperature_note']='transcribe receives best_of=1; installed code removes it at t=0, effective DecodingOptions.best_of=None'
        d['input_processing']='Full stored PCM16 divided by32768, float32; no gain/trim/resample/VAD. Whisper internal fixed log-mel and padding permitted.'
        checkpoint()
        model=whisper.load_model(cfg['model_path'],device='cuda:0');d['model_loads']=1;model.eval()
        require(str(next(model.parameters()).device)=='cuda:0' and next(model.parameters()).dtype==torch.float32,'Frozen CUDA/float32 model mismatch')
        original_decode=model.decode;decode_records=[]
        def counted_decode(audio,actual_options=decoding.DecodingOptions(),**kwargs):
            d['internal_decode_api_calls']+=1
            actual=dataclasses.asdict(actual_options);require(actual==d['effective_decoding_options_before_inference'],'Actual decode options differ from recorded effective options')
            decode_records.append({'index':d['internal_decode_api_calls'],'options':actual})
            return original_decode(audio,actual_options,**kwargs)
        model.decode=counted_decode
        for sample,ref in zip(cfg['samples'],refs):
            checkpoint();sid=sample['id'];require(sid==ref['id'],'Frozen sample/reference order')
            require(header(sample['audio'])==sample['header'],'Input header changed')
            audio,sr=sf.read(sample['audio'],dtype='float32');d['audio_decodes']+=1
            require(sr==16000 and audio.shape==(sample['header']['frames'],) and np.isfinite(audio).all(),'Full input geometry')
            input_stats={'array_SHA256':hashlib.sha256(audio.tobytes()).hexdigest(),'frames':len(audio),'sample_rate':sr,'dtype':str(audio.dtype),'peak':float(np.max(np.abs(audio))),'rms':float(np.sqrt(np.mean(audio.astype(np.float64)**2))),'input_WAV_SHA256':sha(sample['audio'])}
            require(d['transcribe_api_calls']<4,'Four transcribe API cap')
            d['transcribe_api_calls']+=1;d['current_id']=sid;checkpoint();count_before=d['internal_decode_api_calls']
            with torch.inference_mode():result=model.transcribe(audio,**options)
            torch.cuda.synchronize()
            require(isinstance(result,dict) and isinstance(result.get('text'),str),'Missing raw transcript')
            raw={'id':sid,'publisher_reference':ref,'input':input_stats,'raw_result':result,'decode_calls':decode_records[count_before:]}
            write(OUT/'outputs'/f'{sid}_raw.json',raw)
            scored=score(ref['original_text'],result['text']);require(scored['word']['N']>0 and scored['character']['N']>0,'Empty fixed reference')
            write(OUT/'outputs'/f'{sid}_scoring.json',scored)
            d['samples'][sid]={'input':input_stats,'score':scored,'raw_file':f'outputs/{sid}_raw.json','decode_api_calls':d['internal_decode_api_calls']-count_before};checkpoint()
        require(d['transcribe_api_calls']==d['audio_decodes']==4 and list(d['samples'])==IDS,'All four complete items required')
        require(not d['network_attempts'],'Attempted forbidden network')
        d['aggregate']=aggregate([d['samples'][sid]['score'] for sid in IDS]);d['status']='completed'
    except BaseException as exc:
        d['status']='failed';d['failures'].append({'error':repr(exc),'traceback':traceback.format_exc()});traceback.print_exc()
    finally:
        try:
            d['source_hashes_post']=verify_locks(cfg['source_locks']);d['prepared_hashes_post']=verify_locks({str(OUT/n):h for n,h in read(OUT/'frozen_manifest.json').items()})
            d['git_status_after']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        except BaseException as exc:d['status']='failed';d['failures'].append({'postcheck_error':repr(exc)})
        write(OUT/'child_result.json',d)
    return 0 if d['status']=='completed' else 1

if __name__=='__main__':
    if sys.argv[1:]==['--child']:sys.exit(runtime())
    require(not sys.argv[1:],'No overrides');sys.exit(supervise())
