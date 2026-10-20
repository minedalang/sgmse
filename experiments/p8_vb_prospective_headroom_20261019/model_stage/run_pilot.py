"""Prepared native 32-case pilot. Only supervisor runtime may import model libraries."""
from pathlib import Path
import datetime,gc,hashlib,io,json,os,signal,subprocess,sys,time,traceback,weakref
from blocked_events import BlockedEvents
from helpers import require,sha,read,write,verify,module,observe,summarize,weight_path,allow_local_build,aggregate
STAGE=Path(__file__).resolve().parent
OUT=STAGE
BASE=STAGE.parent
INPUT=BASE/'input_stage'
ROOT=Path('/data/coding/sgmse')
CAP=64*1024**2

def size():return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
def check():
    require(time.monotonic()<float(os.environ['CONTENT_PILOT_DEADLINE']),'180s model deadline:170s execution reserve10s')
    require(size()<CAP-1_048_576,'64MiB total cap reserve')
def freeze():return read(STAGE/'execution_manifest.json')

def supervise():
    manifest=freeze();verify(manifest['locks'])
    for n in ['STARTED.json','outputs','model_inputs','metrics.json','experiment_report.md','model_progress.json','model_child_result.json','model_run.log']:
        require(not (OUT/n).exists() and not (OUT/n).is_symlink(),'Existing model execution artifact '+n)
    start=time.monotonic()
    with (OUT/'STARTED.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_manifest_sha256=sha(STAGE/'execution_manifest.json'),scope='model phase only; prior CPU preparation preserved'),f,indent=2)
    env=os.environ.copy();env['PATH']=read(STAGE/'execution_config.json')['native_path'];env.update(CONTENT_PILOT_SUPERVISOR=str(os.getpid()),CONTENT_PILOT_DEADLINE=str(start+170),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NUMBA_CACHE_DIR=str(OUT/'outputs/numba_cache'))
    reason=None
    with (OUT/'model_run.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            if time.monotonic()-start>=170 or size()>=CAP-1_048_576:
                reason='170s execution/64MiB reserve limit';os.killpg(p.pid,signal.SIGKILL);p.wait();break
            time.sleep(.05)
    d=read(OUT/'model_child_result.json') if (OUT/'model_child_result.json').exists() else read(OUT/'model_progress.json') if (OUT/'model_progress.json').exists() else dict(status='failed',failures=[])
    if reason or p.returncode:d['status']='failed';d.setdefault('failures',[]).append(dict(supervisor_error=reason or 'child exit '+str(p.returncode)))
    try:d['supervisor_post_locks']=verify(manifest['locks']);require(sha(STAGE/'execution_manifest.json')==d.get('execution_manifest_sha256',sha(STAGE/'execution_manifest.json')),'Manifest changed')
    except BaseException as e:d['status']='failed';d.setdefault('failures',[]).append(dict(postcheck=repr(e)))
    d.update(supervisor_seconds=time.monotonic()-start,supervisor_pid=os.getpid(),child_pid=p.pid,time_cap=180,execution_seconds_cap=170,finalization_reserve_seconds=10,child_exit=p.returncode,total_bytes_before_report=size())
    d['artifact_manifest']={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in ['metrics.json','experiment_report.md']}
    write(OUT/'metrics.json',d)
    (OUT/'experiment_report.md').write_text('Native matched-content measurement pilot: '+d['status']+'\n\nAll eight utterances/two reused noises/two full-utterance SNRs retained. See metrics.json for all104 raw ASR/scoring rows, S/D/I/N, word-weighted corpus and sentence means, noise/SNR strata, per-case OFF−frontend differences and canonical reference-position transitions. These are single-ASR proxies, not verified speech deletion/recovery, human truth or phoneme times. Eight scaled-clean controls remain measurement controls; discrepancies exclude no cases. Basenames cause different SNR seeds. Four speakers and reused historical noises;32conditions are not32independent samples, pretraining unknown. Prior local-conversion gate remains failed. Raw/clipping/serialization proofs and partial failures retained. Blocked local IPv6 probe refusals are counted separately in metrics, with full safe caller evidence; zero unknown network refusals and zero network bytes are required. Every socket probe remains refused; no retry/training/test access.\n')
    with (OUT/'experiment_report.md').open('a') as report:report.write('\nBlocked local IPv6 probe refusals: '+str(d.get('blocked_local_probe_count',0))+'; unknown network refusals: '+str(d.get('unknown_network_refusal_count',len(d.get('network_attempts',[]))))+'; actual network bytes: '+str(d.get('network_bytes',0))+'. All socket attempts remain refused.\n')
    require(size()<=CAP,'Final total exceeds64MiB');require(time.monotonic()-start<=180,'Final runtime exceeded180s');print(json.dumps(dict(status=d['status'],seconds=d['supervisor_seconds'],bytes=size(),failures=d.get('failures'))));return 0 if d['status']=='completed' else 1

def enhance_native(cfg,registry,d,checkpoint):
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0,str(ROOT));import joint_enhance_by_dir as native
    import sgmse.sampling as sampling
    from sgmse.model import ScoreModel
    cls=native.JointSemambaSgmseModel
    inp=OUT/'model_inputs';inp.mkdir();output=OUT/'outputs';output.mkdir(exist_ok=True)
    (output/'off').mkdir();(output/'frontend').mkdir();(output/'raw').mkdir()
    for r in registry['cases']:(inp/r['basename']).symlink_to(INPUT/r['wav'])
    require(len(list(inp.glob('*.wav')))==32,'Exactly32 noisy links required')
    order=sorted(registry['cases'],key=lambda r:r['basename']);context={};weakmodels=[]
    def arr(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    def desc(x):
        a=np.ascontiguousarray(arr(x));return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    def save(arm,x):
        a=np.ascontiguousarray(arr(x));require(a.dtype==np.float32 and np.isfinite(a).all(),'Finite rawfloat32 required');np.save(output/'raw'/(context['basename']+'__'+arm+'.npy'),a,allow_pickle=False);return dict(path='outputs/raw/'+context['basename']+'__'+arm+'.npy',**desc(a))
    def clipstats(x):
        a=arr(x).reshape(-1);return dict(peak=float(np.abs(a).max()),rms=float(np.sqrt(np.mean(a.astype(np.float64)**2))),above_one=int(np.sum(a>1)),below_minus_one=int(np.sum(a< -1)),at_positive_endpoint=int(np.sum(a==1)),at_negative_endpoint=int(np.sum(a== -1)),serialization_above_max_pcm16=int(np.sum(a>32767/32768)),serialization_below_min_pcm16=int(np.sum(a< -1)))
    original_load=native.load_mono;original_front=cls._frontend_forward;original_match=native._match_length;original_enhance=cls.enhance;original_sampler=sampling.get_pc_sampler;original_loader=cls.__dict__.get('load_from_checkpoint');bound_loader=cls.load_from_checkpoint;original_eval=cls.eval
    def load(path,*a,**kw):
        nonlocal context
        checkpoint();index=len(d['native_cases']);require(index<32,'Native input count cap');r=order[index];require(Path(path).name==r['basename'] and sha(path)==r['wav_sha256'],'Native source/order mismatch');context=dict(basename=r['basename'],id=r['id'],speaker=r['speaker'],seed=r['seed'],input_sha256=r['wav_sha256']);d['native_cases'].append(context)
        result=original_load(path,*a,**kw);require(tuple(result.shape)==(r['samples'],),'Native loader geometry');context['input']=desc(result);context['input_stats']=clipstats(result);context['actual_torch_seed']=int(torch.initial_seed());require(context['actual_torch_seed']==r['seed'],'Native seed mismatch');return result
    def loader(*a,**kw):
        checkpoint();require(d['enhancement_model_loads']==0,'One enhancement model load');d['enhancement_model_loads']+=1;d['loader_kwargs']={k:(v.__name__ if isinstance(v,type) else v) for k,v in kw.items()};require(kw.get('strict') is True and kw.get('weights_only') is False,'Native strict loader required');m=bound_loader(*a,**kw);weakmodels.append(weakref.ref(m));return m
    def eval_observer(self,*a,**kw):
        d['native_eval_args']=dict(positional=list(a),kwargs=kw);require(kw.get('no_ema') is False,'EMA must be enabled');result=original_eval(self,*a,**kw);d['native_model_flags']=dict(training=bool(self.training),train_type=self.train_type,residual_refine_enable=bool(self.residual_refine_enable),frontend=self.hparams.get('frontend'),frontend_ckpt=self.hparams.get('frontend_ckpt'));require(not self.training and self.hparams.get('frontend')=='storm' and self.hparams.get('frontend_ckpt') is None,'Unexpected runtime frontend');return result
    def front(self,*a,**kw):
        checkpoint();d['frontend_calls']+=1;require(d['frontend_calls']<=32,'32 frontend cap');context['frontend_call_input']=desc(a[0]);require(self.train_type=='mode2' and not self.residual_refine_enable,'Expected native mode2 nonresidual');result=original_front(self,*a,**kw);context['frontend_before_match']=desc(result);return result
    def match(a,b):
        result=original_match(a,b);T=int(b.shape[-1]);cur=int(a.shape[-1]);context['native_length_match']=dict(input=cur,reference=T,output=int(result.shape[-1]),operation='center_crop' if cur>T else 'center_pad' if cur<T else 'identity',left=(abs(cur-T)//2),right=abs(cur-T)-abs(cur-T)//2);context['noisy_peak_native_reference']=float(b.abs().max().clamp_min(1e-12).item());context['frontend_raw']=save('frontend',result);context['frontend_preclip']=clipstats(result)
        bio=io.BytesIO();sf.write(bio,arr(result).reshape(-1),16000,format='WAV',subtype='PCM_16');path=output/'frontend'/context['basename'];path.write_bytes(bio.getvalue());context['frontend_pcm']=dict(path=str(path.relative_to(OUT)),sha256=sha(path),method='same native SoundFile float32 PCM_16; no loudness normalization or clipping',serialized=clipstats(sf.read(path,dtype='float32')[0]));checkpoint();return result
    def sampler(*a,**kw):
        require(kw.get('anchor_calibrator') is None and kw.get('score_calibrator') is None and kw.get('regeneration_strength') is None,'OFF calibrators/regen forbidden');fn=original_sampler(*a,**kw)
        context['sampler_args']=dict(predictor=a[0] if a else kw.get('predictor_name'),corrector=a[1] if len(a)>1 else kw.get('corrector_name'),sde=type(a[2] if len(a)>2 else kw.get('sde')).__name__,N=int((a[2] if len(a)>2 else kw.get('sde')).N),snr=kw.get('snr'),trace_sink_present=kw.get('trace_sink') is not None)
        def run(*aa,**kk):
            result=fn(*aa,**kk);context['nfe']=int(result[1]);require(context['nfe']==30,'NFE30 required');return result
        return run
    def enhance(self,*a,**kw):
        checkpoint();d['backend_calls']+=1;require(d['backend_calls']<=32,'32 OFF cap');require(kw.get('timeit',False) is False,'No timeit override')
        for k,v in cfg['expected_enhance_kwargs'].items():require(kw.get(k)==v,'Native kwargs mismatch '+k)
        context['enhance_kwargs']={k:v for k,v in kw.items() if k!='y_cond'};context['backend_anchor']=desc(a[0]);context['backend_condition']=desc(kw['y_cond']);require(context['backend_anchor']['sha256']==context['frontend_raw']['sha256'],'Backend must receive actual matched raw frontend');require(context['backend_condition']['sha256']==context['input']['sha256'],'Backend condition must be actual native noisy')
        require(sys.getprofile() is None,'Unexpected profiling observer')
        def profile(frame,event,result):
            if event=='return' and frame.f_code is ScoreModel.enhance.__code__:
                context['backend_actual_norm_factor']=float(frame.f_locals['norm_factor']);context['backend_internal_T_orig']=int(frame.f_locals['T_orig'])
        sys.setprofile(profile)
        try:result=original_enhance(self,*a,**kw)
        finally:sys.setprofile(None)
        context['backend_raw']=save('off',result);context['off_preclip']=clipstats(result);require(context['backend_actual_norm_factor']==context['noisy_peak_native_reference'],'Actual backend/native frontend normalization reference mismatch');checkpoint();return result
    native.load_mono=load;cls.load_from_checkpoint=staticmethod(loader);cls._frontend_forward=front;native._match_length=match;cls.enhance=enhance;cls.eval=eval_observer;sampling.get_pc_sampler=sampler
    argv=['joint_enhance_by_dir.py','--ckpt',cfg['checkpoint'],'--in_dir',str(inp),'--out_dir',str(output/'off'),'--device','cuda:0','--sample_rate','16000','--train_type','mode2','--sampler_type','pc','--corrector','none','--N','30','--snr','0.5','--seed','2026','--use_ema'];d['native_argv']=argv;oldargv=sys.argv
    try:checkpoint();sys.argv=argv;native.main()
    finally:
        sys.argv=oldargv;native.load_mono=original_load;cls._frontend_forward=original_front;native._match_length=original_match;cls.enhance=original_enhance;cls.eval=original_eval;sampling.get_pc_sampler=original_sampler
        if original_loader is None:delattr(cls,'load_from_checkpoint')
        else:setattr(cls,'load_from_checkpoint',original_loader)
    require(d['frontend_calls']==d['backend_calls']==len(d['native_cases'])==32,'Complete32 native pairs')
    for r in d['native_cases']:
        raw=np.load(OUT/r['backend_raw']['path'],allow_pickle=False);path=output/'off'/r['basename'];bio=io.BytesIO();sf.write(bio,np.clip(raw,-1.,1.),16000,format='WAV',subtype='PCM_16');require(bio.getvalue()==path.read_bytes(),'Native OFF libsndfile reencoding mismatch');r['off_pcm']=dict(path=str(path.relative_to(OUT)),sha256=sha(path),memory_reencoded_sha256=hashlib.sha256(bio.getvalue()).hexdigest(),exact_native_serialization=True,serialization_source=clipstats(np.clip(raw,-1.,1.)),serialized=clipstats(sf.read(path,dtype='float32')[0]));require(raw.shape==(next(x['samples'] for x in registry['cases'] if x['basename']==r['basename']),),'OFF output length')
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();gc.collect();d['enhancement_release']=dict(model_weakrefs_dead=all(w() is None for w in weakmodels),cuda_allocated=int(torch.cuda.memory_allocated()),cuda_reserved=int(torch.cuda.memory_reserved()));require(d['enhancement_release']['model_weakrefs_dead'],'Enhancement model still referenced before ASR');checkpoint()


def asr_requests(registry,native_cases,input_root,out_root):
    requests=[dict(key=g['id']+'__scaledclean',id=g['id'],arm='scaledclean',path=str(input_root/g['scaledclean']['wav'])) for g in registry['groups']]
    native={r['basename']:r for r in native_cases}
    for r in registry['cases']:
        for arm,path in [('noisy',str(input_root/r['wav'])),('frontend',str(out_root/native[r['basename']]['frontend_pcm']['path'])),('off',str(out_root/native[r['basename']]['off_pcm']['path']))]:
            requests.append(dict(key=r['basename']+'__'+arm,id=r['id'],arm=arm,path=path,basename=r['basename'],noise=r['noise'],snr_db=r['snr_db'],speaker=r['speaker']))
    require(len(requests)==104 and len({r['key'] for r in requests})==104,'Complete104 unique requests required')
    return requests

def child():
    require(os.environ.get('CONTENT_PILOT_SUPERVISOR')==str(os.getppid()),'Supervisor required');cfg=read(STAGE/'execution_config.json');manifest=freeze();registry=read(INPUT/'input_registry.json');require(sha(INPUT/'input_registry.json')==cfg['actual_input_registry_sha256'],'Input registry lock mismatch')
    d=dict(status='running',failures=[],execution_manifest_sha256=sha(STAGE/'execution_manifest.json'),network_attempts=[],network_bytes=0,training_calls=0,test_content_accessed=False,enhancement_model_loads=0,asr_model_loads=0,frontend_calls=0,backend_calls=0,transcribe_api_calls=0,internal_decode_api_calls=0,neural_forward_count='not measured; API counts are not forward counts',native_cases=[],items=[])
    def checkpoint():
        d['blocked_local_probe_count']=len(d.get('local_probe_refusals',[]));d['unknown_network_refusal_count']=len(d['network_attempts']);write(OUT/'model_progress.json',d);check();require(not d['network_attempts'],'Unknown/forbidden network refusal; stop before next scientific call');require(d['network_bytes']==0,'Network byte budget')
    try:
        d['source_pre']=verify(manifest['locks']);check();require(os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize>100_000_000,'Insufficient free disk')
        d['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv'],text=True,timeout=5);d['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();d['git_status_before']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        cal=module('frozen_clean_calibration',cfg['calibration_driver']);events=BlockedEvents(cfg['blocked_local_probe_source']['path'],cfg['blocked_local_probe_source']['sha256'],cfg['calibration_driver']);d['network_attempts']=events;d['blocked_events']=events.all_events;d['local_probe_refusals']=events.local_probe_refusals;refuse=cal.offline_guard(events);events.bind_refuse(refuse);checkpoint()
        def audit(event,args):
            if event in ['socket.connect','socket.getaddrinfo','socket.sendto','socket.bind']:
                d['network_attempts'].append(event);raise RuntimeError('Network forbidden')
            if event=='subprocess.Popen':d.setdefault('local_native_subprocesses',[]).append(allow_local_build(args[0],args[1],args[2],cfg))
            if event in ['os.system','os.posix_spawn']:raise RuntimeError('Shell/spawn forbidden')
        sys.addaudithook(audit)
        import dataclasses,inspect,random
        import numpy as np
        import soundfile as sf
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()>=1,'Fixed cuda:0 unavailable');torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();require(free>=cfg['minimum_free_gpu_bytes'],'Insufficient CUDA0 free memory; no fallback');d['gpu_preflight']=dict(device='cuda:0',free=free,total=total)
        d['environment']=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,soundfile=sf.__version__,libsndfile=sf.__libsndfile_version__,cuda=torch.version.cuda)
        oldload=torch.load
        def locked_load(path,*a,**kw):
            checkpoint();actual_path=weight_path(path,manifest['locks']);d.setdefault('weight_load_paths',[]).append(dict(path=actual_path,source_type=type(path).__name__));return oldload(path,*a,**kw)
        torch.load=locked_load
        try:enhance_native(cfg,registry,d,checkpoint)
        finally:torch.load=oldload
        os.environ['XDG_CACHE_HOME']=str(OUT/'outputs/runtime_cache')
        import whisper
        import whisper.decoding as decoding
        from whisper.transcribe import transcribe
        require(Path(whisper.__file__).resolve()==Path(cfg['calibration']['whisper_package'])/'__init__.py','Wrong Whisper source');whisper._download=refuse
        random.seed(cfg['calibration']['seed']);np.random.seed(cfg['calibration']['seed']);torch.manual_seed(cfg['calibration']['seed']);torch.cuda.manual_seed_all(cfg['calibration']['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        options=dict(cfg['calibration']['transcribe_options']);options['temperature']=tuple(options['temperature']);d['transcribe_passed_options']=options
        explicit=inspect.signature(transcribe);d['transcribe_installed_defaults']={k:v.default for k,v in explicit.parameters.items() if v.default is not inspect.Parameter.empty}
        effective={k:v for k,v in options.items() if k in decoding.DecodingOptions.__dataclass_fields__};effective.pop('temperature',None);effective.pop('best_of',None);effective['prompt']=[];d['effective_decoding_options']=dataclasses.asdict(decoding.DecodingOptions(**effective,temperature=0.))
        require(d['effective_decoding_options']==cfg['calibration_effective_options'],'Effective options differ from completed clean calibration');checkpoint()
        model=whisper.load_model(cfg['calibration']['model_path'],device='cuda:0');d['asr_model_loads']+=1;model.eval();require(str(next(model.parameters()).device)=='cuda:0' and next(model.parameters()).dtype==torch.float32,'Frozen ASR device/dtype')
        decode_records=[];original_decode=model.decode
        def decode(audio,actual_options=decoding.DecodingOptions(),**kwargs):
            d['internal_decode_api_calls']+=1;actual=dataclasses.asdict(actual_options);require(actual==d['effective_decoding_options'],'Actual decode options changed');decode_records.append(dict(index=d['internal_decode_api_calls'],options=actual));return original_decode(audio,actual_options,**kwargs)
        model.decode=decode
        refs={r['id']:r for r in registry['references']};requests=asr_requests(registry,d['native_cases'],INPUT,OUT)
        (OUT/'outputs/asr').mkdir()
        for request in requests:
            checkpoint();require(d['transcribe_api_calls']<104,'104 transcribe API cap');audio,sr=sf.read(OUT/request['path'],dtype='float32');require(sr==16000 and audio.ndim==1 and np.isfinite(audio).all(),'ASR full native PCM input');request['input']=dict(wav_sha256=sha(OUT/request['path']),samples=len(audio),array_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),peak=float(np.abs(audio).max()),rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2))))
            before=d['internal_decode_api_calls'];d['transcribe_api_calls']+=1;d['current_item']=request['key'];checkpoint()
            with torch.inference_mode():result=model.transcribe(audio,**options)
            torch.cuda.synchronize();require(isinstance(result,dict) and isinstance(result.get('text'),str),'Raw transcript missing');sc=cal.score(refs[request['id']]['original_text'],result['text']);raw=dict(request=request,publisher_reference=refs[request['id']],raw_result=result,decode_calls=decode_records[before:]);rawpath='outputs/asr/'+request['key']+'_raw.json';scorepath='outputs/asr/'+request['key']+'_score.json';write(OUT/rawpath,raw);write(OUT/scorepath,sc);d['items'].append(dict(**request,score=sc,raw_file=rawpath,score_file=scorepath,decode_api_calls=d['internal_decode_api_calls']-before));checkpoint()
        require(d['transcribe_api_calls']==len(d['items'])==104 and d['asr_model_loads']==1 and d['enhancement_model_loads']==1,'Required complete call counts');require(not d['network_attempts'],'Forbidden network attempt');d['blocked_local_probe_count']=len(d['local_probe_refusals']);d['unknown_network_refusal_count']=len(d['network_attempts']);d['summary']=summarize(d['items'],registry['cases'],aggregate)
        from headroom import analyze
        from quality import measure_all
        d['headroom']=analyze(d['items'],registry['cases'],aggregate)
        d['quality']=measure_all(registry,d['native_cases'],INPUT,OUT,checkpoint);checkpoint()
        d['pipeline_resampling_calls']=0;d['metric_internal_preprocessing']='STOI standard16k-to10k resampling/silence removal; PESQ standard joint peak scaling and intrinsic processing'
        d['status']='completed'
    except BaseException as e:d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()));traceback.print_exc()
    finally:
        d['blocked_local_probe_count']=len(d.get('local_probe_refusals',[]));d['unknown_network_refusal_count']=len(d['network_attempts'])
        try:d['source_post']=verify(manifest['locks']);require(sha(STAGE/'execution_manifest.json')==d['execution_manifest_sha256'],'Manifest changed')
        except BaseException as e:d['status']='failed';d['failures'].append(dict(postcheck=repr(e)))
        write(OUT/'model_child_result.json',d)
    return 0 if d['status']=='completed' else 1

if __name__=='__main__':
    require(sys.argv[1:] in [[],['--child']],'No overrides');sys.exit(child() if sys.argv[1:]==['--child'] else supervise())
