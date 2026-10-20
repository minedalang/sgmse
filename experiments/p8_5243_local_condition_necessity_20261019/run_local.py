"""Frozen 5243 local-condition necessity diagnostic. Preparation-safe import."""
from pathlib import Path
import datetime, functools, gc, hashlib, json, os, signal, subprocess, sys, time, traceback
ROOT=Path('/data/coding/sgmse'); OUT=Path(__file__).resolve().parent
CAP=30_000_000; SECONDS=240

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def scalar(v):
    if hasattr(v,'item'): return v.item()
    if isinstance(v,Path): return str(v)
    raise TypeError(type(v).__name__)

def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,default=scalar,allow_nan=False)+'\n')

def read(path): return json.loads(Path(path).read_text())

def size(): return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())

def check():
    if time.monotonic()>=float(os.environ['LOCAL_DEADLINE']): raise TimeoutError('240s deadline')
    if size()>CAP-150_000: raise RuntimeError('30MB artifact budget')

def require(condition,message):
    if not condition: raise RuntimeError(message)

def ah(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def reconstruct(noisy,clean,noise,selection,expected):
    """Exactly the frozen Worker float64 edit -> float32 assignment, no search."""
    import numpy as np
    require(noisy.dtype==np.float32 and noisy.shape==clean.shape==noise.shape==(480000,),'Input geometry')
    w=np.ones(selection['window_samples'],dtype=np.float64)
    fade=selection['internal_fade_samples'];ramp=.5-.5*np.cos(np.linspace(0,np.pi,fade));w[:fade]=ramp;w[-fade:]=ramp[::-1]
    require(ah(w)==selection['taper_float64_sha256'],'Frozen taper hash')
    original_peak=np.abs(noisy).max();oldres=noisy.astype(np.float64)-clean-noise
    require(float(original_peak)==selection['original_reference_peak'],'Original peak')
    require(np.abs(oldres).max()<=3/32768,'Original decomposition bound')
    conditions={'native':noisy};details={};segments={}
    rows={'target':selection['target'],**selection['selected_controls']}
    for name,row in rows.items():
        lo=row['start_sample'];hi=row['end_sample_exclusive'];alpha=row['alpha'] if name=='target' else row['alpha_if_energy_eligible']
        edit=alpha*w*clean[lo:hi];condition=noisy.copy()
        condition[lo:hi]=(noisy[lo:hi].astype(np.float64)-edit).astype(np.float32)
        outside=np.ones(len(noisy),bool);outside[lo:hi]=False
        residual=condition[lo:hi].astype(np.float64)-(clean[lo:hi]-edit)-noise[lo:hi]
        change=residual-oldres[lo:hi]
        roundbound=np.finfo(np.float32).eps*max(float(original_peak),float(np.abs(edit).max()))
        actual_delta=noisy[lo:hi].astype(np.float64)-condition[lo:hi].astype(np.float64)
        energy=float(actual_delta@actual_delta);target=selection['target']['weighted_clean_energy']
        require(np.array_equal(condition[outside],noisy[outside]),'Outside input changed '+name)
        require(np.abs(condition).max().tobytes()==original_peak.tobytes(),'Reference peak bits changed '+name)
        require(np.isfinite(condition).all(),'Nonfinite input')
        require(np.abs(change).max()<=roundbound and np.abs(residual).max()<=3/32768+roundbound,'Noise decomposition '+name)
        require(abs(energy-target)/target<1e-5,'Input perturbation energy '+name)
        require(ah(condition)==expected[name]['condition_float32_sha256'],'Frozen condition hash '+name)
        conditions[name]=condition
        details[name]={'samples':[lo,hi],'alpha':float(alpha),'condition_sha256':ah(condition),'mask_sha256':ah(w),'ideal_delta_sha256':ah(edit),'actual_delta_sha256':ah(actual_delta),'ideal_energy':float(edit@edit),'actual_energy':energy,'energy_relative_error':abs(energy-target)/target,'outside_bitwise_identical':True,'peak_bitwise_identical':True,'reference_peak':float(original_peak),'physical_noise_sha256':ah(noise),'noise_residual_change_max':float(np.abs(change).max()),'float32_roundoff_bound':float(roundbound),'mixture_residual_max':float(np.abs(residual).max()),'mixture_bound':3/32768+float(roundbound)}
        segments[name]={'mask':w,'ideal_removed_clean':edit,'actual_condition_delta':actual_delta}
    return conditions,details,segments

def summarize_probs(probs,token_id):
    import numpy as np
    require(probs.ndim==2 and probs.shape[0]==99 and probs.shape[1]==392,'Frozen classifier frame/vocabulary geometry')
    ids=probs.argmax(axis=-1);fixed=probs[35:36,token_id];near=probs[25:46,token_id]
    return {'target_index':11,'token':'d','token_id':token_id,'fixed_frames':[35,36],'neighborhood_frames':[25,46],'fixed_posterior_mean':float(fixed.mean()),'fixed_top1_fraction':float(np.mean(ids[35:36]==token_id)),'neighborhood_posterior_max':float(near.max()),'neighborhood_posterior_mean':float(near.mean()),'neighborhood_top1_fraction':float(np.mean(ids[25:46]==token_id)),'neighborhood_peak_frame':25+int(np.argmax(near)),'neighborhood_d_top1_frames':[int(i) for i in range(25,46) if ids[i]==token_id]}

def native_measurement_gate(probs,old,ids):
    import numpy as np
    require(probs.shape==(99,392),'Native classifier geometry')
    delta=float(np.max(np.abs(probs[:,ids]-old['probabilities'])))
    argmax_equal=bool(np.array_equal(probs.argmax(-1),old['argmax']))
    return {'max_cached_target_probability_difference':delta,'tolerance':1e-5,'all_frame_argmax_equal':argmax_equal,'frames':99,'vocabulary':392,'passed':delta<=1e-5 and argmax_equal}

def supervise():
    for name,h in read(OUT/'frozen_execution_manifest.json').items(): require(sha(OUT/name)==h,'Prepared file changed '+name)
    for name in ['STARTED.json','outputs','metrics.json','experiment_report.md','run.log','child_result.json','progress.json','launch_provenance.json']:
        require(not (OUT/name).exists() and not (OUT/name).is_symlink(),'Refuse existing run artifact '+name)
    require(os.environ.get('PATH','').split(':')[0]==str(ROOT/'env/bin'),'Worker PATH must start with existing env/bin')
    started=time.monotonic()
    with (OUT/'STARTED.json').open('x') as f: json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':[sys.executable,str(Path(__file__).resolve())],'frozen_execution_manifest_sha256':sha(OUT/'frozen_execution_manifest.json')},f,indent=2)
    env=os.environ.copy();env.update(LOCAL_DEADLINE=str(started+SECONDS),LOCAL_SUPERVISED=str(os.getpid()),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    reason=None
    with (OUT/'run.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            if time.monotonic()-started>=SECONDS or size()>CAP-200_000:
                reason='240s supervisor timeout' if time.monotonic()-started>=SECONDS else '30MB supervisor artifact cap'
                os.killpg(child.pid,signal.SIGKILL);child.wait();break
            time.sleep(.1)
    d=read(OUT/'child_result.json') if (OUT/'child_result.json').exists() else read(OUT/'progress.json') if (OUT/'progress.json').exists() else {'status':'failed','failures':[]}
    if reason or (child.returncode!=0 and d.get('status') not in ('failed','measurement_gate_failed')):
        d['status']='failed';d.setdefault('failures',[]).append({'supervisor_error':reason or 'child exit '+str(child.returncode)})
    d.update(supervisor_seconds=time.monotonic()-started,child_exit_code=child.returncode,artifact_bytes_before_final=size())
    d['artifact_manifest']={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in ('metrics.json','experiment_report.md')}
    write('metrics.json',d)
    (OUT/'experiment_report.md').write_text('Frozen 5243 local condition diagnostic: '+d['status']+'\n\nSee metrics.json for native raw/PCM and measurement gates, all measured variants, call counts, geometry/RNG/source guards and failures. No retry, training, PER/WER or human recovery flags. 200ms windows are not manually isolated /d/; contradictory conditions are OOD, and one case/recognizer cannot establish general recovery or hallucination.\n')
    print(json.dumps({'status':d['status'],'seconds':d['supervisor_seconds'],'calls':d.get('calls'),'failures':d.get('failures')},default=scalar))
    return 0 if d['status']=='completed' else 1

def runtime():
    require(os.environ.get('LOCAL_SUPERVISED')==str(os.getppid()),'Supervisor required')
    cfg=read(OUT/'execution_config.json');sel=read(OUT/'selection_frozen.json');base=read(cfg['baseline_metrics']);d={'status':'running','failures':[],'calls':{'frontend':0,'backend':0,'recognizer':0},'variants':{},'config':cfg}
    locks=cfg['source_locks'];out=OUT/'outputs'
    def verify_locks(label):
        hashes={}
        for path,h in locks.items():
            check();hashes[path]=sha(path);require(hashes[path]==h,label+' lock '+path)
        for name,h in read(OUT/'frozen_execution_manifest.json').items(): require(sha(OUT/name)==h,label+' prepared file '+name)
        d[label+'_source_hashes']=hashes
    try:
        check();verify_locks('pre')
        d['environment']={'python':sys.version,'executable':sys.executable,'PATH':os.environ.get('PATH'),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'HF_HUB_OFFLINE':os.environ.get('HF_HUB_OFFLINE'),'TRANSFORMERS_OFFLINE':os.environ.get('TRANSFORMERS_OFFLINE')}
        d['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();d['git_status_before']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        d['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.used,memory.total','--format=csv'],text=True,timeout=5)
        out.mkdir();d['phase']='imports';write('launch_provenance.json',d);write('progress.json',d)
        import numpy as np
        import soundfile as sf
        import torch
        sys.path.insert(0,str(ROOT));os.chdir(ROOT)
        import joint_enhance_by_dir as mod
        import sgmse.sampling as sampling
        d['environment'].update(torch=torch.__version__,numpy=np.__version__,soundfile=sf.__version__,cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark)
        def arr(v): return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        def desc(v):
            v=np.ascontiguousarray(arr(v));return {'sha256':ah(v),'shape':list(v.shape),'dtype':str(v.dtype)}
        def save(path,value):
            check();a=np.ascontiguousarray(arr(value));require(size()+a.nbytes+1024<CAP-200_000,'Output array budget');np.save(out/path,a,allow_pickle=False);return desc(a)
        def rng(): return {'cpu':desc(torch.get_rng_state()),**{'cuda'+str(i):desc(v) for i,v in enumerate(torch.cuda.get_rng_state_all())}}
        def restore_rng():
            torch.set_rng_state(torch.from_numpy(np.load(cfg['rng_cpu'],allow_pickle=False)))
            torch.cuda.set_rng_state_all([torch.from_numpy(np.load(p,allow_pickle=False)) for p in cfg['rng_cuda']])
            require(rng()==base['rng_before_backend'],'Restored before_backend RNG differs')
        def frontend_forbidden(self,*a,**kw):
            d['calls']['frontend']+=1;raise RuntimeError('Frontend forbidden: frozen raw anchor only')
        mod.JointSemambaSgmseModel._frontend_forward=frontend_forbidden
        noisy,sr=sf.read(cfg['noisy'],dtype='float32');clean,cs=sf.read(cfg['clean'],dtype='float64');noise,ns=sf.read(cfg['noise'],dtype='float64');require(sr==cs==ns==16000,'Source sample rates')
        conditions,details,segments=reconstruct(noisy,clean,noise,sel,cfg['expected_conditions'])
        for name,payload in segments.items():
            for key,value in payload.items(): save(name+'_'+key+'.npy',value)
        d['input_guards']=details
        frontend=np.load(cfg['anchor'],allow_pickle=False);require(frontend.shape==(1,1,480000) and frontend.dtype==np.float32,'Frozen anchor geometry')
        anchor=torch.from_numpy(frontend.reshape(-1)).to('cuda:0');require(desc(anchor)['sha256']==base['frontend_raw']['sha256'],'Raw anchor hash')
        provenraw=np.load(cfg['baseline_raw'],allow_pickle=False)
        load_kwargs={'map_location':'cpu','strict':True,'data_module_cls':mod.JointWaveDataModule,'base_dir':'.','batch_size':1,'num_workers':0,'dummy':True,'weights_only':False}
        check();d['phase']='backend_model_loading';write('progress.json',d);model=mod.JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint'],**load_kwargs);model.eval(no_ema=False).to('cuda:0')
        require(model.train_type=='mode2' and not model.residual_refine_enable,'Model mode2/nonresidual')
        d['model_loading']={'class':type(model).__module__+'.'+type(model).__qualname__,'checkpoint':cfg['checkpoint'],'use_ema':True,'map_location':'cpu','strict':True,'dummy_data_module':True}
        active={};old_sampler=sampling.get_pc_sampler
        def identify(fn):
            if isinstance(fn,functools.partial): fn=fn.func
            return {'module':getattr(fn,'__module__',type(fn).__module__),'name':getattr(fn,'__qualname__',type(fn).__qualname__),'source_sha256':cfg['sampler_source_sha256']}
        def traced_sampler(*args,**kw):
            check();record=active['record'];require(kw.get('trace_sink') is None,'Unexpected trace sink')
            sde=kw['sde'];record['sampler_settings']={'predictor':args[0],'corrector':args[1],'N':int(sde.N),'T':float(sde.T),'sde_class':type(sde).__module__+'.'+type(sde).__qualname__,'score_function':identify(kw['score_fn']),**{k:kw.get(k,default) for k,default in {'eps':.03,'snr':.1,'corrector_steps':1,'denoise':True,'probability_flow':False,'intermediate':False,'regeneration_strength':None}.items()}}
            require(kw.get('anchor_calibrator') is None and kw.get('score_calibrator') is None,'OFF calibrators')
            record['normalized_anchor']=desc(kw['y']);require(record['normalized_anchor']==base['first_trace']['y'],'Normalized anchor geometry mismatch')
            record['trace_calls']=0
            def trace(payload):
                record['trace_calls']+=1
                if record['trace_calls']==1:
                    record['first_trace']={'step_index':int(payload['step_index']),'t':arr(payload['t']).tolist(),'stepsize':float(payload['stepsize']),'x_t':desc(payload['x_t']),'x_next':desc(payload['x_next']),'x_mean':desc(payload['x_mean']),'rng_post_predictor':rng(),'semantics':'Callback is after predictor; x_t is pre-predictor after identity corrector; RNG here is post-predictor.'}
                    for key in ['step_index','t','stepsize','x_t','rng_post_predictor']:
                        require(record['first_trace'][key]==base['first_trace'][key],'First trace baseline mismatch '+key)
            kw['trace_sink']=trace;fn=old_sampler(*args,**kw)
            def run():
                record['rng_before_sampler']=rng();require(record['rng_before_sampler']==base['rng_before_sampler'],'Before sampler RNG mismatch');check();result=fn();check();return result
            return run
        sampling.get_pc_sampler=traced_sampler
        def pcm_gate(current,ref):
            a,sa=sf.read(current,dtype='int16');b,sb=sf.read(ref,dtype='int16');r={'bytes_equal':sha(current)==sha(ref),'pcm_equal':bool(np.array_equal(a,b)),'sample_rates':[sa,sb]}
            require(r['bytes_equal'] and r['pcm_equal'] and sa==sb==16000,'Historical PCM gate '+str(current));return r
        raw_outputs={}
        for name in ['native','target','left','right']:
            check()
            if name!='native': require(d.get('native_backend_gate_passed'),'Native gate required before interventions')
            record={};d['variants'][name]=record;active['record']=record
            condition=torch.from_numpy(conditions[name]).to('cuda:0');ref=condition.abs().max()
            require(np.float32(ref.item()).tobytes()==np.abs(noisy).max().tobytes(),'Torch reference peak bits '+name)
            record['reference_peak']=float(ref.item());record['anchor']=desc(anchor);record['condition']=desc(condition)
            restore_rng();record['rng_before_backend']=rng();record['enhance_settings']=cfg['enhance_settings']
            require(d['calls']['backend']<4,'Backend call cap');d['calls']['backend']+=1;d['phase']='backend_'+name;write('progress.json',d)
            with torch.inference_mode(): result,nfe,rtf=model.enhance(anchor,y_cond=condition,timeit=True,**cfg['enhance_settings'])
            torch.cuda.synchronize();record['nfe']=int(nfe);record['rtf']=float(rtf);record['rng_after_backend']=rng()
            require(record['rng_after_backend']==base['rng_after_backend'],'End backend RNG differs from proven native '+name)
            if name!='native': require(record['rng_after_backend']==d['variants']['native']['rng_after_backend'],'Random consumption differs across variants '+name)
            record['end_backend_rng_matches_native']=True
            require(nfe==30 and record.get('trace_calls')==30,'NFE/trace count '+name)
            require(np.isfinite(result).all() and result.shape==(480000,),'Output geometry/finite')
            raw_outputs[name]=result;record['raw_output']=save(name+'_raw.npy',result)
            record['clipped_sample_count']=int(np.count_nonzero(np.abs(result)>1))
            sf.write(out/(name+'.wav'),np.clip(result,-1.,1.),16000,subtype='PCM_16')
            saved,saved_sr=sf.read(out/(name+'.wav'),dtype='int16');sf.write(out/(name+'_context.wav'),saved[300000:332000],saved_sr,subtype='PCM_16')
            if name=='native':
                record['raw_exact_baseline']=bool(np.array_equal(result,provenraw));require(record['raw_exact_baseline'],'Native raw exact baseline failed')
                record['historical_full_pcm_gate']=pcm_gate(out/'native.wav',cfg['historical_off']);record['historical_clip_pcm_gate']=pcm_gate(out/'native_context.wav',cfg['historical_clip'])
                d['native_backend_gate_passed']=True
            else:
                require(d.get('native_backend_gate_passed'),'Native gate required before interventions')
                require(record['sampler_settings']==d['variants']['native']['sampler_settings'],'Sampler settings changed')
            write('progress.json',d);check()
        require(d['calls']['frontend']==0 and d['calls']['backend']==4,'Backend/frontend counts')
        sampling.get_pc_sampler=old_sampler
        del model,condition,anchor;gc.collect();torch.cuda.empty_cache();d['backend_released_before_classifier']=True
        regions={'target':[309600,312800],'left':[300960,304160],'right':[317280,320480]}
        outside=np.ones(480000,bool)
        for lo,hi in regions.values(): outside[lo:hi]=False
        for name,value in raw_outputs.items():
            diff=value.astype(np.float64)-raw_outputs['native'].astype(np.float64)
            d['variants'][name]['output_delta_vs_native']={key:{'rms':float(np.sqrt(np.mean(diff[lo:hi]**2))),'max_abs':float(np.abs(diff[lo:hi]).max())} for key,(lo,hi) in regions.items()}
            d['variants'][name]['output_delta_vs_native']['outside_all_three_windows']={'rms':float(np.sqrt(np.mean(diff[outside]**2))),'max_abs':float(np.abs(diff[outside]).max())}
            if name!='native':
                lo,hi=regions[name];active_outside=np.ones(480000,bool);active_outside[lo:hi]=False
                d['variants'][name]['output_delta_vs_native']['outside_edited_window']={'rms':float(np.sqrt(np.mean(diff[active_outside]**2))),'max_abs':float(np.abs(diff[active_outside]).max())}
        check()
        from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
        import transformers
        d['environment']['transformers']=transformers.__version__
        torch.manual_seed(0)
        d['phase']='classifier_model_loading';write('progress.json',d)
        proc=Wav2Vec2Processor.from_pretrained(cfg['classifier_model'],local_files_only=True)
        classifier=Wav2Vec2ForCTC.from_pretrained(cfg['classifier_model'],local_files_only=True).eval().to('cuda')
        require(proc.tokenizer.convert_tokens_to_ids('d')==cfg['d_token_id'],'Frozen d token mapping')
        cache=np.load(cfg['cached_probabilities'],allow_pickle=False)
        old={'probabilities':cache[cfg['cache_probabilities_key']],'argmax':cache[cfg['cache_argmax_key']]};ids=cache[cfg['cache_token_ids_key']]
        require(np.array_equal(ids,np.asarray(cfg['cached_token_ids'])),'Historical cache token IDs')
        for name in ['native','target','left','right']:
            check();wav,sr=sf.read(out/(name+'_context.wav'),dtype='float32');require(wav.shape==(32000,) and sr==16000,'Classifier clip geometry')
            inp=proc(wav,sampling_rate=sr,return_tensors='pt')
            processor_input=inp.input_values.detach().cpu().numpy()
            rec=d['variants'][name];rec['measurement_processing']={'clip_pcm_float32':{'descriptor':desc(wav),'mean':float(wav.astype(np.float64).mean()),'rms':float(np.sqrt(np.mean(wav.astype(np.float64)**2))),'std':float(wav.astype(np.float64).std())},'processor_input':{'descriptor':desc(processor_input),'mean':float(processor_input.astype(np.float64).mean()),'rms':float(np.sqrt(np.mean(processor_input.astype(np.float64)**2))),'std':float(processor_input.astype(np.float64).std())},'processing':'same official Wav2Vec2Processor on serialized PCM16 2s clip; no added normalization; global processor normalization and attention may affect fixed d posterior'}
            require(d['calls']['recognizer']<4,'Recognizer call cap');d['calls']['recognizer']+=1;d['phase']='classifier_'+name;write('progress.json',d)
            with torch.inference_mode():
                logits=classifier(inp.input_values.to('cuda')).logits[0].float();probs=torch.softmax(logits,dim=-1).cpu().numpy()
            rec=d['variants'][name];rec['logits']=save(name+'_ctc_logits.npy',logits);rec['probs']=save(name+'_ctc_probabilities.npy',probs);greedy=probs.argmax(-1).astype(np.int32);rec['greedy_ids']=save(name+'_ctc_greedy_ids.npy',greedy);rec['greedy_text']=proc.batch_decode([greedy.tolist()])[0]
            require(probs.shape==(99,392),'Classifier token/frame geometry')
            if name=='native':
                d['native_measurement_gate']=native_measurement_gate(probs,old,ids)
                if not d['native_measurement_gate']['passed']:
                    d['status']='measurement_gate_failed';write('progress.json',d);raise RuntimeError('Native cached CTC measurement mismatch; backend outputs retained; no interpretation or rerun')
            rec['fixed_d_metrics']=summarize_probs(probs,cfg['d_token_id']);write('progress.json',d)
        require(d['calls']=={'frontend':0,'backend':4,'recognizer':4},'Final call counts')
        d['status']='completed';d['phase']='completed'
    except BaseException as exc:
        if d['status']!='measurement_gate_failed': d['status']='failed'
        d['failures'].append({'error':repr(exc),'traceback':traceback.format_exc()});traceback.print_exc()
    finally:
        try:
            verify_locks('post');d['git_status_after']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        except BaseException as exc:
            d['status']='failed';d['failures'].append({'postcheck_error':repr(exc)})
        write('child_result.json',d)
    return 0 if d['status']=='completed' else 1

if __name__=='__main__':
    if sys.argv[1:]==['--child']: sys.exit(runtime())
    require(not sys.argv[1:],'No parameter overrides allowed')
    sys.exit(supervise())
