"""Single native replay, supervised and experiment-local. No imports/models on import."""
from pathlib import Path
import datetime, hashlib, io, json, os, signal, subprocess, sys, time, traceback
ROOT = Path('/data/coding/sgmse')
OUT = Path(__file__).resolve().parent
CAP = 15_000_000

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def default(value):
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, default=default, allow_nan=False)+'\n')

def size():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())

def check():
    if time.monotonic() >= float(os.environ['REPLAY_DEADLINE']):
        raise TimeoutError('180 second deadline')
    if size() > CAP:
        raise RuntimeError('15 MB artifact limit')

def supervise():
    frozen = json.loads((OUT/'frozen_manifest.json').read_text())
    for p, h in frozen.items():
        if sha(OUT/p) != h:
            raise RuntimeError('Prepared file changed: '+p)
    for name in ('STARTED.json', 'metrics.json', 'experiment_report.md', 'child_result.json', 'outputs', 'inputs', 'run.log'):
        if (OUT/name).exists() or (OUT/name).is_symlink():
            raise RuntimeError('Refuse existing replay artifact: '+name)
    started = time.monotonic()
    with (OUT/'STARTED.json').open('x') as f:
        json.dump({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': [sys.executable, str(Path(__file__).resolve())], 'frozen_manifest_sha256': sha(OUT/'frozen_manifest.json')}, f, indent=2)
    env = os.environ.copy()
    env['REPLAY_DEADLINE'] = str(started+180)
    env['REPLAY_SUPERVISED'] = str(os.getpid())
    reason = None
    with (OUT/'run.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            if time.monotonic()-started >= 180 or size() > CAP-100_000:
                reason = 'supervisor timeout180s' if time.monotonic()-started >= 180 else 'supervisor artifact cap'
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                break
            time.sleep(.1)
    result_path = OUT/'child_result.json'
    d = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'failed', 'failures': []}
    if reason or child.returncode != 0:
        d['status'] = 'failed'
        d.setdefault('failures', []).append({'error': reason or 'child exit '+str(child.returncode)})
    d.update(supervisor_wall_seconds=time.monotonic()-started, child_exit_code=child.returncode, artifacts_bytes_before_final_report=size())
    d['artifact_manifest'] = {str(p.relative_to(OUT)): {'sha256':sha(p), 'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in ('metrics.json','experiment_report.md')}
    write('metrics.json', d)
    (OUT/'experiment_report.md').write_text('Native 5243 replay gate: '+d['status']+'\n\nSingle authorized no-op reconstruction; historical comparison is PCM/bytes only. Newly recovered raw frontend is not a historical raw equality claim. No intervention, Router, classifier, evaluation or training.\n\nSee metrics.json for exact checks, captures, failures and invocation. Any failure stops without retry.\n')
    print(json.dumps({'status':d['status'], 'wall_seconds':d['supervisor_wall_seconds'], 'failures':d.get('failures',[])}, default=default))
    return 0 if d['status']=='passed' else 1

def child_main():
    if os.environ.get('REPLAY_SUPERVISED') != str(os.getppid()):
        raise RuntimeError('Child must be launched by supervisor')
    cfg = json.loads((OUT/'config.yaml').read_text())  # JSON is YAML 1.2
    d = {'status':'running', 'failures':[], 'frontend_calls':0, 'backend_calls':0, 'trace_calls':0, 'config':cfg, 'literal_historical_command_available':False, 'historical_raw_frontend_available':False}
    locks = cfg['source_locks']
    try:
        check()
        d['pre_hashes'] = {}
        for path, expected in locks.items():
            check()
            d['pre_hashes'][path] = sha(path)
            if d['pre_hashes'][path] != expected:
                raise RuntimeError('Source/checkpoint hash mismatch: '+path)
        d['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        d['git_status_before'] = subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True)
        d['environment'] = {'python':sys.version, 'executable':sys.executable, 'cwd':str(ROOT), 'PATH':os.environ.get('PATH'), 'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
        d['gpu_inventory'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.used,memory.total','--format=csv'], text=True, timeout=5)
        inp=OUT/'inputs'; out=OUT/'outputs'; inp.mkdir(); out.mkdir()
        (inp/cfg['basename']).symlink_to(cfg['noisy'])
        argv=['joint_enhance_by_dir.py','--ckpt',cfg['checkpoint'],'--in_dir',str(inp),'--out_dir',str(out),'--device','cuda:0','--sample_rate','16000','--train_type','mode2','--sampler_type','pc','--N','30','--corrector','none','--snr','0.5','--seed','2026','--use_ema']
        d['argv']=argv
        write('launch_provenance.json',d)
        check()
        import numpy as np
        import soundfile as sf
        import torch
        sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
        import joint_enhance_by_dir as mod
        import sgmse.sampling as sampling
        d['environment'].update(torch=torch.__version__, numpy=np.__version__, soundfile=sf.__version__, cuda_runtime=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(), cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark)
        def array(value):
            return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        def describe(value):
            a=np.ascontiguousarray(array(value))
            return {'dtype':str(a.dtype),'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
        def save(name,value):
            a=np.ascontiguousarray(array(value)); check()
            if size()+a.nbytes+1024 > CAP-150_000:
                raise RuntimeError('Insufficient artifact budget: '+name)
            np.save(out/name,a,allow_pickle=False)
            return describe(a)
        def rng(label):
            states={'cpu':torch.get_rng_state(), **{'cuda'+str(i):v for i,v in enumerate(torch.cuda.get_rng_state_all())}}
            return {k:save(label+'_'+k+'.npy',v) for k,v in states.items()}
        def pcm_gate(path, ref):
            a,sr=sf.read(path,dtype='int16');b,bsr=sf.read(ref,dtype='int16')
            r={'bytes_equal':sha(path)==sha(ref), 'pcm_equal':bool(np.array_equal(a,b)), 'sample_rates':[sr,bsr], 'samples':[len(a),len(b)]}
            if not (r['bytes_equal'] and r['pcm_equal'] and sr==bsr==16000):
                d.setdefault('mismatches',[]).append(r)
                raise RuntimeError('PCM/bytes gate mismatch: '+str(path))
            return r
        oldfront=mod.JointSemambaSgmseModel._frontend_forward
        def frontend(self,*a,**kw):
            check();d['frontend_calls']+=1
            if d['frontend_calls']!=1: raise RuntimeError('Frontend call limit')
            d['rng_before_frontend']=rng('rng_before_frontend')
            result=oldfront(self,*a,**kw)
            d['rng_after_frontend']=rng('rng_after_frontend')
            return result
        mod.JointSemambaSgmseModel._frontend_forward=frontend
        oldmatch=mod._match_length
        def match(a,b):
            noisy_reference_peak=float(b.abs().max().item())
            z=oldmatch(a,b)
            if z.shape[-1]!=480000: raise RuntimeError('Expected full30s frontend')
            d['frontend_raw']=save('frontend_raw.npy',z)
            bio=io.BytesIO();sf.write(bio,array(z).reshape(-1),16000,format='WAV',subtype='PCM_16')
            historical=Path(cfg['frontend_pcm']).read_bytes()
            frontend_pcm,sr=sf.read(io.BytesIO(bio.getvalue()),dtype='int16')
            historical_frontend_pcm,bsr=sf.read(io.BytesIO(historical),dtype='int16')
            d['frontend_gate']={'bytes_equal':bio.getvalue()==historical,'pcm_equal':bool(np.array_equal(frontend_pcm,historical_frontend_pcm)),'sample_rates':[sr,bsr],'reencoded_sha256':hashlib.sha256(bio.getvalue()).hexdigest()}
            if not all((d['frontend_gate']['bytes_equal'],d['frontend_gate']['pcm_equal'],sr==bsr==16000)):
                raise RuntimeError('Frontend PCM/bytes gate mismatch')
            d['normalization_reference_peak']=noisy_reference_peak
            write('progress.json',d);check();return z
        mod._match_length=match
        oldsampler=sampling.get_pc_sampler
        def trace(payload):
            d['trace_calls']+=1
            if d['trace_calls']==1:
                d['first_trace']={'step_index':int(payload['step_index']), 'semantics':'Callback is after predictor; x_t is pre-predictor after corrector; RNG captured here is post-predictor. Not mislabeled initial xT.', 't':array(payload['t']).tolist(), 'stepsize':float(payload['stepsize']), 'x_t':describe(payload['x_t']), 'x_next':describe(payload['x_next']), 'x_mean':describe(payload['x_mean']), 'y':describe(payload['y']), 'rng_post_predictor':rng('rng_first_trace_post_predictor')}
        def get_sampler(*a,**kw):
            check()
            if kw.get('trace_sink') is not None: raise RuntimeError('Unexpected existing trace hook')
            d['sampler_settings']={k:(describe(v) if torch.is_tensor(v) else str(v) if k in ('sde','score_fn') else v) for k,v in kw.items() if k not in ('anchor_calibrator','score_calibrator')}
            d['sampler_calibrators']={k:kw.get(k) is not None for k in ('anchor_calibrator','score_calibrator')}
            if any(d['sampler_calibrators'].values()): raise RuntimeError('OFF requires no calibrators')
            kw['trace_sink']=trace
            fn=oldsampler(*a,**kw)
            def run():
                d['rng_before_sampler']=rng('rng_before_sampler');check()
                result=fn()
                d['rng_after_sampler']=rng('rng_after_sampler');check();return result
            return run
        sampling.get_pc_sampler=get_sampler
        oldenhance=mod.JointSemambaSgmseModel.enhance
        def enhance(self,*a,**kw):
            check();d['backend_calls']+=1
            if d['backend_calls']!=1 or 'frontend_gate' not in d: raise RuntimeError('Backend call gate')
            if self.train_type!='mode2' or self.residual_refine_enable: raise RuntimeError('Unexpected loaded model mode')
            for key,expected in {'sampler_type':'pc','predictor':'reverse_diffusion','corrector':'none','N':30,'snr':0.5,'rrsd':False,'anchor_calib':False,'anchor_v2':False,'regen_strength':None}.items():
                if kw.get(key)!=expected: raise RuntimeError('Unexpected setting '+key)
            d['enhance_settings']={k:v for k,v in kw.items() if k!='y_cond'}
            d['backend_anchor']=describe(a[0]);d['backend_condition']=describe(kw['y_cond'])
            d['rng_before_backend']=rng('rng_before_backend');kw['timeit']=True
            y,nfe,rtf=oldenhance(self,*a,**kw)
            d['backend_raw']=save('backend_raw.npy',y);d['nfe']=int(nfe);d['rtf']=float(rtf)
            d['rng_after_backend']=rng('rng_after_backend')
            if nfe!=30 or d['trace_calls']!=30: raise RuntimeError('NFE/trace count gate')
            check();return y
        mod.JointSemambaSgmseModel.enhance=enhance
        sys.argv=argv;mod.main();check()
        d['off_full_gate']=pcm_gate(out/cfg['basename'],cfg['off_pcm'])
        audio,sr=sf.read(out/cfg['basename'],dtype='int16')
        sf.write(out/'off_context_18p75_20p75.wav',audio[300000:332000],sr,subtype='PCM_16')
        d['off_clip_gate']=pcm_gate(out/'off_context_18p75_20p75.wav',cfg['off_clip'])
        if (d['frontend_calls'],d['backend_calls'],d['nfe'])!=(1,1,30): raise RuntimeError('Call count gate')
        d['status']='passed'
    except BaseException as exc:
        d['status']='failed'; d['failures'].append({'error':repr(exc),'traceback':traceback.format_exc()});traceback.print_exc()
    finally:
        try:
            check();d['post_hashes']={}
            for path,expected in locks.items():
                check();d['post_hashes'][path]=sha(path)
                if d['post_hashes'][path]!=expected: raise RuntimeError('Post hash changed: '+path)
            for path,expected in json.loads((OUT/'frozen_manifest.json').read_text()).items():
                if sha(OUT/path)!=expected: raise RuntimeError('Prepared file changed during replay: '+path)
            d['git_status_after']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        except BaseException as exc:
            d['status']='failed';d['failures'].append({'postcheck_error':repr(exc)})
        write('child_result.json',d)
    return 0 if d['status']=='passed' else 1

if __name__=='__main__':
    if sys.argv[1:]==['--child']:
        sys.exit(child_main())
    if sys.argv[1:]: raise RuntimeError('No parameter overrides permitted')
    sys.exit(supervise())
