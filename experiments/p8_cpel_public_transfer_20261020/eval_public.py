#!/usr/bin/env python3
"""Authorized, frozen source/N/D evaluation through the actual public API."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def state_digest(state):
    """Same canonical name/shape/dtype/bytes encoding as the N/D frozen audit."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update((name + str(tuple(t.shape)) + str(t.dtype)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def aggregate(rows):
    metrics = ('normalized_wave_mse', 'si_sdr_db', 'stoi', 'pesq')
    arm_means = {}
    for arm in ('source', 'N', 'D'):
        arm_means[arm] = {}
        for key in metrics:
            values = [r[key] for r in rows if r['arm'] == arm and r.get(key) is not None]
            arm_means[arm][key] = dict(count=len(values), mean=sum(values)/len(values) if values else None)
    paired = {}
    for left, right in (('D', 'N'), ('N', 'source'), ('D', 'source')):
        paired[left+'-minus-'+right] = {}
        a = {r['dev_index']:r for r in rows if r['arm'] == left}
        b = {r['dev_index']:r for r in rows if r['arm'] == right}
        for key in metrics:
            values = [a[i][key]-b[i][key] for i in sorted(a.keys() & b.keys())
                      if a[i].get(key) is not None and b[i].get(key) is not None]
            paired[left+'-minus-'+right][key] = dict(count=len(values), mean=sum(values)/len(values) if values else None)
    return dict(absolute=arm_means, paired_differences=paired)


def main():
    cfg_path = BASE / 'config.yaml'
    cfg = read_json(cfg_path)
    source, registry = cfg['source'], cfg['registry']
    terminals, historical = cfg['terminals'], cfg['historical_metrics']
    limits = cfg['limits']
    expected = dict(config_sha256=sha(cfg_path), code_sha256=sha(__file__),
        registry_sha256=registry['sha256'], source_checkpoint_sha256=source['checkpoint_sha256'],
        terminal_sha256={a:terminals[a]['sha256'] for a in ('N','D')},
        core_sha256=source['code_sha256'],
        historical_metrics_sha256={a:historical[a]['sha256'] for a in ('N','D')})
    auth = read_json(BASE / cfg['authorization_file'])
    require(auth.get('status') == 'authorized_once', 'One-run Root authorization required')
    for key,value in expected.items():
        require(auth.get(key) == value, 'Authorization mismatch: '+key)
    require(not (BASE/'STARTED.json').exists(), 'One-shot evaluation already started; no retry')
    require(limits['seconds']==600 and limits['artifact_bytes']==4194304,
            'Frozen time/artifact limits differ')
    require(limits['source_pcm_reads']==128 and limits['frontend_calls']==64
            and limits['backend_calls']==192 and limits['score_calls']==5760
            and limits['source_model_loads']==1 and limits['terminal_state_loads']==2
            and limits['weight_loads']==3, 'Frozen call limits differ')
    started = time.monotonic()
    result = dict(status='running', identity=expected, rows=[], inputs=[], injections={},
        counts=dict(weight_loads=0, source_model_loads=0, terminal_state_loads=0, pcm_reads=0, frontend=0, backend=0, score=0),
        frozen_checks=[], source_pre={}, source_post={}, network_refusals=[], failure=None)
    protected = {}
    model = None
    handle = None
    original_sampler = None

    def resources():
        return dict(actual_artifact_bytes=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()),
                    free_bytes=shutil.disk_usage(BASE).free)

    def disk_guard():
        r = resources()
        require(r['actual_artifact_bytes'] < limits['artifact_bytes'], 'Artifact cap exceeded')
        require(r['free_bytes'] >= limits['free_disk_reserve_bytes'], 'Disk reserve below frozen limit')

    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        result['summary'] = aggregate(result['rows'])
        try:
            disk_guard()
            text = json.dumps(result, indent=2, allow_nan=False)
            require(len(text.encode()) < limits['artifact_bytes']//2, 'Metrics reservation exhausted')
            (BASE/'metrics.json').write_text(text)
            disk_guard()
        except BaseException as exc:
            result['status'] = 'failed'
            result['resource_failure'] = dict(message=str(exc), **resources())
            (BASE/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            raise

    def timeout(signum, frame):
        raise TimeoutError('600-second evaluation budget reached')

    disk_guard()
    (BASE/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(), **expected), indent=2))
    old_alarm = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(limits['seconds'])
    try:
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', PATH=cfg['native_path'])
        require(str(PROJECT/'env/bin') in os.environ['PATH'].split(os.pathsep), 'Ninja env/bin missing')
        import socket
        def no_network(*args, **kwargs):
            result['network_refusals'].append(dict(operation='connect_or_resolve', arguments=repr(args)[:300]))
            raise RuntimeError('Network forbidden')
        socket.socket.connect = no_network
        socket.socket.connect_ex = no_network
        socket.create_connection = no_network
        socket.getaddrinfo = no_network
        locked = {str(cfg_path):expected['config_sha256'], __file__:expected['code_sha256'],
                  registry['path']:registry['sha256'], source['checkpoint_path']:source['checkpoint_sha256'],
                  **source['code_sha256']}
        for arm in ('N','D'):
            locked[terminals[arm]['path']] = terminals[arm]['sha256']
            locked[historical[arm]['path']] = historical[arm]['sha256']
        for path,digest in locked.items():
            observed = sha(path)
            require(observed == digest, 'Frozen source hash mismatch: '+path)
            protected[path] = observed
        rows = [json.loads(s) for s in Path(registry['path']).read_text().splitlines()]
        require(len(rows)==64 and len({r['utt'] for r in rows})==64, 'Expected complete dev64 registry')
        for row in rows:
            for key in ('clean_path_loader','wav_path'):
                protected[row[key]] = sha(row[key])
        require(len({r[k] for r in rows for k in ('clean_path_loader','wav_path')})==128,
                'Expected 128 distinct S/X paths')
        result['source_pre'] = protected.copy()
        old = {arm:read_json(historical[arm]['path']) for arm in ('N','D')}
        for arm in ('N','D'):
            require(old[arm]['status']=='completed' and old[arm]['checkpoint_sha256']==source['checkpoint_sha256'],
                    'Historical N/D identity mismatch')
            require(old[arm]['backend_ema']['sha256']==terminals[arm]['sha256'], 'Terminal/history mismatch')
            require(old[arm]['frozen_state_pre']==old[arm]['frozen_state_post']
                    ==historical[arm]['expected_frozen_state'], 'Historical frozen state changed')
        require(old['N']['frozen_state_pre']==old['D']['frozen_state_pre'], 'Historical N/D frozen digests differ')
        save()
        sys.path.insert(0,str(PROJECT))
        import numpy as np
        import soundfile as sf
        import torch
        from sgmse.joint_model import JointSemambaSgmseModel
        from sgmse.joint_data_module import JointWaveScpDataModule
        from sgmse.model import ScoreModel
        try:
            from pystoi import stoi
        except ImportError:
            stoi = None
        try:
            from pesq import pesq
        except ImportError:
            pesq = None
        result['metric_preprocessing'] = ('No external clipping, centering, independent gain normalization or PCM serialization. '
            'MSE float64 arithmetic; SI-SDR centered dot-product projection. STOI(target,y,16000,extended=False) and '
            'PESQ(16000,target,y,wb) use unchanged installed library internal preprocessing, including library resampling/level handling.')
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        device=torch.device(cfg['device'])
        require(device.type=='cuda', 'Explicit CUDA device required')
        torch.cuda.set_device(device)
        result['counts']['weight_loads'] += 1
        result['counts']['source_model_loads'] += 1
        require(result['counts']['source_model_loads']<=limits['source_model_loads'], 'Source load cap')
        model=JointSemambaSgmseModel.load_from_checkpoint(source['checkpoint_path'],
            map_location='cpu', strict=True, weights_only=False,
            data_module_cls=JointWaveScpDataModule, frontend_ckpt=None)
        model.eval(no_ema=False)
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        require(not model.training and not model.frontend.training and not model.dnn.training, 'Expected eval state')
        require(model.train_type=='mode2' and model.frontend_name=='storm' and not model.residual_refine_enable
                and model.sde.__class__.__name__=='OUVESDE' and model.sde.sampler_type=='pc', 'Unexpected source topology')
        require(not getattr(model.sde,'residual_anchor_enabled',False), 'Residual SDE forbidden')
        require(model.loss_type=='score_matching' and model.loss_weighting=='sigma^2', 'Unexpected score parameterization')
        f,dm=model.frontend,model.data_module
        geometry=cfg['geometry']
        for k in ('n_fft','hop_length','transform_type','spec_abs_exponent','spec_factor'):
            require(getattr(f,k)==getattr(dm,k)==geometry[k], 'Frontend/model geometry mismatch: '+k)
        require(torch.equal(f.window.to(device),dm.window.to(device)), 'Window tensor mismatch')
        require(dm.num_frames==geometry['frames']==256 and model.sr==geometry['sample_rate']==16000
                and geometry['samples']==32640, 'Unexpected crop geometry')
        intrinsic=old['N']['frozen_state_pre']['intrinsic_names']
        def frozen_digest():
            named=dict(model.named_parameters())
            return dict(frontend_state_dict_sha256=state_digest(f.state_dict()),
                frontend_state_entries=len(f.state_dict()), intrinsic_names=intrinsic,
                intrinsic_sha256=state_digest({n:named[n] for n in intrinsic}))
        expected_frozen=old['N']['frozen_state_pre']
        def verify_frozen(stage):
            observed=frozen_digest()
            result['frozen_checks'].append(dict(stage=stage, digest=observed, matched=observed==expected_frozen))
            require(observed==expected_frozen, 'Frontend/intrinsic fixed state changed at '+stage)
        verify_frozen('source_loaded')
        states={'source':{n:t.detach().cpu().clone() for n,t in model.dnn.state_dict().items()}}
        source_shapes={n:(t.shape,t.dtype) for n,t in states['source'].items()}
        require(all(bool(torch.isfinite(t).all()) for t in states['source'].values()), 'Source dnn nonfinite')
        for arm in ('N','D'):
            result['counts']['weight_loads'] += 1
            result['counts']['terminal_state_loads'] += 1
            require(result['counts']['terminal_state_loads']<=limits['terminal_state_loads'], 'Terminal load cap')
            require(result['counts']['weight_loads']<=limits['weight_loads'], 'Total weight load cap')
            packet=torch.load(terminals[arm]['path'],map_location='cpu',weights_only=True)
            require(isinstance(packet,dict) and packet.get('schema')=='cpel-backend-ema-v1'
                    and packet.get('arm')==arm and packet.get('source_checkpoint_sha256')==source['checkpoint_sha256'],
                    'Terminal packet identity mismatch')
            state=packet['dnn']
            require(set(state)==set(source_shapes), 'Terminal state key mismatch')
            for name,t in state.items():
                require(torch.is_tensor(t) and (t.shape,t.dtype)==source_shapes[name]
                        and t.device.type=='cpu' and bool(torch.isfinite(t).all()), 'Terminal tensor mismatch: '+name)
            for name in intrinsic:
                local=name.removeprefix('dnn.')
                require(torch.equal(state[local],states['source'][local]), 'Terminal intrinsic W changed')
            states[arm]=state
        state_hashes={arm:state_digest(state) for arm,state in states.items()}
        result['state_hashes']=state_hashes
        def desc(t):
            a=t.detach().cpu().contiguous().numpy()
            return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
        cache=[]
        L=geometry['samples']
        with torch.no_grad():
            for index,row in enumerate(rows):
                disk_guard()
                waves=[]
                for key in ('clean_path_loader','wav_path'):
                    result['counts']['pcm_reads']+=1
                    require(result['counts']['pcm_reads']<=limits['source_pcm_reads'],'PCM read cap')
                    wave,sr=sf.read(row[key],dtype='float32',always_2d=True)
                    require(sr==16000 and wave.shape[1]==1 and np.isfinite(wave).all(), 'Invalid PCM source')
                    waves.append(torch.from_numpy(wave[:,0].copy()).reshape(1,-1))
                s,n=waves
                require(s.shape==n.shape, 'S/X length mismatch')
                start=(s.shape[-1]-L)//2
                if start>=0:
                    s,n=s[:,start:start+L],n[:,start:start+L]
                else:
                    pad=L-s.shape[-1]
                    s,n=(torch.nn.functional.pad(w,(pad//2,pad-pad//2)) for w in (s,n))
                    start=-(pad//2)
                r=float(n.abs().max().clamp_min(1e-12))
                clean=(s/r).contiguous()
                noisy=n.contiguous()
                N=model._sgmse_stft((n/r).to(device))
                result['counts']['frontend']+=1
                require(result['counts']['frontend']<=limits['frontend_calls'], 'Frontend call cap')
                A=f.forward_transformed_spec(N)
                require(tuple(A.shape)==(1,1,256,256) and A.dtype==torch.complex64 and bool(torch.isfinite(A).all()),
                        'Frontend spectrum invalid')
                front=(f.spec_to_waveform(A,length=L)*r).squeeze(1).cpu().contiguous()
                require(tuple(front.shape)==tuple(noisy.shape)==tuple(clean.shape)==(1,L)
                        and bool(torch.isfinite(front).all()), 'Cached input geometry/nonfinite')
                cache.append((front,noisy,clean,r))
                result['inputs'].append(dict(utt=row['utt'],dev_index=index,crop_start=start,
                    seed=20261019+100000+index,normfactor=r,physical_front=desc(front),
                    physical_noisy=desc(noisy),normalized_clean=desc(clean)))
        verify_frozen('inputs_cached')
        result['cache_bytes']=sum(t.numel()*t.element_size() for entry in cache for t in entry[:3])
        active={}
        def score_observer(module,args,output):
            result['counts']['score']+=int(args[0].shape[0])
            require(result['counts']['score']<=limits['score_calls'], 'Score call cap')
        handle=model.dnn.register_forward_hook(score_observer)
        original_sampler=model.get_pc_sampler
        def observe_sampler(predictor,corrector,y,**kwargs):
            require('anchor' not in active, 'Multiple public sampler entries')
            fn=kwargs['score_fn']
            require(getattr(fn,'func',None) is model, 'Unexpected Score function')
            active['anchor']=desc(y)
            active['condition']=desc(fn.keywords['y_cond'])
            require(predictor=='reverse_diffusion' and corrector=='none' and kwargs['N']==30
                    and kwargs['corrector_steps']==1 and kwargs['snr']==.5 and kwargs['intermediate'] is False,
                    'Public sampler settings changed')
            for k in ('regeneration_strength','anchor_calibrator','score_calibrator'):
                require(kwargs[k] is None,'Unexpected public calibration/regen')
            active['sampler_kwargs']=dict(predictor=predictor,corrector=corrector,N=kwargs['N'],
                corrector_steps=kwargs['corrector_steps'],snr=kwargs['snr'],intermediate=kwargs['intermediate'],
                regeneration_strength=None,anchor_calibrator=None,score_calibrator=None)
            run=original_sampler(predictor,corrector,y,**kwargs)
            def sampled():
                output,nfe=run()
                active['nfe']=int(nfe)
                return output,nfe
            return sampled
        model.get_pc_sampler=observe_sampler
        public_descriptions={}
        with torch.no_grad():
            for arm in ('source','N','D'):
                model.dnn.load_state_dict(states[arm],strict=True)
                injected=state_digest(model.dnn.state_dict())
                result['injections'][arm]=dict(expected=state_hashes[arm],actual=injected,matched=injected==state_hashes[arm])
                require(injected==state_hashes[arm], 'Injected state differs from CPU snapshot')
                verify_frozen('injected_'+arm)
                for index,(front,noisy,clean,r) in enumerate(cache):
                    disk_guard()
                    info=result['inputs'][index]
                    out=dict(arm=arm,utt=info['utt'],dev_index=index,crop_start=info['crop_start'],seed=info['seed'],errors={})
                    result['rows'].append(out)
                    require(desc(front)==info['physical_front'] and desc(noisy)==info['physical_noisy']
                            and desc(clean)==info['normalized_clean'], 'Cached input mutated')
                    active.clear()
                    require(sys.getprofile() is None,'Existing profiler would be overwritten')
                    def profile(frame,event,retval):
                        if event=='return' and frame.f_code is ScoreModel.enhance.__code__:
                            active['norm']=float(frame.f_locals['norm_factor'])
                            active['length']=int(frame.f_locals['T_orig'])
                    torch.manual_seed(info['seed']);torch.cuda.manual_seed_all(info['seed'])
                    before=result['counts']['score']
                    result['counts']['backend']+=1
                    require(result['counts']['backend']<=limits['backend_calls'],'Backend call cap')
                    sys.setprofile(profile)
                    try:
                        physical=model.enhance(front.to(device),y_cond=noisy.to(device),N=30,
                            sampler_type='pc',predictor='reverse_diffusion',corrector='none',corrector_steps=1,
                            snr=.5,timeit=False,rrsd=False,anchor_calib=False,anchor_v2=False,regen_strength=None)
                    finally:
                        sys.setprofile(None)
                    calls=result['counts']['score']-before
                    out.update(actual_score_calls=calls,nfe=active['nfe'],actual_norm=active['norm'],
                        actual_length=active['length'],anchor=active['anchor'],condition=active['condition'],
                        sampler_kwargs=active['sampler_kwargs'])
                    require(calls==active['nfe']==30 and active['norm']==r and active['length']==L,
                            'Actual public call/norm/length mismatch')
                    observed=(active['anchor'],active['condition'],active['sampler_kwargs'])
                    if arm=='source':
                        public_descriptions[index]=observed
                    else:
                        require(observed==public_descriptions[index], 'Cross-arm public inputs differ')
                    y=np.asarray(physical,dtype=np.float32).reshape(-1)/r
                    target=clean.numpy().reshape(-1)
                    require(y.shape==(L,) and np.isfinite(y).all(), 'Invalid public output')
                    out['output']=desc(torch.from_numpy(y))
                    out['normalized_wave_mse']=float(np.mean((y.astype(np.float64)-target)**2))
                    ref=target.astype(np.float64);est=y.astype(np.float64)
                    ref=ref-ref.mean();est=est-est.mean()
                    out['si_sdr_db']=None
                    try:
                        require(np.dot(ref,ref)>1e-20,'undefined: silent centered reference')
                        proj=ref*np.dot(est,ref)/np.dot(ref,ref)
                        numerator=np.dot(proj,proj);denominator=np.dot(est-proj,est-proj)
                        require(numerator>0 and denominator>0,'undefined/nonfinite projection-energy ratio')
                        value=float(10*np.log10(numerator/denominator))
                        require(np.isfinite(value),'Nonfinite SI-SDR')
                        out['si_sdr_db']=value
                    except Exception as exc:
                        out['errors']['si_sdr_db']=str(exc)
                    for key,fn in [('stoi',None if stoi is None else lambda:stoi(target,y,16000,extended=False)),
                                   ('pesq',None if pesq is None else lambda:pesq(16000,target,y,'wb'))]:
                        try:
                            require(fn is not None,'Metric package not installed; no installation allowed')
                            value=float(fn())
                            require(np.isfinite(value),'Undefined/nonfinite metric')
                            out[key]=value
                        except Exception as exc:
                            out[key]=None;out['errors'][key]=repr(exc)
                    if (index+1)%8==0:
                        save()
                require(state_digest(model.dnn.state_dict())==state_hashes[arm], 'DNN state changed during inference')
                verify_frozen('completed_'+arm)
                save()
        require(result['counts']==dict(weight_loads=3,source_model_loads=1,terminal_state_loads=2,pcm_reads=128,frontend=64,backend=192,score=5760),
                'Final call totals mismatch')
        require(len(result['rows'])==192 and not result['network_refusals'],'Incomplete/unclean execution')
        result['status']='completed'
    except BaseException as exc:
        result['status']='failed'
        result['failure']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        if handle is not None:
            handle.remove()
        if model is not None and original_sampler is not None:
            model.get_pc_sampler=original_sampler
        try:
            if model is not None and 'verify_frozen' in locals():
                verify_frozen('finally')
            result['source_post']={p:sha(p) for p in protected}
            require(result['source_post']==protected,'Protected source changed')
        except BaseException as exc:
            result['status']='failed';result['protection_failure']=str(exc)
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)
        try:
            save()
        except BaseException as exc:
            result['status']='failed'
            result.setdefault('resource_failure',dict(message=str(exc)))
            try:
                (BASE/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            except OSError:
                pass
    if result['status']!='completed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
