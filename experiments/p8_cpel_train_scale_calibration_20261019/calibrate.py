"""Root-gated, no-update CPEL gradient-scale measurement; not a training run."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
import traceback

BASE=Path(__file__).resolve().parent
PROJECT=Path('/data/coding/sgmse')
PARENT=PROJECT/'experiments/p8_cpel_mp_pilot_20261019'
HELPERS=['run_pilot.py','loss_helpers.py','editing.py']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def summarize_ratios(records):
    """All registered probes, zeros retained. Never silently drop invalid rows."""
    import statistics
    reasons=[]
    for i,row in enumerate(records):
        if 'active' not in row:reasons.append(f'probe {i}: missing activity assessment')
        if row.get('M_norm') is None or row['M_norm']<=0:reasons.append(f'probe {i}: zero/undefined M norm')
        value=row.get('ratio')
        if value is None or not math.isfinite(value):reasons.append(f'probe {i}: undefined/nonfinite ratio')
    if len(records)!=16:reasons.append('not all 16 probes completed')
    if reasons:return dict(status='no_calibration',reasons=reasons)
    median=float(statistics.median(row['ratio'] for row in records))
    if not math.isfinite(median) or median<=0:
        return dict(status='no_calibration',median_ratio=median,reasons=['median ratio is zero or nonfinite'])
    raw=.01/median
    suggested=min(100.,max(.1,raw))
    return dict(status='suggestion_only_not_applied',probes=len(records),median_ratio=median,
                target_effective_ratio=.01,lambda_raw=raw,lambda_suggested=min(100.,max(.1,raw)),
                clip=[.1,100.],zeros_retained=True,
                effective_ratios_current_lambda_0_1=[.1*r['ratio'] for r in records],
                effective_ratios_suggested_lambda=[suggested*r['ratio'] for r in records],
                proposed_effective_median=suggested*median,
                interpretation='gradient norms, not Adam update ratios or semantic reliability')


def gate():
    cfg=json.loads((BASE/'config.yaml').read_text())
    auth=json.loads((BASE/'root_calibration_authorization.json').read_text())
    if auth.get('status')!='authorized':raise RuntimeError('Root calibration authorization required')
    if auth['calibration_config_sha256']!=sha(BASE/'config.yaml'):
        raise RuntimeError('Calibration config lock mismatch')
    if auth['code_sha256']!=sha(BASE/'calibrate.py'):raise RuntimeError('Calibration code lock mismatch')
    if auth['checkpoint_sha256']!=cfg['checkpoint']['sha256']:
        raise RuntimeError('Checkpoint authorization mismatch')
    for name in HELPERS:
        expected=cfg['parent_helpers'][name]['sha256']
        if Path(cfg['parent_helpers'][name]['path'])!=PARENT/name:
            raise RuntimeError('Unexpected parent helper location')
        if sha(PARENT/name)!=expected or auth['parent_helpers_sha256'][name]!=expected:
            raise RuntimeError('Parent helper lock mismatch: '+name)
    if sha(cfg['pilot_config']['path'])!=cfg['pilot_config']['sha256']:
        raise RuntimeError('Frozen parent config changed')
    fixed=dict(seed=20261019,device='cuda:0',source_audio_limit=16,crop_samples=32640,sample_rate=16000,
        draws_per_record=2,native_time_range=[.03,1.],aux_time_range=[.3,.7],
        target_gradient_norm_ratio=.01,current_lambda_reference=.1,lambda_clip_range=[.1,100.],apply_proposal=False)
    for k,v in fixed.items():
        if cfg[k]!=v:raise RuntimeError('Fixed calibration/config mismatch: '+k)
    if cfg['limits']!=dict(seconds=300,max_directory_bytes=2097152,minimum_free_disk_bytes=1073741824,
        network=False,install=False,dev_or_final_test_access=False,automatic_retry=False):
        raise RuntimeError('Fixed calibration caps mismatch')
    if cfg['expected_counts']!=dict(records=8,probes=16,score_sample_forwards=64,reverse_traversals=32,
        normal_sampler_calls=0,optimizer_steps=0,ema_updates=0,parameter_updates=0):
        raise RuntimeError('Fixed call counts mismatch')
    registry=BASE/cfg['registry']['path']
    if sha(registry)!=cfg['registry']['sha256'] or auth['registry_sha256']!=cfg['registry']['sha256']:
        raise RuntimeError('Registry lock mismatch')
    rows=[json.loads(s) for s in registry.read_text().splitlines()]
    if len(rows)!=8:raise RuntimeError('Exactly eight fixed registry records required')
    from collections import Counter
    if sorted(Counter(r['priority_bin'] for r in rows).values())!=[2,2,2,2]:
        raise RuntimeError('Expected exactly two frozen records in each of four bins')
    if (BASE/'CALIBRATION_STARTED.json').exists() or (BASE/'metrics.json').exists():
        raise RuntimeError('Calibration already started; no automatic rerun')
    return cfg,rows


def execute(cfg,rows):
    begin=time.monotonic();protected={};model=None
    result=dict(status='running',probes=[],calls=dict(score=0,reverse=0,optimizer=0,ema_updates=0),
                network_attempts=[],pcm_reads=[])
    def deadline(signum,frame):raise TimeoutError('Calibration 300-second limit')
    old_handler=signal.signal(signal.SIGALRM,deadline);signal.alarm(290)
    def guard():
        size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
        if size>2*1024**2:raise RuntimeError('Calibration 2 MiB directory cap')
        if shutil.disk_usage(BASE).free<1024**3:raise RuntimeError('Calibration 1 GiB free reserve')
    with (BASE/'CALIBRATION_STARTED.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),config_sha256=sha(BASE/'config.yaml')),f)
    try:
        guard()
        import socket
        def refuse(sock,address,*args,**kwargs):
            result['network_attempts'].append(repr(address))
            raise RuntimeError('Calibration network connection forbidden')
        socket.socket.connect=refuse;socket.socket.connect_ex=refuse
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PATH=cfg['native_path'])
        if str(PROJECT/'env/bin') not in os.environ['PATH'].split(os.pathsep):
            raise RuntimeError('Project env/bin missing from authorized PATH')
        audio=[r[k] for r in rows for k in ['clean_path_loader','wav_path']]
        if len(set(audio))!=16:raise RuntimeError('Expected sixteen distinct S/X sources')
        paths=[cfg['checkpoint']['path']]+audio+[str(PARENT/n) for n in HELPERS]+[cfg['pilot_config']['path']]
        protected={p:sha(p) for p in paths}
        result['protected_pre_sha256']=protected.copy()
        if protected[cfg['checkpoint']['path']]!=cfg['checkpoint']['sha256']:
            raise RuntimeError('Source checkpoint bytes differ')
        import numpy as np
        import soundfile as sf
        import torch
        sys.path.insert(0,str(PARENT))
        from run_pilot import enable_backend_training
        from loss_helpers import native_dsm,common_point,pair_loss
        from editing import crop_pair,edit_component,spectrum_mask
        sys.path.insert(0,str(PROJECT))
        from sgmse.joint_model import JointSemambaSgmseModel
        from sgmse.joint_data_module import JointWaveScpDataModule
        device=torch.device(cfg['device']);torch.cuda.set_device(device)
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);np.random.seed(cfg['seed'])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats(device)
        model=JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint']['path'],map_location='cpu',
            strict=True,weights_only=False,data_module_cls=JointWaveScpDataModule,frontend_ckpt=None)
        model.eval(no_ema=False)
        if (model.train_type!='mode2' or model.sde.__class__.__name__!='OUVESDE'
            or model.loss_type!='score_matching' or model.loss_weighting!='sigma^2'
            or model.frontend_name!='storm' or model.residual_refine_enable
            or bool(getattr(model.sde,'residual_anchor_enabled',False))):
            raise RuntimeError('Unexpected source model/task topology')
        result['intrinsic_fixed_parameters']=enable_backend_training(model)
        model.rrsd_freeze_base=False;model.rrsd_freeze_router=True;model.rrsd_disable_aux_loss=True
        model.frontend_trainable=False;model.to(device);model.dnn.eval();model.frontend.eval()
        dm=model.data_module
        if (int(model.sr)!=16000 or int(dm.n_fft)!=510 or int(dm.hop_length)!=128
            or int(dm.num_frames)!=256 or dm.transform_type!='exponent'
            or dm.spec_abs_exponent!=.5 or dm.spec_factor!=.15):
            raise RuntimeError('Unexpected source crop/transform')
        params=[p for p in model.dnn.parameters() if p.requires_grad]
        if not params or any(p.dtype!=torch.float32 for p in params):raise RuntimeError('Expected native float32 trainables')
        def digest():
            h=hashlib.sha256()
            for group,iterator in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
                for n,t in iterator:
                    h.update((group+n+str(tuple(t.shape))+str(t.dtype)).encode())
                    h.update(t.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        result['model_pre_digest']=digest()
        def score_hook(module,args,output):result['calls']['score']+=int(args[0].shape[0])
        hook=model.dnn.register_forward_hook(score_hook)
        def load_audio(path):
            if path not in audio:raise RuntimeError('Unregistered audio access')
            a,sr=sf.read(path,dtype='float32',always_2d=True);result['pcm_reads'].append(path)
            if sr!=16000 or a.shape[1]!=1 or not np.isfinite(a).all():raise RuntimeError('Invalid mono16k source')
            return torch.from_numpy(a[:,0].copy()).reshape(1,-1)
        def spec(x):return model._sgmse_stft(x.to(device))
        def front(n):
            with torch.no_grad():return model.frontend.forward_transformed_spec(n).detach()
        def score(u,a,t,n):return model(u,a,t,y_cond=n)
        def rnd(shape,gen):return torch.randn(shape,dtype=torch.complex64,device=device,generator=gen)
        for index,row in enumerate(rows):
            guard()
            s,x=[load_audio(row[k]) for k in ['clean_path_loader','wav_path']]
            s,x,start,factor=crop_pair(s,x,32640,training=False)
            delta,center=edit_component(s,'hann')
            c1,c0,n1,n0=spec(s),spec(s-delta),spec(x),spec(x-delta)
            f1,f0=front(n1),front(n0)
            mask=spectrum_mask(c1,center,128)
            for draw in range(2):
                probe_seed=int(cfg['seed'])+1000*index+draw
                gen=torch.Generator(device=device).manual_seed(probe_seed)
                cpu_gen=torch.Generator().manual_seed(probe_seed)
                t=.03+.97*torch.rand(1,device=device,generator=gen)
                a,b=model.sde.weights(t);sig=model.sde._std(t).reshape(-1,1,1,1)
                eps=rnd(c1.shape,gen)
                q1=score(a*c1+b*f1+sig*eps,f1,t,n1);q0=score(a*c0+b*f0+sig*eps,f0,t,n0)
                native=(native_dsm(q1,eps,sig)+native_dsm(q0,eps,sig))/2
                ta=.3+.4*torch.rand(1,device=device,generator=gen)
                aa,bb=model.sde.weights(ta);ss=model.sde._std(ta).reshape(-1,1,1,1)
                parent=int(torch.randint(2,(),generator=cpu_gen))
                cp,anchor=(c0,f0) if parent==0 else (c1,f1)
                u=(aa*cp+bb*anchor+ss*rnd(cp.shape,gen)).detach()
                qa1,qa0=score(u,anchor,ta,n1),score(u,anchor,ta,n0)
                aux=(common_point(qa1,u,c1,anchor,aa,ss)+common_point(qa0,u,c0,anchor,aa,ss))/2
                pair,stats=pair_loss(qa1,qa0,c1-c0,mask,aa,ss);m=native+aux
                if not all(bool(torch.isfinite(v)) for v in [native,aux,pair,m]):raise RuntimeError('Nonfinite probe loss')
                gm=torch.autograd.grad(m,params,retain_graph=True,allow_unused=True);result['calls']['reverse']+=1
                gp=torch.autograd.grad(pair,params,allow_unused=True);result['calls']['reverse']+=1
                m2=p2=dot=0.;finite=True
                for g1,g2 in zip(gm,gp):
                    if g1 is not None:
                        finite=finite and bool(torch.isfinite(g1).all());m2+=float(g1.detach().double().square().sum())
                    if g2 is not None:
                        finite=finite and bool(torch.isfinite(g2).all());p2+=float(g2.detach().double().square().sum())
                    if g1 is not None and g2 is not None:dot+=float((g1.detach().double()*g2.detach().double()).sum())
                finite=finite and all(math.isfinite(v) for v in [m2,p2,dot])
                mn=math.sqrt(m2) if finite else None;pn=math.sqrt(p2) if finite else None
                if any(p.grad is not None for p in model.parameters()):raise RuntimeError('autograd.grad unexpectedly accumulated parameter .grad')
                ratio=None if not finite or mn<=0 else pn/mn
                if ratio is not None and not math.isfinite(ratio):ratio=None
                cosine=None if not finite or mn<=0 or pn<=0 else dot/(mn*pn)
                result['probes'].append(dict(index=index,utt=row['utt'],bin=row['priority_bin'],draw=draw,seed=probe_seed,
                    crop_start=start,normfactor=factor,event_center=center,t=float(t),aux_t=float(ta),parent=parent,
                    native=float(native.detach()),aux=float(aux.detach()),M=float(m.detach()),pair=float(pair.detach()),
                    active=stats['valid'],pair_stats=stats,M_norm=mn,pair_norm=pn,ratio=ratio,cosine=cosine,
                    gradient_finite=finite,source_controls_detached=not any(v.requires_grad for v in [c1,c0,n1,n0,f1,f0,u])))
                del gm,gp,q1,q0,qa1,qa0,native,aux,m,pair
                write(BASE/'metrics.json',result);guard()
            del c1,c0,n1,n0,f1,f0
        hook.remove()
        if result['calls']!=dict(score=64,reverse=32,optimizer=0,ema_updates=0):raise RuntimeError('Unexpected scientific call counts')
        result['model_post_digest']=digest()
        if result['model_post_digest']!=result['model_pre_digest']:raise RuntimeError('Model state changed')
        if len(result['pcm_reads'])!=16:raise RuntimeError('Unexpected source PCM decode count')
        if result['network_attempts']:raise RuntimeError('Blocked network requests occurred')
        result['calibration']=summarize_ratios(result['probes'])
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        result['status']='completed_no_updates'
    except BaseException as error:
        result['status']='failed';result['failure']=repr(error);result['traceback']=traceback.format_exc()
        result['calibration']=dict(status='no_calibration',reasons=['execution incomplete or failed'])
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        if model is not None and 'digest' in locals() and 'model_pre_digest' in result:
            if 'model_post_digest' not in result:result['model_post_digest']=digest()
            if result['model_post_digest']!=result['model_pre_digest']:result['status']='failed'
        result['protected_post_sha256']={p:sha(p) for p in protected}
        result['protected_unchanged']=protected==result['protected_post_sha256']
        if not result['protected_unchanged']:result['status']='failed'
        result['seconds']=time.monotonic()-begin
        if result['seconds']>300:result['status']='failed'
        if result['status']=='failed':result['calibration']=dict(status='no_calibration',reasons=['execution or preservation failed'])
        write(BASE/'metrics.json',result)
        (BASE/'experiment_report.md').write_text('# CPEL train-scale calibration: '+result['status']+'\n\n'
            'Eight fixed training records, two seeded stochastic probes each; no parameter/optimizer/EMA update, '
            'no normal inference or audio/weight exports. metrics.json preserves all sixteen rows, zero/invalid '
            'coverage and gradient geometry. Any coefficient is a suggestion only, never applied to the parent '
            'configuration or treated as enhancement evidence.\n')
        guard()
    return 0 if result['status']=='completed_no_updates' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-calibration',required=True,action='store_true')
    parser.parse_args()
    cfg,rows=gate()
    raise SystemExit(execute(cfg,rows))
