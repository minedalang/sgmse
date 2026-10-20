"""Root-gated one-pair integration preflight. Never trains or saves weights."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from run_pilot import BASE, PROJECT, sha, write, validate_fixed_config


def gate():
    pc=json.loads((BASE/'preflight_config.json').read_text())
    cfg=json.loads((BASE/pc['pilot_config_path']).read_text())
    validate_fixed_config(cfg)
    auth=json.loads((BASE/pc['authorization_file']).read_text())
    if auth.get('status')!='authorized':raise RuntimeError('No Root preflight authorization')
    if auth['preflight_config_sha256']!=sha(BASE/'preflight_config.json'):
        raise RuntimeError('Preflight config lock mismatch')
    if pc['pilot_config_sha256']!=sha(BASE/pc['pilot_config_path']) or auth['pilot_config_sha256']!=pc['pilot_config_sha256']:
        raise RuntimeError('Pilot config lock mismatch')
    for name in ['run_pilot.py','loss_helpers.py','editing.py','preflight_pilot.py']:
        if auth['code_sha256'][name]!=sha(BASE/name):raise RuntimeError('Code lock mismatch: '+name)
    if auth['checkpoint_sha256']!=cfg['checkpoint']['sha256']:
        raise RuntimeError('Source checkpoint authorization mismatch')
    for k,v in dict(train_registry_index=0,dev_registry_index=0,source_audio_limit=6,
        native_training_time=.5,auxiliary_time=.5,common_parent=1,
        max_seconds=300,max_additional_output_bytes=2097152,expected_backward_calls=2,
        minimum_free_disk_bytes=1073741824,
        expected_score_sample_forwards=dict(train_native=2,train_auxiliary=2,normal_dev=30),
        optimizer_steps=0,ema_updates=0,parameter_updates=0,weight_or_audio_exports=False,
        network_access=False,install_packages=False,seed=20261019,device='cuda:0').items():
        if pc[k]!=v:raise RuntimeError('Fixed preflight mismatch: '+k)
    if pc['crop_policy']!='fixed center crop for both preflight records; no event or sample reselection':
        raise RuntimeError('Preflight must use both fixed center crops')
    selected={}
    for split in ['train','dev']:
        item=cfg['registries'][split];path=BASE/item['path']
        if sha(path)!=item['sha256'] or auth['registry_sha256'][split]!=item['sha256']:
            raise RuntimeError('Registry lock mismatch')
        with path.open() as f:selected[split]=json.loads(next(f))
    if (BASE/'root_execution_authorization.json').exists():
        raise RuntimeError('Preflight must not run in a training-authorized directory')
    if (BASE/'PREFLIGHT_STARTED.json').exists() or (BASE/'preflight_result.json').exists():
        raise RuntimeError('Preflight already started; no retry')
    return pc,cfg,selected


def main():
    pc,cfg,rows=gate()
    begin=time.monotonic()
    old_files={str(p):p.stat().st_size for p in BASE.rglob('*') if p.is_file()}
    def guard():
        size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and str(p) not in old_files)
        if size>pc['max_additional_output_bytes']:raise RuntimeError('Preflight 2 MiB new output cap')
        import shutil
        if shutil.disk_usage(BASE).free<pc['minimum_free_disk_bytes']:raise RuntimeError('Preflight disk reserve')
    def deadline(signum,frame):raise TimeoutError('Preflight 300-second cap')
    old_handler=signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
    with (BASE/'PREFLIGHT_STARTED.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),preflight_config_sha256=sha(BASE/'preflight_config.json')),f)
    result=dict(status='running',pid=os.getpid(),calls=dict(train_native=0,train_auxiliary=0,
        normal_dev=0,backward=0,optimizer_steps=0,ema_updates=0),network_attempts=[],pcm_reads=[])
    protected={};model=None
    try:
        import socket
        def refuse(sock,address,*args,**kwargs):
            result['network_attempts'].append(repr(address))
            raise RuntimeError('Preflight network connection forbidden')
        socket.socket.connect=refuse;socket.socket.connect_ex=refuse
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        audio_paths=[row[k] for row in rows.values() for k in ['clean_path_loader','wav_path','noise1_path']]
        if len(set(audio_paths))!=6:raise RuntimeError('Expected exactly six distinct selected audio sources')
        paths=[cfg['checkpoint']['path']]+audio_paths+[str(BASE/n) for n in
            ['run_pilot.py','loss_helpers.py','editing.py','preflight_pilot.py']]
        for path in paths:protected[path]=sha(path)
        result['protected_pre_sha256']=protected.copy()
        if protected[cfg['checkpoint']['path']]!=cfg['checkpoint']['sha256']:
            raise RuntimeError('Checkpoint bytes mismatch')
        import numpy as np
        import soundfile as sf
        import torch
        from editing import crop_pair,edit_component,spectrum_mask
        from loss_helpers import native_dsm,common_point,pair_loss
        sys.path.insert(0,str(PROJECT))
        from sgmse.joint_model import JointSemambaSgmseModel
        from sgmse.joint_data_module import JointWaveScpDataModule
        seed=cfg['seed'];device=torch.device(cfg['device'])
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        model=JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint']['path'],map_location='cpu',
            strict=True,weights_only=False,data_module_cls=JointWaveScpDataModule,frontend_ckpt=None)
        model.eval(no_ema=False)
        if (model.train_type!='mode2' or model.sde.__class__.__name__!='OUVESDE'
            or model.loss_type!='score_matching' or model.loss_weighting!='sigma^2'
            or model.frontend_name!='storm' or model.residual_refine_enable
            or bool(getattr(model.sde,'residual_anchor_enabled',False))):
            raise RuntimeError('Unexpected source topology/task')
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('dnn.'))
        model.rrsd_freeze_base=False;model.rrsd_freeze_router=True;model.rrsd_disable_aux_loss=True
        model.frontend_trainable=False;model.to(device);model.dnn.eval();model.frontend.eval()
        dm=model.data_module
        if (int(model.sr)!=16000 or int(dm.n_fft)!=510 or int(dm.hop_length)!=128
            or int(dm.num_frames)!=256 or dm.transform_type!='exponent'
            or dm.spec_abs_exponent!=.5 or dm.spec_factor!=.15):
            raise RuntimeError('Unexpected source crop/spectral geometry')
        if any(p.dtype!=torch.float32 for p in model.dnn.parameters()):raise RuntimeError('Expected float32 dnn')
        def digest():
            h=hashlib.sha256()
            for group,iterator in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
                for name,tensor in iterator:
                    h.update((group+name+str(tuple(tensor.shape))+str(tensor.dtype)).encode())
                    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        result['model_parameter_buffer_pre_digest']=digest()
        def load_audio(path):
            if path not in audio_paths:raise RuntimeError('Unregistered audio access')
            a,sr=sf.read(path,dtype='float32',always_2d=True)
            result['pcm_reads'].append(path)
            if sr!=16000 or a.shape[1]!=1 or not np.isfinite(a).all():raise RuntimeError('Invalid audio geometry/data')
            return torch.from_numpy(a[:,0].copy()).reshape(1,-1)
        data={}
        for split,row in rows.items():
            s,x,n=[load_audio(row[k]) for k in ['clean_path_loader','wav_path','noise1_path']]
            if not s.shape==x.shape==n.shape:raise RuntimeError('No silent length alignment allowed')
            sc,xc,start,factor=crop_pair(s,x,32640,training=False)
            nc=n[...,start:start+32640] if start>=0 else torch.nn.functional.pad(n,(-start,32640-n.shape[-1]+start))
            nc=nc/factor
            error=float((xc-sc-nc).square().mean().sqrt()/xc.square().mean().sqrt().clamp_min(1e-12))
            result.setdefault('crop_additivity',{})[split]=dict(utt=row['utt'],start=start,normfactor=factor,
                relative_rms=error,premise_passed=error<=.001,
                meaning='premise check only, not an identifiability or false-positive result')
            data[split]=(sc,xc)
        phase=['train_native']
        def hook(module,args,output):result['calls'][phase[0]]+=int(args[0].shape[0])
        handle=model.dnn.register_forward_hook(hook)
        def spec(w):return model._sgmse_stft(w.to(device))
        def front(n):
            with torch.no_grad():return model.frontend.forward_transformed_spec(n).detach()
        def score(u,a,t,n):return model(u,a,t,y_cond=n)
        s,x=data['train'];delta,center=edit_component(s,'hann')
        c1,c0,n1,n0=spec(s),spec(s-delta),spec(x),spec(x-delta)
        f1,f0=front(n1),front(n0)
        t=torch.tensor([.5],device=device);a,b=model.sde.weights(t)
        sig=model.sde._std(t).reshape(-1,1,1,1)
        gen=torch.Generator(device=device).manual_seed(seed)
        eps=torch.randn(c1.shape,dtype=c1.dtype,device=device,generator=gen)
        q1=score(a*c1+b*f1+sig*eps,f1,t,n1);q0=score(a*c0+b*f0+sig*eps,f0,t,n0)
        native=(native_dsm(q1,eps,sig)+native_dsm(q0,eps,sig))/2
        phase[0]='train_auxiliary'
        u=(a*c1+b*f1+sig*torch.randn(c1.shape,dtype=c1.dtype,device=device,generator=gen)).detach()
        qa1=score(u,f1,t,n1);qa0=score(u,f1,t,n0)
        aux=(common_point(qa1,u,c1,f1,a,sig)+common_point(qa0,u,c0,f1,a,sig))/2
        pair,stats=pair_loss(qa1,qa0,c1-c0,spectrum_mask(c1,center,128),a,sig)
        total=native+aux+.1*pair
        result['detach_audit']={name:not v.requires_grad for name,v in
            [('clean1',c1),('clean0',c0),('noisy1',n1),('noisy0',n0),('frontend1',f1),('frontend0',f0),('common_state',u)]}
        if not all(result['detach_audit'].values()) or not qa1.requires_grad:
            raise RuntimeError('Detached controls or live Score gradient invariant failed')
        if not bool(torch.isfinite(total)):raise RuntimeError('Nonfinite preflight loss')
        result['losses']=dict(native=float(native.detach()),aux=float(aux.detach()),pair=float(pair.detach()),
            total=float(total.detach()),pair_stats=stats,center=center)
        params=list(model.dnn.parameters())
        pair_grads=torch.autograd.grad(pair,params,retain_graph=True,allow_unused=True)
        result['calls']['backward']+=1
        total.backward();result['calls']['backward']+=1
        grad2=0.;m_grad2=0.;pair_grad2=0.;n_grad=0;bad=[]
        for param,gpair in zip(params,pair_grads):
            gt=param.grad
            if gpair is not None:pair_grad2+=float(gpair.detach().double().square().sum())
            if gt is None:
                if gpair is not None:m_grad2+=float((-.1*gpair.detach()).double().square().sum())
            else:
                gm=gt.detach() if gpair is None else gt.detach()-.1*gpair.detach()
                m_grad2+=float(gm.double().square().sum())
        for name,p in model.named_parameters():
            if p.grad is not None:
                if not name.startswith('dnn.') or not bool(torch.isfinite(p.grad).all()):bad.append(name)
                else:grad2+=float(p.grad.detach().double().square().sum());n_grad+=1
        if bad or not n_grad or not all(np.isfinite(v) for v in [grad2,m_grad2,pair_grad2]) or grad2<=0:
            raise RuntimeError('Only finite nonzero dnn gradients are allowed: '+repr(bad[:5]))
        result['gradients']=dict(l2_norm=grad2**.5,dnn_tensors_with_grad=n_grad,non_dnn_or_nonfinite=bad,
            pair_l2_norm=pair_grad2**.5,M_l2_norm=m_grad2**.5,total_l2_norm=grad2**.5,
            effective_pair_to_M_ratio=None if m_grad2<=0 else .1*(pair_grad2/m_grad2)**.5,
            scope='single crop gradient-scale observation; no weight adjustment or population claim')
        model.zero_grad(set_to_none=True)
        del q1,q0,qa1,qa0,total,native,aux,pair,pair_grads
        torch.cuda.empty_cache();guard();write(BASE/'preflight_result.json',result)
        phase[0]='normal_dev'
        with torch.no_grad():
            ds,dx=data['dev'];noisy=spec(dx);anchor=front(noisy)
            torch.manual_seed(seed+100000);torch.cuda.manual_seed_all(seed+100000)
            sampler=model.get_pc_sampler('reverse_diffusion','none',anchor,N=30,
                score_fn=lambda z,y,t:score(z,y,t,noisy),corrector_steps=1,snr=.5,
                intermediate=False,regeneration_strength=None,anchor_calibrator=None,score_calibrator=None)
            sample,nfe=sampler();wave=model.to_audio(sample.squeeze(),32640)
            if result['calls']['normal_dev']!=30 or int(nfe)!=30:
                raise RuntimeError('Normal dev must actually execute exactly 30 Score forwards')
            if wave.numel()!=32640 or not bool(torch.isfinite(wave).all()):raise RuntimeError('Invalid native output')
            result['normal_dev']=dict(nfe_reported=int(nfe),actual_score_forwards=result['calls']['normal_dev'],
                samples=wave.numel(),finite=True,peak=float(wave.abs().max()))
        handle.remove()
        result['model_parameter_buffer_post_digest']=digest()
        if result['model_parameter_buffer_post_digest']!=result['model_parameter_buffer_pre_digest']:
            raise RuntimeError('Model parameters or buffers changed without authorized update')
        if result['calls']!=dict(train_native=2,train_auxiliary=2,normal_dev=30,backward=2,optimizer_steps=0,ema_updates=0):
            raise RuntimeError('Unexpected call/update counts')
        if result['network_attempts']:raise RuntimeError('Blocked network attempts occurred')
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        result['status']='passed'
    except BaseException as error:
        result['status']='failed';result['failure']=repr(error);result['traceback']=traceback.format_exc()
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        if model is not None and 'digest' in locals() and 'model_parameter_buffer_pre_digest' in result:
            if 'model_parameter_buffer_post_digest' not in result:
                result['model_parameter_buffer_post_digest']=digest()
            if result['model_parameter_buffer_post_digest']!=result['model_parameter_buffer_pre_digest']:
                result['status']='failed'
        result['protected_post_sha256']={p:sha(p) for p in protected}
        result['protected_unchanged']=result['protected_post_sha256']==protected
        if not result['protected_unchanged']:result['status']='failed'
        result['seconds']=time.monotonic()-begin
        if result['seconds']>300:result['status']='failed'
        write(BASE/'preflight_result.json',result)
        (BASE/'preflight_report.md').write_text('# CPEL integration preflight: '+result['status']+'\n\n'
            'One pair / one dev record only. No optimizer, EMA or parameter update; no weight/audio exports. '
            'See preflight_result.json for actual Score/backward counts, gradients, source protection, '
            'additivity limitations and failures. A passed integration check is not evidence of enhancement benefit.\n')
        guard()
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-preflight',action='store_true',required=True)
    parser.parse_args()
    raise SystemExit(main())
