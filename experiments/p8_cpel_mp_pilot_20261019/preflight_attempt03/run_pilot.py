"""CPEL M/P pilot. No model or audio access outside explicit Root-gated execution.

Usage: env/bin/python -B .../run_pilot.py --execute-arm M (or P)
There is deliberately no automatic execution, resume, download or install path.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
PROJECT = Path('/data/coding/sgmse')
CAP = 512*1024**2
RESERVE = 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def disk_guard(extra=0):
    size = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
    if size+extra > CAP or shutil.disk_usage(BASE).free-extra < RESERVE:
        raise RuntimeError('512 MiB total artifact cap or 1 GiB free reserve would be violated')


def validate_fixed_config(cfg):
    """This is a fixed pilot, not a configurable generic training platform."""
    fixed = {
        'checkpoint.sha256':'ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9',
        'device':'cuda:0', 'seed':20261019, 'arms':['M','P'],
        'data.sample_rate':16000, 'data.expected_num_frames':256,
        'data.expected_n_fft':510, 'data.expected_hop_length':128,
        'data.crop_samples':32640, 'data.expected_spec_transform':'exponent',
        'data.expected_spec_abs_exponent':.5, 'data.expected_spec_factor':.15,
        'data.additivity_relative_rms_tolerance':.001,
        'editing.attenuation':.5, 'editing.duration_samples':1280,
        'editing.band_edges_hz':[1500,2500,6500,7500],
        'editing.event_center_edge_margin_samples':4000,
        'editing.response_soft_window_samples':3072,
        'optimization.optimizer':'Adam', 'optimization.learning_rate':1e-5,
        'optimization.betas':[.9,.999], 'optimization.epsilon':1e-8,
        'optimization.weight_decay':0., 'optimization.max_steps_per_arm':500,
        'optimization.original_pairs_per_step':1, 'optimization.gradient_clip_norm':1.,
        'optimization.backend_ema_decay':.999,
        'optimization.native_time_range':[.03,1.], 'optimization.aux_time_range':[.3,.7],
        'optimization.score_sample_equivalents_per_step':4,
        'optimization.aux_clean_weight':1., 'optimization.pair_weight_M':0.,
        'optimization.pair_weight_P':.1, 'optimization.pair_lower_margin':.1,
        'optimization.pair_upper_margin':1.5,
        'optimization.pair_component_weights':dict(lower=1.,upper=1.,orthogonal=.25,outside=.2),
        'optimization.pair_min_raw_target_mean_energy':1e-12,
        'evaluation.records':64, 'evaluation.controlled_time':.5,
        'evaluation.controlled_parents':[0,1], 'evaluation.audio_exports':False,
        'evaluation.ASR_calls':False, 'evaluation.DNSMOS_calls':False,
        'limits.seconds_per_arm_including_evaluation':3600,
        'limits.total_gpu_seconds':7200, 'limits.max_directory_bytes':CAP,
        'limits.minimum_free_disk_bytes':RESERVE,
        'limits.primary_comparison_requires_both_arms_steps':500,
        'limits.network_access':False, 'limits.install_packages':False,
        'limits.original_data_or_checkpoint_writes':False, 'limits.frozen_test_access':False,
    }
    for path, expected in fixed.items():
        actual=cfg
        for key in path.split('.'):actual=actual[key]
        if actual!=expected:
            raise RuntimeError('Fixed implementation/config mismatch: '+path)


def gated_config(arm):
    # Importing this file never reads authorization, data, config or weights.
    import yaml
    cfg_path = BASE/'config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    validate_fixed_config(cfg)
    auth = json.loads((BASE/'root_execution_authorization.json').read_text())
    if arm not in auth['allowed_arms'] or auth.get('status') != 'authorized':
        raise RuntimeError('Root has not authorized this arm')
    if auth['config_sha256'] != sha(cfg_path):
        raise RuntimeError('Config differs from authorized version')
    for name in ['run_pilot.py', 'loss_helpers.py', 'editing.py']:
        if auth['code_sha256'][name] != sha(BASE/name):
            raise RuntimeError('Code differs from authorized version: '+name)
    for split in ['train', 'dev']:
        item = cfg['registries'][split]
        p = Path(item['path'])
        if not p.is_absolute():
            p = BASE/p
        if sha(p) != item['sha256'] or auth['registry_sha256'][split] != item['sha256']:
            raise RuntimeError('Registry lock mismatch')
        item['resolved_path'] = str(p)
    cp = cfg['checkpoint']
    if auth['checkpoint_sha256'] != cp['sha256']:
        raise RuntimeError('Checkpoint authorization mismatch')
    if (BASE/f'STARTED_{arm}.json').exists() or (BASE/f'{arm}_backend_ema.pt').exists():
        raise RuntimeError('Arm already started: no automatic rerun/resume')
    disk_guard()
    # This is the first checkpoint storage access; it is AFTER all Root gates.
    if sha(cp['path']) != cp['sha256']:
        raise RuntimeError('Checkpoint bytes differ')
    return cfg


def execute(arm, cfg):
    started = time.monotonic()
    # Deny actual network connections before importing any ML/model libraries.
    import socket
    network_attempts=[]
    def deny_connect(sock, address, *args, **kwargs):
        network_attempts.append(repr(address))
        raise RuntimeError('Network access forbidden in CPEL pilot')
    socket.socket.connect = deny_connect
    socket.socket.connect_ex = deny_connect
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import random
    import numpy as np
    import soundfile as sf
    import torch
    try:
        from pystoi import stoi
    except ImportError:
        stoi=None
    try:
        from pesq import pesq
    except ImportError:
        pesq=None
    from editing import crop_pair, edit_component, spectrum_mask
    from loss_helpers import native_dsm, common_point, pair_loss
    sys.path.insert(0, str(PROJECT))
    from sgmse.joint_model import JointSemambaSgmseModel
    from sgmse.joint_data_module import JointWaveScpDataModule
    seed = int(cfg['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg['device'])
    if device.type != 'cuda':
        raise RuntimeError('Approved real pilot requires an explicit CUDA device')
    torch.cuda.set_device(device)
    recipe = dict(arm=arm, seed=seed, steps=500, batch_original_pairs=1, lr=1e-5,
                  clip=1., ema=.999, native_t=[.03, 1.], aux_t=[.3, .7],
                  pair_lambda=.1 if arm == 'P' else 0., dtype='float32/complex64',
                  loss_reduction='coefficient mean, then branch mean', score_mode='eval_with_parameter_grad')
    result = dict(status='running', recipe=recipe, checkpoint_sha256=cfg['checkpoint']['sha256'],
                  network_attempts=network_attempts,
                  calls=dict(score_train=0, score_eval_normal=0, score_eval_control=0, frontend=0, backward=0,
                             optimizer=0, sampler_nfe_reported=0), steps=[], evaluations={})
    out = BASE/f'metrics_{arm}.json'
    write(BASE/f'STARTED_{arm}.json', dict(pid=os.getpid(), config_sha256=sha(BASE/'config.yaml'), **recipe))
    def deadline(signum, frame):
        raise TimeoutError('Per-arm 1 GPU-hour wall-clock cap reached')
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(max(1,int(3600-(time.monotonic()-started))))
    model = None
    try:
        # Load with source topology so strict state loading remains meaningful.
        model = JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint']['path'],
            map_location='cpu', strict=True, weights_only=False,
            data_module_cls=JointWaveScpDataModule, frontend_ckpt=None)
        model.eval(no_ema=False)  # Source EMA materialization BEFORE unfreezing dnn.
        if model.train_type != 'mode2' or model.sde.__class__.__name__ != 'OUVESDE':
            raise RuntimeError('Expected full-speech mode2 OUVE source')
        if model.loss_type != 'score_matching' or model.loss_weighting != 'sigma^2':
            raise RuntimeError('Unexpected source score parameterization')
        if model.frontend_name != 'storm' or model.residual_refine_enable:
            raise RuntimeError('Expected native StoRM full-speech frontend')
        if bool(getattr(model.sde, 'residual_anchor_enabled', False)):
            raise RuntimeError('Residual anchor must be disabled')
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith('dnn.'))
        model.rrsd_freeze_base = False
        model.rrsd_freeze_router = True
        model.rrsd_disable_aux_loss = True
        model.frontend_trainable = False
        # Never call model.train(): it would restore source RAW weights/flip F.
        model.to(device); model.dnn.eval(); model.frontend.eval()
        params = list(model.dnn.parameters())
        if not params or any(not p.requires_grad for p in params):
            raise RuntimeError('Only dnn must be trainable')
        if any(p.requires_grad for n,p in model.named_parameters() if not n.startswith('dnn.')):
            raise RuntimeError('Non-dnn trainable parameter')
        if any(p.dtype!=torch.float32 for p in params):
            raise RuntimeError('Backend must be float32, not implicit mixed precision')
        optimizer = torch.optim.Adam(params, lr=1e-5)
        ema = {n:p.detach().clone() for n,p in model.dnn.named_parameters()}
        expected_save = sum(p.numel()*p.element_size() for p in model.dnn.state_dict().values())
        disk_guard(expected_save+1024**2)
        if 2*expected_save+4*1024**2 > CAP:
            raise RuntimeError('Two backend EMA artifacts cannot fit agreed cap')
        length = (int(model.data_module.num_frames)-1)*int(model.data_module.hop_length)
        sr = int(model.sr)
        if sr != 16000 or length != 32640:
            raise RuntimeError('Unexpected source crop geometry; Root review required')
        dm=model.data_module
        if (int(dm.n_fft)!=510 or int(dm.hop_length)!=128 or int(dm.num_frames)!=256
            or dm.transform_type!='exponent' or float(dm.spec_abs_exponent)!=.5
            or float(dm.spec_factor)!=.15):
            raise RuntimeError('Source spectral representation differs from frozen recipe')
        result['crop_geometry'] = dict(sr=sr, samples=length, frames=model.data_module.num_frames,
                                       hop=model.data_module.hop_length)
        phase = ['eval_normal']
        def score_hook(module, args, output):
            result['calls']['score_'+phase[0]] += int(args[0].shape[0])
        handle = model.dnn.register_forward_hook(score_hook)
        train = [json.loads(s) for s in Path(cfg['registries']['train']['resolved_path']).read_text().splitlines()]
        dev = [json.loads(s) for s in Path(cfg['registries']['dev']['resolved_path']).read_text().splitlines()]
        if len(train) != 256 or len(dev) != 64:
            raise RuntimeError('Expected 256 train / 64 dev registry records')
        def read_audio(path):
            a, rate = sf.read(path, dtype='float32', always_2d=True)
            if rate != sr or a.shape[1] != 1 or not np.isfinite(a).all():
                raise RuntimeError('Expected finite mono 16 kHz PCM')
            return torch.from_numpy(a[:,0].copy()).reshape(1,-1)
        def load_pair(row, generator=None, training=False):
            s, x = read_audio(row['clean_path_loader']), read_audio(row['wav_path'])
            return crop_pair(s, x, length, generator, training)
        def spec(w):
            return model._sgmse_stft(w.to(device))
        def front(n):
            result['calls']['frontend'] += int(n.shape[0])
            with torch.no_grad():
                return model.frontend.forward_transformed_spec(n).detach()
        def score(u, a, t, n):
            return model(u, a, t, y_cond=n)
        def geometry(t):
            a,b = model.sde.weights(t)
            sig = model.sde._std(t).reshape(-1,1,1,1)
            return a,b,sig
        def noise_like(z, gen):
            # torch.randn complex64 uses real/imag variance 1/2, as native code.
            return torch.randn(z.shape, device=device, dtype=z.dtype, generator=gen)
        def control(row, kind, index):
            phase[0]='eval_control'
            s,x,start,factor = load_pair(row)
            component_source = s
            status = dict(utt=row['utt'], kind=kind, crop_start=start, normfactor=factor)
            if kind == 'noise_only':
                raw_n = read_audio(row['noise1_path'])
                raw_s = read_audio(row['clean_path_loader']); raw_x = read_audio(row['wav_path'])
                if raw_n.shape != raw_s.shape or raw_s.shape != raw_x.shape:
                    return dict(**status, status='not_evaluable_noise_length_mismatch')
                n = raw_n[...,start:start+length] if start >= 0 else torch.nn.functional.pad(
                    raw_n, (-start, length-raw_n.shape[-1]+start))
                component_source = n/factor
                error = float((x-s-component_source).square().mean().sqrt() /
                              x.square().mean().sqrt().clamp_min(1e-12))
                status['additive_relative_rms'] = error
                if error > 1e-3:
                    return dict(**status, status='not_evaluable_additive_premise_failed')
            if kind == 'noise_only':
                speech_edit,center=edit_component(s,'tukey')
                noise_edit,_=edit_component(component_source,'tukey',center=center)
                speech_energy=speech_edit.square().sum()
                noise_energy=noise_edit.square().sum()
                status.update(speech_edit_energy=float(speech_energy),noise_available_edit_energy=float(noise_energy))
                if float(speech_energy)<=0 or float(noise_energy)<float(speech_energy):
                    return dict(**status,status='not_evaluable_matched_noise_energy_insufficient')
                scale=(speech_energy/noise_energy).sqrt()
                if not bool(torch.isfinite(scale)) or float(scale)>1:
                    raise RuntimeError('Matched noise edit may only be scaled downward')
                delta_wave=noise_edit*scale
                status.update(noise_edit_downscale=float(scale),actual_edit_energy=float(delta_wave.square().sum()))
            else:
                delta_wave,center = edit_component(component_source, 'hann' if kind == 'speech_hann' else 'tukey')
            s0 = s if kind == 'noise_only' else s-delta_wave
            x0 = x-delta_wave
            c1,c0,n1,n0 = spec(s),spec(s0),spec(x),spec(x0)
            f1,f0 = front(n1),front(n0)
            p = spectrum_mask(c1,center,model.data_module.hop_length)
            t = torch.tensor([.5],device=device)
            alpha,beta,sigma = geometry(t)
            gen = torch.Generator(device=device).manual_seed(seed+200000+index*3+['speech_hann','speech_tukey','noise_only'].index(kind))
            eps = noise_like(c1,gen)
            rows=[]
            for parent in (0,1):
                cp,anchor = (c0,f0) if parent==0 else (c1,f1)
                u=alpha*cp+beta*anchor+sigma*eps
                q1,q0=score(u,anchor,t,n1),score(u,anchor,t,n0)
                diff=(sigma.square()/alpha)*(q1-q0)
                if kind == 'noise_only':
                    # Never use zero clean delta as a direction. Report response
                    # projected onto the actual removed input-noise direction.
                    g=p*(n1-n0); denom=g.abs().square().sum().clamp_min(1e-20)
                    rows.append(dict(parent=parent, pair_training_skipped=True,
                        noise_direction_response=float(((p*diff)*g.conj()).real.sum()/denom),
                        clean_output_change_energy=float(diff.abs().square().mean())))
                else:
                    _,stats=pair_loss(q1,q0,c1-c0,p,alpha,sigma)
                    rows.append(dict(parent=parent,**stats,
                        projection_gain=stats['response'],
                        response_energy=float((p*diff).abs().square().mean()),
                        zero_response=bool((p*diff).abs().square().mean()<1e-12),
                        reverse_direction=bool(stats['valid'] and stats['response']<0),
                        overgain=bool(stats['valid'] and stats['response']>1.5),
                        outside_change=float(((1-p)*diff).abs().square().mean())))
            return dict(**status,status='evaluated',center=center,parents=rows)
        def evaluate(label):
            phase[0]='eval_normal'
            rows=[]
            result['evaluations'][label]=rows
            with torch.no_grad():
                for i,row in enumerate(dev):
                    phase[0]='eval_normal'
                    disk_guard()
                    s,x,start,factor=load_pair(row)
                    noisy=spec(x); anchor=front(noisy)
                    torch.manual_seed(seed+100000+i); torch.cuda.manual_seed_all(seed+100000+i)
                    sampler=model.get_pc_sampler('reverse_diffusion','none',anchor,N=30,
                        score_fn=lambda z,y,t:score(z,y,t,noisy),corrector_steps=1,snr=.5,
                        intermediate=False,regeneration_strength=None,
                        anchor_calibrator=None,score_calibrator=None)
                    before_score=result['calls']['score_eval_normal']
                    sample,nfe=sampler()
                    actual_score=result['calls']['score_eval_normal']-before_score
                    if actual_score!=30 or int(nfe)!=30:
                        raise RuntimeError('Native normal inference did not make exactly 30 Score calls')
                    result['calls']['sampler_nfe_reported']+=int(nfe)
                    y=model.to_audio(sample.squeeze(),length).reshape(-1).cpu().numpy()
                    target=s.reshape(-1).numpy()
                    if len(y)!=length or not np.isfinite(y).all():
                        raise RuntimeError('Invalid normal inference waveform')
                    ref=target.astype(np.float64); est=y.astype(np.float64)
                    ref=ref-ref.mean();est=est-est.mean()
                    sisdr=None;si_error=None
                    if np.dot(ref,ref)<=1e-20:
                        si_error='undefined: silent centered reference'
                    else:
                        target_proj=ref*np.dot(est,ref)/np.dot(ref,ref)
                        numerator=np.dot(target_proj,target_proj)
                        denominator=np.dot(est-target_proj,est-target_proj)
                        if numerator<=0 or denominator<=0:
                            si_error='undefined/nonfinite projection-energy ratio'
                        else:sisdr=float(10*np.log10(numerator/denominator))
                    metrics=dict(utt=row['utt'],crop_start=start,nfe=int(nfe),actual_score_calls=actual_score,
                        normalized_wave_mse=float(np.mean((y.astype(np.float64)-target)**2)),
                        si_sdr_db=sisdr,errors={} if si_error is None else {'si_sdr_db':si_error})
                    for key,fn in [('stoi', None if stoi is None else lambda:stoi(target,y,sr,extended=False)),
                                   ('pesq',None if pesq is None else lambda:pesq(sr,target,y,'wb'))]:
                        try:
                            if fn is None:raise RuntimeError('Metric package not installed; no installation allowed')
                            value=float(fn())
                            if not np.isfinite(value):raise RuntimeError('Undefined/nonfinite metric')
                            metrics[key]=value
                        except Exception as error:
                            metrics[key]=None;metrics['errors'][key]=repr(error)
                    metrics['controls']=[control(row,k,i) for k in ['speech_hann','speech_tukey','noise_only']]
                    rows.append(metrics)
                    if (i+1)%8==0:write(out,result)
            result['evaluations'][label]=rows
            write(out,result)
        baseline_path=BASE/'initial_base_metrics.json'
        baseline_identity=dict(config_sha256=sha(BASE/'config.yaml'),checkpoint_sha256=cfg['checkpoint']['sha256'],
                               dev_registry_sha256=cfg['registries']['dev']['sha256'],
                               code_sha256={n:sha(BASE/n) for n in ['run_pilot.py','editing.py','loss_helpers.py']})
        if arm=='M':
            if baseline_path.exists():raise RuntimeError('Baseline already exists; no repeat')
            evaluate('initial_base')
            write(baseline_path,dict(identity=baseline_identity,rows=result['evaluations']['initial_base']))
        else:
            baseline=json.loads(baseline_path.read_text())
            if baseline['identity']!=baseline_identity or len(baseline['rows'])!=64:
                raise RuntimeError('Cannot reuse unmatched baseline')
            result['evaluations']['initial_base']=baseline['rows']
            result['initial_base_reused_sha256']=sha(baseline_path)
        cpu_gen=torch.Generator().manual_seed(seed)
        gpu_gen=torch.Generator(device=device).manual_seed(seed)
        order=[]
        for step in range(500):
            phase[0]='train';disk_guard()
            if not order:order=torch.randperm(len(train),generator=cpu_gen).tolist()
            row=train[order.pop()]
            s,x,start,factor=load_pair(row,cpu_gen,True)
            delta,center=edit_component(s,'hann')
            c1,c0,n1,n0=spec(s),spec(s-delta),spec(x),spec(x-delta)
            f1,f0=front(n1),front(n0)
            p=spectrum_mask(c1,center,model.data_module.hop_length)
            t=.03+.97*torch.rand(1,device=device,generator=gpu_gen)
            alpha,beta,sigma=geometry(t)
            eps=noise_like(c1,gpu_gen)
            q1=score(alpha*c1+beta*f1+sigma*eps,f1,t,n1)
            q0=score(alpha*c0+beta*f0+sigma*eps,f0,t,n0)
            native=(native_dsm(q1,eps,sigma)+native_dsm(q0,eps,sigma))/2
            ta=.3+.4*torch.rand(1,device=device,generator=gpu_gen)
            a,b,sig=geometry(ta)
            parent=int(torch.randint(2,(),generator=cpu_gen))
            cp,anchor=(c0,f0) if parent==0 else (c1,f1)
            u=(a*cp+b*anchor+sig*noise_like(cp,gpu_gen)).detach()
            qa1,qa0=score(u,anchor,ta,n1),score(u,anchor,ta,n0)
            aux=(common_point(qa1,u,c1,anchor,a,sig)+common_point(qa0,u,c0,anchor,a,sig))/2
            pair,stats=pair_loss(qa1,qa0,c1-c0,p,a,sig)
            total=native+aux+(.1 if arm=='P' else 0.)*pair
            if not bool(torch.isfinite(total)):
                raise RuntimeError('Nonfinite loss; no retry')
            optimizer.zero_grad(set_to_none=True);total.backward();result['calls']['backward']+=1
            grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();result['calls']['optimizer']+=1
            with torch.no_grad():
                for name,param in model.dnn.named_parameters():ema[name].mul_(.999).add_(param,alpha=.001)
            result['steps'].append(dict(step=step+1,utt=row['utt'],crop_start=start,
                normfactor=factor,center=center,parent=parent,t=float(t),aux_t=float(ta),
                native=float(native.detach()),aux=float(aux.detach()),pair=float(pair.detach()),
                total=float(total.detach()),grad_preclip=float(grad),pair_stats=stats))
            if (step+1)%10==0:
                result['seconds']=time.monotonic()-started
                write(out,result)
                print(json.dumps(dict(arm=arm,step=step+1,seconds=result['seconds'],loss=float(total.detach()))),flush=True)
        with torch.no_grad():
            for name,param in model.dnn.named_parameters():param.copy_(ema[name])
        del optimizer,ema
        torch.cuda.empty_cache()
        disk_guard(expected_save+1024**2)
        state={n:v.detach().cpu() for n,v in model.dnn.state_dict().items()}
        path=BASE/f'{arm}_backend_ema.pt'
        with path.open('xb') as f:torch.save(dict(schema='cpel-backend-ema-v1',arm=arm,
            source_checkpoint_sha256=cfg['checkpoint']['sha256'],recipe=recipe,dnn=state),f)
        del state
        result['backend_ema']=dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size)
        evaluate('final_'+arm)
        result['status']='completed'
        if network_attempts:
            raise RuntimeError('Blocked network connection attempts occurred; execution is not clean')
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        handle.remove()
    except BaseException as e:
        result['status']='failed';result['failure']=repr(e);result['traceback']=traceback.format_exc()
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous_alarm)
        result['seconds']=time.monotonic()-started
        write(out,result)
        (BASE/f'experiment_report_{arm}.md').write_text('# CPEL '+arm+' pilot: '+result['status']+'\n\n'
            'See metrics_'+arm+'.json for recipe, every train/control/normal-dev row and actual calls. '
            'Fixed pilot, not a phoneme recovery claim. Common-point objective is auxiliary, not '
            'unbiased native DSM. Normal dev uses unchanged OUVE PC N=30 without RRSD. '
            'Noise-only controls failing additivity remain not evaluable, never false-positive counts. '
            'No audio, optimizer, frontend or source checkpoint copies saved.\n')
        disk_guard()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-arm',required=True,choices=['M','P'])
    args=parser.parse_args()
    execute(args.execute_arm,gated_config(args.execute_arm))
