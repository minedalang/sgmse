"""CPEL native-only N/D ablation. No model or audio access outside explicit Root-gated execution.

Usage: env/bin/python -B .../run_pilot.py --execute-arm N (or D)
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
CAP = 320*1024**2
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
        raise RuntimeError('320 MiB total artifact cap or 1 GiB free reserve would be violated')


def enable_backend_training(model, fourier_class=None):
    """Restore learnable dnn weights, preserving intrinsic Fourier frequencies.

    Source frozen-base flags hide the original parameter trainability. Recover
    the known native exception by module type rather than unfreezing every W.
    The optional class argument exists only for a tiny synthetic invariant test.
    """
    if fourier_class is None:
        from sgmse.backbones.ncsnpp_utils.layerspp import GaussianFourierProjection
        fourier_class=GaussianFourierProjection
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('dnn.'))
    excluded=[]
    for name,module in model.dnn.named_modules():
        if isinstance(module,fourier_class):
            direct=dict(module.named_parameters(recurse=False))
            if 'W' not in direct:raise RuntimeError('Native GaussianFourierProjection missing W')
            direct['W'].requires_grad_(False)
            excluded.append('dnn.'+(name+'.' if name else '')+'W')
    excluded=sorted(excluded)
    frozen=sorted(n for n,p in model.named_parameters() if n.startswith('dnn.') and not p.requires_grad)
    if frozen!=excluded:raise RuntimeError('Unexpected frozen dnn parameter outside intrinsic exception')
    if any(p.requires_grad for n,p in model.named_parameters() if not n.startswith('dnn.')):
        raise RuntimeError('Non-dnn trainable parameter')
    if not any(p.requires_grad for p in model.dnn.parameters()):raise RuntimeError('No learnable backend')
    return excluded


def validate_native_config(cfg, historical):
    """Fixed missing-aux contrast, not a general tuning interface."""
    if cfg['arms']!=['N','D'] or cfg['protocol_version']!='native_DSM_without_common_point_pair_ablation_v1':
        raise RuntimeError('Expected isolated N/D protocol')
    for key in ['seed','device','checkpoint','data','editing','native_path']:
        if cfg[key]!=historical[key]:raise RuntimeError('Historical scientific field changed: '+key)
    for split in ['train','dev']:
        for key in ['sha256','records']:
            if cfg['registries'][split][key]!=historical['registries'][split][key]:
                raise RuntimeError('Historical registry identity changed')
    skip={'aux_clean_weight','aux_clean_loss','pair_weight_M','pair_weight_P','pair_weight_N','pair_weight_D',
          'M_control','N_control','selection','invalid_pair','update_rule','scalar_logging'}
    for key,value in historical['optimization'].items():
        if key not in skip and cfg['optimization'].get(key)!=value:
            raise RuntimeError('Historical optimization changed: '+key)
    opt=cfg['optimization']
    if opt['aux_clean_weight']!=0 or opt['pair_weight_N']!=0 or opt['pair_weight_D'] is not None:
        raise RuntimeError('Common-point aux must be absent; only D applies dynamic pair')
    rule=opt['update_rule']
    if (rule['name']!='stop_gradient_raw_gradient_budget' or rule['target_norm_fraction']!=.01
        or rule['max_coefficient']!=100. or rule['min_coefficient']!=0.):
        raise RuntimeError('Gradient budget changed')
    for key,value in historical['evaluation'].items():
        if key!='baseline' and cfg['evaluation'].get(key)!=value:
            raise RuntimeError('Historical evaluation changed: '+key)
    expected=dict(historical['limits']);expected.update(seconds_per_arm_including_evaluation=900,
        total_gpu_seconds=1800,max_directory_bytes=320*1024**2)
    if cfg['limits']!=expected:raise RuntimeError('Fixed N/D resource or safety limits changed')


def protected_ast(text):
    """Compare complete local evaluation closures and original initialization."""
    import ast
    tree=ast.parse(text);execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    trial=next(n for n in execute.body if isinstance(n,ast.Try) and any(
        isinstance(c,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model' for t in c.targets) for c in n.body))
    names={'load_pair','physical_crop','read_audio','spec','front','score','geometry','noise_like','control','evaluate','frozen_state_digest'}
    nodes={n.name:ast.dump(n,include_attributes=False) for n in ast.walk(execute)
           if isinstance(n,ast.FunctionDef) and n.name in names}
    if set(nodes)!=names:raise RuntimeError('Missing protected evaluation closure')
    def assigns(n,name):
        return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
    first=next(i for i,n in enumerate(trial.body) if assigns(n,'model'))
    last=next(i for i,n in enumerate(trial.body) if assigns(n,'length'))
    nodes['initialization']=ast.dump(ast.Module(body=trial.body[first:last],type_ignores=[]),include_attributes=False)
    names2={'enable_backend_training'}
    nodes.update({n.name:ast.dump(n,include_attributes=False) for n in tree.body
                  if isinstance(n,ast.FunctionDef) and n.name in names2})
    return nodes


def validate_baseline_provenance(cfg, auth=None):
    """Metadata/source-only; never reads audio or checkpoint storage."""
    pp=BASE/'baseline_provenance.json';provenance=json.loads(pp.read_text())
    if auth is not None and auth['baseline_provenance_sha256']!=sha(pp):
        raise RuntimeError('Baseline provenance lock mismatch')
    historical=Path(provenance['historical_directory'])
    expected_names={'run_pilot.py','editing.py','loss_helpers.py','gradient_budget.py','config.yaml'}
    if set(provenance['historical_files'])!=expected_names:raise RuntimeError('Incomplete historical source manifest')
    for name,digest in provenance['historical_files'].items():
        if sha(historical/name)!=digest:raise RuntimeError('Historical source differs: '+name)
    old_cfg=json.loads((historical/'config.yaml').read_text())
    validate_native_config(cfg,old_cfg)
    if protected_ast((historical/'run_pilot.py').read_text())!=protected_ast(Path(__file__).read_text()):
        raise RuntimeError('Initialization or complete evaluation source differs')
    for name in ['editing.py','loss_helpers.py','gradient_budget.py']:
        if sha(BASE/name)!=provenance['historical_files'][name]:raise RuntimeError('Helper bytes changed')
    core=provenance['core_sources']
    expected_core={str(PROJECT/'sgmse/joint_model.py'),str(PROJECT/'sgmse/model.py'),str(PROJECT/'sgmse/joint_data_module.py')}
    if set(core)!=expected_core:raise RuntimeError('Expected three fixed core sources')
    if auth is not None and auth['core_sha256']!=core:raise RuntimeError('Core authorization differs')
    for path,digest in core.items():
        if sha(path)!=digest:raise RuntimeError('Core source differs')
    for label in ['historical_protection_pre','historical_receipt']:
        entry=provenance[label]
        if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Historical evidence differs: '+label)
    pre=json.loads(Path(provenance['historical_protection_pre']['path']).read_text())['sha256']
    for path,digest in core.items():
        if pre.get(path)!=digest:raise RuntimeError('Core lacks matching historical protection')
    if pre.get(cfg['checkpoint']['path'])!=cfg['checkpoint']['sha256']:
        raise RuntimeError('Checkpoint lacks historical identity')
    for path,digest in provenance.get('current_source_dependencies',{}).items():
        if sha(path)!=digest:raise RuntimeError('Current dependency differs')
    # Dependencies above are current-only locks, not historical equivalence proof.
    entry=provenance['baseline'];bp=Path(entry['path'])
    if bp.resolve()!=(historical/'initial_base_metrics.json').resolve() or sha(bp)!=entry['sha256']:
        raise RuntimeError('Historical baseline path/hash mismatch')
    baseline=json.loads(bp.read_text());identity=baseline['identity']
    expected=dict(config_sha256=provenance['historical_files']['config.yaml'],
        checkpoint_sha256=cfg['checkpoint']['sha256'],dev_registry_sha256=cfg['registries']['dev']['sha256'],
        code_sha256={n:provenance['historical_files'][n] for n in expected_names if n!='config.yaml'})
    if identity!=entry['identity'] or identity!=expected:raise RuntimeError('Historical identity mismatch')
    if provenance['seed']!=cfg['seed'] or provenance['dev_registry_sha256']!=cfg['registries']['dev']['sha256']:
        raise RuntimeError('Historical seed/registry mismatch')
    if len(baseline['rows'])!=64 or not all(r.get('controls_complete') and len(r.get('controls',[]))==4
            and r['actual_score_calls']==30 and r['nfe']==30 for r in baseline['rows']):
        raise RuntimeError('Historical baseline incomplete')
    return baseline,provenance


def gated_config(arm,config_name='config.yaml'):
    # Importing this file never reads authorization, data, config or weights.
    cfg_path = (BASE/config_name).resolve()
    if cfg_path.parent!=BASE:raise RuntimeError('Config must be in this experiment directory')
    cfg = json.loads(cfg_path.read_text())
    auth = json.loads((BASE/'root_execution_authorization.json').read_text())
    if auth['allowed_arms']!=['N','D'] or arm not in auth['allowed_arms'] or auth.get('status') != 'authorized':
        raise RuntimeError('Root has not authorized this arm')
    if auth['config_sha256'] != sha(cfg_path):
        raise RuntimeError('Config differs from authorized version')
    for name in ['run_pilot.py', 'loss_helpers.py', 'editing.py','gradient_budget.py']:
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
    validate_baseline_provenance(cfg,auth)
    if arm=='D':
        n=json.loads((BASE/'metrics_N.json').read_text())
        ns=json.loads((BASE/'STARTED_N.json').read_text())
        if ns['config_sha256']!=sha(cfg_path) or n['checkpoint_sha256']!=cfg['checkpoint']['sha256']:
            raise RuntimeError('N source/config identity differs')
        if sha(BASE/n['backend_ema']['path'])!=n['backend_ema']['sha256']:
            raise RuntimeError('N terminal EMA artifact differs')
        if (n['status']!='completed' or n['calls']['optimizer']!=500 or len(n['steps'])!=500
            or not n['frozen_state_unchanged'] or len(n['evaluations']['final_N'])!=64
            or not all(r['controls_complete'] and len(r['controls'])==4 for r in n['evaluations']['final_N'])):
            raise RuntimeError('N must complete full training and evaluation before D')
    cp = cfg['checkpoint']
    if auth['checkpoint_sha256'] != cp['sha256']:
        raise RuntimeError('Checkpoint authorization mismatch')
    if (BASE/f'STARTED_{arm}.json').exists() or (BASE/f'{arm}_backend_ema.pt').exists():
        raise RuntimeError('Arm already started: no automatic rerun/resume')
    disk_guard()
    # This is the first checkpoint storage access; it is AFTER all Root gates.
    if sha(cp['path']) != cp['sha256']:
        raise RuntimeError('Checkpoint bytes differ')
    cfg['_config_path']=str(cfg_path)
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
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',PATH=cfg['native_path'])
    if str(PROJECT/'env/bin') not in os.environ['PATH'].split(os.pathsep):
        raise RuntimeError('Authorized PATH must include project env/bin')
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
    from editing import crop_pair, edit_component, spectrum_mask, pcm16_nearest_no_clip, codec_additivity
    from loss_helpers import native_dsm, pair_loss
    from gradient_budget import compose_gradients
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
                  pair_rule='dynamic raw-gradient budget relative to native-only gradient; target .01 cap100; N eta_applied0',
                  common_clean_aux=False, common_state_role='pair only; no pointwise clean regression',
                  reverse_traversals_per_step=2, dtype='float32/complex64',
                  loss_reduction='coefficient mean, then branch mean', score_mode='eval_with_parameter_grad')
    result = dict(status='running', recipe=recipe, checkpoint_sha256=cfg['checkpoint']['sha256'],
                  network_attempts=network_attempts,
                  calls=dict(score_train=0, score_eval_normal=0, score_eval_control=0, frontend=0, backward=0,
                             optimizer=0, sampler_nfe_reported=0), steps=[], evaluations={})
    out = BASE/f'metrics_{arm}.json'
    write(BASE/f'STARTED_{arm}.json', dict(pid=os.getpid(), config_sha256=sha(cfg['_config_path']),
        config_path=cfg['_config_path'], **recipe))
    def deadline(signum, frame):
        raise TimeoutError('Per-arm 900-second wall-clock cap reached')
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(max(1,int(900-(time.monotonic()-started))))
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
        result['intrinsic_fixed_parameters']=enable_backend_training(model)
        model.rrsd_freeze_base = False
        model.rrsd_freeze_router = True
        model.rrsd_disable_aux_loss = True
        model.frontend_trainable = False
        # Never call model.train(): it would restore source RAW weights/flip F.
        model.to(device); model.dnn.eval(); model.frontend.eval()
        def frozen_state_digest():
            frontend=hashlib.sha256();intrinsic=hashlib.sha256()
            def add(h,name,tensor):
                h.update((name+str(tuple(tensor.shape))+str(tensor.dtype)).encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            state=model.frontend.state_dict()
            for name in sorted(state):add(frontend,name,state[name])
            named=dict(model.named_parameters())
            for name in result['intrinsic_fixed_parameters']:add(intrinsic,name,named[name])
            return dict(frontend_state_dict_sha256=frontend.hexdigest(),frontend_state_entries=len(state),
                intrinsic_sha256=intrinsic.hexdigest(),intrinsic_names=result['intrinsic_fixed_parameters'])
        result['frozen_state_pre']=frozen_state_digest()
        params = [p for p in model.dnn.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError('Learnable dnn parameter set must be nonempty')
        if any(p.requires_grad for n,p in model.named_parameters() if not n.startswith('dnn.')):
            raise RuntimeError('Non-dnn trainable parameter')
        if any(p.dtype!=torch.float32 for p in params):
            raise RuntimeError('Backend must be float32, not implicit mixed precision')
        optimizer = torch.optim.Adam(params, lr=1e-5)
        ema = {n:p.detach().clone() for n,p in model.dnn.named_parameters() if p.requires_grad}
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
        def read_audio(path,dtype='float32'):
            a, rate = sf.read(path, dtype=dtype, always_2d=True)
            if rate != sr or a.shape[1] != 1 or not np.isfinite(a).all():
                raise RuntimeError('Expected finite mono 16 kHz PCM')
            return torch.from_numpy(a[:,0].copy()).reshape(1,-1)
        def load_pair(row, generator=None, training=False):
            s, x = read_audio(row['clean_path_loader']), read_audio(row['wav_path'])
            return crop_pair(s, x, length, generator, training)
        def physical_crop(w,start):
            if start>=0:return w[...,start:start+length]
            return torch.nn.functional.pad(w,(-start,length-w.shape[-1]+start))
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
                raw_n = read_audio(row['noise1_path'],'float64')
                raw_s = read_audio(row['clean_path_loader'],'float64'); raw_x = read_audio(row['wav_path'],'float64')
                if raw_n.shape != raw_s.shape or raw_s.shape != raw_x.shape:
                    return dict(**status, status='not_evaluable_noise_length_mismatch')
                n = raw_n[...,start:start+length] if start >= 0 else torch.nn.functional.pad(
                    raw_n, (-start, length-raw_n.shape[-1]+start))
                component_source = (n/factor).float()
                error = float((x-s-component_source).square().mean().sqrt() /
                              x.square().mean().sqrt().clamp_min(1e-12))
                status.update(additive_relative_rms=error,strict_pass=error<=1e-3,
                    processing_chain_verified=False)
                codec=codec_additivity(physical_crop(raw_s,start),physical_crop(raw_x,start),n,
                    [sf.info(row[k]).subtype for k in ['clean_path_loader','wav_path','noise1_path']])
                status['codec_diagnostics']=codec
                if codec['status']!='codec_compatible':
                    return dict(**status,status='not_evaluable_codec_unknown_or_incompatible')
                status['noise_response_scope']='codec-compatible stored noise-instance edit only; not noise purity/true-negative/FPR'
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
            status['event_center']=center
            s0 = s if kind == 'noise_only' else s-delta_wave
            x0 = x-delta_wave
            if kind=='speech_tukey_pcm16':
                # Quantize all four physical-amplitude endpoints, not only the
                # edited branches. Never reconstruct raw parents from a rounded
                # normalized waveform when their exact PCM source is available.
                raw_s=physical_crop(read_audio(row['clean_path_loader'],'float64'),start)
                raw_x=physical_crop(read_audio(row['wav_path'],'float64'),start)
                change=delta_wave.double()*factor
                endpoints=[raw_s,raw_s-change,raw_x,raw_x-change]
                quantized=[];details=[]
                for endpoint in endpoints:
                    value,info=pcm16_nearest_no_clip(endpoint);quantized.append(value);details.append(info)
                status['precision_matching']=dict(endpoints=details,normalization_reference='original X1 peak',
                    processing_chain_verified=False,strict_shared_noise_after_quantization=False)
                if not all(info['valid'] for info in details):
                    return dict(**status,status='not_evaluable_quantizer_range_or_finite')
                qs1,qs0,qx1,qx0=quantized
                residual1=qx1-qs1;residual0=qx0-qs0
                target_change=(qs1-qs0)-change
                target_energy=float(change.square().sum())
                status['precision_matching'].update(
                    physical_target_change_mse=float(target_change.square().mean()),
                    physical_target_relative_l2_change=None if target_energy==0 else
                        (float(target_change.square().sum())/target_energy)**.5,
                    residual1_vs_prequant_mse=float((residual1-(raw_x-raw_s)).square().mean()),
                    residual0_vs_prequant_mse=float((residual0-(raw_x-raw_s)).square().mean()),
                    between_branch_residual_change_mse=float((residual1-residual0).square().mean()))
                original_target=spec(s)-spec(s0)
                s,s0,x,x0=[(v/factor).float() for v in quantized]
            c1,c0,n1,n0 = spec(s),spec(s0),spec(x),spec(x0)
            if kind=='speech_tukey_pcm16':
                change_energy=float(((c1-c0)-original_target).abs().square().sum())
                original_energy=float(original_target.abs().square().sum())
                status['precision_matching'].update(compressed_target_change_mse=change_energy/original_target.numel(),
                    compressed_target_relative_l2_change=None if original_energy==0 else (change_energy/original_energy)**.5)
            f1,f0 = front(n1),front(n0)
            p = spectrum_mask(c1,center,model.data_module.hop_length)
            t = torch.tensor([.5],device=device)
            alpha,beta,sigma = geometry(t)
            seed_kind='speech_tukey' if kind=='speech_tukey_pcm16' else kind
            control_seed=seed+200000+index*3+['speech_hann','speech_tukey','noise_only'].index(seed_kind)
            status['gaussian_seed']=control_seed
            gen = torch.Generator(device=device).manual_seed(control_seed)
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
                    g=p*(n1-n0); denom=g.abs().square().sum()
                    input_energy=float(denom)
                    direction_defined=input_energy>1e-20
                    rows.append(dict(parent=parent, pair_training_skipped=True,
                        noise_direction_response=float(((p*diff)*g.conj()).real.sum()/denom) if direction_defined else None,
                        noise_direction_input_energy=input_energy,
                        noise_direction_defined=direction_defined,
                        noise_direction_status='defined' if direction_defined else 'undefined_input_energy_at_or_below_1e-20',
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
                    metrics['controls']=[]
                    metrics['controls_complete']=False
                    rows.append(metrics)
                    for kind in ['speech_hann','speech_tukey','noise_only','speech_tukey_pcm16']:
                        metrics['controls'].append(control(row,kind,i))
                    metrics['controls_complete']=True
                    if (i+1)%8==0:write(out,result)
            result['evaluations'][label]=rows
            write(out,result)
        baseline,provenance=validate_baseline_provenance(cfg)
        result['evaluations']['initial_base']=baseline['rows']
        result['initial_base_reused_sha256']=provenance['baseline']['sha256']
        result['baseline_provenance_sha256']=sha(BASE/'baseline_provenance.json')
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
            pair,stats=pair_loss(qa1,qa0,c1-c0,p,a,sig)
            m=native
            if not all(bool(torch.isfinite(value)) for value in [native,pair,m]):
                raise RuntimeError('Nonfinite loss; no retry')
            optimizer.zero_grad(set_to_none=True)
            gm=torch.autograd.grad(m,params,retain_graph=True,allow_unused=True)
            result['calls']['backward']+=1
            gp=torch.autograd.grad(pair,params,allow_unused=True)
            result['calls']['backward']+=1
            combined,budget=compose_gradients(gm,gp,stats['valid'],arm=='D')
            for param,value in zip(params,combined):param.grad=value
            surrogate=float(m.detach())+budget['eta_applied']*float(pair.detach())
            grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();result['calls']['optimizer']+=1
            with torch.no_grad():
                for name,param in model.dnn.named_parameters():
                    if param.requires_grad:ema[name].mul_(.999).add_(param,alpha=.001)
            result['steps'].append(dict(step=step+1,utt=row['utt'],crop_start=start,
                normfactor=factor,center=center,parent=parent,t=float(t),aux_t=float(ta),
                native=float(native.detach()),common_clean_aux_computed=False,pair=float(pair.detach()),
                surrogate_value=surrogate,grad_preclip=float(grad),pair_stats=stats,gradient_budget=budget))
            del gm,gp,combined,q1,q0,qa1,qa0,m,native,pair
            if (step+1)%10==0:
                result['seconds']=time.monotonic()-started
                write(out,result)
                print(json.dumps(dict(arm=arm,step=step+1,seconds=result['seconds'],surrogate_value=surrogate)),flush=True)
        with torch.no_grad():
            for name,param in model.dnn.named_parameters():
                if name in ema:param.copy_(ema[name])
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
        result['frozen_state_post']=frozen_state_digest()
        result['frozen_state_unchanged']=result['frozen_state_pre']==result['frozen_state_post']
        if not result['frozen_state_unchanged']:
            raise RuntimeError('Frozen frontend state or intrinsic Fourier parameters changed')
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
        if model is not None and 'frozen_state_digest' in locals() and 'frozen_state_pre' in result:
            try:
                if 'frozen_state_post' not in result:result['frozen_state_post']=frozen_state_digest()
                result['frozen_state_unchanged']=result['frozen_state_pre']==result['frozen_state_post']
                if not result['frozen_state_unchanged']:
                    result['status']='failed'
                    result['frozen_state_verification_failure']='Frozen frontend or intrinsic state changed'
            except BaseException as error:
                result['status']='failed';result['frozen_state_post_error']=repr(error)
        result['seconds']=time.monotonic()-started
        write(out,result)
        (BASE/f'experiment_report_{arm}.md').write_text('# CPEL '+arm+' pilot: '+result['status']+'\n\n'
            'See metrics_'+arm+'.json for recipe, every train/control/normal-dev row and actual calls. '
            'Fixed N/D missing-aux ablation, not a phoneme recovery claim. Common state serves pair only; '
            'gM is native DSM only; logged surrogate is not a fixed global loss derivative. Normal dev uses unchanged OUVE PC N=30 without RRSD. '
            'Noise-only controls failing additivity remain not evaluable, never false-positive counts. '
            'No audio, optimizer, frontend or source checkpoint copies saved.\n')
        disk_guard()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-arm',required=True,choices=['N','D'])
    parser.add_argument('--config',default='config.yaml')
    args=parser.parse_args()
    execute(args.execute_arm,gated_config(args.execute_arm,args.config))
