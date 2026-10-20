#!/usr/bin/env python3
"""One authorized, bounded comparison of raw and public waveform anchors."""
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
CAP = 2 * 1024**2
RESERVE = 1024**3


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def disk_guard():
    require(sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()) < CAP,
            'Artifact budget exhausted')
    require(shutil.disk_usage(BASE).free >= RESERVE, 'Disk reserve below 1 GiB')


def main():
    # All protected model/audio reads occur strictly after this authorization gate.
    cfg_path = BASE / 'config.yaml'
    cfg = read_json(cfg_path)  # JSON is valid YAML; intentionally no YAML import.
    # Normalize the frozen Root schema without modifying its on-disk bytes.
    cfg['checkpoint'] = dict(path=cfg['source']['checkpoint_path'], sha256=cfg['source']['checkpoint_sha256'])
    cfg['manifest'] = dict(path=str(BASE / cfg['manifest_file']), sha256=cfg['manifest_sha256'])
    cfg['source_registry'] = dict(path=cfg['selection']['source_dev_registry_path'],
        sha256=cfg['selection']['source_dev_registry_sha256'])
    cfg['old_baseline'] = cfg['baseline']
    cfg['core_sha256'] = cfg['source']['code_sha256']
    cfg['tolerances'] = dict(interface_relative_l2=cfg['numerical']['roundtrip_relative_l2_tolerance'],
        old_raw_mse_atol=cfg['baseline']['sanity_absolute_tolerance'],
        old_raw_mse_rtol=cfg['baseline']['sanity_relative_tolerance'])
    global CAP, RESERVE
    CAP = int(cfg['limits']['artifact_bytes'])
    RESERVE = int(cfg['limits']['free_disk_reserve_bytes'])
    require(CAP == 2097152 and RESERVE == 1073741824 and cfg['limits']['seconds'] == 180,
            'Frozen resource limits differ')
    auth = read_json(BASE / cfg['authorization_file'])
    expected = dict(config_sha256=sha(cfg_path), code_sha256=sha(__file__),
                    manifest_sha256=cfg['manifest']['sha256'],
                    source_checkpoint_sha256=cfg['checkpoint']['sha256'],
                    core_sha256=cfg['core_sha256'],
                    source_registry_sha256=cfg['source_registry']['sha256'],
                    old_baseline_sha256=cfg['old_baseline']['sha256'])
    require(auth.get('status') == 'authorized_once', 'Missing one-run Root authorization')
    for key, value in expected.items():
        require(auth.get(key) == value, 'Authorization mismatch: ' + key)
    require(not (BASE / 'STARTED.json').exists(), 'One-shot audit already started; no retry')
    disk_guard()
    started = time.monotonic()
    (BASE / 'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(), **expected), indent=2))
    result = dict(status='running', identity=expected, cases=[],
                  counts=dict(model_loads=0, pcm_reads=0, frontend=0, backend=0, score=0),
                  source_pre={}, source_post={}, failure=None, network_refusals=[])
    protected = {}
    model = None
    handle = None

    def save():
        result['elapsed_seconds'] = time.monotonic() - started
        payload = json.dumps(result, indent=2, allow_nan=False)
        require(len(payload.encode()) < CAP // 2, 'Scalar JSON exceeds reserved budget')
        try:
            disk_guard()
            (BASE / 'metrics.json').write_text(payload)
            disk_guard()
        except BaseException as exc:
            result['status'] = 'failed'
            result['resource_failure'] = dict(message=str(exc),
                actual_artifact_bytes=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()),
                free_bytes=shutil.disk_usage(BASE).free)
            # Preserve failure and partial counters even when the guard itself fails.
            (BASE / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            raise

    def timeout(signum, frame):
        raise TimeoutError('180-second audit budget reached')

    old_alarm = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    try:
        # Explicit network rejection without changing filesystem permissions.
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8', PATH=cfg['native_path'])
        require(str(PROJECT / 'env/bin') in os.environ['PATH'].split(os.pathsep), 'Ninja env/bin missing')
        import socket
        def no_network(*args, **kwargs):
            result['network_refusals'].append(dict(operation='network_connect_or_resolve', arguments=repr(args)[:300]))
            raise RuntimeError('Network access forbidden in this audit')
        socket.socket.connect = no_network
        socket.socket.connect_ex = no_network
        socket.create_connection = no_network
        socket.getaddrinfo = no_network
        for item in ('manifest', 'source_registry', 'old_baseline'):
            require(sha(cfg[item]['path']) == cfg[item]['sha256'], item + ' hash mismatch')
        for path, digest in cfg['core_sha256'].items():
            require(sha(path) == digest, 'Core source hash mismatch: ' + path)
        manifest = read_json(cfg['manifest']['path'])
        records = manifest['records']
        require(len(records) == 8, 'Exactly eight frozen cases required')
        registry = [json.loads(s) for s in Path(cfg['source_registry']['path']).read_text().splitlines()]
        require(len(registry) == 64, 'Source dev registry must contain 64 records')
        for row in records:
            source = registry[row['dev_index']]
            require(all(row[k] == source[k] for k in ('utt', 'wav_path', 'clean_path_loader')),
                    'Manifest/source identity mismatch')
            require(row['expected_seed'] == 20261019 + 100000 + row['dev_index'], 'Seed mismatch')
            for key in ('wav_path', 'clean_path_loader'):
                protected[row[key]] = sha(row[key])
        require(len(protected) == 16, 'Expected sixteen distinct S/X source files')
        protected[cfg['checkpoint']['path']] = sha(cfg['checkpoint']['path'])
        require(protected[cfg['checkpoint']['path']] == cfg['checkpoint']['sha256'], 'Checkpoint hash mismatch')
        for path in [str(cfg_path), __file__, cfg['manifest']['path'],
                     cfg['source_registry']['path'], cfg['old_baseline']['path'], *cfg['core_sha256']]:
            protected[path] = sha(path)
        result['source_pre'] = protected.copy()
        save()
        sys.path.insert(0, str(PROJECT))
        import numpy as np
        import soundfile as sf
        import torch
        from sgmse.joint_model import JointSemambaSgmseModel
        from sgmse.joint_data_module import JointWaveScpDataModule
        from sgmse.model import ScoreModel
        from sgmse.util.other import pad_spec
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg['device'])
        require(device.type == 'cuda', 'Explicit CUDA device required')
        torch.cuda.set_device(device)
        result['counts']['model_loads'] += 1
        require(result['counts']['model_loads'] <= cfg['limits']['source_model_loads'], 'Model-load cap exceeded')
        model = JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint']['path'],
            map_location='cpu', strict=True, weights_only=False,
            data_module_cls=JointWaveScpDataModule, frontend_ckpt=None)
        model.eval(no_ema=False)
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        require(not model.training and not model.frontend.training and not model.dnn.training,
                'All modules must remain eval')
        require(model.train_type == 'mode2' and model.frontend_name == 'storm'
                and not model.residual_refine_enable and model.sde.__class__.__name__ == 'OUVESDE'
                and model.sde.sampler_type == 'pc', 'Unexpected source topology')
        require(not getattr(model.sde, 'residual_anchor_enabled', False), 'Residual SDE forbidden')
        require(model.loss_type == 'score_matching' and model.loss_weighting == 'sigma^2',
                'Unexpected source parameterization')
        f, dm = model.frontend, model.data_module
        geometry = dict(n_fft=510, hop_length=128, transform_type='exponent',
                        spec_abs_exponent=.5, spec_factor=.15)
        for k, v in geometry.items():
            require(getattr(f, k) == getattr(dm, k) == v, 'Frontend/model geometry mismatch: ' + k)
        require(dm.num_frames == 256 and model.sr == 16000, 'Unexpected frame/sample geometry')
        require(torch.equal(f.window.to(device), dm.window.to(device)), 'Window contents differ')

        def desc(x):
            x = x.detach().contiguous().cpu().numpy()
            return dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(x.tobytes()).hexdigest())

        def diff(a, b):
            require(a.shape == b.shape, 'Comparison shape mismatch')
            require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'Nonfinite tensor')
            a, b = a.to(torch.complex128) if a.is_complex() else a.double(), b.to(torch.complex128) if b.is_complex() else b.double()
            mse = float((a-b).abs().square().mean())
            energy = max(float(a.abs().square().mean()), float(cfg['numerical']['denominator_floor']))
            return dict(mse=mse, reference_energy=energy,
                        normalized_mse=mse/energy if energy else None,
                        relative_l2=math.sqrt(mse/energy) if energy else None)

        result['geometry'] = dict(**geometry, samples=32640, frames=256,
                                  center=True, window=desc(f.window), exact_inverse=False)
        old_list = read_json(cfg['old_baseline']['path'])['rows']
        require(len(old_list) == 64, 'Historical baseline must have 64 rows')
        old_rows = {r['utt']: r for r in old_list}
        for row in records:
            require(old_list[row['dev_index']]['utt'] == row['utt'], 'Historical dev index mismatch')
        active = {}
        def score_observer(module, args, output):
            result['counts']['score'] += int(args[0].shape[0])
            require(result['counts']['score'] <= cfg['limits']['score_calls'], 'Score-call cap exceeded')
        handle = model.dnn.register_forward_hook(score_observer)
        original_sampler = model.get_pc_sampler
        def observe_sampler(predictor, corrector, y, **kwargs):
            require(active.get('public') and 'anchor' not in active, 'Unexpected public sampler entry')
            active['anchor'] = y.detach().clone()
            fn = kwargs['score_fn']
            require(getattr(fn, 'func', None) is model, 'Unexpected public Score function')
            active['noisy'] = fn.keywords['y_cond'].detach().clone()
            require(kwargs['N'] == 30 and corrector == 'none' and predictor == 'reverse_diffusion',
                    'Public sampler setting mismatch')
            for k in ('regeneration_strength', 'anchor_calibrator', 'score_calibrator'):
                require(kwargs[k] is None, 'Unexpected calibration/regeneration')
            return original_sampler(predictor, corrector, y, **kwargs)
        model.get_pc_sampler = observe_sampler
        tolerances = cfg['tolerances']
        with torch.no_grad():
            for row in records:
                disk_guard()
                out = dict(utt=row['utt'], dev_index=row['dev_index'], bin=row['priority_bin'],
                           seed=row['expected_seed'])
                result['cases'].append(out)
                waves = []
                for key in ('clean_path_loader', 'wav_path'):
                    result['counts']['pcm_reads'] += 1
                    require(result['counts']['pcm_reads'] <= cfg['limits']['source_pcm_reads'], 'PCM-read cap exceeded')
                    wave, sr = sf.read(row[key], dtype='float32', always_2d=True)
                    require(sr == 16000 and wave.shape[1] == 1 and np.isfinite(wave).all(), 'Invalid input')
                    waves.append(torch.from_numpy(wave[:, 0].copy()).reshape(1, -1))
                s, n = waves
                require(s.shape == n.shape, 'Source S/X lengths disagree')
                L = 32640
                start = (s.shape[-1]-L)//2
                if start >= 0:
                    s, n = s[:, start:start+L], n[:, start:start+L]
                else:
                    pad = L-s.shape[-1]
                    s, n = (torch.nn.functional.pad(w, (pad//2, pad-pad//2)) for w in (s, n))
                    start = -(pad//2)
                r = n.abs().max().clamp_min(1e-12)
                s_norm, n_norm = (s/r).to(device), (n/r).to(device)
                N = model._sgmse_stft(n_norm)
                result['counts']['frontend'] += 1
                require(result['counts']['frontend'] <= cfg['limits']['frontend_calls'], 'Frontend-call cap exceeded')
                A = f.forward_transformed_spec(N)
                require(A.dtype == torch.complex64 and tuple(A.shape) == (1,1,256,256), 'Raw anchor geometry')
                require(bool(torch.isfinite(A).all() and torch.isfinite(N).all()), 'Nonfinite frontend spectrum')
                a = f.spec_to_waveform(A, length=L)
                require(bool(torch.isfinite(a).all()), 'Nonfinite frontend waveform')
                require(tuple(a.shape) == (1,1,L), 'Frontend waveform geometry')
                expected_anchor = pad_spec(model._sgmse_stft(a))
                physical_front = (a * float(r)).squeeze(1)
                physical_noisy = n.to(device)
                require(tuple(physical_front.shape) == tuple(physical_noisy.shape) == (1,L),
                        'Public waveform inputs must have [1,L] shape')
                out.update(crop_start=start, normfactor=float(r), raw_anchor=desc(A),
                           expected_roundtrip=desc(expected_anchor), noisy=desc(N))
                torch.manual_seed(row['expected_seed']); torch.cuda.manual_seed_all(row['expected_seed'])
                before = result['counts']['score']
                result['counts']['backend'] += 1
                require(result['counts']['backend'] <= cfg['limits']['backend_calls'], 'Backend-call cap exceeded')
                sampler = original_sampler('reverse_diffusion', 'none', A, N=30,
                    score_fn=lambda z,y,t: model(z,y,t,y_cond=N), corrector_steps=1, snr=.5,
                    intermediate=False, regeneration_strength=None, anchor_calibrator=None, score_calibrator=None)
                sample, nfe = sampler()
                raw = model.to_audio(sample.squeeze(), L).reshape(1,L)
                out['raw_calls'] = result['counts']['score']-before
                require(out['raw_calls'] == nfe == 30, 'Raw branch call mismatch')
                active.clear(); active['public'] = True
                require(sys.getprofile() is None, 'Existing profiler would be overwritten')
                def profile(frame, event, retval):
                    if event == 'return' and frame.f_code is ScoreModel.enhance.__code__:
                        active['norm'] = float(frame.f_locals['norm_factor'])
                        active['length'] = int(frame.f_locals['T_orig'])
                torch.manual_seed(row['expected_seed']); torch.cuda.manual_seed_all(row['expected_seed'])
                before = result['counts']['score']
                result['counts']['backend'] += 1
                require(result['counts']['backend'] <= cfg['limits']['backend_calls'], 'Backend-call cap exceeded')
                sys.setprofile(profile)
                try:
                    public_np = model.enhance(physical_front, y_cond=physical_noisy, N=30,
                        sampler_type='pc', predictor='reverse_diffusion', corrector='none',
                        corrector_steps=1, snr=.5, timeit=False, rrsd=False,
                        anchor_calib=False, anchor_v2=False, regen_strength=None)
                finally:
                    sys.setprofile(None)
                out['public_calls'] = result['counts']['score']-before
                require(out['public_calls'] == 30 and active['length'] == L, 'Public call/length mismatch')
                require(active['norm'] == float(r), 'Public noisy normalization mismatch')
                public = torch.from_numpy(public_np).to(device).reshape(1,L)/float(r)
                out.update(actual_public_anchor=desc(active['anchor']), actual_public_noisy=desc(active['noisy']),
                           actual_public_norm=active['norm'], raw_output=desc(raw),
                           public_normalized_output=desc(public), raw_public_anchor=diff(A,active['anchor']),
                           expected_actual_anchor=diff(expected_anchor,active['anchor']),
                           raw_public_noisy=diff(pad_spec(N),active['noisy']),
                           output_difference=diff(raw,public), raw_clean_mse=diff(s_norm,raw)['mse'],
                           public_clean_mse=diff(s_norm,public)['mse'])
                out['interface_passed'] = all(out[k]['relative_l2'] is not None and
                    out[k]['relative_l2'] <= tolerances['interface_relative_l2']
                    for k in ('expected_actual_anchor', 'raw_public_noisy'))
                old = old_rows[row['utt']]
                require(old['crop_start'] == start, 'Historical crop identity mismatch')
                baseline = old['normalized_wave_mse']
                out['old_baseline_sanity'] = dict(old_mse=baseline,
                    absolute_difference=abs(out['raw_clean_mse']-baseline),
                    passed=math.isclose(out['raw_clean_mse'],baseline,
                        abs_tol=tolerances['old_raw_mse_atol'], rel_tol=tolerances['old_raw_mse_rtol']))
                save()
                require(out['interface_passed'], 'Public interface invariant failed; stop without retry')
        require(result['counts'] == dict(model_loads=1, pcm_reads=16, frontend=8, backend=16, score=480),
                'Final call totals mismatch')
        require(not result['network_refusals'], 'Network refusal occurred; run cannot complete')
        result['status'] = 'completed'
    except BaseException as exc:
        result['status'] = 'failed'
        result['failure'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if handle is not None:
            handle.remove()
        if model is not None and 'original_sampler' in locals():
            model.get_pc_sampler = original_sampler
        try:
            result['source_post'] = {p: sha(p) for p in protected}
            require(result['source_post'] == protected, 'Protected source changed')
            for p,h in cfg['core_sha256'].items():
                require(sha(p) == h, 'Core source changed: ' + p)
        except BaseException as exc:
            result['status'] = 'failed'
            result['protection_failure'] = str(exc)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        try:
            save()
        except BaseException as exc:
            result['status'] = 'failed'
            result.setdefault('resource_failure', dict(message=str(exc)))
            try:
                (BASE / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            except OSError:
                pass
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
