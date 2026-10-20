"""Single bounded same-utterance positive control; execute once via experiment worker."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time, traceback

ROOT = Path('/data/coding/sgmse')
OUT = Path(__file__).resolve().parent
FORENSIC = ROOT / 'experiments/p8_10720_pairing_forensics_20261018'
CKPT = ROOT / 'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
CHECKPOINT_SHA = 'ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9'
SR, LENGTH, A, B, SEED, CAP = 16000, 480000, 220000, 252000, 2997368683, 50_000_000
CONTROLS = {'noise_minus10dB': 10 ** (-10 / 20), 'noise_minus20dB': 10 ** (-20 / 20)}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_json(name, data):
    with (OUT / name).open('x') as f: json.dump(data, f, indent=2, default=str); f.write('\n')
def cap():
    size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert size < CAP, f'Output size cap exceeded: {size}'
    return size

def main():
    # Both failed and completed executions require diagnosis, never an automatic retry.
    allowed = {'run_control.py', 'config.yaml', '__pycache__'}
    unexpected = [p.name for p in OUT.iterdir() if p.name not in allowed and p.name != 'run.log']
    assert not unexpected, f'Refuse existing/partial artifacts: {unexpected}'
    with (OUT / 'run_started.json').open('x') as f: json.dump({'started_at_utc': utc()}, f)
    d = {'experiment_id': OUT.name, 'status': 'running', 'started_at_utc': utc(), 'failures': [],
         'hypothesis': 'Same clean utterance plus its archived paired noise at lower noise gains is a positive control for pipeline content preservation.',
         'limits': 'This offline oracle control uses stored clean speech and paired archived noise; it is not deployable enhancement or performance proof. Success does not establish original extreme-low-SNR recoverability. Original noisy differs from stored clean plus archived noise by at most one PCM16 step; baseline remains the actual original noisy and original outputs. Listening judgments are not collected yet.',
         'method': 'control=stored_clean+alpha*archived_paired_noise; clean gain always1; noise gain alpha=10**(-10/20) or10**(-20/20). The original actual-noisy baseline and its outputs are reused without inference. SNR uses clean power divided by scaled archived-noise power.',
         'source_manifest': str(FORENSIC / 'metrics.json'), 'effective_per_file_seed': SEED,
         'git_head': subprocess.getoutput('git rev-parse HEAD'), 'git_status': subprocess.getoutput('git status --short')}
    t0 = time.monotonic()
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import yaml
        sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
        import joint_enhance_by_dir as mod
        cfg = yaml.safe_load((OUT / 'config.yaml').read_text())
        assert cfg['effective_per_file_seed'] == SEED and cfg['N'] == 30 and cfg['sampling_snr'] == .5
        assert cfg['controls'] == CONTROLS and cfg['focus_samples'] == [A, B]
        original = json.loads((FORENSIC / 'metrics.json').read_text())
        sources = {arm: Path(rec['path']) for arm, rec in original['sources'].items()}
        assert set(sources) == {'clean', 'noisy', 'y_pre', 'off', 'router'}
        d['original_sources'] = {}
        arrays = {}
        def pcm_hash(x): return hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()
        for arm, src in sources.items():
            x, sr = sf.read(src, dtype='float64'); assert sr == SR and x.shape == (LENGTH,) and np.isfinite(x).all()
            assert sha(src) == original['sources'][arm]['sha256']
            arrays[arm] = x
            d['original_sources'][arm] = {'path': str(src), 'sha256': sha(src), 'decoded_float64_sha256': pcm_hash(x), 'samples': len(x), 'sample_rate': sr, 'subtype': sf.info(src).subtype}
        name = sources['off'].name
        assert sources['router'].name == sources['y_pre'].name == name
        assert (2026 + int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), 'little')) % (2**32) == SEED
        assert sha(CKPT) == CHECKPOINT_SHA == cfg['checkpoint_sha256']
        replays = {}
        for arm, rel in [('off', 'replay_off/attempt02'), ('router', 'replay_router')]:
            rp = FORENSIC / rel / 'metrics.json'; r = json.loads(rp.read_text())
            assert r['status'] == 'completed' and r['byte_hash_matches_saved'] is True
            assert r['checkpoint_sha256'] == CHECKPOINT_SHA and r['effective_per_file_seed'] == SEED
            assert r['input_sha256'] == d['original_sources']['noisy']['sha256']
            replays[arm] = r
        common = {k: v for k, v in replays['off']['enhance_kwargs'].items() if k not in ('y_cond', 'timeit', 'rrsd')}
        assert common == {k: v for k, v in replays['router']['enhance_kwargs'].items() if k not in ('y_cond', 'timeit', 'rrsd')}
        assert common['N'] == 30 and common['snr'] == .5 and common['corrector'] == 'none' and common['sampler_type'] == 'pc'
        assert common['predictor'] == 'reverse_diffusion' and common['rrsd_scale'] == 1.0
        d['enhance_kwargs'] = common; d['checkpoint'] = {'path': str(CKPT), 'sha256': CHECKPOINT_SHA}
        codefiles = ['joint_enhance_by_dir.py', 'sgmse/joint_model.py', 'sgmse/joint_data_module.py', 'sgmse/model.py', 'sgmse/frontends/storm.py']
        d['code_sha256'] = {f: sha(ROOT / f) for f in codefiles}
        for r in replays.values():
            assert d['code_sha256'] == r['script_sha256'], 'Code changed since successful exact replay'
        d['control_script_sha256'] = sha(__file__); d['config_sha256'] = sha(OUT / 'config.yaml')
        d['replay_evidence'] = {arm: {'driver_sha256': sha(FORENSIC / rel / 'run_replay.py'), 'metrics_sha256': sha(FORENSIC / rel / 'metrics.json')} for arm, rel in [('off', 'replay_off/attempt02'), ('router', 'replay_router')]}
        d['environment'] = {'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'device': torch.cuda.get_device_name(0), 'PATH': os.environ.get('PATH'), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark}
        x = arrays['clean']
        noise_path = Path(cfg['archived_noise_path'])
        assert noise_path.exists(), 'Required archived paired noise missing; no residual fallback'
        assert sha(noise_path) == cfg['archived_noise_sha256']
        noise, noise_sr = sf.read(noise_path, dtype='float64')
        assert noise_sr == SR and noise.shape == (LENGTH,) and np.isfinite(noise).all()
        error = arrays['noisy'] - x - noise
        assert np.max(abs(error)) <= 1/32768, 'Archived pair is not additive within one PCM16 step'
        gains = np.linalg.lstsq(np.stack([x, noise], axis=1), arrays['noisy'], rcond=None)[0]
        d['source_amplitude_evidence'] = {'archived_noise_path': str(noise_path), 'archived_noise_sha256': sha(noise_path), 'archived_noise_decoded_float64_sha256': pcm_hash(noise), 'samples': len(noise), 'sample_rate': noise_sr, 'max_abs_noisy_minus_clean_minus_archived_noise': float(np.max(abs(error))), 'max_error_pcm16_steps': float(np.max(abs(error))*32768), 'residual_noise_correlation': float(np.corrcoef(arrays['noisy']-x, noise)[0,1]), 'fitted_clean_gain': float(gains[0]), 'fitted_noise_gain': float(gains[1]), 'interpretation': 'Stored mixture equals stored clean plus archived paired noise to independent PCM16 quantization precision; actual original noisy is retained as baseline.'}
        def snr(signal, scaled_noise):
            return float(10 * np.log10(np.sum(signal ** 2) / np.sum(scaled_noise ** 2)))
        def snrs(alpha): return {'full_30s_db': snr(x, alpha*noise), 'focus_13p75_15p75_db': snr(x[A:B], alpha*noise[A:B])}
        d['variants'] = {'original': {'alpha': 1.0, 'clean_gain': 1.0, 'noise_gain': 1.0, 'clean_to_scaled_archived_noise_snr_db': snrs(1.0), 'baseline_mixture_max_error_pcm16_steps': float(np.max(abs(error))*32768), 'arms': {arm: str(src) for arm, src in sources.items()}, 'inference_reused': True}}
        for variant, alpha in CONTROLS.items():
            folder = OUT / variant; (folder / 'inputs').mkdir(parents=True)
            y = x + alpha * noise
            assert np.isfinite(y).all() and np.max(np.abs(y)) < 1
            dst = folder / 'inputs' / name
            sf.write(dst, y, SR, subtype='FLOAT')
            yr, sr = sf.read(dst, dtype='float64'); assert sr == SR and yr.shape == (LENGTH,)
            assert np.max(np.abs(yr - y)) <= np.finfo(np.float32).eps * max(1., float(np.max(abs(y))))
            assert np.max(np.abs(yr - y)) < 1e-8
            d['variants'][variant] = {'alpha': alpha, 'clean_gain': 1.0, 'noise_gain': alpha, 'clean_to_scaled_archived_noise_snr_db': snrs(alpha), 'input_sha256': sha(dst), 'input_decoded_float64_sha256': pcm_hash(yr), 'additive_float_write_max_error': float(np.max(abs(yr-y))), 'input_peak': float(np.max(abs(yr))), 'arms': {'clean': str(sources['clean']), 'noisy': str(dst)}}
        write_json('launch_provenance.json', d)
        model = mod.JointSemambaSgmseModel.load_from_checkpoint(str(CKPT), map_location='cpu', strict=True, data_module_cls=mod.JointWaveDataModule, base_dir='.', batch_size=1, num_workers=0, dummy=True, weights_only=False)
        model.eval(no_ema=False).to('cuda:0')
        assert model.train_type == 'mode2' and not model.residual_refine_enable
        d['loaded_model'] = {'train_type': model.train_type, 'frontend': str(getattr(model.hparams, 'frontend', None)), 'residual_refine_enable': model.residual_refine_enable}
        assert d['loaded_model']['frontend'] == 'storm'
        with torch.inference_mode():
            for variant in CONTROLS:
                r = d['variants'][variant]; folder = OUT / variant
                torch.manual_seed(SEED)
                y = mod.load_mono(r['arms']['noisy'], SR).to('cuda:0')[None, None, :]
                cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
                norm = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                tt = time.monotonic(); pre = model._frontend_forward(y / norm) * norm
                post = mod._match_length(pre, y); torch.cuda.synchronize()
                assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state()), 'Frontend consumes RNG: resetting enhance RNG would differ from exact replay'
                assert post.shape == y.shape
                pre_np = post.squeeze().cpu().numpy()
                frontend_path = folder / 'y_pre' / name; frontend_path.parent.mkdir()
                sf.write(frontend_path, np.clip(pre_np, -1., 1.), SR, subtype='PCM_16')
                r['arms']['y_pre'] = str(frontend_path)
                r['frontend'] = {'before_match_samples': int(pre.shape[-1]), 'after_match_samples': int(post.shape[-1]), 'input_samples': int(y.shape[-1]), 'float32_sha256': hashlib.sha256(pre_np.tobytes()).hexdigest(), 'output_sha256': sha(frontend_path), 'wall_seconds': time.monotonic()-tt, 'rng_unchanged': True, 'clipped_sample_count': int(np.count_nonzero(abs(pre_np)>1))}
                for arm in ('off', 'router'):
                    torch.manual_seed(SEED)
                    tt = time.monotonic()
                    enhanced, nfe, rtf = model.enhance(post.squeeze(), y_cond=y.squeeze(), **common, rrsd=arm=='router', timeit=True)
                    torch.cuda.synchronize()
                    enhanced = np.asarray(enhanced); assert enhanced.shape == (LENGTH,) and np.isfinite(enhanced).all()
                    dst = folder / arm / name; dst.parent.mkdir()
                    sf.write(dst, np.clip(enhanced, -1., 1.), SR, subtype='PCM_16')
                    r['arms'][arm] = str(dst)
                    r[arm] = {'nfe': int(nfe), 'model_reported_rtf': float(rtf), 'wall_seconds': time.monotonic()-tt, 'sha256': sha(dst), 'effective_seed': SEED, 'clipped_sample_count': int(np.count_nonzero(abs(enhanced)>1))}
                    cap()
        clips = OUT / 'clips'; clips.mkdir(); manifest = []
        clean_clip = clips / 'common_clean_13.75-15.75.wav'
        sf.write(clean_clip, x[A:B], SR, subtype='FLOAT')
        for variant, r in d['variants'].items():
            for arm in ('clean', 'noisy', 'y_pre', 'off', 'router'):
                src = Path(r['arms'][arm]); a, sr = sf.read(src, dtype='float64')
                assert sr == SR and a.shape == (LENGTH,)
                dst = clean_clip if arm == 'clean' else clips / f'{variant}_{arm}_13.75-15.75.wav'
                if arm != 'clean': sf.write(dst, a[A:B], SR, subtype='FLOAT')
                clip, sr = sf.read(dst, dtype='float64')
                assert clip.shape == (B-A,) and np.array_equal(clip, a[A:B].astype(np.float32).astype(np.float64))
                manifest.append({'variant': variant, 'arm': arm, 'source_path': str(src), 'source_sha256': sha(src), 'source_decoded_float64_sha256': pcm_hash(a), 'source_samples': LENGTH, 'source_subtype': sf.info(src).subtype, 'start_sample': A, 'end_sample_exclusive': B, 'sample_rate': SR, 'clip': str(dst.relative_to(OUT)), 'clip_sha256': sha(dst), 'clip_decoded_float64_sha256': pcm_hash(clip), 'clip_matches_source': True})
        write_json('source_clip_manifest.json', manifest)
        build_page(d, manifest)
        assert all(sha(sources[arm]) == rec['sha256'] for arm, rec in d['original_sources'].items())
        assert sha(CKPT) == CHECKPOINT_SHA
        assert sha(noise_path) == cfg['archived_noise_sha256']
        d['sources_and_checkpoint_unchanged'] = True; d['output_bytes'] = cap(); d['status'] = 'completed'
        d['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(0)
    except Exception as exc:
        d['status'] = 'failed'; d['failures'].append({'error': repr(exc), 'traceback': traceback.format_exc()}); traceback.print_exc()
    d['completed_at_utc'] = utc(); d['wall_seconds'] = time.monotonic() - t0
    write_json('metrics.json', d)
    report = '# 10720 paired archived-noise attenuation positive control\n\nStatus: '+d['status']+'\n\n'+d['hypothesis']+'\n\n'+d['method']+'\n\n'+d['limits']+'\n\n'
    if d['status'] == 'completed':
        report += 'All five arms use source samples220000:252000 (13.75–15.75s). Original baseline outputs were reused. Stored paired noise was attenuated by10/20dB with clean gain1; clean/scaled archived-noise SNR is recorded below. Original noisy differs from clean+archived-noise by at most1PCM16 step and is reused as saved. Fixed playback gain1, no RMS normalization. No listening judgment has yet been recorded.\n\n|Variant|Full30s SNRdB|Focus SNRdB|\n|---|---:|---:|\n'
        for variant, r in d['variants'].items(): report += f"|{variant}|{r['clean_to_scaled_archived_noise_snr_db']['full_30s_db']:.4f}|{r['clean_to_scaled_archived_noise_snr_db']['focus_13p75_15p75_db']:.4f}|\n"
        report += '\nReproducibility: config.yaml, run_control.py, launch_provenance.json, metrics.json and source_clip_manifest.json include seeds, source/checkpoint/code hashes, timing and NFE. Inputs are FLOAT WAV; frontend/backend saved as original PCM16 pipeline. Source pair and checkpoint hashes verified unchanged.\n'
    else: report += 'Failure recorded in metrics.json. Do not retry without diagnosis.\n'
    with (OUT / 'experiment_report.md').open('x') as f: f.write(report)
    print(json.dumps({'status': d['status'], 'wall_seconds': d['wall_seconds'], 'failures': d['failures']}, indent=2))
    if d['status'] != 'completed': raise SystemExit(1)


def build_page(d, manifest):
    rows = []
    labels = {'original': '原始输入（复用原结果）', 'noise_minus10dB': '配对噪声降低10dB', 'noise_minus20dB': '配对噪声降低20dB'}
    for variant, r in d['variants'].items():
        buttons = ''.join(f'<button data-variant="{variant}" data-arm="{m["arm"]}" data-src="{m["clip"]}">播放 {m["arm"]}</button>' for m in manifest if m['variant']==variant)
        rows.append(f'<section><h2>{labels[variant]}</h2><p>clean增益1；配对噪声增益{r["noise_gain"]:.4f}。整段SNR {r["clean_to_scaled_archived_noise_snr_db"]["full_30s_db"]:.2f}dB；本窗口SNR {r["clean_to_scaled_archived_noise_snr_db"]["focus_13p75_15p75_db"]:.2f}dB</p>{buttons}</section>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10720 同一句话噪声对照</title><style>body{font:18px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 18px}section{border:1px solid #bbb;padding:16px;margin:20px 0}button{font:inherit;padding:10px;margin:6px}#identity{overflow-wrap:anywhere}audio{width:100%}</style><h1>10720 · 同一句话的噪声对照</h1><p>每段都是同一原始时间13.75–15.75秒。先听clean，再比较noisy、y_pre、off、router。三行共用同一clean参考。默认固定增益1×，不做音量归一化；可使用设备音量调节。</p><p>对照输入为clean+α×配对存档噪声，clean增益始终为1；噪声增益分别为0.3162和0.1。原始输入与clean+存档噪声最多相差一个PCM16量化步长，原始结果直接复用。这是使用clean参考的离线对照，不是可部署的增强方法；较容易条件下的结果不能证明原始条件下的内容可恢复。请记录是否还是同一句话、哪些声音相同或不同。</p><button id="stop">停止</button><p id="identity" role="status" aria-live="polite">当前播放：无</p><audio id="player" controls preload="none"></audio>''' + ''.join(rows) + '''<script>
'use strict';
const player=document.getElementById('player'),identity=document.getElementById('identity');let generation=0,current=null;
function stop(){generation++;player.pause();current=null;identity.textContent='当前播放：无';}
async function select(button){stop();const token=generation;const info=button.dataset.variant+' · '+button.dataset.arm+' · 13.75–15.75秒 · '+button.dataset.src;
current={token,info};identity.textContent='正在加载：'+info;player.src=button.dataset.src;player.volume=1;player.playbackRate=1;player.load();
try{await player.play();if(token!==generation)return;identity.textContent='当前播放：'+info;}catch(error){if(token===generation){current=null;identity.textContent='播放失败：'+info+' · '+error.message;}}}
for(const button of document.querySelectorAll('button[data-src]'))button.addEventListener('click',()=>select(button));
document.getElementById('stop').addEventListener('click',stop);
player.addEventListener('play',()=>{if(current&&current.token===generation)identity.textContent='当前播放：'+current.info;});
player.addEventListener('ended',()=>{if(current&&current.token===generation)identity.textContent='播放结束：'+current.info;});
window.addEventListener('pagehide',stop);
</script></html>'''
    with (OUT / 'listening.html').open('x') as f: f.write(html)

if __name__ == '__main__': main()
