#!/usr/bin/env python3
"""CPU-only acoustic measurements of archived WAVs. Never imports model code.

Execute once with the project's env/bin/python. All runtime outputs are exclusive
creates in this directory. No WAVs, checkpoint, or prior records are written.
"""
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import signal
import soundfile as sf
import yaml

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_new(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def csv_new(name, rows):
    with (HERE / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pearson(a, b):
    a, b = a - np.mean(a), b - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else None


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else None


def lag_scan(x, y, rate, max_seconds):
    """Overlap-normalized Pearson r(x[t],y[t+lag]); positive = y later.

    Means and norms are recalculated on each overlapping segment. FFT computes
    only the numerator dot products, avoiding O(samples * lag_count).
    """
    n = len(x)
    assert n == len(y)
    limit = min(int(round(max_seconds * rate)), n - 2)
    lags = np.arange(-limit, limit + 1)
    left_x, right_x = np.maximum(0, -lags), np.minimum(n, n - lags)
    left_y, right_y = left_x + lags, right_x + lags
    count = right_x - left_x
    def sums(a, left, right):
        c = np.r_[0., np.cumsum(a)]
        return c[right] - c[left]
    sx, sy = sums(x, left_x, right_x), sums(y, left_y, right_y)
    vx = np.maximum(0., sums(x*x, left_x, right_x) - sx*sx/count)
    vy = np.maximum(0., sums(y*y, left_y, right_y) - sy*sy/count)
    numer = signal.correlate(y, x, mode='full', method='fft')[n-1+lags] - sx*sy/count
    den = np.sqrt(vx*vy)
    good = den > 0
    r = np.full(len(lags), np.nan)
    r[good] = np.clip(numer[good] / den[good], -1, 1)
    if not np.any(good):
        return {'best_positive_r': None, 'best_absolute_r': None, 'zero_lag_r': None}
    best = int(np.nanargmax(r))
    abest = int(np.nanargmax(np.abs(r)))
    return {'best_positive_r': float(r[best]), 'best_positive_lag_seconds': float(lags[best]/rate),
            'best_absolute_r': float(r[abest]), 'best_absolute_lag_seconds': float(lags[abest]/rate),
            'zero_lag_r': float(r[limit]) if good[limit] else None,
            'overlap_samples_best_positive': int(count[best]),
            'positive_lag_definition': 'output y later: compare clean x[t] with output y[t+lag]',
            'limit': 'Context-wide similarity diagnostic; not proof of timing displacement.'}


def features(wave, cfg):
    fs = cfg['sample_rate']
    lo, hi = cfg['extended_seconds']
    size, hop = round(fs*cfg['frame_ms']/1000), round(fs*cfg['hop_ms']/1000)
    centers = np.arange(round(lo*fs)+size//2, round(hi*fs)-size//2+1, hop)
    win = signal.windows.hann(size, sym=False)
    frames = np.stack([wave[c-size//2:c+size//2] for c in centers])
    weighted = frames * win
    # No epsilon added to signal power: exact silence has exactly zero RMS.
    rms = np.sqrt(np.sum(weighted**2, axis=1) / np.sum(win**2))
    raw_rms = np.sqrt(np.mean(frames**2, axis=1))
    spec = np.abs(np.fft.rfft(weighted, axis=1))**2
    spec[:, 1:-1] *= 2  # one-sided Parseval weights, DC and Nyquist unchanged
    freq = np.fft.rfftfreq(size, 1/fs)
    total = spec.sum(axis=1)
    high = spec[:, (freq >= 4000) & (freq <= 7500)].sum(axis=1)
    frac = np.divide(high, total, out=np.zeros_like(high), where=total>0)
    flat = []
    for s in spec:
        # Flatness uses non-DC bins; a true zero bin makes geometric mean zero.
        bins = s[1:]
        flat.append(float(np.exp(np.mean(np.log(bins)))/np.mean(bins)) if np.all(bins>0) else 0.)
    bands = np.stack([spec[:, (freq >= a) & ((freq < b) if b < fs/2 else (freq <= b))].sum(axis=1)
                      for a,b in zip(cfg['spectrum_band_edges_hz'][:-1], cfg['spectrum_band_edges_hz'][1:])], axis=1)
    floor_amp = 10**(cfg['db_display_floor']/20)
    threshold = cfg['near_quantization_floor_pcm16_steps']/32768
    rows = []
    size40 = round(fs*cfg['periodicity_window_ms']/1000)
    minlag = int(np.ceil(fs/cfg['periodicity_frequency_hz'][1]))
    maxlag = int(np.floor(fs/cfg['periodicity_frequency_hz'][0]))
    for i,c in enumerate(centers):
        z = wave[c-size40//2:c+size40//2].copy()
        z -= z.mean()
        z *= signal.windows.hann(size40, sym=False)
        rr = []
        for lag in range(minlag,maxlag+1):
            den = np.linalg.norm(z[:-lag])*np.linalg.norm(z[lag:])
            rr.append(float(np.dot(z[:-lag],z[lag:])/den) if den>0 else 0.)
        j = int(np.argmax(rr))
        row = {'time_seconds': float(c/fs), 'frame_start_seconds': float((c-size//2)/fs),
               'frame_end_seconds': float((c+size//2)/fs), 'rms_hann': float(rms[i]),
               'rms_unwindowed': float(raw_rms[i]),
               'dbfs_hann': float(20*np.log10(max(rms[i],floor_amp))),
               'highband_4_7p5k_fraction': float(frac[i]) if total[i]>0 else None,
               'spectral_flatness_non_dc': flat[i] if total[i]>0 else None,
               'zero_crossing_rate': float(np.mean(np.signbit(frames[i,1:]) != np.signbit(frames[i,:-1]))),
               'periodicity_40ms_max_r': rr[j] if np.any(z) else None,
               'periodicity_peak_lag_samples': j+minlag if np.any(z) else None,
               'periodicity_peak_hz_proxy': float(fs/(j+minlag)) if np.any(z) else None,
               'near_pcm16_quantization_floor': bool(rms[i] <= threshold)}
        for k,power in enumerate(bands[i]):
            row[f'band_{k:02d}_power'] = float(power)
        rows.append(row)
    return {'rows':rows,'times':centers/fs,'rms':rms,'bands':bands,'wave':wave[round(lo*fs):round(hi*fs)]}


def summary(f, lo, hi):
    mask = (f['times'] >= lo-1e-8) & (f['times'] < hi-1e-8)
    rows = [r for r,m in zip(f['rows'],mask) if m]
    result = {'start_seconds':float(lo),'end_seconds':float(hi),'frame_count':len(rows)}
    for key in ['rms_hann','dbfs_hann','highband_4_7p5k_fraction','spectral_flatness_non_dc','zero_crossing_rate','periodicity_40ms_max_r']:
        a = [r[key] for r in rows if r[key] is not None]
        result[key+'_mean'] = float(np.mean(a)) if a else None
        result[key+'_max'] = float(np.max(a)) if a else None
    result['near_floor_fraction'] = float(np.mean([r['near_pcm16_quantization_floor'] for r in rows])) if rows else None
    return result


def main():
    # Guard all known artifacts before creating any of them. The lock also
    # prevents simultaneous workers from partially interleaving this run.
    outputs = ['metrics.json','experiment_report.md','frames.csv','blocks_100ms.csv',
               'tail_clean_spectrum_scan.csv','provenance.json']
    cfg = yaml.safe_load((HERE/'config.yaml').read_text())
    if cfg['plot']:
        outputs += ['context_original.png','context_noise_minus10dB.png','context_noise_minus20dB.png']
    for name in outputs:
        if (HERE/name).exists():
            raise FileExistsError(f'Refusing to overwrite {HERE/name}')
    with (HERE/'run_started.json').open('x') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat()},f)
    try:
        run(cfg)
    except Exception as exc:
        json_new('failure.json',{'utc':datetime.now(timezone.utc).isoformat(),'type':type(exc).__name__,'message':str(exc)})
        raise


def run(cfg):
    source_record = (HERE/cfg['source_metrics']).resolve()
    manifest_path = (HERE/cfg['source_clip_manifest']).resolve()
    record = json.loads(source_record.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert record['status'] == 'completed' and len(manifest) == 15
    indexed = {(r['variant'],r['arm']):r for r in manifest}
    assert len(indexed) == 15
    provenance = {'config_sha256':sha(HERE/'config.yaml'),'script_sha256':sha(__file__),
                  'source_metrics_sha256':sha(source_record),'source_clip_manifest_sha256':sha(manifest_path),
                  'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                  'soundfile':sf.__version__,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
                  'git_status':subprocess.check_output(['git','status','--short'],cwd=HERE,text=True), 'sources':[]}
    fs = cfg['sample_rate']
    cache = {}
    def load(path, expected_sha, expected_decoded=None):
        path = str(Path(path).resolve())
        if path not in cache:
            h = sha(path)
            assert h == expected_sha, f'Source hash mismatch: {path}'
            wave, rate = sf.read(path,dtype='float64')
            assert rate == fs and wave.ndim == 1 and len(wave) == cfg['source_samples'] and np.all(np.isfinite(wave))
            decoded = hashlib.sha256(wave.tobytes()).hexdigest()
            if expected_decoded:
                assert decoded == expected_decoded, f'Decoded hash mismatch: {path}'
            cache[path] = wave
            provenance['sources'].append({'path':path,'sha256':h,'decoded_float64_sha256':decoded,'samples':len(wave),'sample_rate':rate})
        else:
            assert sha(path) == expected_sha
        return cache[path]
    noise_info = record['source_amplitude_evidence']
    noise = load(noise_info['archived_noise_path'],noise_info['archived_noise_sha256'],noise_info['archived_noise_decoded_float64_sha256'])
    all_frames, all_blocks, scans = [], [], []
    metrics = {'experiment_id':cfg['experiment_id'],'status':'completed','new_model_inference':False,
               'extended_seconds':cfg['extended_seconds'],'reported_tail_seconds':cfg['reported_tail_seconds'],
               'clip_end_is_utterance_end':False,'interpretation_limits':cfg['interpretation_limits'],
               'feature_notes':'RMS=sqrt(sum((frame*Hann)^2)/sum(Hann^2)); epsilon never added to power. dB floor applied only after sqrt. 40ms periodicity is an acoustic proxy, not phoneme truth. Spectral cosines use 16 fixed linear-frequency power bands.',
               'variants':{}}
    tail_lo,tail_hi = cfg['reported_tail_seconds']
    block_edges = np.linspace(tail_lo,tail_hi,8)
    for variant,v in record['variants'].items():
        data = {}
        for arm,path in v['arms'].items():
            m = indexed[(variant,arm)]
            assert str(Path(path).resolve()) == str(Path(m['source_path']).resolve())
            assert m['start_sample'] == 220000 and m['end_sample_exclusive'] == 252000
            wave = load(path,m['source_sha256'],m['source_decoded_float64_sha256'])
            clip = (manifest_path.parent/m['clip']).resolve()
            assert sha(clip) == m['clip_sha256']
            clipwave,cliprate = sf.read(clip,dtype='float64')
            assert cliprate == fs and np.array_equal(clipwave,wave[220000:252000])
            data[arm] = features(wave,cfg)
        data['noise_source'] = features(noise,cfg)
        data['scaled_noise'] = features(float(v['alpha'])*noise,cfg)
        vm = {'alpha':v['alpha'],'clean_gain':v['clean_gain'],'noise_gain':v['noise_gain'],
              'source_snr_metadata':v['clean_to_scaled_archived_noise_snr_db'],'arms':{},'clean_output_context_lags':{},'tail_noise_output_similarity':{}}
        for arm,f in data.items():
            all_frames.extend(dict(variant=variant,arm=arm,**row) for row in f['rows'])
            blocks = [summary(f,a,b) for a,b in zip(block_edges[:-1],block_edges[1:])]
            all_blocks.extend(dict(variant=variant,arm=arm,**b) for b in blocks)
            vm['arms'][arm] = {'fixed_tail':summary(f,tail_lo,tail_hi),
                               'before_tail':summary(f,cfg['extended_seconds'][0],tail_lo),
                               'after_tail':summary(f,tail_hi,cfg['extended_seconds'][1]),'tail_blocks_100ms':blocks}
        clean = data['clean']
        valid_clean = clean['rms'] > cfg['near_quantization_floor_pcm16_steps']/32768
        for arm in ['noisy','y_pre','off','router']:
            f = data[arm]
            vm['clean_output_context_lags'][arm] = {
                'fullband':lag_scan(clean['wave'],f['wave'],fs,cfg['lag_search_seconds']),
                'rms_envelope':lag_scan(clean['rms'],f['rms'],1000/cfg['hop_ms'],cfg['lag_search_seconds'])}
            tail = (f['times'] >= tail_lo-1e-8) & (f['times'] < tail_hi-1e-8)
            n = data['scaled_noise']
            vm['tail_noise_output_similarity'][arm] = {
                'rms_envelope_pearson_r':pearson(n['rms'][tail],f['rms'][tail]),
                'mean_band_power_cosine':cosine(n['bands'][tail].mean(axis=0),f['bands'][tail].mean(axis=0)),
                'noise_near_floor_fraction':float(np.mean(n['rms'][tail] <= cfg['near_quantization_floor_pcm16_steps']/32768)),
                'output_near_floor_fraction':float(np.mean(f['rms'][tail] <= cfg['near_quantization_floor_pcm16_steps']/32768)),
                'blocks_100ms':[]}
            for a,b in zip(block_edges[:-1],block_edges[1:]):
                bm = (f['times'] >= a-1e-8) & (f['times'] < b-1e-8)
                vm['tail_noise_output_similarity'][arm]['blocks_100ms'].append({
                    'start_seconds':float(a),'end_seconds':float(b),
                    'rms_envelope_pearson_r':pearson(n['rms'][bm],f['rms'][bm]),
                    'mean_band_power_cosine':cosine(n['bands'][bm].mean(axis=0),f['bands'][bm].mean(axis=0)),
                    'noise_near_floor_fraction':float(np.mean(n['rms'][bm] <= cfg['near_quantization_floor_pcm16_steps']/32768)),
                    'output_near_floor_fraction':float(np.mean(f['rms'][bm] <= cfg['near_quantization_floor_pcm16_steps']/32768))})
            for i in np.flatnonzero(tail):
                a = f['bands'][i]
                norms = np.linalg.norm(clean['bands'],axis=1)*np.linalg.norm(a)
                sims = np.divide(clean['bands']@a,norms,out=np.full(len(norms),np.nan),where=norms>0)
                good = valid_clean & np.isfinite(sims)
                query_low = bool(f['rows'][i]['near_pcm16_quantization_floor'])
                # Report the unfiltered winner as well as the conservative winner.
                # Low energy queries are explicitly invalidated, never treated as matches.
                rawj = int(np.nanargmax(sims)) if np.any(np.isfinite(sims)) else None
                j = int(np.nanargmax(np.where(good,sims,np.nan))) if np.any(good) and not query_low else None
                scans.append({'variant':variant,'arm':arm,'tail_time_seconds':float(f['times'][i]),
                    'query_rms_hann':float(f['rms'][i]),'query_near_floor':query_low,
                    'best_valid_clean_time_seconds':float(clean['times'][j]) if j is not None else None,
                    'best_valid_cosine':float(sims[j]) if j is not None else None,
                    'best_valid_clean_rms_hann':float(clean['rms'][j]) if j is not None else None,
                    'best_valid_offset_clean_minus_output_seconds':float(clean['times'][j]-f['times'][i]) if j is not None else None,
                    'unfiltered_best_clean_time_seconds':float(clean['times'][rawj]) if rawj is not None else None,
                    'unfiltered_best_cosine':float(sims[rawj]) if rawj is not None else None,
                    'unfiltered_reference_near_floor':bool(not valid_clean[rawj]) if rawj is not None else None,
                    'valid_clean_reference_frame_count':int(valid_clean.sum())})
        metrics['variants'][variant] = vm
        if cfg['plot']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes = plt.subplots(7,2,figsize=(14,14),sharex=True)
            maxamp = max(np.max(np.abs(f['wave'])) for f in data.values())
            for (arm,f),rowaxes in zip(data.items(),axes):
                t = cfg['extended_seconds'][0]+np.arange(len(f['wave']))/fs
                rowaxes[0].plot(t[::8],f['wave'][::8],lw=.35)
                rowaxes[0].set_ylim(-maxamp,maxamp)
                rowaxes[0].set_ylabel(arm)
                # Band display dB power re FFT normalization; common limits and axes.
                band_db = 10*np.log10(np.maximum(f['bands'].T/(320**2),1e-24))
                rowaxes[1].imshow(band_db,origin='lower',aspect='auto',extent=[f['times'][0]-.005,f['times'][-1]+.005,0,8000],vmin=-110,vmax=-30,cmap='magma')
                for ax in rowaxes:
                    ax.axvspan(tail_lo,tail_hi,color='cyan',alpha=.15)
                    ax.axvline(13.75,color='green',lw=.6)
                    ax.axvline(15.75,color='green',lw=.6)
                    ax.set_xlim(*cfg['extended_seconds'])
            axes[0,0].set_title('Waveforms, common amplitude scale; cyan = fixed reported tail')
            axes[0,1].set_title('500 Hz band powers, common dB scale [-110,-30]')
            axes[-1,0].set_xlabel('Source time (s)')
            axes[-1,1].set_xlabel('Source time (s)')
            fig.suptitle(variant+'; periodicity and spectra do not identify phonemes')
            fig.tight_layout()
            with (HERE/f'context_{variant}.png').open('xb') as out:
                fig.savefig(out,format='png',dpi=110)
            plt.close(fig)
    for src in provenance['sources']:
        assert sha(src['path']) == src['sha256'], f'Source changed during probe: {src["path"]}'
    metrics['all_sources_unchanged'] = True
    metrics['source_manifest_entries_validated'] = len(manifest)
    metrics['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    csv_new('frames.csv',all_frames)
    csv_new('blocks_100ms.csv',all_blocks)
    csv_new('tail_clean_spectrum_scan.csv',scans)
    json_new('provenance.json',provenance)
    json_new('metrics.json',metrics)
    report = ['# Archived 10720 acoustic tail probe','',
              'Completed CPU-only analysis of existing sources; no model inference.',
              '', 'The original 13.75–15.75 s clip ends within the full 30 s source. Extended context is 12.75–17.75 s; the fixed reported tail is 15.05–15.75 s.',
              '', 'Raw 20 ms Hann / 10 ms hop values are in frames.csv; fixed tail aggregates and 100 ms blocks are in metrics.json and blocks_100ms.csv. Clean before/after activity is retained without a speech detection threshold.',
              '', 'tail_clean_spectrum_scan.csv scans every tail frame against all clean frames in the five second context. References at or below three PCM16 steps RMS are excluded from valid winners; unfiltered winners and their floor flags remain available. Query frames near that floor have no valid winner.',
              '', 'Context lag searches normalize each overlapping window and report fullband and RMS envelope similarity over ±1 s. Positive lag means output later than clean. These correlations and spectral matches do not prove timing displacement or causal noise conversion. Periodicity is a voicing proxy, not phoneme truth.',
              '', 'Source hashes, decoded hashes, manifest checks, code hashes, environment, and git state are recorded in provenance.json. All 15 original source manifest entries and their listening clip slices were verified; all source WAV hashes were unchanged after analysis.',
              '', '| Variant | Arm | Tail RMS mean | Highband fraction mean | Periodicity mean |',
              '|---|---|---:|---:|---:|']
    for variant,v in metrics['variants'].items():
        for arm,details in v['arms'].items():
            s = details['fixed_tail']
            report.append(f"| {variant} | {arm} | {s['rms_hann_mean']} | {s['highband_4_7p5k_fraction_mean']} | {s['periodicity_40ms_max_r_mean']} |")
    report += ['', 'Scientific interpretation is reserved for the Research Lead; no human words or phoneme identities are inferred.']
    with (HERE/'experiment_report.md').open('x') as out:
        out.write('\n'.join(report)+'\n')


if __name__ == '__main__':
    main()
