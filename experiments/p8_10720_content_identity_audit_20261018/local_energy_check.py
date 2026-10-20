#!/usr/bin/env python3
"""Read-only waveform energy measurements. No inference or voice classification."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent
INTERVALS = [(0., .5), (.5, 1.), (1., 1.3), (1.3, 2.)]
REFERENCE_AMPLITUDE = 1 / 32768  # Fixed PCM16 quantization reference, not a speech threshold.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(samples):
    rms = float(np.sqrt(np.mean(samples * samples)))
    return {'sample_values': int(samples.size), 'rms': rms,
            'rms_dbfs': float(20 * np.log10(rms)) if rms else None,
            'max_abs': float(np.max(np.abs(samples))),
            'exact_nonzero_count': int(np.count_nonzero(samples)),
            'exact_nonzero_fraction': float(np.count_nonzero(samples) / samples.size),
            'exact_zero_count': int(np.count_nonzero(samples == 0)),
            'absolute_amplitude_at_most_one_pcm16_lsb_count': int(np.count_nonzero(np.abs(samples) <= REFERENCE_AMPLITUDE)),
            'absolute_amplitude_at_most_one_pcm16_lsb_fraction': float(np.mean(np.abs(samples) <= REFERENCE_AMPLITUDE))}


def segment(samples, rate, start, stop):
    first, last = round(start * rate), round(stop * rate)
    return {'start_s': start, 'stop_exclusive_s': stop, 'first_sample': first,
            'stop_exclusive_sample': last, **stats(samples[first:last])}


def main():
    dest = BASE / 'local_energy_check.json'
    if dest.exists():
        raise SystemExit('Refusing existing results: ' + str(dest))
    metrics_path = BASE / 'metrics.json'
    metrics_hash = sha(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    result = {'status': 'completed', 'scope': 'offline_read_only_waveform_energy',
              'sources': {'audit_metrics': {'path': str(metrics_path), 'sha256': metrics_hash},
                          'script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}},
              'interval_definition': 'Half-open sample intervals; bounds rounded to exact sample indices.',
              'frame_seconds': .1,
              'fixed_amplitude_reference': {'value': REFERENCE_AMPLITUDE,
                  'meaning': 'One PCM16 quantization step; descriptive fixed reference, not a voice, silence or audibility threshold.'},
              'conditions': {}}
    for condition, source in metrics['source_audio_levels'].items():
        path = Path(source['path'])
        digest = sha(path)
        assert digest == source['sha256']
        samples, rate = sf.read(path, dtype='float64', always_2d=True)
        assert samples.shape[1] == 1 and len(samples) == 2 * rate and np.isfinite(samples).all()
        nonzero = np.flatnonzero(np.any(samples != 0, axis=1))
        last = int(nonzero[-1]) if len(nonzero) else None
        result['conditions'][condition] = {
            'source_path': str(path), 'source_sha256': digest, 'sample_rate': rate,
            'channels': samples.shape[1], 'sample_count': len(samples),
            'soundfile_subtype': sf.info(path).subtype,
            'last_exact_nonzero_sample_index': last,
            'last_exact_nonzero_sample_time_s': last / rate if last is not None else None,
            'trailing_exact_zero_samples': len(samples) - last - 1 if last is not None else len(samples),
            'whole_clip': stats(samples),
            'fixed_intervals': [segment(samples, rate, a, z) for a, z in INTERVALS],
            'frames_100ms': [segment(samples, rate, i / 10, (i + 1) / 10) for i in range(20)]}
        assert sha(path) == digest
    clean = result['conditions']['clean']['fixed_intervals'][-1]
    off = result['conditions']['off']['fixed_intervals'][-1]
    ratio = off['rms'] / clean['rms'] if clean['rms'] else None
    result['late_tail_comparison'] = {
        'start_s': 1.3, 'stop_exclusive_s': 2.,
        'clean': clean, 'off': off,
        'off_over_clean_rms_ratio': ratio,
        'off_minus_clean_rms_db': float(20 * np.log10(ratio)) if ratio and ratio > 0 else None,
        'clean_last_greedy_token_end_s': 1.22,
        'clean_last_token_end_basis': 'Prior cached clean CTC timeline, approximate recognizer run end; not a measured speech offset.',
        'clean_last_exact_nonzero_sample_time_s': result['conditions']['clean']['last_exact_nonzero_sample_time_s']}
    result['limits'] = [
        'Energy, exact nonzero values and the fixed quantization reference do not establish voiced speech, speech identity, silence or audibility.',
        'No acoustic speech endpoint is inferred from the clean final CTC token end at 1.22 seconds.',
        'No optimized energy threshold, spectral model, inference, new download, training or source edit.',
        'Per-clip recognizer normalization remains a confound when comparing waveform levels with decoded token counts.',
        'Exact sample nonzero counts depend on the stored WAV encoding and quantization.']
    assert sha(metrics_path) == metrics_hash
    with dest.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result['late_tail_comparison'], indent=2))
    print('Last nonzero:', result['conditions']['clean']['last_exact_nonzero_sample_time_s'])


if __name__ == '__main__':
    main()
