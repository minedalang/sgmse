"""Offline source-recording-support SNR audit; reads frozen recipe/NPZs only."""
from pathlib import Path
import hashlib
import json
import numpy as np

P = Path(__file__).resolve().parent
RECIPE = P / 'inputs/recipe_frozen.json'
OUT = P / 'support_snr_audit.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ah(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()

def power(array):
    return float(np.mean(np.square(array, dtype=np.float64), dtype=np.float64))

def stats(values):
    return dict(zip(('min', 'median', 'mean', 'max'), map(float, (
        np.min(values), np.median(values), np.mean(values), np.max(values)))))

def main():
    before = sha(RECIPE)
    assert before == '8936e3edc3f337356cba7943745a9ed2a9939a76e7ce0da4afd21eb84749d769'
    frozen = json.loads(RECIPE.read_text())
    assert len(frozen['recipes']) == 144
    sr = frozen['sample_rate']
    assert sr == 16000
    cache, clean_meta, array_hashes = {}, {}, {}
    for row in frozen['clean'] + frozen['noise']:
        path = P / row['array_path']
        array_hashes[row['array_path']] = sha(path)
        assert array_hashes[row['array_path']] == row['array_sha256']
        with np.load(path, allow_pickle=False) as archive:
            wave = archive['waveform']
        assert wave.dtype == np.float64 and len(wave) == 32000
        assert ah(wave) == row['float64_sha256']
        key = row.get('sample_id', row.get('noise_id'))
        cache[key] = wave
        if 'sample_id' in row:
            assert row['onset_sample'] == 8000
            clean_meta[key] = row
    records, groups, pooled, regenerated = [], {}, {}, 0
    for rec in frozen['recipes']:
        clean = cache[rec['sample_id']]
        noise = None if rec['noise_id'] is None else rec['noise_gain'] * cache[rec['noise_id']]
        mixture = clean if noise is None else clean + noise
        assert ah(mixture) == rec['float64_sha256']
        assert ah(mixture.astype(np.float32)) == rec['model_input_float32_sha256']
        regenerated += 1
        if noise is None:
            continue
        meta = clean_meta[rec['sample_id']]
        onset, duration = meta['onset_sample'], meta['resampled_samples']
        stop = onset + duration
        assert 0 < duration and stop <= len(clean)
        assert np.count_nonzero(clean[:onset]) == np.count_nonzero(clean[stop:]) == 0
        full_cp, full_np = power(clean), power(noise)
        supp_cp, supp_np = power(clean[onset:stop]), power(noise[onset:stop])
        full_snr = float(10 * np.log10(full_cp / full_np))
        supp_snr = float(10 * np.log10(supp_cp / supp_np))
        assert abs(full_snr - rec['snr_db']) < 1e-12
        offset = supp_snr - full_snr
        records.append([rec['condition'], rec['sample_id'], rec['noise_id'], duration,
                        full_cp, full_np, supp_cp, supp_np, full_snr, supp_snr, offset])
        groups.setdefault(rec['condition'], []).append(records[-1])
        pooled.setdefault(str(rec['snr_db']), []).append(records[-1])
    assert regenerated == 144 and len(records) == 128
    def summarize(rows):
        return {'n': len(rows), 'full_carrier_snr_db': stats([r[8] for r in rows]),
                'source_recording_support_snr_db': stats([r[9] for r in rows]),
                'support_minus_full_snr_db': stats([r[10] for r in rows])}
    after = sha(RECIPE)
    assert before == after
    assert all(sha(P / rel) == val for rel, val in array_hashes.items())
    result = {
        'status': 'passed_offline_only', 'recipe_sha256_before': before,
        'recipe_sha256_after': after, 'frozen_recipe_unchanged': before == after,
        'all_18_frozen_array_hashes_unchanged': True,
        'all_144_recipe_float64_and_float32_hashes_verified': True,
        'noisy_recipes_quantified': len(records), 'sample_rate': sr,
        'full_carrier_samples': 32000, 'full_carrier_seconds': 2.0,
        'support_definition': 'source_recording_support: [8000,8000+resampled_samples); includes the entire resampled source recording, without phoneme boundaries or clean activity selection',
        'snr_definition': '10*log10(mean(clean**2)/mean((noise_gain*fixed_noise)**2)), float64 powers on the same indicated window; noise is the additive recipe component, not a residual inferred from rounded mixture',
        'normalization': 'Clean full-carrier RMS=0.01; source support occupies only part of the two-second carrier. Full-carrier recipe SNR is not phoneme-local or clean-active SNR.',
        'limits': 'No consonant boundary or clean-active segmentation is available. Source recording support SNR is not true consonant SNR and cannot be equated to natural DNS mixture SNR without matching its measurement convention. No ASR, model, checkpoint, GPU, download, or training access.',
        'source_recording_supports': [{'sample_id': sid, 'start_sample': 8000,
           'end_sample_exclusive': 8000 + row['resampled_samples'],
           'duration_samples': row['resampled_samples'],
           'duration_seconds': row['resampled_samples'] / sr} for sid, row in clean_meta.items()],
        'support_duration_seconds_summary': stats([r['resampled_samples'] / sr for r in clean_meta.values()]),
        'by_condition': {key: summarize(rows) for key, rows in groups.items()},
        'pooled_by_nominal_full_carrier_snr_db': {key: summarize(rows) for key, rows in pooled.items()},
        'record_columns': ['condition', 'sample_id', 'noise_id', 'support_duration_samples',
            'full_clean_power', 'full_noise_power', 'support_clean_power', 'support_noise_power',
            'full_carrier_snr_db', 'source_recording_support_snr_db', 'support_minus_full_snr_db'],
        'records': records, 'audit_script_sha256': sha(Path(__file__).resolve())}
    encoded = json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n'
    assert len(encoded.encode()) + Path(__file__).stat().st_size < 50000
    OUT.write_text(encoded)
    print(json.dumps({'status': result['status'], 'support_duration_seconds': result['support_duration_seconds_summary'],
        'by_condition': result['by_condition'],
        'pooled_minus6_plus6': {s: result['pooled_by_nominal_full_carrier_snr_db'][s] for s in ['-6', '6']},
        'recipe_unchanged': True, 'script_plus_json_bytes': len(encoded.encode()) + Path(__file__).stat().st_size}, indent=2))

if __name__ == '__main__':
    main()
