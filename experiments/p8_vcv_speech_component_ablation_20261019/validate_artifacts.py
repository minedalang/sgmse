"""Read-only CPU validation; no model initialization or inference."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def ah(x):
    return hashlib.sha256(x.tobytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

def main():
    d = read(HERE/'metrics.json')
    cfg = read(HERE/'config.yaml')
    selected = read(HERE/'selection_frozen.json')['cases']
    assert d['status'] == 'completed' and d['native_gate_passed']
    assert (d['frontend_forwards'], d['backend_forwards'], d['recognizer_forwards']) == (0, 10, 5)
    assert len(d['cases']) == 10 and len(d['comparisons']) == 5
    assert d['runtime_flags'] == cfg['runtime_flags']
    assert all(sha(p) == digest for p, digest in d['sha256_locks'].items())
    assert (HERE/'outputs/COMPLETED.json').exists()
    assert not (HERE/'outputs/FAILED.json').exists()
    assert not (HERE/'outputs/RUNNING.json').exists()
    gates = read(HERE/'outputs/NATIVE_GATES_PASSED.json')
    assert len(gates) == 5 and all(r['exact_native_raw_and_pcm_match'] for r in gates)
    assert read(HERE/'outputs/audio_manifest.json') == d['cases']
    proofs = {p['case_id']: p for p in read(HERE/'outputs/source_proofs.json')}
    checks = []
    ctc_count = 0
    for case in selected:
        cid = case['case_id']
        native = next(r for r in d['cases'] if r['case_id'] == cid and not r['ablated'])
        ablated = next(r for r in d['cases'] if r['case_id'] == cid and r['ablated'])
        comparison = next(r for r in d['comparisons'] if r['case_id'] == cid)
        for key in ['normalized_anchor_sha256', 'anchor_raw_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256', 'initial_xT_sha256', 'regeneration_strength', 'seed', 'refscale', 'normalized_physical_noise_sha256']:
            assert native[key] == ablated[key], (cid, key)
        assert native['normalized_condition_sha256'] != ablated['normalized_condition_sha256']
        assert ablated['matched_anchor_initial_state_and_noise_geometry']
        p = proofs[cid]
        assert 0 < p['float32_speech_removal_error_max'] <= p['float32_error_bound']
        assert p['condition_active_change_rms'] > 0 and p['refscale'] == native['refscale']
        assert abs(ablated['condition_normalized_peak']-p['normalized_noise_peak']) <= np.finfo(np.float32).eps
        for r in [native, ablated]:
            assert r['nfe'] == r['trace_steps'] == 30
            x = np.load(HERE/r['raw_path'], allow_pickle=False)
            assert x.dtype == np.float32 and x.shape == (32000,) and np.isfinite(x).all()
            assert sha(HERE/r['raw_path']) == r['raw_sha256'] and ah(x) == r['float32_sha256']
            assert sha(HERE/r['wav_path']) == r['wav_sha256']
            wav, sr = sf.read(HERE/r['wav_path'], dtype='int16')
            assert sr == 16000 and np.max(np.abs(wav.astype(np.float64)/32768-x)) <= 1/32768
            encoded = io.BytesIO()
            sf.write(encoded, np.clip(x, -1, 1), 16000, subtype='PCM_16', format='WAV')
            assert hashlib.sha256(encoded.getvalue()).hexdigest() == r['wav_sha256']
            assert r['pcm_clipping_count'] == 0 and r['raw_peak'] < 1
            assert r['condition_normalized_peak'] <= 1 and not r['outside_native_norm_range']
            q = r['quiet_padding']
            end = q['source_end_sample_exclusive']
            segments = [x[:4800], x[20800:], np.concatenate([x[:8000], x[end:]])]
            for name, seg in zip(['pre_0_0p3', 'post_1p3_2p0', 'outside_true_source_support'], segments):
                assert q[name]['samples'] == len(seg)
                assert q[name]['rms'] == float(np.sqrt(np.mean(seg.astype(np.float64)**2)))
        prior = case['off_audio']
        assert native['raw_sha256'] == prior['raw_sha256'] and native['wav_sha256'] == prior['wav_sha256']
        assert np.array_equal(np.load(HERE/native['raw_path']), np.load(ROOT/case['source_experiment']/prior['raw_path']))
        assert native['ctc'] == case['off_score'] and native['ctc_reused']
        for score, base in [(case['pre_score'], ROOT/case['source_experiment']), (case['off_score'], ROOT/case['source_experiment']), (ablated['ctc'], HERE)]:
            path = base/score['array_path']
            assert sha(path) == score['array_sha256']
            a = np.load(path, allow_pickle=False)
            labels = a['candidate_labels'].tolist()
            logits = torch.from_numpy(a['logits'])
            logp = logits.log_softmax(-1)
            scores = -F.ctc_loss(logp[:, None, :].expand(-1, 24, -1).contiguous(), torch.from_numpy(a['candidate_token_ids']), torch.full((24,), len(logits), dtype=torch.long), torch.full((24,), 3, dtype=torch.long), blank=0, reduction='none').numpy()
            assert np.array_equal(scores, a['candidate_loglikelihoods'])
            assert np.array_equal(logp.exp().numpy(), a['probabilities'])
            assert dict(zip(labels, map(float, scores))) == score['all24_loglikelihoods']
            assert labels[int(scores.argmax())] == score['top1_prediction']
            ctc_count += 1
        target, rival = case['nominal_target'], case['preset_rival']
        for prefix, r in [('native', native), ('ablated', ablated)]:
            scores = r['ctc']['all24_loglikelihoods']
            assert comparison[prefix+'_target_vs_rival'] == scores[target]-scores[rival]
        assert comparison['ablated_prediction'] == ablated['ctc']['top1_prediction']
        assert comparison['target_vs_prelabel_degenerate'] == (target == case['pre_prediction'])
        pred = ablated['ctc']['top1_prediction']
        outcome = ('retained' if pred == target else 'lost') if case['transition_kind'] == 'repair' else ('reverted_to_nominal' if pred == target else 'remained_original_error' if pred == case['off_prediction'] else 'third_class')
        assert comparison['outcome'] == outcome
        diff = np.load(HERE/ablated['raw_path']).astype(np.float64)-np.load(HERE/native['raw_path']).astype(np.float64)
        checks.append({'case_id': cid, 'outcome': comparison['outcome'], 'normalized_noise_peak': p['normalized_noise_peak'], 'native_to_ablation_waveform_rms_difference': float(np.sqrt(np.mean(diff**2))), 'native_to_ablation_waveform_max_abs_difference': float(np.max(np.abs(diff))), 'ablated_minus_native_target_margin': comparison['ablated_target_vs_rival']-comparison['native_target_vs_rival'], 'all_checks_passed': True})
    size = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    assert size < cfg['artifact_cap_bytes'] and d['wall_seconds'] < cfg['budget_seconds']
    print(json.dumps({'status': 'passed', 'validation': 'CPU artifact/cache recomputation only; no model calls', 'immutable_locks_checked': len(d['sha256_locks']), 'audio_pairs_checked': 10, 'ctc_caches_exactly_recomputed': ctc_count, 'artifact_bytes_before_validation_record': size, 'cases': checks}, indent=2))

if __name__ == '__main__':
    main()
