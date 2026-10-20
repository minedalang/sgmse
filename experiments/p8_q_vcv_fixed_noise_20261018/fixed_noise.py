#!/usr/bin/env python3
"""Frozen fixed-noise VCV diagnostic; --prepare performs CPU audio preparation only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRIOR = REPO / 'experiments/p8_q_vcv_clean_calibration_20261018'
MANIFEST = PRIOR / 'retrieval/selected_manifest_frozen.json'
SCORER = PRIOR / 'calibrate_clean.py'
MODEL = REPO / 'experiments/p8_phoneme_probe_20261020/local_model'
NOISE = Path('/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/noise')
SNRS = [-12, -6, 0, 6]
CAP = 75_000_000
LIMITATION = ('Controlled recognizer diagnostic with nominal corpus labels. No human recoverability claim. '
              'Failure to rank the correct identity does not prove that the input contains no identity evidence. '
              'No population, independence, confidence interval or enhancement claims. Pooled conditions are descriptive; '
              'eligibility applies only to a single preset noise/SNR condition, never to oracle mixtures.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def ah(x):
    return hashlib.sha256(x.tobytes(order='C')).hexdigest()


def rms(x):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def budget():
    n = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file() and not p.is_symlink())
    require(n <= CAP, f'Artifact cap exceeded: {n} > {CAP}')
    return n


def prior():
    spec = importlib.util.spec_from_file_location('frozen_prior_clean', SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_config():
    import yaml
    c = yaml.safe_load((HERE / 'config.yaml').read_text())
    require(type(c['status']) is str and c['status'] == 'prepared', 'status must be strict string enum')
    for key in ('training', 'enhancement_inference', 'new_downloads'):
        require(type(c[key]) is bool and c[key] is False, f'{key} must be strict false boolean')
    require(c['snrs_db'] == SNRS and type(c['design_approved_by']) is str, 'Frozen configuration mismatch')
    return c


def recipe_wave(recipe, caches):
    import numpy as np
    clean = caches[recipe['sample_id']]
    x = clean if recipe['noise_id'] is None else clean + recipe['noise_gain'] * caches[recipe['noise_id']]
    require(x.dtype == np.float64 and x.shape == (32000,), 'Expected float64 two second signal')
    require(ah(x) == recipe['float64_sha256'], 'Regeneration hash mismatch')
    x32 = x.astype(np.float32)
    require(ah(x32) == recipe['model_input_float32_sha256'], 'Model input hash mismatch')
    return x32


def prepare():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    cfg = check_config()
    out = HERE / 'inputs'
    out.mkdir(exist_ok=False)
    dump(out / 'RUNNING.json', {'stage': 'CPU_preparation', 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        mod = prior()
        manifest = json.loads(MANIFEST.read_text())
        rows = [r for r in mod.validate(manifest) if r['is_primary']]
        frozen = {'schema_version': 1, 'experiment_id': HERE.name, 'source_input_group': 'controlled_corpus_train_VCV_plus_DNS_source_noise_not_official_test',
                  'source_manifest': str(MANIFEST), 'source_manifest_sha256': sha(MANIFEST),
                  'prior_scoring_module': str(SCORER), 'prior_scoring_module_sha256': sha(SCORER),
                  'runner_sha256': sha(__file__), 'config_sha256': sha(HERE / 'config.yaml'),
                  'design_approved_by': cfg['design_approved_by'], 'pairs': manifest['pairs'],
                  'vowel_map': mod.VOWELS, 'consonant_map': mod.CONSONANTS, 'binary_competitors': mod.COMPETITOR,
                  'model_path': str(MODEL), 'model_files_sha256': {str(p.relative_to(MODEL)): sha(p) for p in sorted(MODEL.rglob('*')) if p.is_file()},
                  'clean': [], 'noise': [], 'recipes': [], 'array_dtype': 'float64_little_endian', 'sample_rate': 16000}
        caches = {}
        for row in rows:
            require(sha(row['wav_path']) == row['sha256'], 'Source hash mismatch')
            w, sr = sf.read(row['wav_path'], dtype='float64', always_2d=True)
            require(sr == 25000 and w.shape[1] == 1 and np.isfinite(w).all(), 'Expected finite mono25k source')
            res = resample_poly(w[:, 0], 16, 25)
            require(0 < len(res) <= 24000, 'VCV exceeds fixed 1.5 second maximum; no tuning allowed')
            padded = np.zeros(32000, dtype=np.float64)
            padded[8000:8000 + len(res)] = res
            base_rms = rms(padded)
            require(base_rms > 0, 'Silent clean')
            gain = .01 / base_rms
            clean = padded * gain
            require(abs(rms(clean) - .01) < 1e-15, 'Clean RMS mismatch')
            require(np.max(np.abs(clean)) < 1, 'Clean peak exceeds one')
            caches[row['sample_id']] = clean
            path = out / (row['sample_id'] + '.npz')
            np.savez_compressed(path, waveform=clean, sample_rate=np.array(16000))
            frozen['clean'].append(dict(row, source_samples=len(w), resampled_samples=len(res), onset_sample=8000,
                                        unnormalized_padded_rms=base_rms, derived_gain=gain, full_2s_rms=rms(clean),
                                        array_path=str(path.relative_to(HERE)), array_sha256=sha(path), float64_sha256=ah(clean)))
        for noise_id, file_id in [('N1', 5243), ('N2', 6339)]:
            path = NOISE / f'noise_fileid_{file_id}.wav'
            w, sr = sf.read(path, dtype='float64', always_2d=True)
            require(sr == 16000 and w.shape == (480000, 1) and np.isfinite(w).all(), 'Noise must be exact mono16k30s source')
            clip = w[80000:112000, 0].copy()
            raw_rms = rms(clip)
            require(raw_rms > 0, 'Silent noise')
            gain = 1. / raw_rms
            unit = clip * gain
            require(abs(rms(unit) - 1.) < 1e-14, 'Unit RMS mismatch')
            caches[noise_id] = unit
            artifact = out / (noise_id + '.npz')
            np.savez_compressed(artifact, waveform=unit, sample_rate=np.array(16000))
            frozen['noise'].append({'noise_id': noise_id, 'source_path': str(path), 'source_sha256': sha(path),
                                    'source_samples': len(w), 'source_sample_rate': sr, 'source_decoded_float64_sha256': ah(w),
                                    'start_sample': 80000, 'end_sample_exclusive': 112000, 'selected_samples': len(clip),
                                    'selected_raw_rms': raw_rms, 'derived_gain': gain, 'full_2s_rms': rms(unit),
                                    'array_path': str(artifact.relative_to(HERE)), 'array_sha256': sha(artifact), 'float64_sha256': ah(unit)})
        for noise_id, snr in [(None, None)] + [(n, s) for n in ('N1', 'N2') for s in SNRS]:
            condition = 'padded_clean' if noise_id is None else f'{noise_id}_snr{snr:+d}'
            for row in rows:
                clean = caches[row['sample_id']]
                gain = None if snr is None else float(.01 * 10 ** (-snr / 20))
                x = clean if snr is None else clean + gain * caches[noise_id]
                actual = None if snr is None else float(20 * np.log10(rms(clean) / rms(gain * caches[noise_id])))
                require(snr is None or abs(actual - snr) < 1e-12, 'Actual SNR mismatch')
                peak = float(np.max(np.abs(x)))
                require(peak < 1, 'Peak exceeds one; no clipping or adaptation permitted')
                frozen['recipes'].append({'condition': condition, 'sample_id': row['sample_id'], 'noise_id': noise_id,
                                          'snr_db': snr, 'noise_gain': gain, 'actual_snr_db_float64': actual,
                                          'peak_absolute': peak, 'float64_sha256': ah(x), 'model_input_float32_sha256': ah(x.astype(np.float32))})
        require(len(frozen['recipes']) == 144, 'Expected 144 recipes')
        dump(out / 'recipe_frozen.json', frozen)
        dump(out / 'COMPLETED.json', {'status': 'CPU_prepared_no_inference', 'recipe_sha256': sha(out / 'recipe_frozen.json'), 'artifact_bytes': budget()})
        (out / 'RUNNING.json').unlink()
        dump(HERE / 'metrics.json', {'status': 'prepared_no_inference', 'experiment_id': HERE.name, 'recipe_sha256': sha(out / 'recipe_frozen.json'), 'recognizer_forwards': 0, 'recipes': 144})
    except BaseException as exc:
        dump(out / 'FAILED.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        (out / 'RUNNING.json').unlink(missing_ok=True)
        dump(HERE / 'metrics.json', {'status': 'preparation_failed', 'error': str(exc)})
        raise


def summarize(rows, pairs):
    n = len(rows)
    by_id = {r['sample_id']: r for r in rows}
    # Called on one condition. Pooling below sums condition summaries, preserving pair units.
    pair_rows = []
    for p in pairs:
        members = [by_id[Path(b).stem] for b in p['members']]
        pair_rows.append({'pair_id': p['pair_id'], 'contrast': p['contrast'],
                          'both_correct_24way': all(r['top1_correct'] for r in members),
                          'both_correct_binary': all(r['binary_correct'] for r in members),
                          'same_prediction_collapse_24way': members[0]['top1_prediction'] is not None and members[0]['top1_prediction'] == members[1]['top1_prediction'],
                          'same_prediction_collapse_binary': members[0]['binary_prediction'] is not None and members[0]['binary_prediction'] == members[1]['binary_prediction']})
    confusion = Counter((r['consonant'], r['top1_prediction'] or 'tie') for r in rows)
    binary_confusion = Counter((r['consonant'], r['binary_prediction'] or 'tie') for r in rows)
    contrast = {c: sum(r['top1_correct'] for r in rows if r['contrast'] == c) for c in ('k/g', 't/d')}
    both = sum(p['both_correct_binary'] for p in pair_rows)
    return {'n': n, 'top1_correct': sum(r['top1_correct'] for r in rows), 'binary_correct': sum(r['binary_correct'] for r in rows),
            'contrast_24way_correct': contrast, 'pairs': pair_rows, 'n_pairs': len(pair_rows),
            'both_correct_pairs_24way': sum(p['both_correct_24way'] for p in pair_rows), 'both_correct_pairs_binary': both,
            'both_correct_pairs_24way_fraction': sum(p['both_correct_24way'] for p in pair_rows) / len(pair_rows),
            'both_correct_pairs_binary_fraction': both / len(pair_rows),
            'same_prediction_collapse_24way': sum(p['same_prediction_collapse_24way'] for p in pair_rows),
            'same_prediction_collapse_binary': sum(p['same_prediction_collapse_binary'] for p in pair_rows),
            'confusion_24way': {a: dict((b, v) for (aa, b), v in confusion.items() if aa == a) for a in ('k', 'g', 't', 'd')},
            'confusion_binary': {a: dict((b, v) for (aa, b), v in binary_confusion.items() if aa == a) for a in ('k', 'g', 't', 'd')},
            'binary_margin': {'minimum': min(r['binary_loglikelihood_margin'] for r in rows), 'maximum': max(r['binary_loglikelihood_margin'] for r in rows),
                              'mean': sum(r['binary_loglikelihood_margin'] for r in rows) / n},
            'top1_margin_mean': sum(r['top1_loglikelihood_margin'] for r in rows) / n}


def eligible(s, pair_gate=True):
    return s['top1_correct'] >= 12 and s['binary_correct'] >= 14 and all(v >= 6 for v in s['contrast_24way_correct'].values()) and (not pair_gate or s['both_correct_pairs_binary'] >= 6)


def pooled(summaries):
    out = {k: sum(s[k] for s in summaries) for k in ('n', 'top1_correct', 'binary_correct', 'n_pairs', 'both_correct_pairs_24way', 'both_correct_pairs_binary', 'same_prediction_collapse_24way', 'same_prediction_collapse_binary')}
    out['both_correct_pairs_24way_fraction'] = out['both_correct_pairs_24way'] / out['n_pairs']
    out['both_correct_pairs_binary_fraction'] = out['both_correct_pairs_binary'] / out['n_pairs']
    out['contrast_24way_correct'] = {c: sum(s['contrast_24way_correct'][c] for s in summaries) for c in ('k/g', 't/d')}
    for metric in ('confusion_24way', 'confusion_binary'):
        out[metric] = {}
        for s in summaries:
            for truth, counts in s[metric].items():
                dest = out[metric].setdefault(truth, {})
                for pred, count in counts.items():
                    dest[pred] = dest.get(pred, 0) + count
    out['binary_margin'] = {'minimum': min(s['binary_margin']['minimum'] for s in summaries), 'maximum': max(s['binary_margin']['maximum'] for s in summaries), 'mean': sum(s['binary_margin']['mean'] * s['n'] for s in summaries) / out['n']}
    return out


def run(device):
    cfg = check_config()
    output = HERE / 'recognition_outputs'
    output.mkdir(exist_ok=False)  # Refuse completed and partial runs alike.
    (output / 'arrays').mkdir()
    dump(output / 'RUNNING.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv})
    start = time.time()
    results = []
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import numpy as np
        import torch
        import torch.nn.functional as F
        import transformers
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        frozen_path = HERE / 'inputs/recipe_frozen.json'
        recipe_hash = sha(frozen_path)
        frozen = json.loads(frozen_path.read_text())
        require(sha(__file__) == frozen['runner_sha256'], 'Runner changed after recipe freeze')
        require(sha(HERE / 'config.yaml') == frozen['config_sha256'], 'Config changed')
        require(sha(MANIFEST) == frozen['source_manifest_sha256'] and sha(SCORER) == frozen['prior_scoring_module_sha256'], 'Prior artifacts changed')
        require({str(p.relative_to(MODEL)): sha(p) for p in sorted(MODEL.rglob('*')) if p.is_file()} == frozen['model_files_sha256'], 'Model files changed')
        for r in frozen['clean']:
            require(sha(r['wav_path']) == r['sha256'], 'Clean source changed')
        for r in frozen['noise']:
            require(sha(r['source_path']) == r['source_sha256'], 'Noise source changed')
        caches = {}
        for r in frozen['clean'] + frozen['noise']:
            p = HERE / r['array_path']
            require(sha(p) == r['array_sha256'], 'Cached array changed')
            x = np.load(p, allow_pickle=False)['waveform']
            require(ah(x) == r['float64_sha256'], 'Cached waveform mismatch')
            caches[r.get('sample_id', r.get('noise_id'))] = x
        mod = prior()
        labels = list(mod.CONSONANTS)
        vocab = json.loads((MODEL / 'vocab.json').read_text())
        require(vocab['<pad>'] == 0, 'Wrong blank')
        require(all(t in vocab for t in list(mod.VOWELS.values()) + list(mod.CONSONANTS.values())), 'Missing fixed token')
        id2token = {v: k for k, v in vocab.items()}
        torch.manual_seed(0)
        np.random.seed(0)
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(MODEL), local_files_only=True).to(device).eval()
        require(model.config.pad_token_id == 0, 'Model blank mismatch')
        source = {r['sample_id']: r for r in frozen['clean']}
        conditions = list(dict.fromkeys(r['condition'] for r in frozen['recipes']))
        summaries = {}
        gate = None
        for condition in conditions:
            if condition != 'padded_clean' and not gate:
                break
            condition_rows = []
            for rec in [r for r in frozen['recipes'] if r['condition'] == condition]:
                row = source[rec['sample_id']]
                x = recipe_wave(rec, caches)
                inp = {k: v.to(device) for k, v in extractor(x, sampling_rate=16000, return_tensors='pt', padding=False).items()}
                left, right = mod.VOWELS[row['vowel_left']], mod.VOWELS[row['vowel_right']]
                targets = torch.tensor([[vocab[left], vocab[mod.CONSONANTS[c]], vocab[right]] for c in labels], dtype=torch.long)
                with torch.inference_mode():
                    logits = model(**inp).logits[0].float().cpu()
                    require(logits.shape[1] == len(vocab), 'Wrong vocabulary shape')
                    logp = logits.log_softmax(-1)
                    scores = -F.ctc_loss(logp[:, None, :].expand(-1, 24, -1).contiguous(), targets,
                                         torch.full((24,), len(logits), dtype=torch.long), torch.full((24,), 3, dtype=torch.long),
                                         blank=0, reduction='none', zero_infinity=False).numpy()
                    probs = logp.exp().numpy()
                require(np.isfinite(scores).all(), 'Nonfinite scores')
                # Rank every candidate before consulting the nominal target.
                ranking = sorted(zip(labels, map(float, scores)), key=lambda z: (-z[1], labels.index(z[0])))
                winners = [c for c, s in ranking if s == ranking[0][1]]
                pred = winners[0] if len(winners) == 1 else None
                truth = row['consonant']
                rival = mod.COMPETITOR[truth]
                margin = float(scores[labels.index(truth)] - scores[labels.index(rival)])
                binary_pred = truth if margin > 0 else rival if margin < 0 else None
                greedy = logits.argmax(-1).tolist()
                runs = []
                i = 0
                while i < len(greedy):
                    j = i + 1
                    while j < len(greedy) and greedy[j] == greedy[i]:
                        j += 1
                    if greedy[i] != 0:
                        runs.append({'token': id2token[greedy[i]], 'token_id': greedy[i], 'start_frame': i, 'end_frame_exclusive': j, 'mean_posterior': float(probs[i:j, greedy[i]].mean())})
                    i = j
                artifact = output / 'arrays' / f"{condition}__{rec['sample_id']}.npz"
                np.savez_compressed(artifact, logits=logits.numpy().astype(np.float32), probabilities=probs.astype(np.float32), candidate_loglikelihoods=scores.astype(np.float32), candidate_labels=np.asarray(labels), candidate_token_ids=targets.numpy(), sample_rate=np.array(16000))
                result = dict(rec, consonant=truth, speaker=row['speaker'], contrast='k/g' if truth in ('k', 'g') else 't/d',
                              all24_loglikelihoods=dict(zip(labels, map(float, scores))), all24_ranking=ranking,
                              all24_candidate_ipa_tokens={c: [left, mod.CONSONANTS[c], right] for c in labels},
                              top1_prediction=pred, top1_tied_labels=winners if pred is None else [], top1_correct=pred == truth,
                              top1_loglikelihood_margin=ranking[0][1] - ranking[1][1], binary_prediction=binary_pred,
                              binary_competitor=rival, binary_loglikelihood_margin=margin, binary_correct=margin > 0, binary_tie=margin == 0,
                              greedy_ipa=' '.join(r['token'] for r in runs), greedy_runs_descriptive=runs,
                              ctc_frames=len(logits), array_path=str(artifact.relative_to(HERE)), array_sha256=sha(artifact))
                results.append(result)
                condition_rows.append(result)
                dump(output / 'per_sample.json', results)
                budget()
                print(f"{condition}/{rec['sample_id']}: scored 24 sequences", flush=True)
            summaries[condition] = summarize(condition_rows, frozen['pairs'])
            if condition == 'padded_clean':
                gate = eligible(summaries[condition], pair_gate=False)
                dump(output / 'padded_clean_gate.json', {'passed': gate, 'summary': summaries[condition], 'failure_action': 'STOP_noisy_without_tuning_or_removing_samples'})
        noisy = {k: v for k, v in summaries.items() if k != 'padded_clean'}
        if gate:
            require(len(results) == 144 and len(noisy) == 8, 'All prespecified conditions must complete')
        stability = {}
        for snr in SNRS if gate else []:
            a = {r['sample_id']: r for r in results if r['condition'] == f'N1_snr{snr:+d}'}
            b = {r['sample_id']: r for r in results if r['condition'] == f'N2_snr{snr:+d}'}
            items = []
            for sid in a:
                r, s = a[sid], b[sid]
                items.append({'sample_id': sid, 'N1_prediction': r['top1_prediction'], 'N2_prediction': s['top1_prediction'],
                              'same_24way_prediction': r['top1_prediction'] is not None and r['top1_prediction'] == s['top1_prediction'],
                              'both_24way_correct': r['top1_correct'] and s['top1_correct'],
                              'stable_incorrect_24way': r['top1_prediction'] is not None and r['top1_prediction'] == s['top1_prediction'] and not r['top1_correct'],
                              'same_binary_prediction': r['binary_prediction'] is not None and r['binary_prediction'] == s['binary_prediction'],
                              'both_binary_correct': r['binary_correct'] and s['binary_correct'],
                              'stable_incorrect_binary': r['binary_prediction'] is not None and r['binary_prediction'] == s['binary_prediction'] and not r['binary_correct'],
                              'binary_margin_N2_minus_N1': s['binary_loglikelihood_margin'] - r['binary_loglikelihood_margin']})
            stability[str(snr)] = {'n': 16, 'samples': items, **{key: sum(r[key] for r in items) for key in ('same_24way_prediction', 'both_24way_correct', 'stable_incorrect_24way', 'same_binary_prediction', 'both_binary_correct', 'stable_incorrect_binary')}}
        metrics = {'status': 'completed' if gate else 'stopped_padded_clean_gate_failed', 'experiment_id': HERE.name,
                   'padded_clean_gate_passed': gate, 'recognizer_forwards': len(results), 'conditions': summaries,
                   'eligible_for_later_pilot_conditions': [k for k, s in noisy.items() if eligible(s)],
                   'by_snr_pooled_noise': {str(s): pooled([noisy[f'{n}_snr{s:+d}'] for n in ('N1', 'N2')]) for s in SNRS} if gate else {},
                   'by_noise_pooled_snr': {n: pooled([noisy[f'{n}_snr{s:+d}'] for s in SNRS]) for n in ('N1', 'N2')} if gate else {},
                   'pooled_all_noisy_descriptive': pooled(list(noisy.values())) if gate else None,
                   'noise_identity_stability': stability, 'recipe_sha256': recipe_hash, 'runtime_seconds': time.time() - start,
                   'environment': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': device},
                   'limitations': LIMITATION, 'artifact_bytes': budget()}
        require(sha(frozen_path) == recipe_hash and sha(__file__) == frozen['runner_sha256'], 'Frozen artifacts changed during run')
        for r in frozen['clean']:
            require(sha(r['wav_path']) == r['sha256'], 'Source changed during run')
        for r in frozen['noise']:
            require(sha(r['source_path']) == r['source_sha256'], 'Noise source changed during run')
        require({str(p.relative_to(MODEL)): sha(p) for p in sorted(MODEL.rglob('*')) if p.is_file()} == frozen['model_files_sha256'], 'Checkpoint changed during run')
        dump(output / 'metrics.json', metrics)
        dump(HERE / 'metrics.json', metrics)
        report = [f'# Fixed-noise VCV diagnostic\n\nStatus: {metrics["status"]}. Recognizer forwards: {len(results)}.\n',
                  '| Condition | 24-way | Binary | Both correct binary pairs | Eligible later |', '|---|---:|---:|---:|---|']
        for name, summary in summaries.items():
            report.append(f'| {name} | {summary["top1_correct"]}/16 | {summary["binary_correct"]}/16 | {summary["both_correct_pairs_binary"]}/8 | {name in metrics["eligible_for_later_pilot_conditions"]} |')
        report.extend(['', LIMITATION, '', 'Frozen recipe: inputs/recipe_frozen.json. Full scores and descriptive greedy runs: recognition_outputs/per_sample.json. Float32 logits/probabilities: recognition_outputs/arrays/. Sources and model hashes verified unchanged. No enhancement or training.'])
        (HERE / 'experiment_report.md').write_text('\n'.join(report) + '\n')
        dump(output / 'COMPLETED.json', {'status': metrics['status'], 'recognizer_forwards': len(results), 'metrics_sha256': sha(output / 'metrics.json')})
        (output / 'RUNNING.json').unlink()
        budget()
    except BaseException as exc:
        failure = {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc(), 'recognizer_forwards_recorded': len(results)}
        dump(output / 'FAILED.json', failure)
        dump(HERE / 'metrics.json', failure)
        (HERE / 'experiment_report.md').write_text(f'# Fixed-noise VCV diagnostic\n\nStatus: failed. {exc}\n\nSee recognition_outputs/FAILED.json. Do not retry unchanged commands or overwrite partial artifacts.\n')
        (output / 'RUNNING.json').unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    if args.prepare:
        prepare()
    else:
        run(args.device)


if __name__ == '__main__':
    main()
