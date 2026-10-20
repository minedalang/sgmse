#!/usr/bin/env python3
"""Frozen clean VCV calibration. Inference only; never changes corpus or model files."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VOWELS = {'a': 'æ', 'i': 'iː', 'u': 'uː'}
CONSONANTS = dict(zip(
    'b d g p t k s sh f v dh th ch z zh h dj m n ng w r y l'.split(),
    'b d ɡ p t k s ʃ f v ð θ tʃ z ʒ h dʒ m n ŋ w ɹ j l'.split()))
COMPETITOR = {'k': 'g', 'g': 'k', 't': 'd', 'd': 't'}
LIMIT = 25_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def budget():
    # Local-model weights live outside this experiment; do not follow directory symlinks.
    size = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file() and not p.is_symlink())
    if size > LIMIT:
        raise RuntimeError(f'Experiment artifact budget exceeded: {size} > {LIMIT} bytes')
    return size


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(manifest):
    rows = manifest['files']
    require(16 <= len(rows) <= 29, 'Expected 16 primary plus at most 13 supplements')
    ids = [r['sample_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'Duplicate sample IDs')
    require(all(re.fullmatch(r'[A-Za-z0-9_.-]+', s) for s in ids), 'Unsafe sample IDs')
    primary = [r for r in rows if r['is_primary']]
    require(len(primary) == 16, 'Exactly 16 unique primary clips required')
    require(len({r['speaker'] for r in primary}) == 4, 'Exactly four primary speakers required')
    require(Counter(r['consonant'] for r in primary) == Counter(k=4, g=4, t=4, d=4), 'Unbalanced primary identities')
    require(len(manifest['pairs']) == 8, 'Exactly eight primary contrast pairs required')
    by_base = {r['basename']: r for r in rows}
    counts = Counter()
    membership = Counter()
    for pair in manifest['pairs']:
        require(pair['contrast'] in ('k/g', 't/d'), 'Unexpected contrast')
        members = [by_base[x] for x in pair['members']]
        require(len(members) == 2 and all(r['is_primary'] for r in members), 'Pair must have two primary members')
        require({r['consonant'] for r in members} == set(pair['contrast'].split('/')), 'Wrong pair labels')
        require(len({(r['speaker'], r['vowel_left'], r['vowel_right'], r['stress']) for r in members}) == 1,
                'Pair speaker, vowels and stress must match')
        require(all(r['speaker'] == pair['speaker'] and pair['pair_id'] in r['primary_pair_ids'] for r in members), 'Pair metadata mismatch')
        counts[(pair['speaker'], pair['contrast'])] += 1
        membership.update(r['sample_id'] for r in members)
    require(len(counts) == 8 and all(v == 1 for v in counts.values()), 'One pair per speaker/contrast required')
    require(membership == Counter({r['sample_id']: 1 for r in primary}), 'Primary clips must belong to one pair each')
    for row in rows:
        require(row['consonant'] in CONSONANTS, 'Unknown nominal corpus consonant')
        require(row['vowel_left'] in VOWELS and row['vowel_right'] in VOWELS, 'Unknown corpus vowel')
        require(row['vowel_left'] == row['vowel_right'], 'Expected identical VCV context vowels')
        require(row['role'] == ('primary' if row['is_primary'] else 'supplementary_manual'), 'Role mismatch')
        if not row['is_primary']:
            require(row['consonant'] in COMPETITOR and row.get('boundary_source') == 'manual_mlf', 'Supplements must be manual k/g/t/d clips')
    return rows


def summarize(rows):
    n = len(rows)
    return {'n': n, 'top1_correct': sum(r['top1_correct'] for r in rows),
            'top1_accuracy': sum(r['top1_correct'] for r in rows) / n if n else None,
            'binary_correct': sum(r['binary_correct'] for r in rows),
            'binary_accuracy': sum(r['binary_correct'] for r in rows) / n if n else None,
            'mean_binary_loglikelihood_margin': sum(r['binary_loglikelihood_margin'] for r in rows) / n if n else None}


def run(args, output):
    # Imports are intentionally delayed; syntax checking never initializes a model.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import scipy
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    start = time.time()
    budget()
    manifest_hash = sha(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    samples = validate(manifest)
    protocol = HERE / 'calibration_protocol.yaml'
    provenance = {'manifest_path': str(args.manifest.resolve()), 'manifest_sha256': manifest_hash,
                  'protocol_sha256': sha(protocol), 'script_sha256': sha(__file__),
                  'design_approved_by': args.design_approved_by,
                  'model_path': str(args.model.resolve()),
                  'model_files_sha256': {str(p.relative_to(args.model)): sha(p) for p in sorted(args.model.rglob('*')) if p.is_file()},
                  'source_audio': []}
    for row in samples:
        path = Path(row['wav_path'])
        require(path.is_absolute() and path.is_file(), 'Manifest WAV path must be absolute and present')
        digest = sha(path)
        require(row.get('sha256') == digest, f'Missing or mismatched frozen WAV hash: {path}')
        provenance['source_audio'].append({'sample_id': row['sample_id'], 'path': str(path), 'sha256': digest})
    write_json(output / 'frozen_inputs.json', provenance)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'Requested CUDA unavailable')
    vocab = json.loads((args.model / 'vocab.json').read_text())
    require(vocab['<pad>'] == 0, 'Preset CTC blank must be zero')
    require(all(x in vocab for x in list(VOWELS.values()) + list(CONSONANTS.values())), 'Preset IPA token absent')
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(args.model), local_files_only=True).to(device).eval()
    require(model.config.pad_token_id == 0, 'Model blank mismatch')
    id2token = {v: k for k, v in vocab.items()}
    labels = list(CONSONANTS)
    result_rows = []
    for row, source in zip(samples, provenance['source_audio']):
        wav, sr = sf.read(row['wav_path'], dtype='float32', always_2d=True)
        require(sr == 25000 and wav.shape[1] == 1, 'Expected original mono 25 kHz corpus WAV')
        require(np.isfinite(wav).all() and len(wav) > 0, 'Invalid waveform')
        derived = resample_poly(wav[:, 0], 16, 25).astype(np.float32)
        inputs = extractor(derived, sampling_rate=16000, return_tensors='pt', padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        left, right = VOWELS[row['vowel_left']], VOWELS[row['vowel_right']]
        targets = torch.tensor([[vocab[left], vocab[CONSONANTS[c]], vocab[right]] for c in labels], dtype=torch.long)
        with torch.inference_mode():
            logits = model(**inputs).logits[0].float().cpu()
            require(logits.shape[1] == len(vocab), 'Unexpected model vocabulary dimension')
            log_probs = logits.log_softmax(-1)
            nframes = logits.shape[0]
            # Each of 24 batch entries scores one exact three-token VCV sequence.
            losses = F.ctc_loss(log_probs[:, None, :].expand(-1, 24, -1).contiguous(), targets,
                                torch.full((24,), nframes, dtype=torch.long), torch.full((24,), 3, dtype=torch.long),
                                blank=0, reduction='none', zero_infinity=False)
            scores = -losses.numpy()
            probs = log_probs.exp().numpy()
        require(np.isfinite(scores).all(), 'Nonfinite CTC sequence likelihood')
        artifact = output / 'arrays' / (row['sample_id'] + '.npz')
        np.savez_compressed(artifact, logits=logits.numpy(), probabilities=probs,
                            candidate_loglikelihoods=scores, candidate_labels=np.asarray(labels),
                            candidate_token_ids=targets.numpy(), sample_rate=np.asarray(16000))
        try:
            budget()
        except Exception:
            artifact.unlink()
            raise
        greedy = logits.argmax(-1).tolist()
        runs = []
        i = 0
        while i < len(greedy):
            j = i + 1
            while j < len(greedy) and greedy[j] == greedy[i]:
                j += 1
            if greedy[i] != 0:
                runs.append({'token': id2token[greedy[i]], 'token_id': greedy[i], 'start_frame': i,
                             'end_frame_exclusive': j, 'mean_posterior': float(probs[i:j, greedy[i]].mean())})
            i = j
        target = row['consonant']
        rival = COMPETITOR[target]
        margin = float(scores[labels.index(target)] - scores[labels.index(rival)])
        best = float(scores.max())
        winners = [c for c, s in zip(labels, scores) if float(s) == best]
        pred = winners[0] if len(winners) == 1 else None
        record = dict(row)
        record.update({'source_sha256': source['sha256'], 'derived_16k_float32_sha256': hashlib.sha256(derived.tobytes()).hexdigest(),
                       'source_samples': len(wav), 'derived_samples': len(derived), 'ctc_frames': nframes,
                       'all24_loglikelihoods': {c: float(s) for c, s in zip(labels, scores)},
                       'all24_candidate_ipa_tokens': {c: [left, CONSONANTS[c], right] for c in labels},
                       'top1_prediction': pred, 'top1_tied_labels': winners if len(winners) > 1 else [],
                       'top1_correct': pred == target, 'binary_competitor': rival,
                       'binary_loglikelihood_margin': margin, 'binary_correct': margin > 0,
                       'binary_tie': margin == 0, 'greedy_ipa': ' '.join(r['token'] for r in runs),
                       'greedy_runs_descriptive': runs, 'array_path': str(artifact.relative_to(HERE)),
                       'array_sha256': sha(artifact)})
        result_rows.append(record)
        write_json(output / 'per_sample.json', result_rows)
        budget()
        require(sha(row['wav_path']) == source['sha256'], 'Original WAV changed during inference')
        print(f"{row['sample_id']}: scored 24 fixed sequences", flush=True)
    primary = [r for r in result_rows if r['is_primary']]
    grouped = defaultdict(list)
    for r in primary:
        contrast = 'k/g' if r['consonant'] in ('k', 'g') else 't/d'
        grouped[f"speaker={r['speaker']}/contrast={contrast}"].append(r)
        grouped[f'speaker={r["speaker"]}'].append(r)
        grouped[f'contrast={contrast}'].append(r)
    summary = summarize(primary)
    gate = summary['top1_correct'] >= 12 and summary['binary_correct'] >= 14 and all(
        summarize(grouped[f'contrast={c}'])['top1_correct'] >= 6 for c in ('k/g', 't/d'))
    confusion = {truth: {pred: 0 for pred in labels + ['tie']} for truth in labels}
    for r in primary:
        confusion[r['consonant']][r['top1_prediction'] or 'tie'] += 1
    require(sha(args.manifest) == manifest_hash, 'Frozen manifest changed during run')
    require(sha(protocol) == provenance['protocol_sha256'], 'Protocol changed during run')
    require(sha(__file__) == provenance['script_sha256'], 'Calibration script changed during run')
    metrics = {'status': 'completed', 'experiment_id': HERE.name, 'gate_passed': gate,
               'next_stage': 'eligible_for_research_lead_review' if gate else 'STOP_NOISY_AND_ENHANCEMENT_STAGE',
               'primary': summary, 'primary_groups': {k: summarize(v) for k, v in sorted(grouped.items())},
               'primary_confusion_matrix': confusion, 'confusion_labels': labels,
               'supplementary_descriptive_only': summarize([r for r in result_rows if not r['is_primary']]),
               'gate_thresholds': {'primary_24way': 12, 'primary_binary': 14, 'each_contrast_24way': 6},
               'runtime_seconds': time.time() - start, 'artifact_bytes_at_completion': budget(),
               'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                               'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                               'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                               'numpy': np.__version__, 'scipy': scipy.__version__, 'soundfile': sf.__version__,
                               'transformers': transformers.__version__, 'seed': 0},
               'limitations': 'Controlled clean corpus diagnostic calibration only; no human recoverability, noisy discrimination, generalization, CI, or independence claims.',
               'provenance': provenance}
    write_json(output / 'metrics.json', metrics)
    budget()
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=HERE / 'retrieval/selected_manifest_frozen.json')
    p.add_argument('--model', type=Path, default=REPO / 'experiments/p8_phoneme_probe_20261020/local_model')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--design-approved-by', required=True, help='Research Lead approval reference, recorded before inference')
    args = p.parse_args()
    output = HERE / 'calibration_outputs'
    # Atomic exclusive creation prevents accidental reruns or partial-output overwrite.
    output.mkdir(exist_ok=False)
    (output / 'arrays').mkdir()
    write_json(output / 'RUNNING.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv})
    try:
        metrics = run(args, output)
        write_json(output / 'COMPLETED.json', {'completed_utc': datetime.now(timezone.utc).isoformat(), 'gate_passed': metrics['gate_passed']})
        (output / 'RUNNING.json').unlink()
    except BaseException as exc:
        write_json(output / 'FAILED.json', {'failed_utc': datetime.now(timezone.utc).isoformat(), 'error': str(exc), 'traceback': traceback.format_exc()})
        (output / 'RUNNING.json').unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    main()
