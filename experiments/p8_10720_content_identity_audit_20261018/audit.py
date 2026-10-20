#!/usr/bin/env python3
"""Offline audit of cached CTC IDs; never loads a model or audio."""
import csv
import hashlib
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
GROUP = '03_fileid_10720_14.50-15.00s'
CONDITIONS = ['clean', 'noisy', 'y_pre', 'off', 'router']
HOP = 0.02
MAX_SHIFT_MS = 100.0
EPS_MS = 1e-6
# Explicit IPA labels, including all consonants observed in this clip's outputs.
# Tokens outside this mapping are unclassified for the consonant-only analysis.
CONSONANTS = {
    'p': 'voiceless bilabial stop', 'b': 'voiced bilabial stop',
    't': 'voiceless alveolar stop', 'd': 'voiced alveolar stop',
    'k': 'voiceless velar stop', 'ɡ': 'voiced velar stop',
    'm': 'bilabial nasal', 'n': 'alveolar nasal', 'ŋ': 'velar nasal',
    'f': 'voiceless labiodental fricative', 'v': 'voiced labiodental fricative',
    'θ': 'voiceless dental fricative', 'ð': 'voiced dental fricative',
    's': 'voiceless alveolar fricative', 'z': 'voiced alveolar fricative',
    'ʃ': 'voiceless postalveolar fricative', 'ʒ': 'voiced postalveolar fricative',
    'h': 'voiceless glottal fricative', 'tʃ': 'voiceless postalveolar affricate',
    'dʒ': 'voiced postalveolar affricate', 'l': 'alveolar lateral approximant',
    'ɹ': 'alveolar approximant', 'r': 'alveolar trill', 'ɾ': 'alveolar tap',
    'w': 'labial velar approximant', 'j': 'palatal approximant',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wav_levels(path):
    """Read RIFF PCM/float samples without normalization or audio inference."""
    raw = path.read_bytes()
    assert raw[:4] == b'RIFF' and raw[8:12] == b'WAVE'
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        kind, size = raw[pos:pos + 4], struct.unpack_from('<I', raw, pos + 4)[0]
        payload = raw[pos + 8:pos + 8 + size]
        if kind == b'fmt ':
            fmt = payload
        if kind == b'data':
            data = payload
        pos += 8 + size + size % 2
    assert fmt is not None and data is not None
    encoding, channels, sr, _, _, bits = struct.unpack_from('<HHIIHH', fmt)
    if encoding == 65534:
        encoding = struct.unpack_from('<H', fmt, 24)[0]
    if encoding == 3 and bits in (32, 64):
        x = np.frombuffer(data, dtype='<f4' if bits == 32 else '<f8').astype(np.float64)
    elif encoding == 1 and bits in (16, 32):
        x = np.frombuffer(data, dtype='<i2' if bits == 16 else '<i4').astype(np.float64) / (2 ** (bits - 1))
    elif encoding == 1 and bits == 24:
        b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = ((packed ^ 0x800000) - 0x800000).astype(np.float64) / 8388608
    else:
        raise ValueError(f'Unsupported WAV encoding {encoding}, bits {bits}')
    assert len(x) and np.isfinite(x).all()
    rms, peak = float(np.sqrt(np.mean(x * x))), float(np.max(abs(x)))
    return dict(path=str(path), sha256=sha(path), sample_rate=sr, channels=channels,
                bits_per_sample=bits, encoding=encoding, duration_s=len(x) / channels / sr,
                rms=rms, rms_dbfs=float(20 * np.log10(rms)) if rms else None,
                peak=peak, peak_dbfs=float(20 * np.log10(peak)) if peak else None,
                nonzero_fraction=float(np.count_nonzero(x) / len(x)))


def runs(ids, vocab):
    out = []
    start = 0
    for end in range(1, len(ids) + 1):
        if end == len(ids) or ids[end] != ids[start]:
            tid = int(ids[start])
            if tid != 0:
                out.append(dict(index=len(out), token_id=tid, token=vocab[tid],
                                start_frame=start, end_frame_exclusive=end,
                                start_s=start * HOP, end_s=end * HOP,
                                midpoint_s=(start + end) * HOP / 2))
            start = end
    return out


def align(ref, out, constrained=False):
    """Suffix DP with forward reconstruction and deterministic operation priority."""
    n, m = len(ref), len(out)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    if not constrained:
        for i in range(n + 1):
            dp[i][m] = n - i
        for j in range(m + 1):
            dp[n][j] = m - j

    def eligible(i, j):
        return (ref[i]['token_id'] == out[j]['token_id'] and
                (not constrained or abs(out[j]['midpoint_s'] - ref[i]['midpoint_s']) * 1000
                 <= MAX_SHIFT_MS + EPS_MS))

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if constrained:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1],
                               1 + dp[i + 1][j + 1] if eligible(i, j) else -1)
            else:
                dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j + 1] + 1,
                               dp[i + 1][j + 1] + (not eligible(i, j)))
    events = []
    i = j = 0
    while i < n or j < m:
        if i < n and j < m and eligible(i, j) and dp[i][j] == dp[i + 1][j + 1] + (1 if constrained else 0):
            events.append(('M', i, j)); i += 1; j += 1
        elif not constrained and i < n and j < m and dp[i][j] == 1 + dp[i + 1][j + 1]:
            events.append(('S', i, j)); i += 1; j += 1
        elif i < n and dp[i][j] == dp[i + 1][j] + (0 if constrained else 1):
            events.append(('D', i, None)); i += 1
        else:
            assert j < m and dp[i][j] == dp[i][j + 1] + (0 if constrained else 1)
            events.append(('I', None, j)); j += 1
    return events


def shift_stats(values):
    if not values:
        return dict(count=0, signed_ms=[], mean_ms=None, median_ms=None, min_ms=None,
                    max_ms=None, median_abs_ms=None, max_abs_ms=None)
    a = np.asarray(values)
    return dict(count=len(values), signed_ms=values, mean_ms=float(a.mean()),
                median_ms=float(np.median(a)), min_ms=float(a.min()), max_ms=float(a.max()),
                median_abs_ms=float(np.median(abs(a))), max_abs_ms=float(abs(a).max()))


def summarize(ref, out, events):
    counts = Counter(op for op, _, _ in events)
    shifts = [(out[j]['midpoint_s'] - ref[i]['midpoint_s']) * 1000
              for op, i, j in events if op == 'M']
    return dict(Nref=len(ref), Nout=len(out), **{k: counts[k] for k in 'MSDI'},
                token_edit_error_proxy=(counts['S'] + counts['D'] + counts['I']) / len(ref) if ref else None,
                matched_count=counts['M'], output_count=len(out),
                matched_output_fraction=counts['M'] / len(out) if out else None,
                match_midpoint_shifts=shift_stats(shifts))


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    outputs = ['run_started.json', 'metrics.json', 'timeline.csv', 'sequence_alignment.csv',
               'fixed_time_clean_targets.csv', 'experiment_report.md']
    existing = [name for name in outputs if (BASE / name).exists()]
    if existing:
        raise SystemExit('Refusing existing or partial results: ' + ', '.join(existing))
    source = BASE.parent / 'p8_q_a_ctc_fixed_time_v2_20261018'
    prior = BASE.parent / 'p8_phoneme_probe_20261020'
    paths = dict(source_metrics=source / 'metrics.json', cache=source / 'frame_probabilities.npz',
                 fixed_time_csv=source / 'fixed_time_token_posteriors.csv',
                 prior_sequences=prior / 'phoneme_results.json', vocab=prior / 'local_model/vocab.json',
                 preprocessor_config=prior / 'local_model/preprocessor_config.json',
                 script=Path(__file__).resolve(), config=BASE / 'config.yaml')
    hashes = {k: dict(path=str(v), sha256=sha(v)) for k, v in paths.items()}
    metadata = json.loads(paths['source_metrics'].read_text())
    assert hashes['cache']['sha256'] == metadata['probability_cache']['sha256']
    assert hashes['vocab']['sha256'] == metadata['tokenizer_vocab_sha256']
    vocab = {v: k for k, v in json.loads(paths['vocab'].read_text()).items()}
    manifest = [r for r in metadata['probability_cache']['manifest'] if r['group'] == GROUP]
    assert sorted(r['condition'] for r in manifest) == sorted(CONDITIONS)
    levels = {entry['condition']: wav_levels(Path(entry['path'])) for entry in manifest}
    for entry in manifest:
        assert levels[entry['condition']]['sha256'] == metadata['source_sha256'][entry['path']]
    preprocessor = json.loads(paths['preprocessor_config'].read_text())
    (BASE / 'run_started.json').write_text(json.dumps(dict(started_at_utc=datetime.now(timezone.utc).isoformat(), inputs=hashes), indent=2))
    cache = np.load(paths['cache'], allow_pickle=False)
    sequences, timeline, cached_confidence = {}, [], {}
    for entry in manifest:
        condition = entry['condition']
        ids = cache[entry['argmax_ids_key']]
        assert len(ids) == entry['frames']
        seq = runs(ids, vocab)
        sequences[condition] = seq
        columns = {int(t): c for c, t in enumerate(cache[entry['token_ids_key']])}
        probs = cache[entry['probabilities_key']]
        for event in seq:
            col = columns.get(event['token_id'])
            confidence = (float(probs[event['start_frame']:event['end_frame_exclusive'], col].mean())
                          if col is not None else None)
            cached_confidence[(condition, event['index'])] = confidence
            timeline.append(dict(condition=condition, **event, consonant=event['token'] in CONSONANTS,
                                 run_token_posterior_mean=confidence,
                                 confidence_status='cached_target_column' if col is not None else 'unavailable_not_cached'))
    ref = sequences['clean']
    old = {r['condition']: r for r in json.loads(paths['prior_sequences'].read_text())['records'] if r['group'] == GROUP}
    results, csv_events = {}, []
    for condition in CONDITIONS:
        out = sequences[condition]
        events = align(ref, out)
        stats = summarize(ref, out, events)
        stats['decoded_tokens'] = [r['token'] for r in out]
        stats['prior_decoded_tokens'] = old[condition]['decoded_tokens']
        stats['prior_sequence_exact_match'] = stats['decoded_tokens'] == stats['prior_decoded_tokens']
        stats['prior_token_edit_distance_to_clean'] = old[condition]['token_edit_distance_to_clean']
        stats['recovered_edit_distance'] = stats['S'] + stats['D'] + stats['I']
        cref = [r for r in ref if r['token'] in CONSONANTS]
        cout = [r for r in out if r['token'] in CONSONANTS]
        stats['consonants_only_independent_alignment'] = summarize(cref, cout, align(cref, cout))
        constrained = align(ref, out, constrained=True)
        stats['secondary_100ms_monotonic_matching'] = dict(
            matched_count=sum(op == 'M' for op, _, _ in constrained),
            matched_pairs=[dict(clean_index=i, output_index=j,
                                shift_ms=(out[j]['midpoint_s']-ref[i]['midpoint_s'])*1000)
                           for op, i, j in constrained if op == 'M'])
        stats['unmatched_output_tokens'] = []
        for event_index, (op, i, j) in enumerate(events):
            a, b = ref[i] if i is not None else {}, out[j] if j is not None else {}
            status = (('absent_anywhere_in_clean' if b['token_id'] not in {r['token_id'] for r in ref}
                       else 'appears_elsewhere_in_clean_but_unaligned') if b and op != 'M' else '')
            row = dict(condition=condition, event_index=event_index, operation=op,
                       clean_index=a.get('index'), output_index=b.get('index'),
                       clean_token_id=a.get('token_id'), output_token_id=b.get('token_id'),
                       clean_token=a.get('token'), output_token=b.get('token'),
                       clean_start_frame=a.get('start_frame'), clean_end_frame_exclusive=a.get('end_frame_exclusive'),
                       output_start_frame=b.get('start_frame'), output_end_frame_exclusive=b.get('end_frame_exclusive'),
                       clean_start_s=a.get('start_s'), clean_end_s=a.get('end_s'),
                       output_start_s=b.get('start_s'), output_end_s=b.get('end_s'),
                       midpoint_shift_ms=(b['midpoint_s']-a['midpoint_s'])*1000 if a and b else None,
                       unmatched_output_status=status,
                       output_run_posterior_mean=cached_confidence.get((condition, b.get('index'))),
                       output_confidence_status=('cached_target_column' if cached_confidence.get((condition,b.get('index'))) is not None
                                                 else 'unavailable_not_cached' if b else 'no_output_token'))
            csv_events.append(row)
            if status:
                stats['unmatched_output_tokens'].append(row)
        results[condition] = stats
    with paths['fixed_time_csv'].open() as handle:
        target_rows = [r for r in csv.DictReader(handle) if r['group'] == GROUP]
    # Validate that the reused CSV really refers to these recovered clean runs.
    clean_rows = [r for r in target_rows if r['condition'] == 'clean']
    assert len(clean_rows) == len(ref)
    for r, event in zip(clean_rows, ref):
        assert (int(r['token_id']), int(r['clean_start_frame']), int(r['clean_end_frame_exclusive'])) == (
            event['token_id'], event['start_frame'], event['end_frame_exclusive'])
    write_csv(BASE / 'timeline.csv', timeline)
    write_csv(BASE / 'sequence_alignment.csv', csv_events)
    write_csv(BASE / 'fixed_time_clean_targets.csv', target_rows)
    metrics = dict(experiment_id=BASE.name, status='completed', group=GROUP,
                   completed_at_utc=datetime.now(timezone.utc).isoformat(), inputs=hashes,
                   source_manifest=manifest, frame_hop_s=HOP, blank_id=0,
                   source_audio_levels=levels, preprocessor_config=preprocessor,
                   normalization_confound='Local feature extractor do_normalize=true. Clip amplitude can be normalized before recognition; many greedy tokens do not establish audibility or intelligibility of the original quiet waveform.',
                   primary='unit-cost unconstrained Levenshtein against clean greedy proxy',
                   tie_breaking='forward suffix DP: match, substitution, deletion, insertion',
                   secondary='maximize monotonic same-ID matches with abs midpoint shift <= 100 ms + 1e-6 ms; prefer match, skip clean, skip output',
                   consonant_ipa_mapping=CONSONANTS, results=results,
                   limits=['Clean sequence is recognizer-derived, not verified phoneme truth.',
                           'Approximate 20 ms CTC frame times are not acoustic boundaries.',
                           'Token edit error proxy is not true PER or WER.',
                           'Unmatched output tokens and insertions are not proof of hallucinations.',
                           'Nonclean output token confidence unavailable if no cached target column.',
                           'Per-clip feature normalization confounds comparison of original waveform loudness with greedy token count.',
                           'No inference, tuning, downloads, training or source edits.'])
    with (BASE / 'metrics.json').open('x') as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, allow_nan=False)
    report = ['# Cached content identity audit', '',
              'This compares recognizer outputs against the clean recognizer proxy. It cannot identify acoustic hallucinations or verify words.', '',
              '| Condition | Nout | M | S | D | I | Edit proxy | M/Nout | <=100ms matches |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        r = results[condition]
        precision = 'undefined (empty)' if r['matched_output_fraction'] is None else f"{r['matched_output_fraction']:.4f}"
        report.append(f"| {condition} | {r['Nout']} | {r['M']} | {r['S']} | {r['D']} | {r['I']} | {r['token_edit_error_proxy']:.4f} | {precision} | {r['secondary_100ms_monotonic_matching']['matched_count']} |")
    report += ['', 'Primary alignment is unconstrained Levenshtein; ties prefer match, substitution, deletion, insertion during forward suffix-DP reconstruction. The secondary diagnostic maximizes monotonic same-ID matches within 100 ms + 1e-6 ms; its threshold is fixed.', '',
               'See metrics.json for consonant analysis, shift distributions, prior-sequence comparisons, unmatched output classifications and input SHA256 hashes; sequence_alignment.csv for every operation; timeline.csv for cached token runs and available confidence; fixed_time_clean_targets.csv for unchanged source rows at clean timestamps.', '',
               'Times are approximate CTC runs using a 20 ms hop. Cached posteriors cover clean-target classes only. An unavailable output-class confidence is left blank. No output mismatch is labeled a hallucination. Research Lead interpretation follows separately.', '',
               'The local feature extractor configuration has do_normalize=true. Its per-clip normalization can amplify very quiet input before recognition: decoded tokens do not establish audibility or intelligibility in the original waveform. Read-only source WAV RMS dBFS, peak and nonzero fractions are recorded in metrics.json.']
    with (BASE / 'experiment_report.md').open('x') as handle:
        handle.write('\n'.join(report) + '\n')
    print(json.dumps({c: {k: r[k] for k in ['Nref', 'Nout', 'M', 'S', 'D', 'I', 'token_edit_error_proxy', 'prior_sequence_exact_match']} for c, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
