#!/usr/bin/env python3
"""Validate saved probe artifacts offline; exclusively create validation.json."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    output = ROOT / 'validation.json'
    if output.exists():
        raise FileExistsError(output)
    metrics = json.loads((ROOT / 'metrics.json').read_text())
    v1 = json.loads((REPO / 'experiments/p8_q_a_ctc_fixed_time_probe_20261020/metrics.json').read_text())
    source_path = REPO / 'experiments/p8_q_a_ctc_forced_alignment_20261020/metrics.json'
    source = json.loads(source_path.read_text())
    source_tokens = {(g['group'], int(t['target_index'])): t
                     for g in source['groups_detail'] for t in g['clean_target_tokens']}
    rows = list(csv.DictReader((ROOT / 'fixed_time_token_posteriors.csv').open()))
    identities = [(r['group'], int(r['target_index']), r['condition']) for r in rows]
    expected = {(g, i, c) for g, i in source_tokens
                for c in ('clean', 'noisy', 'y_pre', 'off', 'router')}
    manifests = {(m['group'], m['condition']): m for m in metrics['probability_cache']['manifest']}
    keys = ('fixed_time_target_posterior_mean', 'fixed_time_target_top1_fraction',
            'nearby_target_posterior_max', 'nearby_target_posterior_mean',
            'nearby_target_top1_fraction')
    errors = {key: 0.0 for key in keys}
    integer_mismatches = 0
    source_mismatches = 0
    clean_errors = []
    endpoint_errors = []
    with np.load(ROOT / 'frame_probabilities.npz', allow_pickle=False) as cache:
        for row in rows:
            group, condition = row['group'], row['condition']
            token = source_tokens[(group, int(row['target_index']))]
            m = manifests[(group, condition)]
            probabilities = cache[m['probabilities_key']]
            token_ids = cache[m['token_ids_key']]
            argmax_ids = cache[m['argmax_ids_key']]
            token_id = int(row['token_id'])
            column = np.flatnonzero(token_ids == token_id)
            if len(column) != 1:
                raise ValueError(f'Invalid token map: {group}/{condition}/{token_id}')
            start, end = int(row['clean_start_frame']), int(row['clean_end_frame_exclusive'])
            nlo, nhi = max(0, start - 10), min(len(probabilities), end + 10)
            valid = (0 <= start < end <= len(probabilities)
                     and len(probabilities) == len(argmax_ids) == m['frames']
                     and probabilities.shape[1] == len(token_ids)
                     and probabilities.dtype == np.float32)
            if not valid:
                endpoint_errors.append([group, condition, row['target_index']])
            integer_mismatches += int(int(row['fixed_window_frames']) != end - start)
            source_mismatches += int(
                (start, end, token_id, row['token']) !=
                (int(token['clean_greedy_start_frame']), int(token['clean_greedy_end_frame_exclusive']),
                 int(token['token_id']), token['token'])
                or float(row['nearby_window_s']) != 0.20)
            fixed = probabilities[start:end, column[0]]
            nearby = probabilities[nlo:nhi, column[0]]
            computed = (float(np.mean(fixed)), float(np.mean(argmax_ids[start:end] == token_id)),
                        float(np.max(nearby)), float(np.mean(nearby)),
                        float(np.mean(argmax_ids[nlo:nhi] == token_id)))
            for key, value in zip(keys, computed):
                saved = float(row[key])
                if not np.isfinite(value) or not np.isfinite(saved):
                    raise ValueError(f'Nonfinite metric: {key}')
                errors[key] = max(errors[key], abs(saved - value))
            if condition == 'clean':
                clean_errors.append(abs(float(row[keys[0]]) - float(token['clean_run_posterior_mean'])))
    audio_checks = {p: sha256(p) == digest for p, digest in metrics['source_sha256'].items()}
    model_path = REPO / 'experiments/p8_phoneme_probe_20261020/local_model/pytorch_model.bin'
    hashes = {
        'audio_all_current_hashes_match': all(audio_checks.values()),
        'audio_manifest_identical_to_v1': metrics['source_sha256'] == v1['source_sha256'],
        'model_current_hash_matches': sha256(model_path) == metrics['model_weight_sha256'],
        'model_hash_identical_to_v1': metrics['model_weight_sha256'] == v1['model_weight_sha256'],
        'source_metrics_hash_matches': sha256(source_path) == metrics['source_metrics_sha256'],
        'config_hash_matches': sha256(ROOT / 'config.yaml') == metrics['config_sha256'],
        'probe_code_hash_matches': sha256(ROOT / 'fixed_time_probe.py') == metrics['probe_code_sha256'],
        'csv_hash_matches': sha256(ROOT / 'fixed_time_token_posteriors.csv') == metrics['token_csv_sha256'],
        'npz_hash_matches': sha256(ROOT / 'frame_probabilities.npz') == metrics['probability_cache']['sha256'],
    }
    checks = {
        '445_rows': len(rows) == 445,
        '89_unique_targets': len({(r['group'], r['target_index']) for r in rows}) == 89,
        'complete_unique_source_coverage': len(identities) == len(set(identities)) and set(identities) == expected,
        'integer_lengths_match': integer_mismatches == 0,
        'source_intervals_tokens_and_nearby_radius_match': source_mismatches == 0,
        'integer_endpoints_and_cache_shapes_valid': not endpoint_errors,
        'clean_posteriors_within_tolerance': len(clean_errors) == 89 and max(clean_errors) <= 1e-6,
        'all_five_cache_metrics_exact': all(value == 0.0 for value in errors.values()),
        **hashes,
    }
    result = {
        'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'validation_code_sha256': sha256(Path(__file__)),
        'mode': 'offline existing artifacts only; no inference',
        'pass': all(checks.values()), 'checks': checks,
        'rows': len(rows), 'unique_targets': len(source_tokens),
        'integer_length_mismatch_count': integer_mismatches,
        'source_interval_token_or_radius_mismatch_count': source_mismatches,
        'endpoint_or_cache_shape_errors': endpoint_errors,
        'clean_posterior_max_absolute_error': max(clean_errors),
        'clean_posterior_absolute_tolerance': 1e-6,
        'cache_metric_max_absolute_errors': errors,
        'cache_metric_absolute_tolerance': 0.0,
        'cache_metric_relative_tolerance': 0.0,
        'nearby_integer_endpoint_definition': '[max(0,start-10), min(number_of_frames,end+10))',
        'audio_files_checked': len(audio_checks), 'audio_hash_checks': audio_checks,
    }
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, indent=2))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
