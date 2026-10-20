#!/usr/bin/env python3
"""CAM++ analysis with speaker-cluster bootstrap on the locked manifest."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path('/data/coding/sgmse')
EXP = ROOT / 'experiments/speaker_headroom_replication_64_20261017'
LOG = ROOT / 'experiment_logs/speaker_headroom_replication_64_20261017'
MANIFEST = EXP / 'manifest.csv'
LOCKED_SHA256 = 'c04657fa21a33c7f2dc749514cec899767101ba53f46222e9f4ab8577e2f4492'
ENCODER = ROOT / 'pre_models/wespeaker-voxceleb-campplus/avg_model.pt'
ARMS = ('noisy', 'off', 'router', 'clean')
REPS = 5000
SEED = 20261017
SR = 16000
SAMPLES = 32000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_wave(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if sr != SR or wav.ndim != 1 or len(wav) < SAMPLES:
        raise ValueError(f'invalid audio {path}: sr={sr}, shape={wav.shape}')
    wav = wav[:SAMPLES]
    if not np.isfinite(wav).all():
        raise ValueError(f'nonfinite samples: {path}')
    return wav


def normalize_wave(wav: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32).copy()).view(1, -1)
    x = x - x.mean(dim=1, keepdim=True)
    return x / (x.abs().amax(dim=1, keepdim=True) + 1e-6)


def auc_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int8)
    pos = labels == 1
    neg = ~pos
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    sorted_pos = pos[order].astype(np.int64)
    sorted_neg = neg[order].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    pos_g = np.add.reduceat(sorted_pos, starts)
    neg_g = np.add.reduceat(sorted_neg, starts)
    neg_below = np.cumsum(neg_g) - neg_g
    auc = np.sum(pos_g * (neg_below + 0.5 * neg_g)) / (pos.sum() * neg.sum())
    desc_order = np.arange(len(starts) - 1, -1, -1)
    tp = np.cumsum(pos_g[desc_order])
    fp = np.cumsum(neg_g[desc_order])
    far = fp / neg.sum()
    frr = (pos.sum() - tp) / pos.sum()
    j = int(np.argmin(np.abs(far - frr)))
    return float(auc), float((far[j] + frr[j]) / 2), float(far[j]), float(frr[j])


def pair_layout(speakers: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    same = []
    different = []
    for spk in sorted(set(speakers)):
        ids = [j for j, s in enumerate(speakers) if s == spk]
        if len(ids) != 2:
            raise ValueError(f'expected two rows for {spk}, got {len(ids)}')
        same.append((ids[0], ids[1]))
    uniq = sorted(set(speakers))
    for a in range(len(uniq)):
        ia = [i for i, s in enumerate(speakers) if s == uniq[a]]
        for b in range(a + 1, len(uniq)):
            ib = [i for i, s in enumerate(speakers) if s == uniq[b]]
            different.extend((i, j) for i in ia for j in ib)
    return same, different


def ci(x: np.ndarray) -> list[float]:
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def main() -> None:
    started = time.time()
    manifest_hash = sha256(MANIFEST)
    if manifest_hash != LOCKED_SHA256:
        raise RuntimeError(f'manifest hash changed: {manifest_hash}')
    if sha256(ENCODER) != '07abeeb5150441995b51ea65c9ccc8feed78b33040012f1d2fad29a0e4f5b8d7':
        raise RuntimeError('CAM++ model hash changed')
    with MANIFEST.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 128 or len({r['speaker'] for r in rows}) != 64:
        raise RuntimeError('locked selection cardinality mismatch')

    speakers = [r['speaker'] for r in rows]
    paths: dict[str, list[Path]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for r in rows:
            if arm == 'noisy':
                p = Path(r['noisy_path'])
                if sha256(p) != r['noisy_sha256']:
                    raise RuntimeError(f'noisy source changed: {p}')
            elif arm == 'clean':
                p = Path(r['clean_path'])
                if sha256(p) != r['clean_sha256']:
                    raise RuntimeError(f'clean source changed: {p}')
            else:
                p = LOG / arm / r['filename']
            if not p.is_file():
                raise FileNotFoundError(p)
            # Check all generated outputs for finite values and source-aligned sample rate/length.
            wav, sr = sf.read(p, dtype='float32', always_2d=False)
            if sr != SR or wav.ndim != 1 or len(wav) != int(r['samples']) or not np.isfinite(wav).all():
                raise ValueError(f'invalid aligned finite audio: {p}, sr={sr}, shape={wav.shape}')
            paths[arm].append(p)

    from sgmse.speaker_encoders import WeSpeakerCAMPPlusEncoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    encoder = WeSpeakerCAMPPlusEncoder(str(ENCODER)).to(device).eval()
    embeddings: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for arm, arm_paths in paths.items():
            parts = []
            for start in range(0, len(arm_paths), 8):
                waves = torch.cat([normalize_wave(load_wave(p)) for p in arm_paths[start:start + 8]], dim=0).to(device)
                parts.append(encoder(waves).float().cpu())
            embeddings[arm] = F.normalize(torch.cat(parts).float(), dim=1).numpy()
            print(f'{arm}: {embeddings[arm].shape}', flush=True)

    same_pairs, diff_pairs = pair_layout(speakers)
    diff_pair_to_index = {pair: idx for idx, pair in enumerate(diff_pairs)}
    speaker_names = sorted(set(speakers))
    speaker_index = {name: idx for idx, name in enumerate(speaker_names)}
    diff_row_cluster_ids = np.asarray([(speaker_index[speakers[i]], speaker_index[speakers[j]])
                                       for i, j in diff_pairs], dtype=np.int64)
    speaker_rows = {s: [i for i, row in enumerate(rows) if row['speaker'] == s] for s in speaker_names}
    rng = np.random.default_rng(SEED)
    boot_ids = rng.integers(0, len(speaker_names), size=(REPS, len(speaker_names)))
    counts = np.zeros((REPS, len(speaker_names)), dtype=np.int16)
    # Repeated cluster draws contribute integer weights.
    for b in range(REPS):
        counts[b] = np.bincount(boot_ids[b], minlength=len(speaker_names))

    result = {
        'experiment_id': 'speaker_headroom_replication_64_20261017',
        'status': 'analysis_complete',
        'manifest_sha256': manifest_hash,
        'encoder_sha256': sha256(ENCODER),
        'n_utterances': len(rows),
        'n_speakers': len(speaker_names),
        'segment_samples': SAMPLES,
        'segment_seconds': SAMPLES / SR,
        'sample_rate_hz': SR,
        'device': device,
        'bootstrap': {'replicates': REPS, 'seed': SEED, 'unit': 'speaker cluster',
                      'cluster_draws_per_replicate': len(speaker_names)},
        'arms': {},
        'paired_contrasts': {},
    }
    per_utt = []
    for i, row in enumerate(rows):
        entry = {'filename': row['filename'], 'uttid': row['uttid'], 'speaker': row['speaker']}
        clean_emb = torch.from_numpy(embeddings['clean'][i:i + 1])
        for arm in ARMS:
            entry[f'{arm}_paired_clean_cosine'] = float(np.dot(embeddings[arm][i], embeddings['clean'][i]))
        entry['noisy_sha256'] = row['noisy_sha256']
        entry['clean_sha256'] = row['clean_sha256']
        entry['off_sha256'] = sha256(paths['off'][i]) if 'off' in paths else ''
        entry['router_sha256'] = sha256(paths['router'][i]) if 'router' in paths else ''
        per_utt.append(entry)
    with (EXP / 'per_utterance_campp.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_utt[0]))
        writer.writeheader(); writer.writerows(per_utt)

    score_arrays = {}
    for arm in ARMS:
        embs = embeddings[arm]
        same_scores = np.array([np.dot(embs[i], embs[j]) for i, j in same_pairs], dtype=np.float64)
        diff_scores = np.array([np.dot(embs[i], embs[j]) for i, j in diff_pairs], dtype=np.float64)
        scores = np.concatenate([same_scores, diff_scores])
        labels = np.r_[np.ones(len(same_scores), dtype=np.int8), np.zeros(len(diff_scores), dtype=np.int8)]
        auc, eer, eer_far, eer_frr = auc_eer(scores, labels)
        paired_cos = np.array([r[f'{arm}_paired_clean_cosine'] for r in per_utt], dtype=np.float64)
        same_boot_w = counts.copy()
        same_boot = (same_boot_w * same_scores[None, :]).sum(axis=1) / same_boot_w.sum(axis=1)
        diff_pairs_by_speaker = []
        uniq_pairs = []
        for a in range(len(speaker_names)):
            for b in range(a + 1, len(speaker_names)):
                ia, ib = speaker_rows[speaker_names[a]], speaker_rows[speaker_names[b]]
                idxs = [diff_pair_to_index[(i, j)] for i in ia for j in ib]
                uniq_pairs.append((a, b))
                diff_pairs_by_speaker.append(diff_scores[idxs])
        diff_by_pair = np.asarray(diff_pairs_by_speaker, dtype=np.float64)  # [2016,4]
        unique_pair_idx = np.asarray(uniq_pairs, dtype=np.int64)
        diff_cluster_w = counts[:, unique_pair_idx[:, 0]] * counts[:, unique_pair_idx[:, 1]]
        diff_mean_boot = (diff_cluster_w * diff_by_pair.mean(axis=1)[None, :]).sum(axis=1) / diff_cluster_w.sum(axis=1)
        paired_boot = np.zeros(REPS, dtype=np.float64)
        for sidx, name in enumerate(speaker_names):
            utt_idx = speaker_rows[name]
            values = paired_cos[utt_idx].mean()
            paired_boot += counts[:, sidx] * values
        paired_boot /= counts.sum(axis=1)

        # Cluster-weighted ROC metrics. Bootstrap resamples clusters, then constructs
        # same-label within-cluster pairs and impostor pairs only across distinct IDs.
        score_order = np.argsort(scores, kind='mergesort')
        sorted_scores = scores[score_order]
        group_starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        group_ids = np.cumsum(np.r_[0, (sorted_scores[1:] != sorted_scores[:-1]).astype(np.int64)])
        n_groups = int(group_ids[-1] + 1)
        auc_boot = np.empty(REPS, dtype=np.float64)
        eer_boot = np.empty(REPS, dtype=np.float64)
        # One same-speaker positive score per cluster; four impostor scores per distinct speaker pair.
        diff_weight_by_pair = (counts[:, diff_row_cluster_ids[:, 0]] *
                               counts[:, diff_row_cluster_ids[:, 1]])
        descending = np.arange(n_groups - 1, -1, -1)
        auc_boot = np.empty(REPS, dtype=np.float64)
        eer_boot = np.empty(REPS, dtype=np.float64)
        for block_start in range(0, REPS, 256):
            block_end = min(REPS, block_start + 256)
            block_counts = counts[block_start:block_end]
            block_pos = np.zeros((block_end - block_start, len(scores)), dtype=np.float64)
            block_neg = np.zeros_like(block_pos)
            block_pos[:, :len(same_scores)] = block_counts
            block_neg[:, len(same_scores):] = diff_weight_by_pair[block_start:block_end]
            pos_groups = np.add.reduceat(block_pos[:, score_order], group_starts, axis=1)
            neg_groups = np.add.reduceat(block_neg[:, score_order], group_starts, axis=1)
            neg_below = np.cumsum(neg_groups, axis=1) - neg_groups
            pos_total = pos_groups.sum(axis=1)
            neg_total = neg_groups.sum(axis=1)
            auc_boot[block_start:block_end] = (
                pos_groups * (neg_below + 0.5 * neg_groups)
            ).sum(axis=1) / (pos_total * neg_total)
            tp = np.cumsum(pos_groups[:, descending], axis=1)
            fp = np.cumsum(neg_groups[:, descending], axis=1)
            far_boot = fp / neg_total[:, None]
            frr_boot = (pos_total[:, None] - tp) / pos_total[:, None]
            best = np.argmin(np.abs(far_boot - frr_boot), axis=1)
            eer_boot[block_start:block_end] = 0.5 * (
                far_boot[np.arange(block_end - block_start), best] +
                frr_boot[np.arange(block_end - block_start), best]
            )
        result['arms'][arm] = {
            'paired_clean_cosine': {
                'mean': float(paired_cos.mean()), 'median': float(np.median(paired_cos)),
                'cluster_bootstrap_95ci': ci(paired_boot),
            },
            'same_speaker_cosine': {
                'mean': float(same_scores.mean()), 'n': len(same_scores),
                'cluster_bootstrap_95ci': ci(same_boot),
            },
            'different_speaker_cosine': {
                'mean': float(diff_scores.mean()), 'n': len(diff_scores),
                'cluster_bootstrap_95ci': ci(diff_mean_boot),
            },
            'auc': {'value': auc, 'cluster_bootstrap_95ci': ci(auc_boot)},
            'eer': {'value': eer, 'far_at_eer': eer_far, 'frr_at_eer': eer_frr,
                    'cluster_bootstrap_95ci': ci(eer_boot)},
        }
        score_arrays[arm] = {'same': same_scores, 'different': diff_scores, 'paired': paired_cos,
                             'same_boot': same_boot, 'different_boot': diff_mean_boot,
                             'paired_boot': paired_boot, 'auc': result['arms'][arm]['auc']['value'],
                             'eer': result['arms'][arm]['eer']['value'], 'auc_boot': auc_boot, 'eer_boot': eer_boot}

    for a, b in (('router', 'off'), ('router', 'clean'), ('off', 'clean')):
        out = {}
        for key, field in (('paired_clean_cosine', 'paired'), ('same_speaker_cosine', 'same'),
                           ('different_speaker_cosine', 'different'), ('auc', 'auc'), ('eer', 'eer')):
            x = score_arrays[a][field]
            y = score_arrays[b][field]
            if key in ('paired_clean_cosine', 'same_speaker_cosine', 'different_speaker_cosine'):
                delta = x.mean() - y.mean()
                boot = score_arrays[a][field + '_boot'] - score_arrays[b][field + '_boot']
            else:
                delta = result['arms'][a][key]['value'] - result['arms'][b][key]['value']
                boot = score_arrays[a][field + '_boot'] - score_arrays[b][field + '_boot']
            out[f'{a}_minus_{b}_{key}'] = {'mean_delta': float(delta), 'cluster_bootstrap_95ci': ci(boot)}
        result['paired_contrasts'][f'{a}_vs_{b}'] = out

    same_deficit = score_arrays['clean']['same'].mean() - score_arrays['router']['same'].mean()
    same_deficit_boot = score_arrays['clean']['same_boot'] - score_arrays['router']['same_boot']
    auc_deficit = result['arms']['clean']['auc']['value'] - result['arms']['router']['auc']['value']
    auc_deficit_boot = score_arrays['clean']['auc_boot'] - score_arrays['router']['auc_boot']
    eer_excess = result['arms']['router']['eer']['value'] - result['arms']['clean']['eer']['value']
    eer_excess_boot = score_arrays['router']['eer_boot'] - score_arrays['clean']['eer_boot']
    result['predeclared_gate'] = {
        'router_same_speaker_deficit_vs_clean': {
            'estimate': float(same_deficit), 'cluster_bootstrap_95ci': ci(same_deficit_boot),
            'threshold': 0.05, 'passes_magnitude_and_ci': bool(same_deficit >= .05 and ci(same_deficit_boot)[0] > 0),
        },
        'router_auc_deficit_vs_clean': {
            'estimate': float(auc_deficit), 'cluster_bootstrap_95ci': ci(auc_deficit_boot),
            'threshold': 0.02, 'passes_magnitude_and_ci': bool(auc_deficit >= .02 and ci(auc_deficit_boot)[0] > 0),
        },
        'router_eer_excess_vs_clean': {
            'estimate': float(eer_excess), 'cluster_bootstrap_95ci': ci(eer_excess_boot),
            'threshold': 0.02, 'passes_magnitude_and_ci': bool(eer_excess >= .02 and ci(eer_excess_boot)[0] > 0),
        },
        'gate_passed': bool(same_deficit >= .05 and ci(same_deficit_boot)[0] > 0 and
                             ((auc_deficit >= .02 and ci(auc_deficit_boot)[0] > 0) or
                              (eer_excess >= .02 and ci(eer_excess_boot)[0] > 0))),
        'interpretation': 'development-only; a pass supports investigating a no-training headroom hypothesis but does not authorize training without Research Lead review',
    }
    result['runtime_seconds'] = time.time() - started
    (EXP / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'arms': result['arms'], 'paired_contrasts': result['paired_contrasts'],
                      'predeclared_gate': result['predeclared_gate'],
                      'runtime_seconds': result['runtime_seconds']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
