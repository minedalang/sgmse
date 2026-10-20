#!/usr/bin/env python3
"""Create and hash-lock the predeclared 64-speaker CV manifest."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
import soundfile as sf

PROJECT = Path('/data/coding/sgmse')
SPLIT = Path('/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/scp_new/cv_synthetic')
SYNTHETIC = SPLIT.parent.parent
REPEAT32 = PROJECT / 'experiments/rrsd_cv_repeat32/manifest.csv'
OUT = PROJECT / 'experiments/speaker_headroom_replication_64_20261017'
SEED = 20261017


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    wav_scp = SPLIT / 'wav.scp'
    utt2spk = SPLIT / 'utt2spk'
    wav = {}
    for line in wav_scp.read_text(encoding='utf-8').splitlines():
        uttid, source = line.split(maxsplit=1)
        if uttid in wav:
            raise ValueError(f'duplicate wav.scp key: {uttid}')
        path = (SPLIT / source).resolve()
        wav[uttid] = path
    speakers = {}
    for line in utt2spk.read_text(encoding='utf-8').splitlines():
        uttid, speaker = line.split()
        if uttid in speakers:
            raise ValueError(f'duplicate utt2spk key: {uttid}')
        speakers[uttid] = speaker
    if set(wav) != set(speakers):
        raise ValueError(f'SCP/utt2spk key mismatch: {len(set(wav)^set(speakers))}')

    repeat_rows = list(csv.DictReader(REPEAT32.open(newline='', encoding='utf-8')))
    excluded = {row['speaker'] for row in repeat_rows} | {'reader_10658'}
    counts = Counter(speakers.values())
    eligible = sorted(s for s, n in counts.items() if n >= 2 and s not in excluded)
    if len(eligible) != 124:
        raise ValueError(f'Expected exactly 124 eligible speakers, found {len(eligible)}')

    rng = random.Random(SEED)
    chosen_speakers = sorted(rng.sample(eligible, 64))
    selected = []
    seen_files = set()
    for speaker in chosen_speakers:
        candidates = sorted(u for u, s in speakers.items() if s == speaker)
        for uttid in sorted(rng.sample(candidates, 2)):
            noisy = wav[uttid]
            match = re.search(r'fileid_(\d+)', noisy.name)
            if not match:
                raise ValueError(f'fileid missing from noisy basename: {noisy}')
            clean = (SYNTHETIC / 'clean' / f'clean_fileid_{match.group(1)}.wav').resolve()
            if noisy.parent != (SYNTHETIC / 'noisy').resolve():
                raise ValueError(f'noisy source escaped required parent: {noisy}')
            if clean.parent != (SYNTHETIC / 'clean').resolve():
                raise ValueError(f'clean source escaped required parent: {clean}')
            if not noisy.is_file() or not clean.is_file():
                raise FileNotFoundError((noisy, clean))
            if noisy.name in seen_files:
                raise ValueError(f'nonunique staged basename: {noisy.name}')
            seen_files.add(noisy.name)
            ni, ci = sf.info(noisy), sf.info(clean)
            if (ni.samplerate, ni.channels, ni.frames) != (16000, 1, ci.frames):
                raise ValueError(f'noisy format/length invalid for {uttid}: {ni}')
            if (ci.samplerate, ci.channels) != (16000, 1):
                raise ValueError(f'clean format invalid for {uttid}: {ci}')
            selected.append({
                'filename': noisy.name,
                'uttid': uttid,
                'speaker': speaker,
                'fileid': match.group(1),
                'noisy_path': str(noisy),
                'clean_path': str(clean),
                'sample_rate_hz': ni.samplerate,
                'samples': ni.frames,
                'duration_seconds': ni.frames / ni.samplerate,
                'noisy_sha256': sha256(noisy),
                'clean_sha256': sha256(clean),
            })
    if len(selected) != 128 or len({r['speaker'] for r in selected}) != 64:
        raise ValueError('selection did not produce 128 rows and 64 speakers')

    manifest_path = OUT / 'manifest.csv'
    fields = list(selected[0])
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)
    manifest_hash = sha256(manifest_path)
    (OUT / 'manifest.sha256').write_text(f'{manifest_hash}  manifest.csv\n', encoding='utf-8')
    lock = {
        'selection_seed': SEED,
        'eligible_speakers': len(eligible),
        'selected_speakers': len(chosen_speakers),
        'selected_utterances': len(selected),
        'excluded_speaker_count': len(excluded),
        'excluded_repeat32_speakers': sorted({row['speaker'] for row in repeat_rows}),
        'additional_exclusion': 'reader_10658',
        'source_scp_sha256': sha256(wav_scp),
        'source_utt2spk_sha256': sha256(utt2spk),
        'repeat32_manifest_sha256': sha256(REPEAT32),
        'manifest_sha256': manifest_hash,
        'duration_seconds': sum(r['duration_seconds'] for r in selected),
        'input_bytes': sum(Path(r['noisy_path']).stat().st_size for r in selected),
        'clean_bytes': sum(Path(r['clean_path']).stat().st_size for r in selected),
        'estimated_two_arm_output_bytes_pcm16': sum(r['samples'] for r in selected) * 2 * 2,
        'estimated_two_arm_output_bytes_float32': sum(r['samples'] for r in selected) * 2 * 4,
    }
    (OUT / 'manifest_lock.json').write_text(json.dumps(lock, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(lock, indent=2))

if __name__ == '__main__':
    main()
