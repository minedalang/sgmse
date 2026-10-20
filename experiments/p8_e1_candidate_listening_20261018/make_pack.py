"""Create a small, read-only listening pack from existing P8 WAVs."""
from pathlib import Path
import json
import pandas as pd
import soundfile as sf

raise RuntimeError(
    "This historical builder is invalidated: it applies central-12s coordinates "
    "to the full 30s noisy file. Use ../p8_e1_candidate_listening_aligned_20261020."
)

ROOT = Path('/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020')
RANK = Path('/data/coding/sgmse/experiments/p8_e1_candidate_ranking_20261018/candidate_windows.csv')
OUT = Path(__file__).resolve().parent
CLIPS = OUT / 'clips'
CLIPS.mkdir(exist_ok=True)

# One candidate per speaker, preserving the highest-ranked candidate available.
ranked = pd.read_csv(RANK)
selected = ranked.drop_duplicates('fileid', keep='first').head(4)
arms = {'clean': 'clean', 'noisy': 'noisy', 'y_pre': 'y_pre',
        'off': 'off', 'router': 'router'}
records = []
for _, row in selected.iterrows():
    fileid = int(row.fileid)
    pattern = f'*fileid_{fileid}.wav'
    # Existing listening_pack files are the same central 12 s for all arms.
    # Their time origin is 9 s in the original 30 s files.
    files = {
        arm: next((ROOT / 'listening_pack').glob(f'*_{arm}_*fileid_{fileid}_center12s.wav'), None)
        for arm in arms if arm != 'noisy'
    }
    files['noisy'] = next((ROOT / 'inputs/noisy').glob(f'*fileid_{fileid}.wav'), None)
    if any(v is None for v in files.values()):
        raise FileNotFoundError((fileid, files))
    start = max(0.0, float(row.start_s) - 9.0 - 0.75)
    end = min(12.0, float(row.end_s) - 9.0 + 0.75)
    for arm, src in files.items():
        audio, sr = sf.read(src, always_2d=False)
        a, b = int(round(start * sr)), int(round(end * sr))
        dst = CLIPS / f'fileid_{fileid}_{row.start_s:.2f}_{row.end_s:.2f}_{arm}.wav'
        sf.write(dst, audio[a:b], sr, subtype='PCM_16')
    records.append({
        'fileid': fileid, 'speaker': str(row.speaker), 'snr_db': float(row.snr_db),
        'candidate_window_s': [float(row.start_s), float(row.end_s)],
        'clip_window_s': [start, end], 'rank': int(row['rank']),
        'oracle_local_snr_db': float(row.oracle_local_snr_db),
        'noisy_clean_wave_corr': float(row.noisy_clean_wave_corr),
    })
(OUT / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
