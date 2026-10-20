"""Rank P8-E1 candidate windows from existing P8 local diagnostics.

This is a read-only analysis.  The join key is fileid/start_s/end_s; the
CSV `index` column is only a per-file/window index and is not globally unique.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "p8_local_content_probe_20261020"
OUT = Path(__file__).resolve().parent
KEYS = ["fileid", "start_s", "end_s"]

oracle = pd.read_csv(ROOT / "oracle_snr_windows.csv")
cue = pd.read_csv(ROOT / "noisy_cue_windows.csv")[
    KEYS + ["noisy_clean_wave_corr", "noisy_clean_spec_cos_window"]
]
joined = oracle.merge(cue, on=KEYS, validate="one_to_one")
suppressed = joined[joined["candidate_frontend_suppression"]].copy()

# Prefer higher noisy-clean cue among candidate suppression windows, while
# retaining all clean-aware fields for later human verification.
suppressed["cue_rank_score"] = (
    suppressed["noisy_clean_wave_corr"]
    + 0.25 * suppressed["noisy_clean_spec_cos_window"]
)
ranked = suppressed.sort_values(
    ["cue_rank_score", "noisy_clean_wave_corr", "oracle_local_snr_db"],
    ascending=False,
).reset_index(drop=True)
ranked.insert(0, "rank", ranked.index + 1)

cols = [
    "rank", "fileid", "speaker", "snr_db", "start_s", "end_s",
    "active_fraction", "oracle_local_snr_db", "noisy_clean_wave_corr",
    "noisy_clean_spec_cos_window", "y_pre_rms_db_vs_clean",
    "off_gain_vs_y_pre_db", "candidate_backend_energy_fill",
    "cue_rank_score",
]
ranked[cols].to_csv(OUT / "candidate_windows.csv", index=False)

summary = {
    "source": str(ROOT),
    "join_keys": KEYS,
    "source_rows": int(len(joined)),
    "suppression_rows": int(len(suppressed)),
    "median_suppressed": {
        c: float(suppressed[c].median())
        for c in ["oracle_local_snr_db", "noisy_clean_wave_corr", "noisy_clean_spec_cos_window", "off_gain_vs_y_pre_db"]
    },
    "median_non_suppressed": {
        c: float(joined.loc[~joined["candidate_frontend_suppression"], c].median())
        for c in ["oracle_local_snr_db", "noisy_clean_wave_corr", "noisy_clean_spec_cos_window", "off_gain_vs_y_pre_db"]
    },
    "top10": ranked[cols].head(10).to_dict(orient="records"),
    "interpretation": "candidate ranking only; no phoneme or recoverability claim",
}
(OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
