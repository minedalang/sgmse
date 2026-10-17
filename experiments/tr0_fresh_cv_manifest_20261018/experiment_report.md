# TR-0 fresh CV manifest and proxy sanity — 2026-10-18

## Status

Completed as a read-only preparation step. No training, inference, checkpoint write, or dataset modification was performed.

## Fresh CV manifest

The manifest is drawn from `dns_snr_-20_-5/scp_new/cv_synthetic`. It deterministically excludes every speaker in the existing 16-speaker `rrsd_cv_repeat32` manifest, every speaker in the existing 64-speaker `speaker_headroom_replication_64_20261017` manifest, and `reader_10658`. From the remaining speakers it selects the first 32 speaker IDs in sorted order and the first two sorted utterance IDs per speaker.

The resulting lock contains 32 speakers and 64 utterances. `manifest_lock.json` records the metadata SCP hashes, selected/excluded speakers, and the SHA-256 of `manifest.csv`. Audio files are referenced in place; they are not copied or changed.

## Proxy sanity using existing 64-speaker output summary

The only complete matched summary available for the existing 64-speaker output set is `speaker_backend_only_matched_500_20261018/control_eval_64/quality_metrics.json`. It reports aggregate PESQ/STOI/SI-SDR and DNSMOS, but not per-utterance DNSMOS. Therefore this is a warning check, not a proxy validation experiment.

For the original Router relative to off, the aggregate deltas are:

| Metric | Router − off |
|---|---:|
| PESQ-WB | +0.010925 |
| STOI | +0.001146 |
| SI-SDR | +0.511487 dB |
| DNSMOS-SIG | −0.030868 |
| DNSMOS-OVRL | −0.029837 |

The directions conflict. No single reconstruction metric is safe as a stand-in for perceptual quality. The adapter screen must therefore retain DNSMOS/PESQ/SI-SDR/STOI as separate endpoints and must not tune a training proxy against the old 64-speaker summary.

The matched control itself is not used to claim an adapter effect: it is a no-loss backend-only control and is already recorded separately. This sanity step only determines the evaluation requirement for TR-0.

## Decision

The fresh CV manifest is ready for a later, independently configured smoke. Before any adapter optimization, the next required check is an exact no-op adapter forward comparison. No long training is authorized by this artifact; the planned screen remains short and must preserve the original 80k checkpoint.
