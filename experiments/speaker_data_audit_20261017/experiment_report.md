# Speaker data audit (2026-10-17)

This is a read-only audit. It did not modify the DNS dataset, source code, or checkpoints.

## Findings

- `tr_synthetic`: 10,800 utterances, 1,470 speakers, no missing `utt2spk` entries, and no mismatch between the `reader_xxxx` parsed from the utterance ID and `utt2spk`.
- 1,303 training speakers have at least two utterances; 167 are singletons.
- `cv_synthetic`: 1,200 utterances, 158 speakers, no missing IDs, and 141 speakers have at least two utterances.
- Train/CV speaker overlap is exactly one speaker: `reader_10658`. The prior speaker-disjoint 32-utterance diagnostic excluded it.
- `tt_synthetic_no_reverb` has 150 unique `fileid_*` speaker labels and no cross-utterance pairs. `tt_real_recordings` has 300 unique recording IDs and no cross-utterance pairs. These test splits are not suitable for selecting a cross-utterance speaker loss.

## Batch sampler check

Using the training `utt2spk` IDs with the repository's `SpeakerBatchSampler(batch_size=8, min_per_speaker=2, drop_last=True)`, 1,250 batches were formed. All batches had at least two utterances per speaker; compositions were `(2,2,2,2)` for 1,246 batches and `(2,2,4)` for 4 batches. No invalid batch was observed.

## Interpretation

The data supports a valid cross-utterance speaker experiment on train/CV. The previous negative 1,000-step masked result is therefore not explained by absent IDs or a broken sampler. A future speaker experiment should use a speaker-disjoint CV split and a checkpoint budget matched to the 80k Router, with thresholds fixed before touching the final test set.
