# Corrected fixed-time CTC probe, v2 (prepared 2026-10-18)

Status: completed by the delegated Experiment Worker on 2026-10-18. Actual
execution timestamps are recorded in metrics.json. Historical
source directory IDs containing 20261020 are preserved as input identifiers.

The original probe converted seconds to frames using floor/ceil. Floating point
rounding expanded 10 of 89 token intervals, affecting 50 of 445 saved rows.
This version uses the source integer half-open token intervals directly and
the integer neighborhood `[max(0,start-10), min(num_frames,end+10))`.
The neighborhood correction also avoids seconds conversion roundoff.

The model, 20 clips, clean-derived token sequence, seed, and five conditions
follow the original protocol. Float32 probabilities across all frames for every clean-target token ID,
the token-ID column map, and full-vocabulary frame argmax IDs are cached
in a compressed NPZ smaller than 1 MB. These arrays preserve every reported
metric for offline recomputation; probabilities for other token classes
and logits are omitted. The manifest in metrics.json maps array keys to clips;
the vocabulary uses the original local vocab.json,
whose hash is recorded. The cache supports subsequent window analyses without
inference. No source audio, datasets, or checkpoints are modified.

From the repository root, the delegated worker runs:

```bash
env/bin/python experiments/p8_q_a_ctc_fixed_time_v2_20261018/fixed_time_probe.py
python3 experiments/p8_q_a_ctc_fixed_time_probe_20261020/summary.py \
  --input experiments/p8_q_a_ctc_fixed_time_v2_20261018/fixed_time_token_posteriors.csv \
  --output experiments/p8_q_a_ctc_fixed_time_v2_20261018/summary.json
```

The probe refuses existing result artifacts, including the start marker left
by an interrupted execution. Diagnose failures before arranging another run.
The summary also refuses an existing output file. It provides macro group
means, pooled token means, per-group values and paired differences; no
confidence intervals. Its indexing audit explicitly distinguishes the
original seconds conversion from the actual saved CSV window lengths.

## Results

CUDA forward time was approximately 1.40 s for the same 20 two-second clips.
The CSV contains 445 condition rows / 89 clean-derived targets. Integer
interval length mismatches are zero. All 20 input hashes and the model weight
hash match v1; clean fixed-window posterior agrees with source clean-run
posterior to a maximum absolute error of 4.77e-7. Cached probabilities occupy
136,188 bytes. Source audio, datasets and checkpoints were not modified.
The immutable configuration captures the pre-execution protocol (its
`prepared_not_run` status describes preparation); metrics.json records the
completed run. The configuration hash is retained unchanged.

Primary summaries average token posterior within each group, then give each
eligible group equal weight. Pooled token means in summary.json give a
different weighting; neither is an accuracy or percentage of recovered
phonemes. The four groups were repeatedly inspected development examples,
not an independent confirmation set.

| Subset | targets | eligible groups | Clean | Noisy | Y_pre | Router-off | Router |
|---|---:|---:|---:|---:|---:|---:|---:|
| All tokens | 89 | 4 | 0.900882 | 0.017640 | 0.116839 | 0.101753 | 0.108031 |
| IPA consonants | 56 | 4 | 0.935309 | 0.028557 | 0.141641 | 0.111785 | 0.122643 |
| k,d combined | 7 | 4 | 0.972219 | 0.003813 | 0.003321 | 0.108883 | 0.109680 |

Paired group differences in fixed-time posterior:

| group/file-id | all: off−Y_pre | consonants: off−Y_pre | k,d: off−Y_pre | all: Router−off |
|---|---:|---:|---:|---:|
| 5243 | −0.127192 | −0.126805 | +0.425293 | +0.027813 |
| 4812 | +0.002209 | −0.044939 | −0.000053 | −0.003251 |
| 11999 | +0.033786 | +0.052241 | −0.001335 | +0.000429 |
| 10720 | +0.030851 | +0.000080 | −0.001655 | +0.000123 |
| Equal-group mean | −0.015086 | −0.029856 | +0.105562 | +0.006278 |

The k,d aggregate is dominated by one d pseudo-target in file-id 5243:
fixed-time posterior is Y_pre 0.018755, off 0.869617, Router 0.875637.
The other three groups do not show the same k,d improvement. This candidate
does not establish recovery from noisy: the target identity comes from the
same recognizer, noisy posterior is low, and model prior/context effects have
not been separated from input-specific evidence.

The +/-200 ms maximum also exposes timing sensitivity. For the 5243 k
pseudo-target, Y_pre is only 0.000294 at the clean frame, but 0.985447 in the
neighborhood. For one 10720 d pseudo-target, off is about 0.000003 at the
clean interval but 0.250458 nearby (Router about 0.000005 / 0.379175).
A nearby high value can reflect a shifted spike or another occurrence;
conversely a low fixed-time value alone does not prove phoneme deletion.

## Research decision

The sequence-forced and fixed-time probes answer different questions and
rank the frontend/backend differently. Neither validates a generally superior
content-recovery mechanism. Low noisy posterior is a limitation of this
recognizer under these conditions, not proof that the original mixture is
unidentifiable. Clean-derived labels remain useful candidate labels; this
experiment alone does not validate their identity or calibration.

Stop further alignment/window tuning on these four examples. Keep the 5243 d
as an exploratory candidate, alongside negative and timing-sensitive examples.
Do not start content-loss training, P8-N/P8-R, or official-test tuning on this
basis. Next investigate a small, legally available independently annotated
speech resource or independently verified events to calibrate the diagnostic;
this data-source audit does not require new training or bulk inference.

## Subsequent listener confirmation: file-id 5243 d

On 2026-10-18 the user listened to the first group of the corrected listening
page and explicitly reported that Y_pre loses d while the backend retains it.
This is a positive single-listener, prompted, nonblind event observation and
limits the earlier global description of backend processing as smoothing only.
The user did not separately score off and Router in this feedback; both have
high automatic posterior. It does not establish a Router-specific effect.

The 2 s clip covers original 18.75–20.75 s. Target index11 d is at clean
CTC frames [35,36), approximately clip0.70–0.72 s / original19.45–19.47 s.
It is in the preceding context, outside the named central candidate interval
19.50–20.00 s. Frame coordinates are not manually verified phoneme boundaries.
Do not conflate this observation with recovery inside that central window.

The observation supports an audible frontend/backend content difference at a
specific location. Distinguishing noisy acoustic evidence, residual frontend
cues and contextual generation remains open. Detailed verbatim feedback is
preserved in `experiments/p8_frontend_backend_audit_20261020/user_listening_observation.md`.
