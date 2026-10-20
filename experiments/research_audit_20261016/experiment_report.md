# Research audit: updated plan

## Confirmed code/data facts

- Train SCP: 10,800 utterances, 1,470 parsed `reader_` IDs; 1,303 speakers have at least two utterances.
- CV SCP: 1,200 utterances, 158 parsed IDs; 141 speakers have at least two utterances.
- Train/CV speaker overlap is one ID (`reader_10658`), so the existing CV is not fully speaker-disjoint.
- Synthetic test SCPs expose no `reader_` IDs; real-recording test has no `spk1.scp`. Test-set cross-utterance speaker evaluation is therefore unsupported without external metadata.
- `WavePairsScp` parses `reader_(\d+)` from the utterance ID and assigns unique negative IDs when parsing fails. The parser is valid for train/CV synthetic data.
- `SpeakerBatchSampler` does create same-speaker groups of two, so cross-utterance samples are available in training.
- Current paired InfoNCE keeps same-speaker off-diagonal items in the denominator, creating false negatives; the prototype branch separately uses other same-speaker clean embeddings. This is an important ablation risk.
- Existing speaker runs used UniSpeech-SAT-SV with `cw=10, sw=0` and `cw=10 or 1, sw=0.1`; prior metrics did not establish an independent speaker benefit.

## A: Harm Teacher

The existing finite-difference records contain 86 rows, 64 with selected conflict tiles. On conflict rows, increasing alpha satisfied the expected Future-down/WavLM-up sign in 71.9% of rows; decreasing alpha satisfied the reverse sign in 82.8%; both directions agreed in 59.4%. The low-t bidirectional rate was 71.4%, versus 50.0% in the mid-t bin. This supports a local harm signal as a falsifiable teacher candidate, but does not prove DNSMOS harm.

The worker's oracle script passed syntax/help checks but its first smoke did not reach model execution: it needs the environment's `ninja` executable on PATH and deterministic clean/noisy fileid mapping. No training or full inference was started.

## Missing fair controls

1. The 80k Router and 300k `5ul7u03z` baseline have unequal training budgets.
2. The complete 150-file matched off/full/selective triplet is not currently archived with paired audio from the same 80k checkpoint.
3. Existing test audio has no speaker IDs, so B's final test identity claim cannot be made from this dataset alone.

## Recommended low-cost order

1. Fix the oracle diagnostic mapping and PATH, run 4 CV files at N=4, then only if signs are finite and separable run N=30 on the same 32-file CV subset.
2. Run an independent SpeechBrain XVector/CAM++ probe on existing 32-file outputs, reporting same/different-reader pairs and PESQ/STOI/SI-SDR together.
3. Generate the missing matched full off/full/selective triplet once, in separate directories, before tuning any threshold.

All outputs are recorded in `metrics.json`; no dataset files or checkpoints were modified.

## A oracle update (2026-10-16)

The CUDA oracle diagnostic now runs. A four-file N=30 comparison used the same 80k EMA checkpoint, seed, and sampler. Full Router had 0/120 harm decisions by construction; the WavLM hard gate marked 19/120 steps (15.8%). Gate minus full was PESQ +0.000158 (95% CI [-0.000006,+0.000388]), STOI +0.000007 ([-0.000005,+0.000025]), and SI-SDR -0.000018 dB ([-0.000053,+0.000011]). This is not evidence of a useful quality gate, so a deployable Harm Head is deferred.

## B independent speaker probe (2026-10-16)

A frozen local WeSpeaker CAM++ encoder was applied to the existing 32-utterance outputs. Mean enhanced-to-clean cosine was 0.88595 (off), 0.88089 (original Router), 0.88289 (speaker pair), 0.88286 (proto01), and 0.88176 (proto01-cw1). The manifest contains only two same-reader cross-utterance pairs versus 494 different-reader pairs, so this cannot establish a speaker-verification gain. The full probe is recorded in `experiments/rrsd_independent_speaker_probe_32/`; a speaker-disjoint multi-utterance evaluation remains necessary before treating B as an independent contribution.

## B repeated-speaker CV diagnostic (2026-10-16)

A 16-speaker/32-utterance CV subset with two utterances per speaker (excluding `reader_10658`) was evaluated with independent CAM++. There were 16 same-speaker and 480 different-speaker enhanced-enhanced pairs. None of pair/prototype/prototype-cw1 exceeded the 80k Router on same-speaker cosine; their SI-SDR deltas versus Router were -0.124, -0.126, and -0.054 dB respectively. This rejects the current 1,000-step speaker variants as an independent gain. Because the speaker checkpoints and Router have unequal training budgets, a universal rejection of the speaker idea would require a matched-budget run; no long run is justified yet.

## False-negative mask implementation check (2026-10-17)

The paired InfoNCE implementation now has an opt-in `--contrastive_mask_same_speaker_negatives` flag. Default behavior is unchanged. Unit checks passed for legacy loss equivalence, masked loss activation, unknown-ID behavior, enhanced-input gradients, and detached clean targets. A 1,000-step masked run completed successfully; an 8-file/4-speaker diagnostic showed +0.000596 CAM++ cosine but -0.122 dB SI-SDR versus Router, so the fix alone is not evidence of a useful B improvement.

## Masked same-speaker negative full CV result (2026-10-17)

The masked 1,000-step checkpoint was evaluated on all 32 speaker-disjoint CV utterances under the existing matched inference settings (CUDA, EMA, RRSD, N=30, SNR=0.5, seed 2026). Enhanced-clean CAM++ increased by `+0.001613` versus Router, but the independent same-speaker cross-utterance mean was `0.840694`, below Router's `0.842322`. PESQ changed by `-0.001500` (95% CI `[-0.006685,+0.003938]`), STOI by `-0.000300` (`[-0.000553,-0.000071]`), and SI-SDR by `-0.144008 dB` (`[-0.163897,-0.124513]`). This does not support an independent Speaker–Content gain; stop the current short-run route unless a matched-budget experiment is explicitly justified by a new hypothesis.

## Conflict-aware matched control (2026-10-17)

The 1,000-step `conflict_weight=0.001` run was compared with a `conflict_weight=0` control initialized from the same 80k Router and trained for the same steps. On the same 150-file test manifest, conflict minus control was P808 `+0.002458` (95% CI `[+0.001406,+0.003565]`), OVRL `+0.000410` (`[-0.000909,+0.001714]`), SIG `+0.000341` (`[-0.000837,+0.001524]`), PESQ `-0.000067` (`[-0.002459,+0.002170]`), STOI `-0.000061` (`[-0.000094,-0.000027]`), and SI-SDR `-0.022527 dB` (`[-0.024848,-0.020325]`). The conflict term therefore has no reliable overall perceptual benefit beyond ordinary 1,000-step adaptation and should not be extended to long training without a new hypothesis.

## Fixed-scale Pareto screen (2026-10-17)

On the same 32-file paired manifest, fixed `rrsd_scale=0.5` improved DNSMOS-OVRL by `+0.011710` but reduced SI-SDR by `0.231586 dB`; fixed `rrsd_scale=0.25` improved OVRL by `+0.017246` but reduced SI-SDR by `0.369448 dB`. Neither fixed multiplier resolves the trade-off, so global scale tuning is closed as a baseline. A useful next A method must use observable state/time-dependent gating and be evaluated against these fixed-scale references.

## Full-test t_cut=0.05 candidate (2026-10-17)

A fixed low-noise cutoff was evaluated on all 150 test files using the same 80k EMA Router and matched inference settings. Relative to Router, `t_cut=0.05` improved DNSMOS-P808 by `+0.005188`, SIG by `+0.003971`, and OVRL by `+0.003873` (95% CI `[+0.001864,+0.005916]`), while changing SI-SDR by `-0.049719 dB` (CI `[-0.053556,-0.046117]`). PESQ and STOI declined slightly. This is the strongest current Pareto point, but it is a hand-designed rule and requires held-out-split validation before being treated as a general innovation.
