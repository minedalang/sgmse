# 10720 tail causal pilot

## Research Lead interpretation

At fixed frontend and common random numbers, replacing only the local noisy
condition reduces tail energy much more than replacing only the frontend
anchor. Input changes are exactly confined to the tail and preserve the global
normalization peak. This supports sensitivity of the extra tail energy to the
observed-condition path in this particular example. It does not prove a
semantic conversion from noise to a phoneme, and does not exclude nonlocal
context or distribution effects. The condition is changed away from its
normal joint relationship with the original frontend, using clean reference;
this is an oracle diagnostic, not a deployable suppression mechanism.

The same direction is replicated within the utterance at20dB noise reduction
in `../p8_10720_tail_condition20_20261018/`: raw tail RMS−29.372dB and
4–7.5kHz energy−37.573dB. This second condition is not an independent
utterance or a listener confirmation of the intervened outputs. Only the
original control outputs received the user's s-like/voice-like judgments.

Seed controls confirm that the sampled waveform is variable, but their two
draws neither identify an artifact rate nor justify an ensemble controller.
ArtiFree already studies seed/embedding consistency; PASE already studies
phonological priors. Project-specific contribution requires independent
mechanism evidence and preservation of genuine weak consonants; neither a
silence gate nor a feature loss is established as a new method by this pilot.

Status: completed

Oracle clean-tail intervention is an OOD mechanism probe, not a deployable method. Seeds diagnose stochastic risk, not ensemble selection or novelty. Prior P8-M 5243 active masking negative is preserved and is a distinct experiment.

Window: samples240800:252000, 15.05–15.75s, internal20ms raised cosine fades. Model loaded once; original FLOAT frontend recomputed once and hash checked. All six runs are backend off, NFE30. Full speech activity uses clean20ms disjoint frames above−70dBFS. Raw metrics compare with new no_op raw; PCM16 values are reference only. frames.csv retains raw fixed frame values. No clean/noisy corpus changes or downloads.

|Run|Tail RMS dBFS|4–7.5kHz mean square|Outside relative L2|Active relative L2 vs no_op|
|---|---:|---:|---:|---:|
|no_op|-59.86049|6.7303204e-07|0|0|
|condition_tail_clean|-85.51716|2.1769572e-11|0.013640146|0.0085752038|
|anchor_tail_clean|-68.42637|7.4148065e-08|0.0095540024|0.0079687633|
|both_tail_clean|-88.84889|1.8939127e-11|0.016698343|0.010243593|
|baseline_seed_plus1|-65.66020|2.1299347e-07|0.32728481|0.32245322|
|baseline_seed_plus2|-62.31304|2.3576037e-07|0.33124752|0.32242659|

## Independent cached artifact verification

Verification used existing arrays only, with no model load or repeated inference. Input/code/checkpoint/config/driver and raw/WAV hashes matched. The saved frontend hash and no_op exact PCM16 baseline gate passed. Cached anchor/condition interventions reconstruct exactly, preserve all input samples outside the window, and preserve the noisy global normalization peak. Every recorded NFE is30. Independent SciPy boxcar periodogram integration and raw RMS/RMSE reproduce reported energies.

The first strict comparison found a small L2 precision discrepancy: the driver uses a float32 reference norm denominator; independent float64 norms differ by at most 1.8639874e-05 relative. All recorded L2 values reproduce at the original precision. The table below reports independent float64-denominator L2 values; metrics.json remains the original execution record. This is a numerical reporting limit, not an inference failure.

| Run | Tail RMS | RMS delta dB | Highband delta dB | Outside L2 | Active L2 vs no_op | Active L2 vs clean | Active RMSE vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_op | 0.00101619127 | 0.000 | 0.000 | 0.000000 | 0.000000 | 0.399892 | 0.00148517572 |
| condition_tail_clean | 5.29836747e-05 | -25.657 | -44.902 | 0.013640 | 0.008575 | 0.399578 | 0.00148400753 |
| anchor_tail_clean | 0.000379036934 | -8.566 | -9.579 | 0.009554 | 0.007969 | 0.399824 | 0.00148492157 |
| both_tail_clean | 3.61040308e-05 | -28.988 | -45.507 | 0.016698 | 0.010244 | 0.399684 | 0.00148440065 |
| baseline_seed_plus1 | 0.000521182829 | -5.800 | -4.997 | 0.327283 | 0.322453 | 0.400240 | 0.00148646727 |
| baseline_seed_plus2 | 0.000766210505 | -2.453 | -4.556 | 0.331246 | 0.322426 | 0.403058 | 0.00149693397 |

The baseline seeds span −65.660 to −59.860 dBFS tail RMS. The two additional seed runs provide a small stochastic contrast, not a precise variance estimate. Input changes are confined to the intervention window; output differences outside it remain possible through the model. Raw no_op is the comparator for all causal and seed deltas. These energy changes are not speech/phoneme truth, evidence of deployment performance, or a novelty claim. Prior negatives are preserved.
