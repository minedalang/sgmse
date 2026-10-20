# Common-scale factorial conditional diagnostic

Status: completed once in 25.28 seconds. Calls: 0 frontend (eight cached anchors reused), 16 OFF backend (all NFE30 and 30 traced steps), 16 new CTC. No training, downloads, Router or seed changes.

All four k-anchor/k-noisy cells exactly match prior raw arrays, NPY hashes and PCM hashes before the remaining twelve cells. All four groups share physical noise and one common reference across all four cells. Each anchor row holds its normalized anchor, seed, CPU/CUDA RNG states and initial xT across noisy donors. All 16 normalized condition peaks are <=1, without clipping or renormalization; zero PCM clipping.

Let q = LL(g)−LL(k). Subscripts below are anchor origin followed by noisy donor. Natural cells gg and kk have matching nominal identities; conflict cells gk and kg have distinct anchor and donor identities. Binary g/k selection comes from the sign of q.

| Speaker/noise | Common reference | gg: top1; q | gk: top1; q | kg: top1; q | kk: top1; q |
|---|---:|---|---|---|---|
| f6 N1_snr-6 | 0.210886478 | g; +4.50295 | g; +4.43180 | k; -5.55462 | k; -4.92392 |
| f6 N2_snr-6 | 0.226388648 | g; +4.00364 | g; +4.37145 | t; -3.17822 | k; -4.21559 |
| m5 N1_snr-6 | 0.172265649 | g; +4.37940 | dj; +3.46306 | k; -4.31909 | k; -4.48188 |
| m5 N2_snr-6 | 0.178400278 | g; +5.22865 | g; +5.21089 | k; -2.69678 | k; -2.42417 |

Every binary decision follows the anchor. Conflicting 24-class cells f6/N2 kg and m5/N1 gk select third classes t and dj; other conflict outputs follow the anchor. These are identity-following descriptions, with no cross accuracy endpoint.

Descriptive factorial contrasts: anchor = ((qgg−qkg)+(qgk−qkk))/2; condition = ((qgg−qgk)+(qkg−qkk))/2; interaction = (qgg−qgk)−(qkg−qkk).

| Speaker/noise | Anchor effect | Condition effect | Interaction |
|---|---:|---:|---:|
| f6 N1_snr-6 | +9.70665 | -0.27978 | +0.70185 |
| f6 N2_snr-6 | +7.88445 | +0.33478 | -1.40517 |
| m5 N1_snr-6 | +8.32172 | +0.53957 | +0.75354 |
| m5 N2_snr-6 | +7.78024 | -0.12743 | +0.29036 |

Anchor contrasts are positive in each group. Condition contrasts vary in sign across noises. The common-scale results preserve the two earlier third-class conflict outputs while removing condition peaks above one. Jointly conflicting anchor/noisy identities remain OOD.

Within-row condition contrasts q(noisy g)−q(noisy k):

| Speaker/noise | g anchor | k anchor |
|---|---:|---:|
| f6 N1_snr-6 | +0.07114 | -0.63071 |
| f6 N2_snr-6 | -0.36781 | +1.03736 |
| m5 N1_snr-6 | +0.91634 | +0.16280 |
| m5 N2_snr-6 | +0.01776 | -0.27261 |

Independent validation passed: 16 raw/WAV pairs including PCM re-encoding, all 384 cached CTC candidate likelihoods recomputed from saved float logits (maximum error 0), probability caches, 158 source/config/driver locks, four exact native k gates, four physical-noise/common-scale groups, eight held-row geometry checks and all twelve factorial contrasts. The independent NumPy peak arithmetic agrees with CUDA normalization within one float32 epsilon; recorded CUDA peaks themselves are all <=1. No extra model forwards.

Eight cached frontend raw hashes and original native provenance remain unchanged. The 43-line helper SHA256 remains 66f61a892e5b4ffc37c1069e3692618a07fb09efc3717c106c1d18c212ddbb82. Driver SHA256 is 0d3f1b265b1c65fc91a1af9381a1e0cf4ac9cf26b84d508595c2c7f5a6114cc3. Environment/times and git state are in execution_environment.json; saved validation is outputs/independent_validation.json.

Prior native-reference comparisons remain in metrics.json with their differing scales. The changed g/g reference does not require exact reproduction; all four g/g outputs change raw arrays from their own-reference native outputs. k/k retains the exact original scale and raw output.

Limits: Only the same selected f6/m5 whole VCV recordings, including duration and coarticulation, and two frozen noise carriers were tested. This is not a surgical single-phoneme intervention, enhancement method, human recovery result, true PER measurement or population estimate. Conflicting cells have no uniquely correct enhancement identity; no correctness flags or accuracy aggregates are assigned. No further normalization switch or design iteration follows this experiment, regardless of outcome.

Source: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0.
# Research Lead assessment and stop decision

The common-scale control removes the previous donor-peak-above-one issue,
but the binary identity pattern and two third-class outcomes remain. In these
selected cases the anchor-origin contrast in output CTC log-likelihood is
much larger than the noisy-donor contrast, whose direction is inconsistent.
This is a bounded recognizer response, not a general physical effect size or
proof that the backbone ignores useful information in natural inputs.

The raw anchor may retain identity cues missed by its own CTC decoding;
learned priors and conflicting-condition distribution shift remain alternative
explanations. Do not equate nominal anchor origin with an already correctly
decoded frontend, or score contradictory cells as having one verified target.

Stop normalization/seed experiments on these two k/g pairs now, regardless
of whether a later score could be made favorable. The next separate design
uses other natural correction/damage cases and removes the known speech
component while retaining exactly the same noise, anchor and scale. It asks
about evidence necessity rather than testing further contradictory identities.
It remains an oracle diagnostic and does not trigger training.

