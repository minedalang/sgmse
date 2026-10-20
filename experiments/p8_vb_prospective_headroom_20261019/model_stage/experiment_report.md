# Prospective fixed-candidate headroom: completed unique scientific run

## Execution and integrity

One native run (no additional smoke/retry), session **33718**, observed supervisor PID **14679**, child **14680**, STARTED **2026-10-19T06:47:45.930133+00:00**, supervisor elapsed **97.644696386 s**, exit 0. Frozen180s total/170s execution reserve10s/64MiB parent respected. Exact command:

`/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/model_stage/run_pilot.py`

One prepared independent CPU validator, session **50900**, exit0/passed. Exact command: `/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/model_stage/validate_model_run.py`. It verified 64 serialization outputs,104 scores,32 native cases,21 headroom strata and10000 exploratory bootstrap draws. SI-SDR independently recomputed; STOI/PESQ reuse the same frozen official metric implementations with independently implemented bookkeeping, not independent published algorithms. Validator elapsed was not separately measured. No neural model/ASR loaded by validator.

All413 frozen locks pre/post/supervisor/current and 368 original runtime artifacts hash/bytes matched. Source data, original metadata/CPU registry/metrics, old VB/B/failure records/checkpoints and dirty git outside current experiment unchanged. Original model metrics SHA256 `577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04`; input registry `cfc2e1b4ac313037f7e324b6e548f0f56cb1ee32a80282475a0823c7a876d998`. Original metrics preserved, not rewritten for report. Bridge/listening processes checked alive after execution; no new listening or human judgments.

32 frontend and32 OFF calls, each NFE30; one enhancement-model load and one ASR-model load after enhancement weakrefs released. 104 transcribe API calls and104 internal decode API calls; **neural forward count not measured, API count is not forward count**. One strictly identified refused local urllib3 IPv6 probe, zero unknown refusals, zero actual network bytes/training/test access. All socket creation remained refused. Project input/enhancement/ASR pipeline resampling0; STOI internally resamples to10k/removes silence, PESQ jointly peak-scales and performs standard internal processing. No added gain/lag fit, threshold or seed search.

## Primary results and essential confounds

All cases retained. Word endpoint single-tiny.en ASR relative to publisher sentence text, **not human content truth or actual phoneme recovery**. Uncapped WER is required, including values above100%. Fixed front/OFF pool only; oracle chooses entire candidate by lower word edits (ties front), not word splicing or deployable selector. 32 conditions comprise8 sentences/4 speakers/2 reused noise recordings; not32 independent utterances. Local16k waveform/publisher-ID association does not prove realized phones, spoken alignment or conversion equivalence; prior2LSB source gate remains failed. Pretraining/external/anonymized exposure unknown.

Primary all: frontend E378, OFF E229, offline oracle E159 over N288; best fixed OFF, headroom70/288=**24.305556 percentage points**. However frontend wins23 cases, OFF wins4, ties5. **One frontend ASR repetition dominates error sums**: p227_030_N1_snr_m18 has222 edits,216 insertions; its literal repetitive hypothesis remains below and in raw JSON. Thus pooled OFF advantage/oracle headroom must not be described as broad real-content restoration. Most cases favor frontend, and three of four speaker totals favor frontend. No posthoc exclusion, WER clipping, replacement, or recalculated favorable endpoint performed.

Scaled-clean controls:72 reference words,2 substitutions/0 deletions/0 insertions (WER2.777778%); character3 edits/312 (0.961538%). Nonzero controls p230_027 and p231_032 are kept as ASR/reference adaptation confounds, not used to change IDs or reference. These controls do not certify all other labels as spoken truth.

Exploratory speaker-cluster bootstrap seed20261019,10000 draws,4 clusters: OFF−front microWER interval **[-253.278689,35.563380]pp**, headroom **[0.373134,29.347826]pp**. Four clusters and outlier sensitivity do not support reliable nominal coverage or confirmatory significance; intervals are descriptive, not an upgrade/train gate.

## All21 strata: WER and S/D/I

| Stratum | N | noisy WER;S/D/I | frontend WER;S/D/I | OFF WER;S/D/I | oracle WER;S/D/I | headroom pp | front/OFF/tie case wins |
|---|---:|---|---|---|---|---:|---|
| all | 288 | 64.236%; 50/125/10 | 131.250%; 115/35/228 | 79.514%; 172/1/56 | 55.208%; 111/35/13 | 24.305556 | 23/4/5 |
| noise_N1 | 144 | 59.722%; 14/69/3 | 207.639%; 48/33/218 | 81.944%; 90/1/27 | 56.250%; 45/33/3 | 25.694444 | 11/3/2 |
| noise_N2 | 144 | 68.750%; 36/56/7 | 54.861%; 67/2/10 | 77.083%; 82/0/29 | 54.167%; 66/2/10 | 0.694444 | 12/1/3 |
| snr_-18 | 144 | 100.000%; 21/123/0 | 247.917%; 95/34/228 | 127.778%; 135/0/49 | 97.917%; 94/34/13 | 29.861111 | 14/2/0 |
| snr_0 | 144 | 28.472%; 29/2/10 | 14.583%; 20/1/0 | 31.250%; 37/1/7 | 12.500%; 17/1/0 | 2.083333 | 9/2/5 |
| N1_-18 | 72 | 100.000%; 5/67/0 | 397.222%; 36/32/218 | 129.167%; 71/0/22 | 98.611%; 36/32/3 | 30.555556 | 7/1/0 |
| N1_0 | 72 | 19.444%; 9/2/3 | 18.056%; 12/1/0 | 34.722%; 19/1/5 | 13.889%; 9/1/0 | 4.166667 | 4/2/2 |
| N2_-18 | 72 | 100.000%; 16/56/0 | 98.611%; 59/2/10 | 126.389%; 64/0/27 | 97.222%; 58/2/10 | 1.388889 | 7/1/0 |
| N2_0 | 72 | 37.500%; 20/0/7 | 11.111%; 8/0/0 | 27.778%; 18/0/2 | 11.111%; 8/0/0 | 0.000000 | 5/0/3 |
| speaker_p227 | 56 | 57.143%; 8/24/0 | 450.000%; 27/5/220 | 73.214%; 31/1/9 | 62.500%; 25/5/5 | 10.714286 | 4/2/2 |
| speaker_p228 | 92 | 57.609%; 9/42/2 | 47.826%; 26/17/1 | 69.565%; 52/0/12 | 46.739%; 25/17/1 | 1.086957 | 6/1/1 |
| speaker_p230 | 64 | 65.625%; 10/28/4 | 59.375%; 30/3/5 | 101.562%; 42/0/23 | 59.375%; 30/3/5 | 0.000000 | 8/0/0 |
| speaker_p231 | 76 | 76.316%; 23/31/4 | 57.895%; 32/10/2 | 77.632%; 47/0/12 | 56.579%; 31/10/2 | 1.315789 | 5/1/2 |
| utterance_p227_030 | 24 | 58.333%; 4/10/0 | 962.500%; 12/0/219 | 75.000%; 13/0/5 | 66.667%; 12/0/4 | 8.333333 | 1/1/2 |
| utterance_p227_031 | 32 | 56.250%; 4/14/0 | 65.625%; 15/5/1 | 71.875%; 18/1/4 | 59.375%; 13/5/1 | 6.250000 | 3/1/0 |
| utterance_p228_026 | 48 | 62.500%; 6/22/2 | 45.833%; 13/9/0 | 68.750%; 29/0/4 | 45.833%; 13/9/0 | 0.000000 | 3/0/1 |
| utterance_p228_032 | 44 | 52.273%; 3/20/0 | 50.000%; 13/8/1 | 70.455%; 23/0/8 | 47.727%; 12/8/1 | 2.272727 | 3/1/0 |
| utterance_p230_027 | 28 | 75.000%; 5/12/4 | 53.571%; 9/3/3 | 89.286%; 15/0/10 | 53.571%; 9/3/3 | 0.000000 | 4/0/0 |
| utterance_p230_029 | 36 | 58.333%; 5/16/0 | 63.889%; 21/0/2 | 111.111%; 27/0/13 | 63.889%; 21/0/2 | 0.000000 | 4/0/0 |
| utterance_p231_029 | 32 | 81.250%; 12/11/3 | 50.000%; 10/5/1 | 68.750%; 15/0/7 | 50.000%; 10/5/1 | 0.000000 | 2/0/2 |
| utterance_p231_032 | 44 | 72.727%; 11/20/1 | 63.636%; 22/5/1 | 84.091%; 32/0/5 | 61.364%; 21/5/1 | 2.272727 | 3/1/0 |

## CER and uncapped sentence means

| Stratum | Arm | CER micro | CER sentence mean | WER sentence mean | S/D/I characters |
|---|---|---:|---:|---:|---|
| all | noisy | 0.559295 | 0.562782 | 0.645833 | 120/561/17 |
| all | frontend | 0.927885 | 0.954671 | 1.693819 | 275/215/668 |
| all | off | 0.560096 | 0.572658 | 0.799147 | 502/37/160 |
| all | WER_oracle | 0.423077 | 0.423330 | 0.560533 | 283/211/34 |
| noise_N1 | noisy | 0.551282 | 0.553198 | 0.598519 | 32/306/6 |
| noise_N1 | frontend | 1.471154 | 1.529927 | 2.836118 | 110/170/638 |
| noise_N1 | off | 0.581731 | 0.595270 | 0.809084 | 268/22/73 |
| noise_N1 | WER_oracle | 0.471154 | 0.476172 | 0.575228 | 117/168/9 |
| noise_N2 | noisy | 0.567308 | 0.572366 | 0.693148 | 88/255/11 |
| noise_N2 | frontend | 0.384615 | 0.379416 | 0.551520 | 165/45/30 |
| noise_N2 | off | 0.538462 | 0.550046 | 0.789209 | 234/15/87 |
| noise_N2 | WER_oracle | 0.375000 | 0.370487 | 0.545838 | 166/43/25 |
| snr_-18 | noisy | 0.927885 | 0.925354 | 1.000000 | 35/544/0 |
| snr_-18 | frontend | 1.743590 | 1.793647 | 3.240993 | 231/194/663 |
| snr_-18 | off | 0.905449 | 0.925498 | 1.297957 | 412/18/135 |
| snr_-18 | WER_oracle | 0.746795 | 0.744178 | 0.995727 | 244/192/30 |
| snr_0 | noisy | 0.190705 | 0.200209 | 0.291667 | 85/17/17 |
| snr_0 | frontend | 0.112179 | 0.115696 | 0.146645 | 44/21/5 |
| snr_0 | off | 0.214744 | 0.219818 | 0.300336 | 90/19/25 |
| snr_0 | WER_oracle | 0.099359 | 0.102481 | 0.125338 | 39/19/4 |
| N1_-18 | noisy | 0.971154 | 0.968985 | 1.000000 | 7/296/0 |
| N1_-18 | frontend | 2.804487 | 2.912238 | 5.485164 | 83/156/636 |
| N1_-18 | off | 0.929487 | 0.948674 | 1.291396 | 214/9/67 |
| N1_-18 | WER_oracle | 0.830128 | 0.831156 | 1.005997 | 95/156/8 |
| N1_0 | noisy | 0.131410 | 0.137410 | 0.197037 | 25/10/6 |
| N1_0 | frontend | 0.137821 | 0.147617 | 0.187072 | 27/14/2 |
| N1_0 | off | 0.233974 | 0.241867 | 0.326772 | 54/13/6 |
| N1_0 | WER_oracle | 0.112179 | 0.121188 | 0.144458 | 22/12/1 |
| N2_-18 | noisy | 0.884615 | 0.881724 | 1.000000 | 28/248/0 |
| N2_-18 | frontend | 0.682692 | 0.675056 | 0.996821 | 148/38/27 |
| N2_-18 | off | 0.881410 | 0.902323 | 1.304518 | 198/9/68 |
| N2_-18 | WER_oracle | 0.663462 | 0.657199 | 0.985457 | 149/36/22 |
| N2_0 | noisy | 0.250000 | 0.263008 | 0.386296 | 60/7/11 |
| N2_0 | frontend | 0.086538 | 0.083775 | 0.106218 | 17/7/3 |
| N2_0 | off | 0.195513 | 0.197769 | 0.273900 | 36/6/19 |
| N2_0 | WER_oracle | 0.086538 | 0.083775 | 0.106218 | 17/7/3 |
| speaker_p227 | noisy | 0.500000 | 0.498842 | 0.572917 | 20/124/0 |
| speaker_p227 | frontend | 2.593750 | 2.536969 | 5.140625 | 78/33/636 |
| speaker_p227 | off | 0.527778 | 0.526834 | 0.734375 | 127/6/19 |
| speaker_p227 | WER_oracle | 0.434028 | 0.434459 | 0.630208 | 87/31/7 |
| speaker_p228 | noisy | 0.530303 | 0.530765 | 0.573864 | 20/188/2 |
| speaker_p228 | frontend | 0.416667 | 0.416633 | 0.479167 | 61/97/7 |
| speaker_p228 | off | 0.484848 | 0.485102 | 0.696023 | 150/17/25 |
| speaker_p228 | WER_oracle | 0.411616 | 0.411633 | 0.467803 | 59/97/7 |
| speaker_p230 | noisy | 0.594262 | 0.599809 | 0.666667 | 28/110/7 |
| speaker_p230 | frontend | 0.446721 | 0.438862 | 0.587302 | 73/24/12 |
| speaker_p230 | off | 0.766393 | 0.757081 | 1.001984 | 105/6/76 |
| speaker_p230 | WER_oracle | 0.446721 | 0.438862 | 0.587302 | 73/24/12 |
| speaker_p231 | noisy | 0.621875 | 0.621711 | 0.769886 | 52/139/8 |
| speaker_p231 | frontend | 0.428125 | 0.426222 | 0.568182 | 63/61/13 |
| speaker_p231 | off | 0.525000 | 0.521617 | 0.764205 | 120/8/40 |
| speaker_p231 | WER_oracle | 0.409375 | 0.408365 | 0.556818 | 64/59/8 |
| utterance_p227_030 | noisy | 0.540541 | 0.540541 | 0.583333 | 14/66/0 |
| utterance_p227_030 | frontend | 4.581081 | 4.581081 | 9.625000 | 41/2/635 |
| utterance_p227_030 | off | 0.560811 | 0.560811 | 0.750000 | 72/0/11 |
| utterance_p227_030 | WER_oracle | 0.418919 | 0.418919 | 0.666667 | 53/2/7 |
| utterance_p227_031 | noisy | 0.457143 | 0.457143 | 0.562500 | 6/58/0 |
| utterance_p227_031 | frontend | 0.492857 | 0.492857 | 0.656250 | 37/31/1 |
| utterance_p227_031 | off | 0.492857 | 0.492857 | 0.718750 | 55/6/8 |
| utterance_p227_031 | WER_oracle | 0.450000 | 0.450000 | 0.593750 | 34/29/0 |
| utterance_p228_026 | noisy | 0.576531 | 0.576531 | 0.625000 | 15/96/2 |
| utterance_p228_026 | frontend | 0.413265 | 0.413265 | 0.458333 | 27/49/5 |
| utterance_p228_026 | off | 0.510204 | 0.510204 | 0.687500 | 79/12/9 |
| utterance_p228_026 | WER_oracle | 0.413265 | 0.413265 | 0.458333 | 27/49/5 |
| utterance_p228_032 | noisy | 0.485000 | 0.485000 | 0.522727 | 5/92/0 |
| utterance_p228_032 | frontend | 0.420000 | 0.420000 | 0.500000 | 34/48/2 |
| utterance_p228_032 | off | 0.460000 | 0.460000 | 0.704545 | 71/5/16 |
| utterance_p228_032 | WER_oracle | 0.410000 | 0.410000 | 0.477273 | 32/48/2 |
| utterance_p230_027 | noisy | 0.648148 | 0.648148 | 0.750000 | 17/46/7 |
| utterance_p230_027 | frontend | 0.370370 | 0.370370 | 0.535714 | 18/16/6 |
| utterance_p230_027 | off | 0.675926 | 0.675926 | 0.892857 | 41/1/31 |
| utterance_p230_027 | WER_oracle | 0.370370 | 0.370370 | 0.535714 | 18/16/6 |
| utterance_p230_029 | noisy | 0.551471 | 0.551471 | 0.583333 | 11/64/0 |
| utterance_p230_029 | frontend | 0.507353 | 0.507353 | 0.638889 | 55/8/6 |
| utterance_p230_029 | off | 0.838235 | 0.838235 | 1.111111 | 64/5/45 |
| utterance_p230_029 | WER_oracle | 0.507353 | 0.507353 | 0.638889 | 55/8/6 |
| utterance_p231_029 | noisy | 0.618421 | 0.618421 | 0.812500 | 31/61/2 |
| utterance_p231_029 | frontend | 0.388158 | 0.388158 | 0.500000 | 23/31/5 |
| utterance_p231_029 | off | 0.453947 | 0.453947 | 0.687500 | 47/2/20 |
| utterance_p231_029 | WER_oracle | 0.388158 | 0.388158 | 0.500000 | 23/31/5 |
| utterance_p231_032 | noisy | 0.625000 | 0.625000 | 0.727273 | 21/78/6 |
| utterance_p231_032 | frontend | 0.464286 | 0.464286 | 0.636364 | 40/30/8 |
| utterance_p231_032 | off | 0.589286 | 0.589286 | 0.840909 | 73/6/20 |
| utterance_p231_032 | WER_oracle | 0.428571 | 0.428571 | 0.613636 | 41/28/3 |

## Quality: actual full-length PCM, fixed same-scale clean reference

All96 arms/3 metrics valid: errors0,warnings0, noSTOI sentinel observed. OFF−front paired means all32: SI-SDR **−1.326095dB**, STOI **+0.009964728**, PESQ **+0.062593758**. Direction tradeoff is explicit; these are not A-style human naturalness scores or phone identities.

| Stratum | Metric | noisy mean/n | frontend mean/n | OFF mean/n | OFF−front paired mean/n |
|---|---|---|---|---|---|
| all | si_sdr | -9.047572/32 | 7.005628/32 | 5.679533/32 | -1.326095/32 |
| all | stoi | 0.616225/32 | 0.700332/32 | 0.710296/32 | 0.009965/32 |
| all | pesq | 1.042800/32 | 1.299943/32 | 1.362537/32 | 0.062594/32 |
| noise_N1 | si_sdr | -9.065082/16 | 4.502622/16 | 2.822903/16 | -1.679719/16 |
| noise_N1 | stoi | 0.608376/16 | 0.644946/16 | 0.622535/16 | -0.022411/16 |
| noise_N1 | pesq | 1.038268/16 | 1.257680/16 | 1.331339/16 | 0.073659/16 |
| noise_N2 | si_sdr | -9.030063/16 | 9.508633/16 | 8.536163/16 | -0.972470/16 |
| noise_N2 | stoi | 0.624074/16 | 0.755718/16 | 0.798058/16 | 0.042340/16 |
| noise_N2 | pesq | 1.047331/16 | 1.342206/16 | 1.393735/16 | 0.051529/16 |
| snr_-18 | si_sdr | -18.085693/16 | 0.912347/16 | -0.849078/16 | -1.761424/16 |
| snr_-18 | stoi | 0.475672/16 | 0.533490/16 | 0.541573/16 | 0.008084/16 |
| snr_-18 | pesq | 1.039533/16 | 1.113909/16 | 1.100515/16 | -0.013394/16 |
| snr_0 | si_sdr | -0.009452/16 | 13.098909/16 | 12.208144/16 | -0.890765/16 |
| snr_0 | stoi | 0.756777/16 | 0.867174/16 | 0.879020/16 | 0.011846/16 |
| snr_0 | pesq | 1.046066/16 | 1.485978/16 | 1.624559/16 | 0.138581/16 |
| N1_-18 | si_sdr | -18.117808/8 | -3.251473/8 | -5.566449/8 | -2.314976/8 |
| N1_-18 | stoi | 0.478816/8 | 0.438656/8 | 0.387041/8 | -0.051615/8 |
| N1_-18 | pesq | 1.031570/8 | 1.084538/8 | 1.064907/8 | -0.019630/8 |
| N1_0 | si_sdr | -0.012356/8 | 12.256718/8 | 11.212255/8 | -1.044463/8 |
| N1_0 | stoi | 0.737936/8 | 0.851236/8 | 0.858030/8 | 0.006793/8 |
| N1_0 | pesq | 1.044967/8 | 1.430823/8 | 1.597771/8 | 0.166948/8 |
| N2_-18 | si_sdr | -18.053579/8 | 5.076167/8 | 3.868293/8 | -1.207873/8 |
| N2_-18 | stoi | 0.472529/8 | 0.628323/8 | 0.696106/8 | 0.067782/8 |
| N2_-18 | pesq | 1.047496/8 | 1.143280/8 | 1.136123/8 | -0.007157/8 |
| N2_0 | si_sdr | -0.006547/8 | 13.941099/8 | 13.204032/8 | -0.737067/8 |
| N2_0 | stoi | 0.775619/8 | 0.883112/8 | 0.900010/8 | 0.016898/8 |
| N2_0 | pesq | 1.047165/8 | 1.541133/8 | 1.651348/8 | 0.110215/8 |
| speaker_p227 | si_sdr | -8.929857/8 | 6.340158/8 | 4.964250/8 | -1.375908/8 |
| speaker_p227 | stoi | 0.649187/8 | 0.742946/8 | 0.729567/8 | -0.013379/8 |
| speaker_p227 | pesq | 1.050614/8 | 1.240648/8 | 1.311297/8 | 0.070649/8 |
| speaker_p228 | si_sdr | -9.149083/8 | 7.040612/8 | 5.960143/8 | -1.080469/8 |
| speaker_p228 | stoi | 0.545735/8 | 0.639907/8 | 0.658053/8 | 0.018146/8 |
| speaker_p228 | pesq | 1.037239/8 | 1.248464/8 | 1.352727/8 | 0.104264/8 |
| speaker_p230 | si_sdr | -9.063381/8 | 6.902313/8 | 5.487610/8 | -1.414704/8 |
| speaker_p230 | stoi | 0.662750/8 | 0.722374/8 | 0.747599/8 | 0.025225/8 |
| speaker_p230 | pesq | 1.048003/8 | 1.392474/8 | 1.438896/8 | 0.046422/8 |
| speaker_p231 | si_sdr | -9.047970/8 | 7.739427/8 | 6.306129/8 | -1.433298/8 |
| speaker_p231 | stoi | 0.607227/8 | 0.696099/8 | 0.705967/8 | 0.009868/8 |
| speaker_p231 | pesq | 1.035343/8 | 1.318187/8 | 1.347228/8 | 0.029041/8 |
| utterance_p227_030 | si_sdr | -9.036468/4 | 5.694741/4 | 4.182628/4 | -1.512113/4 |
| utterance_p227_030 | stoi | 0.692446/4 | 0.776349/4 | 0.755939/4 | -0.020410/4 |
| utterance_p227_030 | pesq | 1.041169/4 | 1.306949/4 | 1.364079/4 | 0.057130/4 |
| utterance_p227_031 | si_sdr | -8.823245/4 | 6.985575/4 | 5.745873/4 | -1.239702/4 |
| utterance_p227_031 | stoi | 0.605929/4 | 0.709544/4 | 0.703195/4 | -0.006349/4 |
| utterance_p227_031 | pesq | 1.060058/4 | 1.174348/4 | 1.258515/4 | 0.084168/4 |
| utterance_p228_026 | si_sdr | -9.054070/4 | 6.215453/4 | 5.471455/4 | -0.743998/4 |
| utterance_p228_026 | stoi | 0.556246/4 | 0.618930/4 | 0.631210/4 | 0.012280/4 |
| utterance_p228_026 | pesq | 1.038297/4 | 1.217891/4 | 1.308109/4 | 0.090218/4 |
| utterance_p228_032 | si_sdr | -9.244095/4 | 7.865772/4 | 6.448831/4 | -1.416941/4 |
| utterance_p228_032 | stoi | 0.535224/4 | 0.660884/4 | 0.684895/4 | 0.024011/4 |
| utterance_p228_032 | pesq | 1.036181/4 | 1.279037/4 | 1.397346/4 | 0.118309/4 |
| utterance_p230_027 | si_sdr | -8.992119/4 | 7.543414/4 | 6.509096/4 | -1.034318/4 |
| utterance_p230_027 | stoi | 0.643871/4 | 0.714546/4 | 0.738416/4 | 0.023870/4 |
| utterance_p230_027 | pesq | 1.048675/4 | 1.482394/4 | 1.527327/4 | 0.044933/4 |
| utterance_p230_029 | si_sdr | -9.134643/4 | 6.261212/4 | 4.466123/4 | -1.795089/4 |
| utterance_p230_029 | stoi | 0.681629/4 | 0.730203/4 | 0.756782/4 | 0.026580/4 |
| utterance_p230_029 | pesq | 1.047330/4 | 1.302554/4 | 1.350465/4 | 0.047911/4 |
| utterance_p231_029 | si_sdr | -9.194441/4 | 8.205510/4 | 7.171864/4 | -1.033646/4 |
| utterance_p231_029 | stoi | 0.631745/4 | 0.731553/4 | 0.735830/4 | 0.004277/4 |
| utterance_p231_029 | pesq | 1.029064/4 | 1.322917/4 | 1.369420/4 | 0.046503/4 |
| utterance_p231_032 | si_sdr | -8.901498/4 | 7.273344/4 | 5.440394/4 | -1.832950/4 |
| utterance_p231_032 | stoi | 0.582709/4 | 0.660646/4 | 0.676104/4 | 0.015458/4 |
| utterance_p231_032 | pesq | 1.041623/4 | 1.313458/4 | 1.325037/4 | 0.011578/4 |

## Per-case oracle and actual native observations

Every32 pair has NFE30 and identity length match. All64 outputs have zero preclip beyond±1, zero endpoint equality and zero serialization saturation/overrange. Real backend norm equals actual native noisy peak for each case. Raw arrays/PCM reencoding proof retained.

| Case | seed | norm | front peak | OFF peak | E front/OFF | fixed oracle arm |
|---|---:|---:|---:|---:|---|---|
| p227_030_N1_snr_m18.wav | 3979081451 | 0.378173828125 | 0.0562202818692 | 0.102899119258 | 222/7 | off |
| p227_030_N1_snr_p0.wav | 3820635049 | 0.155303955078 | 0.14659191668 | 0.14685061574 | 1/1 | frontend |
| p227_030_N2_snr_m18.wav | 3726926100 | 0.889495849609 | 0.101307958364 | 0.128338366747 | 7/9 | frontend |
| p227_030_N2_snr_p0.wav | 3660618754 | 0.173645019531 | 0.146332368255 | 0.140532672405 | 1/1 | frontend |
| p227_031_N1_snr_m18.wav | 2091614519 | 0.411682128906 | 0.135028883815 | 0.165077239275 | 8/10 | frontend |
| p227_031_N1_snr_p0.wav | 1319866293 | 0.196258544922 | 0.178193300962 | 0.164805755019 | 3/1 | off |
| p227_031_N2_snr_m18.wav | 454754968 | 0.898773193359 | 0.15812201798 | 0.155407592654 | 9/10 | frontend |
| p227_031_N2_snr_p0.wav | 4284976580 | 0.188598632812 | 0.173894003034 | 0.17226152122 | 1/2 | frontend |
| p228_026_N1_snr_m18.wav | 59012987 | 0.224090576172 | 0.0153237273917 | 0.0292052272707 | 9/14 | frontend |
| p228_026_N1_snr_p0.wav | 2999706490 | 0.0562744140625 | 0.0452182032168 | 0.0470773391426 | 1/6 | frontend |
| p228_026_N2_snr_m18.wav | 20043341 | 0.899993896484 | 0.0464253239334 | 0.0435570031404 | 11/12 | frontend |
| p228_026_N2_snr_p0.wav | 604179555 | 0.11474609375 | 0.046508025378 | 0.0466867648065 | 1/1 | frontend |
| p228_032_N1_snr_m18.wav | 3913318104 | 0.201538085938 | 0.0253085568547 | 0.0380472578108 | 10/15 | frontend |
| p228_032_N1_snr_p0.wav | 910567177 | 0.05126953125 | 0.0405115894973 | 0.0405202582479 | 1/0 | off |
| p228_032_N2_snr_m18.wav | 3737045503 | 0.899383544922 | 0.0439554899931 | 0.0378196574748 | 10/13 | frontend |
| p228_032_N2_snr_p0.wav | 980657116 | 0.112701416016 | 0.0439691096544 | 0.0443174168468 | 1/3 | frontend |
| p230_027_N1_snr_m18.wav | 3862141435 | 0.435180664062 | 0.052956558764 | 0.0925181508064 | 7/9 | frontend |
| p230_027_N1_snr_p0.wav | 165934678 | 0.105621337891 | 0.0915989503264 | 0.087372019887 | 2/3 | frontend |
| p230_027_N2_snr_m18.wav | 400820221 | 0.899993896484 | 0.0726065710187 | 0.0746796876192 | 6/10 | frontend |
| p230_027_N2_snr_p0.wav | 3384358598 | 0.123352050781 | 0.0946460664272 | 0.0933610275388 | 0/3 | frontend |
| p230_029_N1_snr_m18.wav | 2861508768 | 0.380523681641 | 0.0262572299689 | 0.0535182468593 | 11/15 | frontend |
| p230_029_N1_snr_p0.wav | 3832051784 | 0.0956420898438 | 0.0799101144075 | 0.0833485797048 | 2/6 | frontend |
| p230_029_N2_snr_m18.wav | 485237701 | 0.889587402344 | 0.0667785778642 | 0.0808246582747 | 9/15 | frontend |
| p230_029_N2_snr_p0.wav | 3139967728 | 0.107391357422 | 0.0816055908799 | 0.0861789509654 | 1/4 | frontend |
| p231_029_N1_snr_m18.wav | 3702959134 | 0.392028808594 | 0.117158792913 | 0.116357192397 | 8/10 | frontend |
| p231_029_N1_snr_p0.wav | 588216666 | 0.144104003906 | 0.137712135911 | 0.145207121968 | 0/0 | frontend |
| p231_029_N2_snr_m18.wav | 1649520578 | 0.897735595703 | 0.130688980222 | 0.130048155785 | 8/12 | frontend |
| p231_029_N2_snr_p0.wav | 1344737105 | 0.182373046875 | 0.142086267471 | 0.143970683217 | 0/0 | frontend |
| p231_032_N1_snr_m18.wav | 3674325763 | 0.365234375 | 0.0719798281789 | 0.0842985361814 | 11/13 | frontend |
| p231_032_N1_snr_p0.wav | 2786558628 | 0.125 | 0.110880188644 | 0.115717083216 | 3/8 | frontend |
| p231_032_N2_snr_m18.wav | 2767724923 | 0.893402099609 | 0.0969921499491 | 0.10181504488 | 11/10 | off |
| p231_032_N2_snr_p0.wav | 262657095 | 0.121887207031 | 0.107711344957 | 0.107374191284 | 3/6 | frontend |

## Full104 literal reference/hypotheses and scores

Text below is JSON-escaped exact original strings (not normalized paraphrases). Detailed token/character DP alignments and all decoder option records reside in outputs/asr/*_raw.json and *_score.json.

### p227_030__scaledclean

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Nonetheless, the overall picture is healthy."

Word S/D/I/N=0/0/0/6; WER=0.000000; character S/D/I/N=0/0/0/37; CER=0.000000; decode API calls=1.

### p227_031__scaledclean

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " The Premier League has been a huge success."

Word S/D/I/N=0/0/0/8; WER=0.000000; character S/D/I/N=0/0/0/35; CER=0.000000; decode API calls=1.

### p228_026__scaledclean

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So if we can take anything from tonight it will be fantastic."

Word S/D/I/N=0/0/0/12; WER=0.000000; character S/D/I/N=0/0/0/49; CER=0.000000; decode API calls=1.

### p228_032__scaledclean

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " We must provide a long-term solution to tackle this attitude."

Word S/D/I/N=0/0/0/11; WER=0.000000; character S/D/I/N=0/0/0/50; CER=0.000000; decode API calls=1.

### p230_027__scaledclean

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage of the Unawah."

Word S/D/I/N=1/0/0/7; WER=0.142857; character S/D/I/N=1/1/0/27; CER=0.074074; decode API calls=1.

### p230_029__scaledclean

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " It shouldn't come as a surprise but it does."

Word S/D/I/N=0/0/0/9; WER=0.000000; character S/D/I/N=0/0/0/34; CER=0.000000; decode API calls=1.

### p231_029__scaledclean

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last night Mr Robertson said he was delighted."

Word S/D/I/N=0/0/0/8; WER=0.000000; character S/D/I/N=0/0/0/38; CER=0.000000; decode API calls=1.

### p231_032__scaledclean

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " If we cannot succeed in this endeavor, then so be it."

Word S/D/I/N=1/0/0/11; WER=0.090909; character S/D/I/N=0/1/0/42; CER=0.023810; decode API calls=1.

### p227_030_N1_snr_m18.wav__noisy

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " ."

Word S/D/I/N=0/6/0/6; WER=1.000000; character S/D/I/N=0/37/0/37; CER=1.000000; decode API calls=1.

### p227_030_N1_snr_m18.wav__frontend

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " We're going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to be going to"

Word S/D/I/N=6/0/216/6; WER=37.000000; character S/D/I/N=20/0/631/37; CER=17.594595; decode API calls=1.

### p227_030_N1_snr_m18.wav__off

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " The Bolden Quasselworm Harder Van Dank Lourdes."

Word S/D/I/N=6/0/1/6; WER=1.166667; character S/D/I/N=32/0/3/37; CER=0.945946; decode API calls=1.

### p227_030_N1_snr_p0.wav__noisy

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Nonetheless, the overall picture is helping."

Word S/D/I/N=1/0/0/6; WER=0.166667; character S/D/I/N=5/0/0/37; CER=0.135135; decode API calls=1.

### p227_030_N1_snr_p0.wav__frontend

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Nonetheless, the overall picture is helping."

Word S/D/I/N=1/0/0/6; WER=0.166667; character S/D/I/N=5/0/0/37; CER=0.135135; decode API calls=1.

### p227_030_N1_snr_p0.wav__off

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Nonetheless, the overall picture is helpful."

Word S/D/I/N=1/0/0/6; WER=0.166667; character S/D/I/N=5/0/0/37; CER=0.135135; decode API calls=1.

### p227_030_N2_snr_m18.wav__noisy

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/4/0/6; WER=1.000000; character S/D/I/N=4/29/0/37; CER=0.891892; decode API calls=1.

### p227_030_N2_snr_m18.wav__frontend

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " One, the last of the overall future to establish."

Word S/D/I/N=4/0/3/6; WER=1.166667; character S/D/I/N=11/2/4/37; CER=0.459459; decode API calls=1.

### p227_030_N2_snr_m18.wav__off

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Mumble action, they pulled the oyster's to a big test."

Word S/D/I/N=5/0/4/6; WER=1.500000; character S/D/I/N=30/0/5/37; CER=0.945946; decode API calls=1.

### p227_030_N2_snr_p0.wav__noisy

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Nonetheless, the overall picture is helping."

Word S/D/I/N=1/0/0/6; WER=0.166667; character S/D/I/N=5/0/0/37; CER=0.135135; decode API calls=1.

### p227_030_N2_snr_p0.wav__frontend

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Nonetheless, the overall picture is helping."

Word S/D/I/N=1/0/0/6; WER=0.166667; character S/D/I/N=5/0/0/37; CER=0.135135; decode API calls=1.

### p227_030_N2_snr_p0.wav__off

Reference: "Nonetheless, the overall picture is healthy.\n"

Hypothesis: " Nonetheless, the overall picture is telephused."

Word S/D/I/N=1/0/0/6; WER=0.166667; character S/D/I/N=5/0/3/37; CER=0.216216; decode API calls=1.

### p227_031_N1_snr_m18.wav__noisy

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/6/0/8; WER=1.000000; character S/D/I/N=3/27/0/35; CER=0.857143; decode API calls=1.

### p227_031_N1_snr_m18.wav__frontend

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " Thank you very much."

Word S/D/I/N=4/4/0/8; WER=1.000000; character S/D/I/N=10/19/0/35; CER=0.828571; decode API calls=1.

### p227_031_N1_snr_m18.wav__off

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " that if better you are going to all the seat."

Word S/D/I/N=8/0/2/8; WER=1.250000; character S/D/I/N=22/3/3/35; CER=0.800000; decode API calls=1.

### p227_031_N1_snr_p0.wav__noisy

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " Premier League has been huge success."

Word S/D/I/N=0/2/0/8; WER=0.250000; character S/D/I/N=0/4/0/35; CER=0.114286; decode API calls=1.

### p227_031_N1_snr_p0.wav__frontend

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " Promised lead has been a huge success."

Word S/D/I/N=2/1/0/8; WER=0.375000; character S/D/I/N=3/5/1/35; CER=0.257143; decode API calls=1.

### p227_031_N1_snr_p0.wav__off

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " Premier League has been a huge success."

Word S/D/I/N=0/1/0/8; WER=0.125000; character S/D/I/N=0/3/0/35; CER=0.085714; decode API calls=1.

### p227_031_N2_snr_m18.wav__noisy

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/6/0/8; WER=1.000000; character S/D/I/N=3/27/0/35; CER=0.857143; decode API calls=1.

### p227_031_N2_snr_m18.wav__frontend

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " I'm going to leave the queue for a sec."

Word S/D/I/N=8/0/1/8; WER=1.125000; character S/D/I/N=22/6/0/35; CER=0.800000; decode API calls=1.

### p227_031_N2_snr_m18.wav__off

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " It's climbing in the lead, and then it's you. Tap."

Word S/D/I/N=8/0/2/8; WER=1.250000; character S/D/I/N=31/0/1/35; CER=0.914286; decode API calls=1.

### p227_031_N2_snr_p0.wav__noisy

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " The Premier League has been a huge success."

Word S/D/I/N=0/0/0/8; WER=0.000000; character S/D/I/N=0/0/0/35; CER=0.000000; decode API calls=1.

### p227_031_N2_snr_p0.wav__frontend

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " If Premier League has been a huge success."

Word S/D/I/N=1/0/0/8; WER=0.125000; character S/D/I/N=2/1/0/35; CER=0.085714; decode API calls=1.

### p227_031_N2_snr_p0.wav__off

Reference: "The Premier League has been a huge success.\n"

Hypothesis: " The Premier League has been accused of success."

Word S/D/I/N=2/0/0/8; WER=0.250000; character S/D/I/N=2/0/4/35; CER=0.171429; decode API calls=1.

### p228_026_N1_snr_m18.wav__noisy

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " ."

Word S/D/I/N=0/12/0/12; WER=1.000000; character S/D/I/N=0/49/0/49; CER=1.000000; decode API calls=1.

### p228_026_N1_snr_m18.wav__frontend

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So we can bring that."

Word S/D/I/N=2/7/0/12; WER=0.750000; character S/D/I/N=2/33/0/49; CER=0.714286; decode API calls=1.

### p228_026_N1_snr_m18.wav__off

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " They were 50mm hard way for you to think to have the hardest tires."

Word S/D/I/N=12/0/2/12; WER=1.166667; character S/D/I/N=36/0/4/49; CER=0.816327; decode API calls=1.

### p228_026_N1_snr_p0.wav__noisy

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So if we can take any thing tonight it will be some doctor."

Word S/D/I/N=3/0/1/12; WER=0.333333; character S/D/I/N=8/4/1/49; CER=0.265306; decode API calls=1.

### p228_026_N1_snr_p0.wav__frontend

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So if we can take anything from tonight it will be sunbat."

Word S/D/I/N=1/0/0/12; WER=0.083333; character S/D/I/N=3/3/0/49; CER=0.122449; decode API calls=1.

### p228_026_N1_snr_p0.wav__off

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So if we can take any engine to make, it will be found out."

Word S/D/I/N=4/0/2/12; WER=0.500000; character S/D/I/N=12/6/1/49; CER=0.387755; decode API calls=1.

### p228_026_N2_snr_m18.wav__noisy

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/10/0/12; WER=1.000000; character S/D/I/N=3/41/0/49; CER=0.897959; decode API calls=1.

### p228_026_N2_snr_m18.wav__frontend

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So it's a very common treatment like this and that."

Word S/D/I/N=9/2/0/12; WER=0.916667; character S/D/I/N=22/12/3/49; CER=0.755102; decode API calls=1.

### p228_026_N2_snr_m18.wav__off

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " if there are items through, I'm gonna think the handoff of me."

Word S/D/I/N=12/0/0/12; WER=1.000000; character S/D/I/N=30/5/4/49; CER=0.795918; decode API calls=1.

### p228_026_N2_snr_p0.wav__noisy

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So if we can take anything from tonight, it will be found out."

Word S/D/I/N=1/0/1/12; WER=0.166667; character S/D/I/N=4/2/1/49; CER=0.142857; decode API calls=1.

### p228_026_N2_snr_p0.wav__frontend

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So if we can tweak anything from tonight, it will be fantastic."

Word S/D/I/N=1/0/0/12; WER=0.083333; character S/D/I/N=0/1/2/49; CER=0.061224; decode API calls=1.

### p228_026_N2_snr_p0.wav__off

Reference: "So if we can take anything from tonight, it will be fantastic.\n"

Hypothesis: " So if we can take anything for tonight, it will be fantastic."

Word S/D/I/N=1/0/0/12; WER=0.083333; character S/D/I/N=1/1/0/49; CER=0.040816; decode API calls=1.

### p228_032_N1_snr_m18.wav__noisy

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " ."

Word S/D/I/N=0/11/0/11; WER=1.000000; character S/D/I/N=0/50/0/50; CER=1.000000; decode API calls=1.

### p228_032_N1_snr_m18.wav__frontend

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " We will work."

Word S/D/I/N=2/8/0/11; WER=0.909091; character S/D/I/N=3/40/0/50; CER=0.860000; decode API calls=1.

### p228_032_N1_snr_m18.wav__off

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " I'm actually going to be one key in the whole video. Have a good day."

Word S/D/I/N=11/0/4/11; WER=1.363636; character S/D/I/N=36/3/5/50; CER=0.880000; decode API calls=1.

### p228_032_N1_snr_p0.wav__noisy

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " It must provide a long-term solution to tackle this attitude."

Word S/D/I/N=1/0/0/11; WER=0.090909; character S/D/I/N=2/0/0/50; CER=0.040000; decode API calls=1.

### p228_032_N1_snr_p0.wav__frontend

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " It must provide a long-term solution to tackle this attitude."

Word S/D/I/N=1/0/0/11; WER=0.090909; character S/D/I/N=2/0/0/50; CER=0.040000; decode API calls=1.

### p228_032_N1_snr_p0.wav__off

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " We must provide a long-term solution to tackle this attitude."

Word S/D/I/N=0/0/0/11; WER=0.000000; character S/D/I/N=0/0/0/50; CER=0.000000; decode API calls=1.

### p228_032_N2_snr_m18.wav__noisy

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/9/0/11; WER=1.000000; character S/D/I/N=3/42/0/50; CER=0.900000; decode API calls=1.

### p228_032_N2_snr_m18.wav__frontend

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " much that I have a long time to eat, suffer and practice."

Word S/D/I/N=9/0/1/11; WER=0.909091; character S/D/I/N=25/8/2/50; CER=0.700000; decode API calls=1.

### p228_032_N2_snr_m18.wav__off

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " but the map desired a wrong-hand widget of trap or the spasticed model."

Word S/D/I/N=10/0/3/11; WER=1.181818; character S/D/I/N=30/2/9/50; CER=0.820000; decode API calls=1.

### p228_032_N2_snr_p0.wav__noisy

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " We must provide a long term solution to tackle this attitude."

Word S/D/I/N=0/0/0/11; WER=0.000000; character S/D/I/N=0/0/0/50; CER=0.000000; decode API calls=1.

### p228_032_N2_snr_p0.wav__frontend

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " We must survive a long-term solution to tackle this attitude."

Word S/D/I/N=1/0/0/11; WER=0.090909; character S/D/I/N=4/0/0/50; CER=0.080000; decode API calls=1.

### p228_032_N2_snr_p0.wav__off

Reference: "We must provide a long-term solution to tackle this attitude.\n"

Hypothesis: " We must provide a long-term solution to pack full-less attitude."

Word S/D/I/N=2/0/1/11; WER=0.272727; character S/D/I/N=5/0/2/50; CER=0.140000; decode API calls=1.

### p230_027_N1_snr_m18.wav__noisy

Reference: "He has the courage of the unaware.\n"

Hypothesis: " ."

Word S/D/I/N=0/7/0/7; WER=1.000000; character S/D/I/N=0/27/0/27; CER=1.000000; decode API calls=1.

### p230_027_N1_snr_m18.wav__frontend

Reference: "He has the courage of the unaware.\n"

Hypothesis: " Not one, take care."

Word S/D/I/N=4/3/0/7; WER=1.000000; character S/D/I/N=7/13/0/27; CER=0.740741; decode API calls=1.

### p230_027_N1_snr_m18.wav__off

Reference: "He has the courage of the unaware.\n"

Hypothesis: " the rose matte platter. It could help match through."

Word S/D/I/N=7/0/2/7; WER=1.285714; character S/D/I/N=16/0/15/27; CER=1.148148; decode API calls=1.

### p230_027_N1_snr_p0.wav__noisy

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage of that on the right."

Word S/D/I/N=1/0/2/7; WER=0.428571; character S/D/I/N=6/0/4/27; CER=0.370370; decode API calls=1.

### p230_027_N1_snr_p0.wav__frontend

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage of that ulua."

Word S/D/I/N=2/0/0/7; WER=0.285714; character S/D/I/N=5/2/0/27; CER=0.259259; decode API calls=1.

### p230_027_N1_snr_p0.wav__off

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage of Red or the Y."

Word S/D/I/N=1/0/2/7; WER=0.428571; character S/D/I/N=8/1/0/27; CER=0.333333; decode API calls=1.

### p230_027_N2_snr_m18.wav__noisy

Reference: "He has the courage of the unaware.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/5/0/7; WER=1.000000; character S/D/I/N=4/19/0/27; CER=0.851852; decode API calls=1.

### p230_027_N2_snr_m18.wav__frontend

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage to offer it for no one."

Word S/D/I/N=3/0/3/7; WER=0.857143; character S/D/I/N=6/1/6/27; CER=0.481481; decode API calls=1.

### p230_027_N2_snr_m18.wav__off

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has been stories. I'll just wear it and pull no one."

Word S/D/I/N=5/0/5/7; WER=1.428571; character S/D/I/N=15/0/14/27; CER=1.074074; decode API calls=1.

### p230_027_N2_snr_p0.wav__noisy

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage of that for no less."

Word S/D/I/N=2/0/2/7; WER=0.571429; character S/D/I/N=7/0/3/27; CER=0.370370; decode API calls=1.

### p230_027_N2_snr_p0.wav__frontend

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage of the unaware."

Word S/D/I/N=0/0/0/7; WER=0.000000; character S/D/I/N=0/0/0/27; CER=0.000000; decode API calls=1.

### p230_027_N2_snr_p0.wav__off

Reference: "He has the courage of the unaware.\n"

Hypothesis: " He has the courage of that one aware."

Word S/D/I/N=2/0/1/7; WER=0.428571; character S/D/I/N=2/0/2/27; CER=0.148148; decode API calls=1.

### p230_029_N1_snr_m18.wav__noisy

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " ."

Word S/D/I/N=0/9/0/9; WER=1.000000; character S/D/I/N=0/34/0/34; CER=1.000000; decode API calls=1.

### p230_029_N1_snr_m18.wav__frontend

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " Let's show you what it has to do with the system."

Word S/D/I/N=9/0/2/9; WER=1.222222; character S/D/I/N=22/2/5/34; CER=0.852941; decode API calls=1.

### p230_029_N1_snr_m18.wav__off

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " I thought the journey fit in zero, thank you for the whole heart of turn."

Word S/D/I/N=9/0/6/9; WER=1.666667; character S/D/I/N=22/0/23/34; CER=1.323529; decode API calls=1.

### p230_029_N1_snr_p0.wav__noisy

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " It shouldn't come as a surprise, but it does."

Word S/D/I/N=0/0/0/9; WER=0.000000; character S/D/I/N=0/0/0/34; CER=0.000000; decode API calls=1.

### p230_029_N1_snr_p0.wav__frontend

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " It shouldn't come as a surprise for it to."

Word S/D/I/N=2/0/0/9; WER=0.222222; character S/D/I/N=4/2/0/34; CER=0.176471; decode API calls=1.

### p230_029_N1_snr_p0.wav__off

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " It's servant come as the fries for it though."

Word S/D/I/N=6/0/0/9; WER=0.666667; character S/D/I/N=17/1/2/34; CER=0.588235; decode API calls=1.

### p230_029_N2_snr_m18.wav__noisy

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/7/0/9; WER=1.000000; character S/D/I/N=3/26/0/34; CER=0.852941; decode API calls=1.

### p230_029_N2_snr_m18.wav__frontend

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " which is a function of the bias that's good."

Word S/D/I/N=9/0/0/9; WER=1.000000; character S/D/I/N=28/1/1/34; CER=0.882353; decode API calls=1.

### p230_029_N2_snr_m18.wav__off

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " You know, it's still a clums of time, it's all for a vibe to record soon."

Word S/D/I/N=8/0/7/9; WER=1.666667; character S/D/I/N=20/0/19/34; CER=1.147059; decode API calls=1.

### p230_029_N2_snr_p0.wav__noisy

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " It doesn't come as the size, but it does."

Word S/D/I/N=3/0/0/9; WER=0.333333; character S/D/I/N=8/4/0/34; CER=0.352941; decode API calls=1.

### p230_029_N2_snr_p0.wav__frontend

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " It shouldn't come as a prize, but it does."

Word S/D/I/N=1/0/0/9; WER=0.111111; character S/D/I/N=1/3/0/34; CER=0.117647; decode API calls=1.

### p230_029_N2_snr_p0.wav__off

Reference: "It shouldn't come as a surprise, but it does.\n"

Hypothesis: " It's a student of a surprise, but it does."

Word S/D/I/N=4/0/0/9; WER=0.444444; character S/D/I/N=5/4/1/34; CER=0.294118; decode API calls=1.

### p231_029_N1_snr_m18.wav__noisy

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " I love you."

Word S/D/I/N=3/5/0/8; WER=1.000000; character S/D/I/N=4/30/0/38; CER=0.894737; decode API calls=1.

### p231_029_N1_snr_m18.wav__frontend

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " of my support."

Word S/D/I/N=3/5/0/8; WER=1.000000; character S/D/I/N=8/27/0/38; CER=0.921053; decode API calls=1.

### p231_029_N1_snr_m18.wav__off

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " stuff from it because what I'm doing is bring stuff."

Word S/D/I/N=8/0/2/8; WER=1.250000; character S/D/I/N=25/2/5/38; CER=0.842105; decode API calls=1.

### p231_029_N1_snr_p0.wav__noisy

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last night Mr. Robinson said he was delighted."

Word S/D/I/N=1/0/0/8; WER=0.125000; character S/D/I/N=2/1/0/38; CER=0.078947; decode API calls=1.

### p231_029_N1_snr_p0.wav__frontend

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last night Mr. Robertson said he was delighted."

Word S/D/I/N=0/0/0/8; WER=0.000000; character S/D/I/N=0/0/0/38; CER=0.000000; decode API calls=1.

### p231_029_N1_snr_p0.wav__off

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last night, Mr. Robertson said he was delighted."

Word S/D/I/N=0/0/0/8; WER=0.000000; character S/D/I/N=0/0/0/38; CER=0.000000; decode API calls=1.

### p231_029_N2_snr_m18.wav__noisy

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/6/0/8; WER=1.000000; character S/D/I/N=5/30/0/38; CER=0.921053; decode API calls=1.

### p231_029_N2_snr_m18.wav__frontend

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last pack, we'll start this inside with the light."

Word S/D/I/N=7/0/1/8; WER=1.000000; character S/D/I/N=15/4/5/38; CER=0.631579; decode API calls=1.

### p231_029_N2_snr_m18.wav__off

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " That's the last part, as well as use everything that they might end."

Word S/D/I/N=7/0/5/8; WER=1.500000; character S/D/I/N=22/0/15/38; CER=0.973684; decode API calls=1.

### p231_029_N2_snr_p0.wav__noisy

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last night, this drop is in 30 minutes, the light is..."

Word S/D/I/N=6/0/3/8; WER=1.125000; character S/D/I/N=20/0/2/38; CER=0.578947; decode API calls=1.

### p231_029_N2_snr_p0.wav__frontend

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last night Mr Robertson said he was delighted."

Word S/D/I/N=0/0/0/8; WER=0.000000; character S/D/I/N=0/0/0/38; CER=0.000000; decode API calls=1.

### p231_029_N2_snr_p0.wav__off

Reference: "Last night, Mr Robertson said he was delighted.\n"

Hypothesis: " Last night Mr Robertson said he was delighted."

Word S/D/I/N=0/0/0/8; WER=0.000000; character S/D/I/N=0/0/0/38; CER=0.000000; decode API calls=1.

### p231_032_N1_snr_m18.wav__noisy

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " ."

Word S/D/I/N=0/11/0/11; WER=1.000000; character S/D/I/N=0/42/0/42; CER=1.000000; decode API calls=1.

### p231_032_N1_snr_m18.wav__frontend

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " Now it's good, it was right."

Word S/D/I/N=6/5/0/11; WER=1.000000; character S/D/I/N=11/22/0/42; CER=0.785714; decode API calls=1.

### p231_032_N1_snr_m18.wav__off

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " time we've had to the nose in the boy der Haareat tips, it's sharp."

Word S/D/I/N=10/0/3/11; WER=1.181818; character S/D/I/N=25/1/9/42; CER=0.833333; decode API calls=1.

### p231_032_N1_snr_p0.wav__noisy

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " If you cannot succeed in this endeavor, then so be it."

Word S/D/I/N=2/0/0/11; WER=0.181818; character S/D/I/N=2/1/1/42; CER=0.095238; decode API calls=1.

### p231_032_N1_snr_p0.wav__frontend

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " But we cannot succeed in this endeavor and so be it."

Word S/D/I/N=3/0/0/11; WER=0.272727; character S/D/I/N=5/2/1/42; CER=0.190476; decode API calls=1.

### p231_032_N1_snr_p0.wav__off

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " We'll be kind of succeeding this in devart and so be it."

Word S/D/I/N=7/0/1/11; WER=0.727273; character S/D/I/N=12/2/3/42; CER=0.404762; decode API calls=1.

### p231_032_N2_snr_m18.wav__noisy

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " Thank you."

Word S/D/I/N=2/9/0/11; WER=1.000000; character S/D/I/N=3/34/0/42; CER=0.880952; decode API calls=1.

### p231_032_N2_snr_m18.wav__frontend

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " So you cannot see if the lesson does that, it's over here."

Word S/D/I/N=10/0/1/11; WER=1.000000; character S/D/I/N=19/4/6/42; CER=0.690476; decode API calls=1.

### p231_032_N2_snr_m18.wav__off

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " If you cannot subteap with open dev life to a top in."

Word S/D/I/N=9/0/1/11; WER=0.909091; character S/D/I/N=20/2/1/42; CER=0.547619; decode API calls=1.

### p231_032_N2_snr_p0.wav__noisy

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " But we cannot just dig in this and tell about some phobia."

Word S/D/I/N=7/0/1/11; WER=0.727273; character S/D/I/N=16/1/5/42; CER=0.523810; decode API calls=1.

### p231_032_N2_snr_p0.wav__frontend

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " But we cannot succeed in this endeavor and so be it."

Word S/D/I/N=3/0/0/11; WER=0.272727; character S/D/I/N=5/2/1/42; CER=0.190476; decode API calls=1.

### p231_032_N2_snr_p0.wav__off

Reference: "If we cannot succeed in this endeavour, then so be it.\n"

Hypothesis: " that the cantals succeed in listening diverse and so be it."

Word S/D/I/N=6/0/0/11; WER=0.545455; character S/D/I/N=16/1/7/42; CER=0.571429; decode API calls=1.

## Disposition

Completed fixed measurement, not a learned selector or validated P8-V. Nonzero proxy pool headroom is a necessary-condition diagnostic only; observed repetition/clean-ASR discrepancies and4 clusters require cautious Lead interpretation. No true hallucination/deletion/recovery labels or new human feedback established. Do not train, extend seeds, rerun old/new samples, alter SNR/windows or select a favorable secondary endpoint based on this report. Worker stops after this report/verification; any new experiment requires new Lead design. Final accounting includes metadata/inputs/model artifacts and excludes symlink target duplication, recorded separately.

## Lead-requested descriptive decomposition (primary endpoint unchanged)

Sum of four speaker-specific best-fixed error totals = **167**, compared with global best-fixed229 and oracle159. Thus pooled70 saved errors decompose arithmetically into **62 across-speaker fixed-arm heterogeneity +8 within-speaker oracle margin**. This does not create a new selection endpoint, fitted deployable speaker rule, or proof of70 local recoveries. The planned per-speaker results above remain authoritative; no sample exclusion, cap or ASR rerun. Root will interpret separately.
