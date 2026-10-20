Native matched-content pilot attempt02 — completed; OFF has higher single-ASR error than matched frontend

The one approved scientific run started 2026-10-19T04:07:27.388842+00:00 (12:07:27.388842 Asia/Shanghai), completed in 46.539449716 seconds and exited0. Actual exec session74292, supervisor PID49180 and child PID49186 were observed with ps. No new smoke or scientific rerun occurred. The old failed launch and both dependency smokes remain immutable.

Native main ran once:16 actual frontend calls and16 matching OFF calls, N30/NFE30; one enhancement load with backend EMA enabled, then confirmed enhancement weakrefs dead before one local Whisper tiny.en load. All16 cases and52 publisher-reference ASR measurements are retained. Transcribe API calls52; internal decode API calls53; neither is a neural-forward count. Training0/test-content access false.

Actual blocked socket-constructor attempts: 1 strictly classified local urllib3 IPv6 probe (::1/AF_INET6); unknown network refusals 0; network bytes 0. All socket creation remained refused. The independent validator rechecked exact caller/source/bytecode/host/constructor evidence. The older smoke02 event still lacks historic caller proof; this run's evidence does not retroactively attribute it.

Micro-WER and S/D/I below use publisher text, with N156 repeated reference positions overall, N78 for noise-only or SNR-only strata, N39 for noise×SNR strata. WER may exceed100% through insertions. Each cell is WER%; S/D/I.

| Stratum | N | noisy | Y_pre | OFF |
|---|---:|---|---|---|
| all | 156 | 66.0256%; 40/58/5 | 59.6154%; 57/30/6 | 80.7692%; 82/10/34 |
| noise_N1 | 78 | 66.6667%; 18/32/2 | 65.3846%; 23/27/1 | 85.8974%; 42/9/16 |
| noise_N2 | 78 | 65.3846%; 22/26/3 | 53.8462%; 34/3/5 | 75.6410%; 40/1/18 |
| snr_-18 | 78 | 100.0000%; 21/57/0 | 96.1538%; 44/27/4 | 129.4872%; 64/6/31 |
| snr_0 | 78 | 32.0513%; 19/1/5 | 23.0769%; 13/3/2 | 32.0513%; 18/4/3 |
| noise_N1_snr_-18 | 39 | 100.0000%; 8/31/0 | 97.4359%; 13/25/0 | 128.2051%; 30/6/14 |
| noise_N1_snr_0 | 39 | 33.3333%; 10/1/2 | 33.3333%; 10/2/1 | 43.5897%; 12/3/2 |
| noise_N2_snr_-18 | 39 | 100.0000%; 13/26/0 | 94.8718%; 31/2/4 | 130.7692%; 34/0/17 |
| noise_N2_snr_0 | 39 | 30.7692%; 9/0/3 | 12.8205%; 3/1/1 | 20.5128%; 6/1/1 |

Sentence-mean WER and micro-CER are separate aggregates; cells give sentence-mean WER% / micro-CER%.

| Stratum | noisy | Y_pre | OFF |
|---|---|---|---|
| all | 66.8403 / 56.2147 | 60.4167 / 44.3503 | 83.5069 / 57.0621 |
| noise_N1 | 67.7083 / 60.7345 | 66.3194 / 52.2599 | 88.5417 / 60.7345 |
| noise_N2 | 65.9722 / 51.6949 | 54.5139 / 36.4407 | 78.4722 / 53.3898 |
| snr_-18 | 100.0000 / 90.9605 | 96.5278 / 74.0113 | 132.9861 / 91.8079 |
| snr_0 | 33.6806 / 21.4689 | 24.3056 / 14.6893 | 34.0278 / 22.3164 |
| noise_N1_snr_-18 | 100.0000 / 96.0452 | 97.9167 / 83.6158 | 131.2500 / 89.2655 |
| noise_N1_snr_0 | 35.4167 / 25.4237 | 34.7222 / 20.9040 | 45.8333 / 32.2034 |
| noise_N2_snr_-18 | 100.0000 / 85.8757 | 95.1389 / 64.4068 | 134.7222 / 94.3503 |
| noise_N2_snr_0 | 31.9444 / 17.5141 | 13.8889 / 8.4746 | 22.2222 / 12.4294 |

The four newly scaled-clean controls match all39 reference words and177 scoring characters: S=D=I=0, micro and sentence-mean WER/CER0, no control discrepancies. They are measurement controls, not evidence of low-SNR intelligibility or sample independence.

Across156 canonical reference positions, bothH=58, frontendH→OFFnotH=11, frontendnotH→OFFH=6, bothnotH=81. Frontend insertions6, OFF insertions34. These are deterministic ASR alignment transitions, not verified deleted/recovered speech or time-localized phonemes.

| Case | bothH | frontendH→OFFnotH | frontendnotH→OFFH | bothnotH | frontend/OFF insertions |
|---|---:|---:|---:|---:|---:|
| p226_027_N1_snr_m18.wav | 1 | 0 | 0 | 11 | 0/0 |
| p226_027_N1_snr_p0.wav | 10 | 0 | 0 | 2 | 0/0 |
| p226_027_N2_snr_m18.wav | 0 | 1 | 2 | 9 | 0/0 |
| p226_027_N2_snr_p0.wav | 12 | 0 | 0 | 0 | 0/0 |
| p226_028_N1_snr_m18.wav | 0 | 0 | 1 | 8 | 0/5 |
| p226_028_N1_snr_p0.wav | 7 | 0 | 0 | 2 | 1/1 |
| p226_028_N2_snr_m18.wav | 1 | 4 | 1 | 3 | 3/11 |
| p226_028_N2_snr_p0.wav | 9 | 0 | 0 | 0 | 0/0 |
| p287_037_N1_snr_m18.wav | 0 | 0 | 0 | 9 | 0/9 |
| p287_037_N1_snr_p0.wav | 2 | 1 | 0 | 6 | 0/0 |
| p287_037_N2_snr_m18.wav | 0 | 0 | 1 | 8 | 1/6 |
| p287_037_N2_snr_p0.wav | 4 | 2 | 0 | 3 | 0/0 |
| p287_054_N1_snr_m18.wav | 0 | 0 | 1 | 8 | 0/0 |
| p287_054_N1_snr_p0.wav | 5 | 2 | 0 | 2 | 0/1 |
| p287_054_N2_snr_m18.wav | 0 | 0 | 0 | 9 | 0/0 |
| p287_054_N2_snr_p0.wav | 7 | 1 | 0 | 1 | 1/1 |

All32 float32 raw outputs and32 native PCM16 WAVs remain saved. Every output is finite and full length; all16 native length matches are identity. Every frontend/OFF preclip count outside[-1,1], endpoint count and PCM representability saturation count is0. Actual frontend and backend normalization references match the native noisy peak in every case (range0.062286376953125–0.899993896484375). No output gain, filtering or renormalization was applied. The CPU validator independently reproduced all32 serializations byte-for-byte.

| Case | Native seed | norm factor | NFE | frontend peak | OFF peak |
|---|---:|---:|---:|---:|---:|
| p226_027_N1_snr_m18.wav | 1486336889 | 0.37652587890625 | 30 | 0.0235611405 | 0.0450556278 |
| p226_027_N1_snr_p0.wav | 3131377406 | 0.099212646484375 | 30 | 0.0849187821 | 0.0893018469 |
| p226_027_N2_snr_m18.wav | 3031475568 | 0.899993896484375 | 30 | 0.0727510676 | 0.0825417861 |
| p226_027_N2_snr_p0.wav | 1225915657 | 0.1234130859375 | 30 | 0.0938131139 | 0.0949972942 |
| p226_028_N1_snr_m18.wav | 4166214792 | 0.2086181640625 | 30 | 0.0200619232 | 0.0342322588 |
| p226_028_N1_snr_p0.wav | 2303162876 | 0.062286376953125 | 30 | 0.0602170229 | 0.0570524409 |
| p226_028_N2_snr_m18.wav | 2418667005 | 0.899993896484375 | 30 | 0.044742547 | 0.0467960238 |
| p226_028_N2_snr_p0.wav | 521132841 | 0.113525390625 | 30 | 0.0607208535 | 0.061877314 |
| p287_037_N1_snr_m18.wav | 3194511354 | 0.375946044921875 | 30 | 0.0643080771 | 0.0726197287 |
| p287_037_N1_snr_p0.wav | 447921833 | 0.13665771484375 | 30 | 0.122120768 | 0.127336428 |
| p287_037_N2_snr_m18.wav | 1074088042 | 0.899993896484375 | 30 | 0.10770236 | 0.108215436 |
| p287_037_N2_snr_p0.wav | 949415772 | 0.125762939453125 | 30 | 0.118973888 | 0.119518206 |
| p287_054_N1_snr_m18.wav | 2525130446 | 0.392425537109375 | 30 | 0.0619930625 | 0.0842155144 |
| p287_054_N1_snr_p0.wav | 3211441175 | 0.097686767578125 | 30 | 0.0887380317 | 0.0883479789 |
| p287_054_N2_snr_m18.wav | 4034286842 | 0.89593505859375 | 30 | 0.0888692886 | 0.0854813457 |
| p287_054_N2_snr_p0.wav | 611393327 | 0.11737060546875 | 30 | 0.0921955183 | 0.0908683762 |

All52 measurements follow. Full raw result/segments/tokens/decode options and canonical word/character alignments are in outputs/asr; immutable metrics.json includes every score, per-case OFF−frontend difference and transition position. The table shows N, word S/D/I, WER/CER and the literal ASR text (JSON-escaped).

| Measurement | N | S/D/I | WER% | CER% | ASR hypothesis |
|---|---:|---|---:|---:|---|
| p226_027__scaledclean | 12 | 0/0/0 | 0.0000 | 0.0000 | " They should have a major rethink about the event for next year." |
| p226_028__scaledclean | 9 | 0/0/0 | 0.0000 | 0.0000 | " Suggestions of the action being a punishment were dismissed." |
| p287_037__scaledclean | 9 | 0/0/0 | 0.0000 | 0.0000 | " But, after a dip in sales, growth was recorded." |
| p287_054__scaledclean | 9 | 0/0/0 | 0.0000 | 0.0000 | " In time, the first minister will grow in stature." |
| p226_027_N1_snr_m18.wav__noisy | 12 | 0/12/0 | 100.0000 | 100.0000 | " ." |
| p226_027_N1_snr_m18.wav__frontend | 12 | 7/4/0 | 91.6667 | 72.5490 | " I teach that role in the new buttons." |
| p226_027_N1_snr_m18.wav__off | 12 | 7/4/0 | 91.6667 | 74.5098 | " from Bands Africa, the Van Gogh, of Queen." |
| p226_027_N1_snr_p0.wav__noisy | 12 | 0/0/1 | 8.3333 | 5.8824 | " They should have a major rethink about the event for the next year." |
| p226_027_N1_snr_p0.wav__frontend | 12 | 2/0/0 | 16.6667 | 11.7647 | " They should have a major rethink about the event in next week." |
| p226_027_N1_snr_p0.wav__off | 12 | 2/0/0 | 16.6667 | 11.7647 | " They should have a major rethink about the event the next week." |
| p226_027_N2_snr_m18.wav__noisy | 12 | 2/10/0 | 100.0000 | 86.2745 | " Thank you." |
| p226_027_N2_snr_m18.wav__frontend | 12 | 10/1/0 | 91.6667 | 70.5882 | " which has a 90 reason back to the joint as well." |
| p226_027_N2_snr_m18.wav__off | 12 | 10/0/0 | 83.3333 | 64.7059 | " We should have the main reason to body of rins things again." |
| p226_027_N2_snr_p0.wav__noisy | 12 | 1/0/1 | 16.6667 | 7.8431 | " They should have a major rethink about the events for the next year." |
| p226_027_N2_snr_p0.wav__frontend | 12 | 0/0/0 | 0.0000 | 0.0000 | " They should have a major rethink about the event for next year." |
| p226_027_N2_snr_p0.wav__off | 12 | 0/0/0 | 0.0000 | 0.0000 | " They should have a major rethink about the event for next year." |
| p226_028_N1_snr_m18.wav__noisy | 9 | 0/9/0 | 100.0000 | 100.0000 | " ." |
| p226_028_N1_snr_m18.wav__frontend | 9 | 1/8/0 | 100.0000 | 86.2745 | " Question." |
| p226_028_N1_snr_m18.wav__off | 9 | 8/0/5 | 144.4444 | 88.2353 | " Oh I can break the best chew or fountains I think it's very helpful." |
| p226_028_N1_snr_p0.wav__noisy | 9 | 2/0/1 | 33.3333 | 21.5686 | " suggestions of the action being a punishment for the citizens." |
| p226_028_N1_snr_p0.wav__frontend | 9 | 2/0/1 | 33.3333 | 13.7255 | " suggestions of the action being a punishment for this mess." |
| p226_028_N1_snr_p0.wav__off | 9 | 2/0/1 | 33.3333 | 21.5686 | " Suggestions of the action being a punishment for the animals." |
| p226_028_N2_snr_m18.wav__noisy | 9 | 2/7/0 | 100.0000 | 90.1961 | " Thank you." |
| p226_028_N2_snr_m18.wav__frontend | 9 | 4/0/3 | 77.7778 | 43.1373 | " The question is, the action being written on a term of dismissed." |
| p226_028_N2_snr_m18.wav__off | 9 | 7/0/11 | 200.0000 | 107.8431 | " and they did have some of the best swim that the athletes in Bayley's come out to smess. But I..." |
| p226_028_N2_snr_p0.wav__noisy | 9 | 2/0/1 | 33.3333 | 15.6863 | " suggestions of the action being and the punishment for dismissed." |
| p226_028_N2_snr_p0.wav__frontend | 9 | 0/0/0 | 0.0000 | 0.0000 | " Suggestions of the action being a punishment were dismissed." |
| p226_028_N2_snr_p0.wav__off | 9 | 0/0/0 | 0.0000 | 0.0000 | " suggestions of the action being a punishment were dismissed." |
| p287_037_N1_snr_m18.wav__noisy | 9 | 8/1/0 | 100.0000 | 80.5556 | " I'm going to go to the next one." |
| p287_037_N1_snr_m18.wav__frontend | 9 | 5/4/0 | 100.0000 | 77.7778 | " We'll talk about the problem." |
| p287_037_N1_snr_m18.wav__off | 9 | 9/0/9 | 200.0000 | 122.2222 | " he can't do it, he's like, you have to be too fast, you'll be a silver bat on." |
| p287_037_N1_snr_p0.wav__noisy | 9 | 6/0/0 | 66.6667 | 63.8889 | " but after a different stage, we'll go over it." |
| p287_037_N1_snr_p0.wav__frontend | 9 | 5/1/0 | 66.6667 | 52.7778 | " But after a different size, Wokey, we're covered." |
| p287_037_N1_snr_p0.wav__off | 9 | 5/2/0 | 77.7778 | 61.1111 | " But, after the concities, both the ukulele." |
| p287_037_N2_snr_m18.wav__noisy | 9 | 2/7/0 | 100.0000 | 88.8889 | " Thank you." |
| p287_037_N2_snr_m18.wav__frontend | 9 | 9/0/1 | 111.1111 | 83.3333 | " It's not true, it didn't change. It's grossing the children." |
| p287_037_N2_snr_m18.wav__off | 9 | 8/0/6 | 155.5556 | 125.0000 | " the birth of its sense after it didn't say me the situation that you're talking." |
| p287_037_N2_snr_p0.wav__noisy | 9 | 5/0/0 | 55.5556 | 38.8889 | " But after a different sale, the road is recorded." |
| p287_037_N2_snr_p0.wav__frontend | 9 | 2/1/0 | 33.3333 | 22.2222 | " But after a different sales growth is recorded." |
| p287_037_N2_snr_p0.wav__off | 9 | 4/1/0 | 55.5556 | 41.6667 | " but after a different fails, Frof is recorded." |
| p287_054_N1_snr_m18.wav__noisy | 9 | 0/9/0 | 100.0000 | 100.0000 | " ." |
| p287_054_N1_snr_m18.wav__frontend | 9 | 0/9/0 | 100.0000 | 100.0000 | " ." |
| p287_054_N1_snr_m18.wav__off | 9 | 6/2/0 | 88.8889 | 79.4872 | " the townhouse, motherfawn is a new place." |
| p287_054_N1_snr_p0.wav__noisy | 9 | 2/1/0 | 33.3333 | 20.5128 | " in time, first ministry will grow in structure." |
| p287_054_N1_snr_p0.wav__frontend | 9 | 1/1/0 | 22.2222 | 12.8205 | " In time, first minister will grow his stature." |
| p287_054_N1_snr_p0.wav__off | 9 | 3/1/1 | 55.5556 | 46.1538 | " In time, first minister will grill these staff chairs." |
| p287_054_N2_snr_m18.wav__noisy | 9 | 7/2/0 | 100.0000 | 76.9231 | " I'm going to do a few more." |
| p287_054_N2_snr_m18.wav__frontend | 9 | 8/1/0 | 100.0000 | 66.6667 | " It's been five minutes before we go back." |
| p287_054_N2_snr_m18.wav__off | 9 | 9/0/0 | 100.0000 | 87.1795 | " their pin-kites of respect and entry is owned." |
| p287_054_N2_snr_p0.wav__noisy | 9 | 1/0/1 | 22.2222 | 12.8205 | " In time, the first minutes there will grow in stature." |
| p287_054_N2_snr_p0.wav__frontend | 9 | 1/0/1 | 22.2222 | 17.9487 | " In time, the first minister will grow in the back." |
| p287_054_N2_snr_p0.wav__off | 9 | 2/0/1 | 33.3333 | 17.9487 | " In times, the first minister will grow in the picture." |

Independent CPU validation ran once and passed:16 cases,32 serialization pairs,52 independent word/character DP scores and all aggregates/transitions;260 pre/post/supervisor/current source locks, artifact hashes and probe evidence. It loaded0 models and used0 network bytes. Parent original78 files, including original failed metrics SHA6e70d1bef76eb8ea752f401d5fa102b377580365cda477742e4131d4d25fa929, remain unchanged. Registry SHA0af790bf7392c8b24697cf1cb4c40e537d424c09d82940661a18ad37dfacdf13 and config are byte-identical copies; input directory remains a symlink to frozen CPU inputs.

Original attempt02 metrics SHA256: e0ed9b10c646cc19d28cc8c6c6cd0225fd87c1b19b77bac108b1bb19a17066e0. Full git status remains unchanged; provider codex_tools_compat was checked through read-only SQLite and real tools. Bridge PID39452 and listening PID16022 remain alive. No source edits, guard relaxation, cleanup, commit or push. Budget420s including finalization/40,000,000 bytes over the entire parent directory is respected. Final parent directory bytes: 24153753.

This is an exploratory four-utterance/two-speaker pilot with reused historical noises and complete-utterance SNR[-18,0], not16 independent samples, a confirmation set or a pretraining holdout. Different SNR basenames produce different native sampling seeds; no causal SNR isolation is claimed. The prior historical waveform conversion gate remains failed. No human listening truth was collected; a single ASR with language priors and ambiguous edit alignments cannot prove real swallowed words, hallucinations or universal quality changes. Lower D accompanied by higher S/I is not proof of restored content. Poor outcomes triggered no exclusions, parameter changes or repeat decoding. Research Lead receives the complete results for interpretation; this record authorizes no further experiment.
