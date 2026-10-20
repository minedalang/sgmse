# Exploratory P8 silent-gap screen

## Research Lead assessment

The flagged acoustic pattern is not exclusive to10720:3321 contains a1.04s
quiet gap with elevated off output. It is nevertheless sparse under the
prespecified ranking: only one of the other31 gaps is flagged for off and
none for Router. No phonetic label or hallucination rate follows from an
energy threshold. In particular, the other seven utterances are previously
analyzed development data, not an untouched validation set.

The10720 quiet-gap contribution to backend changes is unusually large
relative to the other seven. This argues for diagnosing specific failure
events and genuine weak-consonant controls, rather than penalizing every
high-frequency residual or building a universal silence gate. The next
bounded diagnostic is a two-arm noisy-condition intervention at the selected
3321 gap, with exact baseline replay and unchanged frontend. Selection by
this screen means the replication remains exploratory and cannot prove
prevalence or unbiased effectiveness of a proposed method.

Exploratory ranking only; no speech or hallucination labels, no learned gate. Other seven are held out with respect to this tail hypothesis, but were used in previous global audits and are not confirmation or virgin cross-validation.

Quiet threshold (three PCM16 steps), minimum run duration (300ms), and backend ranking thresholds (-60dBFS and 10dB above Y_pre) are fixed exploratory choices, not tuned on these results. The active complementary threshold is separately fixed at -70dBFS.

All qualifying gaps appear in gaps.csv. frame_masks.csv includes every frame, boundaries, quiet/active/unclassified masks and exact clean RMS. Gap rank flags require both fixed backend thresholds. Undefined gain never passes the ranking rule.

The prior local probe used 9–21 s clean-active 0.5 s windows, 32 ms overlapping Hann frames and adaptive clean activity. This screen uses full 30 s sources, disjoint 20 ms frames, fixed quiet runs and excluded boundary frames. It does not repeat the prior configuration.

| Fileid | Group | Gaps | Quiet seconds | Off flags | Router flags |
|---|---|---:|---:|---:|---:|
| 5243 | exploratory_other7 | 0 | 0.02 | 0 | 0 |
| 6339 | exploratory_other7 | 4 | 4.64 | 0 | 0 |
| 4812 | exploratory_other7 | 0 | 0.60 | 0 | 0 |
| 11999 | exploratory_other7 | 3 | 2.90 | 0 | 0 |
| 10720 | discovery_10720 | 8 | 7.76 | 2 | 1 |
| 11489 | exploratory_other7 | 6 | 3.92 | 0 | 0 |
| 3321 | exploratory_other7 | 9 | 9.26 | 1 | 0 |
| 3645 | exploratory_other7 | 9 | 6.50 | 0 | 0 |

Clean-active complementary relative L2 metrics use disjoint frames at or above -70 dBFS. Frames above the quiet threshold but below the active threshold are unclassified. Boundary frames are excluded from all aggregate masks. No speech, word, or hallucination label is produced.

## Worker verification and factual aggregate

All8 utterances,48 source hashes and12,000 frame rows verified. All39 gap durations, interior boundaries, clean masks and rank flags match the fixed rules. Direct float64 RMS and independent SciPy periodogram band energies matched source gaps10720 at15.04–16.16s and3321 at22.62–23.66s. No inference or repeated screen was performed.

Other7:31 qualifying gaps totaling21.00s; off has1 flagged gap (3321,22.62–23.66s,1.04s;−59.041dBFS,15.148dB aboveY_pre), Router has0. Thus the fixed off ranking flag is not unique to10720. Discovery10720 has8 qualifying gaps totaling6.38s; off flags15.04–16.16s and18.94–19.40s (1.58s total); Router flags only15.04–16.16s (1.12s). No automatic flag identifies speech or hallucination.

| Group | Arm | Active delta energy | All quiet delta energy | Qualifying gap delta energy | Quiet share of interior delta | Gap share of interior delta |
|---|---|---:|---:|---:|---:|---:|
| discovery_10720 | off | 0.87423986 | 0.21599719 | 0.18422973 | 19.4117% | 16.5567% |
| discovery_10720 | router | 0.51925615 | 0.11654015 | 0.099376504 | 18.0156% | 15.3623% |
| exploratory_other7 | off | 35.298586 | 0.15606112 | 0.067658085 | 0.4274% | 0.1853% |
| exploratory_other7 | router | 19.823913 | 0.078064155 | 0.035142586 | 0.3827% | 0.1723% |

Delta energies above sum squared backend-minus-Y_pre waveform differences and include phase changes. Their fractions use interior active+quiet+unclassified energy; boundary samples are excluded and qualifying gaps are a subset of quiet frames. They are not fractions of human speech errors. All utterances have off active-clean relative L2 aboveY_pre; Router is aboveY_pre in7/8, with6339 the exception. These are descriptive paired waveform distances. The seven comparison items were previously audited and are not fresh confirmation or independent cross-validation.
