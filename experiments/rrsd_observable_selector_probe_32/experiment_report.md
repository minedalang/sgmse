# Observable utterance-policy selector probe

This read-only probe asks whether cheap noisy-waveform statistics can select among already generated Router/t-cut arms. It uses 16-speaker/32-utterance CV, leave-one-speaker-out validation, and no clean signal in the features. Labels use clean-based DNSMOS OVRL subject to a per-file SI-SDR tolerance; labels are validation diagnostics, not deployable supervision.

## SI-SDR tolerance 0.05 dB

Fixed OVRL: Router 3.185399, cut005 3.188660, cut010 3.195690; clean-reference oracle 3.186977 (labels={'router': 22, 'cut005': 10, 'cut010': 0}).

LOSO decision tree: OVRL 3.185760, SI-SDR 7.790093, choices={'router': 25, 'cut005': 7, 'cut010': 0}. Logistic: OVRL 3.185610, SI-SDR 7.792523, choices={'router': 27, 'cut005': 5, 'cut010': 0}.

## SI-SDR tolerance 0.1 dB

Fixed OVRL: Router 3.185399, cut005 3.188660, cut010 3.195690; clean-reference oracle 3.189294 (labels={'router': 11, 'cut005': 19, 'cut010': 2}).

LOSO decision tree: OVRL 3.188702, SI-SDR 7.741625, choices={'router': 5, 'cut005': 24, 'cut010': 3}. Logistic: OVRL 3.187368, SI-SDR 7.753562, choices={'router': 8, 'cut005': 23, 'cut010': 1}.

Exit rule: if both feature selectors fail to exceed the best fixed policy and remain far below the clean-reference Oracle, this does not justify a learned policy head; retain the fixed t-cut as a simple Pareto baseline.
