# Utterance-level Oracle over existing 150-file arms

This is a read-only upper-bound diagnostic. It reuses existing per-file metrics for the same 150-file test manifest; no data, checkpoint, or inference was changed. `same_checkpoint` contains the 80k Router, t-cut=0.05, and hard selective arms. `expanded_with_conflict` adds the 1,000-step conflict-aware arm, which is explicitly confounded by training adaptation.

## same_checkpoint

| metric | best fixed | oracle | headroom | choices |
|---|---:|---:|---:|---|
| PESQ_e | 1.689556 (router) | 1.691983 | +0.002427 | {'router': 115, 'tcut005': 15, 'selective': 20} |
| STOI_e | 0.812773 (router) | 0.812983 | +0.000210 | {'router': 115, 'tcut005': 5, 'selective': 30} |
| SI-SDR_e | 7.915450 (router) | 7.915450 | +0.000000 | {'router': 150, 'tcut005': 0, 'selective': 0} |
| DNSMOS_SIG_e | 3.512336 (selective) | 3.515431 | +0.003095 | {'router': 15, 'tcut005': 27, 'selective': 108} |
| DNSMOS_OVRL_e | 3.231617 (selective) | 3.236623 | +0.005007 | {'router': 23, 'tcut005': 32, 'selective': 95} |
| DNSMOS_P808_e | 4.001844 (selective) | 4.008180 | +0.006336 | {'router': 30, 'tcut005': 21, 'selective': 99} |
| DNSMOS_BAK_e | 4.147991 (router) | 4.154388 | +0.006397 | {'router': 57, 'tcut005': 43, 'selective': 50} |

For OVRL with a per-file SI-SDR tolerance of 0.05 dB, 150/150 files had an allowed arm; the oracle mean OVRL was 3.216353, Router mean on those files was 3.212745, delta +0.003608; choices={'router': 91, 'tcut005': 58, 'selective': 1}.

## expanded_with_conflict

| metric | best fixed | oracle | headroom | choices |
|---|---:|---:|---:|---|
| PESQ_e | 1.689556 (router) | 1.693028 | +0.003472 | {'router': 87, 'tcut005': 6, 'selective': 15, 'conflict1000': 42} |
| STOI_e | 0.812773 (router) | 0.812993 | +0.000220 | {'router': 100, 'tcut005': 3, 'selective': 28, 'conflict1000': 19} |
| SI-SDR_e | 7.915450 (router) | 7.915607 | +0.000157 | {'router': 148, 'tcut005': 0, 'selective': 0, 'conflict1000': 2} |
| DNSMOS_SIG_e | 3.512336 (selective) | 3.515741 | +0.003405 | {'router': 12, 'tcut005': 20, 'selective': 105, 'conflict1000': 13} |
| DNSMOS_OVRL_e | 3.231617 (selective) | 3.237112 | +0.005496 | {'router': 19, 'tcut005': 24, 'selective': 91, 'conflict1000': 16} |
| DNSMOS_P808_e | 4.001844 (selective) | 4.008469 | +0.006625 | {'router': 25, 'tcut005': 16, 'selective': 98, 'conflict1000': 11} |
| DNSMOS_BAK_e | 4.147991 (router) | 4.154947 | +0.006956 | {'router': 44, 'tcut005': 34, 'selective': 49, 'conflict1000': 23} |

For OVRL with a per-file SI-SDR tolerance of 0.05 dB, 150/150 files had an allowed arm; the oracle mean OVRL was 3.217800, Router mean on those files was 3.212745, delta +0.005055; choices={'router': 63, 'tcut005': 44, 'selective': 1, 'conflict1000': 42}.

These are unattainable clean-reference upper bounds and cannot demonstrate a deployable router. A small headroom means the existing strategy set cannot support a large learned innovation; a large headroom only justifies learning if features available at inference can predict the choice. The conflict arm should not be interpreted as a pure routing upper bound.
