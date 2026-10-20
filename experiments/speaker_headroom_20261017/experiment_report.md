# Independent CAM++ speaker headroom probe

Existing 32-utterance / 16-speaker repeated-speaker CV split; first 32,000 samples; no training or new audio.

| arm | paired clean cosine | same-speaker cosine | different-speaker cosine | AUC | EER |
|---|---:|---:|---:|---:|---:|
| noisy | 0.531977 | 0.292115 | 0.077538 | 0.796484 | 0.246875 |
| off | 0.668360 | 0.444714 | 0.117805 | 0.955859 | 0.125000 |
| router | 0.670816 | 0.443383 | 0.122451 | 0.952995 | 0.125000 |
| pair | 0.669821 | 0.443089 | 0.119232 | 0.954688 | 0.125000 |
| proto01 | 0.668902 | 0.445883 | 0.119536 | 0.953776 | 0.125000 |
| proto01_cw1 | 0.670082 | 0.442105 | 0.121974 | 0.951953 | 0.125000 |
| masked | 0.669955 | 0.446302 | 0.119858 | 0.955599 | 0.125000 |
| clean | 1.000000 | 0.622426 | 0.090274 | 0.999349 | 0.003125 |

## Paired clean cosine by estimated input SNR

| SNR stratum | n | mean SNR (dB) | noisy | off | router | clean |
|---|---:|---:|---:|---:|---:|---:|
| <=-15 | 8 | -19.70 | 0.185348 | 0.459399 | 0.462251 | 1.000000 |
| (-15,-10] | 2 | -12.47 | 0.251245 | 0.450072 | 0.453322 | 1.000000 |
| (-10,-5] | 7 | -7.18 | 0.458430 | 0.601240 | 0.601849 | 1.000000 |
| >-5 | 15 | 12.56 | 0.788598 | 0.840235 | 0.843235 | 1.000000 |

`clean` is the clean-clean reference. Same-speaker and different-speaker pairs use all unordered pairs; each speaker has two utterances.

No dataset or checkpoint was modified.
