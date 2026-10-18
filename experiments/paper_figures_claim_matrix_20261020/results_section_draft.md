# Draft results text

All results below use the locked DNS Ins20 synthetic test manifest, with 150 no-reverb and 150 official with-reverb utterances. Inference used the same 80k EMA checkpoint, PC sampler with 30 function evaluations, SNR 0.5, and seed 2026 for every arm. The Router was compared with Router-off and with a fixed early-time cutoff (`t_cut=0.05`).

On the pooled 300-utterance set, Future guidance increased SI-SDR by 0.241 dB (95% bootstrap CI [0.202, 0.282]) and changed PESQ by only +0.0058 (CI crossing zero). The same comparison reduced STOI by 0.0008, DNSMOS-SIG by 0.0149, and DNSMOS-OVRL by 0.0145; the corresponding intervals excluded zero for these three measures. Thus the measured gain is a reconstruction improvement accompanied by a small perceptual-proxy cost under the matched protocol.

The effect depended on the acoustic condition. In the no-reverb split, Router increased PESQ by 0.0264, STOI by 0.0005, and SI-SDR by 0.319 dB. In the official with-reverb split, Router increased SI-SDR by 0.164 dB while reducing PESQ by 0.0147, STOI by 0.0021, and DNSMOS-OVRL by 0.0287. A matched-file interaction analysis quantified the change in the Router effect under reverberation as -0.0411 PESQ, -0.0027 STOI, -0.155 dB SI-SDR, and -0.0285 DNSMOS-OVRL; the 95% intervals for these four interactions excluded zero.

The fixed cutoff preserved the same split-dependent pattern and did not remove the interaction. We therefore use it as a transparent engineering control rather than as evidence of a learned adaptive gate.

These results support a conditional perception–reconstruction trade-off and motivate the subsequent mechanism-boundary analyses. They do not identify a causal reverberation mechanism, establish a global perceptual improvement, or validate a second independent algorithmic innovation.

## Data-source clarification

The `with-reverb` split in this section is the official DNS Ins20 test directory `test_set/synthetic/with_reverb`, whose README states that its clean clips were reverberated with internally recorded room impulse responses. It is not a training split. A separate four-utterance fixed-FIR CV smoke is archived under `experiments/reverb_cv_smoke_20261018/`; it is exploratory and does not reproduce the official interaction.
