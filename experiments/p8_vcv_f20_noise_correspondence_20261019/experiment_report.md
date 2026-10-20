# Fixed case5 noise correspondence audit

CPU-only saved-waveform audit. The physical float64 N2 noise, recorded gain, normalized-noise hash, clean/noisy/noise-only condition hashes, and prior audit locks all match. Native and ablated local conditions are bitwise identical in both fixed tails; clean is exactly zero. No inference, training, downloads or audio outputs.

Projection uses signed zero-alignment inner products, no mean subtraction or lag fitting. Each band is selected from the same unwindowed finite-interval DFT. Lowband is [0,4000) Hz, highband [4000,7500] Hz; fullband also contains >7500 Hz. Weighted Parseval products are mean products in physical output scale. Rho squared is the fraction explained by one unconstrained signed scalar times the original noise. LS residual orthogonality, energy decomposition, bounds, fullband time-domain agreement and identity/opposite/zero checks pass.

|Interval start sample|Arm|Band|Signed rho|LS gain|Projected power %|Residual mean square|Power-spectrum cosine|
|---:|---|---|---:|---:|---:|---:|---:|
|14881|clean|fullband|undefined|0|undefined|0|undefined|
|14881|clean|lowband|undefined|0|undefined|0|undefined|
|14881|clean|highband|undefined|0|undefined|0|undefined|
|14881|y_pre|fullband|0.0638271897|0.00011745249|0.407391014|1.78228623e-09|0.00773147327|
|14881|y_pre|lowband|0.134776211|0.00070748665|1.8164627|1.35100306e-09|0.0408665899|
|14881|y_pre|highband|0.0577299646|5.43462772e-05|0.333274881|4.06013569e-10|0.0477260917|
|14881|native_off|fullband|0.00844170745|6.30586186e-05|0.00712624247|2.94872787e-08|0.00966174669|
|14881|native_off|lowband|0.024408936|0.000589415237|0.0595796158|2.91000566e-08|0.0762660088|
|14881|native_off|highband|0.00939370234|8.39804884e-06|0.00882416437|3.67365374e-10|0.206832551|
|14881|ablated_off|fullband|0.0115971027|9.47101384e-05|0.013449279|3.52429992e-08|0.0593401967|
|14881|ablated_off|lowband|0.0240807567|0.000523486557|0.0579882845|2.35844926e-08|0.0849826242|
|14881|ablated_off|highband|0.0104255097|5.20933589e-05|0.0108691253|1.14756228e-08|0.463398844|
|14881|ablated_minus_native|fullband|0.00641436187|3.16515198e-05|0.00411440382|1.28677225e-08|0.399661008|
|14881|ablated_minus_native|lowband|-0.00855317876|-6.592868e-05|0.00731568669|2.96667201e-09|0.233256605|
|14881|ablated_minus_native|highband|0.00948878801|4.36953101e-05|0.0090037098|9.74680776e-09|0.451387573|
|16481|clean|fullband|undefined|0|undefined|0|undefined|
|16481|clean|lowband|undefined|0|undefined|0|undefined|
|16481|clean|highband|undefined|0|undefined|0|undefined|
|16481|y_pre|fullband|0.0436820611|5.63586532e-05|0.190812246|8.25258692e-10|0.00294726257|
|16481|y_pre|lowband|0.0959604085|0.000387900906|0.92084|7.91042353e-10|0.0206927446|
|16481|y_pre|highband|0.0763081378|1.78675985e-05|0.58229319|2.34861686e-11|0.223364785|
|16481|native_off|fullband|0.00583080235|3.65067791e-05|0.00339982561|1.94706409e-08|0.00790669468|
|16481|native_off|lowband|0.0160917359|0.000318143702|0.0258943966|1.90936256e-08|0.0604438932|
|16481|native_off|highband|0.00659055784|6.10056184e-06|0.00434354526|3.69176653e-10|0.204777118|
|16481|ablated_off|fullband|0.0107128372|8.57191913e-05|0.011476488|3.1798153e-08|0.0743970956|
|16481|ablated_off|lowband|0.0196850327|0.000390358088|0.0387500512|1.92063913e-08|0.0858862197|
|16481|ablated_off|highband|0.0103163495|5.5341762e-05|0.0106427066|1.23983938e-08|0.468958076|
|16481|ablated_minus_native|fullband|0.00970936419|4.92124122e-05|0.00942717529|1.27594336e-08|0.433651596|
|16481|ablated_minus_native|lowband|0.0113201021|7.22143866e-05|0.0128144711|1.98815904e-09|0.228157322|
|16481|ablated_minus_native|highband|0.00992619689|4.92412002e-05|0.00985293848|1.06024558e-08|0.456943765|

Clean has exactly zero energy: correlation, projected fraction and power-spectrum cosine are undefined, while LS gain and residual power are zero. Spectral cosine compares powers only and cannot establish phase-aligned retention.

A high signed correlation supports only a zero-alignment, linearly coherent component. Low correlation leaves nonlinear filtering, time-varying gains and phase changes unresolved; it cannot prove a new phoneme or semantic hallucination. A scalar projection does not identify the entire generation mechanism. This is one previously selected, clean-aware OOD whole-VCV condition ablation with the speech-containing frontend anchor fixed. The same listener spectral followup is provenance, not an independent label. No new classifier or listener measurement is introduced.

Reproduce by copying audit_noise.py into a fresh experiments subdirectory and running /data/miniconda/envs/torch/bin/python audit_noise.py. Completed audit CPU wall time: 0.141184 s. Source files unchanged. Git status unchanged across numerical run: True (concurrent work may affect repository-level status). Exact full precision results and hashes are in metrics.json.


The highband increase has little fixed-zero-lag scalar correspondence to the original noise: ablated rho is 0.01042551 / 0.01031635 in the two intervals, with only 0.0108691% / 0.0106427% of its highband power projected onto the original noise. Native values are 0.00939370 / 0.00659056 (0.00882416% / 0.00434355%). For ablated minus native, the fractions are 0.00900371% / 0.00985294%. Ablated highband power-spectrum cosine is 0.463399 / 0.468958, versus native 0.206833 / 0.204777; these phase-insensitive power similarities do not establish waveform retention. Lowband native/ablated rho remains small (0.02440894 / 0.02408076 first interval; 0.01609174 / 0.01968503 guarded). Thus the larger measured highband power is chiefly outside this single aligned-noise projection; it remains unresolved whether nonlinear or phase-changing noise processing explains the sound. No semantic insertion claim follows.

Independent verification: verify_noise.py reconstructs each selected band with inverse-RFFT and recomputes all 30 arm × interval × band cases in the time domain. Signed rho, LS gain and residual power agree, including explicit zero-energy clean handling; all original source locks pass again. Run audit_noise.py then verify_noise.py in a fresh destination. Both scripts refuse overwriting their completed results.
