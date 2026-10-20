# Worker execution and verification

Completed the authorized single run without failures or retries. Original baseline inference was reused. No training, ASR, DNSMOS, downloads or browser runtime tests were performed. Listening judgments have not been collected.

Listening page: http://127.0.0.1:8766/p8_10720_snr_control_20261018/listening.html

Local HTTP returned 200 and page bytes matched disk. All 15 buttons map to the correct variant/arm manifest entry, clip source path and served disk bytes. The HTML contains one audio player. This checks routing and source PCM; actual browser interaction is not claimed verified. All clips contain source samples [220000:252000] at 16 kHz, exactly after the specified FLOAT serialization. Original five baseline hashes and checkpoint were verified unchanged.

| Variant | Full 30 s SNR dB | Focus SNR dB |
|---|---:|---:|
| original | -15.99988052 | -18.24421037 |
| noise_minus10dB | -5.99988052 | -8.24421037 |
| noise_minus20dB | 4.00011948 | 1.75578963 |

| Variant | Arm | NFE | Wall seconds | Output SHA256 |
|---|---|---:|---:|---|
| noise_minus10dB | off | 30 | 9.177 | `88f4aea51a61c9053c7e0e2b58f84a2449cfaeb162c652e05016595b159bcd65` |
| noise_minus10dB | router | 30 | 9.293 | `e52b8e947d2c19654dd675ed95570e66f48a67913e5fe48efb8eef4116b30b38` |
| noise_minus20dB | off | 30 | 9.218 | `e1f3bc73edb0908db91bb35716341bcd29422ba1807fbde2bc51865705a28c4c` |
| noise_minus20dB | router | 30 | 9.300 | `9705c3952ec54017d6a3a57bf6d5549e75ad3e365e95ddccef88cfea918804ee` |

Total run wall 48.254 s; backend enhancement 36.988 s; frontend 0.971 s. All four full outputs: 480000 mono samples, 16000 Hz, PCM16, effective seed 2997368683, no output saturation. Inputs use FLOAT. Audit size before validation metadata 11,361,892 bytes, below 50 MB.

Reproduction/provenance: config.yaml, run_control.py, launch_provenance.json, metrics.json and source_clip_manifest.json. Independent routing, PCM and HTTP checks: worker_validation.json. This is an offline oracle noise attenuation control at fixed clean gain 1. It does not establish original low SNR speech recoverability or deployable enhancement quality.
