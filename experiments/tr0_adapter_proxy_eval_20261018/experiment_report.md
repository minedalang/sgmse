# TR-0 fresh-CV single-step proxy — 2026-10-18

This evaluation uses the locked fresh 32-speaker × 2-utterance manifest. Each utterance is deterministically center-cropped to the 32,640-sample training window. For each utterance, baseline and all four scaled-screen adapter states receive the same `t`, noise tensor, frontend output, and noisy condition. The endpoint is clean-reference single-step complex-spectral `x0` MSE.

This is a clean-aware diagnostic proxy, not a deployment metric. It does not run the PC sampler and does not produce DNSMOS, PESQ, STOI, SI-SDR, speaker, or content claims. A lack of proxy improvement is a stop signal for the current adapter screen; a proxy improvement would only justify full waveform evaluation.
