# VCV enhancement follow-up

Status: prepared_no_inference. The fixed input gate selected N1+6dB and N2+6dB. All16 frozen primary identities and all4 speakers are retained for each condition, giving32 inputs. This is a recognizer-conditioned positive control, not an unbiased enhancement benchmark or a low-SNR recovery claim. +6dB lies outside the historical nominal training SNR range, and the corpus VCV tokens are out of distribution.

The driver loads80kEMA once, uses the same frontend/backend path and sampler settings as the previous SNR control, verifies runtime flags, runs32 frontends and64 backends, then releases the enhancement model before loading the existing phoneme recognizer for96 raw-output forwards. Clean/noisy scores are reused. Exact float32 input hashes, all8 pair seeds shared across identities/noises/arms, raw outputs, PCM16 listening copies, clipping counts and quiet padding energies are recorded. No human listening judgment is requested tonight.

Source corpus: The Consonant Challenge Corpus, DOI10.5281/zenodo.3538223, CC-BY-4.0. Nominal labels are known corpus identities rather than model truth. No human recoverability, WER or DNSMOS claim is made. Padding energy is not a word-hallucination metric.

Run once from repository root: `env/bin/python experiments/p8_q_vcv_enhance_follow_20261018/run_follow.py`. Existing outputs, including partial runs, are refused. The10minute budget is checked before each new task; an active bounded forward is allowed to finish.
