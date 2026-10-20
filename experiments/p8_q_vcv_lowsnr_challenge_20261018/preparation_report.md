# VCV independent low-SNR challenge

Status: prepared_no_inference. This independent stage uses the frozen N1−6dB and N2−6dB recipes, all16 primary identities and all4 speakers per condition. Both conditions failed the previous input eligibility gate; they remain unqualified. The completed +6dB follow-up protocol and results are preserved and SHA locked.

Externally known nominal corpus identities permit evaluating identity after enhancement despite weak input recognition. This does not establish human recoverability, ignored noisy evidence or input identifiability; recognition failure does not prove absence of identity evidence. The VCV context is out of distribution and does not establish natural DNS generalization.

The same pair seeds,80kEMA,N30 frontend/backend path,32frontend+64backend+96raw-output recognition budget,600second task-boundary limit and100MB cap are retained. Clean/noisy scores are reused. Records include transitions from noisy and backend versus y_pre, binary-correct-but24-way-wrong cases, both-correct pairs, collapse, confusion, raw float32 audio/logits and quiet padding energy. No thresholds or samples are tuned after enhancement. PCM16 listening copies are descriptive, with clipping counts.

Corpus attribution: The Consonant Challenge Corpus, DOI10.5281/zenodo.3538223, CC-BY-4.0. No model initialization or inference occurred during preparation. Existing completed/partial outputs are refused.

Run once from repository root: `env/bin/python experiments/p8_q_vcv_lowsnr_challenge_20261018/run_challenge.py`.
