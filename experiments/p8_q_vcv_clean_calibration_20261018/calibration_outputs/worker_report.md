# Clean calibration execution and validation

Completed once with repository env/bin/python, CUDA on NVIDIA L40S, using the Research Lead approval reference preserved in frozen_inputs.json. No further downloads, training, noise or enhancement experiments were performed. The first timing launcher failed because /usr/bin/time was absent, before Python executed or calibration_outputs existed; Bash builtin time then launched the calibration exactly once.

The preset gate passed. Primary 24-way top-1 and binary forced-choice results were both 16/16, with k/g 8/8 and t/d 8/8 in both tasks. Every speaker scored 4/4. Primary confusion counts are k→k=4, g→g=4, t→t=4, d→d=4; there were no ties or primary errors. Mean binary log-likelihood margin was 3.7582256794 (k/g 3.3891761303, t/d 4.1272752285), range 1.7071924210 to 5.0475654602.

The 12 descriptive supplementary clips scored 7/12 for both tasks, with mean binary margin 2.4913618962 and range −2.3388023376 to 6.3672304153. Their top-1 confusion counts were k→k=3, t→t=3, d→d=1, g→k=1, d→t=3, g→d=1. Errors: m2ugu1 g→k, m4udu1 d→t, m4udu2 d→t, m8ada1 d→t, m8igi1 g→d. Supplementary clips were excluded from the gate.

Measured process wall time was 8.99 seconds. Script runtime was 3.5277204514 seconds, including hashing, model loading, CUDA model evaluation, CPU CTC scoring and writes after imports. CUDA-only forward time was not instrumented and cannot be reported separately; no rerun was made to recover it.

worker_validate.py independently verified all 28 arrays and source WAV hashes, all 24 candidate scores and token mappings per clip, saved probabilities against logits, predictions, margins, grouped summaries, confusion matrix and gate. Recomputing all CTC sequence likelihoods from saved logits produced maximum absolute difference 0.0. All original WAV hashes and the frozen manifest, protocol and calibration script hashes are unchanged. Artifacts occupy approximately 6.62 MB across the experiment, below the 25 MB calibration cap.

These results describe this controlled clean corpus diagnostic and do not establish human accuracy, noisy discrimination or generalization. The next scientific decision remains with the Research Lead.
