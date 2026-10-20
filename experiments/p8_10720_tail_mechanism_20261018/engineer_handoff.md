# Worker handoff

Run once from the repository:

```bash
/data/coding/sgmse/env/bin/python experiments/p8_10720_tail_mechanism_20261018/probe.py
```

This preparation did not run the acoustic experiment or model inference. The script uses only NumPy, SciPy, SoundFile, PyYAML, and Matplotlib; it never loads model code or a checkpoint. Existing WAVs and prior experiment records are read only.

Engineer checks on synthetic arrays passed: Python syntax, exact zero RMS for silence, dB floor applied after RMS, and signed lag recovery for -80, 0, 37, and 80 sample shifts. All archived-source checks are performed by the worker run against the 15-entry SNR control manifest, including source hash, decoded source hash, source path consistency, listening clip hash, and exact slice equality. Source hashes are checked again at completion.

Exclusive file creation protects existing outputs. `run_started.json` also serves as a run lock. A failed run writes `failure.json`; diagnose before any retry. Do not delete a previous attempt or rerun without recording why. Runtime writes are confined to this experiment directory and create the required `metrics.json` and `experiment_report.md` beside `config.yaml`.

Frame coordinates use center times; summaries assign frame centers to half-open intervals. Every 20 ms frame in the full five second context remains in `frames.csv`; periodicity uses a centered 40 ms window from the full source. A displayed frame can overlap a fixed boundary by 10 ms. There are 499 complete 20 ms frames in the five second context, and the reported tail has 70 frame centers. Raw fullband wave correlation uses all 80,000 samples in the extended context.

`clean` denotes x; `noise_source` is the exact archived noise; `scaled_noise` uses the alpha stored in SNR control metrics. Both noise representations are retained for every variant. Near-floor flags use a fixed conservative threshold of three PCM16 steps RMS; this is not a learned activity threshold. No normalized spectral match is accepted when the query or clean reference is at/below that threshold. Unfiltered winners and flags remain visible for diagnosis. Correlation and spectrum matching cannot prove causality or timing displacement. The Research Lead interprets results.
