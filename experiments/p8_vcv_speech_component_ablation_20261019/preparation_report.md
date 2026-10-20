# Speech-component ablation: prepared only

This is a distinct speech-component mechanism diagnostic, not another normalization/seed variant on f6/m5. The old low-SNR and fresh-speaker primary cohorts were exhaustively enumerated for y_pre/OFF 24-way nominal-label correctness transitions at -6 dB, excluding all f6/m5 recordings. Exactly five cases remain and are frozen in `selection_frozen.json`, including the complete screened index:

|Source|Condition|Recording|Nominal target|Frontend|OFF|Transition|Fixed rival|
|---|---|---|---|---|---|---|---|
|Original|N1 -6|f11ata1|t|t|k|Harm|k|
|Original|N1 -6|m2ada1|d|t|d|Repair|t|
|Original|N2 -6|m2aga1|g|g|k|Harm|k|
|Fresh|N1 -6|f20aka1|k|t|k|Repair|t|
|Fresh|N2 -6|f20aka1|k|k|ch|Harm|ch|

Selection uses externally nominal corpus labels and fixed three-token CTC candidates. These are selected transitions, not prevalence, human judgments or independently confirmed phoneme correctness. The five labels and outcomes were frozen before any ablation.

Each case reuses its original cached frontend raw float32 array, original noisy peak reference and original pair-hash seed. The intervention replaces only the noisy condition with `float32(noise_gain * frozen_unit_noise)` for the full two-second recording. The known clean speech component is thereby removed, while physical noise and its normalization are held fixed. The anchor stays unchanged. This is whole VCV speech removal, including coarticulation and duration, not a surgical consonant edit. Joint disagreement is OOD; normalized noise peaks may exceed one and are recorded without clipping or renormalizing. No added variants or cases are allowed.

All five native helper outputs must match prior raw arrays (`np.array_equal`), NPY SHA256 and PCM SHA256 before any of the five noise-only calls. Replays and ablations use the unchanged verified 43-line helper imported from `experiments/p8_vcv_fixed_norm_condition_swap_20261019/fixed_norm_off.py`, SHA256 `66f61a892e5b4ffc37c1069e3692618a07fb09efc3717c106c1d18c212ddbb82`. Native/ablation pairs must match raw and normalized anchor, reference, RNG states, seed, initial sampler state and normalized physical noise hashes. OUVE PC/mode2/noisy concat/OFF runtime guards, NFE30 and 30 trace steps are retained. No new frontend calls are needed.

Five native OFF scores are reused; only five ablation outputs receive new CTC forwards. New float logits/probabilities and all 24 candidate scores are retained, along with raw/PCM audio, source and quiet padding RMS. The target remains the original recording's nominal clean identity even when the condition contains only noise. Repairs are described as retained/lost; harms as original error retained, reverted to nominal, or third class. These are CTC proxy transitions, not hearing, true PER, deployable enhancement or population claims.

A fixed target-versus-rival loglikelihood margin uses the original frontend error label for repairs and the original OFF error label for harms. Target-versus-frontend-label margins are also recorded explicitly; the three harms have frontend label equal to target, so those margins are identically zero and marked degenerate. All24 likelihoods remain available, with no candidate-count statistics or outcome-driven rival changes.

CPU preparation in `/data/coding/sgmse/env/bin/python`, NumPy 2.2.6, passes syntax and all 163 immutable source/model/checkpoint/classifier/cache locks. All five frozen recipes regenerate exactly; known speech removal is bounded by recorded float32 error and changes active speech samples rather than only quiet padding. Cached frontend and OFF 24-candidate scores exactly match saved candidate arrays. JSON smoke roundtrips all five guard structures plus NumPy scalars and Paths, and rejects nonfinite values. No model was loaded or run during preparation. See `preparation_metrics.json` and `preparation_guard_smoke.json`.

Worker command after review, from `/data/coding/sgmse`:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH PYTHONPATH=/data/coding/sgmse PYTHONDONTWRITEBYTECODE=1 /data/coding/sgmse/env/bin/python -B experiments/p8_vcv_speech_component_ablation_20261019/run_ablation.py
```

Expected calls: frontend 0, OFF backend 10, new recognizer 5. Caps: 600 seconds checked between tasks, 30 MB artifact budget. Outputs must not already exist; failures preserve partial outputs and diagnostics. No training, downloads, Router, parameter search or extra cases. Source: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0.
