# Speech-component ablation

Completed in 19.21952 seconds. Calls: 0 frontend, 10 OFF backend (N30), 5 new CTC; no training, downloads, Router, extra cases or inference retries. Five exact native RAW/NPY/PCM replays passed before any noise-only call. Config remains the immutable prepared snapshot; completed state is recorded in metrics.json and outputs/COMPLETED.json.

These are five previously selected natural nominal-label correctness transitions, excluding f6/m5. Native and ablation pairs share cached raw/normalized frontend anchor, original noisy reference, physical noise amplitude and normalized noise geometry, seed, CPU/CUDA RNG and initial sampler state. Only the noisy condition loses its entire known clean VCV speech component. Runtime mode2/noisy concat/OUVE PC/OFF guards passed, with NFE30 and trace30 for all ten outputs.

|Case (carrier SNR −6 dB)|Kind|Nominal|Frontend|Native OFF|Noise-only|Descriptive outcome|Native target−rival|Noise-only target−rival|Δ margin (noise-only−native)|
|---|---|---|---|---|---|---|---:|---:|---:|
|original N1 m2ada1|repair|d|t|d|t|lost|0.892603|-0.090029|-0.982632|
|original N1 f11ata1|harm|t|t|k|k|remained_original_error|-0.467913|-5.958050|-5.490137|
|original N2 m2aga1|harm|g|g|k|g|reverted_to_nominal|-0.740393|0.485981|+1.226374|
|fresh N1 f20aka1|repair|k|t|k|n|lost|1.817366|-3.467693|-5.285059|
|fresh N2 f20aka1|harm|k|k|ch|ch|remained_original_error|-3.258408|-3.146841|+0.111567|

Both selected repairs are lost; two selected harms retain their original error class and one reverts to the nominal identity. Fixed rivals are t for both repairs, k for f11ata1 and m2aga1 harms, and ch for the f20aka1 harm. Target-versus-frontend-label margins for all three harms are identically zero because frontend label equals nominal target; these are explicitly marked degenerate. The fixed-rival margins above remain informative. In particular, f11ata1 stays k while the native speech component supports the target-versus-k margin by +5.490137 relative to noise-only. Unchanged top1 therefore does not imply the condition is unused.

|Case|Noise-only normalized peak|Output RMS difference|Output maximum absolute difference|
|---|---:|---:|---:|
|original N1 m2ada1|0.556843|0.00319047|0.04249004|
|original N1 f11ata1|0.434289|0.00366878|0.06572814|
|original N2 m2aga1|0.877118|0.00341095|0.06189110|
|fresh N1 f20aka1|0.316776|0.00357931|0.10170653|
|fresh N2 f20aka1|0.532948|0.00360803|0.10109259|

All ten normalized condition peaks are ≤1 and every output has zero PCM clipping. Float32 speech-removal residuals lie within their recorded bounds, with nonzero changes over active source support. Quiet pre/post padding RMS and true-support-outside RMS remain separately recorded in metrics.json and outputs/audio_manifest.json; neither energy nor classifier labels measure hallucination rate.

CPU-only independent validation passed: 166 immutable locks, ten raw/WAV pairs with exact hashes and byte regeneration, five native/source equalities, paired anchor/reference/RNG/xT/noise and sampler guards, all fixed rivals/margins/outcomes, and exact recomputation of all 24 CTC likelihoods plus probabilities from 15 saved logits caches (five ablations and ten prior frontend/OFF). See validate_artifacts.py and artifact_validation.json. No model was initialized by validation. Initial validator assertions were refined after diagnosing libsndfile near-threshold PCM conversion (two samples differed from ideal float64 floor by <1 LSB) and a NumPy/CUDA normalized-peak difference of 5.96e−8; exact WAV regeneration, ≤1 LSB quantization error and float32 epsilon peak tolerance pass. These were checker assumptions, not failed experimental runs.

Artifacts remain below the 30 MB cap (about 3.85 MB including validation/report), and runtime is below 600 seconds. Source/core/checkpoint/classifier/cache files and the frozen config/selection/driver are unchanged; pre-existing dirty files were preserved. Reproduction command, from /data/coding/sgmse (the driver intentionally refuses existing outputs):

```bash
PATH=/data/coding/sgmse/env/bin:$PATH PYTHONPATH=/data/coding/sgmse PYTHONDONTWRITEBYTECODE=1 /data/coding/sgmse/env/bin/python -B experiments/p8_vcv_speech_component_ablation_20261019/run_ablation.py
```

This is a clean-aware, whole-VCV-removal OOD mechanism diagnostic, not a surgical consonant intervention. Nominal corpus identity and a single recognizer are proxies, with no human PER or listening judgment. Selection used existing outcomes, so these five cases do not estimate population repair/harm rates. Lost/retained repairs and reverted/retained/third-class harms are descriptive; they do not by themselves establish hallucination, independent noisy recoverability or deployable gain. Removing speech also changes joint consistency, duration and coarticulation. Research interpretation and next experiment remain with Research Lead.

Source: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0. All new raw/PCM audio and new logits/probabilities/candidate arrays are retained under outputs/; prior frontend/OFF scores are reused with frozen hashes. No experiment failures occurred.
