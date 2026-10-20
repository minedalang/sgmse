# Public source/N/D transfer evaluation: completed, exploratory

The unique authorized public deployment-transfer evaluation completed with exit0, PID37136/session8361 (completed), in209.653823s. All192 fixed rows completed:64 source,64 N and64 D, in that order. No scores were used to skip later arms. All four metrics are defined for64/64 in each arm, with no missing/error replacement. This is a post hoc evaluation of the same exploratory development set, not new training or independent confirmation.

```text
env/bin/python -B experiments/p8_cpel_public_transfer_20261020/eval_public.py
```

## Absolute public metrics and paired changes

| Metric | Source public | N public | D public | D−N | N−source | D−source | Matched n |
|---|---:|---:|---:|---:|---:|---:|---:|
| normalized_wave_mse | 0.00050799831489 | 0.000505639621835 | 0.000505188819748 | -4.50802086513e-07 | -2.35869305541e-06 | -2.80949514192e-06 | 64 |
| si_sdr_db | 6.28613772906 | 6.3208390667 | 6.32542842951 | +0.00458936281728 | +0.034701337639 | +0.0392907004563 | 64 |
| stoi | 0.782429250459 | 0.782647611431 | 0.782647662676 | +5.12451307387e-08 | +0.00021836097181 | +0.00021841221694 | 64 |
| pesq | 1.89039843716 | 1.88674326427 | 1.88708594628 | +0.000342682003975 | -0.00365517288446 | -0.00331249088049 | 64 |

D−N changes are small: SI-SDR+0.00458936dB, STOI+5.12451e−8, PESQ+0.000342682 and normalized waveform MSE−4.50802e−7. Against source, D has SI-SDR+0.0392907dB/STOI+0.000218412 but PESQ−0.00331249. These descriptive differences do not show comprehensive deployment/content recovery gains. No significance test, bootstrap, threshold selection or best-interface/state choice was performed.

| Comparison | Metric | n | Mean | Median | Positive / negative / zero files |
|---|---|---:|---:|---:|---|
| D-minus-N | normalized_wave_mse | 64 | -4.50802086513e-07 | -1.66727039812e-07 | 12/52/0 |
| D-minus-N | si_sdr_db | 64 | +0.00458936281728 | +0.00310729431527 | 51/13/0 |
| D-minus-N | stoi | 64 | +5.12451307387e-08 | +8.84014919461e-06 | 38/26/0 |
| D-minus-N | pesq | 64 | +0.000342682003975 | +0.000209271907806 | 42/22/0 |
| N-minus-source | normalized_wave_mse | 64 | -2.35869305541e-06 | -9.86012966662e-07 | 9/55/0 |
| N-minus-source | si_sdr_db | 64 | +0.034701337639 | +0.0166084620589 | 47/17/0 |
| N-minus-source | stoi | 64 | +0.00021836097181 | +1.50262170579e-05 | 34/30/0 |
| N-minus-source | pesq | 64 | -0.00365517288446 | +0.000127673149109 | 33/31/0 |
| D-minus-source | normalized_wave_mse | 64 | -2.80949514192e-06 | -1.39390853642e-06 | 7/57/0 |
| D-minus-source | si_sdr_db | 64 | +0.0392907004563 | +0.0212376726283 | 53/11/0 |
| D-minus-source | stoi | 64 | +0.00021841221694 | +1.98045602366e-05 | 36/28/0 |
| D-minus-source | pesq | 64 | -0.00331249088049 | +0.000380575656891 | 34/30/0 |

Means average per-file values equally. Positive MSE differences indicate higher error; positive SI-SDR/STOI/PESQ differences indicate larger metric values. Every original case remains in the saved results. Physical public output was divided by the same original Noisy peak r used for clean normalization; there was no independent gain fitting, external clipping, centering for other metrics, PCM serialization or resampling added outside metric libraries. SI-SDR uses centered projection; standard STOI/PESQ internal processing remains unchanged.

## Actual public path, shared inputs and state identity

The source checkpoint was strictly loaded once with source EMA materialized once, then all parameters frozen. The original source DNN was cloned in CPU memory, and N/D terminal DNN-only files were CPU-loaded exactly once each. All state keys/shapes/dtypes/finite checks and source-checkpoint identities passed; intrinsic Fourier W equalled source. Each arm was strictly injected, and canonical model-state digest matched its CPU packet. No model.eval() after injection overwrote terminal states with source EMA.

| Arm | Expected canonical DNN digest | Actual canonical DNN digest | Match |
|---|---|---|---|
| source | 4813c3ea5659d4c7f9c58e73107d30d93c7cef21e2e64f267671e9f6430592bb | 4813c3ea5659d4c7f9c58e73107d30d93c7cef21e2e64f267671e9f6430592bb | True |
| N | 15e400d6e775930b9784ab49c4a71f43d3acf6df888169c0f44e6532be4b252c | 15e400d6e775930b9784ab49c4a71f43d3acf6df888169c0f44e6532be4b252c | True |
| D | 11d2df47424ea7075bb96e504e1b687514447ad4de3a41eba23e117962e02aaa | 11d2df47424ea7075bb96e504e1b687514447ad4de3a41eba23e117962e02aaa | True |

Nine frontend/intrinsic frozen checks (source loaded, cache complete, each arm injected/completed, finally) all matched the historical N/D frozen digests. Frontend state has267 entries; fixed parameter dnn.all_modules.0.W remains unchanged. All inference preserved the injected DNN state digest. There are no optimizer steps/gradients/new weights or weight copies.

The64 S/X pairs were each PCM-read once (128 reads), center-cropped to32640 with fixed original index/seed. Source frontend ran exactly64 times; its physical output, original physical Noisy and normalized clean were cached in memory and reused unchanged across all arms. Actual public sampler anchor/Noisy-condition descriptors and all sampler arguments match exactly across3 arms for each of64 cases. Actual norm equals cached r, length32640, and raw seed/crop identities match every row. Every real model.enhance call used PC reverse_diffusion/none/N30 with no RRSD/calibrators; actual NFE and Score count are30, totaling192 backend invocations and5760 Score calls.

Sampler/profile hooks observed the original implementation and were restored in finally. No expectedRoundTrip equality gate from the older failed audit was introduced or broadened: this run directly evaluates public outputs. No controlled edit, noise-source read, ASR/DNSMOS, training, package install, network or final-test use occurred. No output waveform/spectrogram arrays were saved.

## Historical audit and raw reference, kept separate

The source public arm reproduces all7 stored partial-audit norms/actual sampler anchor and Noisy descriptors exactly. Max absolute normalized-MSE difference=2.5477360022e-12, retained as arithmetic/normalization roundoff. This is a saved-scalar repeatability observation, not new model calls. The old audit remains statusfailed after7/8 at its frozen tolerance; its eighth case was not retroactively completed, and no old scalar was rewritten.

Historical raw-spectrum normal metrics (reference only; no raw sampler run here):

| System | Raw MSE | Raw SI-SDR | Raw STOI | Raw PESQ |
|---|---:|---:|---:|---:|
| source historical raw | 0.000521001462016 | 5.86471509897 | 0.793958822617 | 1.89985671081 |
| N historical raw | 0.000518464312372 | 5.87682764877 | 0.79434181294 | 1.89310566522 |
| D historical raw | 0.00051811841631 | 5.91640707722 | 0.79446952942 | 1.89458923787 |

Historical raw N/D pair gates failed (raw/precision reductions about3.48%, below10%). Those failures and old M/P failures remain permanent. Public metrics do not replace the old gate, authorize a more favorable interface/state choice, or turn the old screen into success. The current public D−N is a missing deployment-path measurement, not phone gold, human WER, hallucination protection or innovation proof. Source historical300k training-path provenance is still not established by the current source.

## Complete descriptive SNR groups

| Bin | Contrast | n | ΔMSE | ΔSI-SDR | ΔSTOI | ΔPESQ |
|---|---|---:|---:|---:|---:|---:|
| [-10,-5] | D-minus-N | 16 | -5.35253498e-07 | +0.00571736242 | +8.45012095e-06 | +0.000259868801 |
| [-10,-5] | N-minus-source | 16 | -1.9693137e-06 | +0.0129725057 | +0.000948855502 | -0.00153414905 |
| [-10,-5] | D-minus-source | 16 | -2.5045672e-06 | +0.0186898682 | +0.000957305623 | -0.00127428025 |
| [-15,-10) | D-minus-N | 16 | -5.59968595e-07 | +0.00403593667 | +3.63019597e-05 | +0.000122956932 |
| [-15,-10) | N-minus-source | 16 | -1.09016697e-06 | +0.00240778763 | -0.000143868958 | -0.00561121851 |
| [-15,-10) | D-minus-source | 16 | -1.65013556e-06 | +0.0064437243 | -0.000107566999 | -0.00548826158 |
| [-18,-15) | D-minus-N | 16 | -5.62385897e-07 | +0.0100721259 | -6.57366847e-05 | +0.000311717391 |
| [-18,-15) | N-minus-source | 16 | -4.29413484e-06 | +0.0635093686 | +0.00175752717 | +0.00242570788 |
| [-18,-15) | D-minus-source | 16 | -4.85652073e-06 | +0.0735814945 | +0.00169179049 | +0.00273742527 |
| [-20,-18) | D-minus-N | 16 | -1.45600356e-07 | -0.00146797368 | +2.11895846e-05 | +0.000676184893 |
| [-20,-18) | N-minus-source | 16 | -2.08115672e-06 | +0.0599156886 | -0.00168906983 | -0.00990103185 |
| [-20,-18) | D-minus-source | 16 | -2.22675708e-06 | +0.0584477149 | -0.00166788024 | -0.00922484696 |

All four groups are reported without selection. These are repeated exploration groups, not independent confirmatory strata.

## Preservation and validation

All155 Worker-protected files match pre/post:128 actual S/X WAVs, source checkpoint, two terminal N/D weights, core8, current protocol, registry, current goal and necessary old M/P/N/D/audit-failure records. Script source_pre/source_post also match. No other704-WAV scan or new noise PCM access was performed. Source/weight hashes are byte checks; saved tensors remain descriptors. Existing dirty git state is retained in execution_receipt.json. No Root document, prior result, user answer, source or weight was modified.

Independent JSON arithmetic reconstructed192-item arm means and64 matched per-file deltas for all3 contrasts, verified missing coverage, input equality/norm/length/seed/crop,30calls each,5760total and packet/frozen flags. Maximum mean reconstruction discrepancy=8.881784197e-16. This is scalar/hash bookkeeping, not independent GPU or metric inference. All per-file differences and sign counts are in independent_validation.json; original metrics SHA256 `46f554f248efae36cb2509006fc437f4db60bf2030661f593c90fce46f930dac` remains unchanged.

The unique evaluation is complete and stopped. Bridge39452 and listening server16022 remain alive and untouched. No automatic extra training, new checkpoint, interface sweep, human listening request, ASR endpoint or follow-up experiment was started. Raw screening and partial audit failures are preserved alongside this exploratory public result.

