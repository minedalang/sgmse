# Common-scale factorial conditional diagnostic: prepared only

This is the single bounded follow-up to the completed fixed-native-reference experiment. Four donor cases there exceeded the native normalized input range. This design uses one `max(g_native_peak, k_native_peak)` reference for all four anchor/noisy cells within each speaker/noise group. Conflicting anchor/noisy identities remain jointly OOD. Whole VCV recordings change, including coarticulation and duration. No further design iteration is planned after this experiment, regardless of outcome.

Sources are unchanged: f6 and m5 k/g pairs, N1/N2 at -6 dB, frozen physical noise and gains, original pair seeds and 80k EMA checkpoint. The eight cached frontend float32 arrays come from the verified prior experiment, whose eight native helper diagonals passed exact raw and PCM replay. New driver verifies cached frontend hashes and their prior native anchor provenance. No frontend forward, new input, seed, Router, training or download is introduced.

The original 43-line `fixed_norm_off.py` is imported unchanged from `experiments/p8_vcv_fixed_norm_condition_swap_20261019/`; SHA256 `66f61a892e5b4ffc37c1069e3692618a07fb09efc3717c106c1d18c212ddbb82`. Current runtime/model/core/checkpoint and source arrays are locked. Runtime asserts retain OUVE PC, mode2 noisy concatenation, no cross attention/FiLM/selector/residual refinement, and OFF without calibrators. Every call requires NFE30 and 30 traced steps.

For each of four groups, k has the maximum native peak. All four k-anchor/k-noisy common-reference cells must match prior raw arrays, NPY hashes and PCM hashes before any of the remaining twelve calls. Native g/g cells now use a changed reference; their difference from native-own-reference outputs is recorded descriptively and is not a failed replay gate. Within each anchor row, seed, anchor spectrum and initial sampler state must match across noisy identities. The normalized physical noise is identical across all four cells. Float32 decomposition rounding error is bounded and disclosed. Each normalized condition peak must remain at most one, without clipping or renormalization.

Sixteen OFF backend calls and sixteen new CTC forwards are expected. Raw float32, PCM listening copies, 24 candidate scores and float logits/probabilities are saved. Conflict cells are labeled by anchor origin and noisy donor; no correctness endpoint is assigned. Primary response is `q = LL(g) - LL(k)`. For each group, report these descriptive contrasts with the first subscript denoting anchor:

- Anchor effect: `((qgg - qkg) + (qgk - qkk)) / 2`.
- Condition effect: `((qgg - qgk) + (qkg - qkk)) / 2`.
- Interaction: `(qgg - qgk) - (qkg - qkk)`.

There are no p-values, population claims, human recovery or true PER claims. Previous native-reference rows are compared descriptively with their distinct scales explicitly recorded. Source attribution: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0.

CPU-only validation in `/data/coding/sgmse/env/bin/python` (NumPy 2.2.6): syntax passes, all 156 locks match, unchanged helper verified, eight recipe input hashes match, all four k-native reference gates apply, all eight normalized condition peaks are at most one. JSON smoke roundtrips eight guard structures, a NumPy scalar and a Path; NaN remains rejected with `allow_nan=False`. No model load or forward has run. Preparation records are `preparation_metrics.json` and `preparation_guard_smoke.json`.

Worker command after review, from `/data/coding/sgmse`:

```bash
PATH=/data/coding/sgmse/env/bin:$PATH PYTHONDONTWRITEBYTECODE=1 /data/coding/sgmse/env/bin/python -B experiments/p8_vcv_common_scale_factorial_20261019/run_factorial.py
```

The outputs directory must not exist. Caps are 600 seconds between tasks and 30 MB of experiment artifacts. Failure retains outputs and diagnostics; no unchanged retries. No additional normalization, sweep or design iteration is authorized by this driver.
