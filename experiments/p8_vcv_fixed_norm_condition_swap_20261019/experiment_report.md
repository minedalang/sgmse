# Fixed normalization VCV condition swap

## Research Lead interpretation

All eight swapped binary g/k decisions agree with the nominal identity of
the recording that produced the frozen anchor, not the noisy donor. That
nominal identity is not necessarily the frontend recognizer's decoded
identity: the f6 g-source frontend can be decoded as k. Residual identity
cues or learned priors may explain the output; this test does not distinguish
them. Six cross outputs retain the anchor-origin class among all24
candidates; two choose a third class. Five likelihood-ratio changes move
toward the donor label and three move away, with no consistent identity
switching. Noisy conditioning still changes the trajectory and waveform.

Cross-cell nested correctness is donor-label agreement, not enhancement
accuracy: conflicting anchor/noisy identities have no uniquely verified clean
target. Four donor peaks exceed one, so a single common-reference factorial
control is registered separately to remove that range confound. After that,
stop normalization/seed variants on these two selected pairs regardless of
outcome. Neither conflict test alone establishes a natural-input failure,
deployable correction or new method.

Status: completed. Corrected run: 26.49 seconds; 8 native frontend, 16 OFF backend (NFE30 and 30 traced steps each), 8 new wav2vec2 recognizer forwards. No training or downloads.

All eight native outputs exactly match prior raw float32 arrays, NPY file hashes and PCM hashes before any cross inference. Every cross preserves its row reference, seed, CPU/CUDA RNG state, normalized anchor and traced initial xT. Physical noise waveform and gain match between g/k donors; normalized noise and bounded float32 decomposition were independently checked for all eight rows. Reference scales differ between rows.

q = CTC log likelihood(g) − log likelihood(k), scored over the complete two-second waveform. The final column q(noisy g) − q(noisy k) is positive when the g donor moves the score in the nominal g direction. All rows below are descriptive.

| Noise | Anchor | Native 24-class / gk | Swapped 24-class / gk | Swapped relation | q native | q swapped | q(noisy g)−q(noisy k) | Donor normalized peak |
|---|---|---|---|---|---:|---:|---:|---:|
| N1_snr-6 | f6aga1 | g / g | g / g | follows anchor | 4.48946 | 4.43570 | +0.05376 | 1.04266 |
| N2_snr-6 | f6aga1 | g / g | g / g | follows anchor | 4.03813 | 4.42222 | -0.38409 | 1.14696 |
| N1_snr-6 | f6aka1 | k / k | k / k | follows anchor | -4.92392 | -5.55462 | -0.63071 | 0.95909 |
| N2_snr-6 | f6aka1 | k / k | t / k | third class | -4.21559 | -3.17822 | +1.03736 | 0.87187 |
| N1_snr-6 | m5aga1 | g / g | dj / g | third class | 4.26516 | 3.33819 | +0.92697 | 1.26005 |
| N2_snr-6 | m5aga1 | g / g | g / g | follows anchor | 5.25155 | 5.17105 | +0.08050 | 1.19347 |
| N1_snr-6 | m5aka1 | k / k | k / k | follows anchor | -4.48188 | -4.31909 | +0.16280 | 0.79362 |
| N2_snr-6 | m5aka1 | k / k | k / k | follows anchor | -2.42417 | -2.69678 | -0.27261 | 0.83789 |

Each cross g/k decision follows its frozen anchor. Cross 24-class outputs for f6aka1/N2 and m5aga1/N1 select third classes t and dj. The remaining cross 24-class outputs follow their anchors. The signed q shifts vary in direction; this experiment does not show a binary identity flip toward the donor.

Noise instability matrix (N2−N1; changing noise is a separate comparison):

| Anchor | Native N1→N2 top1 | Cross N1→N2 top1 | Native q difference | Cross q difference |
|---|---|---|---:|---:|
| f6aga1 | g→g | g→g | -0.45134 | -0.01348 |
| f6aka1 | k→k | k→t | +0.70833 | +2.37640 |
| m5aga1 | g→g | dj→g | +0.98640 | +1.83286 |
| m5aka1 | k→k | k→k | +2.05771 | +1.62231 |

Four swapped donor peaks exceed the native normalized range, 1.04266–1.26005; these were recorded without condition clipping or renormalization. Raw outputs are saved and scored before PCM encoding. All output PCM clipping counts are zero.

Independent saved-artifact validation passed: 24 raw/WAV pairs, all 192 cross CTC candidate likelihoods plus 192 reused native likelihoods recalculated from saved logits, probability caches, 107 frozen source/helper/config/driver locks, and eight row geometry/noise checks. No additional model forwards. Artifacts: outputs/independent_validation.json, outputs/execution_environment.json, outputs/audio_manifest.json, outputs/baseline_scores_reused.json and outputs/ctc_arrays/.

A first attempt failed before model loading or any inference because NumPy float32 bounds were not JSON serializable; its exception handler also failed serialization. failed_attempt_01/ preserves the traceback, launch record, original driver and outputs. The engineer changed only two saved bounds to Python float and the corrected run succeeded. Original and corrected driver hashes are recorded in serialization_repair.json. The experiment driver and helper were unchanged during the successful run. The read-only validator initially assumed exact mathematical floor for PCM conversion; a single sub-LSB libsndfile rounding difference required validation by independent in-memory PCM re-encoding. No saved experiment output changed.

Limits: This is a controlled conditional/OOD mechanism diagnostic, not a new enhancement method or Router performance hypothesis. Whole VCV recordings differ in identity, coarticulation and duration, so this is not a surgical single-phoneme intervention. Cross cells conflict in anchor/noisy nominal identity: nested score.top1_correct only denotes agreement with the donor nominal label, not a uniquely correct enhancement target or true error. Use follows anchor/follows noisy/third class, never cross accuracy. One recognizer, four recordings and two fixed noise carriers support descriptive comparisons only; no population statistics, true PER, input identifiability or human recovery claim.

Source: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0.
