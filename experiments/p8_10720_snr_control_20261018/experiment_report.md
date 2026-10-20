# 10720 paired archived-noise attenuation positive control

Status: completed

Same clean utterance plus its archived paired noise at lower noise gains is a positive control for pipeline content preservation.

control=stored_clean+alpha*archived_paired_noise; clean gain always1; noise gain alpha=10**(-10/20) or10**(-20/20). The original actual-noisy baseline and its outputs are reused without inference. SNR uses clean power divided by scaled archived-noise power.

This offline oracle control uses stored clean speech and paired archived noise; it is not deployable enhancement or performance proof. Success does not establish original extreme-low-SNR recoverability. Original noisy differs from stored clean plus archived noise by at most one PCM16 step; baseline remains the actual original noisy and original outputs. Listening judgments are not collected yet.

All five arms use source samples220000:252000 (13.75–15.75s). Original baseline outputs were reused. Stored paired noise was attenuated by10/20dB with clean gain1; clean/scaled archived-noise SNR is recorded below. Original noisy differs from clean+archived-noise by at most1PCM16 step and is reused as saved. Fixed playback gain1, no RMS normalization. No listening judgment has yet been recorded.

|Variant|Full30s SNRdB|Focus SNRdB|
|---|---:|---:|
|original|-15.9999|-18.2442|
|noise_minus10dB|-5.9999|-8.2442|
|noise_minus20dB|4.0001|1.7558|

Reproducibility: config.yaml, run_control.py, launch_provenance.json, metrics.json and source_clip_manifest.json include seeds, source/checkpoint/code hashes, timing and NFE. Inputs are FLOAT WAV; frontend/backend saved as original PCM16 pipeline. Source pair and checkpoint hashes verified unchanged.
# Latest user listening feedback (2026-10-18; after the completed run)

> 我听了 确实降低db的 听着几段语音是匹配的

The user reports corresponding content across the listened-to lower-noise
controls. They did not separately score every attenuation level and arm, so
do not label each individual cell as human-verified or compute a success rate.
This is a reference-informed, nonblind, single-listener positive control.
Structured feedback is saved in `user_listening_feedback.json`.

Together with exact byte-for-byte reconstruction of original off and Router
from the correct noisy input, the observation supports a valid generation/
listening chain and makes a wrong-file explanation for the original10720
content difference unlikely. The remaining original-low-SNR discrepancy is
now a model-output content-deviation candidate, including possible speech-like
artifacts, phonetic substitutions or temporal smearing. It is not a measured
hallucination rate or proof that every noisy phoneme was recoverable.

Close repeated pairing checks on this case. Keep the original-low-SNR negative
example and the5243 d positive observation separately. The next independent
diagnostic should test whether known consonant identity changes are followed
under identical noise, using the already frozen VCV controls rather than
selecting new windows or alignment thresholds on these same four examples.
No additional model inference or training is triggered by this report update.

The run metadata below preserves the status at generation time; statements
that listening had not yet been collected are superseded by this section.

# Latest refinement: local tail artifact despite overall content correspondence

The user subsequently reports that noise-reduced20dB off/Router end with an
extra s-like sound absent from clean and Y_pre; at10dB the corresponding
backend sound is more speech-like, while Y_pre has a faint non-speech-like
residual. Verbatim feedback is in `user_tail_feedback_20261018.json`.

This limits the earlier positive-control interpretation: broadly corresponding
content can coexist with an added local artifact. Investigate noise-to-speech
conversion, generative additions and temporal smearing separately. The playback
clip ends at original15.75s inside a30s source; check surrounding clean context
before describing the event as utterance-final hallucination.

The user is away until tomorrow and requests independent work. No human
response is a prerequisite for the next acoustic and controlled-mechanism
diagnostics; optional checks can be queued for later without interrupting work.

