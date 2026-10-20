# P8 counterexample listening pack (2026-10-20)

This is an optional, targeted listening artifact derived from the completed P8 local probes. It contains four 0.5 s windows with `Y_pre` suppression and clean-aware local SNR above −10 dB. It is intended to find a qualitative counterexample, not to collect another rating score.

The source metrics and selection rule are in `candidates.json`; the full context and caveats are in `LISTENING_GUIDE_CN.md`. No inference or training was run and no dataset/checkpoint was modified.

## Phoneme probe follow-up

The same four windows were run through the local `facebook/wav2vec2-lv-60-espeak-cv-ft` CTC model. On clean-active frames, the noisy clips assigned only `0.000–0.003` mean posterior to the clean CTC target and had `0.0` target argmax fraction in all four windows. The corresponding target posterior for `Y_pre` was `0.148/0.292/0.006/0.003`; Router-off was `0.108/0.134/0.014/0.002`; Router was `0.115/0.141/0.022/0.001` for windows 00–03. The noisy-side acoustic-model evidence therefore did not reveal a clear E1 counterexample, even in these higher-cue windows. This is still a four-window, same-frame pseudo-target diagnostic, not a human phoneme label or proof that no recoverable local evidence exists.

The generated arms have more consonant-like posterior mass than `Y_pre` in windows 00/01, but their decoded strings do not match the clean strings. This remains consistent with back-end naturalization rather than verified phoneme recovery. No content-loss, noisy-centered diffusion, or P8-R training was started.
