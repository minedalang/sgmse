# Optional listening checks

These checks do not block the completed cached audit. No listening judgment has been collected. Times below are relative to the existing 2-second WAVs and approximate CTC run intervals, not verified acoustic boundaries. Compare clean and off in the same time window, and record playback gain; the recognizer uses per-clip normalization, so original quietness and token decoding are different observations. No extra audio files or inference were generated.

| Event | Off cached run | Clean counterpart | Suggested comparison window | Uncertainty to resolve |
|---|---|---|---|---|
| Alveolar tap label | ɾ at 0.14–0.16 s | Clean iː at 0.14–0.16 s; clean d at 0.10–0.12 s | 0.06–0.22 s | Whether any audible transient supports a consonant-like event; the off label ɾ is absent from the clean proxy sequence. |
| Bilabial stop label | p at 0.30–0.32 s | Clean d at 0.28–0.32 s at the same time. Unconstrained sequence alignment instead substitutes against clean s at 0.44–0.46 s. | 0.22–0.50 s | Whether the off event corresponds to the nearby clean event or another sound; sequence alignment alone does not identify the acoustic counterpart. |
| Delayed matching stop label | t at 0.74–0.76 s | Clean t at 0.62–0.66 s, midpoint shift +110 ms; same-time clean l at 0.74–0.76 s | 0.55–0.85 s | Whether this is a shifted realization or another event. The +110 ms pair is outside the preset 100 ms diagnostic. |

Unmatched labels and substitutions are recognizer disagreements against a clean recognizer proxy. They do not prove hallucinations or establish true phoneme errors.
