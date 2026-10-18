# Online frontend representation probe (16 speakers)

Read-only evaluation on 16 fresh-CV utterances, one per speaker, selected by first occurrence in the manifest. The frozen 80k EMA checkpoint was used. Three StormFrontend backbone hooks (`act`, `output_layer`, `pyramid_upsample`) captured noisy/clean pooled statistics. PC N=30 inference ran online with a trace callback at steps 3, 5, 7, and 9; clean-x0 NMSE was computed immediately. No WAVs, trajectories, or caches were written.

Fixed-alpha=1 speaker-LOSO Ridge results:

| features | MAE | R² | Spearman |
|---|---:|---:|---:|
| SNR + noisy RMS + cosine | 0.15479 | 0.7584 | 0.8735 |
| hook pooled features | 0.16416 | 0.6551 | 0.8588 |
| combined | 0.15619 | 0.7417 | 0.9029 |

The target mean four-state NMSE was 0.94128. These estimates are exploratory and low-power despite 16 speakers; they are not a generalization claim. The first attempt incorrectly assumed the first 16 rows were unique speakers (they were 8 speakers) and stopped before evaluation; the corrected run selected first occurrences of 16 speakers. Runtime was 183.7 seconds on CUDA.
