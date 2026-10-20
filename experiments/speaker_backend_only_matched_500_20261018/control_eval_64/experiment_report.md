# Control checkpoint evaluation (locked 64-speaker set)

Read-only evaluation of the 500-step backend-only no-loss control checkpoint. No training, dataset changes, or checkpoint overwrites occurred. The locked manifest has 128 utterances from 64 speakers (16 kHz, 30 s each), SHA256 `c04657fa21a33c7f2dc749514cec899767101ba53f46222e9f4ab8577e2f4492`.

Inference used PC sampler, N=30, no corrector, snr=0.5, seed 2026, EMA. All 128 outputs are finite, mono, aligned PCM16 WAVs. Checkpoint SHA256 is `f42b60d0d25d6cfc0bfef3255ec3c0ef3926968367145c9f271eee1411106740`.

## Speaker metrics

CAM++ control paired-clean cosine is 0.65362 (95% cluster CI 0.61587–0.68992), versus off 0.65216 and router 0.65085. For the primary identity verification measures, control−off was same-speaker cosine `+0.00262` (CI `[-0.00832,+0.01453]`), AUC `+0.00852` (CI `[-0.00462,+0.02486]`), and EER `−0.01736` (CI `[-0.06253,+0.03046]`). All intervals include zero. The paired-clean contrasts were control−off `+0.00146` (CI `[-0.00562,+0.00847]`) and control−router `+0.00277` (CI `[-0.00390,+0.00981]`).

## Quality metrics

PESQ WB: control 1.61383, off 1.62011, router 1.63103. STOI: 0.80630, 0.80677, 0.80792. SI-SDR: 6.25728, 6.25728, 6.76877 dB (control, off, router respectively). Full per-utterance values and speaker-cluster intervals are in `quality_per_utterance.csv` and `quality_cluster_metrics.json`.

## DNSMOS

A single control WAV smoke test resolved `auto` to `cuda:0` and took 3.18 s, but ONNX Runtime warned that no CUDA plugin device was registered; actual CUDA execution was not verified. The smoke scores (P808 4.01195, SIG 3.52890, BAK 4.15420, OVRL 3.24336) are not representative of the three-arm set. Full DNSMOS evaluation was therefore not run; see `dnsmos_smoke.json`.
