# Single 10720 router replay

Completed one full 30 s router inference. Saved router WAV was reproduced byte for byte. Full 480000 samples and 13.75–15.75 s window (32000 samples) are exactly equal, with maximum absolute difference 0 and difference RMS 0. Float correlation is 0.9999999999999998 for full duration and 1 for the window (rounding explains the full value). Frontend output after PCM16 serialization also exactly matches saved Y_pre: True.

This replay was explicitly authorized after the user retested the repaired player and still heard off/router sharing content different from clean/y_pre/noisy. That report remains unresolved as an acoustic/semantic difference. Playback repair does not explain the remaining report. Combined with the previous exact off replay, both backend outputs are reproducibly generated from the intended full noisy input with recovered original settings; this supports causal file pairing and does not establish correct phonemes or explain the content difference.

Original config.yaml, original router_retry.log and original report were inspected. RRSD log: enabled=True, scale1.0, t_cut0, t_cut_scale0, seed2026, N30; config/report: 80k EMA checkpoint, PC, corrector none, SNR0.5, CUDA0. Literal historical command and source script hashes are unavailable; reconstructed settings and current dirty git state/code hashes are preserved in metrics.json. The prior off replay was not repeated.

- Input: `/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/noisy/book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- Input symlink basename: `04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- Input SHA256: `cb714d176ad32c2aa041dc5a71715349ec9c35ae441bf55b0036f37d5bec173c`.
- Effective per-file seed: **2997368683**, computed from original prefixed basename with the script's BLAKE2b+2026 formula.
- Checkpoint: `/data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`.
- Checkpoint SHA256: `ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9`.
- Saved router: `/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020/outputs/router/04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- New router: `/data/coding/sgmse/experiments/p8_10720_pairing_forensics_20261018/replay_router/output/04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav`.
- Identical saved/replay SHA256: `78cd59bfa8695624751bff392c71532b40ece20f229e7cf5640e15a806a14cb6`.
- Actual device: CUDA0 NVIDIA L40S, UUID GPU-f9aa822f-67b6-c2aa-c2f9-eeabe9ab7cde.
- Actual NFE: **30**; enhancement wall 9.272 s; total work wall 22.160 s.

```sh
joint_enhance_by_dir.py --ckpt /data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt --in_dir /data/coding/sgmse/experiments/p8_10720_pairing_forensics_20261018/replay_router/inputs --out_dir /data/coding/sgmse/experiments/p8_10720_pairing_forensics_20261018/replay_router/output --device cuda:0 --sample_rate 16000 --train_type mode2 --sampler_type pc --N 30 --corrector none --snr 0.5 --seed 2026 --use_ema --rrsd --rrsd_scale 1.0 --rrsd_t_cut 0.0 --rrsd_t_cut_scale 0.0
```

Executed through run_replay.py using the existing successful off replay wrapper with router flags enabled and new exclusive input/output directories. Existing env/bin and CUDA/bin were prepended to PATH to avoid the known launcher failure; this launch succeeded without retries. Wrapper observes frontend matching and requests existing timeit=True NFE/timing output without changing the waveform returned to the exporter. No browser dependencies, downloads, ASR, training or model changes. Original input, saved router and checkpoint hashes match their post-run hashes. New audio output is 960044 bytes. All full comparisons, current code hashes, environment, git state and original argument list are in metrics.json.
