# Fixed normalization VCV conditional swap: prepared only

Read `/data/AGENTS.md` and current `/data/RESEARCH_PLAN.md`; inspected prior experiments and dirty git state. No completed experiment was rerun. All implementation is experiment-local; core code, prior outputs, corpus data and checkpoints are unchanged.

Frozen design: f6 k/g from the original low-SNR challenge and m5 k/g from fresh speaker replication; N1/N2 at -6 dB. Each native frontend is recomputed with its native noisy peak and must match saved raw float32 hashes without consuming RNG. Eight native OFF diagonals must match saved raw arrays (`np.array_equal`), NPY hashes and PCM hashes before any of eight crosses runs. A mismatch stops and archives partial outputs, with no retry or cross case.

Each row freezes native frontend anchor, original native noisy peak, seed and normalized anchor spectrum. The noisy donor changes between the original k/g recordings mixed with identical physical noise and gain. Float64 recipe hashes and bounded float32 decomposition errors are checked. Reference scales differ across anchor rows. Donor normalized peaks above one are recorded and retained without clipping or renormalization. This is an OOD conditional mechanism test on whole recordings, including duration and coarticulation, not a surgical phoneme change or method improvement.

`fixed_norm_off.py` copies the native ScoreModel OFF STFT/transform/padding, partial score and OUVE PC sampler call with explicit reference scale. Existing sampler trace callbacks hash the first pre-predictor state; same-row native/cross initial state, anchor spectrum, RNG states, seed and reference must match. No core edits, monkeypatching or AST rewriting. Runtime checks enforce mode2/noisy concat/OUVE PC, NCSN++ v2, no cross attention/FiLM/selector/residual refinement and no Router. N=30, reverse_diffusion, corrector none, snr=0.5; all 16 calls require NFE30 and 30 traced steps.

Only eight cross outputs get new wav2vec2 forwards. Native/clean/noisy/frontend scores are reused from frozen caches. Scores retain all 24 candidates and g-minus-k CTC likelihood ratios, with separate nominal anchor/noisy follow indicators. Cross logits/probabilities are saved. Per-row comparisons are descriptive, with no pooled causal statistics, human correctness, true PER or human recovery claim. Quiet padding and source RMS are retained. Source: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0.

Validation: Python syntax parsing, all source/checkpoint locks, eight CPU-only recipe reconstructions/hash checks and unique scoring-cache key checks pass. The initial attempt to use the base Python for CPU preflight found NumPy absent; preflight then succeeded with the existing torch environment. No model was loaded or run. Runtime no-op and geometry checks remain for the worker.

Worker command after review, from `/data/coding/sgmse`:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/miniconda/envs/torch/bin/python -B experiments/p8_vcv_fixed_norm_condition_swap_20261019/run_swap.py
```

Output directory must not exist. Limits: 600 seconds checked between tasks, 50 MB artifacts, no downloads/training/extra ASR. Preserve any failed run and diagnose; do not retry the same command. Expected forward counts: frontend 8, backend 16, new recognizer 8.
