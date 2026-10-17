# DNSMOS t-cut development screen (2026-10-17)

## Scope

This archival record documents an existing 32-utterance development screen comparing the full Router with two late-step cutoffs. It reused existing enhanced WAVs and metric outputs; it did not train a model, modify the dataset, or evaluate the frozen official test set.

## Protocol

- Development split: 32 utterances (`n=32`)
- Arms: Router, `t_cut=0.05`, and `t_cut=0.10`
- Metrics: PESQ, STOI, SI-SDR, DNSMOS-P808/SIG/BAK/OVRL
- Paired bootstrap: 20,000 replicates, seed `2026`
- Differences are reported relative to Router.

## Results

| arm relative to Router | PESQ | STOI | SI-SDR (dB) | DNSMOS-SIG | DNSMOS-OVRL |
|---|---:|---:|---:|---:|---:|
| `t_cut=0.05` | -0.000287 [-0.005997, 0.005375] | -0.000121 [-0.000185, -0.000057] | -0.057741 [-0.065276, -0.050615] | +0.002578 [+0.000844, +0.004547] | +0.003261 [+0.001272, +0.005515] |
| `t_cut=0.10` | -0.005018 [-0.009941, -0.000201] | -0.000344 [-0.000519, -0.000159] | -0.180515 [-0.203526, -0.158816] | +0.009276 [+0.005983, +0.012801] | +0.010290 [+0.006655, +0.014053] |

The complete metric vector, including P808 and BAK, is in `metrics.json`; per-arm CSV files are under `cut005/` and `cut010/`.

## Interpretation and limits

On this development split, cutting Future guidance improves DNSMOS while reducing SI-SDR. The `t_cut=0.05` arm is a small Pareto point, not a comprehensive quality improvement. The split was used for prior route screening and is not independent evidence. These results do not justify further threshold tuning or a claim that the cutoff is a new diffusion mechanism.

## Reproducibility and safety

The source metric artifact is `metrics.json` with seed and bootstrap count recorded above. No original dataset file or valid checkpoint was changed or overwritten. This record completes the required `config.yaml` / `metrics.json` / `experiment_report.md` set for this archival experiment.
