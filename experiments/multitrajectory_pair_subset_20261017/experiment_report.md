# Two-trajectory subset selector: offline analysis

This reuses the existing four-seed trajectory metrics. It runs no inference or training and reads no clean/reference signal for selection. For each utterance and each of the six seed pairs, it selects the candidate with higher DNSMOS OVRL, then compares the selected output with fixed seed 2026.

Each pair costs 2 × N=30 = 60 score-network evaluations per input, half the four-trajectory selector's 120 evaluations. CIs are paired utterance bootstrap intervals (20,000 replicates).

## Development (16)

| Seed pair | Selected seed counts | OVRL Δ [95% CI] | SI-SDR Δ dB [95% CI] | SI-SDR negative files | PESQ Δ | STOI Δ | SIG Δ | BAK Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026–2027 | {'2026': 7, '2027': 9} | +0.0348 [+0.0150, +0.0580] | +0.0883 [-0.0010, +0.1722] | 1/16 | -0.0115 | +0.0003 | +0.0212 | +0.0202 |
| 2026–2028 | {'2026': 8, '2028': 8} | +0.0412 [+0.0125, +0.0743] | +0.0436 [+0.0026, +0.0946] | 3/16 | +0.0080 | +0.0050 | +0.0293 | +0.0169 |
| 2026–2029 | {'2026': 7, '2029': 9} | +0.0231 [+0.0077, +0.0421] | +0.0620 [-0.0054, +0.1403] | 3/16 | -0.0013 | +0.0018 | +0.0193 | +0.0075 |
| 2027–2028 | {'2027': 8, '2028': 8} | +0.0409 [+0.0077, +0.0767] | +0.0778 [-0.0171, +0.1696] | 7/16 | +0.0071 | +0.0022 | +0.0243 | +0.0263 |
| 2027–2029 | {'2027': 12, '2029': 4} | +0.0327 [+0.0064, +0.0600] | +0.1155 [+0.0198, +0.2070] | 5/16 | -0.0068 | +0.0004 | +0.0168 | +0.0284 |
| 2028–2029 | {'2028': 11, '2029': 5} | +0.0267 [-0.0137, +0.0689] | +0.0793 [-0.0210, +0.1765] | 6/16 | -0.0035 | +0.0043 | +0.0161 | +0.0100 |

## Held-out (16)

| Seed pair | Selected seed counts | OVRL Δ [95% CI] | SI-SDR Δ dB [95% CI] | SI-SDR negative files | PESQ Δ | STOI Δ | SIG Δ | BAK Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026–2027 | {'2026': 5, '2027': 11} | +0.0449 [+0.0240, +0.0677] | +0.0007 [-0.0460, +0.0510] | 7/16 | +0.0081 | -0.0000 | +0.0382 | +0.0229 |
| 2026–2028 | {'2026': 11, '2028': 5} | +0.0313 [+0.0080, +0.0587] | -0.0060 [-0.0315, +0.0227] | 4/16 | -0.0139 | +0.0013 | +0.0246 | +0.0178 |
| 2026–2029 | {'2026': 7, '2029': 9} | +0.0272 [+0.0094, +0.0488] | +0.0030 [-0.0600, +0.0518] | 2/16 | -0.0083 | +0.0035 | +0.0196 | +0.0191 |
| 2027–2028 | {'2027': 12, '2028': 4} | +0.0477 [+0.0162, +0.0793] | +0.0194 [-0.0477, +0.0856] | 8/16 | +0.0133 | +0.0019 | +0.0433 | +0.0194 |
| 2027–2029 | {'2027': 9, '2029': 7} | +0.0463 [+0.0154, +0.0770] | +0.0048 [-0.0588, +0.0624] | 6/16 | +0.0103 | +0.0011 | +0.0333 | +0.0351 |
| 2028–2029 | {'2028': 8, '2029': 8} | +0.0359 [+0.0061, +0.0676] | +0.0159 [-0.0631, +0.0890] | 7/16 | -0.0038 | +0.0024 | +0.0247 | +0.0250 |

## Interpretation

The pairwise selector uses an oracle-free DNSMOS criterion but still requires evaluating both trajectories, so it is a compute/quality trade-off rather than a free selector. Treat these as exploratory results: the seed pairs share the same underlying utterances and the DNSMOS criterion is the selection metric, so OVRL gains are expected to be optimistic. Held-out comparisons against fixed seed 2026 are the more informative check; inspect both OVRL and per-utterance SI-SDR regressions before proposing a learned selector or a full test-set run.

Full per-metric means, paired bootstrap intervals, selection counts, and each utterance's selected seed and paired metric deltas are in `metrics.json`.
