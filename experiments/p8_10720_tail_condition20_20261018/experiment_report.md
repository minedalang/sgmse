# 10720 condition-tail 20dB replication

Status: completed

Oracle clean-tail intervention is an OOD mechanism probe, not a deployable method. This is within-case/SNR replication, not independent generalization or a trained method. Prior P8-M 5243 active masking negative is preserved and is a distinct experiment.

Window: samples240800:252000, 15.05–15.75s, internal20ms raised cosine fades. Model loaded once; original FLOAT frontend recomputed once and hash checked. Both runs are backend off, NFE30. Full speech activity uses clean20ms disjoint frames above−70dBFS. Raw metrics compare with new no_op raw; PCM16 values are reference only. frames.csv retains raw fixed frame values. No clean/noisy corpus changes or downloads.

|Run|Tail RMS dBFS|4–7.5kHz mean square|Outside relative L2|Active relative L2 vs no_op|
|---|---:|---:|---:|---:|
|no_op|-72.44268|5.2087399e-08|0|0|
|condition_tail_clean|-101.81502|9.1075989e-12|0.0022074121|0.00209438|

## Independent cached verification and within-case comparison

Both runs passed NFE30, exact20dB float frontend SHA and no_op PCM16 byte gates. Saved frontend hashes are globally identical across the two runs. Cached condition exactly reconstructs the frozen cosine interpolation, is bitwise unchanged outside the window, and retains the original noisy global peak. Source WAV, checkpoint, code, config, driver and output hashes matched. Independent SciPy periodogram band energy and float64 RMS matched raw and PCM16 metrics.

Original L2 metrics use float32 reference norms. Independent float64 denominators differ by at most 2.9103459e-05 relative; the original precision exactly reproduces saved metrics. No inference was repeated.

| Representation | Run | Tail RMS | Tail dBFS | RMS delta dB | Highband energy | Highband delta dB | Outside L2 | Active L2 vs no_op | Active L2 vs clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | no_op | 0.000238707577 | -72.443 | 0.000 | 5.20873995e-08 | -0.000 | 0.000000 | 0.000000 | 0.218885 |
| pcm16_reference | no_op | 0.000239928906 | -72.398 | 0.000 | 5.21291178e-08 | 0.000 | 0.000000 | 0.000000 | 0.218908 |
| raw | condition_tail_clean | 8.11425704e-06 | -101.815 | -29.372 | 9.10759892e-12 | -37.573 | 0.002207 | 0.002094 | 0.218430 |
| pcm16_reference | condition_tail_clean | 2.86091396e-05 | -90.870 | -18.472 | 3.93788426e-11 | -31.218 | 0.003666 | 0.003439 | 0.218455 |

At10dB noise attenuation the same condition intervention reduced raw tail RMS by -25.657dB and highband energy by -44.902dB. The20dB result reproduces the direction within the same utterance and fixed window. It is not independent generalization. The intervention uses oracle clean condition and is OOD; measured energy reductions do not establish phoneme correctness, a deployable method or novelty. Earlier experiment files and negative results were preserved.
# Research Lead assessment: within-case replication, not a deployment result

The user specifically described an extra s-like sound in the20dB attenuation
control. Keeping its entire frontend and sampling random numbers fixed while
replacing only the local noisy condition reduces raw tail RMS by29.372dB and
highband energy by37.573dB. This agrees directionally with the10dB condition
pilot, strengthening the local-condition sensitivity explanation in the same
utterance. Raw outside-window relative L2 is0.002207 and active relative L2
versus no_op is0.002094; these waveform comparisons do not establish
preserved phoneme identity or audible transparency.

The intervention uses clean reference in a near-quiet gap and changes the
normal joint conditioning distribution. It is not a method available at test
time. Suppression of energy is not by itself a correction of s or human
speech content; the user has not judged the intervened outputs. Quantized
PCM16 statistics are reported separately and cannot replace the raw energy
comparison near the quantization floor. Next examine existing other-utterance
quiet gaps before proposing training or a deployable condition mechanism.

