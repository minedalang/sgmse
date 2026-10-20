# CPEL technical smoke: passed; no full training started

One authorized technical smoke completed with exit0, PID603/session20406 (completed), in90.097805s (limit600s). Both source/CUDA exact-resume checks passed without tolerance fallback; all four complete30s public feasibility calls passed. This verifies bounded engineering behavior, not long-training scientific benefit or a100k training authorization. No full run was started and no runs/ directory exists.

```text
env/bin/python -B experiments/p8_cpel_longrun_20261020/train_longrun.py --mode technical_smoke --arm both --authorization /data/coding/sgmse/experiments/p8_cpel_longrun_20261020/technical_smoke_authorization.json
```

Actual versus retained updates: each arm executes reference steps1–4, saves the committed step2 snapshot, then restores that snapshot into its separately authorized shadow root and replays steps3–4. Thus each arm has6 physical optimizer updates but only4 retained scientific steps; total12 physical/8 retained. Replayed work is explicitly counted.

| Arm | Exact comparison | Retained | Physical updates | Source loads | PCM | F | Score | Reverse | Public |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N | passed_exact | 4 | 6 | 1 | 20 | 16 | 144 | 12 | 4 |
| D | passed_exact | 4 | 6 | 1 | 12 | 12 | 24 | 12 | 0 |

Total actual calls match the authorization exactly:2 source model loads,32PCM reads,28frontend,168Score,24reverse traversals,12optimizer updates and4public backend calls. Actual reference/shadow source starts use only the permitted source checkpoint; no old M/P/N/D terminal weights were read. No metrics, ASR/DNSMOS, network, installation, new samples, audio exports or quality-based decisions were used.

The driver compares complete raw DNN/EMA/Adam/global+local RNG/remaining shuffle order/identity/logcommit snapshots with exact recursive tensor/value equality. It also requires byte-identical4step scientific journals after replay. Worker independently checked only saved JSON/journal bytes and SHA/commit offsets, not a second model run or checkpoint tensor load. Both log commits match, all four draw-field tuples match N/D, and all retained native-only raw-gradient budget values reconstruct within numerical arithmetic tolerance. Fixed frontend/Fourier digests and parameter binding/signature checks passed in the source driver.

| Arm | Frozen digest | Parameter signature | Journal commit SHA |
|---|---|---|---|
| N | db84f369fc851dde1f4b420122e57f96e6fa2120e7bec41c76692d7a21efa8a7 | 164293eae44380fdaee763c9f7518babb726aa081dccfcca72aa7637d0260a97 | 387e5de27cd43361b90270257ee642f35090627ab91aec8e0c927329dfbfa455 |
| D | db84f369fc851dde1f4b420122e57f96e6fa2120e7bec41c76692d7a21efa8a7 | 164293eae44380fdaee763c9f7518babb726aa081dccfcca72aa7637d0260a97 | ae9d47a60f7dac14b2ba0d2ce9098b1f01d4d80e379232dfd0aedef31e9f350e |

Complete source public feasibility (no crop/chunk fallback, no quality metric):

| Utterance | Output samples | NFE / Score | Actual Noisy norm | Public seconds | Recorded CUDA peak bytes |
|---|---:|---|---:|---:|---:|
| reader_11103_fileid_4413 | 480000 | 30/30 | 0.270904541016 | 9.712795 | 7325722624 |
| reader_11332_fileid_10899 | 480000 | 30/30 | 0.983795166016 | 9.736700 | 7325722624 |
| reader_11070_fileid_10064 | 480000 | 30/30 | 0.338012695312 | 9.756781 | 7333457920 |
| reader_11076_fileid_2125 | 480000 | 30/30 | 0.990020751953 | 9.783587 | 7341240320 |

All full-source S/X lengths were480000, each output had480000 finite samples, public length480000 and actual norm equal the original complete-Noisy peak. Output SHA descriptors were recorded but waveform arrays were not saved. F/state was unchanged through these calls; actual public sampler/profiler wrappers restored in finally. Public times measure individual sampler/enhance calls, not total source read/frontend/metric throughput. CUDA peak values are the driver-recorded cumulative public readings (maximum7341240320B); a separate whole-smoke training peak was not recorded and is not invented. Four cases cannot guarantee all1186 utterances or longer-run memory/runtime behavior.

Scope/result limits: this is not a zero-update test (12 physical tiny updates occurred), but is also not100k training. No milestone selection, longer training, further resume or model/quality experiment followed. Exact equality here covers the declared4steps and same process/driver/platform; it does not establish cross-version or across-host bitwise reproducibility. No content/identity/phoneme recovery or perception improvement follows.

All71 necessary protected files match Worker pre/post SHA:52core,6driver/helper/resume files,config/protocol/notes/authorization,4registries/receipt/originalCVmetadata/sourcecheckpoint/currentgoal. Driver source_locks_post_matched=True. No full10797-WAV audit/hash scan or extra WAV/header/PCM read was performed by Worker; only the code read its authorized four full-sentence sources and first-four retained training draws, with replay counted. Existing dirty git state remains recorded; bridge39452/server16022 are alive and untouched. All outputs/cache/TMP are confined to configured/data paths.

The original attempt JSON is immutable evidence at attempt_1792464176377551889.json; metrics.json is a byte-identical copy, not a recomputed result. Technical_smoke_validation.json records scalar/journal checks and byte hashes of the four retained resume checkpoints plus logs; those checkpoints were not deserialized by Worker. Source-style/source/preprocessing/history limitations remain in the frozen plan.

The unique technical smoke is complete and stopped. A separate Root authorization and execution instruction would be required for any full100k run or subsequent resume; this report does not grant them.

