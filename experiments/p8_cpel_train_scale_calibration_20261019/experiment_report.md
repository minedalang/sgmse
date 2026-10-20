# CPEL training-only gradient scale measurement: completed with zero updates

One authorized run completed in 22.663933s, PID38731, exec session79848 (completed), exit0. Eight fixed training records×two draws produced all16 registered probes.64 Score sample forwards,32 autograd.grad traversals,16 S/X PCM reads,0 normal sampler calls,0 optimizer steps,0 EMA updates and0 parameter updates. No dev/noise/final-test source access or audio/weight exports. Peak CUDA allocation9805556736 bytes.

```text
env PATH='/data/coding/sgmse/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/command:/usr/local/cuda/bin' env/bin/python -B experiments/p8_cpel_train_scale_calibration_20261019/calibrate.py --execute-calibration
```

All16 gradients are finite, M norms positive and target coverage active16/16. Zero pair gradients0/16; undefined ratios0/16, none dropped. The originally fixed GaussianFourierProjection parameter `dnn.all_modules.0.W` remains frozen. Sources/controls are detached for all16 probes; no accumulated parameter .grad or non-dnn updates.

The prescribed median of16 paired ratios is 0.000720601299819078. Raw proposal0.01/median=13.8772994199576; clipping to[0.1,100] makes no change. This lambda is a suggestion only and was not applied to the parent config or a training run. Proposed effective median=0.01, but maximum=2.83412246186722. Thus a1% median norm target is not an all-probe cap; one probe would have pair gradient norm2.834 times M under this proposal. There was no retuning to hide that heterogeneity.

| Quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| M_norm | 0.00730676345208 | 0.0140408458897 | 0.0440509552351 |
| pair_norm | 5.28455572384e-07 | 9.16839644177e-06 | 0.0028233573855 |
| ratio | 2.66893518112e-05 | 0.000720601299819 | 0.204227232987 |
| cosine | -0.109933037066 | 0.0143338194914 | 0.422843393752 |
| effective_current | 2.66893518112e-06 | 7.20601299819e-05 | 0.0204227232987 |
| effective_proposed | 0.000370376126408 | 0.01 | 2.83412246187 |

Every registered probe (ratios include any zeros; none occurred):

| Record/draw | Utterance | Bin | Native t | Aux t | Parent | M norm | Pair norm | Unweighted ratio | Cosine | λ=.1 effective | Proposed effective |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0/0 | reader_05264_fileid_4279 | [-20,-18) | 0.08162567 | 0.63766927 | 1 | 0.04405095524 | 1.3125525e-06 | 2.97962324e-05 | 0.00655198359 | 2.97962324e-06 | 0.0004134912386 |
| 0/1 | reader_05264_fileid_4279 | [-20,-18) | 0.48462078 | 0.59494805 | 0 | 0.01291464927 | 1.378527821e-06 | 0.000106741406 | 0.0455326325 | 1.06741406e-05 | 0.001481282452 |
| 1/0 | reader_02071_fileid_8404 | [-20,-18) | 0.71380591 | 0.4094725 | 0 | 0.01025723355 | 2.749709105e-05 | 0.002680751191 | -0.0714665246 | 0.0002680751191 | 0.03720158695 |
| 1/1 | reader_02071_fileid_8404 | [-20,-18) | 0.77847749 | 0.58085895 | 1 | 0.01672285409 | 1.205568987e-05 | 0.000720911024 | 0.0469496518 | 7.20911024e-05 | 0.01000429814 |
| 2/0 | reader_09747_fileid_2307 | [-18,-15) | 0.85744995 | 0.60020804 | 0 | 0.01314323018 | 6.281103018e-06 | 0.0004778964481 | 0.0503512151 | 4.778964481e-05 | 0.006631912103 |
| 2/1 | reader_09747_fileid_2307 | [-18,-15) | 0.25844041 | 0.35796261 | 1 | 0.01425710363 | 3.710943428e-05 | 0.002602873294 | 0.114134449 | 0.0002602873294 | 0.03612085205 |
| 3/0 | reader_09137_fileid_5789 | [-18,-15) | 0.67250186 | 0.60100579 | 1 | 0.0143311931 | 4.536559647e-06 | 0.0003165514284 | -0.0666769444 | 3.165514284e-05 | 0.004392878954 |
| 3/1 | reader_09137_fileid_5789 | [-18,-15) | 0.98625386 | 0.62847972 | 0 | 0.007306763452 | 5.26300016e-06 | 0.0007202915756 | -0.0583042486 | 7.202915756e-05 | 0.009995701865 |
| 4/0 | reader_02047_fileid_339 | [-15,-10) | 0.59727073 | 0.45283842 | 1 | 0.009386561433 | 4.505934013e-05 | 0.004800409655 | 0.00508741585 | 0.0004800409655 | 0.06661672212 |
| 4/1 | reader_02047_fileid_339 | [-15,-10) | 0.6430766 | 0.38613302 | 0 | 0.01382458815 | 0.002823357385 | 0.204227233 | 0.422843394 | 0.0204227233 | 2.834122462 |
| 5/0 | reader_01106_fileid_5756 | [-15,-10) | 0.63953412 | 0.43972409 | 0 | 0.01945091109 | 3.451280201e-06 | 0.0001774354006 | 0.163990271 | 1.774354006e-05 | 0.002462324181 |
| 5/1 | reader_01106_fileid_5756 | [-15,-10) | 0.21413481 | 0.65052402 | 0 | 0.01980024004 | 5.284555724e-07 | 2.668935181e-05 | 0.0221156554 | 2.668935181e-06 | 0.0003703761264 |
| 6/0 | reader_03369_fileid_7621 | [-10,-5] | 0.50930959 | 0.64816695 | 1 | 0.02436673264 | 5.748021181e-06 | 0.0002358962634 | 0.0906904253 | 2.358962634e-05 | 0.003273603079 |
| 6/1 | reader_03369_fileid_7621 | [-10,-5] | 0.67476809 | 0.35908645 | 1 | 0.02187182095 | 2.845008507e-05 | 0.001300764355 | -0.0265332717 | 0.0001300764355 | 0.01805109642 |
| 7/0 | reader_08563_fileid_3938 | [-10,-5] | 0.78098261 | 0.39586639 | 0 | 0.01352833852 | 4.681875503e-05 | 0.003460791209 | -0.109933037 | 0.0003460791209 | 0.04802643584 |
| 7/1 | reader_08563_fileid_3938 | [-10,-5] | 0.62812477 | 0.37180942 | 1 | 0.01280890057 | 4.373258711e-05 | 0.003414234257 | -0.0733766805 | 0.0003414234257 | 0.04738035108 |

Independent scalar validation rebuilt each ratio from saved pair/M norms, sorted all16 values and averaged positions7/8, recomputed raw/clipped coefficient and all32 effective ratios. It imports neither calibrate.py nor model code, calls no model, and does not recompute gradients. All comparisons passed; maximum aggregate absolute difference1.7347234759768071e-18. This checks arithmetic only, not independent GPU gradient correctness or acoustic validity.

Model parameter/buffer digest before and after is `533d10eca54f5ff711f1697768d3e34b557d9cb06873239beceef325cd7f99a4`. The source checkpoint,16 registered WAVs,three parent helpers and parent config have identical pre/post SHA256. Dedicated script/config/registry hashes remain frozen. Source EMA materialized once; weights were never saved or updated. No network attempts occurred. The bridge39452 and listening server16022 remain alive and untouched. Complete source hashes, exact command/PATH/PID/session, git state, versions/actual counts and preservation receipt are retained in metrics.json and execution_receipt.json; original primary metrics were not modified for reporting.

These16 correlated training probes are descriptive, not independent samples, evidence-probability calibration, Adam update ratios or semantic reliability. The measurement does not establish native enhancement, weak-phoneme recovery, hallucination suppression or generalization. Float-edit versus PCM-grid shortcuts remain unresolved; changing the window alone does not eliminate them. No dev result was used to select this coefficient and no parent lambda was changed.

This calibration is complete and stopped. Parent M/P training remains unauthorized and unexecuted; Root decides any next step.
