# Frozen64 dev crops: codec consistency diagnosis

One CPU execution completed, exit0, PID38596, 0.515609s (tool chunk6a22ed; completed in initial call, no running session_id). All64 frozen records and192 unique source files were retained with identical pre/post SHA256. No model, checkpoint, network, gain/lag fit, threshold adjustment, update or audio export.

```text
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cpel_codec_consistency_20261019/diagnose.py
```

Each triplet was mono16k with480000 original frames; fixed center start223680 and32640 float64 samples. No min-length alignment, padding, resampling, row replacement or source-specific normalization occurred. Raw PCM scale was preserved; relative RMS is invariant under a shared gain.

Strict relative RMS≤0.001 remains a separate unchanged gate:43/64 pass,21/64 fail. Both additional conditional bounds are shown together, not selected by width: all64 crops are compatible with each; none has a crop code endpoint. Original processing-chain provenance is unverified for all64. Conditional compatibility does not prove noise purity, exact synthesis identity, rounding mode, or a valid semantic noise-only control, and does not overturn the21 strict failures.

| Scope | N | Strict pass/fail/unknown | Nearest bound compatible/fail/unknown | General codec bound compatible/fail/unknown | Crop endpoint rows | Unverified chain |
|---|---:|---|---|---|---:|---:|
| overall | 64 | 43/21/0 | 64/0/0 | 64/0/0 | 0 | 64 |
| [-10,-5] | 16 | 12/4/0 | 16/0/0 | 16/0/0 | 0 | 16 |
| [-15,-10) | 16 | 9/7/0 | 16/0/0 | 16/0/0 | 0 | 16 |
| [-18,-15) | 16 | 12/4/0 | 16/0/0 | 16/0/0 | 0 | 16 |
| [-20,-18) | 16 | 10/6/0 | 16/0/0 | 16/0/0 | 0 | 16 |

q_i=2^(1−bits_i). Nearest bound=0.5Σq assumes exact prequantized additivity and final nearest rounding only; general bound=Σq assumes per-source quantization error≤1LSB. Exceedances compare each absolute residual with the bound plus1e−12 numerical tolerance. These are explicit alternative assumptions, not verified processing histories. Endpoint equality is a possible clipping warning; absence of endpoints does not rule out earlier clipping/processing. Unknown geometry/format would have remained unknown; none occurred.

| Index / utterance | Bin | Relative RMS | Strict | Max abs residual | Nearest bound | Exceed samples | General bound | Exceed samples |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 0 / reader_11236_fileid_6794 | [-20,-18) | 0.0033555881 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 1 / reader_11826_fileid_6359 | [-20,-18) | 0.000352255343 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 2 / reader_11065_fileid_7519 | [-20,-18) | 0.0019067674 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 3 / reader_11121_fileid_10724 | [-20,-18) | 0.000196317418 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 4 / reader_11880_fileid_9924 | [-20,-18) | 0.000313859284 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 5 / reader_11121_fileid_10605 | [-20,-18) | 0.000473651355 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 6 / reader_11265_fileid_2501 | [-20,-18) | 0.00263664602 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 7 / reader_11117_fileid_8577 | [-20,-18) | 0.000293966465 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 8 / reader_11297_fileid_3545 | [-20,-18) | 0.000226077767 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 9 / reader_11346_fileid_6754 | [-20,-18) | 0.00169810518 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 10 / reader_10929_fileid_5420 | [-20,-18) | 0.00158069208 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 11 / reader_11446_fileid_11151 | [-20,-18) | 0.000946854556 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 12 / reader_10929_fileid_3187 | [-20,-18) | 0.00901486987 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 13 / reader_10907_fileid_11974 | [-20,-18) | 0.000318530439 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 14 / reader_11103_fileid_2057 | [-20,-18) | 0.000666927862 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 15 / reader_11219_fileid_2058 | [-20,-18) | 0.000163494391 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 16 / reader_11617_fileid_6929 | [-18,-15) | 0.000609064981 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 17 / reader_11299_fileid_2399 | [-18,-15) | 0.00046385205 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 18 / reader_11794_fileid_5599 | [-18,-15) | 0.000947087021 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 19 / reader_11963_fileid_10405 | [-18,-15) | 0.00812196045 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 20 / reader_11794_fileid_451 | [-18,-15) | 0.00171006679 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 21 / reader_11057_fileid_7856 | [-18,-15) | 0.000608783202 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 22 / reader_10676_fileid_7317 | [-18,-15) | 0.000223978769 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 23 / reader_11357_fileid_9224 | [-18,-15) | 0.00261592173 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 24 / reader_11713_fileid_3877 | [-18,-15) | 0.000199249771 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 25 / reader_11273_fileid_1479 | [-18,-15) | 0.000217576415 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 26 / reader_11065_fileid_2673 | [-18,-15) | 0.000786468816 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 27 / reader_11450_fileid_4097 | [-18,-15) | 0.000198661535 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 28 / reader_11912_fileid_8148 | [-18,-15) | 0.000461528101 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 29 / reader_11364_fileid_11310 | [-18,-15) | 0.0072883073 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 30 / reader_11128_fileid_5694 | [-18,-15) | 0.000125128277 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 31 / reader_11203_fileid_9673 | [-18,-15) | 0.000166760029 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 32 / reader_11046_fileid_10711 | [-15,-10) | 0.000119853165 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 33 / reader_11077_fileid_1780 | [-15,-10) | 0.00193353695 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 34 / reader_11139_fileid_4052 | [-15,-10) | 0.00156934848 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 35 / reader_11981_fileid_612 | [-15,-10) | 0.000562494154 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 36 / reader_11332_fileid_161 | [-15,-10) | 0.0008917956 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 37 / reader_10715_fileid_5533 | [-15,-10) | 0.00273538786 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 38 / reader_11265_fileid_6999 | [-15,-10) | 0.000401661905 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 39 / reader_11297_fileid_4129 | [-15,-10) | 0.000230021898 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 40 / reader_11301_fileid_9495 | [-15,-10) | 0.000119157473 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 41 / reader_11216_fileid_6365 | [-15,-10) | 0.000327643447 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 42 / reader_10929_fileid_6552 | [-15,-10) | 0.00194695493 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 43 / reader_11917_fileid_1923 | [-15,-10) | 0.0015323007 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 44 / reader_11254_fileid_2505 | [-15,-10) | 0.00199759691 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 45 / reader_11071_fileid_5369 | [-15,-10) | 0.00104060038 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 46 / reader_11301_fileid_8115 | [-15,-10) | 0.000500290772 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 47 / reader_11458_fileid_4985 | [-15,-10) | 0.000458723977 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 48 / reader_10927_fileid_850 | [-10,-5] | 0.000883763993 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 49 / reader_11888_fileid_4088 | [-10,-5] | 0.00172862662 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 50 / reader_11446_fileid_7296 | [-10,-5] | 0.000438027602 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 51 / reader_11901_fileid_1583 | [-10,-5] | 0.000564891 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 52 / reader_10929_fileid_3122 | [-10,-5] | 0.000104919013 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 53 / reader_11195_fileid_7449 | [-10,-5] | 0.000306340419 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 54 / reader_11170_fileid_6209 | [-10,-5] | 0.00169458302 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 55 / reader_10785_fileid_3854 | [-10,-5] | 0.00100526698 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 56 / reader_10777_fileid_6785 | [-10,-5] | 0.00027366802 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 57 / reader_11553_fileid_9309 | [-10,-5] | 0.000522030121 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 58 / reader_11121_fileid_8729 | [-10,-5] | 0.0009398426 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 59 / reader_11961_fileid_529 | [-10,-5] | 0.000252541056 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 60 / reader_11560_fileid_7582 | [-10,-5] | 0.000530557069 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 61 / reader_11535_fileid_2947 | [-10,-5] | 0.00244465341 | False | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 62 / reader_11053_fileid_1482 | [-10,-5] | 0.000693358688 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |
| 63 / reader_11046_fileid_8124 | [-10,-5] | 0.000160181781 | True | 3.05175781e-05 | 4.57763672e-05 | 0 | 9.15527344e-05 | 0 |

Dev0 reproduces the earlier float64 diagnosis exactly: relative RMS0.00335558809712323, max residual3.0517578125e-05; absolute difference0. The attempt03 float32 value0.00335558806546032 and failed premise flag remain unchanged. Both old result JSON hashes match before/after; neither record/report was edited.

Per-source sr/channels/frames/container/subtype/bits and crop endpoint counts, per-record raw residual RMS/max/relative RMS, both bound values/counts, reasons and all192 source pre/post hashes remain in metrics.json. The receipt records actual timing/PID and code/config/registry/old-record hashes. All rows have evaluated geometry, all192 subtypes are PCM16; unknown processing-chain provenance remains explicit for all64. These are source/codec arithmetic checks, not acoustic performance or content recovery results.

This independent diagnostic is complete and stopped. No loss weight or training authorization was selected; Root decides subsequent work.
