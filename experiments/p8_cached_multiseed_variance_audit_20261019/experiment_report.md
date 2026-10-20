# Cached four-seed finite-ensemble decomposition — CPU only

## Provenance and critical historical limitation

Existing64 current cache WAVs for all16 filename×seeds2026–2029, plus16 original clean paths from manifest rows16:32 (0-based),80 unique WAVs. Original metrics SHA89bbe3200c9d658115086d6f7400abb9cde696710dbae623a50b64ae6dc20a89; manifest SHA4349228558395bcdabf9a1e4ab4b23ac6256834169c4338dfd4daca03746558e. Each filename exactly four prescribed seeds/NFE30/path listed in old metadata, no replacement or seed selection. Current PCM16/mono16k/NONE, full equal lengths, no added centering/gain/lag/crop/resample.

**Original per_trajectory contains no WAV-byte SHA. Current80 byte/header/size locks certify present cache and pre/post stability only, NOT byte identity to generation time.** Source config/report/code/path and old metric metadata are provenance clues, weaker than historical audio hashes. No hashes invented, recovery/retrieval/re-generation performed. Results describe the current frozen cache; any historical mutation remains unresolved.

Old configuration is80k EMA **Router active**, manual prepare_specs/build_rrsd_score_calibrator, DNS-CV exploratory16 files/8speakers. It is NOT current VoiceBank32 native Router-off. Historical heldout meant outside old selector development half, not final-test/pretraining-independent. Cache PCM already includes historical clipping and quantization; no raw/preclip uncertainty recovered. No new source noisy/frontend/checkpoint/audio decoded.

## Execution and validation

Read actual AGENTS/lowercase entry/analysis/config and full4551B attachment SHAa235e23451a88273189fb9d649904940a868725db05f3dd38983e49290b73a6b; self readonly SQLite codex_tools_compat, bridge39452/listening16022 live. No existing marker/model process; dirty tree preserved.

Unique main `python3 -B /data/coding/sgmse/experiments/p8_cached_multiseed_variance_audit_20261019/run_audit.py`, session66911 exit0, STARTED2026-10-19T07:27:08.529883+00:00, PID22326 recorded by actual STARTED (not claimed independently ps-observed), **11.891850348s**,80PCM decodes. Unique validator `python3 -B /data/coding/sgmse/experiments/p8_cached_multiseed_variance_audit_20261019/validate_audit.py`, session89363 exit0, **12.374187083s**,additional80PCM decodes. Total160 actual decodes,80unique sources. Main/validator each120s, newdir2MiB. Zero model/ASR/quality/network/training/newwaveform/resampling/retry. Mean signal exists only temporary RAM, no audio or samplearray exported. Source hashes/headers/lengths checked pre/post/final; git outside newdir unchanged; old metadata/code/recentVB32/residual metrics unchanged.

Validator independent implementation imports no business functions. V4 independently from six pairwise distances: sum_k<l mean((Bk−Bl)²)/16, direct temporary scalar four-way-mean residual vs S. It independently checks16 files/4 endpoint MSEs/9 strata/equalfile normalization/ratios/raw IID estimates, exact original current hashes. Mathematical agreement does not close missing historical hash gap. No bootstrap/optimization grid/newseed or newmetric.

## Definitions and principal descriptive result

K=4, M=sample mean (not true conditional mean), P=mean S², E_average=mean_k MSE(Bk,S), E_mean=MSE(M,S), V4=mean_k MSE(Bk,M). Exact finite-set identity **E_average=E_mean+V4**. The lower sample-mean squared error is algebra, not a newly learned method, tested audio system or real content gain.

Equal-file/P-normalized means: E_average **0.249055739459**, E_mean **0.217425239969**, V4 **0.031630499490**. Pooled normalized variance fraction **12.700168869%** = sum_i(V4_i/P_i)/sum_i(E_average_i/P_i). In this current historical Router cache, the complementary sample-mean residual accounts for **87.299831131%** of pooled normalized error; seed dispersion is not the majority. This does not identify the cause of remaining error or prove irreducible bias.

Different statistic, mean per-file V4/E_average **13.567428415%** (range5.954019840–19.300258058%). These two ratios have different weights and must not be conflated. Per-speaker pooled normalized fractions appear below; no generalized model-wide variance estimate.

**Only under an additional IID-same-condition seed-draw assumption**, V_unbiased=(4/3)V4 and raw finite4_bias2_est=E_mean−V_unbiased/4. Across equalfile/P means these are0.042173999321 and0.206881740139; negative raw bias-est count0/16 (negative values would be kept, not clipped). Neither is calibrated posterior uncertainty, true conditional bias, phone uncertainty or hallucination rate. Independence/IID and clean truth not proved by using four seeds. E_mean remains noisy finite-four mean residual. Router/noise/content/PCM confounds prevent transfer to currentVB-OFF.

## All9 strata, equal-file clean-power normalized MSE

| Stratum | files | Eavg/P mean | Emean/P mean | V4/P mean | IID unbiased V/P | raw IID bias2est/P | mean file V4/Eavg | pooled normalized V4/Eavg | negative bias count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 16 | 0.249055739459 | 0.217425239969 | 0.0316304994904 | 0.0421739993206 | 0.206881740139 | 0.135674284154 | 0.127001688695 | 0 |
| speaker_reader_10741 | 2 | 0.105534171201 | 0.0882828403712 | 0.0172513308302 | 0.0230017744402 | 0.0825323967611 | 0.163016494244 | 0.163466777005 | 0 |
| speaker_reader_10774 | 2 | 0.253306646748 | 0.219984923328 | 0.0333217234202 | 0.0444289645602 | 0.208877682188 | 0.155352927592 | 0.131546976157 | 0 |
| speaker_reader_10786 | 2 | 0.310566264539 | 0.268361477304 | 0.0422047872348 | 0.0562730496464 | 0.254293214893 | 0.146362110474 | 0.135896238754 | 0 |
| speaker_reader_10811 | 2 | 0.323346521352 | 0.286934106371 | 0.0364124149812 | 0.0485498866416 | 0.274796634711 | 0.118760908729 | 0.112611123289 | 0 |
| speaker_reader_10820 | 2 | 0.120039401906 | 0.11111683465 | 0.00892256725509 | 0.0118967563401 | 0.108142645565 | 0.0757280918232 | 0.0743303208234 | 0 |
| speaker_reader_10855 | 2 | 0.336082977106 | 0.293081653042 | 0.0430013240648 | 0.0573350987531 | 0.278747878353 | 0.147157535914 | 0.127948533529 | 0 |
| speaker_reader_10891 | 2 | 0.194956963703 | 0.169519477686 | 0.0254374860171 | 0.0339166480228 | 0.16104031568 | 0.147887464466 | 0.130477442477 | 0 |
| speaker_reader_10907 | 2 | 0.348612969119 | 0.302120606999 | 0.0464923621201 | 0.0619898161601 | 0.286623152959 | 0.131128739992 | 0.133363833932 | 0 |

## All16 files and four seed endpoint MSEs (raw amplitude MSE)

| File | samples | clean P | seed2026 | seed2027 | seed2028 | seed2029 | Eavg | Emean | V4 | V4/Eavg | raw IID bias2est | identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 016_reader_10741_fileid_10167.wav | 480000 | 0.000606536681325 | 4.80394592974e-05 | 4.96323202504e-05 | 5.54726624473e-05 | 5.19510281117e-05 | 5.12738675267e-05 | 4.30314145796e-05 | 8.24245294716e-06 | 0.16075348603 | 4.02839302638e-05 | -3.38813178902e-21 |
| 017_reader_10741_fileid_10788.wav | 480000 | 6.86435495639e-05 | 8.93978321304e-06 | 8.66465648481e-06 | 8.6487796274e-06 | 8.48944081226e-06 | 8.68566503438e-06 | 7.25010263898e-06 | 1.43556239539e-06 | 0.165279502457 | 6.77158184052e-06 | 4.23516473627e-22 |
| 018_reader_10774_fileid_11915.wav | 480000 | 6.86950231922e-05 | 6.34771481661e-06 | 6.32898365924e-06 | 6.42341223429e-06 | 6.49300157093e-06 | 6.39827807026e-06 | 5.16339389142e-06 | 1.23488417885e-06 | 0.193002580583 | 4.7517658318e-06 | 2.11758236814e-22 |
| 019_reader_10774_fileid_3627.wav | 480000 | 3.8842850941e-05 | 1.59447948623e-05 | 1.61658473274e-05 | 1.63535766633e-05 | 1.57776537759e-05 | 1.60604681572e-05 | 1.41700984635e-05 | 1.89036969374e-06 | 0.117703274602 | 1.35399752322e-05 | 0 |
| 020_reader_10786_fileid_10314.wav | 480000 | 6.18331753649e-06 | 2.75930970674e-06 | 2.8033712355e-06 | 2.69967513159e-06 | 2.75185605472e-06 | 2.75355303214e-06 | 2.41695482158e-06 | 3.36598210561e-07 | 0.122241411962 | 2.30475541806e-06 | -5.29395592034e-23 |
| 021_reader_10786_fileid_2142.wav | 480000 | 2.86583101105e-05 | 4.96155705963e-06 | 4.99388478347e-06 | 5.04612372024e-06 | 5.15243019521e-06 | 5.03849893964e-06 | 4.17952148734e-06 | 8.58977452299e-07 | 0.170482808985 | 3.8931956699e-06 | 0 |
| 022_reader_10811_fileid_1341.wav | 480000 | 0.000190482676148 | 8.52602010632e-05 | 8.25679897855e-05 | 8.18157324335e-05 | 8.31465474776e-05 | 8.31976176899e-05 | 7.47755588886e-05 | 8.42205880132e-06 | 0.101229566845 | 7.19682059548e-05 | 1.35525271561e-20 |
| 023_reader_10811_fileid_2306.wav | 480000 | 2.45276487782e-05 | 5.0310916539e-06 | 5.25241979825e-06 | 5.29568920416e-06 | 5.01621685689e-06 | 5.1488543783e-06 | 4.447105427e-06 | 7.01748951299e-07 | 0.136292250613 | 4.2131891099e-06 | -5.29395592034e-22 |
| 024_reader_10820_fileid_2971.wav | 480000 | 0.000228723978746 | 2.98278347259e-05 | 2.96491579851e-05 | 2.94324431665e-05 | 3.0397023214e-05 | 2.98266147729e-05 | 2.80507322117e-05 | 1.7758825612e-06 | 0.0595401984006 | 2.7458771358e-05 | 1.05879118407e-21 |
| 025_reader_10820_fileid_7471.wav | 480000 | 0.00272546922345 | 0.000303396529244 | 0.000299876487763 | 0.000296488458355 | 0.000295895299083 | 0.000298914193611 | 0.000271439201002 | 2.74749926097e-05 | 0.0919159852458 | 0.000262280870132 | -1.69406589451e-20 |
| 026_reader_10855_fileid_10269.wav | 480000 | 4.59951400563e-05 | 2.56395816745e-05 | 2.64834805469e-05 | 2.5098713274e-05 | 2.6562530268e-05 | 2.59460764408e-05 | 2.28625086605e-05 | 3.08356778029e-06 | 0.118845243801 | 2.18346527338e-05 | 8.47032947254e-22 |
| 027_reader_10855_fileid_11779.wav | 480000 | 0.00131536890999 | 0.000142140585937 | 0.000143735653065 | 0.000142754653048 | 0.000139930563519 | 0.000142140363892 | 0.000117199018684 | 2.49413452077e-05 | 0.175469828026 | 0.000108885236949 | -2.03287907341e-20 |
| 028_reader_10891_fileid_10483.wav | 480000 | 0.0027267646047 | 0.000148922992558 | 0.00014491414731 | 0.000143616311914 | 0.000148539121321 | 0.000146498143276 | 0.000121312113227 | 2.51860300485e-05 | 0.171920472747 | 0.000112916769878 | 1.01643953671e-20 |
| 029_reader_10891_fileid_7534.wav | 480000 | 0.00154855824607 | 0.000513224520154 | 0.000525884269849 | 0.00052285372573 | 0.000520463860099 | 0.000520606593958 | 0.000456127147377 | 6.44794465808e-05 | 0.123854456185 | 0.000434633998517 | 6.77626357803e-20 |
| 030_reader_10907_fileid_11974.wav | 480000 | 3.13881107431e-05 | 1.1578158359e-05 | 1.18468446715e-05 | 1.16425084571e-05 | 1.15013681624e-05 | 1.16422199125e-05 | 9.708778537e-06 | 1.93344137551e-06 | 0.166071538765 | 9.0642980785e-06 | 0 |
| 031_reader_10907_fileid_2839.wav | 480000 | 0.000122282348499 | 4.00903124149e-05 | 3.92694890538e-05 | 4.03664623562e-05 | 3.98835844244e-05 | 3.99024620623e-05 | 3.60644061919e-05 | 3.83805587044e-06 | 0.0961859412195 | 3.47850542351e-05 | -8.47032947254e-22 |

## Full16 normalized quantities

| File | Eavg/P | Emean/P | V4/P | IID unbiasedV/P | raw IID bias2est/P |
|---|---:|---:|---:|---:|---:|
| 016_reader_10741_fileid_10167.wav | 0.0845354767574 | 0.0709461041755 | 0.013589372582 | 0.0181191634426 | 0.0664163133148 |
| 017_reader_10741_fileid_10788.wav | 0.126532865645 | 0.105619576567 | 0.0209132890783 | 0.0278843854378 | 0.0986484802074 |
| 018_reader_10774_fileid_11915.wav | 0.0931403436951 | 0.0751640170056 | 0.0179763266895 | 0.0239684355861 | 0.0691719081091 |
| 019_reader_10774_fileid_3627.wav | 0.4134729498 | 0.36480582965 | 0.0486671201508 | 0.0648894935343 | 0.348583456266 |
| 020_reader_10786_fileid_10314.wav | 0.44531968735 | 0.390883179994 | 0.0544365073562 | 0.0725820098082 | 0.372737677542 |
| 021_reader_10786_fileid_2142.wav | 0.175812841727 | 0.145839774614 | 0.0299730671134 | 0.0399640894845 | 0.135848752243 |
| 022_reader_10811_fileid_1341.wav | 0.436772620863 | 0.392558317643 | 0.0442143032199 | 0.0589524042932 | 0.37782021657 |
| 023_reader_10811_fileid_2306.wav | 0.209920421842 | 0.181309895099 | 0.0286105267425 | 0.0381473689901 | 0.171773052852 |
| 024_reader_10820_fileid_2971.wav | 0.130404406816 | 0.122640102562 | 0.00776430425412 | 0.0103524056722 | 0.120052001144 |
| 025_reader_10820_fileid_7471.wav | 0.109674396995 | 0.0995935667393 | 0.0100808302561 | 0.0134411070081 | 0.0962332899873 |
| 026_reader_10855_fileid_10269.wav | 0.56410473822 | 0.497063573077 | 0.0670411651431 | 0.0893882201908 | 0.474716518029 |
| 027_reader_10855_fileid_11779.wav | 0.108061215993 | 0.0890997330062 | 0.0189614829865 | 0.0252819773154 | 0.0827792386773 |
| 028_reader_10891_fileid_10483.wav | 0.0537259956445 | 0.0444893970745 | 0.00923659857001 | 0.01231546476 | 0.0414105308845 |
| 029_reader_10891_fileid_7534.wav | 0.336187931761 | 0.294549558297 | 0.0416383734641 | 0.0555178312855 | 0.280670100476 |
| 030_reader_10907_fileid_11974.wav | 0.370911776366 | 0.309313886919 | 0.0615978894472 | 0.082130519263 | 0.288781257103 |
| 031_reader_10907_fileid_2839.wav | 0.326314161872 | 0.294927327079 | 0.0313868347929 | 0.0418491130572 | 0.284465048815 |

## Preservation and stop

Source80 current PCM SHA/header/size in source_PCM_locks.json, old per_trajectory/path evidence in source_registry.json, oldmetadata/code and recentVB32/residual metric protection in source_metadata_pre/post.json. All checked again by independent validator and final reporting, zero old file writes. Main metrics SHA 1ba98c0cd57911327412339baa6b1b46d1ecf96bb73a3a00c587af10d41c942e. No metrics were backfilled into historical records.

This completed cached-only decomposition weakens a majority-seed-dispersion explanation **for this cache and this full-PCM squared-error endpoint**. It is not a currentVB32 matched multiseed experiment, conditional-posterior calibration, phoneme correctness label, selection method, or quality/perception guarantee. Remaining error could reflect bias, reference/conditioning issues, residual noise, phase/timing or deterministic processing; this algebra does not separate them. No averaged audio/quality/ASR evaluation or learnedmethod run. Old multi-trajectory selector/anchor/closedVB32 stay closed. Worker stops without extra experiments, outputs, recruitment, training or follow-on inference.

## Final descriptive clarification requested by Lead

Across all16 prescribed files, median V4/E_average = **0.130073353399** (range0.0595401984006–0.193002580583). This is an unselected full-set descriptive median, not a new objective. Remaining87.2998% is finite-four ensemble-center reference error, **not proven true generator bias or posterior miscalibration**; the explicitly IID-assumed raw bias2 estimate is separate. Four sample trajectories cost4× single30-NFE generation historically; the algebraic mean-MSE reduction is not new empirical enhanced quality and no mean audio is exported or scored.
