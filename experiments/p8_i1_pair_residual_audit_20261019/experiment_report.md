# I1a pair residual audit: completed

All12pairs retained; gain/lag fits descriptive only. X-S algebraic reconstruction is not pure noise proof. OLS orthogonality is enforced on its support, not centered independence or per-frame noise purity. Best-of321 correlation has selection bias; range never expanded. No new audio/model/ASR/training. Scientific/noise/split gates remain false. Full summaries and all lag/frame rows retained.

## Worker execution and exact accounting

Only launch_authorization.json I1a audit scope executed; configfalse fields remain frozen preparation snapshots. Main exactcommand `/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_i1_pair_residual_audit_20261019/audit_pairs.py --execute-after-root-review`, session13042 exit0,3.054512817s, PID40435 from actual STARTED/runtime; first ps query already terminal, no independent observation claimed. Unique completed-only validator `/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_i1_pair_residual_audit_20261019/validate_audit.py --validate-after-completed-run`, session39694 exit0,7.332283943s, PID40525 from validation marker/result. Neither rerun or synthetic call. Eachhard120s and totaldirectory8MiB retained. No guard change or ignored forbidden event (both forbidden_attempts=[]).

Preflight24whole-WAV hashes2,770,724B; main decode-prehash24/2,770,724B+sourceposthash24/2,770,724B; validator decode-prehash24/2,770,724B+posthash24/2,770,724B. Total120fullfile hashes/**13,853,620B**. Wholefile hashes include payload but do not decode samples. Main24PCMdecodes=1,384,834samples/2,769,668payloadB; validator additional24=same; total48decodes/**2,769,668sample values /5,539,336decoded payloadB**. Header parsing is part of each decode; no extra header-only source inspection by Workerfinalization. Root pre-review did not read sourceWAV; inherited locks became actual current-byte verification at Workerpreflight.

Actual full12 summaries, **3,852lag rows** (321×12) and **4,330allframe rows**, including partialtails, were independently reconstructed. Validator uses separate struct.iter_unpack int16 decoding (mainnp.frombuffer), independent centered-dot/residual calculation without importing driver stats; both use NumPyfloat64. Fixed numerical tolerance is consistency checking, not scientific threshold. It verified allOLS full/common/frame values, raw residual and reconstruction, source locks/preparationlocks, originalmetricsSHA. No samples/frames dropped.

## All12 results

Positive lag would mean X delayed relative S on fixed S[160,L−160), X[n+lag] support. Everybestlag=0, none boundary; this is best-of321 descriptive, not proof of global exact alignment or perceptual correctness. Common-supportzeroPearson differs from fullzeroOLS support, explicitly labeled below.

| ID | bestlag(samples) | commonzeroPearson | fullzeroOLSgain | rawX−S power | rawX−S centeredPearson withS | framecount |
|---|---:|---:|---:|---:|---:|---:|
| p233_027 | 0 | 0.947475188472 | 1.00162365876 | 0.000490080662303 | 0.00477664242948 | 389 |
| p233_028 | 0 | 0.852308078913 | 0.989580738706 | 0.00101943678629 | -0.0171227060987 | 346 |
| p236_036 | 0 | 0.681051376989 | 0.997833066478 | 0.00403980440048 | -0.00201551170481 | 342 |
| p236_040 | 0 | 0.66178399325 | 1.01154845198 | 0.00150219517654 | 0.0100551049142 | 375 |
| p239_026 | 0 | 0.944279468146 | 0.995843263257 | 0.000314572564567 | -0.0119092273115 | 344 |
| p239_028 | 0 | 0.680276694317 | 0.912976154184 | 0.0044909120372 | -0.0879231489654 | 325 |
| p243_026 | 0 | 0.981794538863 | 1.00323287103 | 5.46537119835e-05 | 0.0165916806445 | 444 |
| p243_027 | 0 | 0.942964498664 | 1.00556394903 | 0.000480461875236 | 0.0156427920422 | 408 |
| p244_028 | 0 | 0.847474129273 | 1.00580496906 | 0.00165054406994 | 0.0091540844795 | 322 |
| p244_034 | 0 | 0.980006862839 | 1.00042265395 | 0.000205826989024 | 0.00207805145003 | 327 |
| p250_038 | 0 | 0.980840322687 | 1.00108423842 | 0.000135120783367 | 0.00543617742838 | 356 |
| p250_050 | 0 | 0.670681426711 | 0.989791309939 | 0.0050572861042 | -0.0093110365119 | 352 |

FullzeroOLSgain range0.912976154184–1.011548451978, median1.000753446184 (n12). Commonzerocorr range0.661783993250–0.981794538863 (n12). Raw residualpower range0.000054653711984–0.00505728610420, median0.000754758724299 (n12). p239_028 has lower fittedgain0.912976154184 and rawcorr−0.0879231489654; this may warrant future design review but is not a posthoc exclusion, noise-source identity or actual speech-leakage rate. No threshold-adjustment/source repair/realigned output performed.

All24signals have zero exactnegative/positivePCM endpoint counts; this does not prove no prior clipping or processing. X=S+(X−S) reconstructionmaxerror0 for everypair is algebraic identity, NOT residualpure-noise or physicallyindependentadditivity. OLS residual innerproducts forcednear0 on fit support do not force centeredPearson0, nor guarantee per-frameindependence or pureN. Allfull/raw/commonbestOLS statistics plusallframes remain in metrics/CSV; lowcorrelation cannot prove absence of nonlinear/phase speechleakage. Best-of321 comparison has selectionbias even though allbest0; range not expanded.

## Preservation, scientific limits and stop

All8prepared and20I0source locks matched;548priorI0/VB32/residual/cachedexperiment fileshash-identicalpre/post/current. OriginalmetricsSHA `93af91ade77fd6072b40a7e8ef915edf360dc4748f4223eca15287aa1d6a08a6` matches independentvalidation. Registry/I0/source data/checkpoints/oldreports/code/preparationconstantbackup untouched. No sourceWAV reread for reporting. Externalgit differences relativepreflight:

````json
{
  "added": [],
  "removed": []
}
````

Root may concurrently update/data/research documents outside repo; not claim unconditional wholefilesystem equality. Bridge39452/listening16022 retained; no push. Source24currentbyte/header checks establish lockedinputs, not prior official48kconversion equality (oldgatefailure remains).

**noiseidentityunknown, noise_disjoint_ready=false, split_frozen=false, scientific_gate_ready=false**. Technical numeric validation does not establish originalnoise source, residualpurity, event/phoneidentity, localnoisyidentifiability, calibratedevidence or CAEG. Zero newWAV/NPY/residualaudio, eventediting/mixing, frontend/backend/ASR/quality calls, training, resampling or network. FullrawX−S is only computedmemory/statistics; no datasetcreation or counterfactual labels. Workerstopsafteraudit; I1eventediting/D0–D4/CAEG remain unauthorized.

## Lead disposition after reviewing all12 results

Root explicitly retains zero-lag raw a=1 as recorded, authorizes no shift or OLS-gain data modification, and retains p239_028 without deletion. Residual purity remainsunknown. No I1b/edit stage is initiated. Root independently read metric/validation/12summary metadata, not sourceaudio, and may update research documentation concurrently.
