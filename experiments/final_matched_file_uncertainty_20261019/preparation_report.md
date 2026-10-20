# Frozen final metric matched-file uncertainty: software prepared

No true bootstrap/CSV numerical analysis executed. Preparation read only sourceCSV headers and originalmetricsJSON schema/values, not audio/model/testwaveforms. Five small synthetic arithmetic/CSV checks passed. The purpose is a paired-unit uncertainty supplement, not repeated finalevaluation or methodselection. Oldmeans/data/arms/metrics frozen.

Main independently parses six CSV, requires exact150unique numericterminalfileIDs `(fileid_(digits).wav)` and finite sevenmetrics, identicalIDsets acrossallarms/splits. Two-split arm-minus-off differences averaged perID, then150ID equalmean. Means must reproduce originalpooleddelta within2e-10. Default_rng(PCG64)seed20261019 produces10000×150uint16 indices used jointly forall14columns. Main gathermean in250drawchunks, saveindex+10000×14drawmeans NPZ;95linearpercentileintervals. OriginalhistoricalCI and newmatchedCI explicitlysidebyside, never chosen bywidth/outcome.

Independentchecker imports no mainbusiness: separateCSVparse/join/perIDloops, sameNumPyPCG64serializedindexgeneration, everydraw bincount weights dotmatrix, separatelinearorder-statisticquantile. Sharedbootstrap_io onlylifecycle/IO/sourceguards. Tests coverallsyntheticdraws/means/endpoints, preservedhistory, constantcontrast, numericIDsort, duplicate/malformedterminal/nonfinite/countfailure. SameRNGbackend acknowledged, not independent libraryimplementation.

OnceWorkercommands AFTER RootfinalSHA review:
```
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/final_matched_file_uncertainty_20261019/bootstrap_main.py --execute-after-root-review
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/final_matched_file_uncertainty_20261019/bootstrap_check.py --validate-after-completed-run
```
Each60sharddeadline coveringfinalrecords, oneCPUthread, total8MiB. Exclusiveindependentmarkers/noretries. SixCSV+originalmetrics andnewcode/configSHA pre/post. Guarddeniestorch/audio/model/dataset/checkpoint/network/process/sourcewrites; no newmetricscomputation ortraining. ValidatorprotectsmainmetricsSHA. Runtimeconfig/metrics/report plusNPZcomplete. No needmanifest or anotherframework.

MatchedfileID is NOT speaker/noise/roomcluster, multiplecomparisonscorrection, crossseed/datasetconfirmation orcontentrecovery. No speakerlabel fabrication, no droppingbadrows, no tuningdraws/seed/intervalmethod to get favorableCI; failure retained. Existingmeanreproductionaudit did not compute this paired uncertainty.
