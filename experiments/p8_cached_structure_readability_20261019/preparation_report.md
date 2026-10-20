# Cached continuous-structure readability: software prepared only

No real cached WAV/header/PCM read, no real probe fitting or main/validator execution. Only the two frozen metadata JSON hashes were checked. Current code writes only this new directory; previousUI/audio/sciencefiles untouched. No StoRM/Score/ASR/GPU/network/checkpoint/rawdataset/OFF access. Future fitting is16actual linearprobe learning operations, NOT “zero training”; no enhancementnetwork training.

Exact frozen320symmetricHann/160hop/FFT512 features and one-sidedweights1/2/1, HFbins128..240,16array_splitbands/logpowerfloor1e-12, two continuouscleantargets,119full/6simple, fivecurrent±2framecontext in[-2,-1,0,+1,+2] columnorder and fullsentencepopulationmoments. All completeframes retained, tailsreported; no energy/label filtering. Identity twoobservedtargetstatistics and trainmeanconstant zero fits.

Main fits normalizedcase/frameweighted Ridge lambda.01, unpenalizedintercept, trainfoldweightedmu/std with exactconstants mappedstd1,4LOSO×4arms=16dualoutputfits. Predictionsunclipped. Fullvsbaselines, NoisyvsFrontend pairedcaseMSE, eachspeaker/SNR/alltargetwisemeans retained. Train/heldcaseIDs, framecounts, mu/std/coefficients recorded. OOFpredictions+targets only in npz; no rawfeaturematrix/audio/controller export.

Independent business implementation imports no readout functions: separateperframeFFT/featurebookkeeping, augmentedweightedleast-squares vsmainnormal-equation. Shares NumPyFFTbackend and cache_io lifecycle/IO only; independentdecode path struct.iter_unpack vsmainfrombuffer. It recomputes allfeatures/folds/coefficients/caseMSE/OOFpredictions with disclosed arithmetic tolerance1e-8relative/1e-9absolute. This is not independent FFT library or independent statistical confirmation.

10small syntheticchecks passed: zero/constantdimensions/tail, independentfeatures, firstflux0/contextedges, one-sidedParsevalpower, solvevsaugmentedlstsq, exactconstantstd1, caseequalweights, duplicatedframeweight/lambda invariance, unpenalizedintercept/unclippedprediction, fixed16foldscope/OOF and heldtargetisolation.32tiny2framecases enforce fixedfoldscope, not large simulation or scientificevidence. No real source reads/fits.

Guard refuses torch/sklearn/ASR/audio/qualitylibraries, allsocket/process calls, sourceswrites, WAV/rawfeature/audioexports, unapprovedOFF/dataset/checkpointreads; only72cachedPCMpaths joined from frozenregistry+modelmetrics. Captured forbidden attempts fail evenif caught. Main/validator own exclusivemarkers,118s work+absolute120sfinalization,8MiBtotal. Code/config/plan SHApre/post included withoutnewmanifestframework. Cachepre/posthash/72PCMfiledecode/samplebytes recorded separately. Failures retained, no retry/retune. Independentvalidator protects mainmetricsSHA.

Root-reviewed Worker unique commands:
```
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cached_structure_readability_20261019/run_readout.py --execute-after-root-review
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_cached_structure_readability_20261019/validate_readout.py --validate-after-completed-run
```
These two cleantargets are continuousspectral proxies, not phonemes/H1H0/CAEG or enhancedcontentgain. LOSOcontrols currentprobe only;4speakers/8utterances/reusednoises/pretrainingunknown remain exploratory. Frontendfullsentencecontext makes comparisoncontext+representationreadability, notpureinformationloss. No automaticupgrade tohiddenprobe/verifier/CAEG/enhancementtraining.
