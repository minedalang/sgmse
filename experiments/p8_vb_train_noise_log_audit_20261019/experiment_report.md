# Official VoiceBank train28 mixing-log metadata audit

## Status, exact limits and source

Completed one bounded official fetch and one independent offline validation; no new audio experiment. Root froze analysis/config, source recordsverified; actual AGENTS/lowercase entry/full4551B attachment read and selfSQLite codex_tools_compat verified. Bridge39452/listening16022 retained; no duplicateSTARTED/modelprocess at preflight. Source/oldrecords/checkpoints/services not changed, no push.

Unique fetch command `python3 -B /data/coding/sgmse/experiments/p8_vb_train_noise_log_audit_20261019/fetch_official.py`, execsession91363, exit0, **7.985741623s**; PID43571 recorded in actualexclusiveSTARTED, not independently ps-observed. ThreeGET only, allHTTP200, no redirects/retries,20s/requesttimeout, officialHTTPS origin via localproxy7890. Totalresponsebody **136,499B** within512KiB;90s fetchalarm. Exact requestURLs/status/allheaders/bodyhashes in http_requests.json, actualbodies saved. No guessedmirror/contentlink; metadata navigated uniqueORIGINAL→literalbitstreams link(size20)→uniqueexactlogfiles.zip→literalcontentlink, nopaginationfallback.

Archive **115,932B**, publisherMD5 **0d16f5b6afe12d64238d1e4a0e6ff6e8** matches metadata+download, SHA256 **ab37c41f31bc2b6c204cc10c8e5bb8bde800307fdb49f277e366d986fefc9e39**. SharedZIPcontainer includes compressed test/56/readme bytes, **but only directorymetadata inspected and log_trainset_28spk.txt opened/decompressed**. Noextractall/testzip/testCRC/readme/56payload read. Archive and size/CRC directoryinventory retained. Trainentry **218,659B**, CRC32 **faa471b6**, SHA256 **462e42fbd0ab30f77cc5ad54fa97620aabe8cab9a8cd6eff3697708c5c01014e**.

Offlineparse uniquecommand `python3 -B /data/coding/sgmse/experiments/p8_vb_train_noise_log_audit_20261019/parse_training.py`, chunk02174c, 0.03490665403660387s. Independentunique `python3 -B /data/coding/sgmse/experiments/p8_vb_train_noise_log_audit_20261019/validate_offline.py`, chunkdb4604, **0.029835449s**, passed. Main+parse machineprocessing below90s; human/code preparation not reported as networkprocessing. Independentvalidator has0network and only reopenssame train28entry once, independentlyregexparses allrows,12join andunion-findgraph; verifiesarchiveMD5/SHA/trainCRC/raw/sourceprepost. Noaudio/model/ASR/quality/training/noiserecording download.

## What the actually opened schema supports

ASCII, **11,572 lines,11,572 uniqueIDs,0duplicates**, exactlythree whitespacecolumns: ID/literalnoise-label/numericcolumn3. Noheader or column-definition withintrainentry. Column3 rawvalues0/5/10/15 preserved, but **nominal_SNR_dB=null, semantics unconfirmed**: pattern resembles nominalSNR but officialitemabstract doesnot define this entry's columns. Do not infer actual/localSNR or nominaldB certainty from plausiblevalues. AdditionalREADMEwasnotopened. Root confirms remainingcolumnsemanticsneedseparateauthorization, not retroactiveexpansion.

Literalnoise labels are category/scenario-level sourceclues, not uniqueoriginalrecordingID. Nooffset/channel/gain/segmentlength/noise-sourceSHA supplied by openedthreecolumns; retainnull. Everyselectednoise_recording_id remainsunknown. OriginalI0 boundedlocalemptylogscan not wrong: this officialmetadataisnewprovenance outside its root/depthlimit. Oldregistry/metrics are not backfilled.

## All12 fixed-ID joins (no reselection)

| ID | speaker | literalnoise token | rawcolumn3 | parsednumeric | nominalSNRdB | sourceline |
|---|---|---|---|---:|---|---:|
| p233_027 | p233 | restaurant | 10 | 10.0 | null | 4202 |
| p233_028 | p233 | restaurant | 5 | 5.0 | null | 4203 |
| p236_036 | p236 | ssn | 0 | 0.0 | null | 32 |
| p236_040 | p236 | station | 0 | 0.0 | null | 36 |
| p239_026 | p239 | restaurant | 10 | 10.0 | null | 4966 |
| p239_028 | p239 | restaurant | 0 | 0.0 | null | 4968 |
| p243_026 | p243 | restaurant | 15 | 15.0 | null | 7844 |
| p243_027 | p243 | restaurant | 10 | 10.0 | null | 7845 |
| p244_028 | p244 | restaurant | 5 | 5.0 | null | 513 |
| p244_034 | p244 | station | 15 | 15.0 | null | 519 |
| p250_038 | p250 | traffic | 15 | 15.0 | null | 2516 |
| p250_050 | p250 | cafeteria | 0 | 0.0 | null | 2527 |

Rawsource lines retained for selected12 only in metrics; fullpublishertrainlog archived for provenance/schemavalidation, not used to replaceIDs.

## Graph as correlation-risk metadata only

Connect12utterances if same speaker or same literalnoise token; transitivecomponentsizes **10+2**:

* p233_027, p233_028, p236_036, p236_040, p239_026, p239_028, p243_026, p243_027, p244_028, p244_034
* p250_038, p250_050

These components dependonthatdefinedspeaker/labelrelation, **not proven actualrecordingidentity or finalizedsplit**. Samecategory could comprise multiple recordings; distinctlabels also do not certify independentrecordings. No pretendnoiseIDs or splitassignment. noise_disjoint_ready=false, scientific_gate_ready=false, split_frozen=false; noD4covariance/cleanresidualpurity/candidateidentifiabilityclaim.

## Preservation, limitations and stop

Sixsource metadata record locks exactbytes/SHA pre/post/validator/final equal, including I0/I1a/I1b originalmetrics. Root-frozenanalysis/configSHA unchanged. No oldsources/metrics/report/audio overwritten. NewmetricsSHA **ca1ffd3055f8c98e6fd77e3d9c351d7e55c52a743a086324c4fcc5570a8849da** preserved afterindependentvalidation. Sourcefilesystemonlyread; no WAV/hash/header/sampleaccess inthisstage. Runtimeonlystdio/network/ZIPPythonstandardlibrary, never modelcode.

Observedrepositoryoutside-newdir gitdifference:
```json
{
  "added": [],
  "removed": []
}
```

Rootmayconcurrentlywrite/data/research docs/reviewJSON; thisobservationisnotunconditionalwholefilesystemimmutability. An auxiliary readonlyattempt to print absentbitstreamdescription raisedKeyError after successfulfetch; noHTTP/ZIPpayload/schema retry, no sourcechanges. AllmainHTTP3 requests remainedunique. No noise-waveformretrieval, testcontentdecoding, I1bpositionreselection, listeners/platform, detector/ASR/training. Worker stopsafter metadataaudit; anyREADMEinterpretationornewsourceworkmustbeseparatelyapproved.
