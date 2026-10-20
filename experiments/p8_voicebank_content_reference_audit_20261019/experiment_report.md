# VoiceBank independent-content reference audit — local complete, ZIP retrieval capped

The local inventory is complete;publisher training-text mapping is incomplete because the authorized aggregate networkstage deadline was reached during ZIPbody download. No further request/retry was made. No transcript manifest was created,and transcript availability/CRC/actualZIPchecksum/IDmapping remain unverified. This is a source-audit outcome,not a model experiment.

|Local source|IDs/pairs|Speakers|Coverage|
|---|---:|---:|---|
|train.scp,official-train IDs|11,572|28|UniqueIDs;no duplicates or malformed IDs|
|Existing16k/train|10,802pairs|26|Clean/noisy names and headers match|
|Existing16k/valid|770pairs|2(p226,p287)|Disjoint from localtrain IDs/speakers|
|test.scp/filenames only|824|2(p232,p257)|No train/testID or speaker overlap|

The localtrain+valid union exactly equals train.scp with no missing/extra IDs. Testclean/noisy filenames match824SCPIDs,but no testaudio header,decode,transcript or content was opened. All23,144train-source(train+valid)WAVheaders were inspected without waveform decoding:16kHz,mono,PCM16,zeroheadererrors and no paired rate/channel/frame/duration mismatches. Localtrain duration31,528.84825s(8.758h),valid2,279.57931s(.633h);overallindividualduration1.092375–15.1120625s. Existingtrain/valid partition is reported as found,not newly selected. Full filename/SCP inventories and compressed per-file header records are retained. Localinventory took5.15944s;independent header-row/pair/ID checks passed.

The boundedVBroot scan found no localTXT/JSON/TSV/ZIP transcript metadata,onlytrain.scp/test.scp. The present SpeechBrain-style voicebank_prepare.py has SHA256a659ad6f96441f4afe97e05e50638b6f45948b42efc2ed79b3bc545221076d6f. Its static download_vctk uses48k→16k resampling and whole-signal save without explicit crop/trim;create_json removes punctuation,uppercases/strips words,lexicon-maps phones and removes adjacent duplicate phones. It is not a timing/phonetic-groundtruth annotation. Its defaultvalidation speakersp226/p287 match currentfolders,but localdirectorylayout differs from its expected originalfoldernames. Codepresence/configuration compatibility does not prove actual execution,resampling fidelity or absence of undocumented transformations. No preparation code was imported/executed.

Official provenance saved:

- [Publisher item](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757) as publisher/item.html and full itemmetadata JSON.
- [Official end-user license](https://datashare.ed.ac.uk/server/api/core/bitstreams/7e7bc32f-e94a-436a-8e97-416a972c7e1a/content) as publisher/license.txt;actual text verifies CC BY4.0.
- [Training bitstream metadata](https://datashare.ed.ac.uk/server/api/core/bitstreams/fb55b5de-c2db-4451-ac0a-134e98011ffa) as publisher/train_bitstream_metadata.json:trainset_28spk_txt.zip,expected6,489,312bytes,MD5expected48bfcd5945165c60ab9d2f312f4d6e1c.
- [Official training ZIP endpoint](https://datashare.ed.ac.uk/bitstreams/fb55b5de-c2db-4451-ac0a-134e98011ffa/download) respondedHTTP200,but its body did not finish before the cap. No complete archive is saved. ExpectedmetadataMD5 has NOT been checked against an actual archive;owncompleteZIPhash,CRC,safe-entry inspection,duplicates/emptytext checks andIDmapping remain unavailable.

Four complete official documents total602,590savedbytes;all savedSHA256values were independently rechecked. Proxyretrieval recorded4,534,750bodybytes,including3,932,160partialZIPbytes that were not saved. The initial direct itempage body was interrupted after stalling;its traceback demonstrates headers had arrived,but actualstatus/finalURL/partialbytecount were not persisted. Those fields remain unknown rather than reconstructed. This direct failure and a local metadata-recording missing-helper error are preserved in retrieval_attempt01_failure.json/.log and retrieval_attempt01.py. The helper was corrected for metadata recording only;no model/data code changed. Same-host direct unavailability prompted the single authorizedlocalhost7890proxy fallback;no subsequent retry was attempted.

The original180s aggregate network cap was preserved across direct/fallback stages using a conservative start estimate from originalscriptmtime. A body-read boundary returned at181.18s,an explicit1.18sdeadlineovershoot;it stopped before another read/request. Totalreceivedtraffic for the interruptedinitialdirectbody is unknown,so the recordedproxybytes must not be presented as exactall-attempttraffic. The incomplete ZIP was held transiently in memory and discarded when the stage stopped,not written as a corrupt finalarchive. Local preparedsnapshot/caps are preserved;config.yaml was updated to the authorized9MBdownload/20MBartifact/180snetwork budget before retrieval. Final artifacts are about1.6MB,well below20MB.

The train IDs provide a plausible exact-filename join key to independently published text,but no mapping is completed in this capped stage. Publishertext would be independent of ASR outputs;filenameIDmatching alone still would not prove spoken-word alignment or actual local conversion provenance. No ASR labels are substituted. Enhancement/ASR/SSL pretraining exposure is unknown,so a later project-independent pilot cannot be claimed as strict pretraining holdout. No new speaker selection,model calls,training,2×2runs or listening request occurred. Finaldecision/anyfurtherretrieval belong to ResearchLead.

Originaltrain.scp/test.scp/preparationcode hashes remain unchanged;datasetfiles were read only and all writes are confined to this newexperiment directory. Existingdirtyworktree,checkpoints and listeningserver are preserved. Rootresearchplan was not edited,no cleanup/commit/push.
