# VoiceBank training reference audit: attempt 02 completed

One authorized ZIP-only transaction completed in 9.63406 seconds of network time and 11.62429 seconds through verification and mapping. It used the existing proxy, with no retry, resume or refetch of the four saved official documents. Parent attempt 01 remains a preserved deadline failure. No test content, audio decoding, model calls, ASR, training, pilot selection or listening request occurred.

The [official training bitstream](https://datashare.ed.ac.uk/server/api/core/bitstreams/fb55b5de-c2db-4451-ac0a-134e98011ffa/content) returned HTTP 200, with no redirects. Curl streamed a fresh download to a durable `.partial` file. It used a 15-second connection timeout, 420-second total timeout, 7 MB file limit and zero retries, under an independent supervisor. The file was renamed only after complete size, official MD5, ZIP safety and all CRC checks passed. The old partial was never recoverable, so this was not a resumed transfer.

|Verification|Result|
|---|---|
|Archive|trainset_28spk_txt.zip|
|Actual size, matching official metadata|6,489,312 bytes|
|Actual MD5, matching official metadata|48bfcd5945165c60ab9d2f312f4d6e1c|
|Own SHA256|5ec7d6ef99b78a7d039f1ecd6db7d0d5b660877b464de15e1d75de6425fbf3f8|
|Archive members|23,148: 3 directories and 23,145 files|
|Declared uncompressed size|2,928,055 bytes|
|File CRC checks|23,145 passed|
|Resource or hidden metadata entries|11,575, including directories; excluded from labels|
|Nonmetadata transcript files|11,572|
|Missing or unexpected train IDs|0 / 0|
|Duplicate basenames or conflicting texts|0 / 0|
|Empty texts or decoding failures|0 / 0|

Every archive member passed checks for absolute paths, parent traversal, backslashes, drive prefixes, symlinks and declared size limits: at most 1 MB per member and 60 MB total. Mac resource metadata was inventoried and CRC-checked, then excluded from labels. No archive members were extracted into the dataset.

All 11,572 training IDs have one reference, covering the existing local partitions:

|Partition|Rows|Speakers|
|---|---:|---:|
|Local train|10,802|26|
|Local validation|770|2: p226 and p287|
|Combined official training source|11,572|28|

`train_reference_manifest.jsonl` stores each ID and speaker, the publisher and existing local partitions, clean/noisy paths, and the previously audited sample rate, channels, frame count and subtype. The transcript object stores its archive member, original byte count and SHA256, original decoded text, and a separate normalized text and SHA256. Normalization collapses whitespace only; case and punctuation remain intact. Original bytes remain recoverable from the verified ZIP. Manifest SHA256: `7f14a16a20880031932ab4ce7f7d3682c1b16cc0cc2eb8cd141e49ad7de087bc`.

Independent CPU validation checked every manifest row against the original archive bytes, CRC, decoding, normalized text, hashes, existing header metadata, filenames and SCP IDs. It confirmed unique coverage, the existing train/validation split, and no test IDs. See `mapping_validation.json` and `validate_mapping.py`. This validation opened no audio or headers, made no network requests and initialized no model.

The prior local inventory remains applicable: 11,572 train-source IDs and 824 test IDs, with no ID or speaker overlap. All 23,144 training-source clean/noisy headers are 16 kHz, mono PCM16, with no paired frame or duration mismatch. Test was limited to SCP IDs and filenames/counts; no test audio headers or transcript content were opened. The three original SCP/preparation-code hashes and every parent snapshot hash remain unchanged.

Publisher attribution: Cassia Valentini-Botinhao, *Noisy speech database for training speech enhancement algorithms and TTS models*, [University of Edinburgh DataShare](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757), [DOI 10.7488/ds/2117](https://doi.org/10.7488/ds/2117). The saved item metadata and [official end-user license](https://datashare.ed.ac.uk/server/api/core/bitstreams/7e7bc32f-e94a-436a-8e97-416a972c7e1a/content) identify CC BY 4.0. These official documents are reused from the parent directory without refetching.

These references are independent of automatic ASR output, but exact filename matching does not verify spoken words, timing, phonetic boundaries or local conversion fidelity. The local preparation script contains resampling and text normalization code; its presence and compatible default speaker split do not prove actual execution provenance. No waveform consistency gate was performed. Enhancement, ASR and SSL pretraining exposure remain unknown, so a project-independent pilot must not be described as strict pretraining holdout.

Attempt 02 transferred 6,489,312 payload bytes, below 7 MB. Combined with prior recorded proxy traffic, known network payload is 11,024,062 bytes, below 15 MB. The parent direct response's unpersisted byte count remains unknown; this is explicitly known traffic, not an exact all-attempt total. Additional artifacts are about 26.1 MB, below 30 MB. The complete ZIP, archive inventory, manifest, transfer headers, process metadata, progress and validation records are retained.

All writes are confined to attempt02. The prior failure, original datasets, checkpoints, existing dirty workspace, listening server and root research plan were preserved. No cleanup, commit or push occurred. Any clean-text consistency gate, pilot selection or model work requires a separate Research Lead decision.
