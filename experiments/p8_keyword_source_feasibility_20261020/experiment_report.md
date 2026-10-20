# Keyword source feasibility: partial metadata; bounded run failed

The archive endpoint served both requested metadata ranges and its complete central directory was parsed. The run nevertheless FAILED: the third HTTP request for official grouping code timed out, and retained artifacts exceeded4MiB. No retry, mirror, full download, member extraction/read, WAV header/PCM decode, model or classifier training followed. The resource failure is preserved, not repaired by deleting/compressing records or raising the cap.

Archive: [official mini Speech Commands](https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip). [Official tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio) documentation supplied by Root describes8 words/8000 recordings/16k/≤1s and CC BY; the tutorial was not fetched again here. Archive README/license and recording contents were not accessed.

Unique PID5566, exec session47128, exit1, elapsed33.603908s.3 HTTP requests; total response body2143724 bytes≤3MiB. Timeout argument20s each; third request wall20.196192s includes timeout handling overhead. No HEAD, redirect, retry or mirror.

| Request | Requested Range | Status | Content-Range | Body bytes | Seconds |
|---|---|---|---|---:|---:|
| 1 | bytes=-65557 | 206 | bytes 182016796-182082352/182082353 | 65557 | 1.433166 |
| 2 | bytes=180004164-182082330 | 206 | bytes 180004164-182082330/182082353 | 2078167 | 11.881645 |
| 3 | None | timeout/no status | unknown | 0 | 20.196192 |

Archive length182082353; EOCD entries16020, central-directory offset180004164, size2078167. Both responses were206 with exact matching Content-Range. ZIP64/multidisk fallback was not used.

Tail SHA256 `a8539d66e17a28a740c684abd42c439623acbb3d3c389d78199471d6c0f4b7e1`; central-directory SHA256 `b9a648bea6e90561117605e2cfe10a44c38055429ee088c7429520f07a6c179c`.

Artifact gate FAIL: 5556584 observed bytes before final small report/receipt >4194304. CDbin2078167B + parsed entriesJSON3397730B + tail65557B already exceed the cap. This implementation omitted its artifact-size guard. All records remain preserved; mandatory small reporting adds further bytes and is not called compliant.

Offline cached-metadata counts: 8000 labelled WAV paths,8 word directories,1750 prefix groups,1516 groups crossing words. Duplicate paths0, unsafe/malformed paths0. Other entry counts: {'directory': 9, 'macos_metadata': 8010, 'other_nonlabelled_file': 1}. Filename prefixes are group identifiers, not verified speakers.

| Word | WAV paths | Prefix groups | Conditional train files/groups | Conditional val files/groups | Conditional test files/groups |
|---|---:|---:|---|---|---|
| down | 1000 | 702 | 783/552 | 115/79 | 102/71 |
| go | 1000 | 697 | 784/557 | 110/73 | 106/67 |
| left | 1000 | 708 | 784/561 | 107/73 | 109/74 |
| no | 1000 | 721 | 775/564 | 106/75 | 119/82 |
| right | 1000 | 707 | 787/558 | 109/76 | 104/73 |
| stop | 1000 | 683 | 798/553 | 96/65 | 106/65 |
| up | 1000 | 723 | 788/575 | 108/77 | 104/71 |
| yes | 1000 | 704 | 791/551 | 114/81 | 95/72 |

Conditional split arithmetic uses the explicit SHA1/nohash rule in metrics.json,10% validation/10% testing. Root separately web-verified that official rule. This Worker's input_data.py HTTP request timed out and downloaded0 bytes; it is not claimed that this run fetched or verified the source. These are locally calculated group splits, not a packaged split manifest. Group intersections between all split pairs are0 by construction.

Group word-coverage histogram: {3: 489, 4: 293, 8: 88, 6: 37, 7: 56, 2: 437, 1: 234, 5: 116}. Conditional file totals: {'training': 6290, 'testing': 845, 'validation': 865}; prefix group totals: {'training': 1399, 'validation': 177, 'testing': 174}.

Filename metadata supplies a plausible word-labelled, groupable source for a future content-transfer question, but recording labels, archive license, true speakers and pretraining exposure remain unverified. No clean recognizer validity or enhancement/content benefit follows. The bounded run failed its resource/network-source gates; no follow-up download, TensorFlow install, classifier training or clean readout execution is authorized.

A first offline report command had a syntax error before any writes; only local reporting was corrected. No driver/network operation was rerun. Current CPEL artifacts, Root documents, bridge and listening service were untouched. This unique check is stopped.
