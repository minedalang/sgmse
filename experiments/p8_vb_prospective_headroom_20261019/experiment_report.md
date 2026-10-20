# Prospective VoiceBank headroom: metadata-only registration

## Status and authorization
Metadata selection/file/reference verification and scoped project-use classification completed; **adequacy for future scientific use remains for Root review**. This is not a holdout certificate, phonetic alignment validation, input preparation or model execution. No permission for the conditional 32-case scientific proposal is implied.

## Frozen method and exact execution
Read `/data/AGENTS.md`, lowercase entry, Root reassessment memo, metadata_plan.md and unchanged config.yaml. Full attachment read: 4551 bytes, SHA256 a235e23451a88273189fb9d649904940a868725db05f3dd38983e49290b73a6b. Self readonly SQLite confirmed thread 01a152d6-4ba1-7b22-a983-bc96a5514c3d /root/experiment_worker_prospective_registry uses codex_tools_compat; actual ps observed bridge PID 39452 and listening PID 16022. Dirty git state preserved in git_pre.json/git_post.json.

Actual command: `python3 -B /data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/register_metadata.py` (one exclusive METADATA_STARTED; PID/time in execution_provenance.json). Registration processing 0.975467245 seconds. Unique independent check: `python3 -B /data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/validate_metadata.py`, 0.272498317 seconds. Both standard library only, no original selector imported into validator. No rerun/retry.

Source manifest SHA exactly 7f14a16a20880031932ab4ce7f7d3682c1b16cc0cc2eb8cd141e49ad7de087bc. Select train, exclude p226/p287, integer ID>=26, existing metadata mono16k/3–5s inclusive; lexical first four eligible speakers, integer first two eligible utterances. Selection does not consult text semantics, model scores or sound. Selected clean inspected only existence/stat/raw-file SHA; noisy paths retained but never opened. Publisher ZIP original bytes and normalized reference independently checked. No audio parsing/library/header reparsing.

| ID | metadata seconds | raw file bytes | SHA256 |
|---|---:|---:|---|
| p227_030 | 3.300625 | 105664 | `44052c2ff12721252dd217cd6dbafe07ebf22cb55a436267ef914008adfef6f3` |
| p227_031 | 3.1878125 | 102054 | `9b808f75ef4bd6ce7595afd43089845e91cd1da9f98ef5944f4707bcb0153636` |
| p228_026 | 4.61 | 147564 | `b4ee68c3f748f9f04a23e9cdba212771e156da10b8b4b13b155a1f9fdf18e009` |
| p228_032 | 4.1963125 | 134326 | `f2e2a4c41ac0832d668fb7eac2ddbd473f4fdbeaaf4bd2dc3694f257a6839567` |
| p230_027 | 3.197625 | 102368 | `b30813cf7984fe0a3bdb83229f90a399f352790697800490d5731bcbfac32a5d` |
| p230_029 | 3.5131875 | 112466 | `54dde5535bd22502cf7b81713e9bb3734554e4c9da4e62a781d4d3d2e0c19799` |
| p231_029 | 3.1156875 | 99746 | `3a7362c574e846d79d0fd38f3c4f11f746e822801763d4621e7c3aa51db6caa8` |
| p231_032 | 3.48 | 111404 | `0902ca75becb519409829b2ff547a888d2d7b55931dd3836c2361a0f6c38ae26` |

## Use-history evidence and gaps
All current experiment/memo/log file paths enumerated. Text suffixes specified in use_history_scan.json: 1749 files, 61558463 bytes. Selected-ID path hits: 0. Four text-hit files are full filename inventory, SCP inventory, publisher archive-entry inventory and reference audit metrics listing discarded __MACOSX metadata; none is a selected-case enhancement/ASR/listening output. Source reference manifest was read in full, not evidence of scientific use. The other initially skipped >8MiB text, training coverage records, was subsequently streamed in full (10800 lines): zero selected-ID hits, see supplementary_history_review.json. Thus large-text skips are accounted for, not silently omitted.

1465 symlink paths inspected but targets not followed; binary/audio contents not searched. Most symlinks belong to final-eval, speaker, old DNS/VCV trials; no selected ID path hit. Actual old VB pilot IDs are p226_027/028 and p287_037/054, not these eight. Evidence supports **no explicit selected-ID model/listening use found in the stated current-project scan**, not proof of complete non-use: anonymized filenames/text, unindexed external records or pretraining could hide exposure. Registry intentionally leaves project_exposure_review pending; do not replace selected IDs/speakers or call this independent final holdout. Root must resolve adequacy of this limited exposure review before scientific-stage authorization.

## Preservation and limits
Source manifest, frozen config/metadata_plan, old VB attempt02 metrics and B-opening metrics hashes matched pre/post and independent check. Git entries outside this new directory identical. No data/source/checkpoint/old experiment/service edits. All eight retain original reference_status and pretraining_exposure=unknown. Sentence text does not establish realized-phone labels, timestamps or local conversion equivalence.

Counters: network/audio decode/resample/mix/model/ASR/training/retries all 0. No enhanced or prepared audio produced. Budget 60s registration processing, 2MiB new artifacts; final counted bytes in final_freeze.json. Independent validation passed metadata scope only. If Root finds actual prior use or cannot accept the scoped unknowns, stop this fixed selection rather than substituting favorable samples. No scientific conclusion about content headroom is available.

## Final use-history disposition and unique-text units

`final_history_review.json` is the authoritative final scoped classification, superseding provisional pending fields intentionally preserved in the original registry/metrics. Four hit files are metadata/reference inventories; no explicit selected-ID model/listening use found. This completes the scoped review, not global unseen verification. All skip and uncertainty limits above remain. Any discovered actual prior use stops selection without replacement.

Post-selection count only (no reselection): **8 distinct publisher normalized-text SHA256 values across 8 utterances and 4 speakers**. Hash groups are retained in final_history_review.json. These are text identities, not independent model-pretraining exposure or generalization units.
