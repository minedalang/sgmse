# 10720 engineering pairing and playback findings

Scope: current 10720 listening page, prior candidate pages, original P8 generation code and records. No inference or source audio changes. Existing repository modifications were present before this investigation.

## Confirmed playback defect and repair

The preserved `../p8_10720_content_identity_audit_20261018/listening_before_playback_fix_20261018.html` is the exact former current page (SHA256 `5bd54dd6bd830f527dc51c94d57d1a098f113c8b2a9229e31187ed04616dcccc`). Its lines 60–67 pause current players before awaiting AudioContext.resume, metadata, and play(). Another click or Stop during those waits does not cancel the pending request. Thus stale clean/off requests can play together, a stale play rejection can stop the latest request, and multiple RAF loops can run. This is a confirmed implementation defect; whether it occurred during the user's particular listening session is unknown.

Current `../p8_10720_content_identity_audit_20261018/listening.html` lines 59–94 increment a generation on every Stop/play, cancel pending metadata listeners, guard every await continuation and stale rejection, and keep one current RAF loop. It also visibly displays current arm, filename and window. Audio URLs, RMS constants, gain calculation, WAV files and window mappings were preserved. New regression harness `playback_regression_20261018.cjs` executes the actual inline before/after JavaScript with controlled deferred resume, metadata and play promises. Six cases fail before and pass after; stop while play promise is pending passes both. Results are `playback_regression_results_20261018.jsonl`. Browser integration remains a separate validation gate.

## URL and index mapping

Former page lines 45–51 and current lines 46–52 route all buttons to `../p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_{clean,off,y_pre,router,noisy}.wav`. Current context group includes all five conditions and A/B/C groups include clean/off/y_pre. These are static mappings; no condition shuffle or asynchronous URL replacement occurs. The corrected candidate page lines 24–29 uses the same five files with native audio controls. Its preserved before-fix HTML has no global pause handler, so users can also start multiple native audio players simultaneously. The current corrected candidate HTML now pauses all other players on each play event and displays current arm/filename; the exact previous page was preserved as listening_before_single_player_fix_20261018.html.

The candidate prefix `03` and source export prefix `04` refer to different row sequences: candidate row 3 is fileid10720, export row 4 is fileid10720. Corrected candidate metrics.json lines 207–239 explicitly maps this candidate to `outputs/{arm}/04_*fileid_10720.wav`, original noisy and clean_fileid_10720.wav. This difference alone is not an index bug.

## Known prior coordinate bug

Historical `../p8_e1_candidate_listening_20261018/make_pack.py` lines 27–42 opens already-centered 12s files for clean/y_pre/off/router but full 30s noisy, then applies `(original_time - 9s - 0.75s)` to every arm. For 10720 this extracts noisy 4.75–6.75s versus all other arms' original 13.75–15.75s. This is a real prior 9s mismatch. The builder is now explicitly invalidated at lines 7–10 and old HTML has a warning. Its warning quotes 9.75–11.75 versus 18.75–20.75, which applies to the 19.5s candidates, not 10720 specifically.

Original central12s builder `../p8_frontend_backend_audit_20261020/make_listening_pack.py` lines 38–55 computes a centered crop independently from each file length. `listening_pack/key.csv` lines 18–21 records start9 duration12 for all four 10720 arms. Corrected candidate metrics records original13.75–15.75 and 480000-sample originals. These are declarations and artifact identity records; worker independent waveform checks provide stronger actual crop validation.

## Original generation code and causal provenance limit

Original manifest.csv line 6 associates index4 with fileid10720, reader10929 and the correct dataset paths. The present `inputs/noisy/04_*fileid_10720.wav` symlink resolves to that noisy path.

`joint_enhance_by_dir.py` lines280–292 sorts input WAV paths and derives per-file Torch seed from BLAKE2b(input basename, digest_size4)+2026 modulo2^32. Lines304–317 load that same path, run frontend, and pass mode2 y_pre with noisy condition into enhance. Lines342–344 write to the same input basename. No separate index mapping, DataLoader shuffle, clean input or fixed12s crop exists in this script. For reproduction preserve the **04_ prefixed input basename**, since changing a basename changes the random seed. Input processing order does not change this per-file seed.

`joint_ablation_by_dir.py` lines103–122 similarly reads each input path and writes frontend output under the same basename. It has no explicit seed flag; the config's seed2026 is not enforced in this frontend-only script. No randomness is evident in the eval frontend path, but deterministic output must be verified, not assumed.

`sgmse/frontends/storm.py` lines251–262 reconstructs waveform with the incoming length; `joint_enhance_by_dir.py` lines74–86 can center-crop/pad a frontend length mismatch. `sgmse/joint_model.py` lines2903–2911 matches to noisy condition length, and line2956 reconstructs to T_orig. Current code contains no central12s-to30s export repadding. A 30s file header alone would not prove that historical runtime behavior, and current code may differ from the executed version.

Original logs/{off_retry,router_retry,y_pre_retry}.log lines1–8 record successful eight-file runs and aggregate settings. They do not record full command lines, per-file resolved input path/hash, output hashes at write time, script hashes or frontend intermediate hashes. Original metrics list final output hashes and shapes. No preserved manifest selection builder or corrected-candidate builder was located among repository Python files. Therefore current clip matching a declared source and that source matching an old hash establish identity, not that the source was produced from the intended input. The present code and naming support the intended pairing but do not close historical causal provenance.

## Bounded replay design, if independent waveform evidence leaves provenance unresolved

Replay only original fileid10720, full30s, into a new exclusive audit directory. Preserve exact `04_*fileid_10720.wav` basename and its symlink target. Lock checkpoint SHA256, relevant current code hashes, environment, exact command and resolved input hash; use original EMA/PC/N30/corrector none/SNR0.5/seed2026 settings. First capture frontend pre/post-match lengths, hash and max-absolute difference against the existing y_pre output (account for PCM16 write quantization), then run off/router once each with the exact per-file seed. Capture input/output hashes and compare full waveforms with old originals. Exact PCM match strongly closes current replay provenance; if not exact, record differences and diagnose code/checkpoint/runtime drift before any further run. Do not rename the input, use only a 2s crop, run a new seed sweep, or overwrite existing outputs. This design is proposed only; no replay was executed by this engineer.

## Browser integration attempt

The real-browser harness is playback_browser_regression_20261018.cjs alongside the current page. Isolated Playwright and120MiB Chromium headless shell plus2.2MiB FFmpeg were installed under /data/codex-state/tmp/audio-paircheck-browser-20261018 (scratch326MiB). A stalled direct download was cancelled and the one allowed proxy7890 retry succeeded. Chromium then failed before any test: eight required shared libraries are absent (recorded in playback_browser_environment_20261018.json). No system package install or global environment change was made. Real-browser integration is therefore unverified; mocked async regression tests pass.
