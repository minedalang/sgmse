# Consolidated 10720 pairing and playback forensics

Completed. The one authorized router-off replay exactly reproduced the saved full 30 s WAV and its 13.75–15.75 s window; frontend output also exactly reproduced saved Y_pre after PCM16 serialization. This strengthens causal input/output provenance beyond filename and hash checks. The canonical successful record is [replay_off/replay_result.json](replay_off/replay_result.json), pointing to [successful detailed metrics](replay_off/attempt02/metrics.json). The root replay_off/metrics.json preserves a failed import before checkpoint load, and is not the final replay status. See [replay report](replay_off/experiment_report.md).

The dedicated page had a confirmed asynchronous playback race; candidate native audio controls also permit simultaneous playback. The dedicated race was repaired. These are confirmed implementation defects, but the cause of the user's particular symptom is not proven. The [engineer findings](engineer_findings.md) detail the fix. The [Node harness](../p8_10720_content_identity_audit_20261018/playback_regression_20261018.cjs) and [test records](../p8_10720_content_identity_audit_20261018/playback_regression_results_20261018.jsonl) show 6 failures/1 pass before and all 7 passing after. This is a controlled JavaScript regression test, not actual browser validation.

Actual browser integration remains unverified: Chromium could not launch because eight shared libraries are missing. User browser verification remains the practical pending check; no further browser downloads or runtime fixes were attempted. A cache-versioned local GET of the [fixed page](http://127.0.0.1:8766/p8_10720_content_identity_audit_20261018/listening.html?v=playbackfix20261018) returned HTTP 200 and exactly matched current disk HTML. See [HTTP evidence](browser_install_metadata/fixed_page_http_check.json).

The exact isolated browser scratch directory `/data/codex-state/tmp/audio-paircheck-browser-20261018` was removed after saving package.json, package-lock.json and [browser failure summary](browser_install_metadata/browserenvfailure.json). Its allocated size was 340,951,040 bytes (325.16 MiB). Measured filesystem free space changed by 312,672,256 bytes; [cleanup record](browser_install_metadata/cleanup.json) preserves realpath, size and disk-free evidence. No other caches, environment, data or checkpoints were removed.

Prior top-level metrics/report/hash records are preserved in [prior_versions_20261018T131037Z/](prior_versions_20261018T131037Z/). The initial forensic findings below describe the state and observations at that earlier check; the successful replay and playback repair above supersede the earlier open causal provenance/playback questions. Current acoustic differences do not establish phoneme error or hallucination.

---

# 10720 pairing forensics

Status: completed. Read-only local HTTP and waveform checks; no ASR, inference, downloads, enhancement, training, or audio changes. Existing experiment records were inspected, git state captured in metrics.json, and existing source hashes matched candidate metrics. An initial execution failed because the default Python lacked soundfile; the existing repository environment succeeded without installing anything.

## Findings

- Current dedicated and corrected candidate pages serve the exact disk HTML and WAV bytes. All five current clips exactly equal source PCM samples [220000:252000] at 16 kHz, corresponding to 13.75–15.75 s. No crop gain or normalization was applied (least squares scale 1.0; maximum PCM difference 0).
- All 25 amplitude invariant centered waveform correlation searches of each 2 s clip across each entire 30 s source peak at 13.75 s. This independently checks the source range rather than only validating hashes.
- Full 30 s waveform lag searches (all lags with at least 15 s overlap) peak at zero for every arm pair. Off/router match their own clean and y_pre substantially more than any of the other seven P8 input fileids. No straightforward swapped-file mapping or 9 s output shift was detected within this bounded check.
- The historical candidate pack DOES contain a concrete mismatch for 10720: noisy is exactly original 4.75–6.75 s, while its clean/y_pre/off/router are original 13.75–15.75 s. The historical HTML warning gives another candidate’s example (9.75–11.75 vs 18.75–20.75); for 10720 the exact ranges above apply. Its current page is marked invalid but still constructs historical audio routes.
- Noisy RMS envelope alone peaks around 11.1 s away, with weak scores; its waveform peak is at zero and all clip searches peak at 13.75 s. Noise dominates that envelope, so its envelope peak must not be interpreted as a speech delay.

## Current HTTP provenance

- http://127.0.0.1:8766/p8_10720_content_identity_audit_20261018/listening.html — HTTP 200, disk HTML equality True, SHA256 `5bd54dd6bd830f527dc51c94d57d1a098f113c8b2a9229e31187ed04616dcccc`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_noisy.wav — HTTP 200, equals disk True, SHA256 `284b279d34a6a76b742fffc41e6cd90a1c96682426c36d166362b3e652b50ad7`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_clean.wav — HTTP 200, equals disk True, SHA256 `10435d243635f7b4135fbca220d6de45d2e501e6cdd0770abd209652ec85204c`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_y_pre.wav — HTTP 200, equals disk True, SHA256 `559ec2a43e8ad6c694df307023a3e2454a13536c165b29a2cb63e19dbe126c0e`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_off.wav — HTTP 200, equals disk True, SHA256 `368283fb95df9f7e569fde71d11cfc4fc04a967176667a3b14082bb75c70c8bb`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_router.wav — HTTP 200, equals disk True, SHA256 `e841bd82429671c6c7b33768e39d5d3bbcb8884ce989ec6ad9d491e47b4b7c51`.
- http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/listening.html — HTTP 200, disk HTML equality True, SHA256 `209fd9e30ead2a43422b5d696816e0b83031b8df882ab28e2f4ed3a49c18a8f6`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_clean.wav — HTTP 200, equals disk True, SHA256 `10435d243635f7b4135fbca220d6de45d2e501e6cdd0770abd209652ec85204c`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_noisy.wav — HTTP 200, equals disk True, SHA256 `284b279d34a6a76b742fffc41e6cd90a1c96682426c36d166362b3e652b50ad7`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_y_pre.wav — HTTP 200, equals disk True, SHA256 `559ec2a43e8ad6c694df307023a3e2454a13536c165b29a2cb63e19dbe126c0e`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_off.wav — HTTP 200, equals disk True, SHA256 `368283fb95df9f7e569fde71d11cfc4fc04a967176667a3b14082bb75c70c8bb`.
  - http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/clips/03_fileid_10720_14.50-15.00s_context_router.wav — HTTP 200, equals disk True, SHA256 `e841bd82429671c6c7b33768e39d5d3bbcb8884ce989ec6ad9d491e47b4b7c51`.
- http://127.0.0.1:8766/p8_e1_candidate_listening_20261018/listening.html — HTTP 200, disk HTML equality True, SHA256 `31818dccb37c744dc24356b2044325716b5ae3a7ca7713a06f4463ea8098981e`.

Dedicated player default gain is 1; RMS equalization checkbox defaults off. Optional equalization multiplies by clean RMS / arm RMS. Playback windows relative to the 2 s files are [0,2], [0,.55], [.45,1.05], [1.1,2]; browser currentTime and animation-frame stopping are approximate. Page JavaScript was inspected; no browser playback timing was measured.

## Exact full source and clip provenance

| Arm | 30 s full source | Source SHA256 | 2 s clip SHA256 |
|---|---|---|---|
| noisy | `/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/noisy/book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav` | `cb714d176ad32c2aa041dc5a71715349ec9c35ae441bf55b0036f37d5bec173c` | `284b279d34a6a76b742fffc41e6cd90a1c96682426c36d166362b3e652b50ad7` |
| clean | `/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/clean/clean_fileid_10720.wav` | `e2a1d164b64b494c9d99c1d1697360ef5f345c7ad12427d995d9c03b781b6c5e` | `10435d243635f7b4135fbca220d6de45d2e501e6cdd0770abd209652ec85204c` |
| y_pre | `/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020/outputs/y_pre/04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav` | `285a319dfcbea54bdcfce623220ae858ac7d53afd5981db97c7eaddf96725f4d` | `559ec2a43e8ad6c694df307023a3e2454a13536c165b29a2cb63e19dbe126c0e` |
| off | `/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020/outputs/off/04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav` | `912f380d90be61b6b2ad0f8bb4cab166aaf597c5e0423eae26193aa4207d1b03` | `368283fb95df9f7e569fde71d11cfc4fc04a967176667a3b14082bb75c70c8bb` |
| router | `/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020/outputs/router/04_book_04733_chp_0031_reader_10929_13_WIPwL0T99nw-10Ebq7HBVxk-U_5z8BeHx5M_snr-16_fileid_10720.wav` | `78cd59bfa8695624751bff392c71532b40ece20f229e7cf5640e15a806a14cb6` | `e841bd82429671c6c7b33768e39d5d3bbcb8884ce989ec6ad9d491e47b4b7c51` |

All sources and clips: WAV PCM_16, mono, 16000 Hz. Full sources: 480000 samples; current clips: 32000 samples. Current clip basename pattern: `03_fileid_10720_14.50-15.00s_context_{arm}.wav`, under `experiments/p8_e1_candidate_listening_aligned_20261020/clips/`.

## Correlation results

Clip-to-full-source search values (all best starts exactly 13.75 s):

| Clip / source | clean | noisy | y_pre | off | router |
|---|---|---|---|---|---|
| clean | 1.000000 | 0.122873 | 0.502277 | 0.290666 | 0.342417 |
| noisy | 0.122873 | 1.000000 | 0.150193 | 0.183582 | 0.185500 |
| y_pre | 0.502277 | 0.150193 | 1.000000 | 0.500191 | 0.610010 |
| off | 0.290666 | 0.183582 | 0.500191 | 1.000000 | 0.989158 |
| router | 0.342417 | 0.185500 | 0.610010 | 0.989158 | 1.000000 |

| Full pair | Waveform r / lag s | RMS envelope r / lag s |
|---|---|---|
| off_vs_clean | 0.711576 / 0.00000 | 0.786136 / 0.00 |
| router_vs_clean | 0.747928 / 0.00000 | 0.808617 / 0.00 |
| off_vs_y_pre | 0.832899 / 0.00000 | 0.867437 / 0.00 |
| router_vs_y_pre | 0.890324 / 0.00000 | 0.925042 / 0.00 |
| off_vs_router | 0.991584 / 0.00000 | 0.989233 / 0.00 |
| noisy_vs_clean | 0.158584 / 0.00000 | 0.177414 / 11.12 |
| off_vs_noisy | 0.172576 / 0.00000 | 0.228667 / -11.13 |
| router_vs_noisy | 0.171284 / 0.00000 | 0.292668 / -11.13 |

Cross-file test covered all eight P8 fileids, clean/noisy/y_pre inputs against full 10720 off/router, 48 comparisons. Each waveform search used every sample lag retaining >=15 s overlap; 10 ms RMS envelopes used the same overlap rule. Matrix with exact input paths/hashes and all lag values is in metrics.json. Lag convention: output[t+lag] versus input[t].

- Largest absolute waveform correlation with any other fileid for off: 0.033414, input 3321 y_pre (lag -10.37844 s).
- Largest absolute waveform correlation with any other fileid for router: 0.036243, input 3321 y_pre (lag -10.37844 s).

## Historical mismatch details

| Arm | Verified original full-source range s | Exact PCM match | Same as current clip |
|---|---|---|---|
| noisy | [4.75, 6.75] | True | False |
| clean | [13.75, 15.75] | True | True |
| y_pre | [13.75, 15.75] | True | True |
| off | [13.75, 15.75] | True | True |
| router | [13.75, 15.75] | True | True |

All historical routes were HTTP 200 and equal disk bytes; exact URLs and hashes are in historical_metrics.json. Historical filenames: `fileid_10720_14.50_15.00_{arm}.wav`.

## Interpretation limits

Current route, crop, and full-file pairing checks do not explain the user’s reported phonetic difference as a simple file swap or global shift. The local clean/off correlation is only 0.290666 (router 0.342417), versus full-file 0.711576/0.747928; substantial local alteration is measurable. Acoustic model distortion remains a hypothesis to evaluate separately, not an established phoneme error or hallucination. Correlation does not prove word/phoneme identity. This check excludes straightforward mapping errors among these eight P8 inputs and large global offsets retaining >=15 s overlap; it does not prove historical inference tensor routing or exclude arbitrary nonlinear/local timing changes.

Reproduction: use `/data/coding/sgmse/env/bin/python verify.py` and `verify_historical.py` in a fresh new audit directory; scripts refuse overwriting existing result artifacts. Source files are only read. Full source ranges, checksums, headers, statistics, raw matrix, git state, and failure/resolution are persisted.
# 最终更新：两条后端均已精确重放，修复后用户仍报告内容差异

用户在播放器修复后再次反馈“仍然是off和router语音一样，clean/y_pre/noisy语音一样”。该差异不能继续用播放器竞争来解释。

- off重放的完整30s及2s片段均与原输出逐采样完全一致；详见 `replay_off/replay_result.json` 指向的成功记录。
- Router另重放一次，完整480000采样、片段32000采样也逐采样完全一致，WAV SHA256为 `78cd59bfa8695624751bff392c71532b40ece20f229e7cf5640e15a806a14cb6`；见 `replay_router/metrics.json` 与 `replay_router/experiment_report.md`。
- 两次Y_pre都与保存前端输出完全一致；输入与checkpoint均未修改。有效逐文件种子2997368683、PC N30、80k EMA保持相同。
- 这证明当前后端WAV来自正确的10720输入；不证明输出内容正确。当前现象应转入局部内容偏离、拖尾/时间变化或错误生成研究，而非继续假定文件配错。
- 浏览器代码修复有异步模拟7项回归证据，但实际浏览器自动测试因缺库未执行；未把用户的内容反馈冒充完整浏览器行为验证。后续需要人耳判断时直接给可选待办，不再继续安装浏览器依赖。

以下保留之前各层来源检查、旧问题与修复记录。旧段落中只讨论off重放的阶段性状态，以本节和两条replay目录为最新状态。

