# P8-E1 原候选时间轴修复听感包

## 重要更正

旧目录 `p8_e1_candidate_listening_20261018` 已作废：其脚本把完整 30 秒 noisy 按中央 12 秒坐标裁剪，导致 5243 的 noisy 实际为约 9.75–11.75 s，而 clean/Y_pre/Router-off/Router 为约 18.75–20.75 s。

本目录保留原来的四个候选，但五个 arm 均直接从完整 30 秒文件按同一原始时间坐标裁剪，因此可用于重新试听。`metrics.json` 记录了源路径、哈希、样本数和 ±100 ms 对齐扫描；所有 arm 的最佳滞后应为 0 ms。

## 试听

服务器已可从 `http://127.0.0.1:8766/p8_e1_candidate_listening_aligned_20261020/listening.html` 访问；SSH 端口转发：`ssh -L 8766:127.0.0.1:8766 <服务器地址>`。

每组先听 clean，再听 noisy、Y_pre、Router-off、Router。只需判断目标音节是否还在 noisy 中、Y_pre 是否吞掉、后端是否补回同一个音节；不需要填写数值评分。
