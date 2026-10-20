# 旧试听包时间轴错误

本目录保留作审计记录，**不是有效试听材料**。

旧版 `make_pack.py` 对已裁剪到原始 9–21 秒的 clean/Y_pre/Router-off/Router 和完整 30 秒 noisy 使用了同一个裁剪索引。以 file-id 5243 为例：

- clean/Y_pre/Router-off/Router 源文件已经是原始 9–21 秒，索引 9.75–11.75 秒对应原始约 18.75–20.75 秒；
- noisy 源文件是完整 0–30 秒，索引 9.75–11.75 秒对应原始约 9.75–11.75 秒。

因此旧包五个 arm 不在同一时间段。旧页面和旧 clips 的内容比较全部作废。修复包位于 `../p8_e1_aligned_listening_20261020/`，该包直接从五个 30 秒源文件按同一原始时间坐标裁剪，并保存了 `alignment_scan.json`。
