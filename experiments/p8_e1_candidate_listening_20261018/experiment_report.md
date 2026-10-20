# P8-E1 候选窗口听感包

## ⚠️ 事后发现：本包时间轴错误，结果作废

旧版构建脚本对完整 30 秒 `noisy` 和已经裁成原始 9–21 秒的中央 12 秒 enhanced/clean 文件使用了同一个索引。以 file-id 5243 为例，`noisy` clip 实际为原始约 9.75–11.75 s，而其它 arm 为原始约 18.75–20.75 s。因此五个 arm 不在同一时间段；本目录中的试听和任何基于 noisy 的内容判断均不得使用。修复版见 `../p8_e1_aligned_listening_20261020/`。

## 目的

从已有 P8 输出中截取少量候选窗口，便于人工判断“noisy 是否仍有可辨认线索、Y_pre 是否吞音、后端是否只做声学平滑”。本目录不运行模型，不修改原始数据或 checkpoint。

## 构建方式

- 候选来自 `../p8_e1_candidate_ranking_20261018/candidate_windows.csv`。
- 每个 file-id 选择一个排名靠前的窗口，共 4 位 speaker。
- 使用已有中央 12 秒 `clean/Y_pre/off/router` 听感文件，以及已有 30 秒 `noisy` 文件。
- 各窗口按原始时间截取上下各 0.75 秒上下文，得到 2 秒、16 kHz PCM16 片段。
- 每个候选包含 `clean/noisy/y_pre/off/router` 五个 arm，共 20 个 WAV。

## 完整性检查

20 个片段均可读取，采样率为 16 kHz，长度为 2.0 秒。`manifest.json` 保存原始时间窗、中央 12 秒映射时间和代理排序字段。

## 解释边界

clean 文本提示来自 Whisper `tiny.en` 自动转写，不是 ground truth。该听感包不能产生 WER/CER、音素恢复率或 E1 成立证据；它只用于优先人工核实少量候选。
