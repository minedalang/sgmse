# file-id 11999 盲化强迫选择包（P8-E1 辅助核实）

## 目的

这个包只用于人工核实 file-id 11999 的候选窗口，不能单独证明 E1、音素恢复或模型幻觉。五个窗口来自已有 clean-aware 候选排序；音素模型结果只复用已有 `phoneme_results.json`，没有为本包重新推理。

## 使用方法

1. 用浏览器打开 `index.html`。远程 SSH 可在服务器上执行 `python3 -m http.server 8000 --directory /data/coding/sgmse/experiments/p8_11999_forced_choice_pack_20261020`，再做 SSH 端口转发。
2. 每个试次先听参考，再交替听 A/B。只判断目标词句/音素与参考是否一致，不按响度、噪声少或音色平滑选择。
3. A/B 条件和顺序在页面中隐藏并已固定随机化。每个试次必须二选一；听不出差别时仍选择一个，并在备注写“差不多”。
4. 点击页面底部下载 `responses.csv`，或直接填写目录中的模板。

## 试次结构

- 每个窗口有三个比较：`noisy vs Y_pre`、`Y_pre vs Router-off`、`Router-off vs Router`。
- 参考为同一原始时间窗的 clean 片段；A/B 是同窗候选，所有音频均为 16 kHz。
- 目标是内容/音素一致性；这不是音质偏好测试。

## E1 通过条件（预先固定）

至少 3/5 个窗口同时满足以下链条，才允许把结果作为 E1 候选证据：

1. 听者能在 noisy 中相对 clean 参考识别同一目标辅音或词片段；
2. Y_pre 被判断为丢失或明显压低该片段；
3. Router-off 或 Router 被判断为恢复同一片段，而不是只让残余语音更圆润。

CSV 的三个附加列用于记录这三项判断。若 noisy 目标片段听不出、只发生整体清晰度变化、或只凭一个 ASR 代理判断，窗口不算 E1。当前尚无人工结果，不能提前宣称通过。

## 选窗依据

五个窗口是既有 `p8_e1_candidate_ranking_20261018/candidate_windows.csv` 中 file-id 11999 且 `candidate_frontend_suppression=true` 的前五名，未在本包中重新排序或使用人工结果调阈值。窗口如下：

| 窗口 | 时间(s) | 原候选 rank | noisy-clean corr | local SNR (dB) | Y_pre RMS vs clean (dB) | off gain vs Y_pre (dB) |
|---|---:|---:|---:|---:|---:|---:|
| W01 | 19.50–20.00 | 7 | 0.219 | -16.95 | -11.59 | 8.75 |
| W02 | 13.50–14.00 | 15 | 0.151 | -19.51 | -22.86 | 2.98 |
| W03 | 10.50–11.00 | 20 | 0.120 | -19.52 | -21.29 | 12.62 |
| W04 | 11.50–12.00 | 24 | 0.111 | -21.31 | -12.56 | 10.69 |
| W05 | 18.50–19.00 | 34 | 0.082 | -22.90 | -28.79 | -0.50 |

已有音素结果只覆盖 W01（19.50–20.00 s）的 clean/noisy/Y_pre/off/Router 五个片段，保存在 `phoneme_auxiliary_11999.json`。它是 CTC 声学模型代理，不是真实音素标签；其他四个窗口没有在本任务中运行音素推理。

## 输出文件

- `index.html`：盲化播放和 A/B 表单。
- `responses.csv`：空白记录模板。
- `answer_key.json`：A/B 条件映射，仅供审计。
- `metrics.json`：固定选择、源文件哈希、随机种子和验证状态。
- `experiment_report.md`：实验边界和当前待人工结果。
- `phoneme_auxiliary_11999.json`：复用的已有音素结果子集。

未修改 DNS 数据集、原始 P8 WAV 或 checkpoint；没有训练和新推理。
