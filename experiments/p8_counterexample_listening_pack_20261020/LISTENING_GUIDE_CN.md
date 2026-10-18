# P8 反例候选定点听感包

这是一个**可选**的定点核查包，不需要逐条评分。它只挑出 4 个 noisy 局部 SNR 相对较高、但 `Y_pre` 仍明显压低的窗口，用来寻找“输入还有线索、前端却吞音”的反例。

每个编号有 5 个文件：`clean / noisy / y_pre / off / router`。建议顺序是先听 clean，再听 noisy、`Y_pre`、off、Router。只需判断一个问题：**clean 中的主要音节在 noisy 中是否还有可辨认线索，而 `Y_pre` 是否把它压没？** 不需要填写表格。

## 候选窗口

| 编号 | file-id | 原始时间 | 局部 SNR（clean-aware） | noisy-clean 相关 | `Y_pre` RMS / clean |
|---|---:|---:|---:|---:|---:|
| 00 | 5243 | 19.50–20.00 s | −8.03 dB | 0.321 | −18.49 dB |
| 01 | 4812 | 12.50–13.00 s | −9.21 dB | 0.328 | −10.00 dB |
| 02 | 4812 | 15.00–15.50 s | −9.29 dB | 0.284 | −10.25 dB |
| 03 | 4812 | 20.50–20.97 s | −9.77 dB | 0.332 | −21.34 dB |

文件目录：`clips/`。

## Clean 自动转写上下文（只作听音提示）

自动转写来自 clean 的完整/中央 12 秒识别，不是人工真值，也不能保证与 0.5 秒边界精确对齐。

- **00 / 5243**：中央片段大致包含 `...whom she loved with an all-absorbing passion, had caught the war fever, gone to Canada and enlisted. Mrs. McGuire herself was Canadian by birth...`
- **01、02、03 / 4812**：中央片段大致包含 `...Have you told me all you know about them? Far from it. Although I was handicapped in my reconstitution of their friendship by lack of the Archbishop's letters... Throughout the letters...`

如果这 4 个候选仍然听不出 noisy 中有明确音节线索，就更支持“信息受限”解释；如果某个候选中 noisy 明显保留了音节而 `Y_pre` 删除，才值得单独研究前端内容保护。

该包不改变数据集或 checkpoint，也不用于调参。
