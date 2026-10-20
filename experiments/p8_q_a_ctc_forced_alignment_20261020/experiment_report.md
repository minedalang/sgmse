# P8-Q-A clean-derived CTC forced-alignment pilot（2026-10-18）

## 目的和边界

这是一个低成本声学模型探针，不是内容真值实验。它复用已有的 4 组 × 5 条、每条 2 秒对齐片段，使用本地缓存的 `facebook/wav2vec2-lv-60-espeak-cv-ft`，从每组 clean 音频生成高置信 CTC 贪心 token 序列，然后把这个序列固定，对 noisy、`Y_pre`、Router-off 和 Router 做 CTC Viterbi 强制对齐。

本机没有 DNS 原始 clean 文本；clean CTC 序列是模型伪标签，不能称为真实音素、转写、WER/CER 或人类可懂度。没有改数据集或 checkpoint，没有下载新模型，也没有训练。

完整的逐 token 置信度、帧边界和各条件固定目标 token posterior 在 `token_alignments.csv` 与 `metrics.json`；条件级汇总在 `alignments.csv`。

## 方法

1. 对每组 clean 做 greedy CTC argmax，并按连续非 blank run 合并。
2. 仅保留 run 平均 posterior `>= 0.50` 的 token 作为该组固定 pseudo-target；被过滤的低置信 run 也记录在 `metrics.json`。
3. 对每个条件使用标准 CTC 扩展状态 `[blank, token_1, blank, ..., token_K, blank]` 做最大对数概率 Viterbi 路径。每个 token 记录起止帧、起止秒、强制路径上的目标 token posterior 均值/最小值/最大值和 argmax 命中率。
4. 同时记录每个条件的 greedy blank fraction、blank 平均 posterior、强制路径 blank fraction，以及相对 clean 的 blank 变化。

## Clean 固定目标序列

| group | 保留 token 数 | 丢弃低置信 run | clean-derived token sequence（按时间顺序） |
|---|---:|---:|---|
| 00 / fileid 5243 | 19 | 1 | `s ɛ l f w ʌ z k ə n eɪ d i ə n b aɪ b θ` |
| 01 / fileid 4812 | 28 | 3 | `æ p t m aɪ ɹ iː k ɑː n s t ɪ t uː ʃ ə n ʌ v ð f ɹ ɛ n ʃ ɪ p` |
| 02 / fileid 11999 | 28 | 0 | `iː v ə n ɪ n ð ə m oʊ s t p ɜː f ɪ k t k ə m j uː n ɪ ɾ i ð` |
| 03 / fileid 10720 | 14 | 0 | `n d iː d s ɜː t ə n l i w ɪ l` |

每个 token 的 clean run posterior 和 clean greedy起止帧/秒在 `metrics.json → groups_detail[].clean_target_tokens`；同一 token 在五个条件的强制边界和 posterior 在 `token_alignments.csv`。

## 条件级结果

| 条件 | clean-target forced posterior 均值 | fixed-target argmax 命中率 | greedy blank fraction | blank posterior 均值 | 强制对齐失败 |
|---|---:|---:|---:|---:|---:|
| clean | 0.8936 | 1.0000 | 0.7374 | 0.7336 | 0/4 |
| noisy | 0.0267 | 0.0179 | 0.9823 | 0.8971 | 0/4 |
| `Y_pre` | 0.2220 | 0.2871 | 0.8131 | 0.7795 | 0/4 |
| Router-off | 0.2662 | 0.2770 | 0.7449 | 0.7305 | 0/4 |
| Router | 0.2750 | 0.2839 | 0.7500 | 0.7302 | 0/4 |

所有 20 个条件片段都能在动态规划意义下完成路径（无法对齐比例 `0/20 = 0%`）。这只说明目标序列在 2 秒 CTC 帧上存在一条可计算路径；它不表示目标 token 在该条件中被正确听到。

相对 `Y_pre`，Router-off/Router 的固定目标 posterior 均值在这 4 组上分别高约 `+0.0442/+0.0530`；Router 相对 Router-off 约 `+0.0087`。这些差异属于同一声学模型对 clean-derived pseudo-target 的响应，不能解释为恢复了正确音素。noisy 的 blank 比例和 blank posterior 都明显更高，表明该模型在原始极低 SNR 片段上大部分帧偏向 blank；后端条件使 blank 减少，但仍不能区分正确 token、近音或声学自然化。

## 时间边界审计补充

固定目标并不等于固定时间位置。对每个条件重新求得的 Viterbi 路径与 clean 路径的 token 中点绝对漂移为：noisy 中位 `120 ms`、90 分位 `652 ms`，仅 `38.2%` 位于 `±40 ms`；`Y_pre`、Router-off 和 Router 的中位数均为 `20 ms`，位于 `±40 ms` 的比例分别为 `66.3%`、`57.3%` 和 `57.3%`。辅音 token 的 noisy 中位漂移为 `140 ms`。因此固定目标 posterior 只能作为整段序列的探索性声学响应，不能作为同一 clean 时间点的局部音素证据。完整边界表见 `experiments/p8_q_a_ctc_boundary_audit_20261020/`；所有逐 token 的 E1/“正确恢复”解释降级为 unknown。

## 证据等级和限制

- 这是 clean-derived pseudo-label probe，4 组样本极小，未使用真实文本或人类转写。
- CTC 强制路径会在低 posterior 帧上“找一条最不差的路径”；`alignment_success` 不是内容正确性。
- blank 变化和目标 token posterior 不能单独证明删除、幻觉或恢复；CTC 时间边界也可能因增强而移动。
- 因此结果只能记录为：在此代理模型上，后端输出对 clean 伪目标的帧后验通常高于 `Y_pre`，但没有证据证明 Router 相比 Router-off 带来正确内容恢复。
- 不据此启动 content loss、Harm Head、P8-N 或 P8-R 训练；下一步若要形成内容结论，仍需真实文本或盲转写和独立样本。

## 复现

```bash
/data/coding/sgmse/env/bin/python \
  experiments/p8_q_a_ctc_forced_alignment_20261020/forced_alignment_probe.py
```

脚本使用 `local_files_only=True` 的现有权重，输出仅写本实验目录。
