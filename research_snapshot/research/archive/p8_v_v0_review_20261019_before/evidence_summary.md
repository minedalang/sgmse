# 关键证据摘要：已知／未知／停止

> 只保留当前研究判断需要的数值和出处，不记录重复运行日志、完整哈希、后台进程与失败修复流水。**这是一份对用户上传计划的整理，不是对远端实验仓库的重新核验。** 要复核请按路径打开原始 `metrics.json`、`experiment_report.md`、用户反馈 JSON 或 [原文归档](archive/research_plan_full_20261019_044238.txt)。

## 1. 最新 VoiceBank 自然匹配内容 pilot（优先看人工反馈）

**来源**：`experiments/p8_voicebank_native_content_pilot_20261019/attempt02/`；4 句（p226、p287 各两句）× 两个历史噪声 × 全句名义 `-18/0 dB`，共 16 格。16 份 Frontend-only 与 16 份 Router-off，PC `N=30`；四句作者文本与缩放 clean 的小型 Whisper 校准结果为 0 词错误。但这不等于其他噪声条件下的 ASR 或人工识别无误。

| 量化代理 | Noisy | `Y_pre` | Router-off |
|---|---:|---:|---:|
| 全 16 格相对发布者原文微 WER | 66.03% | 59.62% | 80.77% |
| -18 dB 子集 | 100.00% | 96.15% | 129.49% |
| 0 dB 子集 | 32.05% | 23.08% | 32.05% |

- Router-off 相比前端编辑错误增量为 `+33`，其中识别器假设的 `I +28`、`D 30→10`。**不能把 I 直接叫真实新增词，或将 D 下降称为真实恢复**；低 SNR ASR 已接近失效。
- 四句仅两位说话人，重复使用同句及噪声；16 格不是 16 个独立自然句子，156 个评分词包含原 39 词在多条件重复。

**人耳 A：`p287_054/N1/0dB`**：用户认为前端和后端句末都无额外语音，音节差不多；前端高频较少，**后端音节质量好很多**。因此 A 不支持“由 ASR 末段差异推断实际听感内容损伤”。不能把整个 0 dB 子集的 ASR 相同 WER 用作人耳无差异结论。

**人耳 B：`p226_028/N2/-18dB`**：用户认为 `Y_pre` 丢掉很多音节，Router-off 有很多奇怪音节；开头 Clean/`Y_pre` 听起来无语音，off 有难懂音节。它是**人工观察到的风险候选**，但没有独立逐词身份或精确人工时间戳，不能立刻称“噪声被识别成特定词”。

**B 只读 CPU 开头复核已完成并关闭**：`experiments/p8_vb_caseB_opening_existing_audit_20261019/`；既有音频、实际 N2、raw/PCM/播放源匹配。前 0.5s 的 clean/noisy/`Y_pre`/off RMS 为约 `0.00067555 / 0.03340213 / 0.00043358 / 0.00308541`，off / `Y_pre` 为 `7.116×`、`+17.04 dB`；五个 100ms 格都增能，全句 off vs 前端仅 `+0.981 dB`。clean 前 0.5s **并非精确零波形**，听觉“无语音”不能替代逐样本全零判定。该复核不能区分人声生成、残噪、非局部前响或真伪音节；停止围绕 B 的新窗口、lag、频带、seed 与新 ASR 搜索。

幅度/字数的混杂复核：-18 dB 下后端 RMS 相对前端中位 `+1.895 dB`，ASR 字数 `55→103`；0 dB RMS 中位 `-0.008 dB`、字数 `77→77`，但编辑错 `18→25`。不能把 ASR 差异归结为单纯响度，也不能排除其影响。记录：`research_memos/p8_voicebank_native_pilot_interpretation_20261019.md`、`p8_native_gain_length_summary_20261019.*` 与 B 专项 memo。

## 2. DNS 与真实辅音信息的证据状态

| 问题 | 当前证据 | 正确边界 |
|---|---|---|
| 训练低 SNR 覆盖 | `tr_synthetic`/`cv_synthetic` 10,800/1,200 条，90/10 小时；文件名 SNR -20…-6 dB | 真实波形/局部弱辅音 SNR 分布未知；CV 有一位 speaker 跨 split |
| 有无真实句子文本 | 本地 DNS SCP `text=dummy`；原 clean 配对不是带原文的独立语料 | 不能以 Whisper clean 转写、音素 CTC 或伪边界算真实 WER/正确音素恢复 |
| `Y_pre` 抑制窗 | 181 个探索窗中 55 个抑制窗；局部 SNR 中位约 -20.41 dB，非抑制约 -8.30 dB | clean-aware 统计，只用于诊断；不能当部署时条件或证明输入无信息 |
| 10/20ms glimpse | 55 个抑制窗中约 24/19 个含至少一帧 clean-aware 局部 SNR 达 -10dB | 不绑定真实辅音身份，不能等同 `/k/` 或 `/d/` 可辨 |
| 固定时间 CTC | 89 个 clean-derived 伪目标中，noisy/`Y_pre`/off/Router 平均后验约 `.0176/.1168/.1018/.1080` | 时间漂移显著，单模型 posterior 不是真实音素身份，整体正负方向不稳定 |

数据和 CTC 记录：`experiments/p8_training_coverage_audit_20261020/`、`p8_text_source_audit_20261020/`、`p8_e1_candidate_ranking_20261018/`、`p8_q_a_shorttime_probe_20261020/`、`p8_q_a_ctc_fixed_time_v2_20261018/`。

## 3. 外部 VCV 与 Noisy 条件归因

**已有较强的操作性事实**：80k Score 的 Noisy 条件主分支确实生效，不是被完全忽略。但“直接 Noisy 起作用”不等于“解码出正确辅音”。

- 外部真实录音 VCV 主 clean 子集：四位说话人、8 对 k/g、t/d 共 16 条，冻结 wav2vec2 三 token 评分达到标称身份 `16/16`；另 12 条补充仅 `7/12`，不能概括全部语料。
- 固定噪声的名义 -6 dB 2s carrier 下，旧四位 speaker 的 N1 24 类 noisy/`Y_pre`/off/Router 为 `8/12/13/15`（分母16）；但语音实际支撑区平均 SNR 约 `+0.867 dB`，非极低局部辅音 SNR。
- 新四位 speaker 复验未复制 Router 的原有净收益；N1 -6 dB 前端/off/Router 为 `14/15/14`，还出现新的类别错误。**已按预注册停止**“Router 稳定提升辅音身份”的扩大叙事。
- g/k 条件互换的统一尺度 2×2：输出识别器的 anchor 主效应 LL margin 大致 `+7.8` 到 `+9.7`，直接 Noisy 条件主效应约 `-0.28` 到 `+0.54` 且方向不一；条件冲突仍 OOD，不能说模型“只信锚点”或 Noisy“没有信息”。
- 五例直接 Noisy 语音分量消融中，原先一些身份修正/损伤发生改变，说明 Noisy 条件敏感；但移除整段 VCV 包含强分布改变，不是只去掉某一个辅音。

路径：`experiments/p8_q_vcv_clean_calibration_20261018/`、`p8_q_vcv_lowsnr_challenge_20261018/`、`p8_q_vcv_speaker_replication_20261019/`、`p8_vcv_common_scale_factorial_20261019/`、`p8_vcv_speech_component_ablation_20261019/`。

**第五例尾音**：Clean 在 `0.9300625s` 后精确为零；直接 Noisy 条件移除语音分量后，4–7.5kHz 尾部能量比原后端约 `31.24/33.59×`。缓存识别器 `/s/` posterior 某帧约 `.672`，用户表述是高频噪声/摩擦样伪影。不是“模型确实凭空说了 `/s/`”的正式身份真值。`experiments/p8_vcv_f20_tail_existing_audit_20261019/`、`p8_vcv_f20_noise_correspondence_20261019/`。

## 4. 5243 `/d/` 的重要反向证据

`experiments/p8_5243_local_condition_necessity_20261019/`：后端原条件对目标 `/d/` 的 CTC P(d) 约 `.869617`；200ms 目标 Noisy 语音分量移除后反而 `.979222`；前、后位置控制 `.614505/.959156`。预注册基于 CTC 后验的局部必要性方向**没有得到支持**。用户的修正听感是“移除目标版 `/d/` 相比原版稍弱”，两者的声音强度与后验相反；并未完成“哪个音素身份更正确/更像 Clean”的独立判断。不得为证明 P8-V 或过抑制机制而重算同一例阈值或增加 seed。

## 5. 当前负结果、独立性与论文边界

- P8-M `Y_pre` 遮蔽：多 seed 下 SI-SDR/STOI 变差，不是 Noisy 成功独立恢复的证据；停止该 OOD smoke。
- P7-C `Y_pre`/Noisy 简单特征预测局部 NRMSE：speaker grouped OOF Ridge MAE `0.1050`，均值基线 `0.0975`；普通局部门控尚无稳定部署依据。
- P7/感知/身份/多轨迹/adapter 较多方案可改善某些 proxy，却没有同时过真实内容、质量、泛化和算力门；状态见 [archive.md](archive.md)。
- 官方 DNS `test_set/synthetic/no_reverb` 与 `with_reverb` 各 150 条已经用于最终评测且冻结；不得用结果继续调参。对新算法应另备独立、未反复用于筛选的确认集。
- P8-V 当前只是可证伪的新候选，不得把现有 Noisy 消融、人工尾声、VoiceBank WER 或文件重组当作验证成功。

## 6. 状态优先级与溯源规则

旧计划里的“准备中”“待执行”若已被后面的**完成/终止**覆盖，以对应最新可复核实验报告为准；旧段落保留历史但不能重新授权动作。音频听感原话和自动代理数值同时保存、互不覆盖。所有新结果先更新本表及实验报告，再将人类可理解的状态写到 [短总纲](../research_plan.md)。
