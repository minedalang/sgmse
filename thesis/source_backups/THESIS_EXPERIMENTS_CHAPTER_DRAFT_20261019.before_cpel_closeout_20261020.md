# 实验章节草稿：冻结基座路由、内容风险与证据边界

> 本章整合已完成实验与用户反馈，不新增增强、训练、ASR或音频指标。方法对应[Router正文](METHODS_RRSD_IMPLEMENTED_DRAFT_20261019.md)，先例见[相关工作](P8_RELATED_WORK_DRAFT_20261019.md)，机制边界见[P8讨论](P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)。算法新颖性、真实辅音恢复与CAEG有效性不因本章写成而成立。

## 1. 实验问题、数据与独立性

本文分别回答：冻结基座上的Router改变哪些重建／质量指标；现有生成后端的内容代理是否与质量一致；原始输入的局部结构是否可读。三个问题使用不同材料与端点，不能将结果拼成同一个新机制的确认。

| 材料 | 规模与用途 | 数据边界 |
|---|---|---|
| 冻结官方DNS Ins20测试 | no-reverb、with-reverb各150份；同80k checkpoint的off/Router/t-cut配对 | 150个matched file-id的两条件音频，不把pooled300条当300个独立底层内容；不用于新方案/阈值选择 |
| VoiceBank自然内容pilot | 4句/2speaker×2noise×−18/0dB＝16格 | train侧探索；单ASR代理及选中A/B单人反馈，不是最终确认集 |
| VoiceBank候选余量及缓存读出 | 8句/4speaker×2noise×−18/0dB＝32格 | 句子在四条件重复；原增强结果冻结，后续线性拟合LOSO只隔离当前probe，不保证预训练未见 |
| P8-I自然配对及结构听感 | 12句/6speaker，47唯一位置，各Clean/Noisy，共94判断 | 发布者标称0–15dB，非局部实测极低SNR；同人探索，不是独立gold；noise录音身份仍unknown |
| CCC/5243既有诊断 | 受控VCV、单事件Noisy条件消融与人工观察 | 探索性/选择性案例；名义辅音或单识别器代理不等于独立实际发音真值 |

训练历史经用户明确：先训练300k前后端基座，随后冻结前后端，仅训练Router80k。比较同一80k文件的Router动作开关可控制基础权重；不能误写为主干仅训练80k。本文不以另一个300k文件名代替EMA、归一化和采样器一致性核验。

## 2. 冻结Router的客观评价

三臂使用同checkpoint、EMA、PC、reverse-diffusion、N30、corrector none、seed2026及相同输入。正式off为`rrsd=true, scale=0`，不是bypass小网络。t-cut为预先固定的0.05动作屏蔽，不是根据最终测试选择的自适应门。

下表直接转录冻结metrics.json字段，差值均为Router−off。DNSMOS是客观预测器，不是本项目人耳MOS。GPU用于生成；旧评价日志记录DNSMOS请求CUDA失败后实际CPU回退。未在本轮重新测量任何指标。

| 条件 | off / Router / t-cut的PESQ | STOI | SI-SDR(dB) | DNSMOS-SIG | DNSMOS-OVRL |
|---|---|---|---|---|---|
| no-reverb | 3.0350 / 3.0614 / 3.0568 | 0.9752 / 0.9757 / 0.9757 | 18.9885 / 19.3076 / 19.2624 | 3.6352 / 3.6304 / 3.6313 | 3.3946 / 3.3944 / 3.3945 |
| with-reverb | 1.3205 / 1.3058 / 1.3090 | 0.3596 / 0.3575 / 0.3571 | -12.7849 / -12.6211 / -12.7334 | 2.9898 / 2.9649 / 2.9704 | 2.5492 / 2.5205 / 2.5268 |
| pooled | 2.1777 / 2.1836 / 2.1829 | 0.6674 / 0.6666 / 0.6664 | 3.1018 / 3.3432 / 3.2645 | 3.3125 / 3.2977 / 3.3009 | 2.9719 / 2.9574 / 2.9606 |

Router−off配对差与原95%区间：

| 条件 | 指标 | 差值 | 原95%区间 |
|---|---|---:|---|
| no-reverb | PESQ | +0.026383 | [+0.023147, +0.029634] |
| no-reverb | STOI | +0.000522 | [+0.000393, +0.000661] |
| no-reverb | SI-SDR(dB) | +0.319033 | [+0.302189, +0.335560] |
| no-reverb | DNSMOS-SIG | -0.004805 | [-0.007129, -0.002520] |
| no-reverb | DNSMOS-OVRL | -0.000204 | [-0.003241, +0.003024] |
| with-reverb | PESQ | -0.014714 | [-0.036809, +0.003592] |
| with-reverb | STOI | -0.002149 | [-0.003142, -0.001167] |
| with-reverb | SI-SDR(dB) | +0.163791 | [+0.085537, +0.244340] |
| with-reverb | DNSMOS-SIG | -0.024959 | [-0.036654, -0.012990] |
| with-reverb | DNSMOS-OVRL | -0.028741 | [-0.042369, -0.015840] |
| pooled | PESQ | +0.005834 | [-0.005825, +0.015530] |
| pooled | STOI | -0.000813 | [-0.001344, -0.000309] |
| pooled | SI-SDR(dB) | +0.241412 | [+0.202065, +0.282450] |
| pooled | DNSMOS-SIG | -0.014882 | [-0.021161, -0.008582] |
| pooled | DNSMOS-OVRL | -0.014472 | [-0.021656, -0.007403] |

150 matched-file的“混响条件Router差−无混响条件Router差”交互：

| 指标 | 交互差值 | 原95%区间 |
|---|---:|---|
| PESQ | -0.041097 | [-0.063256, -0.022738] |
| STOI | -0.002671 | [-0.003627, -0.001743] |
| SI-SDR(dB) | -0.155243 | [-0.232361, -0.074701] |
| DNSMOS-SIG | -0.020154 | [-0.031802, -0.008179] |
| DNSMOS-OVRL | -0.028537 | [-0.041906, -0.015155] |

无混响条件下PESQ/STOI/SI-SDR有小幅正向变化，同时DNSMOS-SIG下降，OVRL区间跨零；不宜写“全部感知指标改善”。混响条件下SI-SDR仍有正向变化，但STOI和多项质量代理变差。混响PESQ均值负向、区间跨零，不能把单个负均值写成其区间已排除零。

旧配对区间使用utterance重采样、10000次、seed20261018，并非speaker-cluster或多重比较校正的总体结论。原pooled bootstrap未保留matched-file的跨条件配对结构，仅作历史区间展示，不用于据此宣称总体显著性；matched-file交互另按150个配对单位分析。交互说明两条件中的Router效果不同，不单独定位混响失效原因；未训练／测试新的混响模型。

**事后配对不确定性补充**：随后仅从六份冻结逐句CSV，按150个file-id将两条件及所有系统共同重采样（PCG64 seed20261019、10000次、linear百分位）。14个pooled均值最大重现误差5.20×10⁻¹⁸；原指标／方法均不变。关键Router−off新95%区间为PESQ[−0.005429,+0.015219]、SI-SDR[+0.201401,+0.282815]dB、STOI[−0.001339,−0.000313]、SIG[−0.020976,−0.008731]、OVRL[−0.021592,−0.007543]。七指标×两对照的新旧区间全部并列，不按宽度或有利结果挑选；定性跨零方向未变。本补充保留了file-id跨条件关联，仍未解决speaker/noise/room层级依赖、训练seed差异或多重比较，也不是独立确认。[补充报告](experiments/final_matched_file_uncertainty_20261019/experiment_report.md)

t-cut保留同类权衡，没有形成综合修复，本文将其作为固定控制而非有效的第二算法。完整三臂/七指标及区间见[原报告](experiments/final_eval_full_20261018/experiment_report.md)，交互见[原始JSON](experiments/final_eval_reverb_interaction_20261018/metrics.json)。原报告部分均值显示与JSON精确字段有细小不一致，本章表格以JSON逐字段转录，不回写历史报告或修改原指标。

## 3. 低SNR内容代理及候选余量

32格中Noisy/frontend/OFF的微WER分别64.24%/131.25%/79.51%；参考驱动的逐格二选一oracle为55.21%，相对最佳固定臂余量24.31个百分点。8句共72个参考词位置（token，不是72种不同词形），在四条件重复后计288，并非288个独立内容样本。

前端词错误更少的条件有23格，后端4格，5格相等。一个6词句的前端被ASR重复插入216词，不能推断音频真的多说216词。后验分解显示70个oracle错误差中62来自speaker间固定策略异质性，8来自speaker内进一步选择；speaker-best仍用参考，不是部署策略。0dB余量仅2.08pp；−18dB oracle微WER仍97.92%。这些结果约束候选池与评价叙事，不足以启动排序器并宣布真实内容恢复。

同32格的OFF−frontend质量变化为SI-SDR−1.3261dB、STOI+0.009965、PESQ+0.062594，32对均有效；−18dB的PESQ变化为−0.013394，0dB为+0.138581。质量代理与单ASR内容代理不能互相替代。[原指标与literal](experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json) · [混杂解释](research_memos/p8_prospective_headroom_interpretation_20261019.md)

较早16格pilot的微WER为66.03%/59.62%/80.77%，不能跨不同句子和两组总体均值进行机制归因。用户A例报告后端音节质量更好、未听到额外尾音；B例报告前端丢音与后端异常开头音节。它们是有参考的单人后验观察，不是总体发生率。[反馈原话](experiments/p8_voicebank_native_content_pilot_20261019/attempt02/listening/user_feedback_20261019.json)

## 4. 条件干预、结构读出与人工参照

### 4.1 条件敏感性不等于正确证据利用

VCV五例移除了直接Noisy条件中的已知语音，但原含语音前端锚点仍在；不是整个后端只有纯噪声。五例单人偏好为原后端1、相似4；其中相似不等于正确，另有/t/→/k/与/s/样尾音观察。条件冲突/OOD、非局部效应和残噪解释未排除。[原记录](experiments/p8_vcv_speech_component_ablation_20261019/listener_assessment_20261019.md)

5243目标移除后CTC `/d/`后验升高，但用户报告声音稍弱；身份、模型概率及可听强度不是同一量。不能据此宣布Noisy必要性或当前系统从不补音。[最新修正](experiments/p8_5243_local_condition_necessity_20261019/listener_assessment_20261019.md)

### 4.2 原始Noisy结构观察

单听者94判断已完成：Noisy41结构/2非语音/4unknown，Clean45结构/0非语音/2unknown。按唯一位置配对，Clean判结构的45处对应Noisy40结构、1非语音、4unknown。用户补充部分unknown是80ms目标过快、未反应过来；因此不能把unknown全部解释为无声学信息。其“肯定回答约95%正确”是自信估计，不作为实测准确率。

Clean没有明确非语音类别，位置control不能充当negative，无法估计假阳性或真假事件双向判别。Noisy先于Clean并不消除同人记忆/研究期待偏差，且逐trial UI版本不可追溯。原答案全保留，不按后续结论重选窗或筛答卷，也不要求默认重听。[完整配对与风险](experiments/p8_i1b_single_listener_materials_20261019/responses/completed_interpretation_20261019.md)

### 4.3 固定线性结构读出

复用VB32的32Noisy、32frontend、8scaled-clean；320样本对称Hann、hop160、FFT512，保留11396完整帧，尾余数报告。只预测clean高频能量fraction与归一化正向高频谱flux两个连续目标；不是音素/有无事件金标。119维full与6维simple的观测特征采用4折LOSO、固定Ridge lambda0.01、case等权且case内帧等权，训练标准化不看留出speaker。

| 输入／读出 | 高频占比MSE | 高频flux MSE |
|---|---:|---:|
| full_noisy | 0.03138759 | 0.00401377 |
| simple_noisy | 0.03794396 | 0.00440901 |
| full_frontend | 0.03463488 | 0.00413680 |
| simple_frontend | 0.02751642 | 0.00341552 |
| identity_noisy | 0.14564364 | 0.01583160 |
| identity_frontend | 0.03206909 | 0.00492853 |
| train_mean_constant | 0.03968649 | 0.00462343 |

full-Noisy比simple-Noisy两项MSE下降17.28%/8.96%，4speaker同向；但simple-frontend全量更低，full-frontend反而在四位speaker的两目标均逊simple。full-Noisy对full-frontend的speaker方向分别为3/4和2/4，不支持稳定跨表示优势。高维表示较差不能定位丢失的真实内容；前端曾使用整句上下文，比较也不是纯信息论损失实验。

主运行16次双输出线性拟合与独立16次增广最小二乘复算通过；两者共享NumPy FFT、IO范围说明在报告。拟合明确记为probe学习，不能写零训练；0新增强网络/ASR/音频，预测越界保留不clip。结果不触发调lambda、扩hidden probe/verifier或CAEG。[完整报告](experiments/p8_cached_structure_readability_20261019/experiment_report.md)

## 5. 资源与可复现边界

Router为42577参数，相对约5512万非Router参数增量0.07725%；参数少不意味着延时零。普通N30/none-corrector路径静态Score评估30次，路由再加30次小Router。旧正式off(scale0)也执行Router，三臂job耗时958/956/957秒不能证明Router更快；t-cut计算后屏蔽动作，不节省小网络调用。future-grad训练额外一次2B Score重评分与输入梯度另计。[成本细节](THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md#24-复杂度与计时口径区分训练lookahead和推理路由)

当前VoiceBank32全句A/B内容材料只复用104源文件，没有新模型输出。后续已收到单人12例参照辅助回答（6句/4speaker、−18dB8例/0dB4例），其余20例未评且不要求补齐；不算人类WER或正式12独立听者盲转写。frontend checkbox为missing6/preserved4/unknown2；OFF为missing4/different3/added1/preserved4/unknown1，标签多选不能当错误音素数。双方preserved对应同四个0dB案例，未出现frontend missing而OFF preserved的明确组合；missing较少不能解释为正确恢复，因为可能转成不同内容或新增。质量OFF3/frontend1/相似5/unknown3，OFF偏好均在内容双方保留的案例；内容和质量不相互抵消。

首例`p227_031_N2_snr_m18`的用户备注“A在句首The之前就生成了幻觉音素，B没有”，冻结A=OFF；原checkbox两臂都为missing。原备注和选项一并保留，不静默补added，也不将“幻觉音素”措辞当作精确身份的独立裁决。部分、同人、参照辅助及重复句子限制仍在，不外推总体胜率或用20个未评案例补分母。[材料报告](experiments/p8_vb_cached_content_review_20261019/experiment_report.md) · [原反馈与解释](experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md)

本章没有新增Speaker Similarity/EER、峰值显存、部署RTF或端到端真实音素恢复数字。旧研究存在相关试验不等于本批已覆盖这些终点；不得用其他数据/不同checkpoint的数字填空。没有新算法确认、独立身份与双向收益前，不把P8-I、CAEG或一般结构可读性写成完成贡献。所有原数据、checkpoint、冻结测试集与原始指标保持不变。

## 6. 结论与尚欠证据

现有证据支持的是：已实现Router在相匹配基座上的条件性重建／质量权衡；选定材料中的内容风险、条件敏感性与有限结构可读性。它们不共同推出“前端损失的真实弱辅音已由新机制恢复”。

后续若有可定位的内容改善与损伤观察，将与质量偏好分开报告；若没有收益或不可评价，如实收口该候选池，不用调ASR、换seed或扩分类器救结果。实际方法创新还需独立的内容／身份、错误新增控制、质量保真及计算预算证据。长期goal未完成，论文段落是交付，不是机制成功的替代。

## 本次写作来源校验

只读既有JSON字段生成表格，未重算音频指标／bootstrap或拟合probe。表中均值差与matched交互已作字段算术一致性核对；这不替代历史模型/数据复现。

- `experiments/final_eval_full_20261018/metrics.json` SHA256 `15cf7076a0e233042e83e68ff429ff0b949e83ebcaf88297d1448d10b363daaa`
- `experiments/final_eval_reverb_interaction_20261018/metrics.json` SHA256 `f32d9834ad56e299d1659d781e0fac10bb43a8056945fe257d12c45de44abc8d`
- `experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json` SHA256 `577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04`
- `experiments/p8_cached_structure_readability_20261019/metrics.json` SHA256 `235a7b2ac8f15131d25d1f061319f2d8e1c815d00d53ad08e311192e7c67e913`
