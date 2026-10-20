# P8 论文讨论草稿：已观察现象与证据局限

配套写作：[相关工作与研究定位](P8_RELATED_WORK_DRAFT_20261019.md) · [固定版本参考文献BibTeX](P8_RELATED_WORK_REFERENCES_20261019.bib)。

> 状态：既有声学证据的写作整合，§1.6另引用随后完成的静态checkpoint比较；**不是新增强方法有效性或新颖性结论**。本次读取原始指标、已有独立验证回执和人工反馈；没有重新计算音频指标、读取波形、运行模型或收集新判断。研究入口仍为 [/data/research_plan.md](../../research_plan.md)。较早的 [Router论文草稿](THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md) 与本文件分开：其Router-on/off比较不能代替本文件的frontend/native-OFF比较。

## 1. 可用于“问题分析与讨论”的正文

### 1.1 音质改善与内容保真应分别评价

在8句、4位说话人、两种噪声及−18/0 dB组成的32条件探索中，原生扩散后端相对判别式前端的配对平均变化为：SI-SDR −1.326 dB、STOI +0.009965、PESQ +0.062594，三项均有32个有效配对。这些结果呈现重建与质量代理之间的权衡，而非全面改善。分层后，−18 dB的PESQ变化为−0.013394，0 dB为+0.138581，也不能用全量均值宣布极低SNR音质提升。[原始指标](experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json) · [报告与验证范围](experiments/p8_vb_prospective_headroom_20261019/model_stage/experiment_report.md)

已有人工观察同样包含正反两面：在较早16条件pilot的A例中，用户认为前后端音节相近、后端音节质量明显更好，未听到额外尾部语音；在B例中，用户报告前端丢失音节，以及后端开头出现异常、难理解的音节。这是同一位听者对有参考、后验选中案例的观察，不是盲听总体结果；“噪声被当成语音”仍是待验证的机制解释。故质量提升、异常结构、具体音素正确性不能合并为一个指标。[人工反馈原话](experiments/p8_voicebank_native_content_pilot_20261019/attempt02/listening/user_feedback_20261019.json)

### 1.2 ASR候选余量不等于已获得真实内容恢复

32条件下，单一识别器给出的微WER分别为：Noisy 64.24%、frontend 131.25%、native OFF 79.51%；根据参考文本逐条件选择前端或后端的oracle为55.21%，相对最佳全局固定臂的差为24.31个百分点。这个oracle使用评价参考，不是可部署选择器，也不是新增强方法。288个评分参考词来自72个原始词在四条件下重复；样本仅4位说话人，bootstrap不能弥补其独立性与评价有效性限制。[逐条件与分层指标](experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json)

总体数值还受到强烈混杂：前端在23条件的词错误更少，后端仅4条件更少，另5条件相同；一个6词句子的前端转写产生216个重复性插入错误，不能解释成音频真的多说了216个词。后验分解显示，70个oracle错误差中，62来自按说话人选择固定臂的异质性，8来自说话人内进一步选择。0 dB的相同代理余量仅2.08个百分点，−18 dB的oracle WER仍为97.92%。因此，这组结果不支持直接训练候选排序器并声称恢复了真实弱辅音，也不证明原始Noisy在信息论上不可辨识。异常值及原评分均保留，未通过删例或改解码参数修饰结果。[混杂解释与停止决定](research_memos/p8_prospective_headroom_interpretation_20261019.md)

较早的4句、2位说话人、16条件pilot给出Noisy/frontend/OFF微WER为66.03%/59.62%/80.77%。它与32条件使用不同句子，不能合并成独立总体或用两次总体均值反转推断机制改变；两批都没有提供独立人工词错率或音素恢复率。[16条件原始指标](experiments/p8_voicebank_native_content_pilot_20261019/attempt02/metrics.json)

### 1.3 Noisy条件消融说明敏感性，不单独识别正确利用机制

已完成的VCV反事实诊断移除了**直接Noisy条件中的语音成分**，但保留由原始输入得到、仍携带语音信息的前端输出锚点。因此，它不是“整个后端只接收纯噪声”的实验。五例来自既有识别器变化的选择，同一位非盲听者报告1例原后端更接近参考、4例整体相似；其中一例两版本均出现/t/→/k/的判断，另一例消融后尾部更像/s/样结构。相似不表示正确或统计等效，摩擦样残留也不自动等于语义幻觉。[五例听感与原话索引](experiments/p8_vcv_speech_component_ablation_20261019/listener_assessment_20261019.md)

在另一个5243局部消融中，移除目标条件语音后CTC的/d/后验由0.869617升至0.979222，而用户后来报告/d/相对稍弱。这不矛盾：类别后验、可听强度与身份正确是不同端点。现有证据至多支持所选干预下的条件敏感性；由于条件分布失配、锚点冲突和上下文路径尚未排除，不能据此认定后端完全忽略Noisy、或已正确利用音素特异证据。[最新人工修正与实验边界](experiments/p8_5243_local_condition_necessity_20261019/listener_assessment_20261019.md)

### 1.4 重建诊断约束假设，但不提供可部署修复

对32条件已存波形的离线分析发现，所有条件沿frontend到OFF的固定线段，其clean-aware MSE最优点都位于两端之间。共同oracle系数为0.2611，使按case等权、以clean功率归一的MSE由0.26391降到0.25622，约2.92%；完整OFF为0.31783。这是使用clean求得的样本内MSE空间，不是部署系数、听感收益或已验证的α控制器。修改能量中非全局gain部分的case平均占比98.40%，也不是“98.40%幻觉”。没有导出混合音频或重算其ASR/音质。[方向审计报告](experiments/p8_vb_residual_direction_audit_20261019/experiment_report.md)

另一项旧DNS/Router四seed缓存分析中，seed离散度占pooled归一参考误差12.70%；剩余87.30%是有限四样本中心对clean的误差，不是已证实的真实生成器偏差。该缓存使用Router及旧输入处理，与本次VB原生OFF不同，而且缺少生成时原始WAV哈希，当前文件稳定性不能补成历史字节身份。此结果既不支持直接迁移到当前后端，也未验证均值音频的内容或音质。[四seed报告](experiments/p8_cached_multiseed_variance_audit_20261019/experiment_report.md)

### 1.5 P8-I技术准备后取得了受限人耳观察，完整判别仍未验证

用户新增的P8-I先问：对于明确的局部假设，原始Noisy是否有可验证判别信息？目前已完成I0的12对/6位说话人元数据登记、I1a配对/残差描述审计及I1b内存编辑完整性验证。I1a的12对最佳lag均为0，但这不证明残差是纯噪声。I1b保留24目标槽及96安全操作，不导出音频；其位置对照不是无语音负例，实际taper编辑能量未精确配平，48组相关比较的最大相对差82.78%。通过的端点是技术完整性，不是自然事件缺失或可辨识性。[I1a报告](experiments/p8_i1_pair_residual_audit_20261019/experiment_report.md) · [I1b报告](experiments/p8_i1b_event_pair_integrity_20261019/experiment_report.md)

发布者训练日志及后续README语义补证给出这12句标称SNR为0–15 dB，而非已经完成的−18 dB局部辨识任务；具体噪声录音/切片来源仍未知，标称SNR也不是80 ms局部SNR。原始日志审计中的SNR空值没有倒改，解释以独立补证为准。D0–D4、校准/弃权、真假事件双向CAEG与独立人耳结构评价仍无验证结果。原四人188判断仅为历史设计，后经用户授权改为一人94判断并已收口：Noisy41结构/2非语音/4unknown、Clean45/0/2；部分unknown来自定位反应困难，不能当无信息。Clean缺可靠非语音类，不能计假阳性或将同人参考称独立gold。[来源补证](research_memos/p8_vb_train_log_semantics_20261019.json) · [实际反馈](experiments/p8_i1b_single_listener_materials_20261019/responses/completed_interpretation_20261019.md)

后续VB32缓存线性读出已独立复算：full-Noisy比simple-Noisy两连续谱代理MSE下降17.28%/8.96%，但simple-frontend全量优于两full，未支持复杂表示的稳定内容优势。它不是原H1/H0任务，也不是具体辅音结果。[读出报告](experiments/p8_cached_structure_readability_20261019/experiment_report.md)

更直接的12例单人参照辅助F/OFF观察中，双方同四个0dB例均preserved；−18dB仍有missing/different/added风险，未见F missing→OFF preserved的明确checkbox组合。OFF获3例质量偏好均在内容双方preserved例。首例备注OFF在The之前有额外声音，与原missing选项同时保留不改；没有精确身份标签或总体率。20例未评不催补、不中途选新样本救收益。[12例记录](experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md)

### 1.6 “80k原生OFF”不应被简化成只训练80k的StoRM

**用户后续明确确认**：“80k checkpoint 是在300k权重的基础上冻结前后端模型，只训练router的结果”。论文据此采用**300k基础前后端＋80k Router-only增量训练**的历史描述，不再把80k/300k当作两套基础模型各自的总训练步数。下面保留静态审计的独立证据与限制；用户说明补足训练历史，但没有额外指定初始化时RAW/EMA物化的具体操作。

随后完成的只读checkpoint比较进一步限定了基线身份：80k Router文件与本地300k文件的538项同名基础RAW tensor中，42项有效字节相同、496项不同；但300k保存的536项EMA与80k中被其内容覆盖的536项RAW tensor，在dtype、shape及有效字节SHA的**多重集合上完全一致**。其余两项Fourier投影W在两文件RAW中相同，当前源码将W设为不需梯度。这与先物化300k EMA、再冻结基础模型训练Router的解释一致，但仍不是历史初始化路径或参数名—EMA位置映射的证明。重复内容存在，不能把无名集合匹配直接当作命名模型或运行输出等价。[静态比较原始指标](experiments/p8_checkpoint_base_identity_20261019/metrics.json) · [执行及限制](experiments/p8_checkpoint_base_identity_20261019/experiment_report.md)

上述多重集合是Root对本次保存SHA字段的后验算术核对，没有再次读取checkpoint storage或运行模型。80k文件自己的EMA只有8项，当前代码在`rrsd_freeze_base`下仅更新/应用可训练Router参数的EMA；不能把“用了EMA”误写为“又对整个基础模型做了80k EMA”。因此，既不能由RAW差异推断模型损坏，也不能由文件名断言P8失败只是基础模型训练不足。论文继续称其为**指定80k Router checkpoint、关闭Router后的原生OFF**，不外推为所有StoRM模型的已证实瓶颈，不据此次比较自动重跑300k或改旧结果。

同一80k checkpoint的Router-on/off比较控制了基础前后端权重这一因素，是模块开关对照；仍须报告Router额外训练成本、score调用及推理成本，不能称matched-compute已自动成立。若另与独立300k文件比较，实际RAW/EMA、归一化、采样器、seed及数据条件还需一致，不能把用户历史说明扩张为所有运行输出等价。

**成本口径另有一个需保留的区别**：旧正式Router-on/off的off是`rrsd=true, scale=0`，当前路径仍计算Router；本P8的native OFF为`rrsd=false`，不调用Router。二者不能混为同一个延时基线。Future-gradient额外Score重评分只用于训练，正常路由推理不增加基础Score评估数；t-cut仅屏蔽动作，不提前省掉Router计算。静态计数、参数量及原始job计时的完整限定见 [旧论文草稿§2.4](THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md#24-复杂度与计时口径区分训练lookahead和推理路由)，未补跑任何模型或冻结测试集。

## 2. 论文主张的可用与不可用版本

| 可以写 | 不能据此写 |
|---|---|
| 选定材料显示音质、重建和内容代理不一致，需要独立内容端点 | 已实现感知、保真、音素身份的全面改善 |
| 后端对某些Noisy条件干预敏感 | 后端必然忽略Noisy／必然正确使用Noisy |
| 人耳报告了具体异常结构及正向音质案例，保留单人非盲限制 | 已测得总体幻觉率或真实音素恢复率 |
| clean-aware oracle及缓存分解帮助排除简单解释 | 已获得可部署α策略或证实87.30%真实偏差 |
| P8-I完成受限管线的可复现准备 | 已证明Noisy有／无/k/、/s/身份信息，或CAEG有效 |
| 指定checkpoint的原生OFF存在所记录风险；基础权重呈EMA内容对应模式 | 仅凭80k文件名断言主干只训练80k、或仅凭无名EMA内容匹配宣布完整模型等价 |

**本节结论**：已有工作建立的是问题表征、代理评价的失效风险和候选解释边界。局部Noisy证据是否可读、后端是否能可靠利用它，以及新机制能否兼顾补真音、拒假音和音质，仍未验证。上述诊断不能单独确立算法新颖性或宣布长期科研目标完成。

## 3. 本次写作核对范围

§1.1–1.5只读检查原始JSON字段、报告、已有验证回执及反馈，没有重做原声学分析或执行验证器；通过回执仅按其覆盖范围引用，不称作本轮重新验收全部模型/波形。§1.6引用随后一次独立记录的静态checkpoint读取，并对其保存SHA字段作后验算术；只读复审也核对了该多重集合结论，未再次读取storage。以下原metrics SHA256实读匹配已有记录，写作前后保持不变：

- 32条件：`577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04`
- 方向诊断：`06603c8fef1a7ee7b5a7bed89b159f6f97036e06e7d04c241524190892fe3e17`
- 四seed缓存：`1ba98c0cd57911327412339baa6b1b46d1ecf96bb73a3a00c587af10d41c942e`
- I1b：`24c0ee1297e261dbecc4cfa9b2f3441292e45e71709cb2fddb0b60462362464c`
- 新静态checkpoint比较：`78d832dfa3123d1c27a4bc5daa7002521874d2b5d5f0947f1f913de4a9d07858`

下一科学动作仍按既有计划：资源及授权确认后开展固定原始材料的独立结构评价；不重复催问、不追加合同/来源扫描，不为填满研究流程而先训练编辑检测器。本稿服务论文写作，不代替所欠的内容与机制验证。
