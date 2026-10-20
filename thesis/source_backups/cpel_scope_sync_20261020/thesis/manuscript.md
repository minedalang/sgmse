# StoRM / SGMSE：已实现方法与已完成证据整合草稿

> **不是完整毕业论文，也不是已验证新方法成功的宣告。** 本稿整合八份可编辑草稿，含中英文摘要、绪论与结论；学校模板、TeX编译及浏览器显示仍未验收，完整毕业论文所需其他部分尚未齐备。当前没有独立新机制收益确认，科研目标未完成；科学结论未因集成而扩张。

> **公式显示限制**：本地未使用数学渲染器；公式保留原始 LaTeX 文本（灰底公式块），不声称数学排版或真实浏览器显示已验收。

[阅读说明与复现](README.md) · [源文件及 SHA256 清单](sources_manifest.json)


# 导读：研究问题与阶段性结论

来源：[THESIS_FRAMING_DRAFT_20261019.md](../THESIS_FRAMING_DRAFT_20261019.md)

## 研究问题与阶段性结论：整合稿导读

> 本文只整合已实现方法、冻结实验与实际收到的听感，不新增实验结果。它不是学校格式的完整绪论，也不是“创新机制已经成功”的结论。相关工作的具体引文由配套章节承接；尚未完成的身份评价、独立内容确认与新机制验证不能因整理成稿而视为完成。

### 1. 研究背景与问题定义

语音增强不只是使输出更平滑或更像自然语音。对于交流任务，输出还应与实际输入中的说话内容相对应。本文研究的对象是本地StoRM/SGMSE系统：判别式前端先给出估计，扩散后端在前端估计与原始含噪语音的共同条件下进行生成。两阶段的协作带来一个必须通过实验区分的问题：后端使声音更自然时，是恢复了真实但微弱的发音，还是主要改变了残噪、响度及发音外观？同样，前端中不明显的结构，也不能仅凭能量变小就认定为已丢失的某个音素。

因此，本文将评价对象分为三个层次。第一层是重建与感知代理，包括SI-SDR、STOI、PESQ和DNSMOS；第二层是与参考内容的对应，包括识别器词错误和人工观察到的缺失、新增或不同内容；第三层是具体机制，即生成是否受真实音素特异声学证据支持。三层可能相关，但不能互相替代。DNSMOS并非本项目的人耳MOS，识别器错误也不等于人耳实际听到的错误词数。本文不把任何单一指标当作真实内容保真的充分条件。

### 2. 研究问题与本文实际完成的工作

**问题一：冻结基础系统上的局部路由，能够改变哪些质量与重建指标？** 已实现方法是在300k基础系统上冻结前后端、增量训练80k的Router。其学习信号来自局部动作之后的未来状态重评分，不以独立声学似然解释该Score。本文描述真实损失与梯度路径，并采用同一checkpoint上的动作开关与固定t-cut对照报告结果。对照控制基础权重，但单次训练与现有评价仍不构成对所有训练seed、说话人或声学环境的总体保证。

**问题二：现有后端的质量变化，是否同时带来可确认的内容恢复？** 已完成的自然内容pilot、固定候选池和用户参照辅助观察回答了这个问题的一部分。分析既保留内容可对应时的音质正观察，也保留极低SNR下的缺失与额外发音风险。ASR失效、重复句子及单人参照辅助的限制分别披露。实际收到的12例内容判断按12例分析，剩余20例不补齐；已完成的94项局部结构判断也不转写为音素身份标签。

**问题三：输入中是否存在可利用的正确证据，当前模型又是否正确利用？** 已有条件干预与固定线性读出只提供有限诊断。条件改变会影响输出，不足以排除条件失配或证明正确利用；连续谱代理可预测，也不等于真实音素可恢复。本问题仍未完整回答。候选级双向证据验证是后续假设，而不是本文已验证成功的增强算法。

上述工作构成可复核的方法描述、条件性实验结果与问题分析。它们不自动构成新的独立学术贡献：方法先例、内容收益、身份保持以及训练和推理预算仍需与相应主张匹配的证据。

### 3. 阶段性结论

在冻结评价中，Router呈现条件相关的权衡。无混响条件下部分重建指标改善，但并非全部感知代理同向；混响条件下仍有重建改善与质量、可懂度代理下降并存。保留跨条件file-id配对的事后不确定性补充没有改变这一基本方向，也没有成为新的独立确认集。因此，当前结果不支持“感知、重建、内容全面改善”或“已经获得稳健Pareto优势”的结论。

内容分析进一步说明，不能仅凭后端输出更自然或某个ASR分数更好，宣称恢复了真实弱音素。实际参照辅助反馈中，内容均能对应时存在后端音质偏好；困难案例同时存在缺失、不同内容和句首额外声音观察。两面证据需要同时保留。局部结构听感与连续谱读出说明了可听或可预测结构的有限性质，但没有填补真实音素身份与双向事件验证的缺口。

因此，本阶段最可靠的结论是明确了已实现路由的条件性收益及其边界，并收窄了后续内容保真研究必须回答的问题，而不是已经解决了生成式增强的内容错误。下一步新方法若要成立，需要在未用于反复筛选的材料上，同时证明真实内容恢复、错误新增控制及必要的质量和预算权衡；不能通过丢弃所有不确定弱音，或把代理任务改得更容易，来替代这些目标。该验证仍属于未完成工作。

### 4. 整合稿的阅读顺序与未完成范围

阅读顺序为研究问题、相关工作、已实现Router、实验、讨论及本节阶段性结论。方法章节回答“做了什么”，实验章节回答“观察到了什么”，讨论章节区分解释与证据；未实施的P8-I完整判别、CAEG或其他新机制不得混入已完成方法。

整合稿仍不是完整毕业论文：学校模板、完整绪论与摘要、正式参考文献排版、可提交的图表、身份指标的充分验证及满足创新主张的新方法证据尚未齐备。将现有章节整合为可读文件是论文交付进展，不代表长期科研目标完成。


# 摘要：中文与English Abstract

来源：[THESIS_ABSTRACT_DRAFT_20261020.md](../THESIS_ABSTRACT_DRAFT_20261020.md)

## 摘要与关键词

### 中文摘要

生成式语音增强需要在提高声音质量的同时保持输入内容，但感知代理、信号重建与真实内容恢复并不等价。本文以本地指定的StoRM/SGMSE判别式前端与扩散后端系统为对象，研究冻结基座上的局部路由权衡、内容评价的证据边界，以及受控响应训练的独立增量。首先，描述已实现Router的动作、未来重评分监督与成本，并整理同一checkpoint上的模块开关评价。结果呈现条件性多目标权衡：无混响下部分重建和质量代理改善，部分感知代理下降；混响下亦未形成综合优势。其次，结合自然句探索、条件干预、结构读出与单人参照辅助观察，区分结构可听、识别器输出和具体音素证据；这些材料不足以证明真实音素恢复或幻觉降低。最后，固定CPEL小试以匹配辅助任务的M臂对照额外成对约束的P臂，两臂各训练500更新。P相对M的两种留出响应代理损失仅下降约1.67%和1.66%，未达预定10%筛查门，正常增强指标几乎不变。固定64条Clean的tiny.en词读出仅49条匹配，未满足该诊断的适用标准，未用于含噪或增强输出评价。受限于探索材料与评价协议，现有结果明确了指定系统的条件性行为与代理评价限制，尚未确认独立新机制收益，内容、身份、质量与效率的综合目标仍未实现。未来研究需以可靠的独立输出内容读数或不同且可证伪的机制依据支撑，有限负筛查不否定所有研究方向。

**关键词**：语音增强；扩散模型；内容保真；可靠性路由；反事实响应

### English Abstract

Generative speech enhancement must improve sound quality while preserving input content, yet perceptual proxies, signal reconstruction, and genuine content recovery are distinct objectives. This study examines a specified local StoRM/SGMSE system with a discriminative frontend and a diffusion backend, focusing on routing tradeoffs, the evidential limits of content evaluation, and the incremental value of controlled response training. First, it describes the implemented Router, its future rescoring supervision, and computational costs, and consolidates module activation comparisons within the same checkpoint. Results show conditional tradeoffs: some reconstruction and quality proxies improve without reverberation while other perceptual proxies decline; reverberant evaluation also provides no comprehensive advantage. Second, exploratory natural sentences, condition interventions, structural readouts, and reference-assisted observations from one listener distinguish audible structure, recognizer outputs, and phoneme-specific evidence. These materials do not establish genuine phoneme recovery or reduced hallucination. Finally, a fixed CPEL pilot compares a matched auxiliary-task arm M against arm P with additional paired constraints, each trained for 500 updates. P reduces two held-out response proxy losses relative to M by approximately 1.67% and 1.66%, below the predefined 10% screening threshold, while ordinary enhancement metrics remain nearly unchanged. A fixed tiny.en word readout matches only 49 of 64 clean recordings, fails its applicability criterion, and is not applied to noisy or enhanced outputs. Within these exploratory materials and evaluation protocols, results characterize conditional system behavior and evaluation limitations without confirming independent benefits from a new mechanism. The combined content, identity, quality, and efficiency objectives remain unmet. Future work requires reliable independent output-content readouts or a different falsifiable mechanistic basis; bounded negative screening does not invalidate all research directions.

**Keywords**: speech enhancement; diffusion models; content fidelity; reliability routing; counterfactual response


# 第一部分：绪论

来源：[THESIS_INTRODUCTION_DRAFT_20261020.md](../THESIS_INTRODUCTION_DRAFT_20261020.md)

## 绪论：生成式语音增强中的质量、重建与内容证据

> 可编辑正文草稿。本章依据现有方法、实验、讨论与固定版本文献编写；不新增科学结果，不宣告独立新机制成功。引用的本地报告与已有BibTeX共同提供来源，尚未适配学校论文模板。

### 1. 研究背景与意义

语音增强的基本任务是在噪声干扰下保留目标语音，为交流与后续处理提供更清楚的声音。评价这一任务时，输出的自然程度、与参考信号的接近程度以及说话内容是否正确对应，分别描述不同性质。更平滑的波形不一定保留微弱辅音，更高的质量预测分数也不保证没有错误新增发音。在极低信噪比下，这种区别尤其重要：系统需要抑制强噪声，又要避免把仍有证据的弱语音一并删除；生成阶段还需要区分可恢复的细节与缺乏支持的补全。

本文以本地StoRM/SGMSE两阶段系统为研究对象。判别式前端产生初步估计，扩散后端利用该估计与含噪输入生成输出。该框架为噪声抑制与自然细节恢复提供了协作方式，也使两者之间的分工成为可研究的问题。StoRM本身已采用判别估计引导随机再生，本文不把组合前后端当作新思想；对指定本地实现与checkpoint的观察，也不外推为原论文全部系统的性质。[既有相关工作与固定版本引文](../P8_RELATED_WORK_DRAFT_20261019.md)

对于实际交流，核心要求是输出内容与输入中的真实说话相对应。前端可能降低某些局部结构的可听程度，后端可能提高声音质量，也可能改变发音外观。要判断其中是否发生正确恢复，不能仅看能量、频谱细节或识别器概率；还需知道原来是什么内容，输出实际被读成什么，以及变化能否归因于可利用的声学证据。研究这些区别的意义，在于让方法收益和评价结论对应，避免以代理分数替代交流任务的目标。

### 2. 核心问题与评价层次

本文区分三个层次。首先是重建与感知代理，以SI-SDR、STOI、PESQ及DNSMOS等描述指定评价协议下的变化。其中DNSMOS是模型预测，不是本项目的人耳MOS。其次是内容对应，包括识别器转写、参照辅助人工观察中的缺失、不同内容与额外声音。单一ASR的词错误率是该识别器的输出指标，不等于人类词错误率。最后是机制层次：某次生成变化是否利用了足以区分真实事件与替代事件的证据。前两个层次的改善均不能自动证明第三层成立。

在加性混合中存在语音分量，与观察者能否从单份含噪信号区分具体发音，是不同问题。改变直接Noisy条件会改变后端输出，也只能说明条件敏感性；若锚点仍保留原语音信息，或干预造成分布失配，就不能将变化直接解释为音素特异证据的正确利用。同理，训练中响应了人工编辑，不代表自然未编辑输出已经恢复正确内容。

已有工作分别涉及保护可靠观测、音系表征、双教师蒸馏、多候选选择与观测约束生成。相关章节保存了StoRM、GALD-SE、PASE、StuPASE、ArtiFree、Grounded Decoding及L3-SE七项固定版本引文。它们限制一般思想的优先权，但不构成本地机制的收益证据。本文不搬用外部性能数字作直接排名，也不因实现不同就宣布具有独立新颖性。[固定版本BibTeX](../P8_RELATED_WORK_REFERENCES_20261019.bib)

### 3. 研究对象与系统边界

本文的核心科学问题面向无混响、极低SNR中的内容忠实性；已完成材料的范围则更广且并不等价。冻结DNS评价含无混响与混响两个条件，各150份且按file-id匹配，不能当作300个独立底层内容。混响结果用于披露已有方法的适用边界，不代表已完成新的混响建模。VoiceBank自然句、受控VCV、局部条件干预、结构判断与短词Clean读出各有独立用途，不拼接成统一确认集。

基础系统的训练历史为300k前后端基座，加上冻结基座后的80k Router-only增量训练。本文比较同一checkpoint的Router动作开关，控制基础权重这一因素；80k不能误写为整个系统只训练80k。正式Router-off的scale为零但仍执行Router，与P8中完全关闭Router的native OFF不同。参数量、额外训练调用和推理成本需要按各自路径解释，不能以模块较小推断成本为零。

所审mode2配置中，OUVE以判别前端输出为锚点，Score同时接收当前状态、前端估计与原始Noisy条件。前端是同一输入的变换，不是独立观测；压缩谱中的残差也不能直接称为物理噪声。clean、真实噪声分量与参考文本可以服务训练或离线诊断，不属于部署时可用的oracle信息。具体表示、梯度链及采样约定由方法章节给出。

### 4. 研究问题与已完成工作

**RQ1：冻结基础系统上的局部路由如何改变感知质量与重建之间的权衡？** 本文描述已实现的Future-gradient Reliability Routing及其真实监督路径，整理同checkpoint动作开关与固定t-cut的结果、条件差异和成本。该问题关注哪些端点改善、哪些端点下降，以及比较控制了什么；不预设全面改善或稳健Pareto优势。

**RQ2：现有质量变化中，哪些证据涉及真实内容，哪些仅涉及代理或结构？** 本文汇总自然内容pilot、候选余量、条件干预、缓存读出与实际收到的人工反馈。94项判断对应47位置的Clean/Noisy结构观察，12个内容case含重复句与条件，均为单人、非独立gold；它们不产生人类WER或音素恢复率。分析保留正向质量观察与内容异常，并将未知项作为未知，不从ASR失败推断波形物理不可辨。

**RQ3：受控响应训练能否在匹配的普通辅助任务之上提供独立增量？** 固定CPEL小试以M臂的受控逐点辅助任务为比较对象，P臂额外加入成对响应约束，在匹配配方与调用预算下筛查其增量。两臂各500更新，heldout代理下降约1.67%／1.66%，未达预定10%门；正常增强指标几乎不变。这回答的是有界开发筛查，不是所有反事实训练的最终有效性判断。[CPEL解释与退出](../research_memos/p8_cpel_mp_pilot_interpretation_20261020.md)

围绕这些问题，本文完成的是方法实现描述、冻结评价整理、内容与代理的区分以及候选方案的失败退出记录，构成可追溯的工作清单。短词评价端点还先在64条Clean上检查适用性：tiny.en仅49条exact-label匹配，未进入Noisy、前端或后端评价。因此其结果只限制当前读出工具，不诊断增强系统。[短词端点收口](../research_memos/p8_keyword_clean_readout_closeout_20261020.md)

### 5. 论文组织与当前状态

导读之后，本章说明背景、研究问题与边界；相关工作章节讨论既有思想及本文定位；方法章节描述已实现Router；实验章节呈现冻结比较、P8诊断和CPEL小试；讨论章节分析可用解释与混杂；结论章节分别回答三个研究问题，并界定尚未实现的目标。原写作校验记录及固定版本参考文献保留于附录。

现阶段尚无独立新机制收益确认，真实内容恢复、错误新增控制、身份保持及质量与效率的综合目标仍未完成。论文结构补全是文档交付进展，不代替科学验证。未来研究有赖于可靠输出内容读数或不同且可证伪的机制依据，当前结果不足以支持扩大训练或继续调整已关闭配置。[当前战略决定](../research_memos/p8_post_cpel_strategy_20261020.md)


# 第二部分：相关工作与研究定位

来源：[P8_RELATED_WORK_DRAFT_20261019.md](../P8_RELATED_WORK_DRAFT_20261019.md)

## P8相关工作与研究定位：论文段落草稿

> **使用范围**：可并入论文“相关工作／问题定位”，不是已完成CAEG方法章节。本轮复核下列固定版本原文，并补读StuPASE v2；定向查阅不等于穷尽查新，不依据作者摘要的SOTA措辞宣布本项目优劣。自己的结果见 [观察与局限](../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)，研究入口为 [/data/research_plan.md](../../../research_plan.md)。配套 [BibTeX](../P8_RELATED_WORK_REFERENCES_20261019.bib) 保留已核版本，未补造会议卷页。

### 1. 判别式估计与生成式恢复的协作

StoRM利用判别式估计引导后续随机再生，结合预测模型与扩散生成。因此，将前端与后端结合、用前端抑制生成伪影，不能单独构成本研究的新颖性。本文以该协作框架为起点，进一步关注指定低SNR材料中音质收益与内容错误可能并存的现象；本地改动版本及checkpoint的诊断不等于对原论文所有系统的复现或否定。[StoRM原论文](https://arxiv.org/abs/2212.11851) · [作者官方实现及正式引文](https://github.com/sp-uhh/storm/blob/master/README.md)

GALD-SE根据粗略相位敏感mask构造各向异性扰动，减少对可靠时频结构的破坏，再通过扩散恢复。它已具体覆盖“不要重新生成可靠观测成分”的思路。我们的推断是：增加mask、按噪声比例控制扰动或保护弱能量区域，本身不足以区分P8新机制；仍需证明所保护或恢复的是正确内容，而非仅某种能量结构。[GALD-SE v5，§II-B–C](https://arxiv.org/html/2409.15101v5)

### 2. 表征先验与声学条件的分工

PASE通过clean教师／noisy学生的去噪表征蒸馏获得DeWavLM，再用高层音系表征与低层声学表征的双流重建生成波形。这不同于给冻结StoRM附加一个冻结SSL距离损失。两者训练对象、目标及合成路径不同，所以本项目某个SSL-loss负结果不能否定PASE；同样，简单增加WavLM监督也不能据此被称为独立创新。[PASE v1，§3.1–3.3](https://arxiv.org/html/2511.13300v1)

StuPASE进一步采用dry-target微调，并以增强音系表征和noisy Mel为条件，通过flow matching生成增强Mel，再由vocoder生成波形。该路线表明“内容表征与声学生成分工”已有直接先例，不能仅把GAN换成扩散／flow就宣称新贡献。它也不直接证明本项目冻结OUVE后端存在同一种瓶颈。[StuPASE v2，§2.2–2.3](https://arxiv.org/html/2603.09234v2)

L3-SE已将声学／语义双教师去噪蒸馏用于自回归声学token生成；其low-SNR评价为−5至5dB，而非本地−18dB。该架构不同于StoRM/OUVE，不作跨系统收益或成本比较；但仅提出SSL双流／双教师不足以构成明确的新贡献。[L3-SE v1，§3–4](https://arxiv.org/html/2605.08608v1)

### 3. 多候选一致性与观测约束生成

ArtiFree利用多次扩散输出的语音表征方差检测伪影风险，并研究基于候选中心性或参考表征相关性的选择。涉及clean参考的选择必须与部署可获得的noisy参考方案分开解释。因此，多seed、不确定性分数或“生成后筛选”都不是本项目独有机制；候选彼此一致也不逻辑保证事件身份正确。[ArtiFree v1，§3.1–3.2及Algorithms 1–2](https://arxiv.org/html/2509.19495v1)

Grounded Decoding将确定性估计与原观测构造成离散证据，在FSQ空间用Hamming距离惩罚偏离，按残差SNR代理调节约束强度，并在保留锚点的局部候选邻域内细化。由此，“让观测约束生成候选”及“始终保留原锚点”都不是足够的新颖性。其离散自回归实现与本地连续扩散不同，但仅更换生成载体不能自动构成贡献。[Grounded Decoding v1，§III-D–G](https://arxiv.org/html/2609.04245v1)

### 4. 本研究尚待验证的定位

以上先例已经涉及保护观测线索、音系先验、声学条件、多候选选择和生成约束。本研究不主张首次提出这些一般思想。P8-I拟先验证：在明确局部假设和受控条件下，**单份原始Noisy是否提供可复核的判别信息**。只有这个问题得到支持，后续才研究候选特异、可双向支持／反对并允许拒答的声学审查。

这是**待验证问题与可能差异**，不是已证明的文献空白或已实现算法。它必须区别于只检测编辑痕迹、只按候选自然度选择、只降低所有弱音能量，以及只改善单一ASR分数。若不能在相同覆盖率与成本下优于简单残差／谱一致性等有竞争力基线，或补真音与拒假音不能兼顾，就不应升级为新的增强机制。现阶段不能写“CAEG已优于上述方法”，也不能将完整目标缩成更容易通过的语音结构分类。

### 5. 写作与比较口径

| 容易混淆的表述 | 本文采用的限定 |
|---|---|
| 加Noisy、mask、SSL或多候选即创新 | 分别与已有条件输入、GALD-SE、PASE、ArtiFree和Grounded Decoding比较具体机制 |
| 本地80k OFF代表全部StoRM | 只指明checkpoint、EMA路径、采样器和本地实现；不由文件名推断主干训练步数 |
| 外部论文WER较低，所以本地机制被否定 | 不跨数据、ASR和计分协议直接排高低；例如Grounded Decoding §IV-B默认使用capped句均WER，而本地32条件为uncapped微WER |
| dry等于物理无混响，或论文高分即因果证明 | 明确目标构造与评价定义；不将作者解释移植为本地因果事实 |
| 结构可读性证明/k/、/s/恢复 | 结构、事件类别、具体身份及最终增强收益分别验证 |

上表关于计分差异以 [Grounded Decoding v1 §IV-B](https://arxiv.org/html/2609.04245v1) 和 [本地冻结指标](../experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json) 为依据，不新增ASR评分或测试集调参。



# 第三部分：已实现方法

来源：[METHODS_RRSD_IMPLEMENTED_DRAFT_20261019.md](../METHODS_RRSD_IMPLEMENTED_DRAFT_20261019.md)

## 已实现方法正文草稿：冻结基座上的Future-gradient Reliability Routing

> 本章描述已实现的Router，不把P8-I／CAEG候选写成完成方法，也不单独认定新颖性或综合增强成功。训练历史经用户确认：300k前后端基座冻结，仅Router增量训练80k。当前源码、已保存配置及历史结果互相区分；源码对应关系不是重训复现。结果和条件性权衡见[原结果草稿](../THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md)，内容局限见[P8讨论](../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)，先例见[相关工作](../P8_RELATED_WORK_DRAFT_20261019.md)。

### 1. 问题与表示

记含噪波形为 (x)，训练clean为 (s)，冻结前端为 (F)。在`normalize=noisy`的共同标度下，令 (r=\max(\|x\|_\infty,10^{-12}))，clean与noisy使用同一 (r)，而不是分别按各自峰值归一化。以实际STFT及压缩变换 (\mathcal T) 定义

\[
Z_0=\mathcal T(s/r),\qquad Y=\mathcal T(x/r),\qquad A=F_{\rm spec}(Y).
\]

这里 (F_{\rm spec}) 表示前端在同一变换谱域的输出；波形推理入口先完成相同尺度下的前端处理和长度匹配。保存的谱变换为指数幅度压缩，指数0.5、factor0.15。下文全部加减及路由发生在该模型坐标，不将 (Y-A) 称为物理噪声，也不把谱域MSE等同于波形MSE或音素正确率。

所审配置为mode2：OUVE主锚点 (y_{\rm main}=A)，直接条件 (y_{\rm cond}=Y)。Score路径接收状态及两个条件；不是仅看到前端输出。将 (A=F_{\rm spec}(Y)) 同时输入不构成第二份独立物理观测。

### 2. 固定OUVE几何与clean读出

在未启用其他残差锚点变体时，正向过程为

\[
dZ_t=\theta(A-Z_t)dt+g(t)dW_t,
\qquad a_t=e^{-\theta t},\quad b_t=1-a_t,
\]
\[
Z_t=a_t Z_0+b_t A+\sigma_t\varepsilon.
\]

其中 (g(t)) 与扰动标准差 (\sigma_t) 是两个不同量，分别由`OUVESDE.sde`和`_std`给出。本文沿用代码的复噪声与Score输出尺度，记冻结网络的有效输出为 (q_\theta(Z_t,A,Y,t))，不把它另行包装为独立声学似然或改变其factor约定。

代码采用的代数clean读出为

\[
\widehat Z_0=
\frac{Z_t+\sigma_t^2 q_\theta-b_tA}{a_t}.
\]

实际实现对 (a_t) 设 (10^{-5}) 的下界。在本配置 (t\in[0.03,1])、(\theta=1.5) 的范围，该下界不触发。此读出不是已证实的真实clean或波形MMSE；有限网络误差、表示变换和采样过程的限制仍存在。

### 3. 局部路由器

#### 3.1 输入特征与结构

路由器 (Q_\phi) 只接收推理可用量：(Z_t,\widehat Z_0,A,Y,t)。在最多32个频带、16个时间块的低分辨率网格上，构造以下19个实特征通道：

| 特征组 | 通道数 |
|---|---:|
| (\widehat Z_0,A-\widehat Z_0,y_{\rm main}-\widehat Z_0,y_{\rm cond}-\widehat Z_0,Z_t-\widehat Z_0) 各自局部幅度的mean/std/max | 15 |
| 前三种差异向量的三组局部实内积归一化关系 | 3 |
| 归一化扩散时间 (t/T) | 1 |

这些不是19份独立证据；mode2中 (y_{\rm main}=A)，部分特征重复。网络依次使用1×1投影、两个3×3上下文卷积及1×1输出，隐藏通道48，中间采用SiLU。Sigmoid得到低分辨率系数，经双线性插值得到谱网格上的 (\alpha_\phi\in[0,1])。已有checkpoint审计计得42,577个Router参数，基础前后端保持冻结。

#### 3.2 动作及Score变换

路由动作是向当前前端谱移动：

\[
\delta Z_0=\alpha_\phi\odot(A-\widehat Z_0),
\qquad \widehat Z_0^{\rm route}=\widehat Z_0+\delta Z_0.
\]

在相同 (Z_t,A,t) 下，将这一clean读出动作换回Score坐标：

\[
q^{\rm route}=q_\theta+
\frac{a_t}{\sigma_t^2}\delta Z_0.
\]

实现对 (\sigma_t^2) 使用 (10^{-8}) 下界，并有非有限数保护。本配置路由为单alpha模式，不是“gate×strength”双头；表述中不应加入未启用的门控机制。谱域凸组合也不能推出波形逐样本凸组合，更不能保证补回两候选中均缺失的正确音素。

### 4. 一步future重评分监督

#### 4.1 同状态、同未来时刻的对照

训练时，先从原扰动路径得到 (Z_t,q_\theta,\widehat Z_0)，将它们及条件detach。只有 (Q_\phi) 的动作保留梯度。对同一状态分别施加路由动作与零动作，并作一次确定性的反向均值步：

\[
Z_{t'}^{j}=Z_t-f(Z_t,A,t)\Delta t
+g(t)^2\Delta t\,q^j,\qquad
j\in\{0,\mathrm{route}\},
\]
\[
q^0=q_\theta,\quad
\Delta t=\min(\Delta t_{\rm base},\max(t-t_\epsilon,0)),
\quad t'=\max(t-\Delta t,t_\epsilon).
\]

保存的`rrsd_lookahead_dt=0`在当前实现中取

\[
\Delta t_{\rm base}=\max\!\left(\frac{T-t_\epsilon}{\max(N-1,1)},10^{-6}\right).
\]

这不是完整PC轨迹：该训练比较不加predictor随机噪声、不运行corrector，且不越过 (t_\epsilon)。两分支在相同 (t') 使用相同冻结网络重新估计clean，消除把未来误差与当前时刻误差作比分母的跨时间混杂。重新评分时两分支沿batch维堆叠，是一次2B大小的网络调用，不是无成本调用。

#### 4.2 相对目标与平滑

令 (E_i^j=\operatorname{mean}|\widehat Z_{0,i}^j(t')-Z_{0,i}|^2)，实际相对目标为

\[
L_{\rm future}=\frac1B\sum_{i=1}^B
\frac{E_i^{\rm route}}{\max(\operatorname{stopgrad}(E_i^0),\epsilon_o)},
\qquad \epsilon_o=10^{-6}.
\]

分母是clamp，不是 (E_i^0+\epsilon_o)。当 (E_i^0\ge\epsilon_o) 时，比分别小于／大于1可描述相对零动作的改善／损伤；分母下界触发时不能机械沿用“零动作恰好等于1”。优化的是一步、压缩谱域的监督代理，不是DNSMOS或真实音素恢复的可微替代真值。

低分辨率alpha的平滑项为两个方向的平均L1差分之和：

\[
L_{\rm smooth}=\operatorname{mean}|D_f\alpha|
+\operatorname{mean}|D_\tau\alpha|.
\]

当前审计checkpoint的Router有效目标为

\[
L_\phi=\lambda_R\bigl(\lambda_F L_{\rm future}
+\lambda_S L_{\rm smooth}\bigr),\quad
\lambda_R=1,\ \lambda_F=1,\ \lambda_S=0.01.
\]

训练代码还可计算基础DSM及其他分支；在本文所述冻结路径中，未经过Router的项不向 (\phi) 提供梯度。保存的gate cost系数虽为0.001，但`rrsd_use_gate=False`使该项实际为零；`contrastive_weight=0`，不能声称主结果来自WavLM／UniSpeech对比监督。其他后续可选分支不能因为存在于当前源码，就倒填为原80k训练使用的目标。

#### 4.3 冻结参数不等于切断输入梯度

梯度链为

\[
\phi\rightarrow\alpha_\phi\rightarrow q^{\rm route}
\rightarrow Z_{t'}^{\rm route}
\rightarrow q_\theta(Z_{t'}^{\rm route},A,Y,t')
\rightarrow\widehat Z_0^{\rm route}(t')\rightarrow L_{\rm future}.
\]

基础网络参数不更新，但该实现保留其对输入的导数。若将future网络调用放入`torch.no_grad()`，会切断经future Score输入Jacobian的梯度分量；clean读出仍有对未来状态的直接梯度项，因此不应误说Router必然完全没有梯度。这会改变预期的future-gradient监督，而不是等价的省显存实现。clean仅作为训练误差参考，不进入部署Router输入；这里的零动作是训练控制分支，不是P8-V的“该音素不存在”声学假设。

### 5. 推理算法与控制口径

```text
输入：原始noisy、冻结前端/Score、已训练Router；无clean
1. 用noisy共同幅度标度得到Y及前端锚点A。
2. 按既有OUVE prior在A附近初始化随机状态Z_T。
3. 对N个递减时间点：
   a. 计算一次基础Score q_theta。
   b. 若rrsd=true，读出Z0_hat，计算alpha和delta_Z0，转换成q_route；
      scale及可选t-cut作用在动作上，当前实现不提前跳过小网络。
   c. 使用原reverse-diffusion predictor更新状态，保留原随机项。
4. denoise=true时返回最后一跳的离散更新均值，再逆谱变换/ISTFT与恢复尺度。
```

本配置为PC、reverse-diffusion predictor、`corrector=none`、N=30。最后返回的是算法末跳均值，不是整条路径的后验均值，也不等同于直接输出上文的clean读出。训练的一步确定性future目标与最终随机推理目标因此并不完全相同。

同一80k checkpoint上的zero-scale与启用路由控制了基础权重，但成本口径仍要明确：旧正式`off`是`rrsd=true, scale=0`，也运行30次小Router；P8 native OFF的`rrsd=false`才跳过Router。两者基础Score静态评估均为30，不能据此说总计算完全相同或Router零开销。完整计时限制见[复杂度说明](../THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md#24-复杂度与计时口径区分训练lookahead和推理路由)。

### 6. 参数与来源定位

| 配置 | 已保存取值 |
|---|---|
| 基座／增量训练 | 用户确认300k前后端基座冻结＋80k Router训练 |
| SDE / mode / noisy主条件拼接 | OUVE / mode2 / `mode2_include_noisy_main=True` |
| OUVE stiffness、sigma范围 | 1.5；0.05–0.5 |
| t_epsilon / N | 0.03 / 30 |
| 谱变换 / 归一化 | exponent0.5、factor0.15 / noisy |
| Router网格 / 隐藏通道 | 32×16 / 48 |
| teacher / gate | future_grad / False |
| epsilon_o / lookahead_dt字段 | 1e-6 / 0（按上文规则解析） |

保存字段来源：[静态checkpoint记录](../experiments/p8_checkpoint_base_identity_20261019/metrics.json)中`comparison.A_hparams`及[机制审计](../experiments/rrsd_router_mechanism_audit_20261020/metrics.json)。数学与代码对应：`sgmse/joint_data_module.py`归一化，`sgmse/sdes.py:OUVESDE`，`sgmse/reliability_router.py:BandTimeReliabilityRouter`，`sgmse/joint_model.py:_rrsd_score_to_x0/_rrsd_reverse_mean_step/_rrsd_future_grad_loss/_compute_rrsd_loss/build_rrsd_score_calibrator`，`sgmse/sampling`。

本稿没有重新训练、运行增强或验收历史训练环境；未保留的历史运行细节仍不得由当前默认值填补。它是方法正文交付，不是新机制验证。现有质量／重建权衡与P8内容证据缺口继续保留；是否足以构成论文算法贡献，需要结合先例和公平实验证据判断。


# 第四部分：已完成实验结果

来源：[THESIS_EXPERIMENTS_CHAPTER_DRAFT_20261019.md](../THESIS_EXPERIMENTS_CHAPTER_DRAFT_20261019.md)


![图1：冻结原始逐条件区间，非speaker聚类/多重校正；不是新实验，DNSMOS非人耳MOS。](figures/router_condition_tradeoffs.svg)

[独立打开SVG图](figures/router_condition_tradeoffs.svg) · [精确数据JSON](figures/router_condition_tradeoffs.data.json)。仅从冻结JSON转录七指标均值差及原95%CI，每metric独立轴，不含pooled、不排名或合成分数。

## 实验章节草稿：冻结基座路由、内容风险与证据边界

> 本章整合已完成实验与用户反馈，不新增增强、训练、ASR或音频指标。方法对应[Router正文](../METHODS_RRSD_IMPLEMENTED_DRAFT_20261019.md)，先例见[相关工作](../P8_RELATED_WORK_DRAFT_20261019.md)，机制边界见[P8讨论](../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)。算法新颖性、真实辅音恢复与CAEG有效性不因本章写成而成立。

### 1. 实验问题、数据与独立性

本文分别回答：冻结基座上的Router改变哪些重建／质量指标；现有生成后端的内容代理是否与质量一致；原始输入的局部结构是否可读。三个问题使用不同材料与端点，不能将结果拼成同一个新机制的确认。

| 材料 | 规模与用途 | 数据边界 |
|---|---|---|
| 冻结官方DNS Ins20测试 | no-reverb、with-reverb各150份；同80k checkpoint的off/Router/t-cut配对 | 150个matched file-id的两条件音频，不把pooled300条当300个独立底层内容；不用于新方案/阈值选择 |
| VoiceBank自然内容pilot | 4句/2speaker×2noise×−18/0dB＝16格 | train侧探索；单ASR代理及选中A/B单人反馈，不是最终确认集 |
| VoiceBank候选余量及缓存读出 | 8句/4speaker×2noise×−18/0dB＝32格 | 句子在四条件重复；原增强结果冻结，后续线性拟合LOSO只隔离当前probe，不保证预训练未见 |
| P8-I自然配对及结构听感 | 12句/6speaker，47唯一位置，各Clean/Noisy，共94判断 | 发布者标称0–15dB，非局部实测极低SNR；同人探索，不是独立gold；noise录音身份仍unknown |
| CCC/5243既有诊断 | 受控VCV、单事件Noisy条件消融与人工观察 | 探索性/选择性案例；名义辅音或单识别器代理不等于独立实际发音真值 |

训练历史经用户明确：先训练300k前后端基座，随后冻结前后端，仅训练Router80k。比较同一80k文件的Router动作开关可控制基础权重；不能误写为主干仅训练80k。本文不以另一个300k文件名代替EMA、归一化和采样器一致性核验。

### 2. 冻结Router的客观评价

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

**事后配对不确定性补充**：随后仅从六份冻结逐句CSV，按150个file-id将两条件及所有系统共同重采样（PCG64 seed20261019、10000次、linear百分位）。14个pooled均值最大重现误差5.20×10⁻¹⁸；原指标／方法均不变。关键Router−off新95%区间为PESQ[−0.005429,+0.015219]、SI-SDR[+0.201401,+0.282815]dB、STOI[−0.001339,−0.000313]、SIG[−0.020976,−0.008731]、OVRL[−0.021592,−0.007543]。七指标×两对照的新旧区间全部并列，不按宽度或有利结果挑选；定性跨零方向未变。本补充保留了file-id跨条件关联，仍未解决speaker/noise/room层级依赖、训练seed差异或多重比较，也不是独立确认。[补充报告](../experiments/final_matched_file_uncertainty_20261019/experiment_report.md)

t-cut保留同类权衡，没有形成综合修复，本文将其作为固定控制而非有效的第二算法。完整三臂/七指标及区间见[原报告](../experiments/final_eval_full_20261018/experiment_report.md)，交互见[原始JSON](../experiments/final_eval_reverb_interaction_20261018/metrics.json)。原报告部分均值显示与JSON精确字段有细小不一致，本章表格以JSON逐字段转录，不回写历史报告或修改原指标。

### 3. 低SNR内容代理及候选余量

32格中Noisy/frontend/OFF的微WER分别64.24%/131.25%/79.51%；参考驱动的逐格二选一oracle为55.21%，相对最佳固定臂余量24.31个百分点。8句共72个参考词位置（token，不是72种不同词形），在四条件重复后计288，并非288个独立内容样本。

前端词错误更少的条件有23格，后端4格，5格相等。一个6词句的前端被ASR重复插入216词，不能推断音频真的多说216词。后验分解显示70个oracle错误差中62来自speaker间固定策略异质性，8来自speaker内进一步选择；speaker-best仍用参考，不是部署策略。0dB余量仅2.08pp；−18dB oracle微WER仍97.92%。这些结果约束候选池与评价叙事，不足以启动排序器并宣布真实内容恢复。

同32格的OFF−frontend质量变化为SI-SDR−1.3261dB、STOI+0.009965、PESQ+0.062594，32对均有效；−18dB的PESQ变化为−0.013394，0dB为+0.138581。质量代理与单ASR内容代理不能互相替代。[原指标与literal](../experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json) · [混杂解释](../research_memos/p8_prospective_headroom_interpretation_20261019.md)

较早16格pilot的微WER为66.03%/59.62%/80.77%，不能跨不同句子和两组总体均值进行机制归因。用户A例报告后端音节质量更好、未听到额外尾音；B例报告前端丢音与后端异常开头音节。它们是有参考的单人后验观察，不是总体发生率。[反馈原话](../experiments/p8_voicebank_native_content_pilot_20261019/attempt02/listening/user_feedback_20261019.json)

### 4. 条件干预、结构读出与人工参照

#### 4.1 条件敏感性不等于正确证据利用

VCV五例移除了直接Noisy条件中的已知语音，但原含语音前端锚点仍在；不是整个后端只有纯噪声。五例单人偏好为原后端1、相似4；其中相似不等于正确，另有/t/→/k/与/s/样尾音观察。条件冲突/OOD、非局部效应和残噪解释未排除。[原记录](../experiments/p8_vcv_speech_component_ablation_20261019/listener_assessment_20261019.md)

5243目标移除后CTC `/d/`后验升高，但用户报告声音稍弱；身份、模型概率及可听强度不是同一量。不能据此宣布Noisy必要性或当前系统从不补音。[最新修正](../experiments/p8_5243_local_condition_necessity_20261019/listener_assessment_20261019.md)

#### 4.2 原始Noisy结构观察

单听者94判断已完成：Noisy41结构/2非语音/4unknown，Clean45结构/0非语音/2unknown。按唯一位置配对，Clean判结构的45处对应Noisy40结构、1非语音、4unknown。用户补充部分unknown是80ms目标过快、未反应过来；因此不能把unknown全部解释为无声学信息。其“肯定回答约95%正确”是自信估计，不作为实测准确率。

Clean没有明确非语音类别，位置control不能充当negative，无法估计假阳性或真假事件双向判别。Noisy先于Clean并不消除同人记忆/研究期待偏差，且逐trial UI版本不可追溯。原答案全保留，不按后续结论重选窗或筛答卷，也不要求默认重听。[完整配对与风险](../experiments/p8_i1b_single_listener_materials_20261019/responses/completed_interpretation_20261019.md)

#### 4.3 固定线性结构读出

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

主运行16次双输出线性拟合与独立16次增广最小二乘复算通过；两者共享NumPy FFT、IO范围说明在报告。拟合明确记为probe学习，不能写零训练；0新增强网络/ASR/音频，预测越界保留不clip。结果不触发调lambda、扩hidden probe/verifier或CAEG。[完整报告](../experiments/p8_cached_structure_readability_20261019/experiment_report.md)

#### 4.4 训练期反事实响应正则：固定CPEL小试的负筛查

本节为后续完成实验的追加记录；第5节保留的“没有新增峰值显存”等说明指原批次材料，不覆盖本节单列的CPEL成本。CPEL候选尝试在不增加推理网络或扩散步数的前提下，训练后端响应原始Noisy中的局部变化。本轮仅作单seed、既有CV上的开发筛查：训练256条、开发64条，均覆盖四个标称SNR分箱。冻结前端与原生Fourier参数，训练现有Score；保留合法原生DSM，共同锚点／状态调用只作辅助回归，不称为原生DSM的无偏估计。M含反事实编辑与受控逐点辅助任务；P在完全匹配的输入、随机配方和调用预算上额外使用成对响应梯度。P以动态规则把当前preclip全局pair/M梯度范数比限制在1%以内；这不是Adam更新、累计轨迹或声学可靠性的预算。

| 终末P相对终末M的端点 | 保存结果 | 预定筛查判读 |
|---|---:|---|
| 未见窗型raw Tukey响应代理损失 | 下降1.67035% | 未达下降至少10%的候选门 |
| PCM16精度匹配Tukey响应代理损失 | 下降1.65827% | 同样未达10%；不是独立复验 |
| 正常未编辑64例：SI-SDR／STOI／PESQ差 | −0.00028658dB／+0.000004568／+0.00013913 | 几乎不变，不据此宣称等效或增强收益 |

每条文件内先平均两个parent响应，再对64文件等权；双parent及四种控制packet不是新增独立样本。M相对初始基座的raw／精度匹配代理已分别下降约11.35%／11.28%，不能归给pair项；普通DSM及普通增强A/B完整对照未实施，因此也不能声称超过普通数据增强。该损失主要由相位感知的正交响应误差支配，损失下降不等于恢复幅度增加，更不等于辅音身份恢复。自动clean高频事件不是自然前端吞音或音素金标。噪声实例控制仅37文件能量足够、27未评；编码相容不证明噪声纯度，不能写成FPR或拒绝假音成功。

两臂各500更新、2000训练Score调用与1000梯度遍历；M总419.4962秒、P总327.2104秒，合计746.7065秒，峰值显存约10.15GB（十进制）。M包含一次初始64例基线，P复用其绑定结果，故总耗时差不能作为训练加速证据；正常推理仍为原生N30，训练额外调用并不免费。唯一配置未达到增量门，已关闭，不提高权重、加步数、换窗或扩训救结果。这是有界负筛查，不是否定所有反事实训练，也不是确认创新、内容恢复或防幻觉机制。未来重提须先说明不同的可检验机制与自然内容终点。[终态执行与全量结果](../experiments/p8_cpel_mp_pilot_20261019/experiment_report.md) · [解释与退出计划](../research_memos/p8_cpel_mp_pilot_interpretation_20261020.md)

#### 4.5 M/P所检验问题的进一步限定

原构想中“反事实数据加普通DSM”的对照，与上述M并不相同：M还包含共同状态下的逐点clean回归。令每支白化残差为 \(r_h=\sigma q_h+(u-a c_h-(1-a)A)/\sigma\)，每支有 \(d\) 个复数坐标，\(\bar r=(r_1+r_0)/2\)。实际两层平均满足

\[
L_{\mathrm{aux}}=\frac{\|r_1\|^2+\|r_0\|^2}{4d}
=\frac{\|\bar r\|^2}{2d}+\frac{\|r_1-r_0\|^2}{8d},\qquad
r_1-r_0=\sigma(q_1-q_0)-\frac{a}{\sigma}(c_1-c_0).
\]

因此M的辅助项已经隐含精确差分幅度及形状监督，并非没有成对响应信息的普通DSM基线。P再加包络及正交项所测的是该强基线上的进一步增量。原1.67%／1.66%与未达10%门保持不变，不能外推为原构想中“普通DSM之外的方向性pair无效”。这是一项解释范围的纠正，不是改动旧实验、放宽退出门或证明新对照必然成功。

为回答缺失对照问题，补充N/D完全去除共同state逐点回归，只比较普通反事实native DSM与同DSM加原pair；其余编辑、500步、EMA及原梯度预算规则保持。两臂各500更新、64正常例及四控制packet完成，总耗时581.72秒，前端及固定Fourier参数保持不变。相同开发集再次使用仍仅探索，不是独立确认；预算分母从native+aux变为native也使跨四臂差异不能全部归因于数学冗余。[代数审查与范围](../research_memos/p8_cpel_question_fidelity_20261020.md) · [补充对照固定方案](../experiments/p8_cpel_native_pair_ablation_20261020/analysis_plan.md)

| 补充N/D端点 | D相对N | 判断 |
|---|---:|---|
| heldout raw Tukey pair | 下降3.48424% | 未达预定10%推进门 |
| heldout PCM16匹配pair | 下降3.47731% | 同样未达10% |
| 正常64例SI-SDR／STOI／PESQ | +.03957943dB／+.000127716／+.001483573 | 小幅正向，未确认实质内容收益 |

三个speech packet各64file双parent，零响应0、过冲0，raw/precision各仍有1个反向parent；投影gain略降、变化仍由orth形状项主导。D相对source的PESQ仍下降约.00527，不用对N变好掩盖它。noise37可评/strict30，另27能量不足未评；signed gain增加约.00012893、输出变化能量略降，未越+.02工程界不表示拒绝假音成功。两轮均未达各自固定门，关闭相应短程配置，但承认连续代理的小作用，不宣称普遍否定反事实训练。普通无编辑继续训练A仍未实施，真实辅音身份恢复、错误新增与独立确认亦未完成。[N/D完整报告](../experiments/p8_cpel_native_pair_ablation_20261020/experiment_report.md) · [独立复算后的解释](../research_memos/p8_cpel_native_pair_interpretation_20261020.md)

#### 4.6 CPEL正常评价路径与public部署入口的区别

后续接口审计原定8例，按固定停止规则在第7例expected再分析谱与实际public sampler anchor的数值检查略超容差后终止，第8例未读取；整体状态为failed。已执行7例的raw前端预测谱与实际public anchor相对L2差为.101–.245，归一输出相对L2差为.0689–1.0902；后者分母为raw输出能量，不是相同比例内容损伤。实际Noisy峰值参考全部相同，原raw MSE基线sanity7/7通过，未发现重复归一化。

这些partial观测足以限定M/P、N/D“正常评价”为未编辑输入、raw预测谱anchor下的原生OUVE/PC路径，不能直接称为与public波形入口等价的部署性能；它不否定两轮raw路径配对差的算术。两branch Noisy谱也有微小浮点差，不能把输出差全部因果归于STFT投影；public对clean MSE在5例低、2例高，仅作partial描述，不判优劣或内容恢复。整体8例验证未过、不改容差或补跑，当前源码亦不证明原300k历史训练接口。[失败partial完整报告](../experiments/p8_native_anchor_roundtrip_audit_20261020/experiment_report.md) · [评价范围解释](../research_memos/p8_native_anchor_roundtrip_interpretation_20261020.md)

#### 4.7 既有N/D终末权重的public部署迁移补充

接口差异实测后，补做同64开发例的source/N/D三臂真实public入口评价，未训练新权重或选择接口。三臂共享同前端物理波形、原Noisy及逐例seed，实际各64×N30，共5760次Score、209.65秒；source EMA仅物化一次，N/D严格注入已保存终末状态，3次canonical摘要与9次前端/固定W检查全部一致。192行四项指标均定义、无metric error，原始输入与actual anchor/condition跨臂相同。

| public评价 | source | N | D | D−N |
|---|---:|---:|---:|---:|
| SI-SDR dB | 6.28613773 | 6.32083907 | 6.32542843 | +.00458936 |
| STOI | .782429250 | .782647611 | .782647663 | +.0000000512 |
| PESQ | 1.89039844 | 1.88674326 | 1.88708595 | +.00034268 |

public D−N的归一MSE差另为−4.5080e−7；D−source虽SI-SDR+.0392907dB、STOI+.000218412，PESQ仍−.00331249。相对普通DSM对照的部署增量很小，没有建立实质综合增强或内容恢复。与旧raw D−N并列可见SI-SDR/STOI/PESQ增量更小，但MSE降幅略大，不能笼统声称全部按同一比例衰减或作接口因果归因。

这是相同已探索dev上的事后部署迁移补充，不是独立确认；没有总体显著性或等效/安全保证。旧raw与precision的10%门失败及原7partial审计failed均保持，不择优接口、权重或样本，也不据此扩大当前配方。结果只限制这一固定训练机制的实用支持强度，不普遍否定反事实学习或阻止其它科学问题继续研究。[完整public报告](../experiments/p8_cpel_public_transfer_20261020/experiment_report.md) · [独立复算后的解释](../research_memos/p8_cpel_public_transfer_interpretation_20261020.md)

### 5. 资源与可复现边界

Router为42577参数，相对约5512万非Router参数增量0.07725%；参数少不意味着延时零。普通N30/none-corrector路径静态Score评估30次，路由再加30次小Router。旧正式off(scale0)也执行Router，三臂job耗时958/956/957秒不能证明Router更快；t-cut计算后屏蔽动作，不节省小网络调用。future-grad训练额外一次2B Score重评分与输入梯度另计。[成本细节](../THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md#24-复杂度与计时口径区分训练lookahead和推理路由)

当前VoiceBank32全句A/B内容材料只复用104源文件，没有新模型输出。后续已收到单人12例参照辅助回答（6句/4speaker、−18dB8例/0dB4例），其余20例未评且不要求补齐；不算人类WER或正式12独立听者盲转写。frontend checkbox为missing6/preserved4/unknown2；OFF为missing4/different3/added1/preserved4/unknown1，标签多选不能当错误音素数。双方preserved对应同四个0dB案例，未出现frontend missing而OFF preserved的明确组合；missing较少不能解释为正确恢复，因为可能转成不同内容或新增。质量OFF3/frontend1/相似5/unknown3，OFF偏好均在内容双方保留的案例；内容和质量不相互抵消。

首例`p227_031_N2_snr_m18`的用户备注“A在句首The之前就生成了幻觉音素，B没有”，冻结A=OFF；原checkbox两臂都为missing。原备注和选项一并保留，不静默补added，也不将“幻觉音素”措辞当作精确身份的独立裁决。部分、同人、参照辅助及重复句子限制仍在，不外推总体胜率或用20个未评案例补分母。[材料报告](../experiments/p8_vb_cached_content_review_20261019/experiment_report.md) · [原反馈与解释](../experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md)

本章没有新增Speaker Similarity/EER、峰值显存、部署RTF或端到端真实音素恢复数字。旧研究存在相关试验不等于本批已覆盖这些终点；不得用其他数据/不同checkpoint的数字填空。没有新算法确认、独立身份与双向收益前，不把P8-I、CAEG或一般结构可读性写成完成贡献。所有原数据、checkpoint、冻结测试集与原始指标保持不变。

### 6. 结论与尚欠证据

现有证据支持的是：已实现Router在相匹配基座上的条件性重建／质量权衡；选定材料中的内容风险、条件敏感性与有限结构可读性。它们不共同推出“前端损失的真实弱辅音已由新机制恢复”。

后续若有可定位的内容改善与损伤观察，将与质量偏好分开报告；若没有收益或不可评价，如实收口该候选池，不用调ASR、换seed或扩分类器救结果。实际方法创新还需独立的内容／身份、错误新增控制、质量保真及计算预算证据。长期goal未完成，论文段落是交付，不是机制成功的替代。



# 第五部分：讨论与证据边界

来源：[P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md](../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)

## P8 论文讨论草稿：已观察现象与证据局限

配套写作：[相关工作与研究定位](../P8_RELATED_WORK_DRAFT_20261019.md) · [固定版本参考文献BibTeX](../P8_RELATED_WORK_REFERENCES_20261019.bib)。

> 状态：既有声学证据的写作整合，§1.6另引用随后完成的静态checkpoint比较；**不是新增强方法有效性或新颖性结论**。本次读取原始指标、已有独立验证回执和人工反馈；没有重新计算音频指标、读取波形、运行模型或收集新判断。研究入口仍为 [/data/research_plan.md](../../../research_plan.md)。较早的 [Router论文草稿](../THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md) 与本文件分开：其Router-on/off比较不能代替本文件的frontend/native-OFF比较。

### 1. 可用于“问题分析与讨论”的正文

#### 1.1 音质改善与内容保真应分别评价

在8句、4位说话人、两种噪声及−18/0 dB组成的32条件探索中，原生扩散后端相对判别式前端的配对平均变化为：SI-SDR −1.326 dB、STOI +0.009965、PESQ +0.062594，三项均有32个有效配对。这些结果呈现重建与质量代理之间的权衡，而非全面改善。分层后，−18 dB的PESQ变化为−0.013394，0 dB为+0.138581，也不能用全量均值宣布极低SNR音质提升。[原始指标](../experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json) · [报告与验证范围](../experiments/p8_vb_prospective_headroom_20261019/model_stage/experiment_report.md)

已有人工观察同样包含正反两面：在较早16条件pilot的A例中，用户认为前后端音节相近、后端音节质量明显更好，未听到额外尾部语音；在B例中，用户报告前端丢失音节，以及后端开头出现异常、难理解的音节。这是同一位听者对有参考、后验选中案例的观察，不是盲听总体结果；“噪声被当成语音”仍是待验证的机制解释。故质量提升、异常结构、具体音素正确性不能合并为一个指标。[人工反馈原话](../experiments/p8_voicebank_native_content_pilot_20261019/attempt02/listening/user_feedback_20261019.json)

#### 1.2 ASR候选余量不等于已获得真实内容恢复

32条件下，单一识别器给出的微WER分别为：Noisy 64.24%、frontend 131.25%、native OFF 79.51%；根据参考文本逐条件选择前端或后端的oracle为55.21%，相对最佳全局固定臂的差为24.31个百分点。这个oracle使用评价参考，不是可部署选择器，也不是新增强方法。288个评分参考词来自72个原始词在四条件下重复；样本仅4位说话人，bootstrap不能弥补其独立性与评价有效性限制。[逐条件与分层指标](../experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json)

总体数值还受到强烈混杂：前端在23条件的词错误更少，后端仅4条件更少，另5条件相同；一个6词句子的前端转写产生216个重复性插入错误，不能解释成音频真的多说了216个词。后验分解显示，70个oracle错误差中，62来自按说话人选择固定臂的异质性，8来自说话人内进一步选择。0 dB的相同代理余量仅2.08个百分点，−18 dB的oracle WER仍为97.92%。因此，这组结果不支持直接训练候选排序器并声称恢复了真实弱辅音，也不证明原始Noisy在信息论上不可辨识。异常值及原评分均保留，未通过删例或改解码参数修饰结果。[混杂解释与停止决定](../research_memos/p8_prospective_headroom_interpretation_20261019.md)

较早的4句、2位说话人、16条件pilot给出Noisy/frontend/OFF微WER为66.03%/59.62%/80.77%。它与32条件使用不同句子，不能合并成独立总体或用两次总体均值反转推断机制改变；两批都没有提供独立人工词错率或音素恢复率。[16条件原始指标](../experiments/p8_voicebank_native_content_pilot_20261019/attempt02/metrics.json)

#### 1.3 Noisy条件消融说明敏感性，不单独识别正确利用机制

已完成的VCV反事实诊断移除了**直接Noisy条件中的语音成分**，但保留由原始输入得到、仍携带语音信息的前端输出锚点。因此，它不是“整个后端只接收纯噪声”的实验。五例来自既有识别器变化的选择，同一位非盲听者报告1例原后端更接近参考、4例整体相似；其中一例两版本均出现/t/→/k/的判断，另一例消融后尾部更像/s/样结构。相似不表示正确或统计等效，摩擦样残留也不自动等于语义幻觉。[五例听感与原话索引](../experiments/p8_vcv_speech_component_ablation_20261019/listener_assessment_20261019.md)

在另一个5243局部消融中，移除目标条件语音后CTC的/d/后验由0.869617升至0.979222，而用户后来报告/d/相对稍弱。这不矛盾：类别后验、可听强度与身份正确是不同端点。现有证据至多支持所选干预下的条件敏感性；由于条件分布失配、锚点冲突和上下文路径尚未排除，不能据此认定后端完全忽略Noisy、或已正确利用音素特异证据。[最新人工修正与实验边界](../experiments/p8_5243_local_condition_necessity_20261019/listener_assessment_20261019.md)

#### 1.4 重建诊断约束假设，但不提供可部署修复

对32条件已存波形的离线分析发现，所有条件沿frontend到OFF的固定线段，其clean-aware MSE最优点都位于两端之间。共同oracle系数为0.2611，使按case等权、以clean功率归一的MSE由0.26391降到0.25622，约2.92%；完整OFF为0.31783。这是使用clean求得的样本内MSE空间，不是部署系数、听感收益或已验证的α控制器。修改能量中非全局gain部分的case平均占比98.40%，也不是“98.40%幻觉”。没有导出混合音频或重算其ASR/音质。[方向审计报告](../experiments/p8_vb_residual_direction_audit_20261019/experiment_report.md)

另一项旧DNS/Router四seed缓存分析中，seed离散度占pooled归一参考误差12.70%；剩余87.30%是有限四样本中心对clean的误差，不是已证实的真实生成器偏差。该缓存使用Router及旧输入处理，与本次VB原生OFF不同，而且缺少生成时原始WAV哈希，当前文件稳定性不能补成历史字节身份。此结果既不支持直接迁移到当前后端，也未验证均值音频的内容或音质。[四seed报告](../experiments/p8_cached_multiseed_variance_audit_20261019/experiment_report.md)

#### 1.5 P8-I技术准备后取得了受限人耳观察，完整判别仍未验证

用户新增的P8-I先问：对于明确的局部假设，原始Noisy是否有可验证判别信息？目前已完成I0的12对/6位说话人元数据登记、I1a配对/残差描述审计及I1b内存编辑完整性验证。I1a的12对最佳lag均为0，但这不证明残差是纯噪声。I1b保留24目标槽及96安全操作，不导出音频；其位置对照不是无语音负例，实际taper编辑能量未精确配平，48组相关比较的最大相对差82.78%。通过的端点是技术完整性，不是自然事件缺失或可辨识性。[I1a报告](../experiments/p8_i1_pair_residual_audit_20261019/experiment_report.md) · [I1b报告](../experiments/p8_i1b_event_pair_integrity_20261019/experiment_report.md)

发布者训练日志及后续README语义补证给出这12句标称SNR为0–15 dB，而非已经完成的−18 dB局部辨识任务；具体噪声录音/切片来源仍未知，标称SNR也不是80 ms局部SNR。原始日志审计中的SNR空值没有倒改，解释以独立补证为准。D0–D4、校准/弃权、真假事件双向CAEG与独立人耳结构评价仍无验证结果。原四人188判断仅为历史设计，后经用户授权改为一人94判断并已收口：Noisy41结构/2非语音/4unknown、Clean45/0/2；部分unknown来自定位反应困难，不能当无信息。Clean缺可靠非语音类，不能计假阳性或将同人参考称独立gold。[来源补证](../research_memos/p8_vb_train_log_semantics_20261019.json) · [实际反馈](../experiments/p8_i1b_single_listener_materials_20261019/responses/completed_interpretation_20261019.md)

后续VB32缓存线性读出已独立复算：full-Noisy比simple-Noisy两连续谱代理MSE下降17.28%/8.96%，但simple-frontend全量优于两full，未支持复杂表示的稳定内容优势。它不是原H1/H0任务，也不是具体辅音结果。[读出报告](../experiments/p8_cached_structure_readability_20261019/experiment_report.md)

更直接的12例单人参照辅助F/OFF观察中，双方同四个0dB例均preserved；−18dB仍有missing/different/added风险，未见F missing→OFF preserved的明确checkbox组合。OFF获3例质量偏好均在内容双方preserved例。首例备注OFF在The之前有额外声音，与原missing选项同时保留不改；没有精确身份标签或总体率。20例未评不催补、不中途选新样本救收益。[12例记录](../experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md)

#### 1.6 “80k原生OFF”不应被简化成只训练80k的StoRM

**用户后续明确确认**：“80k checkpoint 是在300k权重的基础上冻结前后端模型，只训练router的结果”。论文据此采用**300k基础前后端＋80k Router-only增量训练**的历史描述，不再把80k/300k当作两套基础模型各自的总训练步数。下面保留静态审计的独立证据与限制；用户说明补足训练历史，但没有额外指定初始化时RAW/EMA物化的具体操作。

随后完成的只读checkpoint比较进一步限定了基线身份：80k Router文件与本地300k文件的538项同名基础RAW tensor中，42项有效字节相同、496项不同；但300k保存的536项EMA与80k中被其内容覆盖的536项RAW tensor，在dtype、shape及有效字节SHA的**多重集合上完全一致**。其余两项Fourier投影W在两文件RAW中相同，当前源码将W设为不需梯度。这与先物化300k EMA、再冻结基础模型训练Router的解释一致，但仍不是历史初始化路径或参数名—EMA位置映射的证明。重复内容存在，不能把无名集合匹配直接当作命名模型或运行输出等价。[静态比较原始指标](../experiments/p8_checkpoint_base_identity_20261019/metrics.json) · [执行及限制](../experiments/p8_checkpoint_base_identity_20261019/experiment_report.md)

上述多重集合是Root对本次保存SHA字段的后验算术核对，没有再次读取checkpoint storage或运行模型。80k文件自己的EMA只有8项，当前代码在`rrsd_freeze_base`下仅更新/应用可训练Router参数的EMA；不能把“用了EMA”误写为“又对整个基础模型做了80k EMA”。因此，既不能由RAW差异推断模型损坏，也不能由文件名断言P8失败只是基础模型训练不足。论文继续称其为**指定80k Router checkpoint、关闭Router后的原生OFF**，不外推为所有StoRM模型的已证实瓶颈，不据此次比较自动重跑300k或改旧结果。

同一80k checkpoint的Router-on/off比较控制了基础前后端权重这一因素，是模块开关对照；仍须报告Router额外训练成本、score调用及推理成本，不能称matched-compute已自动成立。若另与独立300k文件比较，实际RAW/EMA、归一化、采样器、seed及数据条件还需一致，不能把用户历史说明扩张为所有运行输出等价。

**成本口径另有一个需保留的区别**：旧正式Router-on/off的off是`rrsd=true, scale=0`，当前路径仍计算Router；本P8的native OFF为`rrsd=false`，不调用Router。二者不能混为同一个延时基线。Future-gradient额外Score重评分只用于训练，正常路由推理不增加基础Score评估数；t-cut仅屏蔽动作，不提前省掉Router计算。静态计数、参数量及原始job计时的完整限定见 [旧论文草稿§2.4](../THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md#24-复杂度与计时口径区分训练lookahead和推理路由)，未补跑任何模型或冻结测试集。

### 2. 论文主张的可用与不可用版本

| 可以写 | 不能据此写 |
|---|---|
| 选定材料显示音质、重建和内容代理不一致，需要独立内容端点 | 已实现感知、保真、音素身份的全面改善 |
| 后端对某些Noisy条件干预敏感 | 后端必然忽略Noisy／必然正确使用Noisy |
| 人耳报告了具体异常结构及正向音质案例，保留单人非盲限制 | 已测得总体幻觉率或真实音素恢复率 |
| clean-aware oracle及缓存分解帮助排除简单解释 | 已获得可部署α策略或证实87.30%真实偏差 |
| P8-I完成受限管线的可复现准备 | 已证明Noisy有／无/k/、/s/身份信息，或CAEG有效 |
| 指定checkpoint的原生OFF存在所记录风险；基础权重呈EMA内容对应模式 | 仅凭80k文件名断言主干只训练80k、或仅凭无名EMA内容匹配宣布完整模型等价 |

**本节结论**：已有工作建立的是问题表征、代理评价的失效风险和候选解释边界。局部Noisy证据是否可读、后端是否能可靠利用它，以及新机制能否兼顾补真音、拒假音和音质，仍未验证。上述诊断不能单独确立算法新颖性或宣布长期科研目标完成。



# 第六部分：结论与展望

来源：[THESIS_CONCLUSION_DRAFT_20261020.md](../THESIS_CONCLUSION_DRAFT_20261020.md)

## 结论与展望：阶段性结果及未完成目标

> 本章总结已保存的实现、冻结结果与实际反馈，不新增实验或独立有效性结论。所有“改善”“失败”均限定于相应材料、端点和比较协议。

### 1. 主要工作与研究问题的回答

本文围绕两阶段生成式语音增强中的质量、重建与内容证据开展整理与分析。已完成工作包括冻结基座上的局部Router实现描述、匹配模块开关评价、P8内容风险与条件诊断，以及固定CPEL响应训练小试和短词读出适用性检查。这些工作说明了当前系统的条件性行为与评价限制，尚未形成独立新机制收益确认。

对于RQ1，Router呈现的是多目标权衡，并未获得稳健Pareto优势。基础训练历史为300k前后端基座与80k Router-only增量。同checkpoint动作开关控制了基础权重；正式两条件各150份按file-id匹配，不能作为300个独立内容。无混响中PESQ、STOI、SI-SDR有小幅正向变化，DNSMOS-SIG下降、OVRL原区间跨零；混响中SI-SDR改善与STOI及多项质量代理下降并存。固定t-cut未形成综合修复。事后保留跨条件配对的不确定性补充未改变定性方向，也未变成独立确认或训练seed层面的保证。[冻结评价](../experiments/final_eval_full_20261018/experiment_report.md) · [配对不确定性补充](../experiments/final_matched_file_uncertainty_20261019/experiment_report.md)

对于RQ2，P8表明质量、结构和内容代理必须分开解释。自然句观察同时保留后端音质更好的正案例与缺失、不同内容和额外声音的风险；单ASR候选余量受到异常重复插入及说话人策略异质性影响，不能变成真实弱音恢复空间。条件消融支持部分敏感性，却未排除锚点冲突、分布失配和上下文路径。缓存线性结构读出只预测连续谱目标，不能证明具体phone可辨或已恢复。[P8讨论与原始来源](../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)

94项结构判断来自单听者对47位置的Clean/Noisy观察：Noisy为41结构、2非语音、4未知，Clean为45、0、2。Clean缺可靠负类，同人记忆与期待偏差仍在，不能称独立gold或计算真假事件双向能力。另12个参照辅助内容case含重复句与条件，并非12个独立utterance；双方均保留的是同四个0dB案例，未见明确frontend missing到OFF preserved组合。质量偏好OFF3、frontend1、相似5、未知3，不能抵消内容错误；首例句首额外声音备注和原missing选项同时保留。其余20例未评，分母不补齐。两项观察均非人类WER、音素恢复率或总体幻觉率。[94项解释](../experiments/p8_i1b_single_listener_materials_20261019/responses/completed_interpretation_20261019.md) · [12例解释](../experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md)

对于RQ3，固定CPEL的M/P各500更新后，P相对M的heldout raw与精度匹配pair代理仅下降约1.67%与1.66%，未达到预定10%候选门；正常64例增强指标几乎不变。M自身约11%的代理下降包含编辑数据、受控逐点任务与继续训练的联合改变，各部分贡献未被分离，不能计为pair增量。损失主要受响应正交项支配，pair下降不等于恢复幅度增加，更不等于自然辅音身份恢复。工程损伤门未越界也不证明等效、安全或防幻觉成功。当前配置已关闭，不加步、加权或换窗救结果；单seed短程负筛查不否定所有反事实训练。[CPEL终态与局限](../research_memos/p8_cpel_mp_pilot_interpretation_20261020.md)

短词端点提供的是评价工具适用性负结果。固定64条Clean上tiny.en仅49/64匹配，即76.56%，低于至少58/64且每词6/8的门；down为4/8、no为5/8。样本分属8个filename prefix group，未经真人speaker身份验证。本项仅执行Clean读出，没有Noisy/F/OFF结果，不能证明真实发音错误、标签错误、Noisy物理不可辨或StoRM失败，也不能通过换prompt、识别器或样本救当前端点。[短词收口](../research_memos/p8_keyword_clean_readout_closeout_20261020.md)

### 2. 局限与总体结论

现有证据覆盖的是指定实现、checkpoint、材料和评价协议。重复内容、少量说话人、单人非独立观察、单识别器失效、探索性案例选择及有限训练seed，均限制外推；通过文档身份或转换检查也不能补足这些科学缺口。Router参数增量较小，不代表训练和推理免费；正式scale-zero off与完全bypass的native OFF不能混作延时基线。身份保持、部署效率与独立内容收益仍没有由本批结果充分确认。

因此，本文不能宣称已恢复真实弱phone、已实现可靠防幻觉或已完成感知、重建、内容、身份与效率的综合改善。当前没有独立新机制收益确认，长期科研目标尚未实现。负筛查的价值是约束当前主张和停止不值得扩大的配置，而非宣布所有生成增强、内容约束或反事实方法无效。已有七项固定版本文献也提示，保护观测、表征蒸馏和候选约束的一般思想已有先例；本文的工作清单不写成首次提出或显著成功的创新宣言。

### 3. 有条件的未来问题

未来仅在具备可靠、独立的输出内容读数时，才值得检验任务级可修复缺口：在未按增强结果挑选的新自然整句中，原始Noisy仍能被正确读出，而前端与native OFF是否重复损失相同内容。若成立，它只支持继续研究可修复缺口，不证明局部phone可辨或Noisy通道的因果定位；若OFF已经修复前端，则应保留恢复事实；若各臂均错或未知，也不能据此推断波形无信息。

这一问题目前仅作为条件性展望，尚未形成新的实验设计。既有94项与12例观察已经收口，不以追加听者判断或新建转写平台弥补当前证据，也不通过扩大CPEL、替换短词读出或重开缓存选择等路线追求有利结果。只有未来资源与区分机制的依据发生实质变化，才值得重新明确假设、对照与退出条件，并分别评价正确恢复、错误新增、质量和成本。学校模板、正式参考文献编译及浏览器排版验收仍待完成；结构补全和生成阅读稿不等于论文可提交或科研目标达成。[战略决定与边界](../research_memos/p8_post_cpel_strategy_20261020.md)


# 附录：原写作来源记录

以下是原草稿的来源校验／版本说明，不是新增实验或正文结果。

## P8_RELATED_WORK_DRAFT_20261019.md：原写作记录

### 6. 本轮版本补证与停止边界

- 既有方法审计曾读StuPASE v1。本轮补核v2后，保留dry-target与flow-matching的总体描述，但具体初始化须以v2 §2.2为准：教师来自预训练WavLM，学生来自DeWavLM；不能沿用v1“两者均从DeWavLM”的说法。配套BibTeX使用v2，不倒改旧审计所读版本。[v2方法](https://arxiv.org/html/2603.09234v2)
- 本节不搬用外部性能数字，也不声称训练数据量、预算或模型规模与本项目匹配。StuPASE的dry目标定义还允许原录音中的轻微自然反射，不能写成绝对无混响。[v2 §1、§3.1](https://arxiv.org/html/2603.09234v2)
- 这是有网络文献访问的写作工作，不能记作零网络；没有下载权重／语料、运行音频或模型、训练检测器、导出人耳材料或新增人类反馈。既有P8-I听者及播放授权未变化。
- 本轮到此完成相关工作段落与引文整合，不再为本节增加通用查新平台、解析器或无终点的先例扫描；科学下一步仍按当前计划与真实资源推进。
- 引文文件已作7个唯一键、必需字段与括号平衡的基础文本检查，文内本地链接已核存在。当前环境未发现BibTeX/biber，**未进行TeX编译验收**；正式排版时仍需在论文模板中编译。

**2026-10-20定向补证**：新增L3-SE v1正文方法与实验范围核读。该补证限制一般思想优先权，不证明本地机制的成功或所有具体方案无效；未运行其代码／音频／权重。[CPEL后战略决定](../research_memos/p8_post_cpel_strategy_20261020.md)。

## THESIS_EXPERIMENTS_CHAPTER_DRAFT_20261019.md：原写作记录

### 本次写作来源校验

只读既有JSON字段生成表格，未重算音频指标／bootstrap或拟合probe。表中均值差与matched交互已作字段算术一致性核对；这不替代历史模型/数据复现。

- `experiments/final_eval_full_20261018/metrics.json` SHA256 `15cf7076a0e233042e83e68ff429ff0b949e83ebcaf88297d1448d10b363daaa`
- `experiments/final_eval_reverb_interaction_20261018/metrics.json` SHA256 `f32d9834ad56e299d1659d781e0fac10bb43a8056945fe257d12c45de44abc8d`
- `experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json` SHA256 `577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04`
- `experiments/p8_cached_structure_readability_20261019/metrics.json` SHA256 `235a7b2ac8f15131d25d1f061319f2d8e1c815d00d53ad08e311192e7c67e913`

#### CPEL终态补充来源（2026-10-20）

本次仅转录已保存的终态报告与收口memo，不重算科学统计或执行模型。上述原写作来源记录全部保留。

- `experiments/p8_cpel_mp_pilot_20261019/metrics_M.json` SHA256 `e50f682208921de8cbe557a7781e14c46a4c9be1ec68133a3850bf16b71ac48a`
- `experiments/p8_cpel_mp_pilot_20261019/metrics_P.json` SHA256 `79ec90e6cc369d25b64a2b2f89ccf9877f2d75845175be16e1076513e15dedf2`
- `experiments/p8_cpel_mp_pilot_20261019/metrics.json` SHA256 `58c9bfbadd35416b0ca694244ce394689cd04610396aa9e4ea74246332e61767`

## P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md：原写作记录

### 3. 本次写作核对范围

§1.1–1.5只读检查原始JSON字段、报告、已有验证回执及反馈，没有重做原声学分析或执行验证器；通过回执仅按其覆盖范围引用，不称作本轮重新验收全部模型/波形。§1.6引用随后一次独立记录的静态checkpoint读取，并对其保存SHA字段作后验算术；只读复审也核对了该多重集合结论，未再次读取storage。以下原metrics SHA256实读匹配已有记录，写作前后保持不变：

- 32条件：`577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04`
- 方向诊断：`06603c8fef1a7ee7b5a7bed89b159f6f97036e06e7d04c241524190892fe3e17`
- 四seed缓存：`1ba98c0cd57911327412339baa6b1b46d1ecf96bb73a3a00c587af10d41c942e`
- I1b：`24c0ee1297e261dbecc4cfa9b2f3441292e45e71709cb2fddb0b60462362464c`
- 新静态checkpoint比较：`78d832dfa3123d1c27a4bc5daa7002521874d2b5d5f0947f1f913de4a9d07858`

下一科学动作仍按既有计划：资源及授权确认后开展固定原始材料的独立结构评价；不重复催问、不追加合同/来源扫描，不为填满研究流程而先训练编辑检测器。本稿服务论文写作，不代替所欠的内容与机制验证。


# 附录：固定版本参考文献

[原始 BibTeX](../P8_RELATED_WORK_REFERENCES_20261019.bib)；未作 BibTeX / TeX 编译。

```bibtex
% Fixed versions actually consulted; not a claim that every entry is a peer-reviewed publication.
% arXiv year/version refers to the cited preprint. No invented conference volume/pages.

@article{lemercier2023storm,
  author = {Lemercier, Jean-Marie and Richter, Julius and Welker, Simon and Gerkmann, Timo},
  title = {{StoRM}: A Diffusion-Based Stochastic Regeneration Model for Speech Enhancement and Dereverberation},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year = {2023},
  volume = {31},
  pages = {2724--2737},
  doi = {10.1109/TASLP.2023.3294692},
  url = {https://arxiv.org/abs/2212.11851}
}

@misc{wang2025galdse_v5,
  author = {Wang, Chengzhong and Gu, Jianjun and Yao, Dingding and Li, Junfeng and Yan, Yonghong},
  title = {{GALD-SE}: Guided Anisotropic Lightweight Diffusion for Efficient Speech Enhancement},
  year = {2025},
  eprint = {2409.15101v5},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  note = {Cited version v5, 22 January 2025; original arXiv submission was in 2024},
  url = {https://arxiv.org/abs/2409.15101v5}
}

@misc{rong2025pase_v1,
  author = {Rong, Xiaobin and Hu, Qinwen and Yesilbursa, Mansur and Wojcicki, Kamil and Lu, Jing},
  title = {{PASE}: Leveraging the Phonological Prior of {WavLM} for Low-Hallucination Generative Speech Enhancement},
  year = {2025},
  eprint = {2511.13300v1},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  note = {Fixed arXiv version v1},
  url = {https://arxiv.org/abs/2511.13300v1}
}

@misc{rong2026stupase_v2,
  author = {Rong, Xiaobin and Gao, Jun and Wang, Zheng and Yesilbursa, Mansur and Wojcicki, Kamil and Lu, Jing},
  title = {{StuPASE}: Towards Low-Hallucination Studio-Quality Generative Speech Enhancement},
  year = {2026},
  eprint = {2603.09234v2},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  note = {Cited version v2, 17 June 2026},
  url = {https://arxiv.org/abs/2603.09234v2}
}

@misc{chhaglani2025artifree_v1,
  author = {Chhaglani, Bhawana and Gao, Yang and Richter, Julius and Li, Xilin and Zadissa, Syavosh and Pruthi, Tarun and Lovitt, Andrew},
  title = {{ArtiFree}: Detecting and Reducing Generative Artifacts in Diffusion-based Speech Enhancement},
  year = {2025},
  eprint = {2509.19495v1},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  note = {Fixed arXiv version v1},
  url = {https://arxiv.org/abs/2509.19495v1}
}

@misc{shi2026grounded_v1,
  author = {Shi, Hao and Gao, Yuan and Ni, Zhaoheng and Peng, Junyi and Huang, Gongping and Tsao, Yu and Lu, Xugang},
  title = {Grounded Decoding for Autoregressive Speech Enhancement via Adaptive Code-Space Grounding and Local {LLM} Refinement},
  year = {2026},
  eprint = {2609.04245v1},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  note = {Fixed arXiv version v1},
  url = {https://arxiv.org/abs/2609.04245v1}
}

@misc{wang2026l3se_v1,
  author = {Wang, Zheng and Rong, Xiaobin and Su, Hang and Tan, Tianyi and Wu, Junnan and Fan, Lichun and Luo, Zhenbo and Luan, Jian and Lu, Jing},
  title = {Reducing Linguistic Hallucination in {LM}-Based Speech Enhancement via Noise-Invariant Acoustic-Semantic Distillation},
  year = {2026},
  eprint = {2605.08608v1},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  note = {Fixed arXiv version v1, 9 May 2026; preprint, no conference venue inferred},
  url = {https://arxiv.org/abs/2605.08608v1}
}
```
