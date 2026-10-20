# StoRM / SGMSE：研究总纲（Codex 入口）

> **唯一入口与路径约定**：当前入口为 `/data/research_plan.md`，旧大写根入口已删除，历史原文仅在 `research/archive/` 归档。项目根目录为 `/data/coding/sgmse/`；本文及分册中的 `experiments/...`、`research_memos/...` 均相对该项目根目录，Markdown 导航链接相对各文档所在位置。

> **文档用途**：这是给人和 Codex 首先阅读的短入口，不是实验流水账。更新来源为用户最新上传的 964 行研究计划 `research/archive/research_plan_full_20261019_044238.txt`，另保留我们已讨论的 P8-V 反事实声学证据引导。**拆分只改变文档组织，不构成实验授权、实验结果或论文新颖性证明。**

## 一句话目标

研究在**无混响、极低 SNR** 下，如何让 StoRM 的生成式后端在保留音质的同时，**补回有真实声学证据的弱音素，避免无证据的错误补音**。优先解决词句内容忠实性，而不是只优化自然度或客观音质分数。

## 文档导航：按任务读取，不要默认全读

| 什么时候读 | 文件 | 包含什么 |
|---|---|---|
| 每次开始、判断下一步 | **本文件** `research_plan.md` | 当前问题、最新决策、边界及索引 |
| 研究吞音、低 SNR、P8-C/Q | [research/p8_low_snr.md](research/p8_low_snr.md) | 机制假设、已有验证、下一轮证据门 |
| 验证Noisy局部有无判别信息 | [research/p8_noisy_identifiability.md](research/p8_noisy_identifiability.md) | 用户新增P8-I方案；先元数据/配对准备，D0–D4与CAEG未实施 |
| 讨论算法创新 P8-V | [research/p8_v_counterfactual.md](research/p8_v_counterfactual.md) | 候选级反事实声学验证的设计与查新风险 |
| 研究训练期反事实成对响应 | [research/p8_paired_evidence_training.md](research/p8_paired_evidence_training.md) | CPEL训练索引；M/P、N/D两轮结果与结论范围 |
| 需要具体数字/失败原因 | [research/evidence_summary.md](research/evidence_summary.md) | 近期关键实验、人工听感、代理限制、可追溯路径 |
| 找旧听感原话/试听 | [听感记录直达](research/evidence_summary.md#听感记录直达) | VCV原后端/去语音条件五例、5243局部消融、VB A/B，三组分开 |
| 写论文、区分已做与未证实 | [方法正文](../METHODS_RRSD_IMPLEMENTED_DRAFT_20261019.md)／[实验章节](../THESIS_EXPERIMENTS_CHAPTER_DRAFT_20261019.md)／[P8讨论](../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)／[相关工作](../P8_RELATED_WORK_DRAFT_20261019.md) | 现有实现、冻结结果与局限及固定版本BibTeX；不是新增强结果 |
| 连续阅读现有论文材料 | [整合阅读稿](../thesis/manuscript.html)／[说明](../thesis/README.md) | 导读、中英文摘要、绪论、相关工作、方法、实验、讨论、结论及冻结结果图；学校格式未定、非机制成功 |
| 开始混响 P9 | [research/p9_reverb.md](research/p9_reverb.md) | RIR/数据合成/重训的先决条件 |
| 旧 P7、Router、GAN 等 | [research/archive.md](research/archive.md) | 已暂停路线、旧计划、原文归档的位置 |
| 核对全部原话/哈希/流程 | `research/archive/research_plan_full_20261019_044238.txt` | **最新上传版逐字节保留**；只在需要时打开 |

阅读规则：一般任务先读**本文件 + 对应一个专题**。只在需要精确数据时读 `evidence_summary` 或单次 `experiments/.../experiment_report.md`；不要常规加载整个历史原文。**现有仓库的 `AGENTS.md`、`/goal`、实验冻结协议和权限约束仍独立有效，本摘要不能取代它们。**

> **当前执行状态：active**。用户已恢复并澄清：不得把缺学校模板或待人工评价扩展成整个科研目标的阻塞。模板仅影响最终排版；强内容主张的独立评价可以列待办，不禁止代码、数学、训练设计及有界客观实验继续推进。原[阻塞审计](research/current_blocker_audit_20261020.json)仅保留历史错误及撤回记录，不再作为停止依据。目标附件已按用户最新原文重读，SHA `269a3ba1f319601e1130da8da9ff2c91f0cc5e27d95b72948c9d52ca1195548b`；试听仅在必须等待且没有其它可并行任务时才可能阻塞。

## 当前问题和三条主线

1. **P8，第一优先级**：前端 `Y_pre=F(noisy)` 可能吞掉弱辅音；后端也可能使残留更圆润但未恢复真实音节，或在没有明确语音的位置补出异常结构。两种风险和后端的真实音质收益必须同时记录。
2. **P8-V，条件性创新候选**：对扩散模型提出的具体声学事件进行“存在 / 不存在 / 近音替代”的反事实证据比较，让原始 Noisy 既能**支持恢复**也能**反对错补**，允许 `unknown`。目前只是方法假设，**没有训练或验证成功**。
3. **P9，第二阶段**：混响目前缺可用训练 RIR，现有 80k 权重无明确混响训练记录；需要合法 RIR、合成训练配对、匹配预算重新训练后才讨论机制创新。

**当前决定：P8-CPEL v2唯一M/P小试已完成并关闭本配置**。[结果与退出](research/p8_paired_evidence_training.md)。两臂各500更新、64正常例；P对M的raw/精度匹配pair仅下降1.67%/1.66%，未达预定10%候选门，正常增强指标几乎不变。M本身已有约11%代理下降，不能计作pair收益；不提高权重/加步/换窗救结果，不立即启动普通增强A/B扩训。

**N/D补充对照也已完成，明确结论范围而非全局停工**：N为普通反事实native DSM，D额外原pair，均无共同点clean回归。各500步/64例，raw/精度匹配代理下降3.48%/3.48%，未达固定10%推进门；D−N正常SI-SDR+.03958dB、STOI+.000128、PESQ+.001484，仍是小幅变化，非内容恢复证明。按门关闭此固定配方，不扫权/窗/步数救场；旧M/P失败保持，但不再把强aux增量问题偷换为全部CPEL。全部训练结论统一看 [CPEL专题与两轮结果](research/p8_paired_evidence_training.md)。其它研究继续按新证据与决策价值推进，试听/模板不作为全局前置。

**新短词端点也已收口，不再轮换代理救场**：固定64个Clean的tiny.en读出仅49/64（76.56%），down4/8、no5/8，未达预定58/64且每词6/8。独立复算通过；技术取得成功不是端点通过。不进入Noisy/F/OFF，不改prompt/ASR、训KWS或去testing找通过。[结论与限定](../research_memos/p8_keyword_clean_readout_closeout_20261020.md)。此前完整包预算失败与source metadata失败保留，不因运输修正倒改。当前没有新训练/增强许可；需要不同且可证伪机制或可靠内容评价资源，不把单ASR不匹配说成真实发音错/物理不可辨，也不追加用户试听。

**接口审计已给出新的评价范围证据**：未发现重复peak归一化；但raw预测谱与public实际anchor在已测7例相对L2差约10%–25%，输出亦不同。原定8例审计在第7expected/public数值容差略超界后失败停止，第8未读，不补跑/改tol。CPEL旧正常指标因此限为raw谱接口评价，不直接称public部署性能；这不是归一化bug、吞音原因或新方法收益。[事实与限定](../research_memos/p8_native_anchor_roundtrip_interpretation_20261020.md)。后续研究从这种已测实现边界选择有决策价值的问题，不靠继续调CPEL门或增加试听。

**Public部署迁移评价已完成**：同全部64dev的source/N/D真实public入口三臂完成，0新训练。publicD−N的SI-SDR+.00459dB、STOI约+5e−8、PESQ+.000343；D相对source仍有PESQ−.00331代价。三臂权重注入及F/W、相同输入/seed核通过；原raw10%失败和audit7partial失败不改，不择优接口救场，当前固定pair配方到此收口、不扩训。[完整迁移结果](../research_memos/p8_cpel_public_transfer_interpretation_20261020.md)。这不普遍否定反事实学习，也不全局阻塞其它有证据的研究。

**当前仍聚焦CPEL，不转向Router**：用户指出从CPEL验证自行转向Router属于范围漂移，已停止该新增支线；其历史CSV复盘不是CPEL证据或当前下一步。保存JSON的解释性核查现已完成：尺度不变的Z=orth/response_energy均值下降约1.39%（precision1.36%），支持有限P加权方向改善，不只是总幅度收缩；不证明内容恢复、不改变旧10%失败或public微小增量，不追加训练。[CPEL解释与独立复核](../research_memos/p8_cpel_response_shape_interpretation_20261020.md)。

**当前下一步：普通训练对照组100k已单独授权，等待Worker启动**。完整10797条训练、冻结前端与Fourier参数，原配方/随机流/学习率不变；CPEL组须普通组完整100k与五快照通过后另签授权。唯一真实技术检查90.10s通过：同CUDA实例保存恢复重放精确一致（非跨进程/跨设备证明），4条完整30s public均长度/NFE/有限/归一化通过，单次后端9.71–9.78s。训练约22GPU小时双组外推，3558次最终完整推理按实测单次仅后端线性估约9.63小时，前端/I/O/评分额外；非工期保证。最终100k为主、所有五点训完统一评价，不以早期代理小停正常训练。所有输出仅/data，约154.9GBfree，不再清理权重/WAV。[技术报告](../experiments/p8_cpel_longrun_20261020/experiment_report.md)／[普通组单次授权](../experiments/p8_cpel_longrun_20261020/full_training_N_authorization.json)。

**空间扩容及两包清理已完成**：用户最终批准仅删除原始clean备份压缩包与失败partial下载，Worker核两精确路径后完成；释放约50.06GB，目前/data约156.7GB可用。所有权重、WAV、已合成数据和记录均保留，旧752项清单没有执行。原始clean外部备份由用户确认，未独立核验。[删除回执](research/storage_review_20261020/deletion_receipt_20261020.md)。

**文稿阶段已交付的范围与待输入项**：八份可编辑源稿已整合，中英文摘要、绪论、结论均已补齐；原实验/图表、14公式文本（含新增aux分解）和本地目录链接已核，未做TeX或真实浏览器验收。学校模板/章节规范已异步询问，尚未收到；不能据通用稿声称可提交。强内容结论仍缺独立输出读数，但不据此停止其它研究；不重复问听者或增加准备平台。长期目标未完成，也不因文稿交付收缩为单项写作成功。

## 截至本次上传，最重要的真实进展

- **P8候选余量测量已完成并关闭**：8句/4speaker、32条件唯一运行97.645s，独立验证通过。全量ASR代理余量24.31pp受单例216插入及speaker固定策略异质性支配；frontend赢23/OFF赢4/tie5，0dB余量仅2.08pp。OFF平均SI-SDR−1.326dB、STOI+0.009965、PESQ+0.062594，不是全面改善。**不升级selector训练/重排，不改ASR重跑**；先设计既有输出独立内容评估。详见项目根 `research_memos/p8_prospective_headroom_interpretation_20261019.md`，原始32格与既有旧实验均冻结；完整P8-V强门不降低。

- **单人参照辅助内容观察已按12例收口**：32例材料中已收到12例，余20例未评，不要求补齐或重听。双方在同四个0dB案例保留内容，尚无明确的frontend missing→OFF preserved组合；质量偏好OFF3／frontend1／相似5／unknown3。首例OFF句首额外声音的备注独立保留，不静默改标签。[反馈与限定](../experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md)。正式独立盲转写未实施，单人参照辅助观察不是人类WER、音素恢复率或新方法确认，也不阻塞科研。

- **无需听者的机制审查已推进**：条件Score不自动等于独立声学似然。静态checkpoint字节证据补核保存幅度指数0.5/factor0.15，likelihood不可在压缩谱直接假装noise加性；共享标量variance积分与最小平方残差同排序，不包装成新verifier。DPS/显式noise-prior先例边界仍有效，未批准新调用/训练；项目根 `research_memos/{p8_score_vs_evidence_audit_20261019.md,p8_noise_evidence_contract_20261019.md}`。

- **方向—幅度CPU审计完成并关闭**：全部32例沿现存修改方向的小步可降低MSE，但31/32全量OFF的MSE比F高；共同clean-oracle alpha0.2611只给约2.92%归一MSE空间，非部署系数。修改非纯gain能量占比case均值98.40%，不等于幻觉量。独立重建/497原件保护通过，零新波形/ASR/模型/质量重算；不据此训练α门控或重开anchor路线。解释见项目根 `research_memos/p8_residual_direction_interpretation_20261019.md`，需区分偏差修复与生成方差，不能拿MSE替代内容/质量。

- **旧四seed缓存诊断已完成并关闭**：16句旧Router缓存的seed离散度占pooled归一参考误差12.70%，有限四样本中心误差占87.30%，后者不是已证实true bias。80当前WAV/160实际解码与独立pairwise核验通过；原生成WAV hash缺口仍在，且不是VB32原生OFF，不能直接迁移结论。不导出均值音频/追加seed/复活selector；下一步先审训练与推理读出估计的对象，避免把确定性输出或单次Tweedie读出错叫E[S|X]。项目根 `research_memos/p8_cached_variance_interpretation_20261019.md`。

- **估计对象审查已收口，未升级新实验**：逆压缩与条件均值不交换的数学差别成立，但native PC返回随机轨迹末跳的离散转移均值，不是精确clean Tweedie均值。不能据此宣称发现吞音原因或收益；二阶Tweedie/Jensen已有先例，不为通用公式建立新工具链。暂不改读出/采样器或跑Jacobian/训练；[具体审查与退出条件](../research_memos/p8_estimand_readout_audit_20261019.md)。

- **条件利用先例补核（非实验）**：DOSE替换的是中间x_t，不是删除Noisy；不能只改当前DSM输入而保留原噪声监督就称合法score。条件敏感也不能排除“利用不足”，已有多层传递/state-drop/预测引导需作先例，不凭旧消融重开训练。[具体更正与方法门](../research_memos/p8_condition_utilization_priorart_20261019.md)。

- **基线训练历史已由用户确认**：80k checkpoint是在300k基础权重上冻结前后端、仅训练Router所得；应写“300k基础系统＋80k Router增量”，不是主干只训练80k。同一checkpoint的Router-on/off为冻结基础系统上的模块开关对照，额外训练/推理成本另列。静态权重比较与EMA物化相容，但本句用户说明不指定RAW/EMA初始化细节，也不替代跨文件运行等价核验。[来源与限定](../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md#16-80k原生off不应被简化成只训练80k的storm)；不因此重跑300k或改变P8-I授权。

- **人耳反馈的 A/B 双面性**：A=`p287_054/N1/0dB`，用户没有听到前后端额外语音，后端音节质量明显更好；B=`p226_028/N2/-18dB`，用户听到前端丢音节、后端异常音节，尤其开头。不能只挑 B 讲“后端总会幻觉”，也不能只看 A 宣称内容全部正确。
- **B 开头既有音频复核已完成并关闭**：前 0.5 s 后端相对前端 RMS 约 **+17.04 dB**，但增能、残噪、前响、先验补全和真实音素身份不能互相替代；不开新窗口/seed/ASR 继续搜索此例。
- **VoiceBank 自然匹配 pilot 已完成**：4 句×2 噪声×2 SNR，共 16 条；相对发布者原文，noisy / `Y_pre` / Router-off 微 WER 约 **66.03% / 59.62% / 80.77%**。单一识别器在 -18 dB 已接近失败区；四句重复、仅两位说话人，不是独立样本总体或人耳 WER。A 的人工反馈也否定了凭单例 ASR 错误推断末段有幻觉的简单解释。
- **VCV/Q 与 P8-C**：Noisy 直接条件分支确实影响输出；但已有条件互换/语音移除包含 OOD 或锚点冲突，不能证明它能正确“否决 `/s/`”。新说话人复验**没有复现**原先 Router 的辅音身份净收益；名义 -6 dB VCV 的语音支撑区平均 SNR 约 +0.87 dB，不能冒充极低 SNR 音素证据。
- **5243 的 `/d/`**：用户报告过后端补音的正候选，但目标 Noisy 语音成分移除后，CTC `/d/` 后验反而升高，用户听到的目标版 `/d/` 却稍弱。音素身份、强度、CTC 后验是三种不同指标，预注册的“CTC 必要性”方向**未获支持**。
- **老路线阶段性结论**：P7 的局部风险线性 probe、若干 Router/Anchor/感知 loss/多轨迹/adapter 方案尚无可部署的独立内容恢复证据；不要因代码能跑就重开训练。官方 DNS `test_set/synthetic` 的 no-reverb/with-reverb 各 150 对已冻结，不能用于选择方法或阈值。

详细数字、来源、边界见 [证据摘要](research/evidence_summary.md)。

## 当前科学问题（区分三个层次）

1. `noisy` 在物理上包含 clean 语音分量吗？加性混合通常**有**。
2. 观察到的 `noisy` 是否有**足以区分** `/k/`、`/g/`、`/s/` 或缺失的证据？**未知，需受控正反例。**
3. 后端是否利用了音素特异证据，还是由 `Y_pre` 锚点、上下文先验或条件冲突影响输出？**部分条件敏感性成立，正确证据利用机制尚未证实。**

绝不能从 1 推出 2，或从“生成了辅音结构”推出 3。ASR/CTC、能量提升、MOS 与人类真实内容判断须分开。

## 当前建议的工作顺序（均需另行确认是否实施）

**当前P8-I：技术准备与94项单人结构观察均已收口**：[状态与证据](research/p8_noisy_identifiability.md#6-最小执行次序与预算)。I0/I1a/I1b独立验证完成；24目标/96内存操作safe但编辑能量未配平（差max82.78%），不等于可辨识，保持原gain1/zero-lag、不扫参数。94份原始PCM裁窗完成独立核验；用户答卷已通过聊天提交归档，未改写制作时指标，也不声称自动上传或真实浏览器专项测试通过。D0–D4/CAEG仍未验证，未因此训练新检测器或修改StoRM/OUVE。

**单听者后续94题已全部提交并收口**：Noisy41结构／2非语音／4不确定，Clean45结构／0非语音／2不确定；Clean结构45处对应Noisy40结构、1非语音、4unknown。[完整反馈与限定](../experiments/p8_i1b_single_listener_materials_20261019/responses/completed_interpretation_20261019.md)。仅同人自然结构观察，非独立gold或音素结果；Clean缺负类，不据此评假阳性/真假双向，不重选control或要求重听。黄色区理解风险保留，原生成指标不倒改；不自动批准CAEG，缓存probe独立推进。

**来源补证改变了材料范围判断**：[官方训练日志](../research_memos/p8_vb_train_noise_provenance_interpretation_20261019.md)补到12句噪声类别与发布者标称SNR（0–15dB，不是已完成的极低SNR试验）；录音/切片身份仍unknown。同speaker或同日志类别图只有10+2两个分量，不能照搬三方联合隔离草案。不换旧12句或自动开训，原审计与README后续语义补证分开保存。

**停止过度准备原则继续有效**：不再给已收口池追加合同、校验器、来源扫描或试听数量；同人记忆/期待偏差保留，不构成独立参考、跨听者一致性或总体准确率，原188判断方案不计作实施。强内容主张仍须真实证据，有限观察不能替代长期目标；无依赖的论文整合与研究继续推进。

**并行缓存结构读出已完成并关闭**：[完整结果](../experiments/p8_cached_structure_readability_20261019/experiment_report.md)。32格/4speaker、11,396帧、16次固定线性拟合，主0.475s/独立5.375s复算通过；无新增强/ASR/音频。full Noisy胜其simple基线17.28%/8.96%，但simple frontend全量MSE比两种full更低，full frontend在四位speaker均逊simple；不支持凭复杂表示宣称前端丢失内容。仅连续代理可读性，不能当音素/H1-H0/CAEG收益；不重调特征/lambda或自动升级hidden probe/verifier/增强训练，原32格与听感记录继续保存。

**直接内容观察的材料与反馈分开归档**：否决CCC旧输入分类后，原VB32输出全量制作单人参照辅助A/B材料，区别于原始结构听感。128入口/104源文件独立核验通过；[制作记录](../experiments/p8_vb_cached_content_review_20261019/experiment_report.md)保留制作时0回答，后续12例反馈另存如下。不挑ASR输赢、不重编码，原增强/ASR实验仍冻结。

**后续已收到12例并收口**：[反馈解释](../experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md)。6句/4speaker、8例−18dB/4例0dB，双方同四个0dB例preserved，尚无F missing→OFF preserved明确组合；质量OFF3/F1/相似5/U3，不能替代内容。首例OFF句首额外声音的备注与missing原选项同时保留，不自动补改标签。余20未评、不催补或重听；单人案例不外推WER/phone/总体胜率，不据此改seed/ASR或扩训。

用户反馈新试听例子偏多：32例仅为全量材料，不再默认期待全部完成；可只做2–4例或停止，部分回答只作实际案例观察，剩余未评。内容验证重要但本项单人全量不是goal前置门，不继续以任务数量增加用户负担。

**论文统计收尾，不是新方法结果**：[150matched-file不确定性补充](../experiments/final_matched_file_uncertainty_20261019/experiment_report.md)已从冻结CSV完成并独立复核全部10000×14draw。保留两条件配对后原均值不变、定性CI方向未变，新旧区间并列；修正pooled独立抽条口径，不借结果择方法，不扩展为speaker/noise/roomcluster、多重校正或独立确认。论文已同步，0音频/模型/训练、旧结果不覆盖。

1. **保持已完成实验冻结**：B 开头、16 格 VoiceBank pilot、反复分析的 VCV 五例、5243 局部消融均停止追加搜索。不会因本文档重组触发重跑。
   **下一研究工作先完成对照问题保真审查**：旧M/P关闭保持，但M额外包含强共同state逐点监督，不能把该负筛查替代原普通DSM→pair问题。N/D缺失对照已完成并未达固定推进门，后续不盲扫该配方；自然内容确认列待办，不增加试听/平台准备，也不以模板或单ASR失败全局阻塞。
2. **仅在有独立材料和新批准时**，做带有可信辅音身份的固定噪声正控制：同噪声更换真实辅音，辅音不变更换噪声，关注“输入是否可辨 / 输出是否跟随”，不把受控 VCV 等同真实 DNS 极低 SNR。
3. 若证据支持存在可修复但未修复的真实音素，分别考虑 **P8-N**（合法 noisy-centered 模式互补）和 **P8-R**（前端过抑制定位后定向利用 noisy）；二者都尚未启动。P8-V 是独立的**候选级证据审查**机制，可先作方法与混杂审计，不自动以旧 Score 修改 OUVE。
4. **P9 不抢占 P8**：先获得合规真实 RIR 和新训练/验证 split，确定 wet/dry 目标，匹配预算重新训练；当前不下载或开训。

## 不得越过的边界

- **本次拆分只有文件整理**，未批准任何模型推理、数据下载、训练、实验修改、测试集调参或源码变更。
- 训练、验证和最终测试分开；已反复筛选的 8 条 DNS/181 窗口、VCV 五例、VoiceBank 16 格是探索材料，不是新方法的独立确认集。
- 推理时不能使用 clean、真实噪声分量、真实局部 SNR 或人工 oracle 标签；这类信息只允许训练/离线机制分析。
- StoRM 的 Score 实际同时接收 `[x_t,noisy,Y_pre]`，OUVE 以 `Y_pre` 为中心；不能声称后端“看不到 Noisy”，也不能在冻结权重上把中心随意改为 Noisy 就宣称合法第二模式。
- **优先真实语音内容**：同时报告真实音素恢复率、错误新增、`unknown`、人耳可懂度、音质与保真及预算；不允许通过删除所有低能量高频来获得虚假的“防幻觉改进”。
- **论文新颖性尚未证实**：GALD-SE、PASE、Grounded Decoding、噪声引导及一般 measurement consistency 已覆盖部分空间；P8-V 的潜在差别是**候选级、有符号、双向、可拒绝的声学证据**。

## 维护约定

本文件只更新：**当前结论、任务状态、优先级、决策变更、专题链接**；目标保持 100–150 行。每次实验的命令、哈希、失败重试、长表和完整审计放进各自实验报告/对应专题，不再在总纲顶部追加几十段日志。若新结果改变旧判断，先在 `evidence_summary` 写清“旧结论 → 新证据 → 新结论”，再改本总纲；禁止无说明地覆盖历史。原始上传版及我们此前含 P8-V 的整合版均已保留于 `research/archive/`。
