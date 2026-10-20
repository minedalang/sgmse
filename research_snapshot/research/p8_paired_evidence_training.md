# P8-CPEL：反事实成对声学证据训练

> **最新：两轮训练及source/N/D全64例真实public部署评价均完成。N/D代理增量约3.48%未达10%门，public D−N的SI-SDR仅+.00459dB、STOI近零、PESQ+.000343；未建立实质综合部署或内容收益。当前配方不扩大，不普遍否定CPEL，也不全局阻塞其它研究。**

> **用户提出正式训练预算后的更新**：500步不能证明训练充分；用户说明既往创新常需100k步起才显效，故2000步准备不启动、仅保留未执行工件。当前审查足量训练数据、同预算普通训练与CPEL长程对照，以及可恢复断点/存储；尚未批准100k运行。旧500步数值和10%门不改，但不能据此普遍否定CPEL或认定2000已足够。[预算审查及最新状态](../../research_memos/p8_cpel_training_budget_audit_20261020.md)。

## 先看白话版：两轮到底比较什么

两轮都从同一个已有模型开始，只短程微调后端，不是四个新网络。每个训练组都做500次参数更新，前端保持冻结。技术编号只用于对上原始文件，不代表额外方法。

| 训练轮次 | 不加CPEL的对照组 | 加CPEL的实验组 | 这轮回答什么 |
|---|---|---|---|
| 第一轮（技术编号M/P） | 普通去噪＋额外的共同状态干净语音重建任务 | 左边全部训练，再加CPEL成对约束 | 在已经较强、且部分重叠的重建监督上，CPEL还能多带来多少作用？ |
| 第二轮（技术编号N/D） | 使用同样的反事实编辑数据，只做普通去噪训练；去掉上述额外重建任务 | 左边相同训练，再加CPEL成对约束 | CPEL是否比普通反事实数据增强多带来作用？更贴近原始问题。 |

**结果怎么理解：** 第一轮的响应代理改善约1.67%，第二轮约3.48%，均没达到事先规定的10%推进门。这里是专门的响应检查分数，不是音质、内容正确率或人耳可懂度提高了这些百分比。后续真实部署评价也只看到很小的客观变化，没有建立实质综合增强或内容收益。

**也不是完全没有作用：** 既有记录支持有限的响应方向对齐改善，但不足以支持扩大当前配方。普通数据增强本身相对“不编辑、只继续训练”的收益尚未被单独分离；没有真实辅音恢复、防幻觉或所有CPEL都无效的结论。

下面再列技术细节、精确数值和原始报告，阅读概况时不必先记住这些编号。

## 1. 这是什么、与P8-V如何区分

CPEL是用户提出的**训练期**方法：同背景noise下，从clean和Noisy同时削弱局部声学成分，固定共同state/前端条件，训练后端对Noisy变化产生正确方向响应；保留合法native DSM，不增加部署网络/NFE。

[P8-V专题](p8_v_counterfactual.md)则是**推理期候选验证/引导**：后端提出候选，再审查Noisy是否支持该候选。两者不是一个已验证算法，不相互搬用成功或许可。

自动高频事件不等于辅音身份；压缩谱差分不是波形Delta的压缩；共同state仍含特权结构，不能当部署信息论证据。条件敏感性不等于正确恢复，pair相消的共同错误也不能被它自动抑制。

## 2. 当前结果总表：不要混淆对照问题

| 对照 | 不加pair的基线 | pair相对基线的heldout raw/precision下降 | 判定 |
|---|---|---:|---|
| M/P（原小试） | native DSM＋共同state逐点clean回归 | 1.67035%／1.65827% | 未达固定10%门，关闭此配置 |
| N/D（缺失对照补充） | 普通反事实两端native DSM，无共同点回归 | 3.48424%／3.47731% | 未达固定10%门，关闭此配置 |

两轮均single seed，256train/64既有CV，三个speech packet各64file×2parent；file内先平均parent、再file等权，不能按128独立样本推总体。CV复用只探索、不是独立确认；raw/precision也是同案例反证，不是独立复验。

### 最新N/D正常未编辑结果（raw谱接口，不等同public部署）

**评价路径补证**：后续固定8例接口审计在第7例数值容差失败后停止（第8未执行）；已测7例raw预测谱与public实际anchor相对L2差10.11%–24.52%，对应输出亦不同。归一化参考一致、旧raw基线sanity7/7通过。因此本页M/P、N/D“正常评价”指未编辑输入、raw前端预测谱anchor下的原生OUVE/PC路径，不能直接写成public波形部署性能。整体审计仍failed，不能偷改为8例通过，也不把差异当吞音原因或收益。[完整审计](../../experiments/p8_native_anchor_roundtrip_audit_20261020/experiment_report.md)／[解释](../../research_memos/p8_native_anchor_roundtrip_interpretation_20261020.md)。


| 指标 | N | D | D−N |
|---|---:|---:|---:|
| PESQ | 1.89310567 | 1.89458924 | +.00148357 |
| STOI | .794341813 | .794469529 | +.000127716 |
| SI-SDR dB | 5.87682765 | 5.91640708 | +.03957943 |

四项正常指标均64/64定义；MSE另见原JSON。D相对N小幅正向，工程损伤门未越界，但PESQ仍低于source约.00527；不称实质内容收益、感知等效或安全保证。未运行新ASR/phone/人耳评价。

raw/precision投影gain均略降，未变成更大恢复幅度；zero0/128、reverse1/128、overgain0/128。主要变化仍为orth形状项，不是首次激活Noisy或正确恢复弱辅音。

noise固定64中37可评、27同位置匹配能量不足；37中strict30。D−N signed gain约+.00012893，输出变化能量略降；signed可抵消。编码相容不证明纯noise、未评不计零/通过，不报FPR或防幻觉率。

## 3. 最新public部署迁移补充（事后探索，非独立确认）

实测接口不等价后，一次固定source/N/D全64例真实public enhance已完成，0新训练/权重选择。实际192backend/5760Score、209.65秒，3次state严格注入摘要匹配、F/W9次检查均不变，原始输入/anchor/Noisy/seed跨3臂一致，四项指标各64/64定义。

| public指标 | source | N | D | D−N |
|---|---:|---:|---:|---:|
| SI-SDR dB | 6.28613773 | 6.32083907 | 6.32542843 | +.00458936 |
| STOI | .782429250 | .782647611 | .782647663 | +.0000000512 |
| PESQ | 1.89039844 | 1.88674326 | 1.88708595 | +.00034268 |

相对source，D的SI-SDR+.0392907dB、STOI+.000218412，但PESQ−.00331249；不省略相对基座代价。public D−N 的SI-SDR、STOI和PESQ增量比原raw更小，但归一MSE降幅略大；不能笼统说所有指标按同一方向衰减。不据此做纯anchor因果或总体显著性主张，未建立实质综合部署/内容收益。旧raw10%失败和7partial audit失败保持，不因public表而重新判过门或择优接口。

[public完整实验](../../experiments/p8_cpel_public_transfer_20261020/experiment_report.md)／[含source代价和raw并列表的解释](../../research_memos/p8_cpel_public_transfer_interpretation_20261020.md)。本固定pair配方到此验证收口，不扩训/扫参，仍不普遍否定CPEL或全局阻塞其它研究。

### 进一步解释：小幅作用不只是总幅度缩小

仅从已有N/D JSON复算的Z=orth/response_energy，对响应正幅度缩放不变。raw／precision的file等权Z均值下降约1.3902%／1.3643%；127个两臂正gain的parent中122下降、5上升，另1个仍负gain，无符号翻转。独立逐项复核通过；这支持有限的P加权响应方向对齐改善，但不代表完整向量或真实内容恢复。Z百分比不是cos或音素准确率百分比，也不是3.48%收益的因果分解；相关parent不当作独立成功样本。旧10%失败和public未建立实质收益保持，不因该事后解释扩训。[公式与限定](../../research_memos/p8_cpel_response_shape_interpretation_20261020.md)。

## 4. 为什么补N/D，而不是重跑旧M/P

原用户B是普通反事实DSM。实际M还加共同state逐点clean回归，精确分解已经隐含gain1差分及orth监督。因此旧M/P只检验该强基线上的进一步pair增量，不能替代原B→C问题。该范围纠正不倒改旧门/旧结果，也不保证新对照成功。

[精确代数与问题保真审查](../../research_memos/p8_cpel_question_fidelity_20261020.md)。N/D完全删除common_point调用，GM仅native；两臂同4Score/2grad/500步/原编辑、margin、EMA及1%规则。N计算pair梯度但应用0，D实际应用；native与pair图支持不符/非有限即失败。共同state只服务pair，不能用native+0*aux假装删除。

预算分母由旧native+aux变为native-only，故两轮增量差不能全归因数学冗余。M自身约11%代理下降属于编辑/继续训练/共同任务的联合变化，也不能作单因素aux因果。无编辑原DSM继续训练A仍未做，增强数据本身收益未被分离。

## 5. 执行完整性与成本

N/D实际各2000训练Score、1000梯度遍历、500optimizer；F及原生Fourier W前后摘要不变。两臂合计581.72秒；显式复用绑定旧source initial64，无初始推理重算，不复用旧终末M/P权重。来源provenance保留旧identity、旧code/config、registry/checkpoint与core历史保护；附加SDE/sampler当前锁不冒充完整历史保证。

D的preclip pair/native向量范数预算范围约.0007434–.01，中位.01；不是Adam逐参数更新或累计轨迹界，不是声学可信度权重。源码/原数据/旧答卷不改。部署N30不变，但训练额外调用并不免费。旧M/P746.71秒含一次额外initial评价，不能直接作训练速度比较。

## 6. 完整资料入口

- **最新N/D完整实验**：[目录与报告](../../experiments/p8_cpel_native_pair_ablation_20261020/experiment_report.md)；[固定方案](../../experiments/p8_cpel_native_pair_ablation_20261020/analysis_plan.md)；同目录`config.yaml`、`metrics_N.json`、`metrics_D.json`及汇总`metrics.json`（报告/汇总完成状态以目录实际为准）。
- **最新N/D科学解释**：[独立核算后的结论与局限](../../research_memos/p8_cpel_native_pair_interpretation_20261020.md)。
- **旧M/P完整实验**：[报告](../../experiments/p8_cpel_mp_pilot_20261019/experiment_report.md)；[解释](../../research_memos/p8_cpel_mp_pilot_interpretation_20261020.md)，原结果不覆盖。
- **原构想、OUVE数学与先例**：[独立分析](../../research_memos/p8_counterfactual_paired_training_assessment_20261019.md)。
- **配对编辑/量化风险**：[编码与精度控制](../../research_memos/p8_cpel_quantization_and_artifact_controls_20261019.md)。
- **前期软件与接口证据**：[合成原型](../../experiments/p8_cpel_objective_prototype_20261019/experiment_report.md)、[真实接口attempt03](../../experiments/p8_cpel_mp_pilot_20261019/preflight_attempt03/preflight/experiment_report.md)、[训练内尺度诊断](../../experiments/p8_cpel_train_scale_calibration_20261019/experiment_report.md)、[64dev编码诊断](../../experiments/p8_cpel_codec_consistency_20261019/experiment_report.md)。软件/编码通过不作声学有效性证明。

## 7. 决策与后续

保留小正向作用，但该固定phase-aware正则不足以支持扩大训练。不提高权重、加步、换窗/seed搜过门，不声称CPEL普遍无效。真实内容、身份、新增错误、质量/效率综合目标保持，连续谱代理不能替代它们；强内容评价可列待办，不把模板、单ASR或未确认听者变为全局阻塞。

下一研究须针对新证据和可部署瓶颈提出不同且可证伪的问题，不能只围绕同64例优化代理；当前无新增后续训练许可。方法新颖性仍须区分DOSE、Corrective Forcing、CADT及一般导数监督，不凭工作名称或有限差分公式宣称原创。
