# 硕士论文创新点与结果草稿

日期：2026-10-20  
依据：真实 StoRM/SGMSE 代码、80k checkpoint、冻结 DNS Ins20 官方测试集及已归档实验

## 1. 建议的论文定位

当前证据支持一条可靠的主线：**Future guidance 使判别式前端与生成式后端形成局部、时间相关的协作，但这种协作带来条件性的重建收益，并伴随感知代理代价；代价在官方混响条件下显著放大。**

论文应把贡献写成“机制加验证边界”，不把 Router 描述成所有指标都提升的通用增强器。

如果学校要求列出两个创新点，建议采用：

1. **算法创新：Future-gradient reliability routing。** Router 根据当前条件输出时频路由系数，使用冻结扩散后端的一步 future re-score 学习路由行为。
2. **分析/评估创新：条件化 Pareto 诊断协议。** 通过 matched no-reverb/with-reverb 评测、逐文件交互项、headroom Oracle 和系统性反证实验，识别“重建收益—感知代价—混响交互”的边界。

第二项应明确写成诊断与分析贡献。当前证据不足以支持第二个新的训练模块或新的 SDE。

## 2. 可写入论文的第一创新点

### 2.1 机制描述

当前 80k checkpoint 的关键配置为：

- OUVE；
- `rrsd_teacher=future_grad`；
- `rrsd_freeze_base=True`；
- `rrsd_weight=1.0`；
- `rrsd_future_weight=1.0`；
- `rrsd_smooth_weight=0.01`；
- `rrsd_future_gate_cost=0.001`；
- PC sampler `N=30`。

Router 根据 noisy 输入、判别式前端输出及扩散状态生成局部 `alpha(t,f)`。路由动作改变当前反向状态后，再经过一次冻结 Score 网络，得到 future `x0` 估计。训练的核心相对误差为：

\[
L_{future}=\frac{1}{B}\sum_i
\frac{\|\hat{x}_{0,i}^{route}-x_{0,i}\|_2^2}
{\operatorname{stopgrad}(\|\hat{x}_{0,i}^{base}-x_{0,i}\|_2^2)+\epsilon}.
\]

其中 `base` 是同一当前状态、同一 future 时间点下的 `alpha=0` counterfactual，分母停止梯度。训练时通过 Score 网络的输入梯度把 future 误差传回 Router；扩散主干保持冻结。额外的平滑项约束局部路由图，最终 checkpoint 的 `contrastive_weight=0`，因此当前主结果不是由 WavLM/UniSpeech 对比损失得到的。

### 2.2 官方冻结评测结果

评测使用 300 条官方锁定语音：no-reverb 和官方 with-reverb 各 150 条；同一 80k EMA checkpoint、PC `N=30`、seed `2026`，比较 Router-off、Router 和 `t_cut=0.05`。

Router 相对 Router-off 的 pooled 结果：

| 指标 | 差值 |
|---|---:|
| PESQ | `+0.005834`，95% CI 跨零 |
| STOI | `−0.000813` |
| SI-SDR | `+0.241412 dB` |
| DNSMOS-SIG | `−0.014882` |
| DNSMOS-OVRL | `−0.014472` |

分条件后，no-reverb 中 PESQ/STOI/SI-SDR 均提高；官方 with-reverb 中 SI-SDR 提高，但 PESQ、STOI、DNSMOS-SIG/OVRL 下降。匹配 file-id 的 Router×reverb 交互为：PESQ `−0.04110`、STOI `−0.00267`、SI-SDR `−0.15524 dB`、DNSMOS-OVRL `−0.02854`，主要交互项的 95% CI 排除零。

### 2.3 论文中可以使用的表述

> Future-gradient routing improves reconstruction under the matched Router protocol, but its effect is conditional rather than globally perceptual. The same routing action changes from a small reconstruction benefit in the official no-reverb split to a reconstruction–perception trade-off in the official with-reverb split.

中文可写为：

> Future guidance 并非对所有条件都产生一致的感知增益。它在无混响条件下带来小幅重建改善，在官方混响条件下则表现为 SI-SDR 与感知代理之间的权衡。该现象说明局部路由系数改变了前端信息与生成后端之间的协作方式，但当前 Router 尚不能识别所有需要保护或修复的局部区域。

## 3. 第二创新点的当前证据边界

### 3.1 已验证但不能作为新算法的方向

| 方向 | 结果 | 论文处理 |
|---|---|---|
| Anchor V2 | clean Oracle SI-SDR `+0.448234 dB`，DNSMOS-OVRL `−0.070951` | 作为重建 headroom 与感知 headroom 分离的反证 |
| WavLM Harm Teacher | Gate−Full OVRL `−0.000953`，CI 跨零 | 说明当前 harm 标签没有显示可利用感知 headroom |
| Speaker/Content loss | 身份差异未稳定改善；伪 WER 相对差约 `+0.00516` | 作为辅助监督路线的负结果 |
| Rollout adapter | unseen-speaker clean-x0 NMSE `−54.8%`，OVRL `−0.03465` | 只能称 proxy 瓶颈候选，不能称感知泛化或 CoF |
| P7-A/B/C | 排除了简单求解器、噪声断点和可观测 gate 解释 | 作为扩散链路瓶颈诊断，不称新 SDE |

### 3.2 不应使用的论文表述

- “Router 全面提升 PESQ、STOI、SI-SDR 和 DNSMOS。”
- “WavLM/SSL-MSE 已证明能改善 Router 的感知质量。”
- “Anchor V2 已经是第二个有效控制器。”
- “Rollout adapter 已验证 Corrective Forcing。”
- “P7 已证明 StoRM 存在新的扩散动力学或 predictor kink。”

## 4. 论文实验组织建议

1. **主结果表**：Router-off、Router、`t_cut=0.05`，分别报告 pooled/no-reverb/with-reverb。
2. **交互图**：报告 matched file-id 的 Router×reverb 交互，而不是只比较两个 split 的绝对均值。
3. **机制边界表**：Anchor Oracle、Harm Oracle、Speaker/Content、Rollout adapter、P7 诊断，统一列出假设、对照、结果和停止原因。
4. **复现材料**：保留配置、manifest、逐句 CSV、bootstrap JSON、源文件 hash 和 GPU/评估环境说明。
5. **听感表述**：当前 24 对单听者 A/B 全部 tie，只能说明差异小，不能声称听感等价或人耳改善。若论文必须使用“听感改善”，需要新锁定样本和多位独立听者。

## 5. 当前最合理的结论

当前工作已经足以支撑一篇严谨的硕士论文，但论文贡献应聚焦于：

- Future-gradient reliability routing 的实现和配对验证；
- 重建与感知目标不一致时的条件性 Pareto 行为；
- 混响条件对 Router 效果的显著调制；
- 对多个看似合理但未形成综合收益的第二创新候选进行可复现反证。

这比继续添加一个未经验证的 loss 或 controller 更容易形成可信、可复现、经得起审稿的研究叙事。

## 6. 证据索引

- [论文图表、表格和 claim matrix](./experiments/paper_figures_claim_matrix_20261020/)
- [冻结最终评测](./experiments/final_eval_full_20261018/)
- [混响交互分析](./experiments/final_eval_reverb_interaction_20261018/)
- [冻结指标算术复核](./experiments/final_metrics_repro_audit_20261020/)
- [Anchor V2 Oracle](./experiments/anchor_v2_real_oracle_20261017/)
- [Rollout adapter replication](./experiments/rollout_adapter_replication_20261018/)
- [P7 诊断记录](./experiments/storm_error_attribution_p7_20261017/)

