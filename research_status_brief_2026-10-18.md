# 当前研究状态简报（2026-10-18）

## 1. 当前最可信的正向结果

### Future-grad Router

在 64-speaker CV 复核中，Router 相对 Router-off 的结果为：

- PESQ `+0.010925`
- STOI `+0.001146`
- SI-SDR `+0.511487 dB`
- DNSMOS-SIG `-0.030869`
- DNSMOS-OVRL `-0.029837`

这说明 Future Router 对重建指标有小幅收益，但感知指标存在代价。它目前还不能被描述为综合质量提升。

### 末端时间校准 `t_cut=0.05`

在已用于筛选的 150 条配对数据上，`t_cut=0.05` 相对原始 Router 的 DNSMOS-OVRL 增加约 `+0.003873`，SI-SDR 下降约 `0.049719 dB`；在另一个低 SNR 配对 split 上方向一致。

它是一个可复现的工程 Pareto 对照，但仍是手工时间规则，不是已经学会的 Harm Teacher 或风险门控器。

## 2. 已完成的瓶颈诊断

- P7-0：确认 Router-off 仍保留 StoRM 前端 `Y_pre`；PC sampler 为 `N=30`、无 corrector，实际 Score forward 为 30 次；单文件重放逐采样一致。
- P7-A：ODE 容差阶梯没有证明收敛，严格容差带来很高 NFE 和耗时，未继续扩展。
- P7-B：没有发现跨语音稳定的 DSM/Score 异常层，未触发 Score 干预。
- P7-C：955 个 1 秒片段中只有 200 个局部重建 proxy 得益；noisy/`Y_pre` 的 12 个简单特征无法预测风险，Ridge OOF MAE `0.1050`，差于均值基线 `0.0975`。

结论是存在局部前后端冲突，但尚未找到可观测、可干预的系统特有扩散瓶颈。

## 3. 已停止的候选路线

- WavLM Harm Head：真实 Harm Oracle 的 PESQ/STOI/SI-SDR 增益接近零。
- Router-only Speaker InfoNCE / prototype：没有独立身份收益，并带来重建代价。
- Anchor V2：clean Oracle 提高 SI-SDR，但 DNSMOS-OVRL 下降约 `0.071`。
- Multi-trajectory selector：有 clean-aware headroom，但存在逐条 SI-SDR 下降和自选自评问题。
- backend adapter trust-region：fresh CV 单步 `x0` MSE 四个 adapter 均高于冻结 baseline，trust-region 不优于 score-only。
- P3b 听感：单听者 24/24 A/B 全部 tie，没有可重复偏好标签。
- P3c 内容 smoke：8 条开发语音的 Whisper 伪 WER 为 noisy `0.5932`、off `0.4323`、Router `0.4375`；Router−off 为 `+0.00516`，bootstrap CI 跨零，不支持 phonological anchor。

## 4. 论文贡献判断

目前只有一个相对稳妥的正向主线：

> Future guidance 在扩散末端存在感知—重建权衡；简单末端校准可以得到小幅、可复现的 DNSMOS—SI-SDR Pareto 点。

这还不足以支持第二个独立的学习模块。P7 的诊断结果可以作为机制分析和负消融，但不能包装成“已经定位并修复了扩散瓶颈”。如果论文要求两个正向创新点，应先提出新的、独立且可证伪的机制；不能把已失败的 Harm、Speaker、Anchor 或 Adapter 重新组合。

## 5. 下一步

1. 固定一个未参与路线筛选的最终评测 manifest，统一 checkpoint、EMA、采样器、NFE 和指标版本。
2. 整理 Future Router、Router-off、`t_cut` 的逐句质量、内容和身份指标，形成论文主表及 paired bootstrap CI。
3. 只有出现新的系统特有失败机制和独立监督证据，才重新启动第二机制的短程验证；当前不启动长训练。

详细实验记录见 `/data/RESEARCH_PLAN.md` 及各 `experiments/<id>/experiment_report.md`。
