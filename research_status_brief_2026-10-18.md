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

### 官方最终集的新证据

锁定的 300 条 synthetic 测试集已完成 off / Router / `t_cut=0.05` 三臂评测。Router 相对 off 的 pooled 结果为 PESQ `+0.005834`（CI 跨零）、STOI `-0.000813`、SI-SDR `+0.241412 dB`、DNSMOS-SIG `-0.014882`、OVRL `-0.014472`。分 split 后，no-reverb 有 PESQ/STOI/SI-SDR 收益；with-reverb 的 PESQ、STOI、SIG、OVRL 均下降。

对 150 个匹配 file-id 的事后交互分析进一步得到 `(Router-off)_with_reverb - (Router-off)_no_reverb`：PESQ `-0.04110`、STOI `-0.00267`、SI-SDR `-0.15524 dB`、DNSMOS-OVRL `-0.02854`，主要指标的 matched-file bootstrap CI 均排除零。这支持“Router 与混响条件存在稳定交互”，但尚不能区分前端条件失配、幅相重建问题或训练分布差异。SNR 分层也显示该现象不能由单一 SNR 阈值解释，因此暂不设计 reverb/SNR gate。记录见 `experiments/final_eval_full_20261018/`、`experiments/final_eval_reverb_interaction_20261018/` 和 `experiments/final_eval_strata_20261018/`。

80k checkpoint 的保存配置审计显示训练/验证使用单一 `tr_synthetic`/`cv_synthetic` noisy SCP（映射后 10,800/1,200 条），没有记录 with-reverb 配对训练项或混响增强开关。它使训练—测试条件失配成为后续可证伪假设，但不构成因果证明；审计记录见 `experiments/checkpoint_data_condition_audit_20261018/`。

在排除既有 manifest 的 4 个 CV speaker 上完成固定 FIR 合成混响 smoke。Router 相对 off 的 SI-SDR interaction 相对 identity 为 short `-0.0448 dB`、medium `-0.0714 dB`，但 PESQ interaction 为正，未复现官方最终集 PESQ 与 SI-SDR 同时变差的模式。该 smoke 仅 4 条且使用人工 FIR，按停止条件不扩大此路线，也不启动 reverb adapter/gate；详见 `experiments/reverb_cv_smoke_20261018/`。

随后完成本地混响资源只读审计：官方 README 说明 `with_reverb` 使用内部录制 RIR，但当前配置引用的原机器 `/home/xyf/...` RIR 目录不存在；`/data/coding/sgmse`（排除环境目录）也没有可用 RIR 文件。因此目前无法把人工 FIR smoke 升级为独立真实 RIR fresh-CV；在获得外部、固定且来源明确的 RIR 前，不扩展人工 FIR、不启动 reverb adapter 或混响训练。记录见 `experiments/reverb_resource_audit_20261018/`。

对冻结的 150 个 matched file-id 进一步做了连续混响严重度事后分析。以 no-reverb/with-reverb clean 的波形、复谱、log-magnitude 和 RMS 变化作代理时，Router 与 DNSMOS-OVRL 交互的 Spearman 相关均接近零且 CI 跨零；只有与 SI-SDR 交互存在有限相关。因此不能把“混响越强导致感知退化”作为已定位机制，也不设计 severity gate；分析记录见 `experiments/reverb_severity_posthoc_20261018/`。

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

1. 官方 DNS Ins20 `test_set/synthetic` 两个 split 已锁定为 300 条候选最终评测集；manifest SHA256 为 `4c91782fb1459bd14f2b8581b1f4b537ca98bafd907ecc961226883213bcfa2b`。
2. 已完成 4 条、三臂（Router-off/Router/`t_cut`）真实推理链路 smoke；统一 checkpoint、EMA、PC、NFE 和 seed 均可执行。完整质量评估仍需在方法协议冻结且复查磁盘后进行。
3. 整理 Future Router、Router-off、`t_cut` 的逐句质量、内容和身份指标，形成论文主表及 paired bootstrap CI。
4. 只有出现新的系统特有失败机制和独立监督证据，才重新启动第二机制的短程验证；当前不启动长训练。

详细实验记录见 `/data/RESEARCH_PLAN.md` 及各 `experiments/<id>/experiment_report.md`。

代码层面的 Residual-Anchored OUVE 数学审计也已完成：其时间变化锚点满足 `beta(t)=(1-alpha)lambda(t)` 与 `a(t)=beta'(t)+theta*beta(t)`，平移 `U_t=X_t-beta(t)y` 后还原为标准 OUVE；它改变的是条件表示和有限容量 Score 学习问题，不是新的随机动力学。因此不把该分支作为第二个独立 SDE 创新，也不因公式变化启动训练。记录见 `experiments/residual_anchor_equivalence_audit_20261018/`。

复用已保存的四轨迹结果做了固定一致性惩罚选择：`z(DNSMOS-OVRL)-0.5 z(inter-trajectory consistency)`。held-out OVRL 为 `+0.05875`（CI 全正），SI-SDR 为 `+0.02107 dB`（CI 跨零），SI-SDR 负例从纯 OVRL 选择的 6/16 降到 4/16。该结果只证明一致性具有风险排序价值，仍不构成可部署 Pareto verifier；未启动新推理或训练。记录见 `experiments/multitrajectory_consistency_verifier_20261018/`。

基于 Anchor V2、TR-0 adapter、混响资源/严重度和多轨迹 verifier 的联合审计，当前不启动 NCSN++ adapter、Distill-MOS 训练、learned verifier 或官方集四轨迹推理。除非出现新的独立监督或外部固定 RIR，论文主线保留 Future Router 与 `t_cut=0.05` 的 Pareto 结果，并把后续机制作为已验证边界和负诊断。
