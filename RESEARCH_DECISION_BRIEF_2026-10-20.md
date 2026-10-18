# StoRM / SGMSE 当前研究决策简报

日期：2026-10-20  
项目：`/data/coding/sgmse`  
数据集：`/data/datasets/dns_ins20`（未修改）

## 一句话结论

目前最可靠、最适合写进论文的主线是：

> **Future guidance 带来小幅重建收益，但存在稳定的感知—重建权衡，并且该权衡与混响条件显著交互。**

目前还没有第二个经过跨说话人、感知指标验证的独立算法创新。继续盲目添加 loss 或训练 controller 的证据不足。

## 已完成的关键任务与建设性成果

### 1. 官方冻结集最终对照

300 条锁定语音，使用同一 80k EMA checkpoint、PC `N=30`、`snr=0.5`、seed `2026`，完成 `off / Router / t_cut=0.05` 三臂评测。

Router 相对 off 的 pooled 差值：

| 指标 | Router − off |
|---|---:|
| PESQ | `+0.005834`（95% CI 跨零） |
| STOI | `−0.000813` |
| SI-SDR | `+0.241412 dB` |
| DNSMOS-SIG | `−0.014882` |
| DNSMOS-OVRL | `−0.014472` |

匹配 file-id 的混响交互项为 PESQ `−0.04110`、STOI `−0.00267`、SI-SDR `−0.15524 dB`、DNSMOS-OVRL `−0.02854`。这支持“效果依赖混响条件”的机制分析，但不能声称 Router 全面提升感知质量。

记录：[`experiments/final_eval_full_20261018`](./experiments/final_eval_full_20261018)、[`experiments/final_eval_reverb_interaction_20261018`](./experiments/final_eval_reverb_interaction_20261018)。

### 2. Anchor V2 已完成上限诊断并停止

32 条开发语音的 clean-reference Oracle 显示二维 residual 子空间仍有重建空间：SI-SDR `+0.448234 dB`；但 DNSMOS-OVRL `−0.070951`，其他感知指标也下降。

建设性结论：**前后端融合系数还有重建 headroom，不代表存在感知 headroom。** 因此不继续训练 Anchor controller。

记录：[`experiments/anchor_v2_real_oracle_20261017`](./experiments/anchor_v2_real_oracle_20261017)。

### 3. WavLM Harm Teacher / 对比特征路线已停止

4 条 `N=30` 音频的 WavLM harm oracle 补测中，Gate−Full 的 DNSMOS-OVRL 为 `−0.000953`，bootstrap 区间跨零；PESQ、STOI、SI-SDR 也没有稳定的可利用感知收益。

建设性结论：当前证据不支持训练 Harm Head，也不支持继续把 WavLM 特征 MSE 直接作为 Router 优化目标。

记录：[`experiments/rrsd_harm_oracle_n30_4_dnsmos_20261019`](./experiments/rrsd_harm_oracle_n30_4_dnsmos_20261019)。

### 4. Speaker 与内容方向完成 headroom 检查

- 64-speaker CV：Router 相对 off 的 SI-SDR `+0.5115 dB`，DNSMOS-OVRL `−0.0298`；CAM++ 身份差异没有稳定改善。
- 8 条 ASR smoke：Router 伪 WER 相对 off 约 `+0.00516`，没有内容保持收益证据。

建设性结论：不继续重复 Speaker loss、prototype、phonological anchor 或 content loss 的加权/延长训练。

记录：[`experiments/speaker_headroom_replication_64_20261017`](./experiments/speaker_headroom_replication_64_20261017)、[`experiments/content_headroom_asr_smoke_20261018`](./experiments/content_headroom_asr_smoke_20261018)。

### 5. Rollout adapter 只改善 proxy，未证明感知泛化

两轮 pilot 都降低 clean-`x0` proxy；独立 4-speaker holdout 的 NMSE 下降约 `54.8%`，但 unseen-speaker DNSMOS-OVRL 下降 `0.03465`。

建设性结论：可以记录为“proxy 瓶颈可被修正的候选”，不能写成 Corrective Forcing 成功或第二创新点；当前停止扩大训练。

记录：[`experiments/rollout_adapter_pilot_20261018`](./experiments/rollout_adapter_pilot_20261018)、[`experiments/rollout_adapter_replication_20261018`](./experiments/rollout_adapter_replication_20261018)。

### 6. P7 扩散链路诊断完成一轮有界筛查

P7-A 没有证明默认 PC sampler 是主要瓶颈；P7-B 在噪声功率 `M={0.5,0.75,1,1.25,1.5}` 下质量平滑下降，没有 `M≈1` 附近可重复的 predictor kink；P7-C 的简单可观测特征不能可靠预测局部 repair gain。

建设性结论：P7 的成果是排除若干错误解释、缩小研究范围，不是已验证的新扩散 SDE 或新 controller。

### 7. 感知代理审计完成

Distill-MOS 在独立 64-speaker CV 中与当前 DNSMOS-OVRL 退化方向不一致，不能作为单独 reward；MR-STFT 对 OVRL 有一定排序能力，但没有足够的人类偏好监督。盲听 pilot 的 24 对均为 tie，说明当前差距很小，不能证明等价。

## 当前明确暂停的路线

- WavLM Harm Teacher / harm gate
- Speaker InfoNCE、prototype、Router-only speaker adaptation
- Anchor V2 controller
- rollout-aware adapter 扩大训练
- 普通 perceptual loss、GAN、Distill-MOS trust-region
- mismatch-aware training、reverb gate、未验证的新 SDE

这些路线不是“理论上永远无效”，而是当前数据和验证结果没有显示足够的独立收益，继续投入会增加计算成本和论文叙事风险。

## 需要你决定的事项

### 决定 A：论文主线（推荐 A1）

**A1：收敛到当前实证主线。**

- 把 Future guidance 写成条件性的重建收益；
- 报告感知—重建 Pareto 权衡；
- 强调混响交互；
- 将 Anchor、Harm、Speaker、Rollout、P7 结果作为系统性失败路线/机制边界分析；
- 下一步只做图表、claim matrix、统计表和复现实验材料。

**A2：继续寻找第二个算法创新。**

只建议做零训练、低成本的 fresh-CV 诊断，并设置明确退出条件：如果前端中间状态在 speaker-grouped OOF 下不能超过 SNR/RMS/余弦等简单特征，就停止该方向，不启动 adapter 训练。

### 决定 B：是否补充独立盲听

如果论文必须声称“听感改善”，建议增加独立听者和独立样本的 paired listening。现有 24 对 tie 只能说明差距小，不能作为等价性证明或训练监督。

### 决定 C：是否清理可再生文件

`/data` 当前约剩 `4.1 GB`，其中实验目录和日志占用较大；当前没有自动删除，以免误删可复现实验记录。若需要继续实验，建议只清理 WAV、缓存和可再生中间文件，保留每个实验的 `config.yaml`、`metrics.json`、`experiment_report.md`、manifest 和 hash。

## 当前运行状态与安全边界

- 当前没有训练、推理或 DNSMOS 后台任务。
- 所有协作实验任务均已结束；没有任务在等待用户确认。
- 未修改 `/data/datasets/dns_ins20`。
- 未覆盖有效 checkpoint。
- 没有向远端 push；最近代码提交为本地 commit `addc75f`（Distill-MOS independent CV audit）。
- 正式推理使用 GPU；DNSMOS 请求过 CUDA，但当前 ONNX Runtime 未注册 CUDA provider，因此评估实际回退 CPU。这只影响评估速度，不改变已记录的推理配置。

## 推荐执行顺序

1. 你先决定采用 A1 还是 A2。
2. 若选 A1，我生成论文图表、claim matrix、统计表和最终实验索引，不再启动高成本训练。
3. 若选 A2，我只运行 fresh-CV、低成本、可停止的状态表征诊断；只有出现跨 speaker 的增量解释力，才提出小型 adapter 实验。

