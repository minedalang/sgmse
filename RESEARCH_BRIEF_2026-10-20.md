# StoRM / SGMSE 研究简报

记录覆盖：2026-10-17 至 2026-10-20（依据当前仓库的最新实验记录）

## 一句话结论

目前最可靠的论文结果是：**Future guidance 带来小幅重建收益，但存在稳定的感知—重建权衡，并且与混响条件显著交互。** 目前没有第二个已经通过感知泛化验证的独立算法创新。P7 已完成一轮有界诊断，结果主要用于排除错误机制解释和缩小下一步范围。

## 你刚才问的：P7 是什么？

P7 不是“证明扩散模型机制已经成立”的实验，也不是一个已经训练好的新模型。它是围绕以下问题设计的诊断阶段：

> 判别式前端已经恢复了一部分语音后，扩散后端是否在某些状态、噪声条件或局部区域中没有正确利用这些信息？

P7 的具体分支：

- **P7-0：调用链与实现审计。** 检查 Router-off、前端、OUVE/PC sampler 和条件路径是否按预期运行。没有发现足以解释结果的代码路径错误。
- **P7-A：数值/ODE 容差诊断。** 严格容差会增加代价，但没有证明默认 PC sampler 是主要质量瓶颈，因此不改 SDE 或默认采样器。
- **P7-B：噪声功率 mismatch / predictor-kink smoke。** 在 4 个 fresh-CV speaker 上改变噪声残差比例 `M={0.5,0.75,1,1.25,1.5}`。质量随噪声平滑下降，曲率主要来自低噪声端，前端和扩散输出相近；没有发现 `M≈1` 附近可重复的扩散 predictor 断点。因此不启动 mismatch-aware training 或新 controller。
- **P7-C：可观测局部修复信号。** 用输入、前端输出等可部署特征预测局部收益，Ridge MAE `0.1050`，还不如均值基线 `0.0975`；没有得到可用的局部 gate。
- **Matched-time rollout 诊断：** reverse rollout 状态的 clean-x0 NMSE `0.5140`，解析 forward 状态为 `0.1356`，差值 `+0.3785`，8/8 fresh-CV 语音为正。这是一个值得记录的 proxy 瓶颈，但不是最终质量结果，也不等于 Corrective Forcing 已被验证。

所以，P7 的建设性产出是**定位和排除候选瓶颈**，而不是声称“我们提出了一个新的扩散动力学”。

## 已完成且值得保留的工作

### 1. 官方锁定集的最终对照

300 条锁定语音、同一 80k EMA checkpoint、PC `N=30`、seed 2026，已完成 `off / Router / t_cut=0.05` 三臂评估。Router 相对 off 的 pooled 差异为：

| 指标 | Router − off |
|---|---:|
| PESQ | `+0.005834`（CI 跨零） |
| STOI | `−0.000813` |
| SI-SDR | `+0.241412 dB` |
| DNSMOS-SIG | `−0.014882` |
| DNSMOS-OVRL | `−0.014472` |

no-reverb 上 PESQ/STOI/SI-SDR 改善；with-reverb 上 PESQ/STOI/DNSMOS 下降但 SI-SDR 仍有小幅改善。匹配 file-id 的混响交互项为 PESQ `−0.04110`、STOI `−0.00267`、SI-SDR `−0.15524 dB`、OVRL `−0.02854`。这支持论文中“条件相关的 Pareto 权衡”这一主张，但不能写成 Router 全面提升感知质量。

记录：[`final_eval_full_20261018`](./experiments/final_eval_full_20261018)、[`final_eval_reverb_interaction_20261018`](./experiments/final_eval_reverb_interaction_20261018)。

### 2. Anchor V2 的真实 Oracle 上限诊断

32 条开发语音的 clean-reference Oracle 显示二维 residual 子空间确有重建 headroom：SI-SDR `+0.448234 dB`；但 DNSMOS-OVRL `−0.070951`，并伴随 SIG/P808 下降。因此训练 Anchor controller 没有感知收益空间，已归档。这个结果很有用：它说明“再精细地控制前端/后端融合系数”不能自动解决感知问题。

记录：[`anchor_v2_real_oracle_20261017`](./experiments/anchor_v2_real_oracle_20261017)。

### 3. Rollout-state proxy 的复现及其边界

两轮小规模 adapter pilot 都降低了 rollout-state clean-x0 proxy；独立 4-speaker 复现中，2 条 unseen-speaker holdout 的 NMSE 下降 `54.8%`。但同一 holdout 的 DNSMOS-OVRL 下降 `0.03465`，因此不能把它写成感知收益、Corrective Forcing 成功或第二个创新点。该方向当前停止扩大。

记录：[`rollout_prediction_gap_20261018`](./experiments/rollout_prediction_gap_20261018)、[`rollout_adapter_replication_20261018`](./experiments/rollout_adapter_replication_20261018)。

### 4. Harm Teacher / WavLM 路线被实证停止

已有的 WavLM harm Oracle 在 4 条 `N=30` 音频上只读补测 DNSMOS，Gate−Full 的 OVRL 为 `−0.000953`，区间跨零；PESQ/STOI/SI-SDR 也没有显示可利用的稳定感知 headroom。当前没有理由训练 Harm Head。

记录：[`rrsd_harm_oracle_n30_4_dnsmos_20261019`](./experiments/rrsd_harm_oracle_n30_4_dnsmos_20261019)。

### 5. Speaker 与内容方向的 headroom 审计

64-speaker CV 诊断确认模型距离 clean speaker 表征仍有 headroom，但 Router 相对 off 的 CAM++ 身份结果没有稳定改善，同时 DNSMOS-SIG/OVRL 下降。8 条 ASR 内容 smoke 中 Router 的伪 WER 相对 off 增加约 `0.00516`，没有内容收益证据。因此不继续通过加权、换编码器或延长步数重复 Speaker loss；也不启动 phonological/content loss。

记录：[`speaker_headroom_replication_64_20261017`](./experiments/speaker_headroom_replication_64_20261017)、[`content_headroom_asr_smoke_20261018`](./experiments/content_headroom_asr_smoke_20261018)。

### 6. 感知表示筛选与听感试验

MR-STFT log-magnitude 对 DNSMOS-OVRL 的成对排序一致率约 `0.802`，但对 PESQ/STOI/SI-SDR 较弱；波形/复谱距离对 SI-SDR 很强，却对 OVRL 只有 `0.208`。盲听 pilot 中唯一听者的 24 对全部选择 tie。当前没有足够的人类偏好监督来训练自建感知 encoder、GAN 或感知 loss。

记录：[`perceptual_representation_probe_20261017`](./experiments/perceptual_representation_probe_20261017)、[`p3b_blind_listening_pilot_20261017`](./experiments/p3b_blind_listening_pilot_20261017)。

## 当前已经排除或暂停的路线

- WavLM Harm Teacher / harm gate：Oracle 没有感知 headroom。
- Speaker InfoNCE、prototype、Router-only speaker adaptation：没有稳定跨语句身份收益，并有重建代价。
- Anchor V2 controller：Oracle 有 SI-SDR headroom，但感知指标恶化。
- 多轨迹 selector / consistency verifier：有 OVRL 排序空间，但没有可靠 SI-SDR 保护。
- P7-A/B/C 的简单数值、噪声断点和可观测 gate 假设：未形成可行动证据。
- 普通 GAN、感知 loss 或直接增加 adapter：没有独立的人类偏好或部署风险标签支撑。
- Residual-Anchored OUVE：在审计假设下等价于标准 OUVE，不能作为新 SDE 声称。
- Corrective Forcing 直接复现：已有近期先例，且当前 rollout adapter 感知泛化未成立。

### 新增：Frontend state representation probe

已在 4 个 speaker、8 条 fresh-CV 语音上成功提取 StoRM 前端三个中间 hook，并复用已有 rollout cache 做 speaker-LOSO OOF。随后在 16 个不同 speaker 上在线复核：简单 SNR/RMS/余弦基线 MAE `0.15479`，hook-only `0.16416`，combined `0.15619`。中间表征没有稳定超过简单可观测特征，state-conditioned adapter 不启动。详见 [`frontend_state_representation_probe_20261020`](./experiments/frontend_state_representation_probe_20261020) 和 [`frontend_state_representation_probe_20261020_online16`](./experiments/frontend_state_representation_probe_20261020_online16)。

## 目前最建设性的科研结论

1. **Future guidance 的收益是条件性的，不是全局感知提升。** 这本身可以成为论文中的限制性机制分析和严谨实验贡献。
2. **重建 headroom 与感知 headroom 分离。** Anchor Oracle 和 rollout adapter 都显示，proxy 或 SI-SDR 改善不能直接推出 DNSMOS/听感改善。
3. **当前主要缺口不是“再加一个 loss”，而是缺少可部署、跨说话人稳定的失真/风险信号。** 这解释了为什么 WavLM、Speaker loss、简单感知距离和局部门控都没有形成综合收益。
4. **P7 的正结果应写成“可修正 proxy 瓶颈候选”，不能写成已验证的新扩散机制。**

## 需要你决定的事项

### 决定 A：下一步研究定位（推荐先选这个）

- **A1（推荐）：** 接受当前最强论文定位：Future guidance 的条件性 Pareto 权衡 + 混响交互 + 系统性失败路线审计。继续做论文整理、图表和公平对照，不启动高成本训练。
- **A2：** 继续寻找第二个算法创新。建议先做零训练、低成本的 **DFM-inspired frontend state representation diagnostic**：在 8–16 条 fresh-CV、speaker-grouped OOF 下提取 StoRM 前端中间表示，比较它对 rollout x0 proxy / 后端局部误差的解释力是否超出 diffusion time、输入 SNR、RMS 等简单特征。只有中间表征有跨 speaker 增量解释力，才考虑小型 state-conditioned backend adapter；否则归档。

### 决定 B：是否进行独立盲听

若论文必须主张“听感”而不能只报告 DNSMOS，当前最小必要动作是增加独立听者/独立样本的 paired listening。现有 24 个 tie 只能说明当前差距很小，不能证明等价，也不能作为训练监督。

### 决定 C：资源处理

`/data` 当前约剩 `2.3 GB`，`experiments` 约 `3.6 GB`、`experiment_logs` 约 `1.4 GB`；`/root` 仍有约 `12 GB` 可用。当前没有自动删除实验产物，因为其中包含可复现记录和未提交历史改动。若选择 A2，建议先清理明确可再生的 WAV/缓存副本，保留 `config.yaml`、`metrics.json`、`experiment_report.md` 和 manifest/hash。

## 当前运行状态

- 没有正在运行的训练或推理。
- 最近一次实验提交：`5561511 record noise power mismatch smoke`。
- 未修改 `/data/datasets/dns_ins20`，未覆盖 checkpoint，未 push 云端。
- DNSMOS 的评估请求过 CUDA，但当前 mymetrics 的 ONNX Runtime 未注册 CUDA provider，实际多次回退 CPU；正式推理仍使用 GPU。

## 建议的下一步执行顺序

1. 先由你决定论文主线采用 A1，还是继续执行 A2 的低成本状态表征诊断。
2. 若选 A2，先审计现有 frontend hook 和 fresh-CV manifest，不训练、不写大批 WAV，输出 `config.yaml`、`metrics.json`、`experiment_report.md`。
3. 只有诊断出现跨 speaker、超越 time/SNR 基线的增量解释力，才设计小型 adapter；否则停止该路线并回到论文主线。
