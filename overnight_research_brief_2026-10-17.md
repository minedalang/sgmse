# StoRM / RRSD 夜间研究简报

日期：2026-10-17  
基准：80k EMA Router checkpoint `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`  
约束：未修改原始数据集，未覆盖有效 checkpoint，未 push 到远端。

## 一、今晚完成的工作

### 1. Future Harm Teacher / gate 审计

- 有限差分诊断中，64 个冲突样本的双向符号一致率为 `59.4%`；低扩散时间段为 `71.4%`，说明 WavLM 代理能够提供一定局部方向信息，但不是可靠的 DNSMOS 标签。
- 4 条语音、N=30 的真实 Harm Oracle 相对 Router：PESQ `+0.000158`（95% CI `[-0.000006,+0.000388]`）、STOI `+0.000007`、SI-SDR `-0.000018 dB`，没有可重复的真实指标提升。
- 结论：当前证据不足以训练 Harm Prediction Head；WavLM harm 只能作为待验证代理，不能宣称感知风险标签成立。

记录：`experiments/rrsd_harm_teacher_fd_probe/`、`experiments/rrsd_harm_oracle_n30_4_paired/`。

### 2. 固定强度和时间门控

| 策略 | DNSMOS-OVRL 变化 | SI-SDR 变化 | 结论 |
|---|---:|---:|---|
| 固定 scale=0.5 | `+0.011710` | `-0.231586 dB` | 可控 Pareto 点，代价较大 |
| 固定 scale=0.25 | `+0.017246` | `-0.369448 dB` | 感知收益更大，但不实用 |
| `t_cut=0.05`，150 条测试集 | `+0.003873`，CI 全为正 | `-0.049719 dB` | 当前最好的简单 Pareto 基线 |
| `t_cut=0.05`，留出低 SNR 集 | `+0.003221`，CI 全为正 | `-0.061079 dB` | DNSMOS 方向可泛化，但重建代价略增 |
| late-step partial scale=0.25 | `+0.001211`，CI 跨零 | `-0.043100 dB` | 没有稳定 OVRL 收益 |

结论：抑制末端 Future correction 确实能稳定提高 DNSMOS，但目前只是手工时间规则，不能称为已学习的 Harm Teacher。

记录：`experiments/rrsd_tcut005_full_20261017/`、`experiments/rrsd_tcut005_alt15_full_20261017/`、`experiments/rrsd_tcut005_soft_screen_20261017/`。

### 3. Oracle 上限与策略选择

- 150 条语音的整句策略 Oracle，在 SI-SDR 容忍 `0.05 dB` 时，DNSMOS-OVRL 比固定 Router 多 `+0.00361`；选择 Router 91 条、`t_cut` 58 条、selective 1 条。
- 仅使用 noisy 波形统计量的 leave-one-speaker-out selector 没有复现这部分空间：
  - 0.05 dB 容忍下，Decision Tree OVRL `3.185760`，Logistic `3.185610`，固定 Router `3.185399`；差异很小且没有统计证明。
  - 0.10 dB 容忍下，Tree `3.188702`，Logistic `3.187368`，仍低于 clean-reference Oracle `3.189294`。

结论：已有策略之间的可学习空间很小，暂不训练复杂 utterance policy head 或 tile-level selector。

记录：`experiments/rrsd_utterance_oracle_150/`、`experiments/rrsd_observable_selector_probe_32/`。

### 4. 其他路线的否定结果

- K=4 Future：相对 Router 的 SI-SDR `-0.17996 dB`，STOI `-0.000563`，PESQ `-0.00243`；不继续增加 lookahead 数量。
- matched conflict-aware control：冲突权重没有可靠 OVRL/SIG 增益，并带来约 `-0.0225 dB` SI-SDR 代价；不继续长训。
- Speaker B：训练集确实有可靠 speaker ID 和同人异句样本，但 masked InfoNCE 和 speaker prototype 两个 1000-step 变体都没有独立身份收益，且 SI-SDR 分别下降约 `0.144` 和 `0.129 dB`；暂不继续当前短程路线。

记录：`experiments/rrsd_kstep_k4_1000/`、`experiments/rrsd_control1000_full_matched_20261017/`、`experiments/rrsd_cv_repeat32/`。

## 二、建设性成果

1. **找到了可重复的工程 Pareto 基线。** `t_cut=0.05` 在原测试集和低 SNR 留出集都提高 DNSMOS-OVRL，代价约 `0.05–0.06 dB` SI-SDR。它适合做后续任何自适应 gate 的最低对照线。
2. **明确了当前 Harm Teacher 不能直接成立。** 局部 WavLM 梯度方向有机制信号，但真实音质 Oracle 的收益接近零，因此避免了继续投入高成本、低依据的 Harm Head 训练。
3. **量化了创新上限。** 现有 Router、t-cut、selective 的整句 Oracle headroom 只有约 `+0.0036 OVRL`（受 SI-SDR 约束），说明复杂路由器即使完美学习也不会带来很大的收益。
4. **排除了多个容易重复的方向。** 全局 scale、K-step 扩展、conflict penalty、当前 speaker contrastive/prototype 和 noisy 统计 selector 均没有形成独立、稳定的综合提升。

## 三、当前研究判断

- A 当前最可信的表述是：**Future Router 的末端可靠性校准可形成小幅 DNSMOS—SI-SDR Pareto 改善**。暂时不能表述为 WavLM Harm Teacher 已成功。
- B 当前应作为负消融保留，除非提出新的机制并使用匹配 80k 预算验证；不要仅靠增加 speaker loss 权重或更换负样本继续扩训。
- 如果论文必须保留两个正向创新点，需要从提供的近两年论文调研中寻找与“固定 t-cut / SSL-MSE / speaker contrastive”不同的新机制；在此之前不启动长训练。

## 四、状态与下一步

- 当前没有训练或推理进程。
- `/data` 剩余约 7 MB，新的批量音频实验暂不安全；`/dev/shm` 仍可用于临时小型诊断。
- selector probe 已提交：Git commit `7691902`，未 push。
- 下一步优先级：阅读论文调研报告，筛选 1–2 个可用现有 80k checkpoint、无需大规模训练即可证伪的机制；先做代码审计和 4 条语音 smoke，再决定是否训练。
