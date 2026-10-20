# P7-C 首轮前端—后端配对归因

## 目的与范围

在同一 32 条既有开发 manifest 上比较 Frontend-only `Y_pre`、Router-off 完整 StoRM 后端输出和 RRSD Router 输出。该 split 先前已参与方法筛选，因此结果只用于探索性误差归因，不是独立盲测。实验没有训练，也没有修改源音频或 checkpoint。

详细复现参数见 `config.yaml`，逐条配对结果见 `per_utterance_metrics.csv`，音频路径/hash 见 `paired_audio_manifest.json`。PESQ/STOI、DNSMOS 和复谱归因各自的设置与汇总分别记录在 `pesq_stoi_metrics.json`、`dnsmos_metrics.json`、`metrics.json`。

## 主要结果

| 指标 | Frontend-only `Y_pre` | Router-off | Router |
|---|---:|---:|---:|
| PESQ | 1.494 | 1.729 | 1.749 |
| STOI | 0.8216 | 0.8310 | 0.8319 |
| SI-SDR (dB) | 8.089 | 7.278 | 7.801 |
| DNSMOS OVRL | 2.996 | 3.206 | 3.185 |

相对 Router-off，Frontend-only 的配对差为：SI-SDR `+0.811 dB`（95% bootstrap CI `[+0.614,+1.004]`），PESQ `−0.236`（`[−0.309,−0.165]`），STOI `−0.00936`（`[−0.01420,−0.00419]`），DNSMOS OVRL `−0.211`（`[−0.283,−0.144]`）。在这组开发语音上，扩散后端改善了这些感知代理指标，但 Router-off 的 SI-SDR 低于 Frontend-only；这不能外推为总体感知—重建不可能同时改善。

STFT clean-aware 诊断中，Frontend-only 的复谱相对误差均值为 `0.413`、Router-off 为 `0.482`；有效能量区域相位误差分别为 `0.186` 和 `0.234`。幅度相对误差则分别为 `0.349` 和 `0.334`，方向上更有利于后端。Router 的幅度误差为 `0.325`、复谱误差为 `0.452`、相位误差为 `0.227`。这些指标支持继续测试两个固定幅相交叉波形，但不能证明特定相位相消或可听伪影，也没有证明可部署的可靠性门控。

## DNSMOS GPU 迁移状态

本目录现有完整 DNSMOS 结果仍由 CPU 计算，运行时为 303.4 秒，未覆盖。新 GPU-auto 代码及一条已有 30 秒音频的验证记录见 `../dnsmos_gpu_smoke_20261017/`：该样本的 DNSMOS 组件热启动平均耗时 CPU 3.213 秒、CUDA 0.199 秒；ONNX profiling 确认两个模型的节点都由 CUDA provider 执行，CPU/GPU 分数最大差为 `0.000131`。该数字仅描述一个样本的 DNSMOS 组件，不能称为整套评测或完整语料的速度提升。Mel 特征提取仍在 CPU 上进行。

## 结论边界与下一步

- **已观测**：三个输出在重建与感知代理指标间有差异；谱幅度误差偏向完整后端，复谱/相位误差偏向 Frontend-only。
- **尚未证明**：幅相差异是否是最终质量变化的原因；幅相交换是否能改善波形；这些差异能否由部署时可见条件预测；是否存在真实可听改善。
- **下一步**：只用两个预先固定的幅相交叉组合 `|P|∠G` 与 `|G|∠P` 做 iSTFT 后波形与配对指标诊断；不搜索混合系数、不训练、不按评价指标挑选并宣传最佳版本。若未见跨样本稳定互补或只有单一代理改善，则归档并停止 P7-C。
