# RRSD 阶段 A：冻结预训练编码器特征敏感性测试

## 目的

不训练、不更改 Router、不重复推理。利用 `experiment_logs/rrsd_tcut_screen/` 中之前已增强好的相同 32 条语音（`off/`, `router/`, `clean/`, `paired_metrics.csv`, `manifest.csv`），验证原对比学习特征是否感知噪声和语音失真，并是否对 Off→Router 的变化敏感。

默认 `unispeech_sv`：**严格使用**之前双表征对比学习代码的 UniSpeech-SAT-SV 最后一层隐藏状态时间均值（`hidden_mean`），以及预训练 X-Vector 头输出（`speaker_embedding`）。如果训练时没有开启 `--unispeech_sv_head`，请使用 `ENCODER=unispeech`，避免声称原来用了未启用的分支。CAM++ / SpeechBrain X-Vector 只输出说话人表征。

## 1. 上传与环境

把 `rrsd_stageA_feature_probe.py` 和 `run_rrsd_feature_stageA.sh` 放到 `/data/coding/sgmse/`。沿用现有 `env`（应已有 torch、transformers、soundfile、numpy、scipy）。**不要把整个 SGMSE 80k checkpoint 当作 Hugging Face 编码器路径。** `MODEL_PATH` 填冻结编码器的预训练仓库或本地权重目录。

```bash
cd /data/coding/sgmse
python -c 'import torch, transformers, soundfile, scipy; print("deps ok")'
python -m py_compile rrsd_stageA_feature_probe.py
bash -n run_rrsd_feature_stageA.sh
```

## 2. 先跑 4 条检查，再跑 32 条

**双表征 UniSpeech-SAT-SV**（仅当你之前启用了 `--unispeech_sv_head`）：

```bash
COUNT=4 ENCODER=unispeech_sv \
  MODEL_PATH=microsoft/unispeech-sat-base-plus-sv \
  bash run_rrsd_feature_stageA.sh

COUNT=32 ENCODER=unispeech_sv \
  MODEL_PATH=microsoft/unispeech-sat-base-plus-sv \
  bash run_rrsd_feature_stageA.sh
```

如果预训练模型已下载到本地，把 `MODEL_PATH` 换成本地目录；服务器不能联网且本地也没有预训练模型时，这项实验无法正常进行，**脚本不会退回随机编码器**。

**只用 UniSpeech-SAT 最后一层隐藏表征**（原模型未打开 SV 头）：

```bash
COUNT=32 ENCODER=unispeech MODEL_PATH=microsoft/unispeech-sat-base-plus-sv bash run_rrsd_feature_stageA.sh
```

**CAM++**（使用当前工程中的 `sgmse/speaker_encoders.py`，预训练模型路径须真实存在）：

```bash
COUNT=32 ENCODER=campplus \
  MODEL_PATH=/data/coding/sgmse/pre_models/wespeaker-voxceleb-campplus \
  bash run_rrsd_feature_stageA.sh
```

**SpeechBrain X-Vector**（如原训练用的确实是它）：

```bash
COUNT=32 ENCODER=xvector \
  MODEL_PATH=/data/coding/sgmse/pre_models/speechbrain-xvector \
  bash run_rrsd_feature_stageA.sh
```

可设 `DEVICE=cpu`（很慢）、`MAX_PLM_LENGTH=32000`（**建议不要改，保持与原 CL 一致**）、`ROOT=.../rrsd_tcut_screen`。若使用 UniSpeech-SAT，单条语音约 2 秒，合计每条 7 次模型前向；先 4 条检查显存/速度。

## 3. 实际比较的音频

同一条语音的 `clean`、`off`、`router`，及以下四种由 `clean` 直接生成的合成退化：

- 加独立高斯残余噪声，25 dB SNR；
- 使用**同一个噪声波形**增大到 15 dB SNR，测试单调敏感性；
- 3.2 kHz 低通滤波（6阶 Butterworth）；
- 中心处置零 50 ms 模拟局部缺失。

这些是诊断扰动，**并不严格等价于 DNSMOS 下降的失真类型**。不能只因为干净语音经人为大幅破坏后特征变化，就宣称特征能预测 DNSMOS。音频各自去均值+峰值归一化，故对整体增益变化不敏感；这与现有 CL 的波形预处理一致。

## 4. 输出

```
experiment_logs/rrsd_tcut_screen/feature_stageA/<encoder>/
├── feature_distances.csv    # 每条语音、每种特征的全部配对距离
├── feature_report.txt       # 均值、中位数、bootstrap CI、噪声剂量敏感性、质量指标相关
└── console.log
```

优先阅读以下字段：

1. `Delta d=Router-Off` 的均值和 CI：负值意味着 Router 与 Clean 在该特征下更接近；正值则更远。
2. `Noise-dose sanity`：15 dB 比 25 dB 更大的特征距离是否占明显多数。
3. 隐藏层 `hidden_mean` 与身份嵌入 `speaker_embedding` 的敏感性差异。
4. `Spearman(Delta feature distance, Delta pesq)`：理想方向通常是负号，但 n=32 的相关会很不稳定。显著性和因果关系不能由此确认。

**适用性判断：** 如果 `hidden_mean` 对不同类型失真呈合理剂量响应，并且与配对 PESQ 的变化方向有一定一致性，则可以优先拿它试 Router 的相对特征约束。若只有 `speaker_embedding` 稳定，且对明显退化几乎不敏感，它更适合身份保护而非 DNSMOS 感知约束。两者若都对质量变化不敏感，暂停 RPC 训练。

## 5. 重要限制

- 这只是**阶段 A 的筛选**，不涉及梯度回传或损失权重。
- 使用的是**已训练的冻结编码器**，不是新训练的质量评估网络。
- 32 条样本上的 bootstrap CI 是探索性不确定性描述；不能当作普遍结论。
- 离 DNSMOS 是否提高仍有间隔；如果可能，可增加已跑通的 DNSMOS per-utterance 分数做进一步验证。
- 如果“你的预训练模型”是自训练的 checkpoint（而不是这里的公开 UniSpeech-SAT/CAM++/XVector），需要提供该 checkpoint 的模型构造/forward 接口，不能将其直接当作官方预训练目录加载。
