# RRSD × ICASSP 2024 CCR: A/B方案第一阶段（免训练）

本文献: *Curricular Contrastive Regularization for Speech Enhancement with Self-Supervised Representations* (ICASSP 2024, DOI 10.1109/ICASSP48485.2024.10445912)。论文 Table 3 的 WavLM Base、UniSpeech-SAT Base 与本方案实际 Hugging Face 权重版本可能不同，不视为严格复现。

## 输入

默认读取 `experiment_logs/rrsd_tcut_screen/` 的现有 32 条 `clean/`, `off/`, `router/`, `manifest.csv`, `paired_metrics.csv`。无需重新推理或训练。固定 16kHz、2s、逐条语音 DC removal + peak normalization；人工 15/25dB 噪声、3.2kHz 低通、50ms 缺失使用统一随机种子。使用相同语音配对。

核心：每个编码器每层的**整个时间特征张量逐帧 MSE**（论文最后一层 MSE 的候选实现）；附加逐帧 cosine 作为敏感性分析，绝不称为论文原实现。Transformer 隐藏层 1..12，主要关注第12层，3、6、9层是额外消融。不同模型之间绝对 MSE 尺度不相同，只比较变化方向、噪声剂量敏感度和与指标的配对关系。

## 操作

解压 `SGMSE_RRSD_CCR_论文对照_阶段B工具包.zip` 到 `/data/coding/sgmse`。在已安装 transformers/torch/scipy/soundfile 环境下：

```bash
cd /data/coding/sgmse
python -m py_compile rrsd_ccr_ssl_feature_compare.py

# 先复现已经缓存的官方 UniSpeech-SAT-SV：4条，再32条
COUNT=4 NAME=sat_sv_4 MODEL_TYPE=unispeech_sv MODEL_PATH=microsoft/unispeech-sat-base-plus-sv bash run_rrsd_ccr_ssl_screen.sh
COUNT=32 NAME=sat_sv_32 MODEL_TYPE=unispeech_sv MODEL_PATH=microsoft/unispeech-sat-base-plus-sv bash run_rrsd_ccr_ssl_screen.sh

# 官方 WavLM Base+ (实际是 Base+，并不是论文 Table 3 的 WavLM Base)
COUNT=4 NAME=wavlm_plus_4 MODEL_TYPE=wavlm MODEL_PATH=microsoft/wavlm-base-plus bash run_rrsd_ccr_ssl_screen.sh
COUNT=32 NAME=wavlm_plus_32 MODEL_TYPE=wavlm MODEL_PATH=microsoft/wavlm-base-plus bash run_rrsd_ccr_ssl_screen.sh
```

**离线服务器注意**：默认 `LOCAL_ONLY=1`，若 WavLM 未缓存将明确报错。这种情况需事先通过联网机器下载官方 `microsoft/wavlm-base-plus` 到服务器目录，并将 `MODEL_PATH=/path/to/wavlm`；请勿设置 `LOCAL_ONLY=0` 后假设断网服务器可以自动下载。建议用 `huggingface_hub.snapshot_download` 在联网机器下载，然后 scp 整个目录（配置和权重必须齐全）。

输出在 `$ROOT/feature_ccr_paper/<NAME>/` 的 `report.txt`, `frame_features.csv`, `console.log`。每组对应层和 MSE/Cosine 的结果。把 SAT、WavLM 两份 `report.txt` 发回即可。

## 如何决策

重点先看最后一层 MSE 的 (a) 15dB > 25dB 是否单调，(b) 对低通/短时缺失是否敏感，(c) Router-Off 距离变化是否与 PESQ/STOI 方向冲突，(d) 逐条 Δ 距离和质量指标的关系。32 条只能筛选，不能宣布统计显著改善 DNSMOS。需进一步在独立数据上检验实际 DNSMOS 的感知一致性。

**不要现在做多步 PC 训练**。如果没有可靠 SSL 质量信号，停止路线；若信号有用，再进行冻结编码器、冻结前后端、只更新 Router 的 1500–3000 step 配对 control/CCR 微调。
