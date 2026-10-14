# SGMSE SpeechBrain X-Vector 接入（不修改 site-packages）

本包基于 `SGMSE_CAMPlus_完全独立整合版.zip` 添加 X-Vector，保留原 UniSpeech-SAT 和 CAM++ 的对比学习路径、双向 InfoNCE、`contrastive_speaker_weight`、RRSD/Future-grad，不新增对比学习计算间隔参数。

## 1. 安装 SpeechBrain

先检查是否已安装：

```bash
python -c "from speechbrain.inference.speaker import EncoderClassifier; print('SpeechBrain OK')"
```

如果没有：

```bash
python -m pip install --dry-run speechbrain==1.1.0
# 确认不会换掉现有 torch / torchaudio 后再执行：
python -m pip install speechbrain==1.1.0
```

不要为了安装本插件直接升级/降级已有的 PyTorch。如果当前环境与 SpeechBrain 仍存在兼容性错误，先贴具体 traceback 排查。

## 2. 准备官方预训练权重

官方预训练模型：https://huggingface.co/speechbrain/spkrec-xvect-voxceleb

服务器能联网时：

```bash
hf download speechbrain/spkrec-xvect-voxceleb \
  --local-dir /data/coding/sgmse/pre_models/speechbrain-xvector
```

服务器不能联网时，手工下载模型仓库中的全部文件（包括 `hyperparams.yaml`, `embedding_model.ckpt`, `classifier.ckpt`, `mean_var_norm_emb.ckpt`, `label_encoder.txt`）并上传同一目录。不要把 CAM++ 的 `avg_model.pt` 填进 X-Vector 模型路径。

本适配器对于本地目录会覆盖 SpeechBrain 超参数中的 `pretrained_path`，以便真正从本地离线加载。

## 3. 梯度检查（正式训练前必须先通过）

```bash
cd /data/coding/sgmse
python -m scripts.check_xvector_grad \
  /data/coding/sgmse/pre_models/speechbrain-xvector
```

期待 `embeddings: (2, 512)`、增强波形 grad norm 非零、`PASS`。与之前 CAM++ 一样，只有增强分支保留 autograd，干净分支不需要梯度。

## 4. 训练

在之前的训练命令基础上，将旧的 `--plm_type` 和 `--plm_name_or_path` 参数替换为：

```bash
--plm_type xvector \
--plm_name_or_path /data/coding/sgmse/pre_models/speechbrain-xvector \
--contrastive_weight 10 \
--contrastive_temperature 0.1 \
--contrastive_clean_weight 1.0 \
--contrastive_speaker_weight 0.5 \
--max_plm_length 32000
```

普通非说话人感知 InfoNCE：将 `--contrastive_speaker_weight 0`；训练脚本其他参数原样保留。**用 `--init_ckpt` 加载原始 300k 基线，并开一个新实验，不要用 `--ckpt` 直接恢复另一个编码器训练的 optimizer/checkpoint。**

## 5. 单独测 X-Vector 的推理和训练式开销

```bash
python -m scripts.benchmark_speaker_encoders \
  --plm_type xvector \
  --model /data/coding/sgmse/pre_models/speechbrain-xvector \
  --device cuda:0 --batch-size 16 --length 32000 \
  --warmup 3 --steps 12 --amp off
```

脚本输出：
- `INFERENCE 1 forward`: **X-Vector 特征提取器单次前向推理**，不计算梯度。
- `Enhance forward` + `Clean forward` + `Backward`: **一次 InfoNCE 对比学习中两次编码、一次输入反向传播**的近似计算开销（采用 MSE 代替 InfoNCE 本身，小型矩阵 loss 的时间不计）。
- `TRAIN 2fw+1bw`: 以上三项之和；此脚本 **不包含 STORM/NCSN++、STFT/iSTFT、DataLoader、分布式同步**。

对照组（保持完全相同的 GPU / batch / 长度 / 精度 / 预热 / 步数）：

```bash
python -m scripts.benchmark_speaker_encoders \
  --plm_type campplus \
  --model /data/coding/sgmse/pre_models/wespeaker-voxceleb-campplus \
  --device cuda:0 --batch-size 16 --length 32000 --amp off

python -m scripts.benchmark_speaker_encoders \
  --plm_type unispeech_sat \
  --model microsoft/unispeech-sat-base-plus-sv \
  --device cuda:0 --batch-size 16 --length 32000 --amp off
```

如果正式训练使用 BF16，可将三条命令都替换为 `--amp bf16` 进行第二轮测试。请空闲 GPU 上依次运行，避免同时运行训练进程干扰测量。

**注意**：这里的推理是*说话人嵌入提取推理*，不是 SGMSE 语音增强推理。CAM++ / UniSpeech-SAT / X-Vector 仅用于训练损失时，部署增强模型的前向推理速度基本不受影响。

## 6. 验证范围和依赖

源码参考 SpeechBrain 官方模型定义（24-bin FBank + 5 TDNN blocks + 512-D embedding）；不复制或修改 SpeechBrain 源码。预训练权重由 `speechbrain` 加载，仅选用 `.mods.compute_features`、`.mods.mean_var_norm`、`.mods.embedding_model`，避免其高层 classify_file 的不必要开销。引入 SpeechBrain (Apache-2.0) 外部 Python 包。

已使用安装的 SpeechBrain 1.1.0 以**合成网络权重**验证编码、梯度回传与冻结。尚未在用户 GPU / 真实 VoxCeleb 预训练权重上跑通；请以第 3 节实测为准。测速脚本只提供实测方法，**不承诺一定比 UniSpeech-SAT 快**。
