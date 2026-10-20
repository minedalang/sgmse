# P8-0 训练/CV 分布覆盖审计（2026-10-18）

## 目的与边界

本审计检查 80k checkpoint 对应的 synthetic train/CV manifest、音频头信息和实际 `joint_scp` 数据加载路径，回答低 SNR、说话人、时长和噪声清单是否覆盖。它是只读审计：没有读取冻结测试 manifest，没有训练、推理或重写 `/data/datasets/dns_ins20`。

这里的 `clean_fileid_*.wav` 是 DNS 合成任务的配对目标文件。它们不是本机保留的 DNS 原始 clean 语料，也不带可追溯的逐句 transcript 或音素标注；因此本审计不能把这些 WAV 当作自然语料库来构造或确认最小音素对。

名义 SNR 从 noisy 文件名的 `_snr-XX` 解析。没有对每个文件做功率估计，因此 **实际 SNR 记为 unknown**；不能把下面的名义分布称为实测局部 SNR。

可复现脚本、配置、逐条记录和汇总分别位于：

- `audit_training_coverage.py`
- `config.yaml`
- `metrics.json`
- `tr_synthetic_records.jsonl` / `cv_synthetic_records.jsonl`

## 数据规模与 SNR 覆盖

所有可读的 train/CV 音频均为 16 kHz、30.0 s，音频头和清洁/噪声配对没有缺失或时长不一致。

| split | utterances | 总时长 | speakers | noise_fileid 数量 |
|---|---:|---:|---:|---:|
| train (`tr_synthetic`) | 10,800 | 90.0 h | 1,470 | 10,800 |
| CV (`cv_synthetic`) | 1,200 | 10.0 h | 158 | 1,200 |
| 合计 | 12,000 | 100.0 h | 1,628（split 并集） | 12,000 |

按优先区间统计（区间右端点规则为 `[-20,-18)`, `[-18,-15)`, `[-15,-10)`, `[-10,-5]`）：

| nominal SNR 区间 | train | CV | 合计 |
|---|---:|---:|---:|
| `[-20,-18)` | 1,393 | 163 | 1,556 |
| `[-18,-15)` | 1,930 | 219 | 2,149 |
| `[-15,-10)` | 3,395 | 359 | 3,754 |
| `[-10,-5]` | 4,082 | 459 | 4,541 |

因此训练和 CV 都覆盖了 −20 到 −6 dB 的全部文件名 SNR 档位，但分布偏向较高的 `[-10,-5]` 区间；该结论仅针对文件名名义 SNR。完整每个 dB 档位计数见 `metrics.json`。

## 说话人和 split 独立性

train 有 1,470 个 speaker，CV 有 158 个 speaker。utterance ID、file ID 和 `noise_fileid_*.wav` 在两个 split 间均无重叠；但 speaker `10658` 同时出现在 train（3 条）和 CV（7 条），所以这不是严格 speaker-disjoint 的 CV。其余 speaker 集合不重叠。该重叠会削弱以“未见说话人”解释 CV 结果的依据，应在后续泛化分析中单独标注。

每位 speaker 的 utterance 数量并不均匀：

- train：每位 1–33 条，中位数 6；167 位只有 1 条，300 位不超过 2 条。
- CV：每位 1–24 条，中位数 7；17 位只有 1 条，33 位不超过 2 条。

这说明总体 speaker 数量较多，但少数说话人的覆盖很稀疏；不能只用 speaker 总数代表每个 speaker 都有足够的低 SNR/辅音样本。

## 噪声清单覆盖

`noise1.scp` 为每个 utterance 提供唯一的 `noise_fileid_<id>.wav` 路径：train 10,800 个、CV 1,200 个，两个 split 间无 basename 重叠。这个 ID 数量表示合成噪声实例的覆盖，不等于语义噪声类别数量。

从 noisy 文件名提取到的弱标签只能作线索，不能视为数据集的正式噪声类别标注。train/CV 中都有 `door`、`fan`、`typing`、`squeak`、`breath` 等 token，且大量文件名没有这些可解析 token（train 6,057 条、CV 646 条）；混合噪声文件名可能含多个 token，`squeak`/`squeakyChair` 也会被同时命中。因此本审计不声称已经完成语义噪声类别覆盖。

## 实际训练调用链

checkpoint 的保存超参数记录了：

- `train_wav_scp` / `valid_wav_scp` 指向原机器上的 `scp/tr_synthetic` / `scp/cv_synthetic`；当前可读的 `scp_new` 清单与其 utterance ID、顺序和音频 basename 一致，差异仅是路径前缀。
- `clean_dir` 与默认模板 `clean_fileid_<fileid>.wav` 用于匹配 clean。
- `frontend=storm`、`train_type=mode2`、`sde=ouve`；本审计不重新加载 checkpoint。
- 旧 checkpoint 审计记录 `contrastive_weight=0`，因此默认训练 loader 使用普通 shuffle，而不是 speaker-balanced contrastive sampler。

代码路径的关键行为如下：

1. `joint_train.py` 在 `--dataset joint_scp` 时实例化 `JointWaveScpDataModule`。
2. `WavePairsScp` 读取 `wav.scp`，按 file ID 解析 `clean_fileid_<id>.wav`；只有显式传入 `train_clean_scp`/`valid_clean_scp` 时才读 `spk1.scp`。
3. `noise1.scp` 不在 `WavePairsScp.__getitem__` 中读取，因此当前训练不是每步重新抽噪或重采样 SNR，而是直接加载预先合成的 noisy WAV。
4. 训练长于目标片段时随机裁剪；CV 使用中心裁剪；clean/noisy 以同一边界对齐，并默认按 noisy 峰值归一化。

## 诊断结论和限制

审计没有发现清单缺失、音频采样率/时长配对错误或名义低 SNR 档位缺失。可作为 P8 内容问题竞争解释的分布事实是：训练确实包含大量名义 −20 到 −10 dB 样本，但 `[-20,-18)` 仅占 train 1,393/10,800，且辅音级、局部实际 SNR 和音素文本没有在训练清单中标注；无法据此证明弱辅音已经充分暴露或模型看见了正确的内容证据。

另一个需要保留的验证风险是 CV 存在一个与 train 重叠的 speaker（10658）。后续独立内容验证应使用新的 speaker/音频，并把训练覆盖与人工音素证据分开报告。只有在有真实文本或盲转写后，才能判断“吞音”“补能量但未恢复音素”等内容结论。

## 复现

在 `/data/coding/sgmse` 下运行 `config.yaml` 中的 invocation 即可重建 JSON 汇总和逐条记录。脚本只写本实验目录；它不会写入原始数据、checkpoint 或冻结测试目录。
