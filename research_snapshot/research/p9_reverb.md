# P9：混响数据建设、训练与机制验证

> **状态**：P8 之后的第二阶段；**当前本地缺少可复用的真实训练 RIR，80k checkpoint 未记录明确的 with-reverb 训练/增强配置，尚未启动新混响训练。** 本文件是条件式路线图，不是资源下载或训练许可。总入口：[research_plan.md](../research_plan.md)。

## 1. 已知事实

- 官方冻结 DNS synthetic 测试集 `no_reverb` / `with_reverb` 各150条。为避免将“Router减off”误读成名为Router-off的单一模型，交互明确记为：`[metric(Router,with_reverb)−metric(off,with_reverb)]−[metric(Router,no_reverb)−metric(off,no_reverb)]`，按相同file-id配对。PESQ约`−0.04110`、STOI`−0.00267`、SI-SDR`−0.15524dB`、DNSMOS-OVRL`−0.02854`，对应主要matched-file CI排除零。**这是Router相对off效果的条件交互，不是off模型自身的混响前后差，也不是已定位的失效因果机制。**
- 作为不同量，原JSON中off自身的`with_reverb−no_reverb`均值为PESQ`−1.71449`、STOI`−0.61558`、SI-SDR`−31.77345dB`、OVRL`−0.84537`；不得拿这些绝对条件差替代上述交互。本次仅澄清现有字段与简写，原报告/指标/测试集未改，未新增混响实验。
- 4 条 fresh-CV 上的临时 short/medium FIR smoke 没有复现官方测试集的完整退化模式，不能据此设计 severity gate。
- 现有 80k checkpoint 保存的配置指向 `tr_synthetic/cv_synthetic` noisy SCP，没有可确认的 with-reverb 配对训练条目；不能因此推断所有历史 checkpoint 都从未见过混响，其他权重须逐个审计。
- 旧 RIR 路径在本机不存在，临时 FIR 不是实测房间 RIR。此前 `/data` 余量报告约 6.2 GB，重新启动前必须复查实际可用空间。

出处：`experiments/final_eval_reverb_interaction_20261018/`、`checkpoint_data_condition_audit_20261018/`、`reverb_cv_smoke_20261018/`、`reverb_resource_audit_20261018/`、`reverb_severity_posthoc_20261018/`。

## 2. 资源门：先有合规 RIR，才谈训练

- 仅检查许可证、来源、可重复获取方式和数据容量；候选可以考虑 OpenSLR [SLR28](https://www.openslr.org/28/)（实测/模拟 RIR）和 [SLR26](https://www.openslr.org/26/)（模拟 RIR）。公开可用不意味着已下载或已同意相关条款。
- 确定足够多样的房间、RT60、DRR、源—麦距离、噪声空间传播与房间反射。**同一个 RIR 卷积几千条语音不等于独立房间泛化。**
- 先冻结按**房间/RIR、speaker、noise** 来源隔离的 train/dev/holdout；官方冻结测试集中的 RIR/混响不能混入训练、开发或选择阈值。
- 用在线 convolution + mixing + deterministic seed/manifest 比预先存几十万条 WAV 更能控制磁盘；但实现前需明确 CPU/GPU、I/O 和恢复空间预算。暂不下载、不清理有效模型/数据、不生成批量样本。

## 3. 提前锁定增强目标

| 任务 | 增强目标 | 重要约束 |
|---|---|---|
| 只降噪，允许残留混响 | `reverberant clean`（wet） | 不能用干声参考暗中评价去混响 |
| 同时降噪 + 去混响 | `dry clean` | 必须处理 RIR 直达声时延、尾音/截断和参考对齐 |

记录噪声是否空间卷积、SNR 的计算范围、RMS/峰值缩放、相位/时序对齐及训练前端与后端的参考定义。RIR 数据域不匹配时，不能只用一个新 gate 掩盖训练分布问题。

## 4. 分阶段路线（未来需另行批准）

1. **独立新开发集**：小规模真实 RIR，固定 room/source/noise 划分，比较 `Y_pre`/Router-off/Router 与无混响对应条件，先证实新集真实复现问题。
2. **等预算训练对照**：无混响继续训练 vs 加 RIR 数据继续训练；分别检查前端、后端或联合训练的必要性，保持初始化、步数、模型、EMA、采样 NFE 与硬件预算匹配。
3. **内容与质量联合评估**：按房间、RT60、噪声/SNR 分层报告人工内容转写、辅音删除/插入、PESQ/STOI/SI-SDR、DNSMOS 和计算成本，不以单一均值掩盖不同房间的退化。
4. **再问是否存在方法创新**：如果普通数据域匹配已经解决问题，应当诚实归为训练数据修正；只有独立数据及匹配训练后仍有明确模型机制缺口，才提出新条件/Score/采样方法。

## 5. 当前停止条件

没有合规 RIR、独立真实房间集、明确 wet/dry 监督目标或足够磁盘空间时，不启动大规模合成/训练。P8 低 SNR 内容忠实性仍为第一优先级。**简单卷积、局部混响 mask 或普通 reverb gate 不能直接称算法创新。**
