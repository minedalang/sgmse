# CPEL收口后的瓶颈复盘：同例重建—预测质量冲突

> **范围纠正（用户指出后）**：本Router复盘由助手自行扩展，并非CPEL验证。现已停止这条新增支线，不作为CPEL结论依据或当前后续工作授权。下文保留历史事实和当时推断，关于“下一候选”的建议不再作为当前执行安排。

## 已完成的工作与决策

M/P、N/D及public迁移已完成；当前固定pair配方关闭，不追加步数、权重、窗口或接口选择。本复盘不生成新波形、模型评价或训练，不接触最终测试集。它回答一个更小但会改变动作的问题：旧Router的质量代价是否仅由少数尾部负例或不同句均值混合造成？

Root与独立Lead分别按既有CSV复算，答案是否定的：**同例冲突在此CV中很普遍**。因此不以“过滤少量坏例”叙事重开selector、tail-risk或SNR门控，也不再新增质量×CAM++相关表。尚未定位具体语音失真，不足以指定新loss或控制器。

## 精确对象与独立复算

对象是20261017固定64speaker、每人2句的128句开发CV；同80k EMA checkpoint、相同seed2026和PC N30，比较Router scale1与scale0。这里scale0仍执行Router路径，**不冒充原生bypass，也不是F对后端，更不是CPEL N/D**。原基座训练历史为300k基础系统＋80k Router-only增量。

按两表`filename`严格等值联结，128项均唯一、无缺失：

- `experiments/speaker_headroom_replication_64_20261017/quality_per_utterance.csv`，SHA256 `a8300cc9c90d063259556ec340470bebbdcd84dcf1b105971c1e12e20dc2a1a6`。
- `experiments/distillmos_surrogate_independent64_20261020/per_file.csv`，SHA256 `bb0b9bd7d9939a7a8a8ab86b2bec0540f4c24946ca7a5e729d2069d8ca720e13`。

定义每项差为Router减off，严格比较符号，不设新的效应大小门：

| 同一条记录满足的条件 | 条数 |
|---|---:|
| SI-SDR差大于0 | 128/128 |
| SI-SDR差大于0且DNSMOS-OVRL差小于0 | 105/128 |
| 上一行同时Distill-MOS差小于0 | 83/128 |

SI-SDR最小增益+.1251378dB、中位+.4758603dB；OVRL差中位−.0182389。先file内再按speaker求OVRL差均值，60/64个speaker为负。Root核两表读取前后SHA一致；Distill原`distill_delta`与`router_distillmos-off_distillmos`相符。机器记录见[p8_post_cpel_bottleneck_counts_20261020.json](p8_post_cpel_bottleneck_counts_20261020.json)。复算只用Python标准库，没有音频、权重、重抽样或新增指标推理。

这些是反复探索CV的描述性符号计数，不是独立确认、总体概率、显著性或可感知效应大小。DNSMOS和Distill-MOS都是预测代理，同降不是两份独立人类质量证据，更不证明身份丢失或幻觉。

## 负结果范围必须保真

- [P3a](../experiments/perceptual_representation_probe_20261017/experiment_report.md)显示简单听觉/谱距离不能可靠共同排序重建和DNSMOS，不普遍否定感知监督。
- [P3b](../experiments/p3b_blind_listening_pilot_20261017/experiment_report.md)24次全tie比较Router、t-cut、selective，**没有Router对off**；不能将其解释为本次冲突在人耳中不存在，也不证明等效。无需要求用户补听。
- [H4](../experiments/h4_distortion_diagnostic_20261018/experiment_report.md)没有找到跨对照且包含DNSMOS的稳定伪影解释，不能凭谐波相关重开GAN/伪影loss。
- [Distill检查](../experiments/distillmos_surrogate_independent64_20261020/experiment_report.md)不支持将其直接作为独立reward。再与CAM++相关也不能定位干预，且CAM++前2秒与质量30秒的范围不同。

## 未解问题、下一动作与退出条件

尚未回答的是：同句重建更好、预测SIG/OVRL更差时，具体什么语音失真被改变？当前表格不能区分细节抹除、残噪变化或时相/纹理异常，不能把任何一种写成原因。

**当前不开新训练，也不为显得推进而造另一张关联表。** 后续仅审查能区分至少两种具体误差来源、具有可部署意义的假设。方案必须说明现存证据、不同结果如何改变下一决策、最低调用成本与反证退出；若只是重复证明代理冲突、改善同一代理或换名复活旧门控/anchor/多seed，则不进入执行。预测质量代价的归因与独立内容恢复是不同问题，不将其中任一证据缺口泛化为整个goal阻塞。

同时，摘要、绪论、结论已最小同步M/P基线范围、N/D失败及public微小增量。文档保护及转换检查见[检查记录](../thesis/cpel_scope_sync_checks_20261020.json)，浏览器、TeX和学校模板仍未验收；文稿同步不替代科研成功。
