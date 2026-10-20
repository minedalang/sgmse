# P8-CPEL：反事实成对声学证据训练候选

> **当前：唯一v2 M/P各500步及64例评价均已完成，未达到预定pair增量信号，关闭本配置。** 没有实质增强、辅音恢复或防幻觉收益证明；不追加调权重/加步/换窗，不据此扩大A/B训练。

## 科学问题与范围

在原生DSM之外训练受控的Noisy局部变化响应，是否比相同样本、共同输入和逐点辅助任务更有价值，并迁移到正常前端/正常采样的真实内容保真？当前后端已经接收Noisy，不能把有条件影响与充分/正确利用混为一谈。

这是训练期假设，不是P8-V外部候选verifier。自动声学成分不是音素标签；配对相减具有训练特权，不能当单观测可辨性。后续真实音素、身份、错误新增、音质和效率要求不因这个小试被缩小。

## 已取得的依据

- [合成目标原型](../coding/sgmse/experiments/p8_cpel_objective_prototype_20261019/experiment_report.md)只验证算术/梯度，明确零clean差分不会自动惩罚noise直通。
- [真实接口attempt03](../coding/sgmse/experiments/p8_cpel_mp_pilot_20261019/preflight_attempt03/preflight/experiment_report.md)实测34Score/2反向、0更新、约9.1GiB峰值，参数/buffer不变。旧Ninja PATH和DataModule构造失败均保留，未因此修改strict、样本或损失。原生固定Fourier W的训练范围问题随后已修复。
- [训练内尺度测量](../coding/sgmse/experiments/p8_cpel_train_scale_calibration_20261019/experiment_report.md)在正确原生可学习集合上完成8记录×2draw、64Score/32grad、0更新。r=||grad pair||/||grad M||中位.0007206、最大.2042；global lambda13.877虽给中位1%，尾部达2.834，建议未应用。16个相关probe不是总体或Adam更新保证。基线响应在这16次均同向，lower hinge仅2次生效：不能叙述成“首次激活完全忽略的Noisy通道”。
- [64dev编码诊断](../coding/sgmse/experiments/p8_cpel_codec_consistency_20261019/experiment_report.md)保留原strict .001的43pass/21fail；全部64同时与nearest/general两种量化误差界相容。均PCM16、无crop码端点，但处理链仍未验证，不能由此证明noise纯度或撤销旧失败。

## v2只做一次有辨识价值的小试

[固定方案](../coding/sgmse/experiments/p8_cpel_mp_pilot_20261019/analysis_plan_v2.md)／[配置](../coding/sgmse/experiments/p8_cpel_mp_pilot_20261019/config_v2_draft.yaml)。256train/64既有CV四个名义SNR组均衡，排除旧181窗全部8个file ID和speaker10658；speaker/fileid/clean/noise实例不交叉。CV非新确认集，noise录音/类别来源未知。

共同背景编辑在波形进行，两臂同裁窗/Noisy标度，再分别变换；目标为两clean变换之差，不是Phi(Delta)。保留合法原生DSM，共同parent前端/state仅作辅助响应，不能照搬同epsilon标签或称无偏DSM。共同条件仍有失配，完全原生对应与严格固定前端一般不可兼得。

M/P同4Score、两次grad遍历、一次Adam/clip/EMA；只训练原生可学习Score参数，F与固定Fourier W不动。P采用一次明确的动态原始梯度更新：eta=min(100,.01||gM||/||gp||)，无下限，零/inactive设0、非finite失败；M做同样计算但应用eta0。它只限制当前参数向量clip/Adam之前的pair分量≤1%，不是静态全局loss完整导数、Adam预算或声学可信权重，不作为独立创新贡献。

noise实例诊断并列strict/nearest/general/处理链字段，codec-compatible只作工程扰动解释，不作真负类或FPR。原PCM与float编辑可能有精度格点捷径，所以在raw Tukey之外固定增加两支同PCM16量化的heldout packet，重算实际target与残差变化；不改训练编辑或正常推理，也不假称量化后严格同noise。[编码与捷径决策](../coding/sgmse/research_memos/p8_cpel_quantization_and_artifact_controls_20261019.md)。

单seed、每臂最多500更新/3600秒、总≤2GPU小时；新产物≤512MiB且留1GiB。仅终止EMA，不扫lambda/时间/窗型或按CV选checkpoint。正常64dev质量/重建与raw/精度匹配响应/噪声实例响应同时报告；只在raw编辑变好、正常无迁移、追随noise或M已解释收益则不扩大。不追加预检/标定或用户试听。Root签唯一v2 M/P许可后，Worker已完成两臂并收口；本节为实施方案记录，终态与退出决定见下节，不构成重跑许可。

## 唯一v2终态与退出决定

[完整结果](../coding/sgmse/experiments/p8_cpel_mp_pilot_20261019/experiment_report.md)／[独立数值核查后的解释](../coding/sgmse/research_memos/p8_cpel_mp_pilot_interpretation_20261020.md)。M/P均500更新，全部64正常例及三个speech packet双parent覆盖完整，前端与固定W不变。P相对M：raw Tukey pair下降1.67035%，precision下降1.65827%，均未达预定≥10%。M本身相对initial已下降约11.35%/11.28%，不可计入pair贡献。P−M正常PESQ+.00013913、STOI+.000004568、SI-SDR−.00028658dB，未越损伤界但未建立实质收益。

pair主要由orth形状项主导，P投影gain略降，不是激活此前没有的Noisy通道或恢复更多辅音。noise工程37例/strict30例未越损伤界，另27例能量不足未评，不算拒绝假音成功。两臂总746.71秒包含不对称的initial评价，不能比较训练速度；500步数据/随机量完全配对、1%原始梯度预算实测有效，不代表Adam更新预算。

**关闭当前配置，遵守预定退出，不扩大、不扫权重/时间/窗型或换seed救结果。** 结果仅限单seed短程相位感知响应正则，未否定全部CPEL；未来重提须先提出不同且有自然内容终点的可证伪机制。完整普通增强对照和真实身份/错误新增仍未证实，本轮不足以立即启动该成本。

## 完整贡献仍未证明

A=原生DSM、B=普通反事实增强的完整对照仅在M/P出现真实增量信号后考虑；M/P正向也不能直接声称超过普通增强。真实内容/身份与未编辑输出不可被连续谱代理替代，最终标准测试在方案冻结后评价，旧探索池不升级确认集。DOSE/CoF/CADT/一般导数监督的先例边界仍有效；详见[独立评估](../coding/sgmse/research_memos/p8_counterfactual_paired_training_assessment_20261019.md)。

## 原始问题保真补正（恢复后）

旧固定M/P的关闭不变，但不将其外推为整个CPEL或普通DSM之上pair的问题已失败。M含权重1共同state clean回归，其精确分解已监督gain1差分及orth；用户原B普通DSM不含此任务。此前把局部负筛查扩大为无独立输出读数就不能继续研究，属于过强封闭，已撤回。

[详细推导](../coding/sgmse/research_memos/p8_cpel_question_fidelity_20261020.md)。当前仅审查一个不同的缺失对照N/D（aux0）：普通反事实DSM vs 同DSM+原pair，旧M/P原样作为aux1历史参照；不增加步/权重/换窗救旧配置，未批准新训练。哪怕机制代理通过也不等于真实辅音恢复/防幻觉，独立内容评价是强主张待办而非所有科学动作的统一门。
