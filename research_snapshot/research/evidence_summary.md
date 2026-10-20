# 关键证据摘要：已知／未知／停止

> 只保留当前研究判断需要的数值和出处，不记录重复运行日志、完整哈希、后台进程与失败修复流水。**这是一份对用户上传计划的整理，不是对远端实验仓库的重新核验。** 要复核请按路径打开原始 `metrics.json`、`experiment_report.md`、用户反馈 JSON 或 [原文归档](archive/research_plan_full_20261019_044238.txt)。

**CPEL当前证据（非增强结果）**：[真实接口预检](../../experiments/p8_cpel_mp_pilot_20261019/preflight_attempt03/preflight/experiment_report.md)与[8记录训练内尺度测量](../../experiments/p8_cpel_train_scale_calibration_20261019/experiment_report.md)均完成、0更新；正确保留原生Fourier W固定。global lambda13.877的尾部有效梯度比达2.834，未应用；v2改为明确的一步原始梯度预算规则，不称静态loss或Adam预算。[64dev编码检查](../../experiments/p8_cpel_codec_consistency_20261019/experiment_report.md)原strict仍43pass/21fail，两量化界均64相容但处理链未知，不证明noise纯度。全部旧失败和科学限定保留于原目录。

**下一唯一小试**：[CPEL v2状态与方案](p8_paired_evidence_training.md)采用同预算M/P、raw及精度匹配编辑、正常输出和noise实例响应；配置与独立脚本已冻结；M已实际启动，P待M完整成功，具体阶段见运行记录，不再扩软件检查或试听。既有16次基线响应均同向，不能将目标说成激活完全未用的Noisy。新颖性、真实内容与综合收益仍未证明；[论文整合稿](../../thesis/manuscript.html)不把该候选写成结果。

## 听感记录直达

论文写作可从 [P8观察与局限讨论草稿](../../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md) 进入：对应当前原始指标与人工反馈，区分内容代理、条件消融、技术准备和未验证机制，不新增实验或独立听者结果。

| 你要找的对比 | 人耳结论与原话入口 | 音频页面 |
|---|---|---|
| **原始后端 vs 去掉直接 Noisy 语音条件的后端（VCV 五例）** | [五例听感汇总](../../experiments/p8_vcv_speech_component_ablation_20261019/listener_assessment_20261019.md)：1/3相似，2原后端更近，4两者t→k，5消融尾部/s/样声；后续高频噪声观察及原话JSON均有索引 | [五例试听](http://localhost:8766/p8_vcv_speech_component_ablation_20261019/listening/index.html) |
| **5243 `/d/` 局部去语音证据** | [听感及最新修正](../../experiments/p8_5243_local_condition_necessity_20261019/listener_assessment_20261019.md)：目标移除版/d/稍弱，不等于身份删除 | [5243试听](http://localhost:8766/p8_5243_local_condition_necessity_20261019/listening/index.html) |
| **VoiceBank A/B：前端 vs 原生后端** | [反馈原话JSON](../../experiments/p8_voicebank_native_content_pilot_20261019/attempt02/listening/user_feedback_20261019.json)，下面§1概述正/负观察 | [A/B试听](http://localhost:8766/p8_voicebank_native_content_pilot_20261019/attempt02/listening/index.html) |

这些是三组不同的已完成试听，不是新盲听结果；Noisy去语音消融仍保留原语音 `Y_pre` 锚点。入口补全不新增样本/听者，不要求重听，不覆盖原始记录。

## 1. 最新 VoiceBank 自然匹配内容 pilot（优先看人工反馈）

**来源**：`experiments/p8_voicebank_native_content_pilot_20261019/attempt02/`；4 句（p226、p287 各两句）× 两个历史噪声 × 全句名义 `-18/0 dB`，共 16 格。16 份 Frontend-only 与 16 份 Router-off，PC `N=30`；四句作者文本与缩放 clean 的小型 Whisper 校准结果为 0 词错误。但这不等于其他噪声条件下的 ASR 或人工识别无误。

| 量化代理 | Noisy | `Y_pre` | Router-off |
|---|---:|---:|---:|
| 全 16 格相对发布者原文微 WER | 66.03% | 59.62% | 80.77% |
| -18 dB 子集 | 100.00% | 96.15% | 129.49% |
| 0 dB 子集 | 32.05% | 23.08% | 32.05% |

- Router-off 相比前端编辑错误增量为 `+33`，其中识别器假设的 `I +28`、`D 30→10`。**不能把 I 直接叫真实新增词，或将 D 下降称为真实恢复**；低 SNR ASR 已接近失效。
- 四句仅两位说话人，重复使用同句及噪声；16 格不是 16 个独立自然句子，156 个评分词包含原 39 词在多条件重复。

**人耳 A：`p287_054/N1/0dB`**：用户认为前端和后端句末都无额外语音，音节差不多；前端高频较少，**后端音节质量好很多**。因此 A 不支持“由 ASR 末段差异推断实际听感内容损伤”。不能把整个 0 dB 子集的 ASR 相同 WER 用作人耳无差异结论。

**人耳 B：`p226_028/N2/-18dB`**：用户认为 `Y_pre` 丢掉很多音节，Router-off 有很多奇怪音节；开头 Clean/`Y_pre` 听起来无语音，off 有难懂音节。它是**人工观察到的风险候选**，但没有独立逐词身份或精确人工时间戳，不能立刻称“噪声被识别成特定词”。

**B 只读 CPU 开头复核已完成并关闭**：`experiments/p8_vb_caseB_opening_existing_audit_20261019/`；既有音频、实际 N2、raw/PCM/播放源匹配。前 0.5s 的 clean/noisy/`Y_pre`/off RMS 为约 `0.00067555 / 0.03340213 / 0.00043358 / 0.00308541`，off / `Y_pre` 为 `7.116×`、`+17.04 dB`；五个 100ms 格都增能，全句 off vs 前端仅 `+0.981 dB`。clean 前 0.5s **并非精确零波形**，听觉“无语音”不能替代逐样本全零判定。该复核不能区分人声生成、残噪、非局部前响或真伪音节；停止围绕 B 的新窗口、lag、频带、seed 与新 ASR 搜索。

幅度/字数的混杂复核：-18 dB 下后端 RMS 相对前端中位 `+1.895 dB`，ASR 字数 `55→103`；0 dB RMS 中位 `-0.008 dB`、字数 `77→77`，但编辑错 `18→25`。不能把 ASR 差异归结为单纯响度，也不能排除其影响。记录：`research_memos/p8_voicebank_native_pilot_interpretation_20261019.md`、`p8_native_gain_length_summary_20261019.*` 与 B 专项 memo。

## 2. DNS 与真实辅音信息的证据状态

| 问题 | 当前证据 | 正确边界 |
|---|---|---|
| 训练低 SNR 覆盖 | `tr_synthetic`/`cv_synthetic` 10,800/1,200 条，90/10 小时；文件名 SNR -20…-6 dB | 真实波形/局部弱辅音 SNR 分布未知；CV 有一位 speaker 跨 split |
| 有无真实句子文本 | 本地 DNS SCP `text=dummy`；原 clean 配对不是带原文的独立语料 | 不能以 Whisper clean 转写、音素 CTC 或伪边界算真实 WER/正确音素恢复 |
| `Y_pre` 抑制窗 | 181 个探索窗中 55 个抑制窗；局部 SNR 中位约 -20.41 dB，非抑制约 -8.30 dB | clean-aware 统计，只用于诊断；不能当部署时条件或证明输入无信息 |
| 10/20ms glimpse | 55 个抑制窗中约 24/19 个含至少一帧 clean-aware 局部 SNR 达 -10dB | 不绑定真实辅音身份，不能等同 `/k/` 或 `/d/` 可辨 |
| 固定时间 CTC | 89 个 clean-derived 伪目标中，noisy/`Y_pre`/off/Router 平均后验约 `.0176/.1168/.1018/.1080` | 时间漂移显著，单模型 posterior 不是真实音素身份，整体正负方向不稳定 |

数据和 CTC 记录：`experiments/p8_training_coverage_audit_20261020/`、`p8_text_source_audit_20261020/`、`p8_e1_candidate_ranking_20261018/`、`p8_q_a_shorttime_probe_20261020/`、`p8_q_a_ctc_fixed_time_v2_20261018/`。

## 3. 外部 VCV 与 Noisy 条件归因

**已有较强的操作性事实**：80k Score 的 Noisy 条件主分支确实生效，不是被完全忽略。但“直接 Noisy 起作用”不等于“解码出正确辅音”。

- 外部真实录音 VCV 主 clean 子集：四位说话人、8 对 k/g、t/d 共 16 条，冻结 wav2vec2 三 token 评分达到标称身份 `16/16`；另 12 条补充仅 `7/12`，不能概括全部语料。
- 固定噪声的名义 -6 dB 2s carrier 下，旧四位 speaker 的 N1 24 类 noisy/`Y_pre`/off/Router 为 `8/12/13/15`（分母16）；但语音实际支撑区平均 SNR 约 `+0.867 dB`，非极低局部辅音 SNR。
- 新四位 speaker 复验未复制 Router 的原有净收益；N1 -6 dB 前端/off/Router 为 `14/15/14`，还出现新的类别错误。**已按预注册停止**“Router 稳定提升辅音身份”的扩大叙事。
- g/k 条件互换的统一尺度 2×2：输出识别器的 anchor 主效应 LL margin 大致 `+7.8` 到 `+9.7`，直接 Noisy 条件主效应约 `-0.28` 到 `+0.54` 且方向不一；条件冲突仍 OOD，不能说模型“只信锚点”或 Noisy“没有信息”。
- 五例直接 Noisy 语音分量消融中，原先一些身份修正/损伤发生改变，说明 Noisy 条件敏感；但移除整段 VCV 包含强分布改变，不是只去掉某一个辅音。

路径：`experiments/p8_q_vcv_clean_calibration_20261018/`、`p8_q_vcv_lowsnr_challenge_20261018/`、`p8_q_vcv_speaker_replication_20261019/`、`p8_vcv_common_scale_factorial_20261019/`、`p8_vcv_speech_component_ablation_20261019/`。

**第五例尾音**：Clean 在 `0.9300625s` 后精确为零；直接 Noisy 条件移除语音分量后，4–7.5kHz 尾部能量比原后端约 `31.24/33.59×`。缓存识别器 `/s/` posterior 某帧约 `.672`，用户表述是高频噪声/摩擦样伪影。不是“模型确实凭空说了 `/s/`”的正式身份真值。`experiments/p8_vcv_f20_tail_existing_audit_20261019/`、`p8_vcv_f20_noise_correspondence_20261019/`。

## 4. 5243 `/d/` 的重要反向证据

`experiments/p8_5243_local_condition_necessity_20261019/`：后端原条件对目标 `/d/` 的 CTC P(d) 约 `.869617`；200ms 目标 Noisy 语音分量移除后反而 `.979222`；前、后位置控制 `.614505/.959156`。预注册基于 CTC 后验的局部必要性方向**没有得到支持**。用户的修正听感是“移除目标版 `/d/` 相比原版稍弱”，两者的声音强度与后验相反；并未完成“哪个音素身份更正确/更像 Clean”的独立判断。不得为证明 P8-V 或过抑制机制而重算同一例阈值或增加 seed。

## 5. 当前负结果、独立性与论文边界

- P8-M `Y_pre` 遮蔽：多 seed 下 SI-SDR/STOI 变差，不是 Noisy 成功独立恢复的证据；停止该 OOD smoke。
- P7-C `Y_pre`/Noisy 简单特征预测局部 NRMSE：speaker grouped OOF Ridge MAE `0.1050`，均值基线 `0.0975`；普通局部门控尚无稳定部署依据。
- P7/感知/身份/多轨迹/adapter 较多方案可改善某些 proxy，却没有同时过真实内容、质量、泛化和算力门；状态见 [archive.md](archive.md)。
- 官方 DNS `test_set/synthetic/no_reverb` 与 `with_reverb` 各 150 条已经用于最终评测且冻结；不得用结果继续调参。对新算法应另备独立、未反复用于筛选的确认集。
- P8-V 当前只是可证伪的新候选，不得把现有 Noisy 消融、人工尾声、VoiceBank WER 或文件重组当作验证成功。

## 6. 状态优先级与溯源规则

**冻结最终指标的配对不确定性补充已完成**：仅六份逐句CSV+原metrics，150matched-file保持两条件与三系统一起抽样，10000draw×14列主gather/独立bincount全验；主0.469s、复核0.103s，原均值最大差5.20e−18。新旧CI全部并列，Router SI-SDR新CI[+.201401,+.282815]dB、OVRL[−.021592,−.007543]、PESQ仍跨零，定性方向未变。只修重采样单位，不改变方法/指标/数据，不是独立确认、speaker/noise/roomcluster或多重校正，不混入12人耳案例。[报告](../../experiments/final_matched_file_uncertainty_20261019/experiment_report.md)，原新metrics SHA`cff1cb2ef8bf3919d64908d3e4f110945aef02acb8f0328dfb05265435f104a0`；无音频/模型/训练访问。

**已收到VB32中的12例内容反馈，本轮收口**：6句/4speaker，8例−18dB/4例0dB；20例未评不外推。checkbox frontend missing6/preserved4/unknown2，OFF missing4/different3/added1/preserved4/unknown1（多选不等于错误音素数）；双方preserved同四个0dB例，未见F missing→OFF preserved组合，不能将少音标签变少当恢复。质量OFF3/F1/相似5/U3，OFF3均在内容双方保留例。首例`p227_031_N2_m18`用户备注A在The前额外声音，A=OFF；与两臂原missing一起保存，不静默补added。详见`experiments/p8_vb_cached_content_review_20261019/responses/partial12_interpretation_20261019.md`。单人参照辅助、重复内容、非WER/phone恢复；不要求补20或重听，不启动调参/扩训。

**混响交互记号消歧**：专题中原`(Router-off)`表示相减，易与Router-off模型名混淆；现明确写为`(Router效果−off效果)_with−(Router效果−off效果)_no`。PESQ−.04110等是差中之差，不是off自身的条件差（PESQ−1.71449等）；两组原JSON字段分别保留，未更改原指标或重跑。[明确定义](p9_reverb.md#1-已知事实)。

**VB32参照辅助内容材料已完成（覆盖下方准备状态）**：`experiments/p8_vb_cached_content_review_20261019/`唯一生成0.100s，128软链入口/104原PCM的hash、头、映射及7本地GET通过；新增常规213,668B，原音频不复制/改写，0PCM解码/模型/ASR/训练。匿名A/B frontend置A14例、置B18例按固定hash未重平衡；不是完全盲测，质量另记。制作时0/32人类回答、浏览器未实测，设可选TODO，不等答卷；不是正式12听者WER评测或内容恢复证据。原metrics SHA`171bb9f017e4bb886c3b4108da75493e72056f16899594dc0c56e0c010f89871`。

**下一内容观察是新材料准备，不是新输出**：独立Lead认可复用VB32全部F/OFF缓存，另列单人参照辅助匿名A/B（与原12听者正式盲转写分开）。32case四臂共128入口/104源，0新增强/ASR/训练、原PCM仅symlink引用；内容标签需明确听到少音/新增/不同，单纯听不清归unknown，质量最后另记。方案在`experiments/p8_vb_cached_content_review_20261019/analysis_plan.md`，当前仅准备无回答，不因匿名AB称为独立盲测/WER/phone或CAEG验证。

**读出后的下一方案复审，避免重复小分类**：曾拟CCC旧32primary/8speaker的缓存k/g、t/d线性分类；独立Lead核原Noisy已具11/14、10/13（两组各两噪声，各16例）的标称身份可读代理，新模型成功/失败均难改变决策，故不执行、不调bin/lambda/分类器。不是Noisy无信息，也不是将完整P8-V门强加到所有小试验。下一有价值的问题需直接联系可听内容参照与前后端失真，而非追加输入校准；[决策](../../research_memos/p8_post_readability_decision_20261019.md)。

**缓存连续结构读出已完成并关闭**：`experiments/p8_cached_structure_readability_20261019/`唯一主0.475s、独立5.375s；72缓存PCM主/复核各读一次，16固定双输出线性拟合＋16独立增广最小二乘复算，32case/11,396帧全保留，原指标及代码源锁保持。full Noisy相对simple Noisy两目标MSE下降17.28%/8.96%，4speaker同向，−18dB差较小；full frontend反而在两个目标、四位speaker均差于simple frontend。**simple frontend全量MSE最低**（HF0.0275164/flux0.00341552），故不能拿full Noisy胜full frontend的均值就宣称前端丢失可恢复内容。full Noisy与full frontend的speaker方向为HF3/4、flux2/4，未展示稳定双代理优势。仅有限结构可读性，不是音素/H1-H0/增强收益；不调lambda/特征救结果，不自动升级hidden probe/verifier/CAEG。原metrics SHA`235a7b2ac8f15131d25d1f061319f2d8e1c815d00d53ad08e311192e7c67e913`，完整报告保留越界预测及操作性预检错误。

**人耳unknown解释补充**：完成94题后，用户说明部分“不确定”因黄色80ms太快、来不及反应，并自评肯定的结构判断约95%正确。保存原话于单听者目录`responses/user_qualitative_clarification_20261019.json`；不将unknown当作无语音／无信息，不在未指定题号时追填原因，不把自信估计写成独立实测准确率。原答卷不改、不要求重听。

**单听者94题已收到，更新上述/后续未评历史**：Noisy41结构／2非语音／4unknown，Clean45结构／0非语音／2unknown；冻结位置配对中，Clean结构45处对应Noisy40结构、1非语音、4unknown。同人判断非独立gold，不能称准确率；缺Clean明确非语音类别，固定池不支持假阳性/真假双向检验，不重选control/重复要求试听。保留用户先前黄色区定位疑问与UI版本无法逐题追溯风险。详见`experiments/p8_i1b_single_listener_materials_20261019/responses/completed_interpretation_20261019.md`；材料原metrics不倒改，缓存probe独立继续。

**单听者实际反馈增量（不是材料通过数）**：用户聊天提交export `2026-10-19T11:36:10.712Z`，94题ID/material匹配；Noisy47题锁定，41结构／2非语音／4unknown，Clean47未评。无准确率、音素身份或独立gold结论；此前黄色区定位/理解问题无法由导出追溯到逐题版本，原样保留并披露，不静默筛答卷。独立归档见`experiments/p8_i1b_single_listener_materials_20261019/responses/stage1_received_20261019.md`；原生成metrics不回填。Clean后续可选，不影响缓存probe进展。

**用户授权的单听者材料已生成，非新听感结果**：47固定区间×原Noisy/Clean共94裁窗，0.172s制作，94源PCM独立重核及97本地GET通过；约1.58MB，0模型/训练/外网。原四人设计不实施。制作时无判断，浏览器播放未实测，localStorage/JSON下载不等于服务端保存；[页面](http://127.0.0.1:8766/p8_i1b_single_listener_materials_20261019/index.html)已设可选TODO，不催答、不阻塞科研。报告在`experiments/p8_i1b_single_listener_materials_20261019/experiment_report.md`。

**Router成本口径澄清（仅代码/旧日志）**：`future_grad`额外Score重评分在训练，不进入普通路由推理。当前N30/none-corrector静态路径中，正式`off(scale0)`、Router与t-cut均30 Score＋30 Router；P8的`rrsd=false`才是30 Score＋0 Router。t-cut不提前跳过小网络。旧三臂总job时间958/956/957秒包含启动/前端/I/O，不能证明Router零开销或加速；参数少也不是延时证据。[论文成本说明与来源](../../THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md#24-复杂度与计时口径区分训练lookahead和推理路由)。未重新推理或测试集调参。

**用户后续确认训练历史（更新基线解释）**：用户明确说明：“80k checkpoint 是在300k权重的基础上冻结前后端模型，只训练router的结果”。因此采用**300k基础前后端＋80k Router-only增量训练**的描述；不再把80k与300k误当成两套主干各自只训80k/300k的比较。同一80k checkpoint的Router-on/off是同一冻结基础系统的模块开关对照，额外训练及推理成本仍须单列。该说明补足训练历史来源，下面静态审计保持原样；用户本句未指定初始化时RAW/EMA物化的细节，不能替其补写。与另一个300k文件作运行等价比较仍需核实际EMA、配置与采样设置，不由文件名自动放行。

**P8基线身份补核（静态，不是增强实验）**：`experiments/p8_checkpoint_base_identity_20261019/`唯一标准库比较3.190s、0模型/音频/训练；80k与300k同名基础RAW共538项，42同/496不同/0未知。300k EMA536项与所覆盖80k RAW536项的dtype/shape/有效SHA多重集合一致（Root后验算术），另两项固定Fourier W的RAW一致；与EMA物化初始化相符，仍不证明名字—EMA位置映射或历史copy。不能仅据80k文件名说主干训练不足，也不把集合匹配当运行输出/公平预算证明。不重跑300k或改P8原指标；[论文限定](../../P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md#16-80k原生off不应被简化成只训练80k的storm)。

**战略复审（非实验）**：新的独立Research Lead实际只读核查goal/计划/报告，认为已出现过度门控和文档工程；Root采纳，停止给该池追加合同/校验器/来源扫描。下一科学运行待明确4位未接触材料听者与播放授权，复用47位置/188判断；0已确认参与者/导出/播放。noise身份及联合split限制不是人耳评价或整个goal的禁令。可以整理已有论文证据，不将结构端点、写作或理论审查改写为完整goal成功。[决策](../../research_memos/p8_strategy_reset_20261019.md)。

**官方训练日志来源增量**：`experiments/p8_vb_train_noise_log_audit_20261019/` 唯一3GET/7.986s/136499B，ZIP MD5、train28 CRC和独立11572行/12join/图核验通过，0音频/模型/训练。I0限定本机扫描无日志不等于官方无日志。Worker因未读README保留SNRnull；Root另只读115B字段说明确认SNR(dB)，独立补证，不回填原metrics。12句标称0/5/10/15dB各4/2/3/3，不等于本地/局部实测极低SNR；noise类别restaurant7/station2/ssn1/traffic1/cafeteria1，录音/切片身份仍unknown。speaker或同literal类别图为10+2，不能三方同时隔离此二关系；不换ID/拆分量凑split，不升级纯noise/D4/科学gate、不动人耳分配。[解释与补证](../../research_memos/p8_vb_train_noise_provenance_interpretation_20261019.md)。

**条件利用方法审查（无新实验）**：实际读DR-DiffuSE/DOSE官方正文及本机_step/_loss/forward，纠正归档DOSE“条件dropout”易误读：其对象为中间状态而非删除Noisy，并使用clean预测。Root解析表明仅把原x_t换成独立噪声而保留原DSM标签时，已知drop分支最优可为零；隐藏分支不能据此断言全网络零解。该限制不是所有dropout非法，也不证明当前模型条件利用充分。2026 PRL条目仅发布者摘要可核、正文未读，不作精确覆盖结论。详见 [定向先例审查](../../research_memos/p8_condition_utilization_priorart_20261019.md)；不跑模型/训练，不以待听者为由扩建准备工程。

**D4仅解析合同，不是新增检测结果**：[模型与估计条件](../../research_memos/p8_d4_reference_contract_20261019.md)核清已知候选、共同Gaussian均值/协方差下的LLR/错误率/AUC关系；二维同trace反例说明标量能量不能定实际检测难度。独立数学复审通过并补共同均值/d=0退化条件。没有对12句填D4数值、拟合噪声、训练/推理；noise来源unknown未被公式解决。协方差包络只有真实成立才给模型内界，不是现有RMS自动保证。

**新用户计划，尚非实验证据**：P8-I局部可辨识性方案已登记，详见 [独立计划](p8_noisy_identifiability.md)。当前仅元数据/代码与合成软件准备；真实反事实构造、D0–D4训练/评测及CAEG协作均未实施。旧Noisy消融不转写为正确反证据，编辑削弱标签不升级音素有无真值。后续状态须由独立实验目录证据更新。

**P8-I软件准备后续**：I0 driver/独立validator已交，29项合成检查不是实际语料通过。Root实核8锁/来源hash/AST并委派Worker一次受限元数据登记（60秒/2MiB，24文件header/全字节SHA，0PCM解码/模型/网络/训练）；尚不填任何成功或可辨识性数值。独立只读复审补充：X1保留原X、H0仍可能含残差语音泄漏；unknown来源不得伪ID隔离；D0–D3须匹配有效上下文。I1真实编辑/混音与训练均不随I0自动授权。目录 `experiments/p8_noisy_local_identifiability_prep_20261019/`。

**I0真实登记完成（仅元数据结果，覆盖上述待运行）**：12对/6speaker、0.235117秒，原registry与独立validator通过；主24header与2,770,724B全WAV哈希读取，独立另24文件核验，PCM解码/模型/训练/网络0。限定metadata扫描0候选，noise identity unknown；noise-disjoint/split/scientific-ready均false。I0完成不能当作I1的残差纯净/对齐或D0–D4可辨识性通过。原始记录在上述目录；下一步仅按真实来源限制制定I1波形审计协议，不直接生成编辑/混音或训练。

**I1a仅软件准备，非结果**：已冻结audit-only的12对配对/残差设计，目录 `experiments/p8_i1_pair_residual_audit_20261019/`。Engineer仅代码/合成测试，尚未PCM解码或真实执行；主与独立验证未来各120秒/合计8MiB产物，不编辑、混音、训练或调用StoRM。固定共同支持全321lag、raw残差与OLS/全帧描述，不以代数重构或OLS正交宣布noise纯净。原I0/旧实验保持冻结。

**I1a真实审计已完成（覆盖上述准备状态）**：主3.054513秒、独立7.332284秒；主24+独立24=48次PCM文件解码，全12对/3,852lag/4,330帧通过。固定±10ms内bestlag全0，共同支持zeroCorr0.661784–0.981795；全句OLSgain0.912976–1.011548（中位1.000753），最偏p239_028保留，其raw残差相关−0.087923不自动归因泄漏。24原信号PCM端点全0、raw重构误差0，均不足以证明纯noise。Root不shift、不用fit gain改源，继续zero-lag/gain1；noise身份/纯度unknown、三个gatefalse。0新波形/编辑/混音/模型/ASR/训练/网络；原metrics SHA `93af91ade77fd6072b40a7e8ef915edf360dc4748f4223eca15287aa1d6a08a6`。下一步仅设计原观测保留的操作编辑及伪影对照，不建检测器/跨句残差混合或宣称CAEG验证。

**I1b仅协议（无新实验）**：[事件操作/伪影控制设计](../../research_memos/p8_i1b_event_edit_protocol_20261019.md)已写入。独立只读复审无设计层致命问题，要求操作标签/位置对照/最终量化三者不升级为自然内容证据；已采纳。还没有真实选区、编辑音频、操作AUC或D0–D4/CAEG结果；不得用未来技术完整性通过替代可辨识性成功。

**I1b后续仅代码准备**：具体选区/tie、80ms离散mask、位置对照和量化/越界规则已冻结于 `experiments/p8_i1b_event_pair_integrity_20261019/{analysis_plan.md,config.yaml}`，Engineer做最小实现与合成测试，真实选区/PCM/内存编辑均未运行。保留24目标槽/96操作槽的失败及缺失状态，不挑safe子集训练；原/sham字节保持、edited只内存hash，仍0新音频/模型/训练。上段方案状态不是新结果。

**I1b真实完成/技术支线收口（覆盖上述未运行）**：主1.063287s、独立3.474548s；24目标/96操作全保留、全部safe；主24+核验24=48PCM文件解码，只内存计算与hash，0音频文件导出/模型/ASR/分类器/训练。原metrics SHA `24c0ee1297e261dbecc4cfa9b2f3441292e45e71709cb2fddb0b60462362464c`。原/sham字节和mask外样本不变，float重构max2.78e−17，safe单臂量化≤0.5LSB，量化双臂+raw残差重构max1LSB。**未真正配平编辑能量**：48组相关target/control×taper实际delta能量相对差中位12.86%、max82.78%，后验描述不是新终点；共享control依赖保留。不给此池“真辅音/无事件/noise-disjoint”金标，不调规则重跑或直接训练操作检测器。下一步优先确认自然结构评价与听者资源（尚未确认，未招募/导出），或另审有限结构预测方案；不能把后者当CAEG完整验证。全部科学gate仍false。

**自然结构评价仅方案，不是反馈**：[协议](../../research_memos/p8_i1b_human_structure_protocol_20261019.md)与operator-only allocation JSON已写；47唯一位置×2原始arm×2判断=188计划trial，4个未分配听者slot。按句子跨人分臂，避免同人先Clean再对应Noisy；480ms上下文保证原选区边界，0音频读取/导出/播放、0参与者/答卷。原600ms方案在元数据边界断言失败后实施前改480ms，非声学重跑。资源仍未知；若只有2人，最多每arm单判断的探索版，不伪称双人Clean共识。结构参考即使一致也不是音素真值或Noisy信息上界，不新增模型/训练授权。

旧计划里的“准备中”“待执行”若已被后面的**完成/终止**覆盖，以对应最新可复核实验报告为准；旧段落保留历史但不能重新授权动作。音频听感原话和自动代理数值同时保存、互不覆盖。所有新结果先更新本表及实验报告，再将人类可理解的状态写到 [短总纲](../research_plan.md)。

## 7. P8-V 方法审计状态（非声学实验）

**旧状态 → 新工作 → 当前判断**：候选级反事实构想 → V0a 数学/定向先例审计 → 尚不具备启动验证器/引导的条件。固定 Gaussian 假设下的似然差就是白化残差/匹配滤波基线；两候选都错、真身份未被提出及自适应候选选择校准仍待解决。`unknown` 不等于静音，首个验证阶段应只报排序/拒答，不改波形。ArtiFree/Grounded Decoding 已覆盖部分选择/锚定空间；新增核对 StuPASE，不证明本项目新颖性。下一步只读材料/候选来源审计；旧 VoiceBank/B/VCV/5243 全部保持关闭。详见项目根 `research_memos/p8_v_v0_method_audit_20261019.md`，**无新模型或音频结果**。

**材料门后续**：已缓存VCV源清单合计44份/13位speaker，原始字节哈希全部相符；不能只按主试验8位speaker估计还剩8位未用。相对两缓存清单仅 `f13/f24/m17` 未列入，不保证全项目/预训练未见。保存的91份人工MLF在同speaker/context/stress下没有完整k/g或t/d双侧边界对；VoiceBank11,572条train/valid参考是句子标签，不含音素事件标注。名义近音对存在，但完整P8-V材料门尚未齐；缺失/集外候选、proposal及拒答校准需先确定来源。只读来源审计见 `research_memos/p8_v_material_gate_20261019.{md,json}`，未新增模型/音频实验。

**参考语义修正**：外部官方文档可区分实际发音与标准期望；源语料的删除/添加/替换不是增强造成的错误。未来应保留说话人实际音素，不奖励先验“纠正”口音/省音。L2-ARCTIC/Buckeye 完整访问需相应申请，本轮未注册、未获取语料。标准库标签契约已交付，14项合成单元测试及Root另5个literal检查通过，始终不批准真实导入或hard gold；不算声学验证。方案与出处：`research_memos/p8_realized_phone_source_review_20261019.md`。

**后续单例反证了“合成测试通过即可接语料”的推断**：官方主页公开的一个TextGrid已取回，正文3,579B，words7/phones16/IPA7共30interval独立结构检查通过；phones有sp与空标签。旧v0会把未验证字母符号当speech、拒绝空值，所以真实导入需tier/符号profile与未知隔离，不能把14项测试当完整兼容。原软件冻结未改，无音频/模型调用。此例3个替换、0添加/删除标记不代表整个语料分布；不是独立确认集。出处：`experiments/p8_l2_public_annotation_schema_audit_20261019/`及上述来源memo§5。

**验证门复审（不是新增性能结果）**：独立Research Lead批评认为，强音素/缺失标注门被过度前置到所有早期工作。Root采纳：停止通用parser支线，保留完整主张的强门；先准备前瞻性固定候选headroom的有界反证。已有合法VB train清单在序号>=26、3–5秒规则下覆盖26位合格speaker，仅是元数据计数。当前批准8句/4speaker的metadata-only注册，尚未批准混音/模型，不声称独立预训练holdout或真实音素收益。决定与退出条件：`research_memos/p8_validation_gate_reassessment_20261019.md`。

**注册后续结果**：`experiments/p8_vb_prospective_headroom_20261019/` 已冻结8句/4speaker/8个不同参考文本，独立重选、发布者原TXT ZIP与8个clean原文件字节哈希通过。Root接受限定当前项目扫描的未见明确模型/试听使用结论；不是全球/匿名/预训练未见证明。最终使用史在 `final_history_review.json`，原registry/metrics pending字段保留历史。仍0解码/混音/模型；下一步仅科学/执行协议审查，不把metadata通过算增强结果。

**科学协议已冻结，尚非结果**：同目录 `science_plan.md` 定义32条件、32frontend/32OFF NFE30、104次transcribe上限、primary WER-oracle/tie规则与4speaker探索性bootstrap；同时评估SI-SDR/STOI/PESQ并披露标准metric内部预处理。预算模型180s/父目录64MiB、0训练/网络/重试。Engineer仅准备代码与合成测试；CPU真实输入和模型须各自复核后Worker运行，无新性能数值，不把普通句级排序改称完整P8-V。

**CPU阶段已完成**：`p8_vb_prospective_headroom_20261019/input_stage/` 唯一运行4.348s，8个本地clean+2个原始noise解码，32noisy+8scaledclean（80文件）独立重建通过；7准备锁/32源锁/94运行工件核验，输入零饱和。原registry SHA `cfc2e1b4ac313037f7e324b6e548f0f56cb1ee32a80282475a0823c7a876d998`。0ASR/增强/训练/网络/管线重采样；只是合成完整性，不证明内容余量或本地与官方48k转换相等。Root接受用于最终模型代码冻结，模型尚未启动；质量代码额外修正了nonfinite值JSON表示与STOI警告sentinel，只有合成测试，未运行实际metric或模型。

**后续模型已完成（覆盖上述历史“尚未启动”）**：同目录model_stage唯一97.645s，32frontend/32OFF NFE30、104transcribe/104internal decode，0训练/网络；413锁/368工件、64序列化/104评分/96quality臂独立核验。全量N288，noisy/frontend/OFF/oracle错误185/378/229/159，原primary余量70/288=24.31pp；frontend赢23、OFF赢4、tie5。单例p227_030_N1_m18前端转写216I，不能算音频真实插入；4cluster区间宽、clean控制2S/72也全保留。后验描述分解70=62跨speaker固定臂异质性+8speaker内余量，不改主终点。0dB front/off/oracle WER14.58/31.25/12.50%；-18dB247.92/127.78/97.92%，接近失效的代理不证明声学无信息。OFF−front全32对SI-SDR−1.326dB/STOI+0.009965/PESQ+0.062594，64输出无clip/饱和。**关闭本轮，不训练/追加排序或重跑ASR**，下一步先设计独立内容端点评估；`research_memos/p8_prospective_headroom_interpretation_20261019.{md,json}`含全部限制。

**盲转写只是设计，不是新结果**：为避免同一听者记住同句后评其他arm/noise，0dB固定48版本分成6list、每list8不同句、每list2人，共需12人/96判断。Root核48原文件字节SHA、每list底层句子不重复及跨list全部版本覆盖；未播放/复制音频、未招募或实施。不能将跨不同听者的较好转写当作声学候选的人类oracle，不用P.808质量MOS替代内容。0dB范围是后验跟进，不能回答-18dB异常转写或完整P8-V。原始来源与边界见 `research_memos/p8_content_blind_protocol_20261019.md`；分配JSON为操作者设计，禁止直接暴露给听者。

**Score/观测证据不是同义（代码/数学审计，无新测量）**：真实model/SDE/transform源码与独立Lead复审表明，score范数不排序密度、固定t路径积分最多给条件扰动密度比、变t路径另有时间项；y_cond=None实际回填anchor，不能直接变无条件prior。非线性压缩谱差不保留物理加性noise。DPS2209.14687v4与显式speech/noise后验SE2601.09931v1定向原文核查补充先例；普通measurement guidance并非新颖性。80k旧metadata未给transform参数，不能推默认指数；未开模型/重排/梯度实验，不断言所有score经验特征必然无效。详见 `research_memos/p8_score_vs_evidence_audit_20261019.md`。

**上述参数unknown的后续修正（不是runtime复测）**：同一80k checkpoint字节SHA核实，静态ZIP `archive/data.pkl`的唯一key/primitive opcode直接显示transform=exponent、spec_abs_exponent0.5、factor0.15及mode2/OUVE/score_matching；未执行pickle或加载模型，c_out引用未解析，不猜测。噪声likelihood契约明确物理/线性域、来源可见的noise估计、共享nuisance与校准；解析积分共享标量variance仍与最小平方残差同排序，75组算术检查不是语音准确率。否决其作为新verifier的包装，不启动新noise搜索/实验。出处：`research_memos/p8_noise_evidence_contract_20261019.{md,json}`。

**新离线方向—幅度诊断已完成/关闭**：`experiments/p8_vb_residual_direction_audit_20261019/` 主1.985s+独立11.701s、主72+核验72=144次既有PCM解码、32case/21层与497原件pre/post一致，无新模型/ASR/质量/波形输出。全32case clean-oracle alpha在(0,1)，31例全量OFF的MSE更高；等case归一MSE F0.26391/OFF0.31783/共同alpha0.261108 oracle0.25622（比F低2.92%），逐caseoracle0.25391（低3.79%）。非gain能量占比case均值98.40%，gain/rest归一MSE贡献−.007727/+ .061646；不能解读成幻觉能量或解释全部SI-SDR。只是受限MSE几何，**不部署alpha/新混合、不重开anchor或追加seed**；竞争解释含生成方差/系统偏差，非“所有修改都错”。`research_memos/p8_residual_direction_interpretation_20261019.md`保留理想后验风险推导与perception定义边界，不宣称处在理论前沿。

**旧缓存seed分解完成/关闭**：`experiments/p8_cached_multiseed_variance_audit_20261019/` 复用16文件×4seed **Router/manualprepare_specs**缓存（非VB32 nativeOFF），80唯一PCM、主80+独立80=160decode，主11.892s/核验12.374s、9层pairwise恒等式通过。等文件归一Eavg0.249056=Emean0.217425+V4 0.0316305；pooled V4比12.7002%，逐文件比均值13.5674%/中位13.0073%/范围5.954–19.300%。87.2998%只表示有限集合中心参考误差，**不等于已确认真实bias**；IID估计另列。旧pertrajectory无历史WAV hash，只证当前字节pre/post不变，未复现生成。不导出mean audio/新指标/seed/模型；不把代数MSE降低变成新方法或4倍成本的“免费提升”。`research_memos/p8_cached_variance_interpretation_20261019.md`限定了跨旧Router到当前OFF的推断与下一估计对象审查。

**估计对象后续审查已完成（无新声学结果）**：从“中心参考误差的原因未知”出发，代码/数学复核确认压缩谱 clean 条件均值逆变换一般不等于波形条件均值；但native PC默认返回随机路径末次反向离散转移均值，不是该clean条件均值，eps还由model.self.t_eps传入。精确样本的正确逆变换本身不构成分布错误，因此不把反例当作真实吞音因果解释。PyTorch官方/安装文档支持复randn实虚各方差1/2，不能混用real score常数；未做运行时noise或网络协方差校准。2409.10753v1与STSL2312.00852v1原文已有目标/二阶矩相关先例；后者非此具体修正的等价证明。**到此收口，不建Jacobian工具链、不改sampler/训练，不重用旧32格调参**；详见 [估计对象审查](../../research_memos/p8_estimand_readout_audit_20261019.md)。

## 8. CPEL v2终态：执行完整，pair增量不足，关闭本配置

**先前状态 → 新证据 → 当前决定**：M/P已批准并启动 → 唯一两臂各500更新、64正常评价及四控制packet完成，Root与独立Reviewer从终态JSON复算一致 → 不通过事先固定的10%增量信号门，关闭该配置，不扩大/调权重/加步数救结果。[完整执行报告](../../experiments/p8_cpel_mp_pilot_20261019/experiment_report.md)／[科学解释与后续决定](../../research_memos/p8_cpel_mp_pilot_interpretation_20261020.md)。

P相对M的raw Tukey pair仅降1.67035%，PCM16精度匹配降1.65827%（P/M=.9832965/.9834173；门≤.9），均64file×双parent完整覆盖；M相对initial已降11.35081%/11.28388%，不能计作pair贡献。pair约90%为orth，P投影gain略降，非零/反向覆盖未改善，不能称恢复更多弱辅音或更敏感。正常64例P−M：PESQ+.00013913、STOI+.000004568、SI-SDR−.00028658dB；未越工程损伤界，但没有实质增强收益证据。

noise实例37可评/27同位置匹配能量不足，其中strict30；工程组gain差−.00005501，输出变化能量略降。未评不算零/通过，codec相容仍不证明noise纯度或FPR。M/P同500步样本/随机变量，前端及固定Fourier W摘要不变，动态1%preclip梯度预算实测符合容差；不等于Adam更新或声学可信度预算。两臂总746.71秒，M额外包含initial基线，不能据总时长声称P更快。普通DSM/增强A/B、真实内容/身份与独立确认均未实施，不因小试无明显增量而否定所有反事实训练，也不将长期goal收缩为这个已完成小试。


## 9. 新词身份端点：来源部分证据，技术gate失败不掩盖

CPEL关闭后，独立Lead支持一个不同的早期诊断问题：原生后端相对前端的词身份修复/损伤转移，而不是继续优化结构proxy；不用完整自然音素强标注阻塞所有有限问题，也不把词标签成功升级为辅音恢复。来源与退出见 [限定memo](../../research_memos/p8_keyword_endpoint_feasibility_20261020.md)。

[一次官方ZIP来源检查](../../experiments/p8_keyword_source_feasibility_20261020/experiment_report.md)返回两个206 Range，共2143724响应body字节/33.60秒，CD16020项中真实词WAV路径8000（8词各1000）、1750前缀组、88完整8词组，无重复路径；未取成员内容/音频或模型。第三源码请求超时、产物5.57MB超过4MiB，**运行失败**；原件保留，不删/压缩/提限重跑救gate。Root从已浏览官方which_set规则另作离线计数，完整组train/validation/testing=71/9/8，前缀交叉0；不冒称Worker成功取得分组源码，不把prefix当verifiedspeaker。

新的 [64Clean读出固定方案](../../experiments/p8_keyword_clean_readout_20261020/analysis_plan.md)仅在准备：按标签/组元数据冻结8个validation组各8词64条，testing完整8组reserved，不按声音或ASR选样。拟一个新任务取得官方182MB archive，仅抽64原音频用本机tiny.en无prompt校验适配性；门为≥58/64且每词≥6/8。不通过不换prompt/模型、不建分类器；通过也不证明Noisy/增强域可靠。软件与单独许可未完成前0真实音频/ASR；原source失败和CPEL关闭不改判。

**64Clean取得任务后续（技术失败，不是识别失败）**：父唯一GET在120.018s仅保存11.53MB partial，0成员/0ASR，适配性unknown，30保护文件不变，不续传或延时。Root仅准备一次同64/同ASR/同门的固定成员Range运输修正，预计5.54MB body而非182MB全包；并非改评价救场，无完整archive SHA。子code审查/单独许可未完成前不执行，若再失败不追加抓取平台；[独立子方案](../../experiments/p8_keyword_clean_readout_20261020/transport_attempt02/analysis_plan.md)。

**固定64Clean读出最终收口（覆盖前述准备/unknown）**：transport02唯一104.054s完成64Range/5,535,648B、64CRC/header、64transcribe/64内部decode。实际49/64=76.5625%，各词4/7/7/5/6/6/7/7；原门≥58且每词≥6未过。Root与独立Reviewer逐条JSON复算一致，34保护文件不变，父完整包取得失败及metadata失败保留。关闭当前tiny.en词身份端点，不跑Noisy/F/OFF、不改prompt/模型/词表/去testing或训KWS救场。一个prefix组0/8占8/15错，不能据此猜录音/口音原因或删组；ASR不匹配不是标签错、自然发音错误或物理无信息的证明。[完整报告](../../experiments/p8_keyword_clean_readout_20261020/transport_attempt02/experiment_report.md)／[解释与约束](../../research_memos/p8_keyword_clean_readout_closeout_20261020.md)。未产生增强/真实内容恢复结论，长期goal未完成。

## 10. CPEL后独立战略复审（无新实验）

独立Lead与Root采纳：目前不提出立即开训的新机制，不用再换代理填补未建立的内容缺口。只保留一个未来有外部资源时的问题：新自然整句中是否重复存在Noisy可被独立无参考转写保留而F/native OFF均丢失的内容。它只建立任务级可修复性，不证明局部phone信息或通道/OUVE因果，不直接指定loss。缺的是独立输出内容读数，而非将完整phone强标注前置所有早期研究；不重复询问听者、追加旧任务或建下一套平台，0新执行许可。[采纳决定](../../research_memos/p8_post_cpel_strategy_20261020.md)。

定向核读L3-SE v1 §3–4新增先例约束：双教师声学/语义蒸馏后条件化自回归生成已有实现，−5至5dB的所谓low-SNR评价不能直接支持本地−18dB。未复现外部模型或搬用性能，相关工作将明确这一限制；不是穷尽查新、也不由已知先例宣布所有具体机制无效。

## 11. 恢复后的问题保真纠正：M不等于普通DSM增强

用户质疑全局阻塞并恢复目标，最新目标明确只有必须等待试听且无其它任务时才可能阻塞。学校模板只限制最终排版、人工证据只限制对应强内容主张，先前全局停工推断撤回，不重跑闭合实验或倒改失败门。

代码与独立代数复核发现：M=nativeDSM+权重1共同state逐点clean回归，后者精确包含反对称差分 
`aux=||rbar||²/(2D)+||r1−r0||²/(8D)`，`r1−r0=σΔq−(a/σ)δ`。即M已隐含gain1及orth强监督，P主要再改权相同误差；旧1.67%/1.66%不足10%的结论只针对该强基线，不能替代用户原普通DSM增强→方向pair的未执行问题。[精确推导与边界](../../research_memos/p8_cpel_question_fidelity_20261020.md)。这不是新代码bug或新训练结果，亦不保证取消aux后会有收益。

允许审查唯一缺失对照：aux0的N(native edited DSM)与D(同native加原pair)，不改样本/编辑/500步/EMA/1%规则/门限，4forward/2gradient预算匹配。共同state只为pair，不以0*aux留逐点任务；旧M/P不重跑。预算基准随native-only变化，因此跨旧两臂的交互不能纯归因冗余。当前仅静态实施与价值审查，未授权新训练；phonegold/额外试听不作为这项早期机制比较准入条件。

**N/D后续科学决定（新问题，未有结果）**：独立数学审查确认M已含强delta监督，工程审查证明可用新隔离目录删除common_point，不改core或旧结果。Root签唯一N/D：普通反事实nativeDSM vs 同DSM+原pair；同256/64、500步/4Score/2grad/EMA/原margin和1%预算，N完整终态后D，显式复用绑定旧initial64。每臂≤900s/总1800s、320MiB且留1GiB，0ASR/冻测/网络/音频导出；真正图支持与科学有效性仍未知。旧M/P失败门不改，当前CV复用只探索，预算native-only使跨旧交互不能纯归因冗余。不需用户试听，也不直接启动大规模训练。[执行方案](../../experiments/p8_cpel_native_pair_ablation_20261020/analysis_plan.md)。

## 12. N/D缺失对照终态：小幅增量，仍未达固定推进门

N/D各500更新/64正常例/四packet及F/W不变量完整，0共同点clean回归；Root与独立Reviewer从保存JSON复算一致。raw/precision weighted pair下降3.48424%/3.47731%，均未达事先10%门；Hann下降4.46252%。D−N正常PESQ+.00148357、STOI+.000127716、SI-SDR+.03957943dB，小幅正向但PESQ仍低于source约.005267。投影gain略降、形状orth仍占约九成，不能叫恢复幅度更大、真实phone恢复或防幻觉。

noise37可评/strict30、27能量不足未评；D−N signed gain+.00012893（strict+.00014617）未越+.02界，能量略降，signed可抵消不报FPR。500draw全匹配、每步aux computed False；D原始梯度预算max约.01、非Adam界。两臂总581.72秒、复用历史source initial64无新基线调用，同CV再次使用仅探索。旧M/P与新N/D的问题不同，其增量差不能纯归因aux冗余，原A→B数据增强收益仍未分离。

**关闭本固定no-aux配置，不加权/加步/换窗/seed搜过门；承认小作用，不普遍否定CPEL，不全局阻塞其它科研。** [统一CPEL索引](p8_paired_evidence_training.md)／[N/D解释](../../research_memos/p8_cpel_native_pair_interpretation_20261020.md)／[完整实验](../../experiments/p8_cpel_native_pair_ablation_20261020/experiment_report.md)。P8-V页已交叉链接，训练结果不搬作推理期验证成功。

## 13. 下一客观问题：幅度标度已匹配，anchor接口仍未量化

独立只读代码与旧VB32 hook记录实核32/32公共backend norm等于Noisy peak；public frontend乘r与后端除r抵消，Noisy与anchor共用同r，未见重复归一化或anchorpeak独立缩放，不批准“修复归一化”实验。

另一个具体实现边界：当前训练/CPEL为raw前端压缩谱A，public部署为T(W(A))，W是真实frontend逆压缩/ISTFT，T是真实modelSTFT/压缩。任意神经预测谱不保证STFT一致，eps也不精确互逆；旧hooks只有波形hash，没有两个anchor谱差。当前源码不证明300k历史训练接口，也不能从此解释所有吞音或CPEL负结果。

只准备一个固定8dev、原source前端8次、raw/public各8×N30共480Score/180s/2MiB/0训练的接口检查，直接hook公共sampler实际anchor/condition/norm并核expectedRoundTrip；metadata按4bin各2预选，0真实运行。谱差/输出差为0也有效，不改窗/seed找非零，差非零不当创新或质量收益。[方案](../../experiments/p8_native_anchor_roundtrip_audit_20261020/analysis_plan.md)。

**接口audit后续终态（失败partial，不重跑）**：唯一22.149935s，在7/8双branch后第7expectedRT/actualpublic relL2=1.07710e−5>固定1e−5停止；第8未读取。实际1model/14PCM/7F/14backend/420Score、42保护全同、raw历史MSEsanity7/7通过、norm同；大差rawA/actualPublic relL2=.101–.245、output=.0689–1.0902，不能与小interface残差混淆。Noisy条件也有浮点差，非完全只改anchor的因果试验；publicMSE5低2高仅partial，不作质量收益。CPEL正常结果应限为raw预测谱anchor的PC评价，未证明public部署等价；旧两轮门不改，源300k历史代码未知。[报告](../../experiments/p8_native_anchor_roundtrip_audit_20261020/experiment_report.md)／[解释](../../research_memos/p8_native_anchor_roundtrip_interpretation_20261020.md)。不放tol/补第8/改模型或归一化，不由接口差异包装创新。

## 14. 由已测接口差异引出的缺失部署评价（仅准备）

独立Lead复审支持一次固定public部署迁移补充，不是把raw失败门换成新成功门：同全部64dev、source/N/D既有三臂，每臂真正调用公共enhance，固定前端物理输入与逐例seed，不训练或改接口。source第三臂保留相对基座代价；主要publicD−N并列N−source/D−source，不按旧7partial挑子集。旧M/P、N/D10%门和audit第7failed全部保持，不补旧第8/放tol；本run不声称public等于手工expected。

预算1source加载+2已保存terminalstate、128PCM/64F/192backend/5760Score、600s/4MiB/留1GiB，无controlled edit/网络/ASR/冻测/权重或音频输出。sourceEMA仅一次，N/D strict注入后canonicalstate摘要必须正确、F/W不变，不调用会重置sourceEMA的model.eval。原始dev复用属于事后探索非独立确认，不据结果自动开大训练。[固定部署问题与协议](../../experiments/p8_cpel_public_transfer_20261020/analysis_plan.md)。当前仅软件准备，尚无真实执行许可/结果。

**Public部署评价终态（完整192行，非新训练）**：source/N/D各64完成209.6538s、5760Score/128PCM/64F/192backend/3weight，155保护文件前后同、3canonical注入/9FW检查true。Root与独立标量复算publicD−N SI-SDR+.00458936dB、STOI+5.1245e−8、PESQ+.000342682、MSE−4.5080e−7；D−source另为SI+.0392907/STOI+.000218412/PESQ−.00331249，未建立实质综合部署或内容收益。全64每项有效，无metric error，输入anchor/condition/norm/seed跨臂同。旧raw10%失败门及7partial auditfailed不改；不择接口/权重、不自动扩训。原始dev复用属事后探索非独立确认，不给phone或人耳主张。[报告](../../experiments/p8_cpel_public_transfer_20261020/experiment_report.md)／[独立解释](../../research_memos/p8_cpel_public_transfer_interpretation_20261020.md)。


## 15. CPEL后复盘：同例质量—重建冲突并非少数尾部案例

从既有64speaker/128句CV的质量CSV与Distill-MOS CSV按filename严格联结，Root与独立Lead标准库算术一致：Router scale1相对scale0的SI-SDR在128/128句均改善；其中105句同例DNSMOS-OVRL下降，83句还同时Distill-MOS下降。SI-SDR增益最小+.12514dB、中位+.47586dB，OVRL差中位−.01824；60/64 speaker的OVRL差均值为负。两CSV原件SHA前后同，0新音频/模型/指标/重抽样，源路径与SHA见[复盘及计数](../../research_memos/p8_post_cpel_bottleneck_reassessment_20261020.md)。这不是native bypass或CPEL N/D对照，亦非独立最终确认。

该结果排除“少量极端负例或跨句均值混合造成全部冲突”的简单叙事，不据此重开selector、tail-risk或SNR门控；不新增质量×CAM相关表。预测MOS不是人类偏好，83句不能当两份独立人类证据。旧P3b全tie比较Router/t-cut/selective而非Router/off，不能用它否定此对照下潜在人耳差异；不要求补听。

尚缺具体失真与可区分干预的归因，当前不开新loss或训练。下一候选须能区分至少两种具体误差来源、不同结果能改变干预选择，并给最低成本及反证退出；不能只是重复代理冲突。模板、试听、phone gold不作整个科研目标前置。

文稿摘要/绪论/结论另已最小同步M/P强aux范围、N/D未达10%及public微小增量；Root核现存55保护文件与7输出SHA全部一致，14公式/58目录与链接检查记录见[文档检查](../../thesis/cpel_scope_sync_checks_20261020.json)。转换检查不替代科学、浏览器或学校模板验收。


**用户范围纠正（第15节之后）**：用户明确质疑为何从CPEL转到Router。第15节为助手自行扩展的历史旁支，不是CPEL证据；该新增Router分析停止，不继续相关文献/诊断/实验。当前聚焦CPEL已保存结果的可解释范围，旧M/P、N/D及public数值、失败门和原件均不改。


## 16. 回到CPEL：有限方向作用不能全部归为幅度缩小

用户纠正后，停止从CPEL自行转向Router的旁支。只用已保存N/D JSON做一次解释性核查：Z=orth/response_energy，在匹配tuple中共同k=sigma²/alpha²下为(1−cos²theta)/k；纯正幅度缩放应使Z不变。raw/precision的file等权mean Z分别下降1.39020%/1.36427%；两臂均正gain的127parent中，122下降、5上升；另1parent仍为负gain，无符号翻转/未定义。原64file完整保留，precision不是独立复验，parent不是独立样本。

这支持有限的P加权响应方向对齐改善，而非“只有总幅度收缩”；不证明完整向量更近、gain增加或内容恢复，也不是原3.48%收益的因果分解。原raw/precision10%失败与public微小增量不变，不追加训练或换门。Root与独立Reviewer从原JSON逐项重算一致，0新模型/音频/权重/重抽样；1e−5描述guard不是新成功门或严格浮点误差界。[公式、完整标量与复核](../../research_memos/p8_cpel_response_shape_interpretation_20261020.md)。


## 17. 用户质疑短程训练：允许有界预算检验准备，不翻案旧门

500步与EMA .999不能证明训练充分；初始线性平均系数仍60.64%，2000步为13.52%，不等于行为比例。原每步训练pair的固定100步块相对改善4.316/10.118/8.985/14.057/15.577%，两组500项8draw字段匹配；这是变化任务上的训练记录，不是固定验证或过原10%门。原终末仅存EMA，无raw/Adam/RNG/order，不能精确续训。Root与独立科学/工程审查支持仅准备一档从source重做2000步两组、其它科学项不变、最终EMA真实public为主的预算敏感性补充。旧500失败保持；不自动再加步。当前尚无真实授权/训练。[审查与依据](../../research_memos/p8_cpel_training_budget_audit_20261020.md)。


**训练预算进一步纠正（用户100k经验）**：用户补充既往创新常需100k步起才显效。Root不启动未执行2000方案，目录保留STATUS说明，无执行授权/STARTED。旧500步筛查是短程结果，不足以排除常规预算收益；也不能凭用户经验认定100k必然充分或有效。现转审全训练数据与匹配长程设计；若数据规模与步数都变，不能将相对旧500差异纯归因步数。实际/data仅约1.3GBfree，根盘约12GB但持久性未确认；已异步询问输出存储位置，不删旧文件，先推进metadata与断点设计。


## 18. 正常训练预算方案进入实现准备，尚无长程结果

用户100k经验后，不启动2000，保留旧500步负筛查但撤回路线级否定。现固定两组100k/完整10797训练条目，源起点/配方/随机流/计算匹配；扩大数据和预算使新旧不具纯步数因果。新256开发面板排旧64条目却共享43speaker；CV1186既有pretraining/选模材料，不是独立确认。最终主模型100k EMA，全五快照都保存、训完统一评价，不挑中期最好。最终明确完整30s三臂public而非仅全records短窗，预计3558calls/106740Score、FLOAT输出约6.83GB。真实30s显存/吞吐和CUDA exactresume必须先技术检查，不按科学分数做开关。CPU toy恢复已验证但真实模型未跑。训练外推约22GPUh，整句推理另约16GPUh粗估，非实测工期。[固定方案与实现准备](../../experiments/p8_cpel_longrun_20261020/analysis_plan.md)。
