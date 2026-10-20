# 前瞻性固定候选余量：冻结科学设计

**Research Lead决定（2026-10-19）**：采用下述32条件设计；当前批准Engineer准备代码/合成测试，真实CPU输入准备与模型执行须分别经Root复核后委派Worker。不是训练/selector/完整P8-V许可。原metadata配置、registry、metrics、final_freeze及历史pending字段全部冻结，不因本阶段重写。

## 1. 问题与范围

固定候选池只有原生frontend与一次Router-off输出。检验池内是否存在**相对发布者原文的句级ASR代理互补余量**：若池内无更低错误候选，就不为这套proposal/评价端点投入复杂selector。代理在低SNR失效不证明真实内容无余量；质量收益与内容改善分开。不能声称真实音素恢复、幻觉率或P8-V已通过。

材料由metadata规则先选、未看新增强结果。8个ID固定：p227_030、p227_031、p228_026、p228_032、p230_027、p230_029、p231_029、p231_032。4speaker/8个不同文本，train-split探索材料，不是最终独立测试。项目使用史只接受final_history_review.json声明范围，预训练/匿名/外部暴露仍unknown。

原registry SHA256 `6c929a2c2c66e00b0bdf7c15172200f7c37eec522c5adeda29190845362bb6eb`；metadata final_freeze SHA `64ce113923ffdee7ff6bcf5eeebc9cb174f2b5f96bdaff1dbf3c006a1fd5644a`。使用registry的local16k clean，**不是**先前官方48k默认转换派生的四条音频；不再取48k或搜索resampler。以前2LSB来源门仍失败，当前只声称已存本地音频与发布者ID关联，spoken alignment/转换等价未证明。原本配套noisy不使用。

## 2. CPU输入配方（无模型）

- 从registry中8个clean WAV作严格header核对/PCM16解码：16k、mono、NONE压缩、frames与冻结metadata完全相同；不重采样、不裁句、不换样本。违背即技术失败，保留结果不替换。
- 复用旧native pilot配置中原始物理噪声N1=noise_fileid_5243.wav、N2=noise_fileid_6339.wav，两个source SHA与完整float64 decoded hash按旧config锁。噪声取[80000,80000+L)，不循环/换offset；不声称新噪声泛化。
- 以全句均方功率确定g=sqrt(Ps/Pn)*10^(-q/20)，q固定[-18,0]dB。每句4条件；共32个noisy。
- 每句共同scale：M=max(该句clean及全部4条件的noise component和mixture绝对峰值)，a=min(1,0.9/M)。同一a作用于该句所有条件与clean。clean-aware配方只在合成/离线评价使用，不进入部署selector或native推理。
- 保存8个scaledclean +32个noisy，各自PCM16 WAV及little-endian float32原数组，共40份信号/80个文件。沿用旧prepare_inputs的float64混合→floor(x*32768)量化/拒绝输入饱和，记录component SNR、float32/PCM相对原混合残差、峰值与scale；不能把量化后残差SNR冒充物理分量SNR。
- basename固定 `<id>_<N1|N2>_snr_<m18|p0>.wav`；seed=(2026+little_endian_uint32(BLAKE2b(basename,digest_size=4))) mod2^32。记录不同SNR basename导致不同seed；不增seed。
- 核对8条publisher TXT原字节/ZIP entry/hash与registry，保存reference来源；不依赖clean ASR生成reference。标准库独立重建配方与40份文件及参考，不用原业务函数自证。

CPU执行预算120s（118执行+2收尾）、输入阶段<24MiB，整个新实验父目录<64MiB；不计symlink重复。若预计预算不能支持完整32条件，返回Root，不静默缩小范围。

## 3. 模型路径与观察（原生入口不改）

- 复用先前成功attempt02的原生调用与只观察包装设计，**不重跑旧配置或其import smokes**。真实权重/模型库导入只能在Worker模型阶段。
- checkpoint为已用80k `joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`（SHA `ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9`，Root已对照旧manifest并实际重哈希一致）。真实锁以已验证旧manifest/磁盘复核建立；EMA、mode2、storm frontend、no residual-refine；只有原生PC reverse_diffusion、corrector none、N30、snr0.5、seed2026，Router/calibrators/regen全off。
- `joint_enhance_by_dir.main`一次，32个输入按原生basename排序，32 frontend、32 OFF，全部NFE30。包装只观察/返回原参数对象及原输出，不改变计算、不加timeit/trace/noisy权重、不换OUVE中心。
- 留存输入/真实norm/seed/长度匹配、preclip raw float32、native frontend及OFF PCM、clipping/serialization计数。同libsndfile内存再编码核验，native发生的clip如实记录，不按输出峰值排除坏样本。
- 仅1次增强模型装载，释放后再装1次ASR；weakref/显存记录。锁全部实际权重路径与外部frontend配置，未知权重在加载前停止。
- 不增加新smoke，不修旧源码。沿用成功的严格offline/local-build guard与blocked-local-IPv6-probe分类：所有socket仍拒绝；只允许已锁定的本地import探测分类，未知网络事件在下一scientific call前技术停止。0网络字节、0训练、无retry。保留原native扩展缓存/PATH，不因cache位置变化诱发无计划编译。

## 4. ASR和完整评分

沿用已冻结tiny.en本地权重和成功旧pilot有效选项：en/transcribe、T=0、beam_size=1、best_of实际None、fp32、condition_on_previous_text=false、initial_prompt=None/内部prompt=[]、no_speech_threshold=None，不使用真实文本prompt。不是声称greedy decoder或每API一次forward。

8 scaledclean控制 +32×(noisy/frontend/off)=104次transcribe API硬上限；internal decode另计并逐次核对有效options，神经forward未测则明确未测。所有调用固定全句PCM输入与原幅度，不响度匹配/重切窗/时间搜索。

reference normalization：U+2018/U+2019→ASCII apostrophe、lowercase、word regex `[a-z0-9]+(?:'[a-z0-9]+)*`，character regex `[a-z0-9]`（仅word tokens拼接后字符）；DP tie顺序diagonal>delete_reference>insert_hypothesis。保留literal reference/hypothesis、逐word/character alignment及S/D/I/N；WER/CER不cap于1。clean控制不按零错筛句、不换reference。零N输出null并标识，不伪造0率。

## 5. 主要候选余量统计（纯离线，不是selector）

每case以word edit_distance E决定oracle：E_front<=E_off选front（tie固定front），否则off。noisy与scaledclean从不入候选池。不按quality或CER改这个选择。

对all、N1/N2、-18/0、四noise/SNR格、4speaker和8utterance分别报告：

- 三臂原始S/D/I/N、micro WER/CER、uncapped句均；固定臂赢/输/同E的case数及全逐例输出。
- E_best_fixed=min(sum E_front,sum E_off)，tie标front；E_oracle=sum min(E_front,E_off)，N=sum reference words。
- headroom=(E_best_fixed-E_oracle)/N；同时列相对always-front与always-off的原始错误差。该批“best fixed”是诊断参照，不是用确认结果训练的部署策略。
- WER-oracle的S/D/I/N、CER和固定选择列表；不另外寻找CER-oracle，不逐词拼接候选、不挑最有利headroom指标。
- 以speaker为cluster、整簇保留所有句子/条件，固定seed20261019、10000次有放回抽4簇，计算OFF−front micro WER差和primary headroom的2.5/97.5百分位。仅探索性bootstrap区间；4簇不能证明可靠总体覆盖，不当显著性或升级gate；并列4位speaker结果。

无预设“改善X%才算成功”的找通过阈值。无代理余量则不继续这套pool/端点的selector；ASR失效则记录未决，不推出真实不可恢复。若有余量，也只允许提出另批的无X/简单残差/部署X分数排序反证，仍不能开训或改波形。

## 6. 质量/保真对照（不混同人耳自然度）

对每case noisy/front/off的**实际PCM**，以同scale的clean PCM为参考，固定16k/原长度/不额外gain或lag拟合。报告zero-mean SI-SDR、STOI(extended=false)、PESQ(16k,wb)与OFF−front逐例差/分层均值。使用现存本地pystoi0.4.1/pesq0.0.4，冻结实际代码/二进制和API；不安装/下载依赖。

SI-SDR：中心化reference s与estimate x，alpha=<x,s>/<s,s>，10log10(||alpha*s||²/||x-alpha*s||²)。非正reference能量、zero projection或perfect residual产生有原因的null，不用任意epsilon伪造有限分数。STOI/PESQ若因退化信号等计算失败，保留case/异常/null与有效样本数，不用排除样本制造改善。技术管线/源锁不匹配仍是整体技术失败；单metric错误是测量缺失，不能写成完整quality覆盖。

Root已只读核对现装API：`pystoi.stoi(x,y,fs_sig,extended=False)`内部会重采样到10k并去静音帧；帧不足会warning并返回1e-5。必须记录warning与原返回值，将该sentinel情形列为无效测量/null，不无声当作有效极低分。`pesq.pesq(16000,ref,deg,mode='wb')`包装器会共同peak缩放两输入，PESQ内部也有标准处理。保留这些标准metric预处理并披露；“不额外gain/lag/重采样”指本项目不另改信号或搜参数，不声称metric内部从不做缩放/对齐/重采样。区分输入/增强/ASR管线重采样计数与metric内部预处理，不把前者0误写为全部重采样0。不用pesq_batch/多进程。

这些指标不代表用户A式自然度/真实音素身份，未做新人工试听则不伪造。报告质量与WER方向不一致，不只挑内容或自然度之一。

## 7. 成本、安全、复现与阶段许可

模型唯一执行180s总预算（170执行+10收尾），新实验父目录总64MiB（含metadata/输入/模型产物，symlink仅计本身不重复目标；按旧审计惯例排除symlink重复但需披露）。保留至少1MiB收尾余量；最多32 frontend/32OFF/104transcribe，0训练/网络；无追加seed/窗口/采样数/模型/数据替换或自动重试。超限保留部分结果并技术停止。

在 `input_stage/`、`model_stage/` 中各用独立config/metrics/report和独占STARTED marker。元数据原15文件及registry、旧VoiceBank/B/公开TextGrid指标保持锁；只写新阶段。原始数据/checkpoint/dirty树/bridge/试听服务不动，不push。

Engineer只做实现、静态/合成测试与source锁，**不运行真实准备/模型**。先交CPU阶段供Root审/Worker执行；input registry验证后才完成model manifest。Root再审后Worker才能唯一模型运行。独立CPU validator重算完整40输入与32case/104评分/原始序列化、quality与oracle/bootstrap定义；如复用业务函数需披露，不能冒称独立。

本设计通过不证明长期goal完成，试验若无收益应影响下一研究决定，不以换指标/改标签延续路线。
