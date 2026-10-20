# 固定后端修改的方向—幅度审计：仅既有完整PCM，零新模型

Research Lead批准一次有界CPU诊断。**新问题**：已观察到OFF相对frontend的保真下降，是否具有“沿同一修改方向缩短步长即可改善MSE”的oracle空间，还是方向本身不改善MSE？同时分开纯全局gain与非gain修改。此为已完成32条件产物的离线误差几何，不重开生成实验，不改其主指标/ASR，也不运行selector/残差混合方法。

既往先例检查：rrsd_utterance_oracle_150是候选臂选择；anchor_v2_real_oracle_20261017是每步双方向tile clean-oracle并重跑轨迹；multitrajectory_oracle_heldout16是多seed选择，均不重做。当前32自然train句的完整PCM单条F→B线几何未见相同配置结果。本审计不重命名这些旧方法为创新，也不据oracle成立重开anchor训练。

## 锁定来源与全量范围

唯一输入为 `p8_vb_prospective_headroom_20261019/input_stage/input_registry.json`（SHA cfc2e1b4ac313037f7e324b6e548f0f56cb1ee32a80282475a0823c7a876d998）、其8份scaledclean PCM，以及`model_stage/metrics.json`（SHA577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04）内全部32个frontend/OFF PCM路径和hash。模型独立validation源记录也锁。只解码这72个唯一WAV（8S+32F+32B）；不读原noisy/真实noise、不读额外样本、raw.npy，不裁片/重采样/对齐/重归一。严格PCM16 mono16k/header/长度一致，全部32保留。解码次数实际记录，缓存S可复用。

source byte/metadata hashes pre/post一致。旧原metrics、reports、input registry、源码/模型/metadata15/服务保持不动。新config/metrics/report写在本目录；原32格仍关闭，不写回或补它的原metrics。

## 冻结解析量：每case直接算，不输出任何混合音频

波形按int16/32768作float64（标准库float与math.fsum）；不中心化。S=同scale clean PCM，F=frontend PCM，B=OFF PCM，d=B−F，e=F−S，L=样本数。

```text
P = mean(S²) > 0
E0 = mean(e²)
A = mean(e*d)
D = mean(d²)
E1 = mean((B-S)²)
E(alpha) = E0 + 2*alpha*A + alpha²*D
alpha_unconstrained = -A/D       if D>0
alpha_oracle = clip(-A/D,0,1)   if D>0; 0 with no_change flag if D==0
```

记录E0/E1/A/D、E1−E0、identity residual、alpha_unconstrained/oracle、E_oracle、相对front和OFF的MSE差、全部以P归一的对应量。验证oracle<=两端，只由clean作诊断，不生成波形/重新量化/运行SI-SDR/STOI/PESQ/ASR。alpha=0、0<alpha<1、alpha=1只描述固定线段MSE；**不是音素恢复、音质最优或可部署门控**。精确数值/端点，不按结果再设阈值划类别。

全局共同alpha以32case**等权normalized MSE**为目标：求sum(A/P)、sum(D/P)的解析最优alpha，并报告其目标、两端目标及mean逐case oracle目标。按all/noise/SNR/交叉4格/speaker/utterance共21层同定义；不新增搜索网格/多目标选最好。小样本描述，不做新bootstrap或总体显著性推断。

## 纯gain vs其余修改（固定投影，不拟合clean gain）

```text
PF=mean(F²)
beta = mean(d*F)/PF                 if PF>0
d_gain = beta*F
d_rest = d-beta*F
```

beta只描述现存修改在F方向上的投影，不是从clean找最佳缩放；PF==0时无gain定义，beta=null、d_gain=0、d_rest=d。计算两部分能量、mean(d_gain*d_rest)正交检查、各自对MSE差的贡献2mean(e*d_part)+mean(d_part²)，两贡献和应等于E1−E0（保留数值误差）。同时报告非gain修改能量占比D_rest/D（D0则null）。

单独gain可以改变MSE而SI-SDR对非零全局缩放不变，所以**本审计不能把MSE变化全部当SI-SDR下降原因**。只将原quality中的SI-SDR逐例差作为已测描述引用，不再计算或用其筛样。非gain项还可能包含时移/相位/残噪/语音细节，不能从其能量判音素对错。

## 对照、退出与限制

保留两个真实端点F/B、逐case oracle和共同alpha oracle；所有oracle明确使用clean，只作“当前方向有何MSE空间”的上界诊断。禁止导出/试听oracle混合、把alpha按同批ASR/质量优化、增加频段/局部窗口/lag/seed。若无空间就停止缩幅路线；若有空间也只能说明方向—幅度分工值得后续新问题审查，不授权学习alpha或宣称内容/质量改善。

量化PCM是实际已评分信号，本审计不替代preclip raw分析。共同per-utterance scale已冻结，归一MSE等权避免大幅值单句主导，但不把它与SI-SDR/WER混为同一个目标。

## 实施与交付

Worker可直接用标准库实现这一小型分析，不做通用框架。独占STARTED marker；已有结果不重做。预算60秒主分析/2MiB新产物，0模型/ASR/网络/训练/音频输出。独立CPU核验单独实现：直接构造e+alpha*d的临时内存标量和重新求和，检查二次恒等式/投影贡献/全部32case/21strata；**不保存波形**。可读原PCM重验，但准确记录额外解码次数，不声称总计只有72次。

交config.yaml（Root提供不改）、metrics.json、experiment_report.md、源码/独立验证与执行溯源。Root代码/科学复核前后保护哈希、当前git状态与bridge/试听实际存活，无模型并发；第一次技术失败记录停止不自动重试。若标准库实现需超范围，交Engineer另准备，不能自行改变科学问题。
