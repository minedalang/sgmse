# 未编辑缓存的结构可读性：一次有界线性读出反证

## 决策与问题

第一阶段配对/编辑技术准备与单听者待办材料已完成。听感不再阻塞其他研究。本次由Research Lead批准进入**独立的阶段2受限探索**，只拟合小型线性读出，不训练StoRM、Score或新的部署检测网络。拟合本身明确记作probe学习，不宣称“零训练”。独立Lead复审认为问题与旧DNS扩散状态误差probe不同，值得一次有限反证。

可反证问题：固定119维可观测表示是否比相应6维简单基线，更易预测两个clean连续谱结构量？同时描述Noisy与前端表示的差异。不是原H1/H0反事实分类，不是音素身份或Noisy信息论可辨识性证明，不替代完整P8-I/CAEG。

## 材料与不变项

仅现存VoiceBank train缓存：8句/4speaker、两噪声×−18/0dB，共32格；读取32原Noisy、32已生成frontend PCM、8 scaled-clean，共72唯一PCM。路径/SHA从冻结input_registry与model metrics按basename/id一一join。不得读取OFF后端、原始噪声、checkpoint、原始语料或冻结DNS最终测试集。原32格增强/ASR及后续MSE方向、四seed实验仍关闭，不重生成波形、调用模型或改变任何旧件。

原8句/4speaker已经参与其他探索，基础模型预训练暴露未知；LOSO只约束本次线性拟合，不制造全局unseen或noise-disjoint确认集。两噪声重复版本和四条件不是独立句子。相同119维读出容量不代表前端与原观测总表征计算相同；前端曾使用整句上下文，所以只称“上下文＋表示的可读性对比”，不作纯信息损失归因。

## 固定数值定义

- 16k/mono/PCM16原样转float64，除32768，不重新归一化/对齐/增益拟合/重采样。三信号同长度，所有完整320样本帧、hop160，尾余样本数单独报告，不按能量或Clean标签筛帧。
- 窗为对称Hann：`0.5-0.5*cos(2*pi*n/319)`，n=0..319；FFT512，取rfft257 bins。单边能量权DC/Nyquist为1，其余为2。总frame power=`sum(w*|FFT|^2)/(512*sum(window^2))`。
- HF固定bins128..240（4–7.5kHz，含两端）。目标1为HF加权能量／全257bin加权能量；分母精确0时定义0，不视作无语音标签。
- 令`m=sqrt(w)*abs(FFT)`。目标2为`norm(max(m_t[HF]-m_prev[HF],0))/(norm(m_t)+norm(m_prev))`；两norm之和精确0时定义0，首帧prev=current使flux0。目标都不等同音素或语音活动真值。
- 16谱带为`np.array_split(arange(257),16)`：首带0..16，后续各16bin，最后241..256；不重叠覆盖全部bins。16带能量fraction＋`log10(max(frame_power,1e-12))`共17维。
- full表示：17维当前及±1/±2帧平铺85维（边缘重复feature帧，不补音频），加整句17维mean和17维population-std，共119维；Noisy/frontend使用完全相同提取器。
- simple表示：当前logpower/HFfraction/flux加各自全句均值，共6维。另报告无拟合identity（对应观测自身HFfraction/flux）及训练target均值constant；不从确认结果择优拼出新baseline。

## 拟合、终点与记录

四个leave-one-speaker-out折，每折完整留出该speaker的8条件。full-Noisy/full-frontend/simple-Noisy/simple-frontend各折一次双输出Ridge，共16次拟合，固定lambda0.01，无超参数搜索。

每个训练case等权、case内帧等权，权重总和为1；feature标准化的mean/std也用同一权重。精确零std用1，intercept不标准化且不惩罚。目标为`sum(weight*||prediction-target||^2)+0.01*||beta||^2`；两个输出独立但共用设计矩阵，禁止把未归一权重和当成相同lambda。标准化/系数/constant只由训练折获得，probe不读Clean、ASR、真实SNR、speakerID或编辑角色作为输入；SNR仅作离线分层。

主终点为两个目标**分别**的heldout MSE：先逐case平均再等case汇总，并报告每speaker原始差值、−18/0分层、identity/simple/constant对照；full对同臂simple、identity和constant逐项比较，不只对随机或均值。不将两目标择优写成功，不报告4cluster的总体显著性。预测不clip到[0,1]，报告越界比例；所有帧/32case保留。

保存每折训练/留出ID、训练标准化/系数、逐case分数和逐帧heldout预测/target以便复核。不导出原始feature矩阵、音频或可部署controller。不把probe拟合或代理正向当作增强收益。

## 退出与成本

若full没有稳定超过简单/identity对照、speaker方向不稳或只在某一代理上变好，不自动升级hidden probe、verifier、CAEG或增强训练。即便正向，也只是有限结构可读性，不能定位丢失了哪个真辅音；Noisy无增益也不证明无信息。禁止换窗、lambda、层、帧筛选或新seed救结果。

NumPy-only、单CPU线程，主≤120s、独立复核≤120s、全新目录≤8MiB，0GPU/增强模型/ASR/外网/新波形。Engineer实现最小脚本与少量合成检查；Root审后Worker唯一执行。独立复核不import业务函数，使用同NumPy FFT但独立谱/feature bookkeeping、以加权增广least-squares复算Ridge（与主normal-equation区别明确）。不把共享FFT库称完全独立算法。失败保留且不自动重试。
