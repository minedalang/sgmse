# CPEL M/P短程筛查：准备版，尚未执行

## 目的、授权与原范围

检验在相同反事实样本、逐点辅助任务、受控输入及Score调用结构上，加pair项是否比M控制多出可迁移的响应收益。不是最终新方法确认，不以此取代普通增强A/B、真实内容/身份、音质及效率的完整目标。当前仅授权隔离代码准备及已完成的metadata清单；**没有Root执行许可JSON，不能读取真实模型/音频或训练。**

用户暂停时已停止Engineer；本轮Root通过goal接口实核active后恢复软件准备。所有修改仅在本目录，现有脏worktree、主训练入口、原数据/checkpoint及旧实验保持。固定配置详见config.yaml；其中数值是一次开发优化假设，不是验证过的声学识别阈值。

## 数据：冻结清单而非重新挑例

train256/dev64复用已完成coverage audit元数据，四个名义SNR区间分别64/16条，按固定hash排序，排除旧181窗的8个file ID和重叠speaker10658。train/dev分别221/52speaker，所选speaker、fileid、clean/noise实例basename均不交叉。路径仅复制，metadata准备阶段没有重新读取header、PCM或证明当前文件完好。

CV是既有预训练/选模CV，不是新确认集；noise实例隔离不等于噪声录音/类别隔离。文件名SNR不是局部音素可辨性。真实执行需先验证路径、共同裁窗、归一化和加性前提；不调整gain/lag去让检查通过。noise-only需对实际评估crop检查X≈S+N，relative RMS误差>1e-3时该诊断记unknown/失败原因，不重新挑例或假装纯噪声。

训练随机同裁窗；开发固定中心crop。S1/X1共用原X1峰值归一化，构造Delta=.5*w80ms*H(S1)，S0=S1−Delta、X0=X1−Delta，然后两端分别STFT/压缩。H为固定平滑高频带算子，事件中心只由clean高频能量决定，不由F/后端误差或测试结果反选。该编辑不是自然音素存在/不存在金标。

## M/P的共同训练与梯度

同一个80k checkpoint materialize已保存EMA一次；所有臂同初始backend/frontend。训练现有dnn参数，前端及其余参数冻结；前端eval。dnn也用eval关闭随机dropout，但保留参数梯度；不使用no_grad包住待训练Score。新EMA仅跟踪backend，避免沿用Router-only参数集合的EMA或future_grad冻结配置。

每个原样本产生两支合法原生DSM状态，正常t∈[.03,1]。另抽t∈[.3,.7]和parent0/1，以该parent的正常前端及合法OUVE状态作共同A/u，detach固定，仅Noisy条件跨支变化。原生两状态共用Gaussian以减方差，不将其称共享state；aux另抽Gaussian。每步为2native＋2commonaux，共4份Score样本等价前向。M/P均计算相同诊断，只有pair权重0与.1不同。

原生DSM定义为`.5 mean|sigma*q+epsilon|²`，明确是新小试中所有臂一致的均值尺度，而不是悄悄沿用旧sum并加mean aux。逐点aux直接计算白化残差

\[
r_h=\sigma q_h+(u-a c_h-(1-a)A)/\sigma,
\qquad L_{aux}=\tfrac12\operatorname{mean}_h\operatorname{mean}|r_h|^2.
\]

这是共同proposal上的辅助clean回归，不称原DSM无偏估计。双方都保留原生DSM，避免用错误的共同epsilon标签。

## 实际pair项：能量缩放，而非照搬阶段0比例目标

设delta=c1−c0、D为每样本谱坐标数，P为时间软权重。active按原始mean|Pdelta|²>1e-12固定，不随t或白化尺度变。令

\[
e=(a/\sigma)P\delta,\quad w=\sigma P(q_1-q_0),\quad
R=\|e\|,\quad v=e/R,\quad b=\operatorname{Re}\langle w,v\rangle.
\]

在active分支，四项为

\[
L_-= [0.1R-b]_+^2/D,\quad L_+=[b-1.5R]_+^2/D,
\]
\[
L_\perp=\|w-bv\|^2/D,\qquad
L_{out}=\sigma^2\|(1-P)(q_1-q_0)\|^2/D.
\]

pair为L_-、L_+、L_perp、L_out的1、1、.25、.2加权和；总目标=DSM+aux+lambda_pair*pair。active分母用精确非零R，先安全替换inactive分母再计算，不能先产生NaN后mask。inactive整项skip而DSM/aux保留；按所有原样本平均并记录active覆盖，不能随有效例数改变整个batch的步长。

此式等价于把有效目标的增益/正交比例项按(a²/sigma²)*||Pdelta||²/D加权，避免极小事件的倒能量放大；outside按自身谱能量定义。直接白化实现降低大clean读出相减的数值消减。它仍是相位感知的压缩谱响应约束，软inside/outside不是严格正交分解；正margin仍是幅度假设，不能当可辨识性保证。

优化固定Adam1e-5、clip1、新backend EMA.999、pair batch1、float32/complex64，不开AMP/adapter。最大500更新每臂，不作loss/time/window sweep；不会以一个失败配置否定所有CPEL可能性。

## 开发端点、预算与退出

每个臂对全部64开发crop做正常前端、原OUVE PC/reverse_diffusion/none/N30原生推理；统计normalized wave MSE、SI-SDR、STOI、PESQ，所有失败/undefined保留覆盖，不重新挑例。初始backend作为一次基线，用相同配置/seed可复用；不加RRSD、外部Score校准、ASR、DNSMOS或音频导出。

受控诊断固定t=.5、两个parent、三类干预（训练Hann、未见Tukey边缘、noise-only Tukey），M/P完全同输入与随机性。noise-only clean差分为零；另报沿noise-input编辑方向的投影和响应能量，不把它伪称clean方向。零loss/反事实敏感性不能证明拒假音。

noise-only沿用heldout speech编辑的中心、窗与滤波器，并只向下缩放噪声编辑，使其波形能量与heldout speech编辑相等；原noise编辑能量不足/为零则记unmatched/unknown，不放大噪声或换位置以凑对照。其谱响应投影方向是观察到的Noisy两端各自压缩之差Phi(X1/r)−Phi(X0/r)，不是Phi(噪声Delta)，也不是不存在的clean差分。纯噪声与能量匹配均须单列有效覆盖。

一轮开发信号的预定观察条件：M的未见编辑pair组件均值>1e-12且P≤.9M，非零响应覆盖不能解释为退化；noise-input gain均值增加≤.02；正常P−M均值不越过PESQ−.03、STOI−.005、SI-SDR−.2dB的筛查损伤界。这些是工程退出阈值，不是听觉显著性阈值或正式假设检验。缺匹配覆盖/噪声来源不可靠时不得给完整通过结论。无迁移、追随noise或M已解释收益，停止扩大；不能更换窗/权重救本轮。

各臂含评价≤3600s，总GPU时间≤7200s；产物≤512MiB，磁盘至少保留1GiB。仅存终止时backend EMA，不复制原front/checkpoint，不保存多个大optimizer快照。不删除用户文件腾空间。若某臂预算/OOM/错误未到500步，记录实际终止状态，主比较不采用不匹配的两个最终步数，也不自动重启。记录实际Score样本前向、反向、墙钟和显存；“不增推理NFE”不等于训练免费。

软件准备完成后Root核代码/config/registry SHA，再决定是否单独授权真实小型preflight；只有preflight核加载/梯度/尺度/资源后，才考虑M/P执行许可。Worker例行执行，Root不等待长训练；本目录当前不存在这两类真实执行许可或实际训练结果。最终标准测试和真实音素/身份确认仍属于后续完整范围，不能被本试点代替。
