# 已实现方法正文草稿：冻结基座上的Future-gradient Reliability Routing

> 本章描述已实现的Router，不把P8-I／CAEG候选写成完成方法，也不单独认定新颖性或综合增强成功。训练历史经用户确认：300k前后端基座冻结，仅Router增量训练80k。当前源码、已保存配置及历史结果互相区分；源码对应关系不是重训复现。结果和条件性权衡见[原结果草稿](THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md)，内容局限见[P8讨论](P8_EVIDENCE_DISCUSSION_DRAFT_20261019.md)，先例见[相关工作](P8_RELATED_WORK_DRAFT_20261019.md)。

## 1. 问题与表示

记含噪波形为 (x)，训练clean为 (s)，冻结前端为 (F)。在`normalize=noisy`的共同标度下，令 (r=\max(\|x\|_\infty,10^{-12}))，clean与noisy使用同一 (r)，而不是分别按各自峰值归一化。以实际STFT及压缩变换 (\mathcal T) 定义

\[
Z_0=\mathcal T(s/r),\qquad Y=\mathcal T(x/r),\qquad A=F_{\rm spec}(Y).
\]

这里 (F_{\rm spec}) 表示前端在同一变换谱域的输出；波形推理入口先完成相同尺度下的前端处理和长度匹配。保存的谱变换为指数幅度压缩，指数0.5、factor0.15。下文全部加减及路由发生在该模型坐标，不将 (Y-A) 称为物理噪声，也不把谱域MSE等同于波形MSE或音素正确率。

所审配置为mode2：OUVE主锚点 (y_{\rm main}=A)，直接条件 (y_{\rm cond}=Y)。Score路径接收状态及两个条件；不是仅看到前端输出。将 (A=F_{\rm spec}(Y)) 同时输入不构成第二份独立物理观测。

## 2. 固定OUVE几何与clean读出

在未启用其他残差锚点变体时，正向过程为

\[
dZ_t=\theta(A-Z_t)dt+g(t)dW_t,
\qquad a_t=e^{-\theta t},\quad b_t=1-a_t,
\]
\[
Z_t=a_t Z_0+b_t A+\sigma_t\varepsilon.
\]

其中 (g(t)) 与扰动标准差 (\sigma_t) 是两个不同量，分别由`OUVESDE.sde`和`_std`给出。本文沿用代码的复噪声与Score输出尺度，记冻结网络的有效输出为 (q_\theta(Z_t,A,Y,t))，不把它另行包装为独立声学似然或改变其factor约定。

代码采用的代数clean读出为

\[
\widehat Z_0=
\frac{Z_t+\sigma_t^2 q_\theta-b_tA}{a_t}.
\]

实际实现对 (a_t) 设 (10^{-5}) 的下界。在本配置 (t\in[0.03,1])、(\theta=1.5) 的范围，该下界不触发。此读出不是已证实的真实clean或波形MMSE；有限网络误差、表示变换和采样过程的限制仍存在。

## 3. 局部路由器

### 3.1 输入特征与结构

路由器 (Q_\phi) 只接收推理可用量：(Z_t,\widehat Z_0,A,Y,t)。在最多32个频带、16个时间块的低分辨率网格上，构造以下19个实特征通道：

| 特征组 | 通道数 |
|---|---:|
| (\widehat Z_0,A-\widehat Z_0,y_{\rm main}-\widehat Z_0,y_{\rm cond}-\widehat Z_0,Z_t-\widehat Z_0) 各自局部幅度的mean/std/max | 15 |
| 前三种差异向量的三组局部实内积归一化关系 | 3 |
| 归一化扩散时间 (t/T) | 1 |

这些不是19份独立证据；mode2中 (y_{\rm main}=A)，部分特征重复。网络依次使用1×1投影、两个3×3上下文卷积及1×1输出，隐藏通道48，中间采用SiLU。Sigmoid得到低分辨率系数，经双线性插值得到谱网格上的 (\alpha_\phi\in[0,1])。已有checkpoint审计计得42,577个Router参数，基础前后端保持冻结。

### 3.2 动作及Score变换

路由动作是向当前前端谱移动：

\[
\delta Z_0=\alpha_\phi\odot(A-\widehat Z_0),
\qquad \widehat Z_0^{\rm route}=\widehat Z_0+\delta Z_0.
\]

在相同 (Z_t,A,t) 下，将这一clean读出动作换回Score坐标：

\[
q^{\rm route}=q_\theta+
\frac{a_t}{\sigma_t^2}\delta Z_0.
\]

实现对 (\sigma_t^2) 使用 (10^{-8}) 下界，并有非有限数保护。本配置路由为单alpha模式，不是“gate×strength”双头；表述中不应加入未启用的门控机制。谱域凸组合也不能推出波形逐样本凸组合，更不能保证补回两候选中均缺失的正确音素。

## 4. 一步future重评分监督

### 4.1 同状态、同未来时刻的对照

训练时，先从原扰动路径得到 (Z_t,q_\theta,\widehat Z_0)，将它们及条件detach。只有 (Q_\phi) 的动作保留梯度。对同一状态分别施加路由动作与零动作，并作一次确定性的反向均值步：

\[
Z_{t'}^{j}=Z_t-f(Z_t,A,t)\Delta t
+g(t)^2\Delta t\,q^j,\qquad
j\in\{0,\mathrm{route}\},
\]
\[
q^0=q_\theta,\quad
\Delta t=\min(\Delta t_{\rm base},\max(t-t_\epsilon,0)),
\quad t'=\max(t-\Delta t,t_\epsilon).
\]

保存的`rrsd_lookahead_dt=0`在当前实现中取

\[
\Delta t_{\rm base}=\max\!\left(\frac{T-t_\epsilon}{\max(N-1,1)},10^{-6}\right).
\]

这不是完整PC轨迹：该训练比较不加predictor随机噪声、不运行corrector，且不越过 (t_\epsilon)。两分支在相同 (t') 使用相同冻结网络重新估计clean，消除把未来误差与当前时刻误差作比分母的跨时间混杂。重新评分时两分支沿batch维堆叠，是一次2B大小的网络调用，不是无成本调用。

### 4.2 相对目标与平滑

令 (E_i^j=\operatorname{mean}|\widehat Z_{0,i}^j(t')-Z_{0,i}|^2)，实际相对目标为

\[
L_{\rm future}=\frac1B\sum_{i=1}^B
\frac{E_i^{\rm route}}{\max(\operatorname{stopgrad}(E_i^0),\epsilon_o)},
\qquad \epsilon_o=10^{-6}.
\]

分母是clamp，不是 (E_i^0+\epsilon_o)。当 (E_i^0\ge\epsilon_o) 时，比分别小于／大于1可描述相对零动作的改善／损伤；分母下界触发时不能机械沿用“零动作恰好等于1”。优化的是一步、压缩谱域的监督代理，不是DNSMOS或真实音素恢复的可微替代真值。

低分辨率alpha的平滑项为两个方向的平均L1差分之和：

\[
L_{\rm smooth}=\operatorname{mean}|D_f\alpha|
+\operatorname{mean}|D_\tau\alpha|.
\]

当前审计checkpoint的Router有效目标为

\[
L_\phi=\lambda_R\bigl(\lambda_F L_{\rm future}
+\lambda_S L_{\rm smooth}\bigr),\quad
\lambda_R=1,\ \lambda_F=1,\ \lambda_S=0.01.
\]

训练代码还可计算基础DSM及其他分支；在本文所述冻结路径中，未经过Router的项不向 (\phi) 提供梯度。保存的gate cost系数虽为0.001，但`rrsd_use_gate=False`使该项实际为零；`contrastive_weight=0`，不能声称主结果来自WavLM／UniSpeech对比监督。其他后续可选分支不能因为存在于当前源码，就倒填为原80k训练使用的目标。

### 4.3 冻结参数不等于切断输入梯度

梯度链为

\[
\phi\rightarrow\alpha_\phi\rightarrow q^{\rm route}
\rightarrow Z_{t'}^{\rm route}
\rightarrow q_\theta(Z_{t'}^{\rm route},A,Y,t')
\rightarrow\widehat Z_0^{\rm route}(t')\rightarrow L_{\rm future}.
\]

基础网络参数不更新，但该实现保留其对输入的导数。若将future网络调用放入`torch.no_grad()`，会切断经future Score输入Jacobian的梯度分量；clean读出仍有对未来状态的直接梯度项，因此不应误说Router必然完全没有梯度。这会改变预期的future-gradient监督，而不是等价的省显存实现。clean仅作为训练误差参考，不进入部署Router输入；这里的零动作是训练控制分支，不是P8-V的“该音素不存在”声学假设。

## 5. 推理算法与控制口径

```text
输入：原始noisy、冻结前端/Score、已训练Router；无clean
1. 用noisy共同幅度标度得到Y及前端锚点A。
2. 按既有OUVE prior在A附近初始化随机状态Z_T。
3. 对N个递减时间点：
   a. 计算一次基础Score q_theta。
   b. 若rrsd=true，读出Z0_hat，计算alpha和delta_Z0，转换成q_route；
      scale及可选t-cut作用在动作上，当前实现不提前跳过小网络。
   c. 使用原reverse-diffusion predictor更新状态，保留原随机项。
4. denoise=true时返回最后一跳的离散更新均值，再逆谱变换/ISTFT与恢复尺度。
```

本配置为PC、reverse-diffusion predictor、`corrector=none`、N=30。最后返回的是算法末跳均值，不是整条路径的后验均值，也不等同于直接输出上文的clean读出。训练的一步确定性future目标与最终随机推理目标因此并不完全相同。

同一80k checkpoint上的zero-scale与启用路由控制了基础权重，但成本口径仍要明确：旧正式`off`是`rrsd=true, scale=0`，也运行30次小Router；P8 native OFF的`rrsd=false`才跳过Router。两者基础Score静态评估均为30，不能据此说总计算完全相同或Router零开销。完整计时限制见[复杂度说明](THESIS_INNOVATION_RESULTS_DRAFT_2026-10-20.md#24-复杂度与计时口径区分训练lookahead和推理路由)。

## 6. 参数与来源定位

| 配置 | 已保存取值 |
|---|---|
| 基座／增量训练 | 用户确认300k前后端基座冻结＋80k Router训练 |
| SDE / mode / noisy主条件拼接 | OUVE / mode2 / `mode2_include_noisy_main=True` |
| OUVE stiffness、sigma范围 | 1.5；0.05–0.5 |
| t_epsilon / N | 0.03 / 30 |
| 谱变换 / 归一化 | exponent0.5、factor0.15 / noisy |
| Router网格 / 隐藏通道 | 32×16 / 48 |
| teacher / gate | future_grad / False |
| epsilon_o / lookahead_dt字段 | 1e-6 / 0（按上文规则解析） |

保存字段来源：[静态checkpoint记录](experiments/p8_checkpoint_base_identity_20261019/metrics.json)中`comparison.A_hparams`及[机制审计](experiments/rrsd_router_mechanism_audit_20261020/metrics.json)。数学与代码对应：`sgmse/joint_data_module.py`归一化，`sgmse/sdes.py:OUVESDE`，`sgmse/reliability_router.py:BandTimeReliabilityRouter`，`sgmse/joint_model.py:_rrsd_score_to_x0/_rrsd_reverse_mean_step/_rrsd_future_grad_loss/_compute_rrsd_loss/build_rrsd_score_calibrator`，`sgmse/sampling`。

本稿没有重新训练、运行增强或验收历史训练环境；未保留的历史运行细节仍不得由当前默认值填补。它是方法正文交付，不是新机制验证。现有质量／重建权衡与P8内容证据缺口继续保留；是否足以构成论文算法贡献，需要结合先例和公平实验证据判断。
