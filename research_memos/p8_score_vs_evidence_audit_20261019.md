# P8-V：条件score不是独立的声学似然

**性质：真实代码+数学/原始文献审计，无模型运行或新声学指标。** 32条件测量及旧VB/VCV/5243/B继续关闭；不实施score重排、likelihood guidance、noise模型训练或额外ASR。盲听未实施，不把缺听者当作只能停止科研的理由。

## 1. 本轮明确的实现事实

- `sgmse/model.py:1164–1255`：`forward(x_t,y,t,y_cond)`可在mode2主分支接收原noisy；网络输出还经preconditioning/目标分支转换，不能把任意DNN原始张量都叫score。`_loss`的score_matching分支以`sigma*score+z`为扰动噪声目标，而不是显式观测noise N的概率。
- `model.py:1415`之后：mode2归一化参考为原noisy峰值，anchor/condition共同缩放再作STFT/transform。已完成native run的实际norm/条件已观察核验，不仅是注释推测。
- `sgmse/sdes.py:289–375`：标准OUVE漂移朝anchor、mean为a(t)*x0+(1−a(t))*anchor，prior_sampling以anchor加Gaussian启动；`prior_logp`仍NotImplemented。这个终端过程prior不等于已训练的无条件clean-speech语料先验。
- `sgmse/joint_model.py:2965–3010`：已有`_score_step_geometry`用mu_hat=x_t+sigma²*score，再经OUVE weights反解x0；不是把mu_hat直接当clean。该helper存在不等于这次native OFF调用了它，不能把静态辅助方法当实际运行trace。
- `sgmse/data_module.py`的`spec_fwd/spec_back`实现可选非线性幅度指数/log变换。**本轮没有重新加载checkpoint，也没有从默认值认定80k实际指数。** 已存checkpoint审计记录对应同一SHA，列mode2/OUVE/score_matching，但没有列spec_abs_exponent/transform_type，故精确运行表示参数仍需合法元数据证据，不能猜成默认0.5。

上述四源码SHA及原checkpoint审计文件由本轮审计回执冻结。无源码修改。训练目标与代码接口允许讨论其理想score语义，但不是证明当前网络恰好等于某个真实密度的梯度。

## 2. “用生成器自证有声学证据”的逻辑缺口

令X为实际混合、A=F(X)为冻结确定性前端。理想条件密度满足 p(S|X,A)=p(S|X)，因为A是X的函数；A没有增加独立观测。这不表示计算上anchor没用，也不表示网络真的达到这个理想分布。

在固定同一个t、固定X/A、同一表示/归一化下，若学习score恰是保守场∇z log q_t(z|X,A)，沿路径积分最多得到：

```text
integral score(z,t,X,A) dot dz = log q_t(z1|X,A) - log q_t(z0|X,A)
```

它是**有限扰动时刻的条件密度比**，不是自动得到log p(X|S1)−log p(X|S0)。干净候选与z_t之间还要说明OUVE均值/扰动/表示，不能把t>0的网络输入直接当原波形。近似神经score未强制保守，数值路径积分可能依赖路径；score范数本身甚至不是密度高低的排序（mode处score可为零）。

独立Lead复审进一步指出：沿真实采样轨迹t也在改变时，d log q_t(z_t)/dt还含显式∂t log q_t项，不能只累加score·dz就把结果当上述固定t的endpoint密度比。该问题在实施路径特征前必须命名清楚，不靠观测NFE30就省掉时间项。

即使假定可以获得真实clean-space posterior比，Bayes仍给：

```text
Δ log p(S|X) = Δ log p(X|S) + Δ log p(S)
```

因此模型更喜欢一个候选，可能是更强先验，不是更强观测证据。这是概率语义的区别，不是已经证明当前StoRM一定自我强化、或所有score-based特征必然无用。

**解析反例（非声学实验）**：S先验N(−2,0.25)、X|S为N(S,1)，观察X=1，比较S0=−1/S1=1。纯likelihood log-ratio为+2，prior log-ratio为−16，posterior log-ratio为−14。真实posterior score为−5S−7，积分−1到1恰为−14。这个理想模型都能因先验偏好与纯观测排序相反，不需要假设网络出错。

## 3. DPS不能不加推导地贴到当前条件Score上

[DPS v4，§2.2–3，Eqs.5/7/15](https://arxiv.org/html/2209.14687v4)从数据prior score与观测likelihood分解出发，并近似有限t的likelihood；它不等于“只要有任意已条件化的增强score，就可以再加同一X残差梯度”。OpenReview PDF本轮遇browser验证，随后读arXiv指定v4，未绕过验证或下载模型。

若在理想clean空间已有p(S|X)上再乘同一个p(X|S)^λ，得到p(S)*p(X|S)^(1+λ)（归一化常数略），是观测再加权/温度化，不是原posterior。这只是明确前提下的代数；实际OUVE噪声时刻/离散数值更新不是这一公式的自动实现。**再加权可能有经验收益，不能仅因“重复观测”就断言一定错误**；但必须作为新目标/近似重推、校准和matched-compute对照，不能用DPS名称担保合法性。

同样，把y_cond置零、删除anchor或用noise-only条件来提取“无条件prior”，并无训练支持保证。原网络训练条件和OUVE中心都变了，差分可能是OOD冲突；不是Bayes意义上从posterior精确减去prior。旧条件移除负结果不因这个新解释重开。

本机`forward`还会在`y_cond=None`时回填`y`，不是执行无条件分支。即使条件score精确，缺少另有约束的prior或noise模型，也一般不能唯一拆分成二者；不要把接口默认值当condition-drop训练能力。

## 4. 压缩谱的残差不等于物理noise

线性复STFT可保留X=S+N，但非线性幅度映射G一般不满足G(X)=G(S)+G(N)。例如正实数S=4,N=5且G=sqrt时，G(9)=3而G(4)+G(5)=4.236…；因此G(X)−G(S_candidate)不能无说明当作G(N)或白Gaussian观测残差。

观测likelihood应在波形/线性复谱等明确定义域建立，或推导非线性观测映射/协方差/梯度与变化测度；不能把当前score坐标中的L2距离直接称物理加性noise似然。含冗余STFT与复杂相位的表示转换也不能随手补一个标量Jacobian就声称完整概率正确。若只用近似残差启发式，应明确命名和适用限制。

## 5. 补充查新改变了哪些优先权判断

[Ayilo等2026 v1，§III–V](https://arxiv.org/html/2601.09931v1)已在单通道SE中区分speech prior与NMF noise likelihood，并研究显式speech/noise联合后验与diffusion noise prior；它也讨论了有限t likelihood近似及计算代价。**“给speech diffusion加noise模型/残差一致性”不是本项目的空白创新点。** 其unconditional/类型条件prior与本项目mode2 noisy-conditioned OUVE不是同一模型，不能将对方结果直接搬到80k条件Score上。

这项定向阅读不构成穷尽查新或对方方法在本项目极低SNR内容上的复现；只核其分解/噪声建模机制，不搬用性能指标。Uni-ArrayDPS检索摘要涉及多通道SCM，不作为本轮已核全文机制或单通道可直接复用证据。

## 6. 决策与最小可证伪路径（当前均不授权新科学运行）

1. **否定快捷主张**：不把score norm、condition-drop差分、posterior路径分数或压缩谱残差当作已校准的独立声学证据；不立即写likelihood梯度接入旧采样器。
2. **允许未来作为基线受测，而非理论禁用**：若score特征在新的可信内容端点、固定proposal和matched-compute下有增量预测价值，可以是经验自一致性特征；但需超过no-X先验/简单物理残差，对实际真音保留与错补两方向都评价，不把高置信当真值。
3. **真正下一方法门**：先明确“评分对象/观测域/noise假设/有限t近似/校准与内容端点”五项，才讨论有符号证据训练或推理。known-noise Gaussian仅是oracle/合成基线，部署noise应从可见X估计，不能偷用合成真N。
4. 盲听未落实前，不新增该32格的score调用、ASR修参、噪声估计搜索或selector；可先分析先例的误差/成本和识别条件。普通grounding或显式noise先例没有消除本项目真实内容证据缺口，不能因公式合法就扩大训练。

本轮推进的是：排除一种不成立的“免费用现成Score验证声学证据”论证，并补足先例与坐标边界；不是宣称方法已失败或长期目标完成。保留与音质/保真/身份/成本兼容的替代路线。
