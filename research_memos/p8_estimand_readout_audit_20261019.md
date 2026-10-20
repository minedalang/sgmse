# 压缩谱条件均值与原生后端输出：估计对象审查

**性质：代码、解析推导与定向原文审查；不是增强实验。** Root 接受独立 critic 的数学复审，并补核真实 sampler/wrapper 及复高斯文档。没有运行模型、Jacobian、音频解码、ASR、质量指标或训练。旧 VB32、四 seed、VCV、5243 均保持关闭。

## 1. 本轮决定

**数学差别成立，但当前不足以把“逆压缩均值修正”升为实验或创新点。** 原生 PC 输出不是下面推导中的精确 clean 条件均值；不能把一个正确的 Jensen/delta 推导直接当成已经发现当前后端吞音的原因。暂不实现协方差估计、修改最后一步、追加 seed 或训练新 loss。

上一轮缓存分解只显示有限四样本中心参考误差较大，不能识别其原因。此次新增证据进一步排除“因此输出一定是估计对象偏差、加二阶矩就能修好”的跳步推断。保留估计对象区别作为后续方法设计的检查项，而非已证实性能瓶颈。

## 2. 训练读出的对象：必须写全条件

令 X 为 noisy，A=F(X)，n=peak(X) 为固定观测归一化量，Z₀=G(STFT(S/n))。保存的 exponent=.5、factor k=.15 来自既有 checkpoint 静态 opcode 证据，见 `p8_noise_evidence_contract_20261019.json`；不是拿当前默认值冒充保存值。

对固定 c=(X,A,t)，OUVE 扰动为 zₜ=aZ₀+bA+ε，a=exp(−θt)、b=1−a；此处 A 表示对应的压缩谱锚点。源码 `model.py:_step/_loss` 使用 ε=σz 与损失 ||σq+z||²。因此理想无限容量条件下：

```text
q*(z_t,c) = -E[ε | z_t,c] / σ²
μ(z_t,c) = E[Z₀ | z_t,c] = (z_t - bA + σ²q*) / a
```

这是压缩谱 clean 的、还条件于随机状态 zₜ 的均值。它既不是 raw DNN 输出，也不是已经对 zₜ 积分的 E[S|X]。实际网络不必等于 q*，DSM 目标也没有保证内容、校准或后验精确性。

## 3. 非线性逆变换不与均值交换，但不等于采样器有错

在每个实虚二维 bin，G(u)=k u/√||u||，其逆为 h(u)=u||u||/k²。一般：

```text
h(E[Z₀|z_t,c]) != E[h(Z₀)|z_t,c]
```

**解析反例（不是数据实验）**：在某固定条件下，未压缩正实系数 S 等概率取 1、9，压缩系数为 k、3k。逆均值 h(2k)=4，均值的正确逆读出 E[h(Z₀)]=5。沿此正实射线 h 是二次函数，方差项 k²/k²=1 精确补足差值；一般二维分布没有此精确保证。

若 Z 真是目标压缩谱后验的一个**精确样本**，则 h(Z) 是相应未压缩谱的样本；再经固定线性 ISTFT 得到对应波形分布。非线性本身不会让正确的分布变换自动“有偏”。必须区分：变换一个样本、变换一个条件均值、平均多条已逆变换波形。这三种操作不能混写。精确采样前提也不适用于未经校准的当前网络/有限 NFE。

## 4. 实际 native PC 到底返回什么

真实调用链：冻结 `model_stage/run_pilot.py` → `joint_enhance_by_dir.main` → `ScoreModel.enhance` → `get_pc_sampler`。固定配置为 PC / reverse_diffusion / none / N30，三个 calibrator 与 regen 均关闭。没有传 denoise=False。

- `model.py:1296–1306` 将 eps 设为 **self.t_eps**，因此不能只用 sampling 函数的默认 .03 断言 checkpoint 实际末时刻；本轮不加载模型、不新增其运行时测量。
- `sampling/__init__.py:get_pc_sampler` 最后一跳从 eps 到 0；默认 denoise=True 返回 `xt_mean`。
- `sampling/predictors.py:56–65` 中 `xt_mean=x−f_reverse`，另返回 `xt=xt_mean+Gz`；先前路径仍是随机的。denoise 不是把整条随机轨迹积分成后验均值。
- 普通 OUVE、无 calibrator 时，源码末跳均值为 `(1+θh)z_t−θhA+g(t)²h qθ`（最后 h=t）；clean Tweedie 读出却为 `(z_t−(1−a)A+σ(t)²qθ)/a`。两式系数一般不等，即使 qθ 精确也不能直接把前者命名为后者。
- `model.py:1562–1564` 将返回值送入 to_audio，再乘 noisy norm；native 入口随后 clip/写 PCM。既有输出的零裁切验证不改变上述估计对象区别。

**结论**：原生返回的是随机路径上的末跳离散转移均值，经逆变换的结果。不能用 §3 的反例推断它存在同等大小或方向的失真，也不能凭 denoise=True 就叫波形 MMSE。§4 公式只是源码代数，不是实际输出误差测量。

## 5. 二阶矩公式成立的条件与成本缺口

μ≠0，r=||μ||，C=Cov(Z₀|zₜ,c) 为一个 bin 的实虚 2×2 块。二阶 delta 展开为：

```text
E[h(Z₀)|z_t,c] ≈ h(μ)
 + { (2Cμ + μ tr C)/r - μ(μᵀCμ)/r³ } / (2k²)
```

这是近似，可能改变方向，不只是放大幅度。若真实实坐标噪声 ε~N(0,vI)，真实实坐标 score s=∇log p(zₜ|c)，则完整协方差恒等式 C_full=v/a²(I+vJ_s)，取同 bin 的块即可。

**复数约定补核**：安装的 `torch/_torch_docs.py` 与 [PyTorch 2.14 randn 文档](https://docs.pytorch.org/docs/2.14/generated/torch.randn.html) 都写明复标准噪声实/虚部各方差 1/2；[randn_like](https://docs.pytorch.org/docs/2.14/generated/torch.randn_like.html) 按输入 dtype 使用对应采样。源码 `_step` 是直接 complex randn_like。由该文档约定推导 v=σ²/2、s=2q*，故 C_full=σ²/(2a²)(I+σ²J_q*)。这是文档/源码支持的常数推导，**没有做本机随机数实测或实际网络协方差校准**；不能据此指控原 sampler 有 factor-2 bug。

求 J 时固定 X/A/t 及其他坐标。每 bin 的全 2×2 块足以作上述逐 bin 逆变换与线性 ISTFT 的一阶均值近似；不需要所有跨 bin 协方差来求此均值，但联合不确定性或非线性质量指标另当别论。获取这些块不等于一次额外 forward，需先设计并核算实际 autograd/JVP/VJP 成本，不能写“零额外 NFE”省略反向计算。

实际 J 可能非对称、导出非 PSD 矩阵；对称化/PSD 投影是额外假设。μ 近零、后验宽/多峰/跨原点、小 a 均可能使展开或估计不可靠。不能用选择性裁掉坏 bin、clean oracle 阈值或旧样本搜参数挽救结果。

## 6. 先例与否决的包装方式

- [Richter 等，2409.10753v1，§II-A/II-B、IV](https://arxiv.org/html/2409.10753v1)：已讨论 score/noise/denoiser 目标关系、时间域辅助项和感知项，且报告感知与重建权衡。因此“换成预测 clean/加波形或 PESQ loss”不构成此项目独立创新。本文不是当前 StoRM 的复现实验。
- [Rout 等，STSL，2312.00852v1，§2–4、Eq.(3)/(5)](https://arxiv.org/pdf/2312.00852)：已研究第一阶 Tweedie 的 plug-in 近似、二阶矩与 Hessian-trace surrogate，用于 latent diffusion 逆问题。该似然/采样修改**不等于**本 memo 的逐 bin 逆压缩均值修正；但足以否定“首次将二阶 Tweedie/Jensen 修正用于 diffusion”的宽泛主张。此次读到正文，不再只依赖搜索摘要。CVPR URL读取失败，arXiv HTML为错误标题页，故只使用实际可读 PDF，不冒充已逐式核对会议终版。

尚未做覆盖所有 speech/非线性 decoder 工作的系统查新；不作“无人做过该具体修正”的反向结论。独立 critic 同样认为通用 delta/Tweedie 本身不建立新颖性。

## 7. 下一步与退出条件

本支线到此完成**估计对象审查**，不再为它构造通用协方差工具链。未批准新读出、sampler、Jacobian 或训练实验。

若未来提出一个具体新方法，先同时回答：①操作的随机变量/目标风险/原生路径是哪一个；②为何实际误差而非通用数学差别是瓶颈；③相对 STSL、已有预测/时域目标到底新增什么；④与简单末步/增益/匹配训练基线怎样公平比较；⑤怎样保护真实弱音节且不只降低 MSE。若只能给通用公式、旧32格参数搜索或无成本核算，不进入工程准备。

独立内容端点仍未实施，不把缺听者伪装成理论证实。后续优先研究实际条件利用/内容错误的可证伪机制；保留已有正面音质反馈、负实验和 test 冻结边界。长期 goal 仍未完成。

源码/现有保护记录的本轮 hash 与文档检查见同名 `.json`；它是静态审查收据，不是实验指标。
