# 原始预测谱与public部署接口：部分完成揭示不等价，保持失败记录

## 执行事实

固定8例的唯一audit于22.1499秒、第7例触及事先接口容差门后终止：实际1source模型、14PCM、7前端、14backend、420Score，0训练/梯度/网络/ASR/新质量指标或音频导出。第8例没有读取或评价，不能作8/8完成。完整7个双分支记录保留，不删第7、不放宽门补跑。

失败的是第7例expectedRoundTrip与actualPublic anchor相对L2=1.0771018688e−5，略超过固定1e−5；绝对complex MSE=2.5655839897e−13。此前6例该接口检查通过，7例原Noisy条件误差均在固定范围内，norm全部相同，7例raw输出MSE全部与旧source baseline容差sanity匹配。小门失败与浮点乘r/除r及压缩稳定化误差相容，但本次没有独立证明具体数值来源，仍记技术失败，不倒改协议。

## 两种差量不能混为一谈

- **大差：raw预测谱A与public实际sampler接到的anchor。** 7例相对L2为10.11%–24.52%。这不是1e−5量级的实现容差差。
- **小差：用真实frontend逆变换和modelSTFT计算的expectedRoundTrip，与public实际sampler anchor。** 前6在容差内，第7略超界。该检查限定是否确实沿预期public路径，并不要求raw谱等于再分析谱。
- 两分支归一输出相对L2差为6.89%–109.02%（分母为各例raw输出能量，不是clean能量）；低能量参照可能放大比值，不能读成相同百分比内容损伤。

raw与public都用同source EMA/前端、原始crop/Noisy peak和同seed、每branch30Score；因此这些保存数据足以否定“raw-CPEL接口输出天然等于public部署输出”的假定，至少在已测7例上。除了anchor合成再分析，两branch Noisy谱也存在约1e−7至4e−6相对L2的浮点差；不能把输出差精确全部因果归到某个投影算子，更不能因此解释吞音或定位音素身份。

## 不是归一化bug，也不是方法收益

7例public实际norm全部等于同一原Noisy peak，先前静态审计及旧VB32也已证实乘r/除r抵消；没有anchor peak独立缩放证据，不进行所谓修复重复归一化实验。

public相对raw的归一clean MSE在5例降低、2例升高。全部7例均值raw约.00030999776、public约.00026199891；这是固定顺序partial的描述，不是预定完整8例或总体性能估计，更不是PESQ/STOI/内容收益。不能以此选择更优接口、翻转旧M/P或N/D门，或包装成新的STFT一致性创新。

## 对现有结果解释的必要纠正

CPEL M/P及N/D的训练与正常评价使用raw谱anchor，二者内部匹配；本audit不否定其配对差的算术。应把其中“正常native”准确写作**未编辑输入、原生OUVE/PC采样器下的raw谱接口评价**，而不是已经证明与public waveform入口等价的部署结果。

当前源码能证明当前代码接口差别，不能反推原300k训练时的代码路径。旧正式public评价和VB32仍保持原接口、原输出与原指标；不由新差别自动修改历史记录。

**本次到此按失败收口，不重试、放宽容差或补第8。** 保留已证实的接口作用域信息，后续研究可据此分清训练条件与部署流程，但差非零本身不授权新训练、不保证内容/质量提升。真实内容确认仍为对应强主张待办，不能借此再次全局停止其它研究。

[完整原始结果与执行报告](../experiments/p8_native_anchor_roundtrip_audit_20261020/experiment_report.md)。本解释只用保存JSON/源码与标准库算术，没有新增模型/音频调用。

原metrics SHA256：`c925182dd71d157469cf90c623d0331502412fd6b2d489e6abd5b07ea080c249`（失败partial原件保留）。
