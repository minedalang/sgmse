# raw谱anchor与public波形再分析接口审计

## 为什么做，明确不做什么

定向静态审查与既有VB32 hooks已经否决“前后端使用不同peak导致重复归一化”的假设；不做所谓修复归一化试验。另一处未量化的实际差别为：当前训练及CPEL正常评价用前端原始预测谱A，public入口先将A合成波形，再STFT/压缩得到anchor。任意预测谱不保证合成再分析不变，逆压缩又含eps差异，因此两者不能直接假定等价。

本次只判断这个接口差别的数值与输出影响，不宣称吞音因果、内容恢复或新方法创新；当前源码不能证明原300k历史训练采用何种接口。旧VB32只记录波形及norm/NFE，没有比较两谱，不重跑旧VB或5243池。M/P及N/D结果原样保留，明确其raw谱评价范围，不用该差别解释所有负结果。

## 固定8例与唯一双路径

从既有64dev依四个SNR metadata bin各取2条，按sha256(native-anchor-roundtrip-v1|utt)排序，保留原dev_index/seed；在看任何新输出前已登记。每例仅读取S/X两个PCM，固定center32640/同原Noisy peak，source checkpoint strict/EMA加载1次，F及Score全部冻结。前端预测8次，每例A仅生成1次。

raw分支按原CPEL接口以A和Noisy谱进入原生N30。public分支把frontend.spec_to_waveform(A,L)恢复物理标度并传model.enhance，前端wave及原Noisy形状必须[1,L]，不误传[1,1,L]。临时hook观察实际sampler接到的anchor/Noisy谱及ScoreModel.enhance实际norm，原实现执行不改、finally恢复hook。两branch分别恢复相同CPU/CUDA seed=20261019+100000+原devindex，实际各30Score。

模型与frontend的FFT/hop/window tensor/压缩参数及长度必须runtime匹配，不能凭默认值；expectedRoundTrip沿真实frontend逆变换和model STFT/pad，不手写逆谱。物理乘r/除r的浮点误差保留；实际public与expected的相对L2按预定1e-5数值范围记录，超界不改参数重试。原Noisy条件差、raw/publicanchor差、两归一输出差及各自对clean MSE全部记录。旧initial64对应8例rawMSE仅按1e-8+1e-4*old值的容差作sanity，不因不符换样本或重跑。谱差为0或输出差小同样是有效结果，不预设必须发现问题。

## 成本和解释

最多1source模型加载、16PCM、8前端、16backend×30=480Score；180秒、新产物2MiB、留1GiB，0训练/梯度/网络/安装/测试集/权重复制/音频或谱数组输出。仅标量、shape/dtype/hash、原始来源与调用计数。异常保留partial且终止，无重试/窗口/seed搜索；这不是新质量指标或ASR实验。

若实际public等于预期再分析且两谱差/输出差仅数值级，排除该边界的实质影响。若明显不同，仅确认raw-CPEL与public部署接口不等价；不能自动说哪个更好或将它包装为创新，需另审是否存在有价值的新研究问题。完整实验资料须config/metrics/report，当前仅软件准备不授权真实执行。
