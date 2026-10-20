# N/D：普通反事实DSM之外pair增量的缺失对照

## 一个不同的问题，不重跑或重判旧M/P

原始用户B仅普通反事实DSM，而已完成M额外含共同state逐点clean回归，精确分解已含强差分/gain1/orth监督。旧M/P的1.67%/1.66%未达10%门结论原样保留。本次只回答：**没有强共同点回归时，原pair是否仍在普通DSM增强之外有增量？** 不重新宣称这是未检验的一般新颖思想，也不以此覆盖旧负结果。

N：两变体各自合法OUVE native DSM。D：同native DSM，增加原来phase-aware有限方向pair。共同u/A/t只用于pair，不import/call common_point，不写native+0*aux。N/D均算4次Score/两次参数梯度，N施加native梯度，D施加native+eta*pair梯度。相同Score主干、冻结F与原生Fourier W；旧helper保留原定义不代表使用该目标。

## 固定不变与公平性

沿用已注册256train/64dev、同checkpoint物化EMA、源参数集合、样本次序/crop/归一化/事件/RNG/time、80ms Hann编辑、heldout Tukey与PCM16匹配、noise实例控制，500步、Adam1e-5/clip1/EMA.999及原pair margin .1/1.5与分量1/1/.25/.2不改。aux0两臂同数据同计算，旧M/P只作aux1历史参照，不重新训练或选择终末权重。

D仍用eta=min(100,.01*||g_native||/||g_pair||)，N同算但应用0，非有限/图支持不符终止。None/None保留None，nativeNone而pair非None不补零。预算分母改为native-only是本问题的定义，跨旧M/P增量差不能全部归因数学冗余；相同forward/两梯度也不意味着实际反传成本与旧M/P相等。

该pair仍是相位感知响应形状+有限幅度包络，而非纯身份或纯符号约束。共用parent state仍含特权信息，不证明部署可辨性；共同幻觉相消、noise直通、编辑捷径等限制继续存在。当前高频事件多数原本同向，不能把试验写成首次激活Noisy。

## 现有基线显式复用

两臂只复用旧source initial64，不重新跑基线、也不复用旧终末M/P权重。baseline_provenance绑定旧baseline完整SHA/identity、旧config/四code、相同checkpoint/dev/seed和三核心源码的历史pre保护；新旧初始化及完整eval闭包AST相同。sdes/sampler等附加当前锁不是缺失历史hash的替代，来源限制显式保留。旧identity原样，不伪装新config生成。

N完整500+64final四packet完成且冻结状态不变、终末artifact SHA验证后才D；不因N的科学分数好坏跳过D。同一64既有CV再次使用仅作探索，不是独立确认；开发SNR分组只全量描述不择组。正式DNS测试集不读取。

## 预先固定的筛查与解释

64 file内先平均2parent，再file等权；三speech packet不合并冒充更多独立样本。weighted pair为lower+upper+.25orth+.2outside。主要门为raw Tukey和precision matched同时D/N<=.9，N均值>1e-12、覆盖匹配且无zero-response退化；gain/反向/overgain/四分量全部报告。它仅是预定工程信号，不是声学正确性或显著性门。

正常未编辑nativePC N30指标D−N不得低于PESQ−.03/STOI−.005/SI-SDR−.2dB；同file/crop/seed/有效分母。noise实例匹配覆盖的signed input gain增加不得超过.02，同时报告输出变化能量、strict及codec子组和全部未评原因。编码相容不证明纯noise或FPR；27能量不足若再次出现不补零。precision与raw为同案例反证，不是独立复验。

若未达到增量门，关闭这一no-aux配置，不扫权重/窗/步数/seed；若只合成代理变好、native不迁移，不能进入真实内容成功主张。若达到且正常/噪声损伤门未越界，只支持进一步检查适用机制/真正迁移，仍不自动大训练。A原native继续训练→B普通增强收益未由此试验分离，phone/身份/错误新增/真实人耳与质量效率完整目标仍待证，试听列待办而不阻塞这项客观机制比较。

## 成本与执行边界

各500步、每臂900秒、总1800秒上限，产物320MiB且始终留1GiB；只存两个终末backend EMA(预计合计222375234B)，不存raw/optimizer或音频。0网络/安装/ASR/DNSMOS/冻结测试集/旧源写入。失败保留partial、无自动resume/retry或降级。初始化/图支持已有静态及tiny纯tensor证据，但真实CUDA尚未验证，不额外开预检链，首次真实异常就完整记录并停。

Root审最终SHA签单次N/D许可后才由Worker运行并监控，同handle续看不因观测超时重启。全部原始数据、旧实验、用户反馈、服务和兼容桥保持不变。文稿与上一轮宽泛关闭表述后续注明本次纠正，不倒改冻结实验结论。
