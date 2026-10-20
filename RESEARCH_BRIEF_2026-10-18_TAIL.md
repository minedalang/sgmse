# 夜间研究简报：补对辅音与额外尾音

## 跨午夜追加：VCV对照与真实noisy通道（2026-10-19）

**最新复验决定**：新四位说话人已经跑完，原四人中Router的标签增益没有复现。新N1−6dB，Y_pre/off/Router为14/15/14（24类，分母16）；N1+6为16/16/15。N2两条件Router与off没有变化。Router分别新增一个g→dj和t→d标签错误。因此停止将当前现象升级为“Router稳定提升辅音身份”，不以换speaker/seed或继续扩训寻找正例。完整结果见[新说话人复验报告](experiments/p8_q_vcv_speaker_replication_20261019/experiment_report.md)。

研究仍有用的证据是：noisy直接卷积条件既可能帮助纠错，也能推动额外结构；需要针对二者做控制变量的机制实验。已发现跨音素raw条件互换会改变峰值归一化，不能声称只改变音素身份；后续设计须固定归一化锚点/噪声尺度，或者用不改变全局峰值的局部分量干预。原5243/10720人耳观察、两语音尾音诊断和所有负结果继续保留。

- checkpoint确认没有启用条件cross-attention、FiLM或selector；noisy的实/虚部直接与扩散状态、前端锚点拼成6通道，并进入progressive input skip。之前尾段Oracle干预针对的是这条普通卷积证据路径，不能归因未启用模块。
- 16条已冻结VCV、两噪声、四档完整2秒carrier SNR的输入诊断共144次前向；只有两噪声的+6dB同时满足全部可辨输入预设门。
- +6dB前后端正控制：N1的Y_pre/off为15/16，Router16/16；N2三臂都是16/16。off没有改变前端身份正确性，Router修正一个N1代理标签错误。
- 独立的−6dB已知标签挑战：N1的noisy/Y_pre/off/Router为8/12/13/15（24类、分母16）；N2为6/12/12/12。Router在N1修正off两处错误、未新增，N2无同类收益；off同时有修正和损伤，不能只看净均值。
- **SNR口径**：−6dB指含静音padding的完整2秒；VCV录音支撑区间约0.34–0.51秒，实际支撑区间SNR平均+0.867dB。它不是−6dB单辅音局部SNR，不能当极低SNR语音恢复实证。
- 已预先选定未在之前28条VCV中加载的四位说话人f18/m5/f20/m15进行复验，保持aCa重音1、两噪声、±6dB设计不变，不按识别结果换样本；最新完成结果与退出决定见本节开头。

这些是有外部标称身份的受控诊断，不是新的独立算法创新或无偏增强榜单。真实弱辅音保护与静音伪影两条证据均保留，暂不训练新loss/controller。

## 新增的人耳证据

- 5243：用户定点确认Y_pre缺d，而后端有d；这是后端只圆润、不补音的具体例外。
- 10720原始极低SNR：用户确认off/Router与clean/Y_pre/noisy听起来内容不同。
- 同一句10720降低配对噪声10/20dB后，用户听到整体内容可对应；但20dB组后端末尾新增s样声，clean/Y_pre没有，10dB组该段更像人声。整体匹配不等于局部无伪影。

用户已离线，今晚不等待额外试听。新干预输出未由用户判断，不能写成人工确认伪影已消失。

## 配对问题已单独查明

历史旧包noisy确有9秒裁剪bug；两个播放器分别存在多音频叠播和过期异步播放竞争，已修复留证。当前10720源/采样裁剪/HTTP文件一致。相同输入、权重、有效逐文件seed分别重放off/Router，都与原WAV逐字节一致；源文件和checkpoint未改。用户修复后仍报告内容差异，故该剩余差异不能继续归咎于文件配错。

## 有建设性的机制结果

固定完整float前端、采样随机数、归一化峰值，仅在clean近静音间隙替换局部noisy条件（离线clean Oracle）：

| 例子 | 后端 | 间隙RMS变化 | 4–7.5kHz能量变化 |
|---|---|---:|---:|
| 10720，噪声降低10dB | off | −25.657dB | −44.902dB |
| 10720，噪声降低20dB | off | −29.372dB | −37.573dB |
| 3321，原始输入 | off | −29.751dB | −20.908dB |

10720只替换前端锚点的下降较小（RMS−8.566dB）；改变seed也会改变尾能量，但两条额外seed不足以估计概率。

这些结果支持额外输出对noisy条件路径敏感；不证明逐音素的噪声语音化，不是可部署抑制方法，更不是新颖性证明。3321真实语音区域与clean距离略升，不能靠压低局部能量宣称增强质量改善。

## 跨例筛查的范围

已有8条开发语音共39个固定规则clean静音间隙。10720之外7条有31处；按预设能量排序，只有3321一处off突出，Router没有。10720较突出、问题较稀疏。自动flags不是人类幻觉标签，其他7条也不是全新盲测集。不能推广为后端在所有静音处都会生成语音。

## 下一科学步骤

已有外部VCV冻结主集16条clean在24类和二选一均16/16，额外12条只7/12一致；不按结果换样。下一步在原冻结配对中固定噪声，检查真实k/g或t/d改变时，noisy和前后端是否跟随身份；同时比较同一音素跨噪声是否稳定。

目标是区分noisy提供的有效弱辅音证据与导致额外结构的线索。保留5243补对d的正例，防止抑制机制把真实辅音也删掉。当前不训练gate、controller或内容loss，不用官方冻结测试集调阈值。ArtiFree/PASE已覆盖生成伪影、采样一致性和音系先验，不能把发现尾音或加SSL损失本身当新创新。

## 证据索引

- [10720来源与修复](experiments/p8_10720_pairing_forensics_20261018/experiment_report.md)
- [同句SNR对照及用户反馈](experiments/p8_10720_snr_control_20261018/experiment_report.md)
- [尾段声学诊断](experiments/p8_10720_tail_mechanism_20261018/experiment_report.md)
- [10dB条件/锚点/seed诊断](experiments/p8_10720_tail_causal_pilot_20261018/experiment_report.md)
- [20dB条件复验](experiments/p8_10720_tail_condition20_20261018/experiment_report.md)
- [8条静音间隙筛查](experiments/p8_silent_gap_screen8_20261018/experiment_report.md)
- [3321成功attempt02](experiments/p8_3321_tail_condition_20261018/attempt02/experiment_report.md)
- [VCV clean校准](experiments/p8_q_vcv_clean_calibration_20261018/experiment_report.md)

所有原始数据和有效checkpoint未修改，无push。长期goal保持active；不存在等待用户评分的任务门槛。
