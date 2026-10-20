# CPEL N/D保存响应解释：不只是正幅度收缩，但仍是小幅代理作用

## 本次只回答什么

用户指出CPEL验证不应自行转入Router研究，已撤回该旁支作为当前安排。本次只使用已保存N/D终态JSON，回答：orth项下降能否全部解释为响应总幅度的正比例缩小？不新增模型、音频、权重、训练或评价端点，不改变原10%门，不开启新配置。

## 可识别的数学量

实际`loss_helpers.py:31–38`定义单位目标向量v、加权响应w=sigma*P*(q1-q0)，b=Re<w,v>，O=orth=||w-bv||²/d。`run_pilot.py`保存的E=response_energy=(sigma²/alpha²)*||w||²/d。记共同正数k=sigma²/alpha²，则

`Z = O/E = (1-cos(theta)^2)/k`。

复数空间按实内积解释；soft P不妨碍w相对单位v的正交分解，但inside/outside不能因此称互补正交分区。匹配N/D的同一utt/packet/parent固定t=.5、几何和目标，k相同，无需从权重或SDE重建其数值。

任何正数c下，若完整响应满足Deltaq_D=c*Deltaq_N，则O_D=c²*O_N、E_D=c²*E_N，因而Z_D=Z_N。此必要条件允许每个tuple有不同c，并不限于一个全局缩放。**反过来，Z相同不能证明纯缩放**，因为正交分量方向可能变化。

Z下降只在两臂gain均正时表示对该P加权clean差分实轴的相对方向对齐增加；不等于完整响应更接近clean差分，不等于gain增大、更多恢复或语义正确。负gain必须另列。

## 固定分析与全部覆盖

主要为既定heldout raw Tukey，precision为同案例敏感性核对，不作为独立复验，不新增Hann表。按utt/parent匹配64×2条，crop、norm、center、gaussian seed、valid均一致。各parent先算Z，再file内平均两parent，最后64file等权。全部E>0且zero_response=False；没有缺失/补零或Z比值分母为0。

| 保存标量统计 | raw Tukey | PCM16匹配Tukey |
|---|---:|---:|
| N的file等权平均Z | 5.594512379 | 5.644776960 |
| D的file等权平均Z | 5.516737266 | 5.567766767 |
| 两均值之比减1 | −1.390204% | −1.364273% |
| parent相对Z变化中位 | −0.748026% | −0.727390% |
| 两臂gain均正 | 127/128 | 127/128 |
| 其中Z下降／上升 | 122／5 | 122／5 |
| 两臂gain均负／符号翻转 | 1／0 | 1／0 |

符号统计另用相对1e−5的浮点记录guard核对，122/5不变，无落在guard内的正gain项；这不是新的科学成功门、显著性检验或严格浮点误差界。最小正gain项相对Z变化绝对值raw为6.68e−5、precision为4.33e−4。128parent共享64file，不当作128独立样本，计数仅作描述。

负gain项为reader_11273_fileid_1479/parent0：raw gain −.04731772→−.04222943，Z反而增加约.1581%；precision同方向。该项仍为反向响应，没有被移除或写成正确恢复，也不能套用“Z下降才好”的正gain解释。

## 可靠解释与仍未证明的部分

本批保存统计**不支持把全部变化说成仅正比例幅度收缩**：多数正gain响应的相对orth比例有有限下降。但5项方向相反，平均gain仍略降；不称所有案例一致改善，更不称恢复幅度增强。1.39%是Z均值的相对变化，不是余弦角、语义准确率或音素恢复率提高1.39%。

没有完整响应张量，不能识别具体频时形状、完整N/D向量距离或纯收缩解释的残差大小；此事后标量检查也不识别训练因果细节或总体显著性。**有限的相对方向改善与未达到推进门可以同时成立**。

旧raw/precision weighted pair仍仅改善3.48424%/3.47731%，未达10%；public D−N SI-SDR仅+.00459dB、STOI近零、PESQ+.000343，D相对source PESQ仍低.00331。新Z表不替换旧门、不择端点救场、不追加训练。准确结论是“此固定配方有小幅连续响应代理作用，但尚无实质综合部署或内容收益”，不是“完全没学到方向”，也不是“CPEL普遍无效”。

## 可复算资料

- [分析脚本](p8_cpel_shape_ratio_analysis_20261020.py)：只导入Python标准库，固定两源SHA，不导入项目/torch。
- [完整256条raw/precision记录与汇总](p8_cpel_shape_ratio_analysis_20261020.json)：99,160字节，SHA256 `3ec4489fce1f55d110cf6c6a02c6b6c20392500bdb0f77bf4f7d97ad378897da`。
- 源N SHA `eae9d47fcf6e5387428918a18d020a6b104a7dbbec97a30f0521fc783689b598`；源D SHA `1174f563fb610acfd27055ebec27d5135c89bf17065b4f6598a28511bba93fc1`，读取前后相同。

命令：`python3 -B research_memos/p8_cpel_shape_ratio_analysis_20261020.py`。这里只是既有结果解释性核查，没有新的模型实验、指标推理或bootstrap。

## 独立复核

独立Reviewer从原N/D JSON用标准库重新计算，而非执行本脚本，256条派生行逐项最大差0，匹配覆盖、raw/precision均值、符号及原weighted pair数值均一致。首次复算命令因Python语法不兼容退出、未写文件或产生科学调用，修正该语法后完成；未改分析规则。Reviewer确认soft P下的实内积恒等式和有限解释范围，并明确Z变化不是3.48%收益的因果分解。
