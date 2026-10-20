# 既有多seed缓存的有限集合误差—离散度分解

Root批准一次**仅缓存CPU分析**。新问题：现存四seed输出的平均参考MSE，有多少可由seed间离散度这个精确代数项解释？不重新生成、不挑seed、不运行mean waveform的新指标。目的只是评估“生成方差可能贡献参考误差”的竞争解释，不能代替当前VB32 Router-off上的直接实验。

## 来源与已知混杂

历史 `experiments/multitrajectory_oracle_heldout16/` config/report/metrics：16个DNS CV探索文件、8speaker、seed2026–2029、PCN30、80k EMA **Router启用**。生成路径代码 `experiments/multitrajectory_oracle_20261017/multitrajectory_oracle.py`用了build_rrsd_score_calibrator与手工prepare_specs，**不是**最近VB32的原生Router-off入口。旧名heldout只指旧选择器开发半集外，不是预训练/最终测试独立。

冻结metadata来源：旧metrics SHA `89bbe3200c9d658115086d6f7400abb9cde696710dbae623a50b64ae6dc20a89`；`experiments/rrsd_cv_repeat32/manifest.csv` SHA `4349228558395bcdabf9a1e4ab4b23ac6256834169c4338dfd4daca03746558e`，取原rows16:32（0-based）。按metrics的per_trajectory一一匹配16filename×4seed，输出路径均在`/dev/shm/multitraj_router_wavs_heldout16/`。Root只读确认当前64条列出的WAV及16个clean路径均存在。

**历史hash缺口必须明示**：原per_trajectory没有WAV SHA；本轮只能冻结当前字节并检查前后不变，不能证明它们与原生成时逐字节相同。config/源码/路径和已有数值提供来源线索，不补造历史hash/不可变证书。若成员缺失/不唯一/头或长度不合，技术失败停止；不另寻目录/更换文件/恢复或重跑模型。

先保存旧config/report/metrics、manifest和生成代码当前hash。仅解码上述64cache PCM与16clean（80唯一WAV），标准库wave/struct，要求16k mono PCM16/NONE、相同完整长度。旧生成时有clip和量化，本审计以现存PCM为对象，不反推preclip float方差。不加载noisy、frontend、checkpoint或模型。所有16/4种seed保留，不按quality选臂或排序。

## 冻结计算（每文件，样本均方，未中心化）

S为clean，B_k为4份cache，K=4：

```text
M[t] = (1/K) sum_k B_k[t]
P = mean_t(S[t]^2) > 0
E_average = (1/K) sum_k mean_t((B_k-S)^2)
E_mean = mean_t((M-S)^2)
V_population4 = (1/K) sum_k mean_t((B_k-M)^2)
E_average = E_mean + V_population4
```

保留4个独立端点MSE、E_average、E_mean、V_population4、identity residual、V_population4/E_average（分母0则null），及除P归一量。M只存在临时内存，不写音频或样本数组，不作STOI/PESQ/SI-SDR/MOS/ASR。不能把恒等式造成的E_mean较低当学习到的新算法或内容改善。

如果附加“seed可视为同一条件生成器的独立同分布抽样”假设：V_unbiased=(K/(K−1))*V_population4；finite4_bias2_est=E_mean−V_unbiased/K。保留raw bias2_est，即使负值也不截0。它是有限seed/固定参考下的估计，不是真正posterior bias或语音身份不确定度。不能把seed间离散度都叫幻觉，也不能将E_mean称真实conditional mean。

汇总all与8speaker：**equal-file clean-power normalized MSE**均值、均值恒等式/平均ratio及pooled normalized variance fraction（明确两种ratio不同），所有文件/seed表、负bias2_est个数。无新bootstrap/阈值/频段窗口/lag/听感筛选。4seed只是本集合，不能从这一个比例推整模型总体方差。

## 成本、保护与执行

标准库逐文件处理；主120s/独立验证120s、新目录2MiB，禁止模型/ASR/quality库、网络/训练/重采样/新波形/mean audio/任意清理。独占STARTED；已有marker/results不重做，失败记录停止无retry。

pre/post冻结当前80源WAV字节和头；旧metadata/report/code不改，最近VB32 metrics SHA577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04及residual audit metrics SHA06603c8fef1a7ee7b5a7bed89b159f6f97036e06e7d04c241524190892fe3e17保持；git目录外/bridge/试听保留。仅写本新目录。记录实际唯一WAV与解码计数，独立验证重读的解码另计，不写成0真实音频处理。

交config.yaml（Root冻结）、metrics.json、experiment_report.md、原始来源清单/hash、实施代码、独立CPU重建/溯源。独立实现可用等价pairwise恒等式：V_population4=(1/K²)sum_{k<l}mean((Bk-Bl)²)核验，并直接重算M-S；不import业务函数自证。scalar正确不证明cache历史完整性。

**判定/退出**：此缓存若离散度很小，削弱“该历史Router池的误差主要是seed变化”的解释；若很大，也不证明当前OFF或true posterior variance，更不授权平均/多seed训练或选择。若来源不可核，保持未知而非复跑。无论哪种结果都不向关闭VB32加seed或复活旧多轨迹selector；后续方法仍需可信内容/质量与匹配计算支持。
