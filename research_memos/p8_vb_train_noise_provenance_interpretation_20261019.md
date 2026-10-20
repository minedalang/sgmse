# 官方训练日志补证：类别已知，不等于噪声录音已知

## 新证据及其范围

I0的本机root/depth2/白名单扫描未找到日志，只能说明该范围未发现。Root随后在[官方完整文件页](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757/full)看到logfiles.zip，批准了新来源审计，而非重跑波形或开训。

`experiments/p8_vb_train_noise_log_audit_20261019/` 唯一3×HTTP200、7.986秒、正文136499B；ZIP115932B与发布者MD5 `0d16f5b6afe12d64238d1e4a0e6ff6e8`吻合。Worker只打开train28日志，CRC通过，全11572行无重复，固定12句全部join，独立离线验证通过。[官方日志容器](https://datashare.ed.ac.uk/server/api/core/bitstreams/11185dc8-9cf1-405b-b858-35bd6a04aedd/content)来自实际metadata链接。没有音频/模型/训练或测试/56spk内容解码。

## 字段语义后续补证，原metrics不回填

Worker冻结范围没有打开README，故其原metrics的nominal_SNR_dB=null保留历史。Root另作已有容器内的来源文档只读审查，精确读取log_readme.txt的115B，CRC31dd03c1、SHA `58d41e2a300445471342a1f236e0ed5f993cf0ab6c343228f17e354e20770319`；字段定义是 `wavfile noise SNR(dB)`。

因此第三列现在可记为**发布者标称SNR(dB)**，不是仅靠数值模式猜测。Root该次0网络/测试/56spk/audio读取；范围与Worker阶段分开记录在 `p8_vb_train_log_semantics_20261019.json`。项目累计打开过train28日志与README，不能无限宣称所有人始终只读train条目。

| 发布者标称全句SNR | 固定12句数量 |
|---|---:|
| 0dB | 4 |
| 5dB | 2 |
| 10dB | 3 |
| 15dB | 3 |

这些不是本地16k或80ms事件实测SNR；原48k转换门限制未因此解决。**I0/I1a/I1b只是数据管线验证，并未完成极低SNR可辨识性实验。** 用户提出的后续多noise/低SNR构造仍待做，不因本轮日志而追加混音。

## 类别、录音和分组可行性

12句noise标签为restaurant7、station2、ssn1、traffic1、cafeteria1；全train日志有10种token。但三列没有录音ID、offset、channel、segment length、gain或noise波形SHA。类别不能冒充唯一录音，异类也不自动证明噪声独立；原I0/noise_recording_id=unknown不倒改。这不能证明X−S纯净或提供D4协方差。

12句按“同speaker或同literal noise标签”连接并传递闭包，只有**10句与2句两个分量**。所以若要同时严格隔离speaker与该标签，现池不能拆成三个均非空的开发/校准/留出组；此前2/2/2 speaker草案不能直接执行为三方联合隔离。

这只是该关系下的可行性，不是已查明真实recording关联，也不是整个VB或未来设计都无法划分。不拆连通分量、不换当前ID找有利split。人耳方案不需训练分割，但也不能因增加听者就声称噪声泛化。

## 当前决定

旧12句/I1b窗口不换，原指标与未知字段不倒改；用新补充记录体现“限定本机未找到→官方有日志→类别/标称SNR已知但录音身份未知”。不自动开D0–D4、拟合噪声或建独立split；正式模型验证仍需来源关联、独立校准与匹配设计。

人耳分配不按此结果挑SNR/类别，资源仍未确认，未招募/导出。技术完整性、自然结构评价、极低SNR内容机制是不同证据，不互相替代。

原审计metrics SHA `ca1ffd3055f8c98e6fd77e3d9c351d7e55c52a743a086324c4fcc5570a8849da`；全部source/独立验证见上述目录。无新声学性能、训练或goal完成声明。
