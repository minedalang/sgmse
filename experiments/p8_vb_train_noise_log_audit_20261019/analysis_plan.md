# VoiceBank官方训练混音日志：有界来源审计

## 为什么是新任务，而非重跑

I0仅在本机固定root/depth2/文本白名单内发现0日志候选，身份仍unknown；那不等于官方无日志。Root实际浏览发布者当前item/full页发现logfiles.zip113.21KB，MD5明确。此阶段仅填真实来源元数据缺口，不重做波形审计、调参、模型/检测器或人耳实验。

## 批准范围

Worker先全文读协议/entry/该设计/config，核provider/桥/进程/git/resources和来源锁。仅本新目录可写。网络总3次GET/90秒/正文512KiB/总产物4MiB，0重定向/自动重试；超过预期schema/URL/分页或网络限制就记录失败停止，不换镜像/反复请求。归档所有实际请求URL/status/headers/正文SHA/字节；同一stage不允许重跑已有STARTED。

1. 官方Dspace item bundles metadata，选唯一ORIGINAL；2. 仅跟其字面bitstreams href(允许size=20)，按name精确选logfiles.zip；3. 仅跟其字面content链接，全部URL仍在官方HTTPS origin。不得猜测其他下载链接、触碰音频大包。metadata标size/MD5与发布者页预期核验；archive大小≤256KiB且MD5必须一致。所有HTTP请求实际登记，不把urllib自动重定向或重试漏计。

archive共享container允许取回并保留，但仅读取ZIP目录(路径/size/CRC)；只解压basename恰为log_trainset_28spk.txt的一条安全entry并核CRC，≤2MiB，拒绝多匹配/zip路径逃逸/符号链接/加密/异常压缩。绝不extractall/testzip，test/56spk条目不打开、不解码、不计算内容CRC。它们的compressed字节位于共享container不等于读取其测试行；报告清楚这一范围。不获取test音频/文本/指标，不训练/选择阈值。

## 离线解析与结论范围

保留原训练log全文bytes/hash/编码与行格式，先检查schema，再解析全部publisher train28行的ID/字面noise token/nominalSNR(若存在)。不得预猜列义后强行套schema；若没有足够语义依据，只输出raw列与未确定项。允许记录总行数/duplicates/12目标ID覆盖，不输出其他目标的结果来改选12句；原12不换。ID应精确joinI0登记，nominalSNR不是wave实测或局部SNR。

noise token若仅类别/场景名，noise_recording_id仍unknown，不能倒写旧registry或宣称真实noise-disjoint。可在新JSON保存12句按speaker/同字面token连接的连通分量，作为相关性风险，而非已证明的独立split。offset/channel/gain/segmentlength/sourceSHA缺失逐项记录，不能因日志存在就把D4协方差或残差纯净gate改true。

## 独立验证与交付

成功后一次标准库离线验证：官方MD5/新SHA、只读目标entry CRC与raw字节、独立解析12join与nominal数值、独立graph分量、源锁pre/post核对。独立validator不得网络或音频。实际动作与counts保真，失败保留不修schema重跑HTTP。config.yaml/metrics.json/experiment_report.md必备，源文件/有效checkpoint/服务/脏树不改、不push；Root可能并发更新研究文档，按观测范围披露，不伪称全filesystem相同。

## 未授权

噪声波形下载/解码、源音频、信号重建/校准、I1b编辑位置/强度重选、独立听者招募/页面、D0–D4性能评测、任何模型/训练。若官方请求失败或训练entry缺失，报告具体失败；不得把失败当整个goal工具缺失或伪造噪声身份。
