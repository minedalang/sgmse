# 固定Clean词读出适配性：一次取得与校验，不是增强实验

## 问题和来源

mini Speech Commands在元数据层提供8个目录词标签和可追溯分组。前次来源运行请求源码超时且派生产物超4MiB，保留失败，不重跑或改判通过；本次独立动作只判断既有tiny.en无prompt读出是否足以支撑后续词身份转移诊断。

冻结selected64_manifest：官方which_set条件规则validation完整8词组有9组，按sha256(p8-word-calibration-v1|prefix)排序前8组，每组8词各首文件；64全部保留。第9组不用，testing完整8组保留而不提取/解码。分组prefix不是已验证speaker，public资料相对预训练暴露未知。完整官方ZIP下载/整体字节hash必然接触其所有压缩字节；“不读reserved音频”指不抽取、解析或解码reserved成员，不冒称未接触压缩数据。

## 一次运行

Root许可绑定config、脚本、64manifest、receipt、来源CD/entries及本机ASR权重/package源文件hash。未有许可不取音频/权重。一次官方HTTPS GET精确182082353B，必须Content-Length与ETag符合已查元数据，无redirect/retry/mirror；只保存此archive，核ZIP所选64记录与来源CD字段一致，CRC核后仅64原WAV保存。禁止extractall/Mac侧车或任何替换样本。全包取得不能宣称所有8000录音已核内容。

只对64个原始WAV核mono PCM16、16kHz、1–16000样本；异常terminal、不重采样/裁剪/外部padding。用本机固定tiny.en，English、temperature0、beam1/best_of1、no prompt、无前文、no_speech_threshold=None、fp16=False；不将标签或八词字典喂给模型，不约束候选或做同音字/拼写替换。保存raw text与lowercase/标点分词后的token。每条正确须恰有一个token且等于目录label；多词/空/集外输出单列，不能当作已证实的真实内容错误。

## 端点和退出

唯一门：全量至少58/64正确，且每个词至少6/8正确。门在任何本数据ASR结果前固定，只测Clean读出适配性，不说明Noisy/增强域的可靠性，更不是人耳正确率。任何技术失败如实终止；读出达不到则不采用该读出/端点，不调prompt、换ASR或训练KWS找通过。

当前不加噪、不跑F/后端、不测试CPEL M/P权重，不新增loss/网络。只有未来另批，才考虑在固定评估器下读Noisy/F/nativeOFF的修复、损伤、共同失败转移，而非只报总体准确率。CPEL v2继续关闭。

## 成本及保存

总300秒、下载120秒、1HTTP/182082353响应body字节；新常规产物≤200MiB并始终留1GiB；64transcribe上限，内部decode实数另报（不假装1API=1网络forward），ASR只加载1次。0StoRM/frontend/backend/optimizer，0模型复制/安装/后续网络。执行前预计源ZIP+所选音频+日志尺寸，写入前检查资源，不能重演元数据产物超cap。保存config/metrics/report、命令/终态PID或session、完整64行与技术/识别失败状态。旧源目录、CPEL与论文原指标均不改。本计划不是运行成功或完整goal证明。
