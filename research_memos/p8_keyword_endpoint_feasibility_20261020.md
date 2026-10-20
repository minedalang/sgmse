# CPEL收口后的有界问题：词身份转移，而非继续调差分loss

## 为什么改变下一问题

CPEL v2固定小试已经关闭：P相对M的增量约1.67%/1.66%，不能通过更换评价语料挽救该配置。本轮仅问一个新的基线诊断问题：**在具有词标签的新材料上，原生后端相对前端是在修复词身份，还是增加词身份错误？** 不先提出新的loss或模块。

自然音素身份、实际删除和人工边界是强音素主张的必要证据，但不应前置为所有早期诊断的统一门。真实词标签可提供有限内容端点，避免继续把高频结构、正交谱误差或ASR置信当辅音恢复。独立Research Lead复审认为值得一次有界来源决策，不等于该词集已证明适合。

## 官方依据与限制

[TensorFlow官方教程](https://www.tensorflow.org/tutorials/audio/simple_audio)展示mini Speech Commands：八个词文件夹、8000个短录音、16kHz/不超过1秒，并给出官方存储URL及约182MB下载示例；文档说明Speech Commands以CC BY发布。这里转述官方声明，不是本机已核文件数、录音内容或完整许可归档。教程开头的“ten words”与随后明确八文件夹的实例不一致，本方案不照抄前者。

教程示例按文件作随机划分，不能据此声称speaker隔离。[官方分组源码which_set](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/speech_commands/input_data.py)忽略文件名中_nohash_及其后缀，以稳定SHA1分组。若mini保留该命名，可按prefix检查组交叉；prefix仍只是发布者分组键，不是本轮已验证的真人身份。原[Speech Commands论文](https://arxiv.org/abs/1804.03209)提供任务来源，本文不把数据集引文当成标签绝对正确或模型预训练未见的证明。

本次批准的来源动作仅为官方ZIP中心目录元数据：最多3个HTTP请求、3MiB响应body、60秒、4MiB产物；只206匹配Range，不支持即停止，不回退完整下载/镜像。统计词文件数、命名分组与固定10%/10%组划分覆盖，0WAV抽取/解码、0模型/ASR/训练。结果目录为 `experiments/p8_keyword_source_feasibility_20261020/`。这只是是否采用的决策，不扩大成通用parser或下载平台。

## 若来源可行，才另定一个固定读出检查

优先复用本机已存在的tiny.en及既定无prompt、无历史文本、固定解码参数。目录词标签不得作为prompt、候选限制或输出纠错字典；只作事后规范化的精确词匹配。先固定样本/全部保留，再核Clean适配性及预设Noisy条件是否落入评估器失效区，不按识别输赢换样本。Clean正确也不保证增强域可靠，必须保留这一限制。

这里**没有批准音频获取、ASR或增强调用**，更没有新分类器训练。若现有读出不适合，不训练一整套识别平台补洞；该小词汇诊断可直接不采用。读取已有wrapper确认tiny.en路径只是静态可行性证据，不是本数据准确率。

## 真正能改变决策的后续端点（尚未执行）

如以后通过固定读出检查，才比较同输入的Clean/Noisy/F/native backend，主表是逐样本转移而非单一总体准确率：

- F错→backend对：固定评估器下的修复候选；
- F对→backend错：损伤候选；
- Noisy对、F错、backend仍错：未被当前系统利用的内容候选，但不是单观测可恢复性的数学证明；
- 各臂都错或Clean错误：评估器/难度限制，不能说物理无信息。

保留净收益、修复和损伤两方向以及未判/多词输出，不用总体准确率相互抵消。不把CPEL M/P终末权重拿到此词集择优；首次问题只涉及既有原生基座，不包含重开Router、旧CCC输入分类或旧VCV搜索。

孤立短词、闭集词汇、padding、合成加噪与自然句子吞弱辅音存在域差。单识别器可能依赖元音、背景或生成伪影；结果不叫人耳准确率、辅音恢复率、FPR或防幻觉成功。没有真正决策信号便停止该来源，不因为新材料便默认存在可研究创新。

## 本次来源结果与后续有限决定

[来源报告](../experiments/p8_keyword_source_feasibility_20261020/experiment_report.md)：两个Range返回206，实际archive182082353B，中心目录16020项；真实词WAV路径8000、每词1000、1750个prefix组、88个完整8词组，无重复路径。0成员内容读取/音频解码。第三次源码请求超时，且保存CD和派生JSON后产物5.57MB超过4MiB，**执行gate失败**，原件保留、不再请求或删件救结果。

Root另用已浏览的官方which_set规则对保存metadata作标准库算术：training/validation/testing分别1399/177/174组，完整8词组71/9/8，组无交叉。此为离线条件计数，不谎称Worker抓取了源码或录音已验证。

信息足以支持一个不同于训练的最低成本适配性问题：Worker已按事先规则登记validation8组×8词=64，testing8完整组reserved；[固定Clean读出方案](../experiments/p8_keyword_clean_readout_20261020/analysis_plan.md)只准备一次取得官方archive、仅抽64原音频和现存tiny.en读出。前次预算失败完整保留，新的200MiB上限针对包含182MB源包的新任务，不倒改原4MiB。

新任务未执行且需代码SHA复核/单独许可：若至少58/64且每词≥6/8则仅判Clean读出可用；失败不换prompt/模型、不过渡到新classifier平台。尚未证明自然弱辅音可恢复，也未批加噪/增强调用。CPEL终末权重不参与。

**执行许可后续**：Root已核188行脚本及14份既有Whisper源码锁，签单次独立许可；Worker负责取得/64Clean读出与监控。未因此批准加噪或增强，也没有准确率结果。resource failure状态与中途活动计数已在新代码静态修正；这不是旧source失败变通过。

## 完整archive取得预算失败：适配性仍未知

父Clean任务唯一运行实际在120.018秒下载期限终止，HTTP200的Length/ETag匹配，但仅保存11,534,336字节不完整archive；0成员抽取/header/ASR加载/转写。所有保护文件不变。适配性是unknown，不是0/64或Clean识别失败，完整失败与partial保留。

Root不延时/续传父任务。经独立静态意见，仅准备一次新的固定成员Range取得修正：相同64manifest、识别器/参数/评分/门均不变，只从同官方URL取得64段，总预算8MiB，按offset/局部header精确切压缩体、限定raw-deflate并核CRC/原大小。元数据预计body5,535,648字节，原WAV总2,050,816字节；非已取得。段内可能过取邻近压缩字节而不解码，不能验证完整archive SHA。子目录 `p8_keyword_clean_readout_20261020/transport_attempt02` 保留独立config/plan，实际许可尚待代码审核。若再失败不追加第三种抓取策略；0既有ASR结果可用于择样或救评分。

**Range许可后续**：Root已核最终224行脚本与父科学字典、选择/评分函数完全一致，签唯一子任务许可。Worker执行并监控，仍无读出结果。父全包取得技术失败不变，若此运输再失败不追加第三策略；不批加噪/增强。
