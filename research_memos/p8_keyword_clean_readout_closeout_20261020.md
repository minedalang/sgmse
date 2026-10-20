# 固定短词Clean读出：技术完成，适用性失败，停止当前端点

## 可靠结论

唯一固定64成员Range取得与tiny.en Clean读出于104.054秒完成：64GET、5,535,648响应body字节，64成员CRC/header及原WAV SHA联结通过；64 transcribe/64内部decode、ASR加载1，0加噪/F/backend/训练。保留过取邻近压缩字节而不解析/解码其他成员的限定，完整archive SHA不可得。34保护文件前后不变；父完整包下载预算失败和原source metadata失败不改判。

原定门为至少58/64且每词至少6/8。实际49/64=76.5625%；down4、go7、left7、no5、right6、stop6、up7、yes7。全64固定样本保留，未换词/组、未调prompt、beam或模型。Root和独立Reviewer从保存JSON逐条复算tokens/flags/exact-label完全一致。[终态完整报告](../experiments/p8_keyword_clean_readout_20261020/transport_attempt02/experiment_report.md)。

## 未采用端点，不误判语料或模型增强

**关闭当前tiny.en读出的词身份转移端点，不进入Noisy/F/native backend，不改prompt/ASR、训练KWS、另挑样本或去testing救场。** 尚未测任何增强内容收益或损伤。这一结果不能证明Speech Commands目录标签错误、某种真实发音错误、Noisy物理不可辨或StoRM失败；只说明该固定读出在Clean上已不满足本诊断的适用门。

empty1、multiword1、OOV12（标志可重叠），另有2项单token集内替换。`dbb40d24`组0/8，占15项不匹配中的8项；其余组7/6/7/7/7/7/8。此异质性是观察，不定位口音/录音/年龄/身份原因，不在未听/未测时猜测。不能删此组或把now映成no等来改变门。即使Clean正确，增强域读出可靠性原本仍未证明；现在没有理由把单识别器作为真实词保真的裁判。

## 对下一研究动作的约束

CPEL小试提示一般响应形状代理没有足够独立收益；本次又排除了一个未经校验便可直接使用的词读出捷径。不要再轮换代理、分类器和词表直到出现漂亮恢复，也不因此退回只优化DNSMOS/MSE便声称内容问题解决。

当前没有新训练或增强调用授权。若再次研究内容恢复，应由新的可靠参考/独立读出资源或不同且可证伪机制支撑，不能把已关闭端点当确认集。完整音素强标注并非一切早期实验的先决条件，但更换评估器救当前失败也不是科学进展。论文可保留这项适用性负结果，长期机制/内容/身份/效率目标仍未达成。

数据锁：`metrics.json` SHA256 `2113502e6ae2523b673a731fda001ad90bef9d789cd07904b5cdb3479136322d`。所有本段算术仅用保存JSON，无新音频/模型/ASR调用。
