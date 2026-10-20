# 最终确认：仅两个压缩包，尚未删除

用户扩容已在文件系统生效，/data可用约106.6GB，训练不再因旧空间不足而必须删实验权重。此次改用更保守方案：**所有权重、验证WAV、已合成数据和科研记录全部保留**。

仅提议删除以下两项，仍等待最后确认：

| 文件 | 预计释放 | 删除代价 |
|---|---:|---|
| `/data/aliyunpan-v0.4.0-linux-amd64/Downloads/86e8209211ee43c18d73dccdf862dfb9/dataset/dns_ins20/clean_testandtrain.gz` | 50.05GB | 用户确认有备份；当前训练不读取该包。以后重新合成数据需从备份重新下载。代理未独立验证远端备份。 |
| `/data/coding/sgmse/experiments/p8_keyword_clean_readout_20261020/mini_speech_commands.zip` | 11.53MB | 是超时失败的不完整下载，非实际64条读出材料；仅失去partial原始字节，失败报告/SHA/receipt保留。 |

**不执行旧清单中的A01–A08或任何WAV删除。** 这两项确认若获批，只适用于这里两个精确文件，不代表批准旧752项清单；执行前需核metadata未变化，遇变化即停。没有递归目录删除、没有压缩或移动旧工件。

机器清单：[final_confirmation_two_archives.json](final_confirmation_two_archives.json)。
