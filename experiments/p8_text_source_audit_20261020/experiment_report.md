# P8 本地原始文本来源审计

## 目的

检查 P8 八条 clean 文件在本地 DNS Ins20 树中是否有可核实的原始文本或非 `dummy` SCP 标注，避免把 Whisper clean 转写当成人工真值。

## 方法

只读扫描 `/data/datasets/dns_ins20` 的 SCP `text`、README 和常见文本/元数据文件，并核对 `experiments/p8_frontend_backend_audit_20261020/manifest.csv` 中的 8 个 file-id。未下载网络资源，未写入数据集，未重新推理。

## 结果

- P8 manifest：8 条 clean/noisy 配对。
- 本地 SCP `text` 文件存在，但其转写字段为 `dummy`；没有发现包含真实句子的非 dummy SCP 文本。
- 未发现能按这 8 个 file-id 直接映射到原始书籍文本的本地元数据文件。
- 已有的 `clean_center12s_beam5_reference.md` 和相关 Whisper 文件只能作为听音提示，不能作为 WER/CER 真值。

## 结论与影响

当前无法仅依靠本地 DNS Ins20 文件完成正式人工转写真值核验。P8-1 的正式内容结论仍需独立文本来源，或由听者在不先看 clean 的条件下盲转写后再进行 clean 原句核实。当前用户的听感观察可用于更新 H8-E 假设，但不支持启动内容损失、Harm Head 或大规模训练。

数据集与 checkpoint 未修改。详细哈希和逐文件扫描统计见 `metrics.json`。
