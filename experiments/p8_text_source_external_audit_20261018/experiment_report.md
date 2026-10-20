# P8 原始文本来源外部审计

## 目的

确认 P8 的 `book_*/reader_*` 文件名和 clean 自动转写片段能否恢复到可核实的原句文本。该审计只读，不修改数据集、音频或 checkpoint。

## 本地结果

此前对 `/data/datasets/dns_ins20` 和 P8 manifest 的扫描已确认可见 SCP 的 `text` 全部为 `dummy`。本轮再次对 `/data` 的非音频/非模型文本和元数据进行关键词扫描，没有发现与 `fileid_5243`、`reader_10658` 等匹配的原始书籍文本。

## 外部检索结果

通过用户提供的 Mihomo 代理尝试检索 clean 自动转写中的片段：

- DuckDuckGo 返回 bot challenge，未获得结果正文。
- Bing 对精确短语 `"But there are no sharp lines in nature"` 返回无结果。
- Google Books API 返回 quota exceeded，未获得结果。

这些失败不能证明原文本不存在，只能说明本轮没有得到可验证来源；也没有把任何搜索片段写入数据或实验标签。

## 判定

当前仍缺少可引用的原始文本来源。现有 Whisper clean 转写只能用于帮助人工听音，不能用于正式 WER/CER、E1 判定或训练监督。P8-1 的正式人工核实时，需要用户/听者从 clean 独立核实文本，或提供合法原始转写文件。
