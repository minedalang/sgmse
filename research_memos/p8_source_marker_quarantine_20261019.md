# 标注特殊符号隔离规则：到此停止扩建解析器

本规则是 Research Lead 的接入前置约束，不是声学算法，不授权导入完整语料。依据既有公开单例审计 `experiments/p8_l2_public_annotation_schema_audit_20261019/`：来源/30个interval结构验证成功，但这不能证明旧v0对每种来源符号都有正确语义。

| 输入情况 | 必须保留 | 处理决定 |
|---|---|---|
| 非phones tier（words/IPA/comments等） | tier名称、范围、原标签及来源 | 只保存元数据；不送入phones转换函数，也不计为phones错误 |
| phones空标签 | 原始空字符串、interval索引及起止 | `unlabeled_interval`隔离；不补成sil、不丢interval、不当声学缺失真值 |
| phones `sp` 或其他未在来源profile中明确注册的符号 | raw symbol、tier、来源、时间 | `unregistered_source_symbol`隔离；本规则不猜测其含义，不映射speech或silence |
| err/带星偏离 | 原始标记与expected/perceived字段 | 保留源不确定/偏离；不是模型abstention，更不自动给hard gold |
| 结构矛盾/非法字段/不合法时间 | 可审计错误及原输入 | 失败并隔离，禁止静默修复或为了覆盖率修改标签 |
| 来源profile已核实的普通音素/操作 | raw、expected、perceived、source operation与provenance | 才可进入有限语义转换；仍不等于标注绝对正确、噪声中可辨、候选完整或增强有效 |

三种状态必须分开：**标注未定义/来源不确定**、**声学观测难以辨识**、**模型选择拒答**。前者不能拿来训练后两者的真值。`real_data_approved`与`hard_identity_eligible`不得由语法解析成功自行置true。

## 实施边界

- v0函数/14项原测试/原文件哈希继续冻结；不得直接把完整TextGrid喂给它。本规则目前是审核/接入禁令，不声称已经实现新的完整reader或完成实际语料转换。
- 不为单个公开示例再开发通用TextGrid parser、自动词表、标注修复框架或新音频试验。只有实际获准的新科研协议需要这些依赖时，才按协议实现最小adapter并另验。
- 首先恢复科学决策：判断候选生成是否有可利用的内容/质量互补、部署输入能否超过简单基线。局部音素身份及双向否决的强主张仍需对应标注，但不应把它们不加区分地设成所有句子级诊断的硬前置条件。

此规则防止静默误标，不构成任何增强收益。软件工程支线到此收口，长期目标仍是实际增强内容/质量/效率的可复现改善。
