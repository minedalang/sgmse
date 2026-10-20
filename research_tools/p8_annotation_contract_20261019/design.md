# P8：实际发音标签契约（仅软件准备）

Research Lead 批准的范围：标准库纯函数 + 合成单元测试；不是语料导入、候选生成、verifier 或增强实验。只在本目录新增文件；总产物 <1 MB、合成测试 <60s。禁止网络、注册/邮件、安装、真实音频/标注导入、模型调用、训练、改动旧实验/源码。来源尚未获取，不能称可用真实训练集。不得通过第三方镜像绕过数据访问流程。

## 科学目标

增强应保存录音实际发音，而不是偷偷做口音/发音纠正。把期望音素与实际感知音素分开，使“把原本未说出的目标音补出来”和“恢复被噪声掩盖的已说出音”不混淆。标签契约只保护数据语义，不能验证声学证据或算法新颖性。

官方 [L2-ARCTIC 标注说明](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/) 给出：替换 `CPL,PPL,s`，添加 `sil,PPL,a`，删除 `CPL,sil,d`；CPL 是期望音素、PPL 是感知实际音素。`err` 与带 `*` 的偏离不得自动变成确定身份。`/annotation` 与 forced-alignment 的 `/textgrid` 必须区分。实际文件格式尚未取回验证，本目录不实现整个 TextGrid 读取器。

## API 与输出

实现 `translate_annotation(raw_label, *, source_kind, source_id, start_s, end_s)`，纯函数，不读写文件。`source_kind` 仅 `manual_annotation` / `forced_alignment`；`source_id` 非空。时间接受有限 Decimal-compatible 数值/字符串，拒绝 bool/NaN/Inf/负值/倒序，精确保留为十进制字符串；允许零长但显式标记，不推断其代表真实静音波形。

输出至少保存：原始完整标签、期望音素、实际音素（不确定可为 null）、原始感知符号、语料标注操作、manual/forced来源、source_id、起止及零长标志、`identity_uncertain`、`deviation`、`realized_class`（speech/silence/uncertain）、`real_data_approved=false`、`hard_identity_eligible=false`。

语音符号保持原样另给大小写规范化字段即可；本阶段只校验标签结构与字符，不声称确认真实语料音素全集。无模型 abstention 输出；源标签 uncertain 不等于模型 unknown，不产生波形动作。`hard_identity_eligible` 保持 false：没有实际语料/标注来源审核，不自动升级 hard gold。

## 必须满足的语义

1. manual `K,G,s`：期望 K、实际 G；保真目标是 G，不是补 K。forced 源不得接受这种人工错误标签。
2. manual `K,sil,d`：期望 K、实际标为 silence/目标缺失；不得把“补 K”计为恢复；不推断逐样本零或精确声学边界。
3. manual `sil,K,a`：实际 K 存在，虽被语料称添加也应保存；不能当增强新插音。
4. manual `K,err,s`：实际身份不确定；不是 silence、不是确定 K。manual `K,G*,s`：保留星号，标记偏离/身份不确定，不能不加说明地升级为确定 G。
5. 单符号 manual `K`：记录源声明 K；单符号 forced `K`：仅记录对齐估计。二者不能共用“人工确认”标志。单独 `sil` 仅是源静音标签，不证明任何特定音素的匹配缺失反事实。
6. 显式拒绝空字段/非法操作/额外逗号/无效类型及矛盾操作：a 的期望非 sil，d 的实际非 sil，s 含 sil。不要猜测修复。源文件将来若有新编码，先拒绝并审计，不静默扩展。
7. `annotation_operation` 仅是语料相对标准发音的标记，不等于增强输出错误。声学是否可辨、人工分歧、候选覆盖、模型拒答另设字段/流程；本工具不计算。
8. 合成测试固定 expected literals 独立断言以上语义与错误路径，不用同一函数生成自身 expected。标明 synthetic-only，不捏造真实样本/人数/错误率；不得扫描本地数据运行导入。

## 交付

`annotation_contract.py`、`test_annotation_contract.py`、`preparation_report.md`、机器可读测试/文件哈希回执。报告写清测试数、实际命令/环境、边界、未验证真实编码；代码不依赖 torch/numpy/音频库。无需 subprocess/socket allowlist框架、全局服务、dataset CLI 或通用 TextGrid parser。

Root 复核通过只证明软件遵守该契约，不批准实际语料执行、verifier 训练或修改 StoRM。
