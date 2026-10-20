# 前瞻性候选余量：仅注册元数据

Root只批准此阶段，不批准混音/音频解码/模型执行。完整科学设计动机与未决门见项目根 `research_memos/p8_validation_gate_reassessment_20261019.md`；独立建议不是运行许可。

## 选择规则（任何新音频/模型前冻结）

唯一源：`experiments/p8_voicebank_content_reference_audit_20261019/attempt02/train_reference_manifest.jsonl`，SHA256 `7f14a16a20880031932ab4ce7f7d3682c1b16cc0cc2eb8cd141e49ad7de087bc`。

1. 仅local_partition=train，排除历史pilot speaker p226/p287；ID末序号>=26，已有clean_header为16kHz单声道且frames/16000在闭区间[3,5]秒。
2. 先按speaker字典序，取有>=2条合格记录的前4位；每位按整数序号升序取前2条。不得读标签语义/ASR结果/音质选句。选择后若任何一条不合格/被分析过，报告失败停止，不自动换句/换speaker。
3. 保存8条全部metadata、发布者原文及其源SHA、speaker/ID/时长/本地partition、clean/noisy路径、原清单reference_status与pretraining_exposure。原noisy只记录路径，不读样本或用于新混音；未来物理噪声及输入尚未冻结。
4. 对选中8个local clean仅作existence/stat/整文件字节SHA；不调用音频库、不解析/解码音频样本。不将字节哈希等价为spoken alignment或复采样等价。

## 使用史

查当前实验目录/研究memo中的选中ID输出文件路径、已完成模型case/utterance清单，区分全量reference清单/SCP/元数据引用与实际模型输出/听感使用。先记录搜索范围、跳过的大文件/二进制与发现。不得因grep无结果笼统宣称全球未见；已知旧VB模型为p226_027/028、p287_037/054。若无法确定项目内未分析状态，标为待审，不伪报新holdout。完整预训练暴露仍unknown。

## 边界与交付

标准库、0网络/音频解码/重采样/混音/模型/ASR/训练、新目录<=2MiB；单次元数据处理预算60s。独占METADATA_STARTED marker；若已有completed/marker不要重做。若发现异常，保留失败记录而不改选择规则。

Worker交付metadata-only `config.yaml`（Root提供不改）、`metrics.json`、`experiment_report.md`、`prospective_registry.json`、源码/实际命令/时间及独立CPU核验。核验原清单SHA、重建同样8条选择、byte hash及原文字段，不用原函数自证。这里只能证明选择/文件/来源，不证明实际发音或P8-V有效性。

pre/post保留当前git状态与源清单、旧VB/B metrics及所有旧实验；不改标签软件、数据、checkpoint或试听/bridge。该8句属train-split探索材料，不是新方法最终确认集；无模型执行批准。
