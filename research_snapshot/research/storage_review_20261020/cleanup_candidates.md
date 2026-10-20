# /data 清理候选清单（待用户审批，尚未删除）

> **本清单不是删除授权。当前没有删除、移动、压缩任何权重或WAV。** 仅根据文件元数据与既有报告判断当前依赖和再用价值，没有加载模型、解码音频或核对权重张量等价。所有原始指标、配置、日志、答卷、源码与报告都建议保留。

## 先给建议

**最保守的第一批：仅审查 A01。** 它包含5条旧 `rrsd_speaker` 验证路线的10份额外 `last` 保存件，预计释放 **6.259 GB (5.829 GiB)**；每条路线仍保留报告引用的代表权重。仅这一批已足够为此前估计的长程断点空间留出余量，不必先删原始数据、正式评价或全部历史权重。

注意：这些额外保存件**没有证明与保留件张量相同或字节相同**；部分文件大小/SHA已不同。这里的建议是“接受失去该保存件恢复能力，保留代表权重”，不是“已验证重复文件”。

## 汇总

| 范围 | 估计可回收已分配空间 | 含义 |
|---|---:|---|
| A01–A08：39份旧额外权重保存件 | 18.298 GB (17.041 GiB) | 每条路线至少留代表权重；非已证明重复 |
| W01–W10：448条旧验证WAV | 0.431 GB (0.402 GiB) | 丢失对应直接试听及新指标复评能力 |
| T01：失败下载partial | 0.012 GB (0.011 GiB) | 不完整包，不用于训练或实际64条读出 |
| 以上优先候选合计 | **18.740 GB (17.453 GiB)** | 不含下面的高代价候选 |
| B01–B03、V01–V05另审 | 4.372 GB (4.072 GiB) | 会丢代表权重、整次运行恢复或历史试听链接 |
| 两级全部候选合计 | 23.113 GB (21.525 GiB) | **不是推荐全部删除** |

盘点时 `/data` free 为 1.339 GB (1.247 GiB)。空间使用 `st_blocks×512` 静态估算，不含目录元数据，快照/COW/打开文件等可能使实际df变化不同。所有752条候选均为普通文件、nlink=1，没有重复inode；不将符号链接目标重复计量。

## A. 旧权重：优先审查的额外保存件

以下路径统一前缀为 `/data/coding/sgmse/joint_train_logs/`，A08例外。每个目录只选择清单列出的 `…-last.ckpt` 和 `last.ckpt`；**不删除整个目录**。

| ID | 内容 | 文件数 | 估计回收 |
|---|---|---:|---:|
| A01 | rrsd_speaker额外10份last保存件 | 10 | 6.259 GB (5.829 GiB) |
| A02 | SSL/WavLM旧小试额外保存件 | 10 | 5.990 GB (5.578 GiB) |
| A03 | 旧conflict/gradient probe额外保存件 | 4 | 2.396 GB (2.231 GiB) |
| A04 | 旧projection smoke/gate额外保存件 | 4 | 0.886 GB (0.825 GiB) |
| A05 | 旧k-step/debug额外保存件 | 6 | 1.328 GB (1.237 GiB) |
| A06 | 旧control1000额外保存件 | 2 | 0.443 GB (0.412 GiB) |
| A07 | Anchor V2旧小试额外保存件 | 2 | 0.442 GB (0.412 GiB) |
| A08 | 旧speaker backend-only对照额外last | 1 | 0.554 GB (0.516 GiB) |

### A01精确文件与保留件

五条路线已归档/暂停，不是当前CPEL训练源。依据：[路线归档](../archive.md)、[speaker参数审计](../../../experiments/speaker_parameter_audit_20261017/experiment_report.md)、[旧复验](../../../experiments/rrsd_cv_repeat32/experiment_report.md)。

**候选10件：**
```text
/data/coding/sgmse/joint_train_logs/5l5cb2a0-rrsd_speaker_pair_masked1000/epoch=0-step=1000-last.ckpt
/data/coding/sgmse/joint_train_logs/5l5cb2a0-rrsd_speaker_pair_masked1000/last.ckpt
/data/coding/sgmse/joint_train_logs/ftyjnqu1-rrsd_speaker_pair1000/epoch=0-step=1000-last.ckpt
/data/coding/sgmse/joint_train_logs/ftyjnqu1-rrsd_speaker_pair1000/last.ckpt
/data/coding/sgmse/joint_train_logs/kn1hgnni-rrsd_speaker_svproto_masked_sw05_1000/epoch=0-step=1000-last.ckpt
/data/coding/sgmse/joint_train_logs/kn1hgnni-rrsd_speaker_svproto_masked_sw05_1000/last.ckpt
/data/coding/sgmse/joint_train_logs/q2ilpruq-rrsd_speaker_proto01_cw1_1000/epoch=0-step=1000-last.ckpt
/data/coding/sgmse/joint_train_logs/q2ilpruq-rrsd_speaker_proto01_cw1_1000/last.ckpt
/data/coding/sgmse/joint_train_logs/qpf32q6y-rrsd_speaker_proto01_1000/epoch=0-step=1000-last.ckpt
/data/coding/sgmse/joint_train_logs/qpf32q6y-rrsd_speaker_proto01_1000/last.ckpt
```

**对应必须保留的5件：**
```text
/data/coding/sgmse/joint_train_logs/5l5cb2a0-rrsd_speaker_pair_masked1000/epoch=0-step=1000.ckpt
/data/coding/sgmse/joint_train_logs/ftyjnqu1-rrsd_speaker_pair1000/epoch=0-step=1000.ckpt
/data/coding/sgmse/joint_train_logs/kn1hgnni-rrsd_speaker_svproto_masked_sw05_1000/epoch=0-step=1000.ckpt
/data/coding/sgmse/joint_train_logs/q2ilpruq-rrsd_speaker_proto01_cw1_1000/epoch=0-step=1000.ckpt
/data/coding/sgmse/joint_train_logs/qpf32q6y-rrsd_speaker_proto01_1000/epoch=0-step=1000.ckpt
```

A02–A08的逐文件路径同样已列在下方CSV/JSON中。A08的两保存件已有不同SHA记录，因此不称重复；保留历史报告使用的 `epoch=0-step=500-last.ckpt`。正式测试使用的 `01maaifx-conflict_w0.001_s1000` 整组未放进本候选。

## W. 旧验证WAV：只删WAV，不删指标/目录

统一前缀 `/data/coding/sgmse/experiment_logs/`。它们是旧speaker/SSL/k-step/scale小试输出，不含基座off/router、正式150条full评价或当前CPEL原始输入。

| ID | 目录 | WAV数 | 估计回收 |
|---|---|---:|---:|
| W01 | `rrsd_speaker_eval` | 96 | 0.092 GB (0.086 GiB) |
| W02 | `rrsd_speaker_eval_svproto_masked_sw05_1000` | 32 | 0.031 GB (0.029 GiB) |
| W03 | `rrsd_ssl_mse_eval` | 96 | 0.092 GB (0.086 GiB) |
| W04 | `rrsd_ssl_rel_eval` | 32 | 0.031 GB (0.029 GiB) |
| W05 | `rrsd_kstep_k4_eval` | 32 | 0.031 GB (0.029 GiB) |
| W06 | `rrsd_tcut005_scale075_screen_20261017` | 32 | 0.031 GB (0.029 GiB) |
| W07 | `rrsd_tcut005_scale05_screen_20261017` | 32 | 0.031 GB (0.029 GiB) |
| W08 | `rrsd_tcut005_scale025_screen_20261017` | 32 | 0.031 GB (0.029 GiB) |
| W09 | `rrsd_scale05_screen_20261017` | 32 | 0.031 GB (0.029 GiB) |
| W10 | `rrsd_scale025_screen_20261017` | 32 | 0.031 GB (0.029 GiB) |

删除代价：无法直接重听、追查旧波形或增加新指标；已有CSV、配置、报告不丢。没有证明所有WAV可逐字节再生成。定向symlink扫描未发现外部链接指向这些文件，但不是全部文本引用的穷尽证明。

## T. 小型失败产物

T01仅为：
```text
/data/coding/sgmse/experiments/p8_keyword_clean_readout_20261020/mini_speech_commands.zip
```
该文件只有11.53MB，为超时留下的不完整下载。后来成功读出的64条材料与Range原件不在此项中；失败报告、哈希与receipt仍保留。

## B/V. 需单独决定：会损失整次运行或历史材料

不优先推荐本次删除；A01已能满足当前空间需求。

| ID | 内容 | 文件数 | 估计回收 | 删除代价 |
|---|---|---:|---:|---|
| B01 | rrsd_speaker每组保留的代表权重 | 5 | 3.129 GB (2.915 GiB) | 若连A01也删除，则5组共15份权重全消失；旧路线不能直接模型复评或恢复。 |
| B02 | Anchor retry全部两份权重 | 2 | 0.442 GB (0.412 GiB) | 删除后此retry运行没有任何权重可供直接复评/恢复。 |
| B03 | speaker backend-only报告所用权重 | 1 | 0.554 GB (0.516 GiB) | 与A08一起删除会失去该完整控制的全部权重。 |
| V01 | 旧speaker no-loss控制64-speaker输出：仅WAV | 128 | 0.123 GB (0.115 GiB) | 丢失对应直接试听/指标复评；V05会使历史盲听页面音频链接失效。 |
| V02 | 旧rollout pilot输出：仅WAV | 16 | 0.015 GB (0.014 GiB) | 丢失对应直接试听/指标复评；V05会使历史盲听页面音频链接失效。 |
| V03 | 旧rollout复验输出：仅WAV | 16 | 0.015 GB (0.014 GiB) | 丢失对应直接试听/指标复评；V05会使历史盲听页面音频链接失效。 |
| V04 | 旧混响smoke临时生成及输出：仅WAV | 48 | 0.046 GB (0.043 GiB) | 丢失对应直接试听/指标复评；V05会使历史盲听页面音频链接失效。 |
| V05 | 已完成P3b盲听匿名副本：仅WAV | 48 | 0.046 GB (0.043 GiB) | 丢失对应直接试听/指标复评；V05会使历史盲听页面音频链接失效。 |

其中V05是报告明确的匿名试听副本，若保留 `rrsd_tcut_screen` / `rrsd_selective_screen` 源输出可以考虑它；但删除会使旧P3b盲听页面失效，答卷和answer_key必须留。服务器PID16022及响应记录不在候选中，也不会被停止。

## 明确保留／本次不建议删

- 原始数据、训练数据和 `/data/datasets` 全部内容。
- 300k基座：`joint_train_logs/5ul7u03z/epoch=444-step=300000.ckpt`。
- 当前80k来源：`joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`，以及同目录100k权重。
- CPEL两轮的4份终末EMA、全部实验JSON/配置/代码/provenance与当前长程方案。
- 正式final评价输出、source/off/router基线与 `01maaifx` 正式测试来源权重。
- `z6mc9ska*`、`u4ij4si9*`、`7x0m2u68*` 历史长训权重：归属/恢复价值没有核清，不凭体积或年代判没用。
- 94项结构反馈、12例内容反馈、原VCV/5243/VB32及其试听源；删除符号链接本身不会释放目标音频的大块空间。
- `speaker_headroom_replication_64`、`rrsd_cv_repeat32`、`rrsd_tcut_screen` 等混有基座输出的目录，不整目录清理。
- 两套运行环境、度量工具、Whisper/HF权重缓存、兼容桥及会话备份，本轮不建议删除。

### 一个很大的文件，但没有列为推荐删除项

`/data/aliyunpan-v0.4.0-linux-amd64/Downloads/86e8209211ee43c18d73dccdf862dfb9/dataset/dns_ins20/clean_testandtrain.gz` 约 **50.05GB**。它是数据源压缩包；尚未验证解压完整性、是否唯一备份或能否重新取得，因此不把它当临时垃圾，不进入候选总量。若用户另行确认数据备份策略，才值得单独审查。

## 逐文件附件与审批方式

- [逐文件候选CSV](candidate_files.csv)：752条，包含精确路径、组ID、逻辑/已分配字节、inode/nlink、mtime，所有决策默认 `NOT_APPROVED_DO_NOT_DELETE`。
- [分组摘要JSON](summary.json)：每组理由、代价、保留项和来源证据。
- [全部项目实验权重库存CSV](weight_inventory.csv)：104条，包含未推荐删除的权重，便于核对遗漏。
- [只读库存生成脚本](build_inventory.py)：只写清单，没有删除、移动或压缩操作。

用户可以只批准某些组（如“A01”）或指出个别文件保留。**审批前不执行任何删除；若获准，执行前还须按清单重新核路径、大小、inode/mtime和当前依赖，遇变化即停，不扩大到整个目录。**
