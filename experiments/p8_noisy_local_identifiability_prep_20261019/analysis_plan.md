# P8-I / I0：元数据软件准备冻结设计

来源为用户本轮新增提案，集成计划 `/data/research/p8_noisy_identifiability.md`。该文件与config共同约束本目录；实现准备不是运行授权。

## 当前唯一任务

Engineer仅在本目录实现标准库元数据注册driver、独立validator及合成测试。不得读取真实WAV/header、进行真实语料选择注册、产生STARTED/registry/metrics.json、解码音频、导入torch/numpy/whisper/质量库、训练/推理、网络/下载、重采样、编辑原数据/旧实验。允许只读协议、冻结config/manifest与现有文档以确认schema。

## 将来Root另批、Worker唯一注册的行为

- 使用config冻结来源和选择规则，12对为上限且预期精确数量；不足按失败/不足报告，不放宽。仅local train；原本valid/test不得作为候选。
- 真实注册时核对来源hash、按声道/采样率/时长/ID/speaker选择，冻结全ID。对24个选中clean/noisy仅文件完整字节hash及标准库WAV header核验，不读取PCM样本；要求16k/mono/PCM16/匹配长度。manifest现存header不等于实际文件重核。
- 记录本地clean/noisy路径、bytes/SHA/header、源行hash；不声称已完成对齐/增益/近似加性或纯noise验证。header匹配不足以过I1波形门。
- 按config的严格根/深度/数目/总读取字节预算，检查本机noise来源元数据线索。禁止进入test/valid、读取音频或自动扩大搜索。无来源只能标noise_recording_id=unknown；存在日志也须核内容才能填身份，不能按噪声类别代替录音来源。若关联不足则noise_disjoint_ready=false，不把登记成功算该门通过。
- 原使用史只披露本次排除了六个已用speaker，其余外部/匿名/预训练unknown；不声称全球unseen。split未冻结，不能根据12行相邻顺序假装speaker/noise分组已通过。
- 仅一次进程、独占marker、60秒/2MiB监督、0网络/自动重试。原数据与既有实验只读。metadata成功与科学门ready分别报告。
- 成功执行必须有config.yaml、metrics.json、experiment_report.md；软件准备用preparation_*另名。独立validator从冻结源行/排序重新推导选择，不导入driver，不重新哈希整语料，不调用模型。已有STARTED不得重跑。

## 软件准备验收（合成，不读真实语料）

覆盖train-only、speaker排除、数值后缀排序、3–5秒边界、重复/坏schema/header错、缺源/坏SHA、数量不足、已有marker、来源扫描限额/路径逃逸、noise unknown不得升级ready、禁止网络/模型与写源、产物/超时。只用临时目录和合成JSON/WAV header字节。

## 后续未授权范围

I1真实对齐/残差/编辑/混音、D0–D3/hidden features/ASR、D4数值评测、StoRM/OUVE修改均不在I0权限中。I1的事件数/noise/SNR只是总计划草案，需来源核查后另冻，不得顺带实现或执行。
