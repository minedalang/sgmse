# I1b：局部操作配对完整性（软件准备冻结）

本设计落实P8-I用户提案的第一阶段，仅在将来一次有界CPU运行中选区和验证内存操作，不训练探针、不改变StoRM/OUVE、不输出新音频。完整规则见同目录冻结config.yaml；上位科学解释见锁定research_memos/p8_i1b_event_edit_protocol_20261019.md。

## 当前授权与禁止

Engineer仅本目录代码+合成数组/临时WAV测试，可NumPyCPU，禁止真实WAV/header/sample/选区/main试跑、网络/模型/ASR/训练/委派；不改本设计/config或旧件。Root另审后Worker才能真实执行一次主程序+一次独立validator，各硬120秒，总目录8MiB，原24文件主解码+独立24，0新WAV/NPY/PCM文件。完整payload只内存hash，禁止伪称未读音频bytes。准备≤256KiB，独占marker、全deadline覆盖收尾、无自动重试，原来源pre/posthash。

## 计算规则与不可偷换

仅I0固定12对、gain1/lag0、rawN=X-S，noise来源unknown不跨句reuse/noise定标，不调用任何前端或新模型。config明确320/160/512 symmetricHann rfft、一侧归一功率、2k–7.5k频带、transient正谱差与sustained7帧比例均值、80ms窗口和0.2s边界、160gap/tie。完整候选分数保存，不只top；至多24目标槽，任何零分母/无正分候选/无非重叠位置control均null+reason，不能换句或调规则补数。它们是自动声学结构候选，不是真实音素/弱辅音标签。

位置control不是负例：来自同句但可能含语音，仅未加窗Clean能量尽量接近。两family共用同一位置，另完整报告各真实taper后的delta能量差与局部谱/包络描述。两family预设不是独立自然发音，sham也只校验零操作。

每个event×family×site计算S0=S+delta、X0=X+delta及原/sham，对X1不重算重编码；所有slot保留。masklinear/cosine离散端点和plateau严格按config，没有窗长/深度搜索。浮点重构不证明pureN。int16量化先检查预量化范围及rint整数范围，越界记录unsafe，无clamp/no音频输出；不删unsafe对/更改强度或归一化。safe行保存内存PCM payloadSHA、量化误差，mask外相等及原/sham完整WAVSHA，strictJSON null/reason。

技术完整性通过≠操作可辨识性≠音素真实存在；不计算分类准确率/AUC/dprime，不冻结训练split，所有科学/noise/split/traininggate保持false。该工作没有替代将来多noise、多SNR、D0–D4、校准/拒答和独立内容确认。

## 实现/独立验证

最小driver/独立validator分离，不共享候选/统计/mask/量化业务函数。允许同NumPyFFT后端但独立帧/窗/选择与mask实现；合成小FFT再用直接DFT核reference以避免只比同错误。真实validator逐项独立重建全帧分数、24event槽、控制排名/坐标、所有operation行、mask和float/PCMhash，原metrics不可变。源码/锁/原音频Hash与解码counts独立记录。

合成覆盖已知HF瞬态/持续段、零/constant/无candidate、最早tie、边界/重叠/缺control、两mask端点/镜像/plateau、不同位置能量误差、原/shambytecopy、delta0和mask外精确相等、半LSB ties-even/负数、float越界而round未越界与round越界均unsafe、不得clip、残差并非pureN的限制、坏source/path/header、existingmarker、deadline/bytes/网络模型捕获拒绝。不得因synthetic通过声称真实事件通过。

## 交付和停止

preparation_report/preparation_metrics（不混metrics.json）、测试源码/回执、manifest列锁。原config和本设计不改；遇科学定义冲突先报Root。完成交SHA与精确future命令后停止，当前无真实运行/导出许可。
