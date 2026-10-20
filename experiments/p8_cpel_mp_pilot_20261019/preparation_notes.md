# CPEL M/P 软件准备：未执行真实实验

本目录新增 `run_pilot.py`、`loss_helpers.py`、`editing.py`，不改项目源码、旧实验、原始数据或 checkpoint。已实读当前 Root `config.yaml` / `analysis_plan.md`。本次未实例化真实模型、读取 checkpoint storage/WAV header/PCM、跑真实训练/评价、访问网络或创建执行授权。

## 实现边界

- Tensor-only loss 包含原生均值 DSM、共同 proposal 的逐点辅助回归（不称无偏 DSM）以及能量缩放白化 pair 四项。batch 为一个原始 pair；inactive 整项为可反传的零，不先除零再 mask，也不除以 active 数。
- 编辑在波形进行，原 Noisy 同归一因子；双端各自 STFT/压缩后相减作 clean 谱目标。rFFT 平滑带通、80 ms Hann/Tukey、固定 clean 能量中心、192 ms 全频软响应窗均独立于模型输出。移动能量用累积和，避免长卷积构造。
- noise-only 控制先检查实际中心 crop 的 `X≈S+N`；同 heldout speech 中心/滤波/Tukey，用向下缩放匹配 speech-edit 波形能量。不足、零或前提不成立均保留 unknown，不放大、不改 gain/lag、不换例。方向为实际两端 Noisy 压缩谱差，不拿 clean 的零差分或 `Phi(delta_noise)`代替。
- 唯一可调用真实执行入口为 `--execute-arm M/P`，必须先存在 Root 手工创建的 `root_execution_authorization.json`。脚本不会创建许可。许可要求 `status: authorized`、`allowed_arms`、`config_sha256`、`code_sha256`（三个 Python 文件各 SHA）、`registry_sha256`（train/dev）及 `checkpoint_sha256`。全部先核，之后才读取 checkpoint bytes。STARTED 与 EMA 文件防止自动重跑；没有恢复或救参路径。
- source 模型严格原 topology/statekeys 加载，使用训练映射 `JointWaveScpDataModule`（两次失败后的静态接口修复详见 `interface_fix_report.md`），仅将外部 `frontend_ckpt=None`；源 EMA 物化一次后只解冻 `dnn.*`，前端和 dnn 均保持 eval（参数梯度仍开）。后续不调用旧 `model.train/eval`，避免旧 Router-only EMA 恢复/覆盖新 dnn。新 EMA 仅跟踪 dnn；仅保存 terminal backend EMA state，不复制前端、源 checkpoint 或 optimizer。
- 已静态读现存 `p8_checkpoint_base_identity_20261019/metrics.json` 的源 A_hparams：contrastive_weight=0、final_content_weight=0、selector_content_weight=0、frontend_ckpt=null；未重新读权重。runtime 仍需真实小 preflight 证明严格加载、EMA 初值及冻结/梯度。若 constructor 出现本地缺文件或外部模型请求，只报失败，不 strict=False、不下载或安装。
- 入口在导入 ML/model 库前拒绝 socket connect/connect_ex，并设离线变量；任何连接尝试记录为失败。CUDA float32/complex64，固定确定性、CUBLAS workspace；不 AMP。实际 CUDA kernel 的确定性支持尚未测试，失败不自动降级或重试。
- 每步独立调用原生两支和共同 aux 两支，共四份 Score；M 同样计算 pair 诊断但权重零。Adam 1e-5、500 步、clip1、EMA .999。初始化和训练采样/裁窗/parent/random draws 使用相同 seed 与独立显式 generator，dev 不消耗训练 generator。
- 正常评价直接使用既有 `get_pc_sampler(reverse_diffusion, none, anchor, N=30, score_fn=...)` 和 `to_audio`，不调用会再次归一化的 public enhance，不调用 RRSD/其他校准器或旧 training_step/aux hook。初始 base64 由 M 唯一计算，P 仅在 config/code/registry/source SHA 全同后复用。正常/控制 Score 调用分别实计，sampler NFE、前端、backward、optimizer 次数也记录。
- 所有64 dev 都保留；STOI/PESQ 未安装或 undefined 逐指标记 null/error，不安装或换例。正常采样/音频几何/显存等致命错误终止本臂并保留已有部分记录，不默默当零分或成功。无 ASR/DNSMOS、音频输出或独立音素恢复主张。
- 每臂墙钟含初始化及评价限制3600秒；512 MiB 总目录及至少1 GiB free runtime guard，保存前按 dnn 大小预留并检查两臂可装下。SIGALRM 为 Unix CLI 保护，未实际触发测试。

## 已做的少量新合成检查

仅 CPU 新随机 tensor：
1. 精确目标方向 `q1-q0=a*delta/sigma²` 在 P=1 时 response≈1、pair≈0；反传有限。
2. clean delta=0 时 pair 完整 skip、返回零且梯度有限。
3. 共同逐点辅助回归的解析残差零解。
4. 原生 DSM 的解析残差零解。
5. 纯合成波形编辑保持 `(X−delta)−(S−delta)=X−S`，固定中心边界及 shape。
6. 软谱窗范围/shape、中心裁窗及共同归一几何。

六组通过；三个 Python 文件最新 AST 解析通过。不是实际 SGMSE API 集成、真实训练、GPU 梯度、模型性能、真实音频或真实浏览器验证。

## 待 Root 审核/未来执行验证

初始准备阶段没有真实 preflight 许可；后续Root/Worker另行授权的两次真实preflight均失败并终止，M/P未开。本次静态修复不复用旧许可、不运行第三次。已另备 `preflight_pilot.py --execute-preflight`，不调用500步入口；其专用 `root_preflight_authorization.json` 独立绑定两config、四代码、两registry与source SHA。仅train[0]/dev[0]六个WAV，两个中心crop加性检查、四Score训练图、一次pair autograd.grad及一次总loss backward（共两反向遍历）、零optimizer/EMA/参数更新；一条dev真实30Score原生推理。只核输出finite/几何，不跑STOI/PESQ或64评价、不存波形/权重。模型参数/buffers与源文件pre/post digest、300秒与新产物2MiB保护均已准备，**完整成功路径尚未实际验证**。单crop梯度尺度记录gPair/gM/gTotal及effective ratio，不调权或作总体判断。

本次补充三个runner静态修正：固定config数值断言；正常dev每row实际Score计数必须30；SI-SDR的silent reference/非有限投影比明确null/error，不用eps伪造有限结果。新增微型复杂梯度代数检查验证`gTotal−.1*gPair=gM`，四文件AST及当前config断言通过；不运行真实preflight gate或真实数据代码。

仍须Root审代码后决定是否单独授权Worker新的隔离小preflight；不能把合成检查当真实API/GPU验证，更不能直接开展M/P。现有STARTED/失败结果及attempt01/02快照保持；没有M/P训练曲线或新增强权重，不把软件准备或失败预检描述为CPEL成功。

后续Root报告attempt03真实集成通过（4+30Score、两反向、零更新）；结果与快照保持。Root随后发现原生Gaussian Fourier `W`被旧全dnn解冻误打开。本次仅静态修复其原生冻结语义、trainable-only optimizer/grad集合和新EMA；详见`intrinsic_fixed_parameter_fix.md`。attempt03旧271梯度记录不倒改成新集合已验证。一个tiny CPU固定W例子通过；没有第四预检或M/P授权/执行。
