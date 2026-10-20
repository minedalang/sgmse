# P7-0：StoRM 采样调用链与 Router-off 基线审计

## 状态与范围

**状态：已运行，范围完成但不包含质量评价。** 本实验只审计固定 80k checkpoint 的真实采样路径，并对一个文件做确定性 Router-off 重放。没有训练、没有批量重新生成质量评估音频、没有修改数据集，也没有覆盖 checkpoint。

实验目录中的 `config.yaml`、`metrics.json` 和 `baseline_manifest.csv` 是本报告的机器可读依据。

## 固定对象

- Checkpoint：`joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`
- Checkpoint SHA256：`ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9`
- 训练/模型配置：`train_type=mode2`、`sde=ouve`、`sampler_type=pc`、`N=30`、`theta=1.5`、`sigma_min=0.05`、`sigma_max=0.5`、`t_eps=0.03`、`loss_type=score_matching`、`loss_weighting=sigma^2`、`network_scaling=1/t`。
- 采样器：`reverse_diffusion` predictor，`none` corrector，`snr=0.5`，`use_ema`，seed 基准为 `2026`。
- 开发 manifest：`experiment_logs/rrsd_tcut_screen/manifest.csv`，32 条；manifest SHA256 记录在 `config.yaml`。该 split 曾用于路由筛选，只能作开发诊断，不能作为最终盲测。

## 已核查的调用链

1. `joint_enhance_by_dir.py` 读取 noisy 波形，按该文件的最大绝对值归一化，调用 `model._frontend_forward(y/norm)` 得到 `Y_pre`，再乘回归一化因子并中心裁剪/补齐到 noisy 长度。
2. `train_type=mode2` 时，扩散主输入是 `Y_pre`，条件是 `Y_noisy`。因此 Router-off 关闭的是额外 RRSD score 修正，仍然保留 StoRM 前端及 noisy 条件；它不是“无前端”基线。
3. `ScoreModel.enhance` 根据 checkpoint 保存的 `self.sde.sampler_type` 选择 PC。推理脚本的 `--sampler_type` 不会把保存的 `pc` 改成 ODE。
4. PC sampler 在每个时间步先调用 corrector，再调用 predictor。`NoneCorrector` 不调用 Score；`ReverseDiffusionPredictor` 每个时间步调用一次 reverse-SDE 离散化，因此本配置的 Score forward 次数为 `N * (0 + 1) = 30`。
5. 对 checkpoint 的 EMA 做了实际加载；推理调用 `model.eval(no_ema=False)` 会将保存的 EMA shadow 参数复制到当前模型。该 checkpoint 的 EMA state 有 8 个 shadow tensors，checkpoint/模型加载检查确认它们是 Router 参数，不是完整 Score backbone 或前端参数。因此应表述为「Score backbone 与 frontend 使用 checkpoint state_dict 权重、Router 使用 EMA」，不能笼统称作完整模型 EMA。

## 运行证据

- 带 forward hook 的单文件运行摘要报告：返回的 `nfe=30`，DNN forward hook 计数为 `30`；耗时约 `9.18 s`，RTF 约 `0.306`。`config.yaml` / `metrics.json` 中保存了调用数和返回 NFE，但独立 hook 脚本与原始 stdout 未归档；该 hook 计数暂属**未保存原始记录的历史运行证据**，不能作为完全独立复现的 instrumentation artifact。当前代码调用链给出的计数公式仍可直接审计。
- 文件 `000_fileid_2673.wav` 使用相同 checkpoint、EMA、PC、`N=30`、`snr=0.5`、seed=2026 做 Router-off 重放。
- 重放写出的 WAV 与历史 `experiment_logs/rrsd_tcut_screen/off/000_fileid_2673.wav` 重新读入后逐采样一致：`max_abs_diff=0.0`、`rmse=0.0`。
- 32 条 manifest 中 noisy、clean、Router-off 均为 16 kHz，三者长度一致；未修改原始 noisy/clean 数据。

## ODE 分支审计结果

仓库存在 probability-flow ODE sampler，但当前 checkpoint 不是 ODE 配置，因此本实验没有把 ODE 结果混入 PC 基线。代码审计还发现：

- ODE sampler 的 `z` 参数没有用于固定初始状态，调用时仍会重新采样 prior；因此不能直接声称已完成共享 `x_T` 的 ODE 收敛比较。
- ODE 返回的 `nfev` 没有加上末端可选 denoise Score call。

这两点在修复或建立受控 wrapper 前，不能用来解释 PC 结果。

## 结论边界

本实验证明了固定 Router-off 重放的调用链、NFE 计数和单文件输出一致性，足以继续做有边界的 P7-C 或 P7-A 诊断。它没有证明 Score 学习正确、Router 有效、前端或后端造成了某种听感损伤，也没有产生 PESQ、STOI、SI-SDR 或 DNSMOS 结论。现有 150 条 Router-off 目录缺少完整生成命令和配置，不能替代本审计的权威复现基线。

## 下一步

先导出与同一 32 条 manifest 严格配对的 Frontend-only `Y_pre`，使用和 `joint_enhance_by_dir.py` 相同的归一化与长度对齐；只在音频和 STFT 一致性核验通过后，做小规模幅相/子带归因。若没有稳定、可由部署时可观测量识别的互补规律，停止 P7-C，不启动新的训练模块。
