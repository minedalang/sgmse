# RRSD Future–Perceptual Gradient Agreement（FPGA）：冻结模型只读诊断

## 目的和边界
- 利用现有 80k RRSD Router、冻结 StoRM/NCSN++，及官方冻结 `microsoft/wavlm-base-plus` 第 3 层；只检查引导方向的两类梯度是否冲突。
- **不训练、不更新、不保存任何 checkpoint；不增加课程负样本，不实现 InfoNCE**。
- 这是科研假设的可行性诊断，而非证明 WavLM 梯度等于真实感知质量梯度。只用复数谱 Future MSE 和 WavLM 第 3 层逐帧 MSE，不含 DNSMOS。
- 每条语音从 DNS `cv_synthetic` 加载一个随机扩散时间步；默认只对 `0.05 <= t < 0.50` 计算 WavLM 梯度。其余时间步记为 SKIP。

## 诊断定义
同一个 `x_t` 和同一个小时间间隔 `dt`，针对 **同一个未来状态**：

`alpha_tile -> bilinear -> x0_F - x0_B -> routed score -> reverse_mean_step -> frozen next score -> x0_next`

- `g_F = d [E_future(alpha) / stopgrad(E_future(alpha_anchor))] / d alpha_tile`
- `g_P = d MSE(H_WavLM3(iSTFT(x0_next(alpha))), H_WavLM3(iSTFT(X_clean))) / d alpha_tile`

默认 `PG_ANCHOR=zero`，即在 alpha=0 处测局部变化，便于衔接之前的 zero-guidance Future 梯度诊断。

`PG_ANCHOR=router` 可重复同一实验，在已训练 Router 的真实 α 附近测局部方向，检查零引导梯度结论是否仍然适用。**两个 Anchor 的比较为不同实验，不能在同一统计中混合**。

- `g_F < -thr_F && g_P > thr_P`：Future 有利、SSL 不利（候选冲突）。
- `g_F < -thr_F && g_P < -thr_P`：Future 与 SSL 同时有利（方向一致）。
- `g_F > thr_F && g_P > thr_P`：两者都认为引导有害。
- 每条样本 `thr = max(1e-10, 0.1 * mean(abs(gradient)))`，可用 `PG_REL_TOL` 等覆盖。
- 还统计 `alpha_conflict_budget = sum(alpha_router * conflict) / sum(alpha_router)`，衡量部署中的引导预算实际落在冲突位置的比例；和“全部时频 tile 的冲突比例”一起看。
- `fd_*` 对梯度幅值较大的一组 tile 加 `0.02`，测实际 Future 与 SSL 误差变化和一阶预测符号。数值变化太小的记 `fd_*_valid=0`，不要把不稳定的有限差分视为失败。

## 安装（不打源码补丁）
解压 ZIP 后，将三个 Python 文件和 Bash 运行脚本放到 `/data/coding/sgmse/` 根目录。不要修改 `sgmse/joint_model.py`。

需要环境能从本地 cache 载入 `microsoft/wavlm-base-plus`；如需下载一次，设 `PG_LOCAL_ONLY=0`，网络下载完后改回 1。

## 先跑 4 批
```bash
cd /data/coding/sgmse
BATCHES=4 BATCH_SIZE=1 PG_LOCAL_ONLY=1 CUDA_VISIBLE_DEVICES=0 bash run_rrsd_pg_agreement.sh
```
寻找 `[RRSD-PG-INIT]`、`[RRSD-PG]`、`[RRSD-PG-DONE]`。注意随机 t 过滤，只有 1–3 条入选也属正常。

## 正式数据：先在零引导状态测试
```bash
BATCHES=96 BATCH_SIZE=1 PG_ANCHOR=zero PG_FD_BATCHES=8 \
  PG_LOCAL_ONLY=1 CUDA_VISIBLE_DEVICES=0 bash run_rrsd_pg_agreement.sh
python rrsd_pg_summarize.py experiment_logs/rrsd_pg_zero_wavlm_l3_cv_synthetic_96b_seed2026.log \
  | tee experiment_logs/rrsd_pg_zero_summary.txt
```

## 如果零引导位置存在冲突，再检查当前 Router 的实际 α
```bash
BATCHES=96 BATCH_SIZE=1 PG_ANCHOR=router PG_FD_BATCHES=8 \
  PG_LOCAL_ONLY=1 CUDA_VISIBLE_DEVICES=0 bash run_rrsd_pg_agreement.sh
python rrsd_pg_summarize.py experiment_logs/rrsd_pg_router_wavlm_l3_cv_synthetic_96b_seed2026.log \
  | tee experiment_logs/rrsd_pg_router_summary.txt
```

以上两次用相同 `--seed 2026`、相同数据及抽样；需要配对时以 `[RRSD-PG] batch/sample/t` 核对（不要只比均值）。

## 显存或速度
- 默认 FP32 和 BATCH_SIZE=1。WavLM 只提取 2 秒音频第 3 层，但实现仍运行所有 WavLM 层；反向传播的开销会明显增加。
- 如果 OOM，请提供显卡型号和错误堆栈；不要先切到 bf16，否则微小差分更容易被精度噪声掩盖。
- 如果 FD 有效样本极少，不要急着改阈值；先检查实际 `fd_..._obs` 是否稳定大于浮点噪声。

## 解释/停止标准
优先看：
1. MID 与 LOW 的 `alpha_conflict_budget`：冲突是否发生在 Router **实际使用** 的位置？
2. `fd_p_sign_ok / valid`、`fd_f_sign_ok / valid`：表征与未来梯度符号是否经实际扰动支持？
3. `PG_ANCHOR=router`：当前动作附近是否依然冲突，而不是仅在 α=0 处冲突？

仅当“冲突足够多 + Router 确实使用了这些位置 + 两个有限差分方向基本可信”成立，才值得考虑针对性训练损失。否则暂时不加 WavLM 感知正则。

**注意**：0.01% 的小变化不能视为感知质量改善；最终仍必须测独立数据上的 DNSMOS/PESQ/STOI/SI-SDR。此实验不验证模型创新性、模型性能提升，也不使用语音质量分数作为训练目标。
