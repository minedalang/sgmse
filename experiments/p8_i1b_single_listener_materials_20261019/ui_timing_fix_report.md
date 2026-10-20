# 单听者页面：时间定位最小修复

针对用户无法把黄色区对应到听觉时刻的问题，仅修改 index.html 与 player.js：增加只读 0–480 ms 刻度、随 audio.currentTime 更新的红色游标和时间数字；200≤t<280 ms 时显著显示“黄色目标区”，其他时刻明确“上下文”。使用 requestAnimationFrame，暂停/结束/切题停止并清理过期帧。没有拖动 seek、原生 controls、倍速、beep 或额外播放。

醒目说明：不是整段有人声就选；只判断黄色 80 ms；定位不了请选不确定。没有修改现有 schema/version/localStorage key、答案、播放次数或锁定条件，没有读取/清理浏览器草稿，没有默认要求重听。刷新后沿原有流程恢复草稿；若刷新正在播放的片段，原有中断处理仍适用，不伪造成功播放。

修改前 UI 原件与 SHA 存 ui_timing_previous/。原 metrics、配置、94 裁窗、已有状态测试等全部原文件哈希核对未变。没有读来源 WAV、重导音频、运行 builder/validator、改服务器或创建人耳反馈。

实际验证：原 16 项 Node state/单播放器 mock 全通过；新增 8 项时间边界、currentTime 驱动、stale RAF、render/end 清理与原状态核心/key 不变检查通过。结果见 ui_timing_validation.json。不声称实机浏览器播放或真人定位测试已通过。

轻量核验（不触碰音频或草稿）：
```
node /data/coding/sgmse/experiments/p8_i1b_single_listener_materials_20261019/test_player.js
node /data/coding/sgmse/experiments/p8_i1b_single_listener_materials_20261019/test_timing.js
```
Root 可委派 Worker 仅做现有 8766 的 index.html/player.js HTTP GET 字节核验与静态保护核验；不要重新运行 builder 或完整材料 validator，不重启服务，不生成答卷。最终页面仍只有浏览器草稿/JSON 下载，未上传服务器。
