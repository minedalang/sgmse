# 单听者局部结构材料制作完成；人耳判断仍待办

本次仅完成已授权的材料制作，不是可辨识性实验成功。固定 47 个区间分别导出 Noisy / Clean 共 94 份原始 PCM16 裁窗；全部为 16 kHz、单声道、480 ms（7680 样本），标记 200–280 ms（80 ms）。无归一化、补零、混音或数值转换。第一阶段 47 Noisy 锁定后才开放另序 47 Clean；同人记忆和期待偏差仍在，不构成独立 gold、跨听者一致性、总体准确率或音素恢复证据。

## 实际执行与验证

唯一生成命令：`python3 -B /data/coding/sgmse/experiments/p8_i1b_single_listener_materials_20261019/build_materials.py --build-once`。生成 PID 54877，耗时 0.172036414 秒，状态 completed，无重试或失败。

一次独立命令：`python3 -B /data/coding/sgmse/experiments/p8_i1b_single_listener_materials_20261019/validate_materials.py`。94/94 WAV 头、固定排序、窗边界、样本字节与 SHA 核验通过；24 份源 WAV 按冻结 allocation SHA 重核通过。公开 trials 仅含 opaque trial_id、stage、audio，页面和脚本未含原 utterance/speaker 身份；不公开目标/对照角色映射。operator_sources.json 是操作者追溯文件，静态服务不提供防窥安全边界。

通过已有 PID 16022 的 127.0.0.1:8766 服务进行 97 次 loopback GET（index / player / trials / 94 WAV），共 1,469,875 bytes，全部与磁盘原件一致，94 份 HTTP WAV 头通过。未播放音频、未 POST、未外网访问、未新建或重启服务。生成 metrics 的 network_bytes=0 仅记录生成阶段；后续本地 GET 独立记录于 http_and_protection_validation.json。

preflight_protection.json 保存最小执行前锁；执行后 24 份 source、5 份代码/config、8 份旧 metrics 及 1 份旧反馈记录共 38 文件 SHA 均未变化。未修改既有科学指标。桥接 PID 39452 与静态服务 PID 16022 均存活；工作线程 provider 已核为 codex_tools_compat。执行记录见 execution_receipt.json。

## 页面与结果边界

入口：http://127.0.0.1:8766/p8_i1b_single_listener_materials_20261019/index.html 。浏览器 localStorage 保存草稿，支持 partial / complete JSON 本地下载，没有服务器答卷上传。用户可按需完成，不作为继续科研的等待门。

Root / Engineer 已报告 Node 16 项、Python 5 项合成检查通过，本 Worker 未重复这些检查。未运行真实浏览器，未验证实际浏览器音频播放或下载行为，未安装浏览器依赖。本次仅核验静态资源与 PCM；没有 autoplay、真实回答或伪造答卷。

人耳回答 **0 / 94**，状态 **unrated_optional_TODO**。模型、ASR、GPU、训练调用均 0；所有新产物限本任务目录且总量低于 2 MiB。长期科学结论及原始实验保持冻结。
