# 既有 VB 32 格：单听者参照辅助材料已完成，判断待办

复用关闭的 32 格缓存，不重跑科学实验。全部 8 句、4 speaker、2 noise、−18/0 dB，共 32 case；每 case 提供完整原幅度 Clean、Noisy、匿名 A、B。128 个不透明 symlink 入口引用 104 份唯一 PCM（8 Clean、32 Noisy、32 frontend、32 native OFF）。不裁窗、复制、重编码、归一化、滤波或混音。

固定 case hash 排序与 A/B parity，无结果筛选或事后平衡。实际 A=frontend 14 case、A=OFF 18 case，B 对应相反。公开页面/manifest 不含系统身份、speaker/原 basename 或旧指标；发布者原文用于内容参照，可能与真实发音不完全一致。operator_mapping.json 保留追溯映射；静态服务不是防窥安全边界。

## 执行与独立检查

唯一生成命令：`python3 -B /data/coding/sgmse/experiments/p8_vb_cached_content_review_20261019/build_review.py --build-once`。
生成 PID 7604，耗时 0.100404208 秒，completed，无失败或重试。

独立检查仅一次：`python3 -B /data/coding/sgmse/experiments/p8_vb_cached_content_review_20261019/check_materials.py`。128/128 link 对应、104/104 源文件 SHA/WAV header、32 case join、顺序/parity及公开字段通过；PCM 样本解码 0。源文件前后 SHA 一致。材料 metrics SHA `171bb9f017e4bb886c3b4108da75493e72056f16899594dc0c56e0c010f89871` 保持不变。

现有 127.0.0.1:8766 服务 PID 16022 未更改或重启。仅 GET index、player、cases 和首 case 四个 arm，共 7 requests / 440,625 bytes，HTTP 200 且 SHA 与磁盘一致；只下载校验，不播放或提交答案。0 POST、外网、模型、ASR、GPU、训练。

最小保护记录核 8 份旧 metrics、前 94 材料已归档的两份用户 export 与 clarification 源、5 份代码/config及2份源 metadata 均未变化；104 音频源前后 SHA 未变。preflight_protection.json、independent_validation.json、http_and_protection_validation.json 和 execution_receipt.json 保存实际证据，不改原科学 primary metrics。

## 时长与空间

所有入口均为完整原文件；单文件时长 3.1156875–4.61 秒。104 唯一源累计 371.81625 秒；128 入口各完整播放一次累计 457.62 秒（约 7.627 分钟），不含作答或可选第二次播放时间。无需解码 PCM，由原 WAV frame count / 16000 得到。

引用唯一源 11,902,696 bytes；按 128 入口计引用 14,649,472 bytes；128 symlink 自身 14,416 bytes。它们与新常规文件占用分开记录。常规文件最终数见 execution_receipt.json，低于 2 MiB，不通过跟随 symlink 把旧音频当成新复制空间。

## 可选使用与证据边界

入口：http://127.0.0.1:8766/p8_vb_cached_content_review_20261019/index.html 。独立 schema/key，不触碰旧 94 题草稿；先完成 Clean，再听其他来源，内容锁定后可选质量判断；允许分次和 partial JSON 下载。答案只在浏览器 localStorage / 本地导出，未自动上传。

本页是单人、重复句、Clean/文字参照辅助观察，隐藏系统身份不等于内容盲测。人类判断 **0 / 32**；未答保持未评/unknown，不用材料制作宣称 WER、独立 gold、音素恢复或 CAEG 成功。原正式 12 听者 protocol 未实施，未以新页替代。可选待办不阻塞其他研究。

Engineer 已报告 12 项 Node 合成检查通过，本 Worker 未重跑；真实浏览器播放、存储、下载及人耳判断未测试，无生成的真人答卷。
