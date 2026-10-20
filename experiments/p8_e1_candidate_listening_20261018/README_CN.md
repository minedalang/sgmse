# ⚠️ 已作废：P8-E1 候选窗口听感包（时间轴错误）

> **不要使用本目录的 `listening.html` 或 `clips/` 做试听结论。** 旧脚本把完整 30 秒 `noisy` 当成中央 12 秒片段来裁剪，导致 noisy 实际取到约 9.75–11.75 秒，而 clean/Y_pre/Router-off/Router 对应约 18.75–20.75 秒。该包的 noisy 与其它 arm 不同时间段，之前基于它的内容比较全部无效。请改用 [修复后的对齐听感包](../p8_e1_aligned_listening_20261020/listening.html)。

这是从已有 P8 WAV 中截取的少量人工核实包，不是新推理结果。每个候选包含相同时间段的 `clean / noisy / Y_pre / Router-off / Router`，并额外保留约 0.75 秒上下文。

## 使用方式

1. 先听 `clean`，只把它当作原句提示；不要用它先记住音节后再判断其它音频。
2. 随机播放 `noisy / Y_pre / off / router`，记录能听出的词或音节、吞音位置、是否只是变圆润，以及是否出现新音节。
3. 这是定性核实，不需要逐个量化打分；重点是判断 noisy 是否还保留可辨认的音素线索。

## 候选

| rank | file-id | speaker | 窗口 | 局部 SNR | noisy-clean 相关 |
|---:|---:|---|---|---:|---:|
| 1 | 5243 | reader_10658 | 19.5–20.0 s | −8.03 dB | 0.321 |
| 2 | 4812 | reader_10921 | 12.5–13.0 s | −9.21 dB | 0.328 |
| 7 | 11999 | reader_10927 | 19.5–20.0 s | −16.95 dB | 0.219 |
| 8 | 10720 | reader_10929 | 14.5–15.0 s | −15.41 dB | 0.187 |

排名来自 clean-aware 代理，只用于安排试听优先级；不能解释成音素可恢复概率。

文件在 `clips/`，机器可读清单在 `manifest.json`。对应的 clean 自动转写参考见 `../p8_frontend_backend_audit_20261020/clean_center12s_beam5_reference.md`，其中 sample 00、02、03、04 分别对应上述四个 file-id；它不是人工真值。

候选窗附近的更精确自动词提示见 [clean_timestamp_prompt.md](../p8_e1_clean_timestamp_prompt_20261018/clean_timestamp_prompt.md)：5243=`Canadian by`，4812=`reconstitution of`，11999=`most perfect`，10720=`certainly will`。这些词只用于定位试听。

中文听音提示见 [LISTENING_GUIDE_CN.md](LISTENING_GUIDE_CN.md)。

## 浏览器播放（SSH）

在服务器上执行：

```bash
cd /data/coding/sgmse/experiments/p8_e1_candidate_listening_20261018
/data/coding/sgmse/env/bin/python -m http.server 8765 --bind 127.0.0.1
```

然后在本地 SSH 连接中建立端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 <服务器地址>
```

浏览器打开 `http://127.0.0.1:8765/listening.html`。若你使用 VS Code Remote SSH，也可以直接对 8765 端口做端口转发。
