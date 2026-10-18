"""Create a labeled, same-window listening pack for the P8 three-arm audit.

The source WAVs and generated outputs are never modified.  A fixed centered
12-second window is used for every arm of an utterance so audible differences
are not caused by different crop positions.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.csv"
OUT = ROOT / "listening_pack"
SAMPLE_SECONDS = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    arms = ("y_pre", "off", "router")
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    key_rows = []
    for row in rows:
        index = int(row["index"])
        stem = Path(row["noisy_basename"]).stem
        for arm in arms:
            src = ROOT / "outputs" / arm / f"{index:02d}_{row['noisy_basename']}"
            wav, sr = sf.read(src, always_2d=False)
            if wav.ndim != 1:
                raise ValueError(f"expected mono WAV: {src}")
            target = min(len(wav), SAMPLE_SECONDS * sr)
            start = max(0, (len(wav) - target) // 2)
            clip = wav[start : start + target]
            dst = OUT / f"{index:02d}_{arm}_{stem}_center12s.wav"
            sf.write(dst, clip, sr, subtype="PCM_16")
            key_rows.append(
                {
                    "sample": f"{index:02d}",
                    "speaker": row["speaker"],
                    "fileid": row["fileid"],
                    "arm": arm,
                    "label_cn": {"y_pre": "Y_pre（前端）", "off": "Router-off", "router": "Router"}[arm],
                    "start_seconds": start / sr,
                    "duration_seconds": target / sr,
                    "source_sha256": sha256(src),
                    "clip": str(dst.relative_to(ROOT)),
                }
            )

    with (OUT / "key.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_rows[0].keys())
        writer.writeheader()
        writer.writerows(key_rows)

    guide = """# P8 三臂听感对照包

每个编号有三段同一时间片段：`Y_pre（前端）`、`Router-off`、`Router`。
所有片段都取原始 30 秒语音的中央 12 秒，采样率 16 kHz。

## 只需要听这些现象

- 人声是否突然中断、吞字或少一截；
- 是否出现突然变调、金属声、机械声或不自然的重复；
- 哪个版本整体更容易听清人声；
- 如果三段听起来差不多，直接记为“差不多”。

不需要听懂英文，不需要猜原句，也不需要填写转写。
反馈时只需写类似：`样本03：Router 有明显断续，Y_pre 和 off 差不多`。

`key.csv` 是条件映射和文件校验记录；音频文件名已经标注条件，便于直接播放。
"""
    (OUT / "LISTENING_GUIDE_CN.md").write_text(guide, encoding="utf-8")


if __name__ == "__main__":
    main()
