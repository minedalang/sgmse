"""Transcribe the eight clean reference WAVs for human listening context.

The DNS Ins20 SCP text is dummy, so this deliberately produces an ASR
reference rather than a ground-truth transcript or a WER label.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path

import torch
import whisper


ROOT = Path(__file__).resolve().parent
CACHE = Path("/data/.rootcache/whisper")
MODEL_NAME = "tiny.en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = list(csv.DictReader((ROOT / "manifest.csv").open(encoding="utf-8")))
    model_path = CACHE / f"{MODEL_NAME}.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model_sha = sha256(model_path)
    model = whisper.load_model(MODEL_NAME, device=DEVICE, download_root=str(CACHE))
    rows = []
    started = time.time()
    for row in manifest:
        path = Path(row["clean_path"])
        t0 = time.time()
        result = model.transcribe(
            str(path), language="en", task="transcribe", temperature=0,
            beam_size=1, best_of=1, condition_on_previous_text=False,
            fp16=(DEVICE == "cuda"), verbose=False,
        )
        segments = result.get("segments", [])
        rows.append({
            "index": row["index"],
            "speaker": row["speaker"],
            "fileid": row["fileid"],
            "snr_db": row["snr_db"],
            "clean_path": str(path),
            "clean_sha256": sha256(path),
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "avg_logprob": (sum(float(s.get("avg_logprob", 0.0)) for s in segments) / len(segments)) if segments else None,
            "no_speech_prob_max": max((float(s.get("no_speech_prob", 0.0)) for s in segments), default=None),
            "runtime_seconds": time.time() - t0,
        })
        print(json.dumps({"index": row["index"], "text": rows[-1]["text"]}, ensure_ascii=False), flush=True)

    out = {
        "experiment_id": "p8_frontend_backend_audit_20261020",
        "purpose": "clean_audio_asr_reference_for_listening",
        "interpretation": "Automatic reference only; DNS Ins20 SCP text is dummy and this is not ground-truth transcription or a WER label.",
        "model": {"name": MODEL_NAME, "sha256": model_sha, "path": str(model_path), "device": DEVICE, "whisper_version": getattr(whisper, "__version__", "unknown")},
        "decode": {"language": "en", "task": "transcribe", "temperature": 0, "beam_size": 1, "best_of": 1, "condition_on_previous_text": False, "fp16": DEVICE == "cuda"},
        "n": len(rows),
        "runtime_seconds": time.time() - started,
        "rows": rows,
    }
    (ROOT / "clean_asr_reference.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Clean 音频自动识别参考", "", "这是 clean 音频的 Whisper `tiny.en` 自动转写，仅供听感时对照。DNS Ins20 的 `text` 文件为 `dummy`，以下内容不是人工真值。", ""]
    for r in rows:
        lines += [f"## 样本 {r['index']}（speaker {r['speaker']}，SNR {r['snr_db']} dB）", "", r["text"] or "（未识别出文本）", ""]
    (ROOT / "clean_asr_reference.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
