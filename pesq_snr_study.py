"""对比两个 joint 权重在低 SNR (v2) 与高 SNR 含混响 (with_reverb) 测试集上的 PESQ 收益。

输出:
  analysis_pesq/per_file_pesq.csv   -- 每个文件: dataset, filename, snr, pesq_noisy, pesq_<model...>
  终端打印: 总体均值 + 按 SNR 分档的 (enhanced - noisy) PESQ 收益
"""
import os
import re
import glob
import numpy as np
import pandas as pd
from soundfile import read
from pesq import pesq

V2 = "/home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15_v2_testset"
WR = "/home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/datasets/test_set/synthetic/with_reverb"

SETS = {
    "v2_lowsnr": {"noisy": f"{V2}/noisy", "clean": f"{V2}/clean"},
    "with_reverb": {"noisy": f"{WR}/noisy", "clean": f"{WR}/clean"},
}

MODELS = {
    "bl0pegi4_300k_ema": {
        "v2_lowsnr": "enhanced/enahnced_bl0pegi4_300k_useema",
        "with_reverb": "enhanced/enahnced_bl0pegi4_300k_useema_with_reverb",
    },
    "yqjz1nx1_300k_ema": {
        "v2_lowsnr": "enhanced/enahnced_yqjz1nx1_300k_useema",
        "with_reverb": "enhanced/enahnced_yqjz1nx1_300k_useema_with_reverb",
    },
}

# 已有 _results.csv 的直接复用, 不重复算 PESQ
EXISTING_CSV = {
    ("bl0pegi4_300k_ema", "v2_lowsnr"): "enhanced/enahnced_bl0pegi4_300k_useema/_results.csv",
    ("bl0pegi4_300k_ema", "with_reverb"): "enhanced/enahnced_bl0pegi4_300k_useema_with_reverb/_results.csv",
    ("yqjz1nx1_300k_ema", "with_reverb"): "enhanced/enahnced_yqjz1nx1_300k_useema_with_reverb/_results.csv",
}


def fileid(name):
    return re.search(r"(fileid_\d+)", name).group(1)


def snr_of(name):
    return float(re.search(r"_snr(-?[\d.]+)_", name).group(1))


def pesq_of(clean_path, test_path):
    x, sr_x = read(clean_path)
    y, sr_y = read(test_path)
    assert sr_x == sr_y == 16000
    n = min(len(x), len(y))
    return pesq(16000, x[:n], y[:n], "wb")


rows = []
for ds, paths in SETS.items():
    wavs = sorted(glob.glob(os.path.join(paths["noisy"], "*.wav")))
    print(f"[{ds}] {len(wavs)} files", flush=True)
    for i, wp in enumerate(wavs):
        b = os.path.basename(wp)
        cp = os.path.join(paths["clean"], f"clean_{fileid(b)}.wav")
        row = {"dataset": ds, "filename": b, "snr": snr_of(b)}
        row["pesq_noisy"] = pesq_of(cp, wp)
        for model, dirs in MODELS.items():
            csv_path = EXISTING_CSV.get((model, ds))
            if csv_path:
                continue  # 后面统一从 csv 填
            ep = os.path.join(dirs[ds], b)
            row[f"pesq_{model}"] = pesq_of(cp, ep)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(wavs)}", flush=True)

df = pd.DataFrame(rows)

# 从已有 _results.csv 补充
for (model, ds), csv_path in EXISTING_CSV.items():
    sub = pd.read_csv(csv_path)[["filename", "pesq"]].rename(columns={"pesq": f"pesq_{model}"})
    mask = df["dataset"] == ds
    df.loc[mask, f"pesq_{model}"] = df.loc[mask, "filename"].map(sub.set_index("filename")[f"pesq_{model}"])

os.makedirs("analysis_pesq", exist_ok=True)
df.to_csv("analysis_pesq/per_file_pesq.csv", index=False)

model_cols = [c for c in df.columns if c.startswith("pesq_") and c != "pesq_noisy"]

print("\n===== 总体 PESQ =====")
for ds in SETS:
    sub = df[df["dataset"] == ds]
    line = f"{ds:12s} noisy={sub['pesq_noisy'].mean():.2f}"
    for c in model_cols:
        line += f"  {c.replace('pesq_', '')}={sub[c].mean():.2f} (Δ{sub[c].mean() - sub['pesq_noisy'].mean():+.2f})"
    print(line)

print("\n===== 按 SNR 分档: PESQ 均值 (Δ = enhanced - noisy) =====")
for ds in SETS:
    sub = df[df["dataset"] == ds].copy()
    if ds == "v2_lowsnr":
        bins = [-21, -19, -17, -15]
        labels = ["[-20,-19)", "[-19,-17)", "[-17,-15]"]
    else:
        bins = [-1, 2, 6, 10, 14, 19]
        labels = ["[0,2]", "(2,6]", "(6,10]", "(10,14]", "(14,18]"]
    sub["snr_bin"] = pd.cut(sub["snr"], bins=bins, labels=labels, include_lowest=True)
    print(f"\n--- {ds} ---")
    print(f"{'snr_bin':12s} {'n':>4s} {'noisy':>6s}" + "".join(f"  {c.replace('pesq_', ''):>22s}" for c in model_cols))
    for b, g in sub.groupby("snr_bin", observed=True):
        cells = f"{str(b):12s} {len(g):>4d} {g['pesq_noisy'].mean():>6.2f}"
        for c in model_cols:
            cells += f"  {g[c].mean():>10.2f} (Δ{g[c].mean() - g['pesq_noisy'].mean():+6.2f})"
        print(cells)

print("\nDone. per-file 结果: analysis_pesq/per_file_pesq.csv")
