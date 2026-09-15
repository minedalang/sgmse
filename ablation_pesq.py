"""消融结果的 PESQ 统计: 前端 only / 后端 only / 完整链路, 按 SNR 分档对比。

读取 analysis_pesq/per_file_pesq.csv 中已有的 noisy 和 joint PESQ,
新算 4 个消融目录的 PESQ, 合并写回 analysis_pesq/ablation_pesq.csv。
"""
import os
import re
import glob
import numpy as np
import pandas as pd
from soundfile import read
from pesq import pesq

WR = "/home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/datasets/test_set/synthetic/with_reverb"

ABL_DIRS = {
    "pesq_bl0_frontend": "enhanced/abl_bl0pegi4_frontend_wr",
    "pesq_bl0_backend": "enhanced/abl_bl0pegi4_backend_wr",
    "pesq_yqjz_frontend": "enhanced/abl_yqjz1nx1_frontend_wr",
    "pesq_yqjz_backend": "enhanced/abl_yqjz1nx1_backend_wr",
}

df = pd.read_csv("analysis_pesq/per_file_pesq.csv")
df = df[df["dataset"] == "with_reverb"].copy().reset_index(drop=True)

for col, d in ABL_DIRS.items():
    vals = {}
    for wp in sorted(glob.glob(os.path.join(d, "*.wav"))):
        b = os.path.basename(wp)
        fid = re.search(r"(fileid_\d+)", b).group(1)
        x, sr_x = read(f"{WR}/clean/clean_{fid}.wav")
        y, sr_y = read(wp)
        assert sr_x == sr_y == 16000
        n = min(len(x), len(y))
        vals[b] = pesq(16000, x[:n], y[:n], "wb")
    df[col] = df["filename"].map(vals)
    print(f"{col}: {df[col].notna().sum()}/{len(df)} files", flush=True)

df.to_csv("analysis_pesq/ablation_pesq.csv", index=False)

groups = {
    "bl0pegi4": ("pesq_bl0_frontend", "pesq_bl0_backend", "pesq_bl0pegi4_300k_ema"),
    "yqjz1nx1": ("pesq_yqjz_frontend", "pesq_yqjz_backend", "pesq_yqjz1nx1_300k_ema"),
}

bins = [-1, 2, 6, 10, 14, 19]
labels = ["[0,2]", "(2,6]", "(6,10]", "(10,14]", "(14,18]"]
df["snr_bin"] = pd.cut(df["snr"], bins=bins, labels=labels, include_lowest=True)

for model, (fcol, bcol, jcol) in groups.items():
    print(f"\n===== {model} (with_reverb, PESQ) =====")
    print(f"{'snr_bin':10s} {'n':>3s} {'noisy':>6s} {'frontend':>9s} {'backend':>9s} {'joint':>9s}")
    for b, g in df.groupby("snr_bin", observed=True):
        print(f"{str(b):10s} {len(g):>3d} {g['pesq_noisy'].mean():>6.2f} "
              f"{g[fcol].mean():>9.2f} {g[bcol].mean():>9.2f} {g[jcol].mean():>9.2f}")
    print(f"{'ALL':10s} {len(df):>3d} {df['pesq_noisy'].mean():>6.2f} "
          f"{df[fcol].mean():>9.2f} {df[bcol].mean():>9.2f} {df[jcol].mean():>9.2f}")
