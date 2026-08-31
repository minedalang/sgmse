import zipfile
from pathlib import Path

import pandas as pd
import wandb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_PATH = "chenwh-tianjin-university-of-science-and-technology/sgmse_joint/bl0pegi4"
OUT_DIR = Path("wandb_export_bl0pegi4")


def safe_filename(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("=", "_")
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_dir = OUT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(RUN_PATH)

    print(f"Run name: {run.name}")
    print(f"Run state: {run.state}")
    print("Downloading full history...")

    rows = []
    for row in run.scan_history(page_size=10000):
        rows.append(row)

    if not rows:
        raise RuntimeError("No history rows found. Check run path or W&B permissions.")

    df = pd.DataFrame(rows)

    if "_step" in df.columns:
        df = df.sort_values("_step")

    all_csv = OUT_DIR / "history_all.csv"
    df.to_csv(all_csv, index=False)
    print(f"Saved full history: {all_csv}")

    # 保存全部 metric 名称
    metric_keys = sorted(df.columns.tolist())
    keys_txt = OUT_DIR / "metric_keys.txt"
    with open(keys_txt, "w", encoding="utf-8") as f:
        for k in metric_keys:
            f.write(k + "\n")
    print(f"Saved metric keys: {keys_txt}")

    # 自动筛选 anchor 相关指标
    anchor_keys = [
        k for k in metric_keys
        if "anchor" in k.lower()
    ]

    # 也把常见 loss 加进去，方便一起看
    extra_keys = [
        k for k in metric_keys
        if any(s in k.lower() for s in ["loss", "pesq", "stoi", "sisdr", "si_sdr", "dnsmos"])
    ]

    selected_keys = sorted(set(anchor_keys + extra_keys))

    if "_step" in df.columns:
        export_cols = ["_step"] + selected_keys
    else:
        export_cols = selected_keys

    export_cols = [c for c in export_cols if c in df.columns]

    anchor_df = df[export_cols].copy()
    anchor_csv = OUT_DIR / "history_anchor.csv"
    anchor_df.to_csv(anchor_csv, index=False)
    print(f"Saved selected history: {anchor_csv}")

    # 绘图：每个 metric 单独一张图
    x_col = "_step" if "_step" in df.columns else None

    for key in selected_keys:
        if key not in df.columns:
            continue

        tmp = df[[x_col, key]].copy() if x_col else df[[key]].copy()
        tmp = tmp.dropna(subset=[key])

        if tmp.empty:
            continue

        # 只画数值型数据
        tmp[key] = pd.to_numeric(tmp[key], errors="coerce")
        tmp = tmp.dropna(subset=[key])

        if tmp.empty:
            continue

        plt.figure(figsize=(8, 4.5))

        if x_col:
            plt.plot(tmp[x_col], tmp[key])
            plt.xlabel(x_col)
        else:
            plt.plot(tmp.index, tmp[key])
            plt.xlabel("index")

        plt.ylabel(key)
        plt.title(key)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = chart_dir / f"{safe_filename(key)}.png"
        plt.savefig(fig_path, dpi=180)
        plt.close()

    print(f"Saved charts to: {chart_dir}")

    # 打包
    zip_path = Path(f"{OUT_DIR}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            zf.write(p, p.relative_to(OUT_DIR.parent))

    print(f"Done. Zip file: {zip_path}")


if __name__ == "__main__":
    main()