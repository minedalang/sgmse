#!/usr/bin/env python3
"""Read-only CAM++ speaker headroom probe on an existing repeated-speaker split."""
import csv, json, math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path("/data/coding/sgmse")
MANIFEST = ROOT / "experiments/rrsd_cv_repeat32/manifest.csv"
ENCODER = ROOT / "pre_models/wespeaker-voxceleb-campplus/avg_model.pt"
OUT = ROOT / "experiments/speaker_headroom_20261017"
ARMS = {
    "noisy": None,
    "off": ROOT / "experiment_logs/rrsd_cv_repeat32/off",
    "router": ROOT / "experiment_logs/rrsd_cv_repeat32/router",
    "pair": ROOT / "experiment_logs/rrsd_cv_repeat32/pair",
    "proto01": ROOT / "experiment_logs/rrsd_cv_repeat32/proto01",
    "proto01_cw1": ROOT / "experiment_logs/rrsd_cv_repeat32/proto01_cw1",
    "masked": ROOT / "experiment_logs/rrsd_cv_repeat32/masked_pair32",
    "clean": "clean",
}

def norm(w):
    x = torch.from_numpy(np.asarray(w, np.float32).copy()).view(1, -1)
    x = x - x.mean(dim=1, keepdim=True)
    return x / (x.abs().amax(dim=1, keepdim=True) + 1e-6)

def read(path):
    w, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != 16000 or w.ndim != 1:
        raise ValueError(f"bad audio {path}: sr={sr} shape={w.shape}")
    return w[:32000]

def auc_eer(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = labels == 1; neg = ~pos
    # Rank-based AUC, ties receive their average rank.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores)+1)
    auc = (ranks[pos].sum() - pos.sum()*(pos.sum()+1)/2) / max(1, pos.sum()*neg.sum())
    desc = np.argsort(-scores, kind="mergesort")
    y = labels[desc]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    fn = pos.sum() - tp; tn = neg.sum() - fp
    far = fp / max(1, neg.sum()); frr = fn / max(1, pos.sum())
    j = int(np.argmin(np.abs(far-frr)))
    return float(auc), float((far[j]+frr[j])/2), float(far[j]), float(frr[j])

def pair_stats(emb, speakers):
    z = F.normalize(torch.from_numpy(emb).float(), dim=1).numpy()
    sims = z @ z.T
    same, diff, labels, scores = [], [], [], []
    for i in range(len(speakers)):
        for j in range(i+1, len(speakers)):
            is_same = speakers[i] == speakers[j]
            scores.append(float(sims[i,j])); labels.append(int(is_same))
            (same if is_same else diff).append(float(sims[i,j]))
    auc, eer, far, frr = auc_eer(scores, labels)
    return {"same_mean": float(np.mean(same)), "different_mean": float(np.mean(diff)),
            "same_n": len(same), "different_n": len(diff), "auc": auc,
            "eer": eer, "eer_far": far, "eer_frr": frr}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader(MANIFEST.open()))
    speakers = [r["speaker"] for r in manifest]
    from sgmse.speaker_encoders import WeSpeakerCAMPPlusEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WeSpeakerCAMPPlusEncoder(str(ENCODER)).to(device).eval()
    paths = {}
    for arm, root in ARMS.items():
        paths[arm] = []
        for r in manifest:
            if arm == "noisy": p = Path(r["noisy"])
            elif arm == "clean": p = Path(r["clean"])
            else: p = root / r["filename"]
            if not p.is_file(): raise FileNotFoundError(p)
            paths[arm].append(p)
    embeddings = {}
    with torch.inference_mode():
        for arm, ps in paths.items():
            chunks = []
            for start in range(0, len(ps), 4):
                x = torch.cat([norm(read(p)) for p in ps[start:start+4]], dim=0).to(device)
                chunks.append(model(x).float().cpu())
            embeddings[arm] = torch.cat(chunks).numpy()
            print(f"{arm}: {embeddings[arm].shape}", flush=True)
    clean = embeddings["clean"]
    result = {"split": "rrsd_cv_repeat32", "utterances": len(manifest),
              "speakers": len(set(speakers)), "device": device, "max_samples": 32000,
              "arms": {}}
    per_file = []
    for i, r in enumerate(manifest):
        clean_w = read(Path(r["clean"])); noisy_w = read(Path(r["noisy"]))
        noise_w = noisy_w[:len(clean_w)] - clean_w[:len(noisy_w)]
        snr = 10.0 * math.log10((np.mean(clean_w.astype(np.float64)**2) + 1e-12) /
                                 (np.mean(noise_w.astype(np.float64)**2) + 1e-12))
        row = {"filename": r["filename"], "speaker": r["speaker"],
               "snr_db_est": float(snr), "duration_sec": 32000/16000}
        for arm, emb in embeddings.items():
            row[arm + "_clean_cos"] = float(F.cosine_similarity(
                torch.from_numpy(emb[i:i+1]), torch.from_numpy(clean[i:i+1]), dim=1
            ).item())
        per_file.append(row)
    result["per_utterance"] = per_file
    bins = {"<=-15": lambda x: x <= -15, "(-15,-10]": lambda x: -15 < x <= -10,
            "(-10,-5]": lambda x: -10 < x <= -5, ">-5": lambda x: x > -5}
    result["snr_strata"] = {}
    for name, pred in bins.items():
        rows = [r for r in per_file if pred(r["snr_db_est"])]
        result["snr_strata"][name] = {"n": len(rows), "snr_mean": float(np.mean([r["snr_db_est"] for r in rows])) if rows else None,
                                      "paired_clean_cos": {arm: float(np.mean([r[arm+"_clean_cos"] for r in rows])) for arm in embeddings} if rows else {}}
    for arm, emb in embeddings.items():
        cross = F.cosine_similarity(torch.from_numpy(emb), torch.from_numpy(clean), dim=1).numpy()
        result["arms"][arm] = {**pair_stats(emb, speakers),
                                "paired_clean_cos_mean": float(cross.mean()),
                                "paired_clean_cos_median": float(np.median(cross))}
    result["reference"] = {"clean_clean": result["arms"]["clean"]}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Independent CAM++ speaker headroom probe", "",
             "Existing 32-utterance / 16-speaker repeated-speaker CV split; first 32,000 samples; no training or new audio.", "",
             "| arm | paired clean cosine | same-speaker cosine | different-speaker cosine | AUC | EER |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, m in result["arms"].items():
        lines.append(f"| {arm} | {m['paired_clean_cos_mean']:.6f} | {m['same_mean']:.6f} | {m['different_mean']:.6f} | {m['auc']:.6f} | {m['eer']:.6f} |")
    lines += ["", "## Paired clean cosine by estimated input SNR", "",
              "| SNR stratum | n | mean SNR (dB) | noisy | off | router | clean |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in result["snr_strata"].items():
        if m["n"]:
            lines.append(f"| {name} | {m['n']} | {m['snr_mean']:.2f} | {m['paired_clean_cos']['noisy']:.6f} | {m['paired_clean_cos']['off']:.6f} | {m['paired_clean_cos']['router']:.6f} | {m['paired_clean_cos']['clean']:.6f} |")
    lines += ["", "`clean` is the clean-clean reference. Same-speaker and different-speaker pairs use all unordered pairs; each speaker has two utterances.", "", "No dataset or checkpoint was modified."]
    (OUT / "experiment_report.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "metrics.json")

if __name__ == "__main__": main()
