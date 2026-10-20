#!/usr/bin/env python3
"""Read-only diagnostic of simple perceptual representations.

The script reuses existing aligned WAV outputs and existing utterance metrics.
It writes only compact CSV/JSON artifacts; no audio is copied or modified.
"""
from __future__ import annotations
import argparse, csv, json, math
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable

import librosa
import numpy as np
import soundfile as sf
from scipy import signal, stats

SR = 16000
ARMS = ("noisy", "off", "router", "cut005", "selective")
RANK_ARMS = ("router", "cut005", "selective")
Q_METRICS = ("pesq", "stoi", "si_sdr", "dnsmos_sig", "dnsmos_ovrl", "dnsmos_bak")
DIST_METRICS = (
    "wave_mse", "complex_stft_rel_mse", "log_mel_l1_db", "erb_l1_db",
    "mrstft_l1_db", "mrstft_sc", "envelope_rel_l1", "modulation_l1_db",
)


def read_mono(path: Path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if int(sr) != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}: {path}")
    return x


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    return librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop,
                         win_length=n_fft, window="hann", center=True,
                         pad_mode="reflect")


def erb_filterbank(sr: int, n_fft: int, n_bands: int = 40,
                   fmin: float = 50.0, fmax: float = 7600.0) -> np.ndarray:
    def rate(f):
        return 21.4 * np.log10(1.0 + 0.00437 * f)
    def inv(r):
        return (10.0 ** (r / 21.4) - 1.0) / 0.00437
    e = np.linspace(rate(fmin), rate(fmax), n_bands + 2)
    edges = inv(e)
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for i in range(n_bands):
        lo, mid, hi = edges[i:i + 3]
        left = (freqs - lo) / max(mid - lo, 1e-8)
        right = (hi - freqs) / max(hi - mid, 1e-8)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
    return fb


def log_spectral(S: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.abs(S) ** 2, 1e-10))


def log_band(S: np.ndarray, fb: np.ndarray) -> np.ndarray:
    p = np.abs(S) ** 2
    return 10.0 * np.log10(np.maximum(fb @ p, 1e-10))


def envelope_and_modulation(x: np.ndarray):
    # Envelope is low-pass smoothed then represented at 100 Hz. This avoids
    # making the distance a sample-level waveform proxy again.
    env = np.abs(signal.hilbert(x.astype(np.float64))).astype(np.float32)
    sos = signal.butter(4, 40.0, btype="lowpass", fs=SR, output="sos")
    env = signal.sosfiltfilt(sos, env).astype(np.float32)
    env100 = signal.resample_poly(env, 1, SR // 100).astype(np.float32)
    env100 = np.maximum(env100, 0.0)
    mod = np.abs(np.fft.rfft(env100 - env100.mean())).astype(np.float32)
    return env100, mod


def feature_bundle(x: np.ndarray, clean: np.ndarray) -> Dict[str, float]:
    n = min(len(x), len(clean))
    x = fit_length(x[:n], n)
    c = fit_length(clean[:n], n)
    c_rms = float(np.sqrt(np.mean(c * c) + 1e-12))
    out: Dict[str, float] = {
        "wave_mse": float(np.mean((x - c) ** 2) / (np.mean(c * c) + 1e-12)),
    }
    Sx = stft(x); Sc = stft(c)
    # The same input length gives matching frame shapes. Center-padding can
    # differ by one frame for unusual lengths, so crop explicitly.
    F = min(Sx.shape[0], Sc.shape[0]); T = min(Sx.shape[1], Sc.shape[1])
    Sx = Sx[:F, :T]; Sc = Sc[:F, :T]
    out["complex_stft_rel_mse"] = float(np.mean(np.abs(Sx - Sc) ** 2) /
                                       (np.mean(np.abs(Sc) ** 2) + 1e-12))
    mel_fb = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=80,
                                 fmin=20.0, fmax=7600.0, htk=False)
    lmx = log_band(Sx, mel_fb); lmc = log_band(Sc, mel_fb)
    out["log_mel_l1_db"] = float(np.mean(np.abs(lmx - lmc)))
    erb_fb = erb_filterbank(SR, 1024, n_bands=40)
    erbx = log_band(Sx, erb_fb); erbc = log_band(Sc, erb_fb)
    out["erb_l1_db"] = float(np.mean(np.abs(erbx - erbc)))
    # Multi-resolution magnitude spectral distance.
    l1s, scs = [], []
    for nf in (512, 1024, 2048):
        hop = nf // 4
        X = np.abs(stft(x, nf, hop)); C = np.abs(stft(c, nf, hop))
        ff = min(X.shape[0], C.shape[0]); tt = min(X.shape[1], C.shape[1])
        X = X[:ff, :tt]; C = C[:ff, :tt]
        l1s.append(float(np.mean(np.abs(20.0 * np.log10(X + 1e-5) -
                                  20.0 * np.log10(C + 1e-5)))) )
        scs.append(float(np.linalg.norm(X - C) / (np.linalg.norm(C) + 1e-12)))
    out["mrstft_l1_db"] = float(np.mean(l1s))
    out["mrstft_sc"] = float(np.mean(scs))
    ex, mx = envelope_and_modulation(x); ec, mc = envelope_and_modulation(c)
    m = min(len(ex), len(ec)); ex = ex[:m]; ec = ec[:m]
    out["envelope_rel_l1"] = float(np.mean(np.abs(ex - ec)) /
                                  (np.mean(ec) + 1e-8))
    mm = min(len(mx), len(mc)); mx = mx[:mm]; mc = mc[:mm]
    out["modulation_l1_db"] = float(np.mean(np.abs(20.0 * np.log10(mx + 1e-5) -
                                                 20.0 * np.log10(mc + 1e-5))))
    return out


def estimate_lag(x: np.ndarray, c: np.ndarray, max_lag: int = 400) -> int:
    # Downsample for a cheap alignment audit; report lag in original samples.
    ds = 4
    x = x[::ds].astype(np.float64); c = c[::ds].astype(np.float64)
    x -= x.mean(); c -= c.mean()
    corr = signal.correlate(x, c, mode="full", method="fft")
    center = len(c) - 1
    lo = max(0, center - max_lag // ds); hi = min(len(corr), center + max_lag // ds + 1)
    lag_ds = int(np.argmax(corr[lo:hi]) + lo - center)
    return lag_ds * ds


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est)); ref = ref[:n].astype(np.float64); est = est[:n].astype(np.float64)
    ref -= ref.mean(); est -= est.mean()
    p = np.dot(est, ref) / (np.dot(ref, ref) + 1e-10) * ref
    q = est - p
    return float(10.0 * np.log10((np.dot(p, p) + 1e-10) / (np.dot(q, q) + 1e-10)))


def load_quality(root: Path):
    """Load existing paired metrics and existing DNSMOS; no evaluator run."""
    q = {}
    def put(path, arm, names):
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                name = Path(r.get("filename", r.get("utt", ""))).name
                q.setdefault(name, {}).setdefault(arm, {})
                for out, src in names.items():
                    if src in r and r[src] not in ("", None):
                        try: q[name][arm][out] = float(r[src])
                        except ValueError: pass
    paired = root / "rrsd_tcut_screen" / "paired_metrics.csv"
    put(paired, "off", {"pesq":"off_pesq", "stoi":"off_stoi", "si_sdr":"off_si_sdr"})
    put(paired, "router", {"pesq":"router_pesq", "stoi":"router_stoi", "si_sdr":"router_si_sdr"})
    put(paired, "cut005", {"pesq":"cut005_pesq", "stoi":"cut005_stoi", "si_sdr":"cut005_si_sdr"})
    selective_pair = root / "rrsd_selective_screen" / "metrics" / "paired_metrics.csv"
    put(selective_pair, "selective", {"pesq":"selective_pesq", "stoi":"selective_stoi", "si_sdr":"selective_si_sdr"})
    put(root / "rrsd_selective_screen" / "router_dnsmos_metrics.csv", "router",
        {"dnsmos_p808":"DNSMOS_P808_e", "dnsmos_sig":"DNSMOS_SIG_e", "dnsmos_bak":"DNSMOS_BAK_e", "dnsmos_ovrl":"DNSMOS_OVRL_e"})
    put(root / "rrsd_tcut_screen" / "router" / "metrics" / "dnsmos_batch_metrics.csv", "router_batch",
        {"dnsmos_p808":"DNSMOS_P808_e", "dnsmos_sig":"DNSMOS_SIG_e", "dnsmos_bak":"DNSMOS_BAK_e", "dnsmos_ovrl":"DNSMOS_OVRL_e"})
    put(root / "rrsd_tcut_screen" / "cut005" / "metrics" / "dnsmos_batch_metrics.csv", "cut005",
        {"dnsmos_p808":"DNSMOS_P808_e", "dnsmos_sig":"DNSMOS_SIG_e", "dnsmos_bak":"DNSMOS_BAK_e", "dnsmos_ovrl":"DNSMOS_OVRL_e"})
    put(root / "rrsd_selective_screen" / "selective_dnsmos_metrics.csv", "selective",
        {"dnsmos_p808":"DNSMOS_P808_e", "dnsmos_sig":"DNSMOS_SIG_e", "dnsmos_bak":"DNSMOS_BAK_e", "dnsmos_ovrl":"DNSMOS_OVRL_e"})
    put(root / "rrsd_selective_screen" / "selective_dnsmos_metrics.csv", "noisy",
        {"pesq":"PESQ_n", "stoi":"STOI_n", "si_sdr":"SI-SDR_n"})
    # router_batch is a tiny independent evaluator duplicate; prefer the paired
    # router DNSMOS evaluator above and retain only one set in output.
    for name, arms in q.items():
        if "router" not in arms and "router_batch" in arms:
            arms["router"] = arms.pop("router_batch")
        else:
            arms.pop("router_batch", None)
    return q


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 20000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0: return [float("nan"), float("nan")]
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def rank_summary(rows, reps=20000, seed=2026):
    rng = np.random.default_rng(seed)
    # Nested map filename -> arm -> row.
    by = {}
    for r in rows: by.setdefault(r["filename"], {})[r["arm"]] = r
    out = {}
    for d in DIST_METRICS:
        out[d] = {}
        for qm in Q_METRICS:
            per_rho=[]; per_sign=[]; pair_total=[]; pair_good=[]
            for fn, arms in by.items():
                candidate_arms = RANK_ARMS if qm.startswith("dnsmos_") else ARMS
                use=[a for a in candidate_arms if a in arms and arms[a].get(qm) is not None]
                if len(use) < 2: continue
                ds=np.array([float(arms[a][d]) for a in use]); qs=np.array([float(arms[a][qm]) for a in use])
                rr=stats.spearmanr(-ds, qs).statistic
                if np.isfinite(rr): per_rho.append(float(rr))
                good=tot=0
                for i,j in combinations(range(len(use)),2):
                    qdiff=qs[i]-qs[j]; ddiff=ds[i]-ds[j]
                    if abs(qdiff)<1e-12 or abs(ddiff)<1e-12: continue
                    tot += 1
                    if np.sign(qdiff)==-np.sign(ddiff): good += 1
                if tot: per_sign.append(good/tot); pair_total.append(tot); pair_good.append(good)
            rho=np.array(per_rho,float); sig=np.array(per_sign,float)
            out[d][qm]={
                "utterances":int(len(rho)),
                "mean_spearman_quality_vs_negative_distance":float(np.nanmean(rho)) if len(rho) else float("nan"),
                "spearman_ci95":bootstrap_ci(rho,rng,reps) if len(rho) else [float("nan")]*2,
                "pairwise_rank_agreement":float(np.sum(pair_good)/max(np.sum(pair_total),1)),
                "utterance_agreement_ci95":bootstrap_ci(sig,rng,reps) if len(sig) else [float("nan")]*2,
            }
    return out


def sanity(root: Path, rows, smoke_n: int):
    checks=[]
    for row in rows[:smoke_n]:
        cp=root/"rrsd_tcut_screen"/"clean"/row["filename"]
        c=read_mono(cp); cr=np.sqrt(np.mean(c*c)+1e-12)
        variants={"clean_gain_0.8":0.8*c,"clean_gain_1.2":1.2*c,
                  "clean_shift_80":np.concatenate([np.zeros(80,np.float32),c[:-80]])}
        for name,x in variants.items():
            f=feature_bundle(x,c); checks.append({"filename":row["filename"],"variant":name,**f})
    out={"note":"clean-only perturbation sanity; gain/shift are not model outputs",
         "rows":checks}
    for d in DIST_METRICS:
        vals=[x[d] for x in checks if np.isfinite(x[d])]
        out.setdefault("mean_by_variant",{})
        for v in ("clean_gain_0.8","clean_gain_1.2","clean_shift_80"):
            vv=[x[d] for x in checks if x["variant"]==v]
            out["mean_by_variant"].setdefault(v,{})[d]=float(np.mean(vv)) if vv else float("nan")
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="experiment_logs")
    p.add_argument("--manifest",default="experiment_logs/rrsd_tcut_screen/manifest.csv")
    p.add_argument("--out-dir",default="experiments/perceptual_representation_probe_20261017")
    p.add_argument("--max-files",type=int,default=32)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--bootstrap",type=int,default=20000)
    args=p.parse_args(); root=Path(args.root); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    with open(args.manifest,newline="",encoding="utf-8") as f: manifest=list(csv.DictReader(f))[:args.max_files]
    q=load_quality(root)
    dirs={"clean":root/"rrsd_tcut_screen"/"clean", "noisy":root/"rrsd_tcut_screen"/"noisy", "off":root/"rrsd_tcut_screen"/"off", "router":root/"rrsd_tcut_screen"/"router", "cut005":root/"rrsd_tcut_screen"/"cut005", "selective":root/"rrsd_selective_screen"/"selective"}
    rows=[]
    for idx,m in enumerate(manifest):
        fn=m["filename"]; clean=read_mono(dirs["clean"]/fn)
        for arm in ARMS:
            x=read_mono(dirs[arm]/fn); n=min(len(x),len(clean)); x=x[:n]; c=clean[:n]
            f=feature_bundle(x,c); rec={"filename":fn,"uttid":m.get("uttid",""),"arm":arm,"n_samples":n,
                 "rms_ratio":float(np.sqrt(np.mean(x*x)+1e-12)/(np.sqrt(np.mean(c*c)+1e-12))),
                 "estimated_lag_samples":estimate_lag(x,c),**f}
            for qm in Q_METRICS:
                rec[qm]=q.get(fn,{}).get(arm,{}).get(qm)
            rows.append(rec)
        print(f"{idx+1}/{len(manifest)} {fn}",flush=True)
    fields=list(rows[0].keys())
    with (out/("smoke_per_file.csv" if args.smoke else "per_file.csv")).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    rs=rank_summary(rows,args.bootstrap,2026)
    # Overall descriptive feature/quality means by arm.
    descriptive={}
    for arm in ARMS:
        rr=[r for r in rows if r["arm"]==arm]; descriptive[arm]={}
        for k in DIST_METRICS+Q_METRICS:
            vv=[r[k] for r in rr if r[k] is not None and np.isfinite(r[k])]
            if vv: descriptive[arm][k]=float(np.mean(vv))
    result={"metadata":{"n_files":len(manifest),"arms":list(ARMS),"ranking_arms":list(RANK_ARMS),"quality_sources":["rrsd_tcut_screen/paired_metrics.csv","rrsd_selective_screen/metrics/paired_metrics.csv","rrsd_selective_screen/router_dnsmos_metrics.csv","rrsd_tcut_screen/cut005/metrics/dnsmos_batch_metrics.csv","rrsd_selective_screen/selective_dnsmos_metrics.csv"],"bootstrap":args.bootstrap,"bootstrap_seed":2026,"smoke":bool(args.smoke)},"descriptive_means":descriptive,"rank_summary":rs,"sanity":sanity(root,manifest,min(4,len(manifest)))}
    with (out/("smoke_metrics.json" if args.smoke else "metrics.json")).open("w",encoding="utf-8") as f: json.dump(result,f,indent=2,allow_nan=True)
    print(json.dumps({"n_files":len(manifest),"smoke":args.smoke,"rank_keys":len(rs)},indent=2))

if __name__=="__main__": main()
