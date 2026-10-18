"""消融推理: 判断前端(frontend)还是后端(SGMSE扩散)导致高SNR语音退化。

--mode frontend_only : 只跑前端, 输出 y_pre
--mode backend_only  : 跳过前端, noisy 直接作为扩散输入 (y_in=noisy, y_cond=noisy)
--mode joint         : 完整链路 (等同 joint_enhance_by_dir.py)

用法示例:
CUDA_VISIBLE_DEVICES=4 python joint_ablation_by_dir.py \
  --ckpt joint_train_logs/bl0pegi4-None/epoch=148-step=300000.ckpt \
  --in_dir  .../with_reverb/noisy --out_dir enhanced/abl_bl0pegi4_frontend_wr \
  --frontend storm --mode frontend_only --use_ema
"""
import os
import glob
import argparse

import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
from sgmse.joint_model import JointSemambaSgmseModel
from sgmse.joint_data_module import JointWaveDataModule


def load_mono(path: str, target_sr: int):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(wav).float()


def _match_length(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    T = ref.size(-1)
    cur = y.size(-1)
    if cur == T:
        return y
    if cur > T:
        start = (cur - T) // 2
        return y[..., start: start + T]
    pad = T - cur
    left = pad // 2
    return F.pad(y, (left, pad - left), mode="constant", value=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["frontend_only", "backend_only", "joint"], required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--train_type", type=str, choices=["mode2", "mode3"], default="mode2")
    p.add_argument("--frontend", type=str, choices=["semamba", "storm", "tfgridnet"], default=None)
    p.add_argument("--frontend_root", type=str, default=None)
    p.add_argument("--frontend_config", type=str, default=None)
    p.add_argument("--frontend_ckpt", type=str, default=None)
    p.add_argument("--sampler_type", type=str, default="pc")
    p.add_argument("--corrector", type=str, default="none")
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--snr", type=float, default=0.5)
    p.add_argument("--use_ema", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    load_kwargs = dict(
        map_location="cpu",
        strict=True,
        data_module_cls=JointWaveDataModule,
        base_dir=".",
        batch_size=1,
        num_workers=0,
        dummy=True,
    )
    if args.frontend is not None:
        load_kwargs["frontend"] = args.frontend
    if args.frontend_root is not None:
        load_kwargs["frontend_root"] = args.frontend_root
    if args.frontend_config is not None:
        load_kwargs["frontend_config"] = args.frontend_config
    if args.frontend_ckpt is not None:
        load_kwargs["frontend_ckpt"] = args.frontend_ckpt

    # PyTorch 2.6 defaults to weights_only=True.  Joint checkpoints contain
    # the DataModule class in their hyperparameters, so use the same explicit
    # full-checkpoint load mode as joint_enhance_by_dir.py.
    model = JointSemambaSgmseModel.load_from_checkpoint(
        args.ckpt, **load_kwargs, weights_only=False
    )
    model.eval(no_ema=not args.use_ema).to(args.device)

    h = model.hparams
    print(f"[Info] train_type={getattr(h, 'train_type', '?')} "
          f"frontend={getattr(h, 'frontend', '?')} "
          f"frontend_trainable={getattr(h, 'frontend_trainable', '?')} "
          f"residual_refine_enable={getattr(h, 'residual_refine_enable', '?')}")
    print(f"[Eval] mode={args.mode} EMA={'on' if args.use_ema else 'off'}")

    wavs = sorted(glob.glob(os.path.join(args.in_dir, "*.wav")))
    print("Found wavs:", len(wavs), flush=True)

    with torch.inference_mode():
        for i, wp in enumerate(wavs):
            y = load_mono(wp, args.sample_rate).to(args.device)
            y = y.unsqueeze(0).unsqueeze(0)  # [1,1,T]

            if args.mode == "backend_only":
                # 跳过前端: noisy 同时作为扩散输入与条件
                y_in = y.squeeze(0)   # [1,T]
                y_cond = y_in
            else:
                norm = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                y_pre = model._frontend_forward(y / norm) * norm
                y_pre = _match_length(y_pre, y)
                if args.mode == "frontend_only":
                    out = y_pre.squeeze().cpu().numpy()
                    out = np.clip(out, -1.0, 1.0)
                    sf.write(os.path.join(args.out_dir, os.path.basename(wp)), out, args.sample_rate)
                    if (i + 1) % 25 == 0:
                        print(f"  {i + 1}/{len(wavs)}", flush=True)
                    continue
                # joint
                if args.train_type == "mode2":
                    y_in, y_cond = y_pre.squeeze(0), y.squeeze(0)
                else:
                    y_in, y_cond = y.squeeze(0), y_pre.squeeze(0)

            enhanced = model.enhance(
                y_in.squeeze(0),
                y_cond=y_cond.squeeze(0),
                sampler_type=args.sampler_type,
                predictor="reverse_diffusion",
                corrector=args.corrector,
                N=args.N,
                snr=args.snr,
            )
            enhanced = np.clip(enhanced, -1.0, 1.0)
            sf.write(os.path.join(args.out_dir, os.path.basename(wp)), enhanced, args.sample_rate)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(wavs)}", flush=True)

    print("Done. Output:", args.out_dir)


if __name__ == "__main__":
    main()
