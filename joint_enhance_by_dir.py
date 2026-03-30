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

'''
CUDA_VISIBLE_DEVICES=2 python joint_enhance_by_dir.py --ckpt joint_train_logs/dx4g6bi0-jointtrain_storm_frontend_selectortfgrida_sbveedm/last.ckpt --in_dir test_frontend_output --out_dir test_frontend_output/dx4g6bi0_enhanced --train_type mode2 --frontend storm

'''
def load_mono(path: str, target_sr: int):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)  # to mono
    if sr != target_sr:
        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=target_sr)
    wav = torch.from_numpy(wav).float()
    return wav

def _match_length(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 y 的最后一维长度调成 ref 的长度（center crop/pad），y/ref: [B, 1, T]."""
    T = ref.size(-1)
    cur = y.size(-1)
    if cur == T:
        return y
    if cur > T:
        start = (cur - T) // 2
        return y[..., start : start + T]
    pad = T - cur
    left = pad // 2
    right = left + (pad % 2)
    return F.pad(y, (left, right), mode="constant", value=0.0)
def main():
    p = argparse.ArgumentParser()

    # paths
    p.add_argument("--ckpt", type=str, required=True, help="joint lightning .ckpt")
    p.add_argument("--in_dir", type=str, required=True, help="dir of noisy wavs")
    p.add_argument("--out_dir", type=str, required=True, help="dir to write enhanced wavs")

    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--train_type", type=str, choices=["mode2", "mode3"], default="mode2",
                   help="must match training: mode3=SGMSE(noisy | y_pre), mode2=SGMSE(y_pre | noisy)")

    # frontend loader (match your training args)
    p.add_argument("--frontend", type=str, choices=["semamba", "storm", "tfgridnet"], default="semamba")
    p.add_argument("--frontend_root", type=str, required=None)
    p.add_argument("--frontend_config", type=str, required=None)
    p.add_argument("--frontend_ckpt", type=str, default=None)

    # sgmse sampling params
    p.add_argument("--sampler_type", type=str, default="pc")
    p.add_argument("--corrector", type=str, default="ald")
    p.add_argument("--N", type=int, default=30, help="diffusion steps (bigger=better+slower)")
    p.add_argument("--snr", type=float, default=0.5)
    p.add_argument("--use_ema", action="store_true", help="use EMA weights for SGMSE (recommended)")

    # IMPORTANT: these STFT/transform params must match training
    p.add_argument("--n_fft", type=int, default=510)
    p.add_argument("--hop_length", type=int, default=128)
    p.add_argument("--num_frames", type=int, default=256)
    p.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann")
    p.add_argument("--spec_factor", type=float, default=0.15)
    p.add_argument("--spec_abs_exponent", type=float, default=0.5)
    p.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy")
    p.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent")

    # IMPORTANT: these must match training (backbone/sde/loss/etc). 用你训练时的值覆盖默认值！
    p.add_argument("--backbone", type=str, default="ncsnpp_v2")
    p.add_argument("--sde", type=str, default="ouve")
    p.add_argument("--t_eps", type=float, default=0.03)
    p.add_argument("--loss_type", type=str, default="mse")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--weighting", type=str, default="none")
    p.add_argument("--nonlinear_weighting", action="store_true")

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build minimal data_module kwargs (we won't call setup(); only need STFT funcs)

    # Create model (IMPORTANT: JointSemambaSgmseModel expects backbone & sde directly)
    # model = JointSemambaSgmseModel(
    #     backbone=args.backbone,
    #     sde=args.sde,
    #     train_type=args.train_type,

    #     frontend=args.frontend,
    #     frontend_root=args.frontend_root,
    #     frontend_config=args.frontend_config,
    #     frontend_ckpt=args.frontend_ckpt,

    #     frontend_trainable=False,     # 推理不训练
    #     data_module_cls=JointWaveDataModule,

    #     # ---- 以下这些 kwargs 会传给 ScoreModel + datamodule（尽量与训练一致）----
    #     lr=args.lr,
    #     ema_decay=args.ema_decay,
    #     t_eps=args.t_eps,
    #     loss_type=args.loss_type,
    #     sigma_data=args.sigma_data,
    #     weighting=args.weighting,
    #     nonlinear_weighting=args.nonlinear_weighting,

    #     # datamodule 需要的 STFT/transform 参数（你之前的 data_module_kwargs）
    #     **data_module_kwargs,
    # )

    # ckpt = torch.load(args.ckpt, map_location="cpu")
    # state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # if missing:
    #     print("[WARN] missing keys:", len(missing))
    # if unexpected:
    #     print("[WARN] unexpected keys:", len(unexpected))
    # 直接从 joint ckpt 重建模型（会自动带上训练时的 sbve k/c 等超参）
    data_module_kwargs = dict(
        base_dir=".",                 # 关键：补上必填 base_dir（推理时用不到，给个占位）
        batch_size=1,
        num_workers=0,
        dummy=True,                   #如果你的 JointWaveDataModule 支持 dummy 模式，强烈建议打开

    )

    model = JointSemambaSgmseModel.load_from_checkpoint(
        args.ckpt,
        map_location="cpu",
        strict=True,
        frontend=args.frontend,
        frontend_root=args.frontend_root,
        frontend_config=args.frontend_config,
        frontend_ckpt=args.frontend_ckpt,
        data_module_cls=JointWaveDataModule,
        **data_module_kwargs,         #  把 base_dir 等补进去
    )



    model.eval().to(args.device)

    wavs = sorted(glob.glob(os.path.join(args.in_dir, "*.wav")))
    print("Found wavs:", len(wavs))

    with torch.inference_mode():
        for wp in wavs:
            # y = load_mono(wp, args.sample_rate).to(args.device)  # [T]
            # y = y.unsqueeze(0)  # [1, T]

            # # 1) SEMamba pre-enhance
            # y_pre = model._semamba_forward(y)  # [1, T]
            # 2) SGMSE diffusion enhance (conditional)
            # if args.train_type == "mode2":
            #     y_in, y_cond = y_pre, y
            # else:  # mode3
            #     y_in, y_cond = y, y_pre

            y = load_mono(wp, args.sample_rate).to(args.device)   # [T]
            y = y.unsqueeze(0).unsqueeze(0)      # [1,1,T]
            norm = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)   # [1,1,1]
            y_pre = model._frontend_forward(y/norm)* norm   # [1,1,T']
            y_pre = _match_length(y_pre, y)      # 对齐长度（训练里就是这么做的）
            if args.train_type == "mode2":
                y_in, y_cond = y_pre.squeeze(0), y.squeeze(0)      # [1,T], [1,T]
            else:
                y_in, y_cond = y.squeeze(0), y_pre.squeeze(0)      # [1,T], [1,T]
            

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
            out_path = os.path.join(args.out_dir, os.path.basename(wp))
            sf.write(out_path, enhanced, args.sample_rate)

    print("Done. Output:", args.out_dir)


if __name__ == "__main__":
    main()
