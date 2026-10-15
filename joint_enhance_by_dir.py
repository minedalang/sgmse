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

python joint_enhance_by_dir.py --ckpt /data/coding/sgmse/joint_train_logs/sb61i0gj_RRSD/epoch\=118-step\=80000.ckpt --in_dir /data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5_testset/noisy --out_dir en
hanced/enahnced_sb61i0gj_300k_-5_testset  --frontend storm --N 30 --rrsd --rrsd_scale 0.5
CUDA_VISIBLE_DEVICES=6,7 python -u joint_train.py \
  --backbone ncsnpp_v2 \
  --sde ouve \
  --loss_type score_matching \
  --num_eval_files 5 \
  --loss_weighting sigma^2 \
  --c_in 1 \
  --c_out 1 \
  --c_skip 0 \
  --network_scaling 1/t \
  --train_type mode2 \
  --mode2_include_noisy_main \
  --nf 128 \
  --ch_mult 1 2 2 2 \
  --num_res_blocks 1 \
  --attn_resolutions 0 \
  --selector_type noise_prior_guidance \
  --selector_hidden_ch 32 \
  --selector_attn_levels 3,2 \
  --selector_oracle_weight 0.0 \
  --noise_prior_init_scale 0.05 \
  --anchor_calib_weight 0.1 \
  --anchor_controller_hidden_ch 64 \
  --anchor_k_max 0.08 \
  --anchor_time_power 1.0 \
  --anchor_loss_weight_c 1.0 \
  --anchor_loss_weight_k 1.0 \
  --anchor_loss_weight_recon 0.1 \
  --dataset joint_scp \
  --train_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15_v2/scp/tr_synthetic/wav.scp \
  --valid_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15_v2/scp/cv_synthetic/wav.scp \
  --test_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15_v2/scp/tt_real_recordings/wav.scp \
  --clean_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15_v2/clean \
  --batch_size 8 \
  --accelerator gpu \
  --devices 2 \
  --precision 32-true \
  --frontend_trainable \
  --frontend storm \
  --frontend_aux_loss spec_mse \
  --weighting_denoiser_to_score 0.5 \
  --storm_nf 128 \
  --storm_ch_mult 1 2 2 2 \
  --storm_num_res_blocks 1 \
  --storm_attn_resolutions 0 \
  --wandb_name trainable_likestorm_noise_prior_guidance_anchorcalib
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

    # frontend loader (match your training args) - 推理时不传，让checkpoint里的值生效
    p.add_argument("--frontend", type=str, choices=["semamba", "storm", "tfgridnet"], default=None)
    p.add_argument("--frontend_root", type=str, default=None)
    p.add_argument("--frontend_config", type=str, default=None)
    p.add_argument("--frontend_ckpt", type=str, default=None)

    # sgmse sampling params
    p.add_argument("--sampler_type", type=str, default="pc")
    p.add_argument("--corrector", type=str, default="none")
    p.add_argument("--N", type=int, default=30, help="diffusion steps (bigger=better+slower)")
    p.add_argument("--snr", type=float, default=0.5)
    p.add_argument("--use_ema", action="store_true", help="use EMA weights for SGMSE (recommended)")
    p.add_argument(
        "--regen_strength",
        type=str,
        default=None,
        help="Adaptive regeneration strength. Use 'auto' for learned selector, or a fixed float in [0,1]. None keeps original sampler.",
    )
    p.add_argument(
        "--anchor_calib",
        action="store_true",
        help="Enable state-aware conditional anchor calibration during PC sampling. Requires a checkpoint trained with --anchor_calib_weight > 0.",
    )
    p.add_argument(
        "--anchor_k_max",
        type=float,
        default=None,
        help="Inference-time override for anchor K_max. If omitted, use the value saved in the checkpoint.",
    )
    p.add_argument(
        "--anchor_scale",
        type=float,
        default=1.0,
        help="Inference-time multiplier on K_max. Try 0.25/0.5/0.75/1.0 for strength ablation.",
    )
    p.add_argument(
        "--anchor_v2",
        action="store_true",
        help="Enable fine-grained band-time Anchor V2. Requires a checkpoint trained with --anchor_v2_weight > 0.",
    )
    p.add_argument(
        "--rrsd",
        action="store_true",
        help="Enable Oracle-distilled Reliability-Routed Score Diffusion during PC sampling.",
    )
    p.add_argument(
        "--rrsd_scale",
        type=float,
        default=1.0,
        help="Inference multiplier on predicted alpha routing coefficients; 1.0 uses the trained router as-is.",
    )
    p.add_argument(
        "--anchor_v2_a_max",
        type=float,
        default=None,
        help="Optional inference override for the Y_pre-x0_base gate cap.",
    )
    p.add_argument(
        "--anchor_v2_b_max",
        type=float,
        default=None,
        help="Optional inference override for the Y_noisy-Y_pre gate cap.",
    )
    p.add_argument(
        "--anchor_v2_scale",
        type=float,
        default=1.0,
        help="Inference multiplier for both Anchor V2 gate caps.",
    )
    p.add_argument(
        "--anchor_v2_single_shot",
        action="store_true",
        help="Apply Anchor V2 at exactly one predictor step instead of every reverse step.",
    )
    p.add_argument(
        "--anchor_v2_single_shot_ratio",
        type=float,
        default=0.5,
        help="Reverse progress in [0,1] for Single-shot V2: 0=first/high-noise step, 1=last/low-noise step.",
    )
    p.add_argument(
        "--anchor_v2_single_shot_step",
        type=int,
        default=None,
        help="Optional 1-based predictor step for Single-shot V2; overrides --anchor_v2_single_shot_ratio.",
    )

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
    data_module_kwargs = dict(
        base_dir=".",                 # 关键：补上必填 base_dir（推理时用不到，给个占位）
        batch_size=1,
        num_workers=0,
        dummy=True,                   #如果你的 JointWaveDataModule 支持 dummy 模式，强烈建议打开
    )

    # 推理时不传 frontend 相关参数，让 checkpoint 里的值生效
    # checkpoint 会自带 frontend=storm 和 frontend_ckpt 等信息
    load_kwargs = dict(
        map_location="cpu",
        strict=True,
        data_module_cls=JointWaveDataModule,
        **data_module_kwargs,
    )
    
    # 只有当用户明确指定了 frontend 参数时才覆盖
    # 否则使用 checkpoint 中保存的值
    if args.frontend is not None:
        load_kwargs["frontend"] = args.frontend
    if args.frontend_root is not None:
        load_kwargs["frontend_root"] = args.frontend_root
    if args.frontend_config is not None:
        load_kwargs["frontend_config"] = args.frontend_config
    if args.frontend_ckpt is not None:
        load_kwargs["frontend_ckpt"] = args.frontend_ckpt

    model = JointSemambaSgmseModel.load_from_checkpoint(
        args.ckpt,
        **load_kwargs,
         weights_only=False,
    )

    model.eval(no_ema=not args.use_ema).to(args.device)
    print(f"[Eval] backend EMA {'enabled' if args.use_ema else 'disabled'}")

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
                regen_strength=args.regen_strength,
                anchor_calib=args.anchor_calib,
                anchor_k_max=args.anchor_k_max,
                anchor_scale=args.anchor_scale,
                anchor_v2=args.anchor_v2,
                rrsd=args.rrsd,
                rrsd_scale=args.rrsd_scale,
                anchor_v2_a_max=args.anchor_v2_a_max,
                anchor_v2_b_max=args.anchor_v2_b_max,
                anchor_v2_scale=args.anchor_v2_scale,
                anchor_v2_single_shot=args.anchor_v2_single_shot,
                anchor_v2_single_shot_ratio=args.anchor_v2_single_shot_ratio,
                anchor_v2_single_shot_step=args.anchor_v2_single_shot_step,
            )
            
            enhanced = np.clip(enhanced, -1.0, 1.0)
            out_path = os.path.join(args.out_dir, os.path.basename(wp))
            sf.write(out_path, enhanced, args.sample_rate)

    print("Done. Output:", args.out_dir)


if __name__ == "__main__":
    main()
