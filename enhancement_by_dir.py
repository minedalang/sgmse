import glob
import os
import inspect
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from torchaudio.functional import resample as ta_resample
from os.path import join, dirname
from argparse import ArgumentParser
import numpy as np
from functools import partial

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec

'''
python enhancement_for_dir.py --mode3 --pre_enhanced_dir /hom
e/xyf/cwh_project/SEMamba-main/SEMamba-main/dns1_synthetic_snr-15_-5_from_testset_synthetic_with_reverb_enha
nced_by_vdpth  --noisy_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-15_-5_from_testset_
synthetic_with_reverb/noisy   --enhanced_dir vb_test_enhanced_dir_by_250ksbve_edm_crossattn-film-testmode1_b
s4*8_mh1pndou  --ckpt /home/xyf/cwh_project/sgmse-main/logs/mh1pndou-sgmse-sbve-edm-crossattn-film-testmode1
-batchsize4\*8-12-24/epoch\=741-step\=250000.ckpt
'''


def maybe_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """wav: (C,T) -> (C,T')"""
    if orig_sr == target_sr:
        return wav
    return ta_resample(wav, orig_freq=orig_sr, new_freq=target_sr)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')

    # mode2 / mode3 推理模式开关 + 预增强输入目录
    parser.add_argument("--mode2", action="store_true",
                        help="mode2 inference: observation=pre_enhanced wav, condition=noisy wav.")
    parser.add_argument("--mode3", action="store_true",
                        help="mode3 inference: observation=noisy wav, condition=pre_enhanced wav (reverse of mode2).")
    parser.add_argument("--pre_enhanced_dir", type=str, default=None,
                        help="Directory containing PRE-ENHANCED wavs (required if --mode2 or --mode3). "
                             "Must mirror the relative paths of --noisy_dir files.")

    parser.add_argument("--ckpt", type=str, required=True, help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald",
                        help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
    args = parser.parse_args()

    # --- sanity checks ---
    if args.mode2 and args.mode3:
        raise ValueError("Please enable ONLY one of --mode2 or --mode3 (they are mutually exclusive).")
    if (args.mode2 or args.mode3) and (not args.pre_enhanced_dir):
        raise ValueError("You enabled --mode2/--mode3 but did not provide --pre_enhanced_dir")

    # Load score model
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.t_eps = args.t_eps
    model.eval()

    # Get list of noisy files
    noisy_files = []
    noisy_files += sorted(glob.glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(args.noisy_dir, '**', '*.wav'), recursive=True))
    noisy_files += sorted(glob.glob(join(args.noisy_dir, '*.flac')))
    noisy_files += sorted(glob.glob(join(args.noisy_dir, '**', '*.flac'), recursive=True))

    # Check if the model is trained on 48 kHz data
    if model.backbone == 'ncsnpp_48k':
        target_sr = 48000
        pad_mode = "reflection"
    elif model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        # 相对路径（用于保持目录结构一致）
        rel = os.path.relpath(noisy_file, args.noisy_dir)

        # ---- load noisy ----
        y_noisy, sr = load(noisy_file)   # (C,T)
        y_noisy = maybe_resample(y_noisy, sr, target_sr)

        # ---- load pre_enhanced if needed ----
        y_pre = None
        if args.mode2 or args.mode3:
            pre_file = join(args.pre_enhanced_dir, rel)
            print(f"pre_file:{pre_file}")
            if not os.path.exists(pre_file):
                raise FileNotFoundError(f"pre_enhanced file not found: {pre_file}")
            y_pre, sr2 = load(pre_file)
            y_pre = maybe_resample(y_pre, sr2, target_sr)

        # 选择观察wav和条件wav
        # mode2: y_obs=pre,  y_cond=noisy
        # mode3: y_obs=noisy, y_cond=pre
        # default: y_obs=noisy, y_cond=noisy 
        if args.mode2:
            y_obs = y_pre
            y_cond_wav = y_noisy
        elif args.mode3:
            y_obs = y_noisy
            y_cond_wav = y_pre
        else:
            y_obs = y_noisy
            y_cond_wav = y_noisy

        # 长度对齐 
        T_orig = min(y_noisy.size(1), y_obs.size(1))
        if y_cond_wav is not None:
            T_orig = min(T_orig, y_cond_wav.size(1))

        y_noisy = y_noisy[:, :T_orig]
        y_obs = y_obs[:, :T_orig]
        if y_cond_wav is not None:
            y_cond_wav = y_cond_wav[:, :T_orig]

        # ---- 归一化 (跟训练 normalize="noisy" 对齐：用 noisy 的 max) ----
        norm_factor = y_noisy.abs().max().clamp_min(1e-12)
        y_noisy = y_noisy / norm_factor
        y_obs = y_obs / norm_factor
        if y_cond_wav is not None:
            y_cond_wav = y_cond_wav / norm_factor

        # ---- prepare DNN input ----
        Y_obs = torch.unsqueeze(model._forward_transform(model._stft(y_obs.to(args.device))), 0)
        Y_obs = pad_spec(Y_obs, mode=pad_mode)

        Y_cond = None
        if y_cond_wav is not None:
            Y_cond = torch.unsqueeze(model._forward_transform(model._stft(y_cond_wav.to(args.device))), 0)
            Y_cond = pad_spec(Y_cond, mode=pad_mode)
        # 借助score_fn 传递参数y_cond
        score_fn = partial(model, y_cond=None if Y_cond is None else Y_cond.to(args.device))

        # ---- Reverse sampling ----
        if model.sde.__class__.__name__ == 'OUVESDE':
            if args.sampler_type == 'pc':
                sampler = model.get_pc_sampler(
                    predictor='reverse_diffusion',
                    corrector=args.corrector,
                    score_fn=score_fn,
                    y=Y_obs.to(args.device),
                    N=args.N,
                    corrector_steps=args.corrector_steps,
                    snr=args.snr
                )
            elif args.sampler_type == 'ode':
                sampler = model.get_ode_sampler(
                    score_fn=score_fn,
                    y=Y_obs.to(args.device),
                    N=args.N
                )
            else:
                raise ValueError(f"Sampler type {args.sampler_type} not supported")

        elif model.sde.__class__.__name__ == 'SBVESDE':
            sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
            sampler = model.get_sb_sampler(
                sde=model.sde,
                N=args.N,
                y=Y_obs.to(args.device),
                sampler_type=sampler_type,
                score_fn=score_fn
            )
        else:
            raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")

        sample, _ = sampler()

        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        out_path = join(args.enhanced_dir, rel)
        makedirs(dirname(out_path), exist_ok=True)
        write(out_path, x_hat.detach().cpu().numpy(), target_sr)