import glob
import os
import inspect
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec


def call_with_supported_kwargs(fn, **kwargs):
    """只传入 fn 支持的参数，避免你 model 里还没加 y_cond 时直接炸"""
    sig = inspect.signature(fn)
    filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filt)
def read_kaldi_scp(path):
    """scp 每行: <uttid> <wav_path>"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            utt, wav = line.split(maxsplit=1)
            items.append((utt, wav))
    return items

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_scp", type=str, required=True, help='Kaldi scp: <uttid> <noisy_wav_path>')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
     # mode2 推理模式开关 + 预增强输入目录
    parser.add_argument("--mode2", action="store_true",
                        help="Enable mode2 inference: observation=pre_enhanced wav, condition=noisy wav.")
    parser.add_argument("--pre_enhanced_dir", type=str, default=None,
                        help="Directory containing PRE-ENHANCED wavs (required if --mode2). "
                             "Must mirror the relative paths of --test_dir files.")
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
    args = parser.parse_args()
    if args.mode2 and not args.pre_enhanced_dir:
        raise ValueError("You enabled --mode2 but did not provide --pre_enhanced_dir")

    # Load score model 
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.t_eps = args.t_eps
    model.eval()

   # Get list of noisy files from scp
    noisy_items = read_kaldi_scp(args.test_scp)

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
    for utt, noisy_file in tqdm(noisy_items):
        # 用 uttid 作为输出名（也用于 mode2 时在 pre_enhanced_dir 下找同名文件）
        ext = os.path.splitext(noisy_file)[1]
        if ext == "":
            ext = ".wav"
        rel = f"{utt}{ext}"   # 例如: p232_001.wav

         # ---- load noisy ----
        y_noisy, sr = load(noisy_file)
        if sr != target_sr:
            y_noisy = torch.tensor(resample(y_noisy.numpy(), orig_sr=sr, target_sr=target_sr))

         # mode2模式: 观测是pre_enhahnced_wav
        if args.mode2:
            pre_file = join(args.pre_enhanced_dir, rel)
            if not os.path.exists(pre_file):
                raise FileNotFoundError(f"pre_enhanced file not found: {pre_file}")
            y_obs, sr2 = load(pre_file)
            if sr2 != target_sr:
                y_obs = torch.tensor(resample(y_obs.numpy(), orig_sr=sr2, target_sr=target_sr))
        else:
            y_obs = y_noisy  # 默认推理：观测就是 noisy

        # ---- align length ----
        T_orig = min(y_noisy.size(1), y_obs.size(1))
        y_noisy = y_noisy[:, :T_orig]
        y_obs = y_obs[:, :T_orig]

        # ---- normalize (跟训练 normalize="noisy" 对齐：用 noisy 的 max) ----
        norm_factor = y_noisy.abs().max().clamp_min(1e-12)
        y_noisy = y_noisy / norm_factor
        y_obs = y_obs / norm_factor
        
        # ---- prepare DNN input ----
        # observation
        Y_obs = torch.unsqueeze(model._forward_transform(model._stft(y_obs.to(args.device))), 0)
        Y_obs = pad_spec(Y_obs, mode=pad_mode)
        # 两种推理模式的条件都是Noisy
        Y_cond = torch.unsqueeze(model._forward_transform(model._stft(y_noisy.to(args.device))), 0)
        Y_cond = pad_spec(Y_cond, mode=pad_mode)
           
         # ---- Reverse sampling ----
        if model.sde.__class__.__name__ == 'OUVESDE':
            if args.sampler_type == 'pc':
                sampler = call_with_supported_kwargs(
                    model.get_pc_sampler,
                    predictor='reverse_diffusion',
                    corrector=args.corrector,
                    y=Y_obs.to(args.device),
                    y_cond=None if Y_cond is None else Y_cond.to(args.device),
                    N=args.N,
                    corrector_steps=args.corrector_steps,
                    snr=args.snr
                )
            elif args.sampler_type == 'ode':
                sampler = call_with_supported_kwargs(
                    model.get_ode_sampler,
                    y=Y_obs.to(args.device),
                    y_cond=None if Y_cond is None else Y_cond.to(args.device),
                    N=args.N
                )
            else:
                raise ValueError(f"Sampler type {args.sampler_type} not supported")

        elif model.sde.__class__.__name__ == 'SBVESDE':
            sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
            sampler = call_with_supported_kwargs(
                model.get_sb_sampler,
                sde=model.sde,
                y=Y_obs.to(args.device),
                y_cond=None if Y_cond is None else Y_cond.to(args.device),
                sampler_type=sampler_type
            )
        else:
            raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")

        sample, _ = sampler()

        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        makedirs(dirname(join(args.enhanced_dir, rel)), exist_ok=True)
        write(join(args.enhanced_dir, rel), x_hat.cpu().numpy(), target_sr)

