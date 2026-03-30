"""
测试单独预训练的前端权重 (TFGridNet 或 Storm)
用法:
  # 测试 TFGridNet 前端 (单音频):
  python test_frontend_only.py --frontend tfgridnet --ckpt frontend_pretrain_logs/5bd3pa64-tfgridnet_pretrain_3-20/last.ckpt --test_wav path/to/audio.wav
  
  # 测试 Storm 前端 (单音频):
  python test_frontend_only.py --frontend storm --ckpt frontend_pretrain_logs/g9qnaxy3-storm_lite_frontend_pretrain/last.ckpt --test_wav path/to/audio.wav

  # 批量测试整个目录:
  python test_frontend_only.py --frontend storm --ckpt frontend_pretrain_logs/g9qnaxy3-storm_lite_frontend_pretrain/epoch=74-step=150000.ckpt --in_dir /path/to/noisy_dir --out_dir output_dir
"""

import argparse
import os
import glob
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf

from sgmse.frontends import build_frontend


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr)
    return torch.from_numpy(wav).float()


def save_audio(wav, path, sr=16000):
    # 用 soundfile 保存更稳定
    sf.write(path, wav.numpy(), sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=str, default="storm", choices=["tfgridnet", "storm"], help="前端类型")
    parser.add_argument("--ckpt", type=str, required=True, help="前端预训练 ckpt 路径")
    parser.add_argument("--test_wav", type=str, default=None, help="单测试音频路径")
    parser.add_argument("--in_dir", type=str, default=None, help="批量测试输入目录")
    parser.add_argument("--out_dir", type=str, default="test_frontend_output", help="输出目录")
    # TFGridNet 参数
    parser.add_argument("--frontend_root", type=str, default="/home/xyf/cwh_project/espnet", help="espnet 路径")
    parser.add_argument("--frontend_module", type=str, default="espnet2.enh.separator.tfgridnet_separator")
    parser.add_argument("--frontend_class", type=str, default="TFGridNet")
    # Storm 参数
    parser.add_argument("--storm_nf", type=int, default=128, help="Storm nf 参数")
    parser.add_argument("--storm_ch_mult", type=int, nargs='+', default=[1, 2, 2, 2], help="Storm ch_mult 参数")
    parser.add_argument("--storm_num_res_blocks", type=int, default=1, help="Storm num_res_blocks 参数")
    parser.add_argument("--n_fft", type=int, default=510, help="STFT n_fft")
    parser.add_argument("--hop_length", type=int, default=128, help="STFT hop_length")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # 构建前端模型
    if args.frontend == "tfgridnet":
        print("加载 TFGridNet 前端...")
        frontend = build_frontend(
            frontend_name="tfgridnet",
            frontend_root=args.frontend_root,
            frontend_config="TF-GridNet-config/tfgridnet_lighter.yaml",
            frontend_module=args.frontend_module,
            frontend_class=args.frontend_class,
        )
        # TFGridNet 权重加载
        print(f"加载权重: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state_dict = None
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            state_dict = {k.replace("frontend.", ""): v for k, v in state_dict.items()}
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            state_dict = {k.replace("frontend.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = ckpt
        frontend.load_state_dict(state_dict, strict=False)
    elif args.frontend == "storm":
        print("加载 Storm 前端...")
        frontend = build_frontend(
            frontend_name="storm",
            frontend_ckpt=args.ckpt,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            storm_nf=args.storm_nf,
            storm_ch_mult=args.storm_ch_mult,
            storm_num_res_blocks=args.storm_num_res_blocks,
        )
    else:
        raise ValueError(f"Unknown frontend: {args.frontend}")
    
    frontend.eval()
    frontend.to(device)
    
    # 判断是单音频还是批量
    if args.in_dir is not None:
        # 批量处理
        wav_files = sorted(glob.glob(os.path.join(args.in_dir, "*.wav")))
        print(f"找到 {len(wav_files)} 个 wav 文件")
        
        with torch.no_grad():
            for idx, wav_path in enumerate(wav_files):
                basename = os.path.basename(wav_path)
                
                wav = load_audio(wav_path)
                wav = wav.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
                
                enhanced = frontend(wav)
                enhanced = enhanced.squeeze().detach().cpu()
                
                out_path = os.path.join(args.out_dir, basename)
                save_audio(enhanced, out_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"已处理 {idx + 1}/{len(wav_files)} 个文件")
        
        print(f"完成！输出目录: {args.out_dir}")
        
    elif args.test_wav is not None:
        # 单音频处理
        test_wav = args.test_wav
        print(f"测试音频: {test_wav}")
        
        wav = load_audio(test_wav)
        print(f"原始音频: shape={wav.shape}, mean={wav.mean():.6f}, std={wav.std():.6f}")
        
        wav = wav.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
        
        with torch.no_grad():
            enhanced = frontend(wav)
        
        enhanced = enhanced.squeeze().detach().cpu()
        print(f"增强后: shape={enhanced.shape}, mean={enhanced.mean():.6f}, std={enhanced.std():.6f}")
        
        out_path = os.path.join(args.out_dir, f"{args.frontend}_enhanced.wav")
        save_audio(enhanced, out_path)
        print(f"已保存到: {out_path}")
        
    else:
        print("请指定 --test_wav (单音频) 或 --in_dir (批量目录)")


if __name__ == "__main__":
    main()