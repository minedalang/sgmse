"""
测试前端输出 y_pre 的脚本
用于对比：冻结前端 vs 解冻前端 的前端增强输出

用法:
  # 测试 rojpnxbe 300k (冻结前端)
  python test_frontend_ypre_only.py \
    --ckpt joint_train_logs/rojpnxbe-None/last.ckpt \
    --in_dir enhanced_h8xb26xi_300k_v2_testset \
    --out_dir y_pre_result/rojpnxbe_300k

  # 测试 h8xb26xi 300k (解冻前端)
  python test_frontend_ypre_only.py \
    --ckpt joint_train_logs/h8xb26xi-trainable_startat_rojpnxbe_300k_weight05_timemse/last.ckpt \
    --in_dir enhanced_h8xb26xi_300k_v2_testset \
    --out_dir y_pre_result/h8xb26xi_300k
"""

import argparse
import os
import glob
import torch
import librosa
import numpy as np
import soundfile as sf

from sgmse.frontends import build_frontend


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr)
    return torch.from_numpy(wav).float()


def save_audio(wav, path, sr=16000):
    if hasattr(wav, 'numpy'):
        wav = wav.numpy()
    sf.write(path, wav, sr)


def main():
    parser = argparse.ArgumentParser(description="提取前端输出 y_pre")
    
    # checkpoint 和路径
    parser.add_argument("--ckpt", type=str, required=True, help="joint lightning .ckpt 路径")
    parser.add_argument("--in_dir", type=str, required=True, help="输入 noisy 音频目录")
    parser.add_argument("--out_dir", type=str, required=True, help="输出 y_pre 音频目录")
    parser.add_argument("--num_files", type=int, default=10, help="最多处理多少个文件，-1表示全部")
    
    # Storm 前端参数 (需要匹配训练时的配置)
    parser.add_argument("--frontend", type=str, default="storm", choices=["storm", "semamba", "tfgridnet"])
    parser.add_argument("--storm_nf", type=int, default=128)
    parser.add_argument("--storm_ch_mult", type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument("--storm_num_res_blocks", type=int, default=1)
    parser.add_argument("--n_fft", type=int, default=510)
    parser.add_argument("--hop_length", type=int, default=128)
    
    # TFGridNet 参数
    parser.add_argument("--frontend_root", type=str, default="/home/xyf/cwh_project/espnet")
    parser.add_argument("--frontend_module", type=str, default="espnet2.enh.separator.tfgridnet_separator")
    parser.add_argument("--frontend_class", type=str, default="TFGridNet")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # 构建前端模型并加载权重
    print(f"加载 Storm 前端...")
    frontend = build_frontend(
        frontend_name="storm",
        frontend_ckpt=args.ckpt,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        storm_nf=args.storm_nf,
        storm_ch_mult=args.storm_ch_mult,
        storm_num_res_blocks=args.storm_num_res_blocks,
    )
    
    frontend.eval()
    frontend.to(device)
    
    # 获取测试音频列表
    wav_files = sorted(glob.glob(os.path.join(args.in_dir, "*.wav")))
    if args.num_files > 0 and args.num_files < len(wav_files):
        wav_files = wav_files[:args.num_files]
    print(f"Found {len(wav_files)} wav files, will process {len(wav_files)}")
    
    # 统计信息收集
    stats = []
    
    with torch.no_grad():
        for idx, wav_path in enumerate(wav_files):
            basename = os.path.basename(wav_path)
            
            # 加载 noisy 音频
            wav = load_audio(wav_path)
            wav = wav.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
            
            # 提取 y_pre (前端输出)
            y_pre = frontend(wav)
            y_pre = y_pre.squeeze().detach().cpu().numpy()
            
            # 保存
            out_path = os.path.join(args.out_dir, basename)
            save_audio(y_pre, out_path)
            
            # 收集统计信息
            stats.append({
                'file': basename,
                'y_pre_mean': float(np.mean(np.abs(y_pre))),
                'y_pre_std': float(np.std(y_pre)),
                'y_pre_max': float(np.max(np.abs(y_pre))),
            })
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(wav_files)} files")
    
    # 保存统计信息
    stats_path = os.path.join(args.out_dir, "y_pre_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"{'File':<60} {'Mean':>12} {'Std':>12} {'Max':>12}\n")
        f.write("-" * 100 + "\n")
        for s in stats:
            f.write(f"{s['file']:<60} {s['y_pre_mean']:>12.6f} {s['y_pre_std']:>12.6f} {s['y_pre_max']:>12.6f}\n")
        
        # 平均统计
        avg_mean = np.mean([s['y_pre_mean'] for s in stats])
        avg_std = np.mean([s['y_pre_std'] for s in stats])
        avg_max = np.mean([s['y_pre_max'] for s in stats])
        f.write("-" * 100 + "\n")
        f.write(f"{'AVERAGE':<60} {avg_mean:>12.6f} {avg_std:>12.6f} {avg_max:>12.6f}\n")
    
    print(f"\nDone! Output: {args.out_dir}")
    print(f"Stats saved to: {stats_path}")
    print(f"\nAverage y_pre stats:")
    print(f"  Mean: {avg_mean:.6f}")
    print(f"  Std:  {avg_std:.6f}")
    print(f"  Max:  {avg_max:.6f}")


if __name__ == "__main__":
    main()
