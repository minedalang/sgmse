"""
测试单独预训练的 TFGridNet 前端权重
用法:
  python test_frontend_only.py --ckpt frontend_pretrain_logs/5bd3pa64-tfgridnet_pretrain_3-20/last.ckpt
"""

import argparse
import os
import torch
import torchaudio
import librosa
import numpy as np

from sgmse.frontends import build_frontend


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr)
    return torch.from_numpy(wav).float()


def save_audio(wav, path, sr=16000):
    torchaudio.save(path, wav.unsqueeze(0).cpu(), sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="TFGridNet pretrain ckpt path")
    parser.add_argument("--test_wav", type=str, default="test_samples/noisy.wav", help="测试音频路径")
    parser.add_argument("--out_dir", type=str, default="test_frontend_output", help="输出目录")
    parser.add_argument("--frontend_root", type=str, default="/home/xyf/cwh_project/espnet", help="espnet 路径")
    parser.add_argument("--frontend_module", type=str, default="espnet2.enh.separator.tfgridnet_separator")
    parser.add_argument("--frontend_class", type=str, default="TFGridNet")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    # 构建前端模型
    print("加载 TFGridNet 前端...")
    frontend = build_frontend(
        frontend_name="tfgridnet",
        frontend_root=args.frontend_root,
        frontend_config="TF-GridNet-config/tfgridnet.yaml",
        frontend_module=args.frontend_module,
        frontend_class=args.frontend_class,
    )
    
    # 加载权重
    print(f"加载权重: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    # 尝试不同的 key
    state_dict = None
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        # 移除 "frontend." 前缀
        state_dict = {k.replace("frontend.", ""): v for k, v in state_dict.items()}
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        state_dict = {k.replace("frontend.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt
    
    frontend.load_state_dict(state_dict, strict=False)
    frontend.eval()
    frontend.to(device)
    # 测试音频
    test_wav = args.test_wav
    if not os.path.exists(test_wav):
        # 使用默认测试音频
        test_wav = "/home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15_v2_testset/noisy/book_00198_chp_0017_reader_00957_5_8Vo48iZX8Ro_snr-17.85_tl-26_fileid_147.wav"
    
    print(f"测试音频: {test_wav}")
    
    # 加载并处理
    wav = load_audio(test_wav)
    print(f"原始音频: shape={wav.shape}, mean={wav.mean():.6f}, std={wav.std():.6f}")
    
    # 预处理：添加 batch 维度和 channel 维度
    wav = wav.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
    
    with torch.no_grad():
        enhanced = frontend(wav)
    
    # 输出
    enhanced = enhanced.squeeze().detach().cpu()  # [T]
    print(f"增强后: shape={enhanced.shape}, mean={enhanced.mean():.6f}, std={enhanced.std():.6f}")
    
    # 保存
    out_path = os.path.join(args.out_dir, "tfgridnet_enhanced.wav")
    save_audio(enhanced, out_path)
    print(f"已保存到: {out_path}")


if __name__ == "__main__":
    main()
