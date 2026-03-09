import os
import argparse
import numpy as np
import torch
import soundfile as sf

from sgmse.joint_model import JointSemambaSgmseModel


def load_audio(path, target_sr=None):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # 转单声道
    wav = wav.astype(np.float32)

    if target_sr is not None and sr != target_sr:
        raise ValueError(
            f"采样率不一致: wav sr={sr}, target_sr={target_sr}. "
            f"你当前代码里没做重采样，请先保证输入和模型采样率一致。"
        )
    return wav, sr


def match_length(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    y/ref: [B, 1, T]
    """
    T = ref.size(-1)
    if y.size(-1) > T:
        y = y[..., :T]
    elif y.size(-1) < T:
        pad = T - y.size(-1)
        y = torch.nn.functional.pad(y, (0, pad))
    return y


def print_tensor_stats(name, x: torch.Tensor):
    x_det = x.detach()
    print(
        f"{name}: "
        f"shape={tuple(x_det.shape)}, "
        f"dtype={x_det.dtype}, "
        f"device={x_det.device}, "
        f"mean={x_det.mean().item():.6f}, "
        f"std={x_det.std().item():.6f}, "
        f"min={x_det.min().item():.6f}, "
        f"max={x_det.max().item():.6f}, "
        f"nan={torch.isnan(x_det).any().item()}, "
        f"inf={torch.isinf(x_det).any().item()}"
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="joint 训练得到的 .ckpt")
    parser.add_argument("--input_wav", type=str, required=True, help="输入 noisy wav")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")

    parser.add_argument("--semamba_root", type=str, required=True)
    parser.add_argument("--semamba_config", type=str, required=True)
    parser.add_argument(
        "--semamba_ckpt",
        type=str,
        default=None,
        help="可选。一般测试 joint ckpt 自带 semamba 权重时可设为 None"
    )

    parser.add_argument(
        "--train_type",
        type=str,
        default="mode3",
        choices=["mode2", "mode3"],
        help="必须和训练时一致：mode3=SGMSE(noisy | y_pre), mode2=SGMSE(y_pre | noisy)"
    )

    parser.add_argument("--use_ema", action="store_true", help="是否用 EMA 权重推理")
    parser.add_argument("--N", type=int, default=30, help="扩散采样步数")
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # 1) 加载 joint 模型
    # semamba_trainable=False 只是确保当前推理阶段不训练
    # 加载 ckpt 后我们再统一 .eval() + freeze
    model = JointSemambaSgmseModel.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        map_location=device,
        train_type=args.train_type,
        semamba_root=args.semamba_root,
        semamba_config=args.semamba_config,
        semamba_ckpt=args.semamba_ckpt,   # 若要严格使用 joint ckpt 中的 semamba，一般传 None
        semamba_trainable=False,
        strict=False,
    )

    # 2) 全冻结 + eval
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # 可选：使用/关闭 EMA
    # 你原项目里 eval(no_ema=False) 会切到 EMA 参数
    # 如果你想排除 EMA 干扰，可改成 model.eval(no_ema=True)
    if hasattr(model, "eval"):
        if args.use_ema:
            try:
                model.eval(no_ema=False)
                print("eval with EMA params")
            except TypeError:
                model.eval()
                print("model.eval() done (this class may not support no_ema arg)")
        else:
            try:
                model.eval(no_ema=True)
                print("eval with raw params (EMA disabled)")
            except TypeError:
                model.eval()
                print("model.eval() done (this class may not support no_ema arg)")

    # 3) 读 noisy wav
    noisy_np, sr = load_audio(args.input_wav, target_sr=getattr(model, "sr", None))
    noisy_t = torch.from_numpy(noisy_np).float().to(device)         # [T]
    noisy_t = noisy_t.unsqueeze(0).unsqueeze(0)                     # [1,1,T]

    print("\n===== Input stats =====")
    print_tensor_stats("noisy_wav", noisy_t)

    # 4) 仅跑 SEMamba 前端
    y_pre = model._semamba_forward(noisy_t)                         # [1,1,T']
    y_pre = match_length(y_pre, noisy_t)

    print("\n===== SEMamba output stats =====")
    print_tensor_stats("y_pre", y_pre)

    semamba_out = y_pre.squeeze().detach().cpu().numpy()
    semamba_out = np.clip(semamba_out, -1.0, 1.0)

    semamba_out_path = os.path.join(args.out_dir, "semamba_output.wav")
    sf.write(semamba_out_path, semamba_out, sr)
    print(f"saved: {semamba_out_path}")

    # 5) 串联跑完整 joint pipeline
    # mode3: y_in = noisy, y_cond = y_pre
    # mode2: y_in = y_pre,  y_cond = noisy
    if args.train_type == "mode2":
        y_in = y_pre
        y_cond = noisy_t
    elif args.train_type == "mode3":
        y_in = noisy_t
        y_cond = y_pre
    else:
        raise ValueError(f"unsupported train_type: {args.train_type}")

    print("\n===== Joint input stats =====")
    print_tensor_stats("y_in", y_in)
    print_tensor_stats("y_cond", y_cond)

    # 注意：你的 enhance 里通常输入是一维 [T] 或 [1,T]
    enhanced = model.enhance(
        y_in.squeeze(0).cpu(),          # [1,T]
        y_cond=y_cond.squeeze(0).cpu(), # [1,T]
        N=args.N,
        snr=args.snr,
    )

    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().numpy()

    enhanced = np.asarray(enhanced).squeeze()
    print("\n===== Joint enhanced stats =====")
    print(
        f"enhanced: shape={enhanced.shape}, "
        f"mean={enhanced.mean():.6f}, std={enhanced.std():.6f}, "
        f"min={enhanced.min():.6f}, max={enhanced.max():.6f}, "
        f"nan={np.isnan(enhanced).any()}, inf={np.isinf(enhanced).any()}"
    )

    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=-1.0)
    enhanced = np.clip(enhanced, -1.0, 1.0)

    enhanced_out_path = os.path.join(args.out_dir, "joint_enhanced_output.wav")
    sf.write(enhanced_out_path, enhanced, sr)
    print(f"saved: {enhanced_out_path}")

    print("\n===== Summary =====")
    print(f"1) noisy input          : {args.input_wav}")
    print(f"2) semamba output       : {semamba_out_path}")
    print(f"3) joint enhanced output: {enhanced_out_path}")


if __name__ == "__main__":
    main()