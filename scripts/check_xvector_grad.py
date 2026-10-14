"""Run: python -m scripts.check_xvector_grad /path/to/speechbrain-xvector-dir

Or supply speechbrain/spkrec-xvect-voxceleb to download the pretrained model.
"""
import argparse
import torch
from sgmse.xvector_encoder import SpeechBrainXVectorEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="SpeechBrain Hub ID or local pretrained folder")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    net = SpeechBrainXVectorEncoder(args.model).to(args.device).eval()
    x = torch.randn(2, 16000, device=args.device, requires_grad=True)
    clean = torch.randn(2, 16000, device=args.device)
    out = net(x)
    with torch.no_grad():
        target = net(clean)
    loss = torch.nn.functional.mse_loss(out.float(), target.float())
    loss.backward()
    grad_norm = x.grad.norm().item() if x.grad is not None else 0.0
    print("device:", args.device)
    print("embeddings:", tuple(out.shape))
    print("enhanced waveform grad norm:", grad_norm)
    print("pretrained encoder has trainable parameters:", any(p.requires_grad for p in net.parameters()))
    assert out.shape == (2, 512)
    assert torch.isfinite(x.grad).all() and grad_norm > 0
    assert all(p.grad is None for p in net.parameters())
    print("PASS: gradients reach enhanced audio; pretrained X-Vector remains frozen")


if __name__ == "__main__":
    main()
