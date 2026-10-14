"""Sanity-check real CAM++ gradient flow before a full training run.

Usage: python scripts/check_campplus_grad.py /absolute/path/to/model_dir
The model directory must contain avg_model.pt. (16 kHz input.)
"""
import sys
import torch

from sgmse.speaker_encoders import WeSpeakerCAMPPlusEncoder


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/check_campplus_grad.py /path/to/campplus_model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = WeSpeakerCAMPPlusEncoder(sys.argv[1]).to(device)
    wav = torch.randn(2, 16000, device=device, requires_grad=True)
    clean = torch.randn(2, 16000, device=device)
    enh_emb = model(wav)
    with torch.no_grad():
        clean_emb = model(clean)
    loss = torch.nn.functional.mse_loss(enh_emb, clean_emb)
    loss.backward()
    print('encoder:', model.__class__.__name__, 'device:', device)
    print('embeddings:', tuple(enh_emb.shape))
    print('enhanced waveform grad norm:', wav.grad.norm().item())
    print('pretrained encoder has trainable parameters:', any(p.requires_grad for p in model.parameters()))
    assert wav.grad is not None and wav.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())
    print('PASS: gradient reaches enhanced waveform while pretrained CAM++ stays frozen')


if __name__ == '__main__':
    main()
