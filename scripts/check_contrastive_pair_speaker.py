#!/usr/bin/env python3
"""Sanity checks for the decoupled loss; no model weights needed."""
import torch
import torch.nn.functional as F
from sgmse.model import ScoreModel

class Dummy:
    contrastive_temperature = 0.1

s = Dummy()
torch.manual_seed(13)
enh = torch.randn(8, 768, requires_grad=True)
clean = torch.randn(8, 768)
ids = torch.tensor([101, 101, 202, 202, 303, 303, 404, 404])

pair, speaker, ratio = ScoreModel._contrastive_pair_speaker_losses(s, enh, clean, ids)
z1 = F.normalize(enh.float(), dim=-1)
z2 = F.normalize(clean.float(), dim=-1)
logits = z1 @ z2.T / s.contrastive_temperature
labels = torch.arange(8)
reference = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
assert torch.allclose(pair, reference, atol=1e-6), (pair, reference)
assert speaker.item() > 0 and ratio.item() == 1.0
loss = pair + 0.1 * speaker
loss.backward()
assert enh.grad is not None and torch.isfinite(enh.grad).all() and enh.grad.abs().sum() > 0
assert clean.grad is None

ids_single = torch.arange(8)
pair2, speaker2, ratio2 = ScoreModel._contrastive_pair_speaker_losses(
    s, enh.detach(), clean, ids_single
)
assert speaker2.item() == 0 and ratio2.item() == 0
assert torch.allclose(pair, pair2)

print("PASS: pair loss matches old spk0; speaker loss active; gradients reach enhanced embedding;")
print("      clean targets detached; no same-speaker pairs yield zero auxiliary loss.")
print(f"pair={pair.item():.5f}, speaker={speaker.item():.5f}, valid_ratio={ratio.item():.2f}")
