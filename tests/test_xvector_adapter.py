"""Unit check with synthetic SpeechBrain modules; real weights checked on server."""
import unittest
from unittest import mock
import torch
from torch import nn

from sgmse.xvector_encoder import SpeechBrainXVectorEncoder


class TestXVectorAdapter(unittest.TestCase):
    def test_gradient_and_batch_frontend(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.Xvector import Xvector
        from speechbrain.processing.features import InputNormalization

        class FakeClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.mods = nn.ModuleDict({
                    "compute_features": Fbank(n_mels=24),
                    "mean_var_norm": InputNormalization(norm_type="sentence", std_norm=False),
                    "embedding_model": Xvector(
                        in_channels=24,
                        tdnn_blocks=5,
                        tdnn_channels=[512, 512, 512, 512, 1500],
                        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
                        tdnn_dilations=[1, 2, 3, 1, 1],
                        lin_neurons=512,
                        activation=nn.LeakyReLU,
                    ),
                })

        torch.set_num_threads(2)
        with mock.patch.object(EncoderClassifier, "from_hparams", return_value=FakeClassifier()):
            enc = SpeechBrainXVectorEncoder().eval()
        enh = torch.randn(2, 16000, requires_grad=True)
        clean = torch.randn(2, 16000)
        z_enh = enc(enh)
        with torch.no_grad():
            z_clean = enc(clean)
        self.assertEqual(tuple(z_enh.shape), (2, 512))
        ((z_enh-z_clean)**2).mean().backward()
        self.assertIsNotNone(enh.grad)
        self.assertTrue(torch.isfinite(enh.grad).all())
        self.assertGreater(float(enh.grad.norm()), 0)
        self.assertFalse(any(p.requires_grad for p in enc.parameters()))
        self.assertTrue(all(p.grad is None for p in enc.parameters()))
        enc.train()
        self.assertFalse(enc.training)


if __name__ == "__main__":
    unittest.main()
