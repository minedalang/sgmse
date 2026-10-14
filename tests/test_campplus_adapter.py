# Regression test: standalone CAM++ does not import WeSpeaker and retains autograd.
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from unittest import mock

from sgmse.campplus_standalone import CAMPPlus
from sgmse.speaker_encoders import WeSpeakerCAMPPlusEncoder


class AdapterTest(unittest.TestCase):
    def test_embedding_and_waveform_input_grad(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as path:
            # Use a synthetic, *complete* official-layout state_dict. The real
            # pretrained weights are checked separately on the user's server.
            model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func='TSTP')
            torch.save(model.state_dict(), Path(path) / 'avg_model.pt')
            del model
            enc = WeSpeakerCAMPPlusEncoder(path)
            enh = torch.randn(1, 16000, requires_grad=True)
            clean = torch.randn(1, 16000)
            z_enh = enc(enh)
            with torch.no_grad():
                z_clean = enc(clean)
            self.assertEqual(tuple(z_enh.shape), (1, 512))
            loss = nn.functional.mse_loss(z_enh, z_clean)
            loss.backward()
            self.assertIsNotNone(enh.grad)
            self.assertTrue(torch.isfinite(enh.grad).all())
            self.assertGreater(enh.grad.norm().item(), 0)
            self.assertTrue(all(not p.requires_grad for p in enc.parameters()))
            self.assertTrue(all(p.grad is None for p in enc.parameters()))
            enc.train()
            self.assertFalse(enc.training)

    def test_no_wespeaker_package_is_needed(self):
        import sys
        self.assertNotIn('wespeaker', sys.modules)
        self.assertNotIn('s3prl', sys.modules)


if __name__ == '__main__':
    unittest.main()
