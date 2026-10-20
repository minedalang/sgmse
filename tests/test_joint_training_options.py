import unittest

import torch
from torch.utils.data import Dataset

from sgmse.joint_data_module import _BaseJointWaveDataModule
from sgmse.joint_training_utils import rrsd_aux_loss_enabled
from sgmse.sampler import SpeakerBatchSampler


class _TinySpeakerDataset(Dataset):
    def __init__(self):
        self.spk_ids = [1, 1, 2, 2, 3, 3, 4, 4]

    def __len__(self):
        return len(self.spk_ids)

    def __getitem__(self, index):
        value = torch.tensor(float(index))
        return value, value, torch.tensor(self.spk_ids[index], dtype=torch.long)


class JointTrainingOptionsTest(unittest.TestCase):
    def _module(self):
        module = _BaseJointWaveDataModule(
            batch_size=4,
            num_workers=0,
            gpu=False,
        )
        module.train_set = _TinySpeakerDataset()
        return module

    def test_balanced_batches_are_opt_in_when_contrastive_is_disabled(self):
        loader = self._module().train_dataloader(
            contrastive_weight=0.0,
            speaker_balanced_batches=True,
        )
        self.assertIsInstance(loader.batch_sampler, SpeakerBatchSampler)
        self.assertEqual(len(loader), 2)
        for _, _, speaker_ids in loader:
            _, counts = torch.unique(speaker_ids, return_counts=True)
            self.assertTrue(torch.all(counts == 2))

    def test_default_sampling_remains_shuffled_batches(self):
        loader = self._module().train_dataloader(contrastive_weight=0.0)
        self.assertNotIsInstance(loader.batch_sampler, SpeakerBatchSampler)
        self.assertEqual(loader.batch_size, 4)
        self.assertTrue(loader.drop_last)

    def test_rrsd_auxiliary_loss_gate(self):
        router = object()
        self.assertTrue(rrsd_aux_loss_enabled(router, 1.0, False))
        self.assertFalse(rrsd_aux_loss_enabled(router, 1.0, True))
        self.assertFalse(rrsd_aux_loss_enabled(router, 0.0, False))
        self.assertFalse(rrsd_aux_loss_enabled(None, 1.0, False))


if __name__ == "__main__":
    unittest.main()
