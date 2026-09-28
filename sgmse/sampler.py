from torch.utils.data import Sampler
from collections import defaultdict
import random
import torch.distributed as dist


class SpeakerBatchSampler(Sampler):
    """
    Speaker-balanced batch sampler.

    Each global batch contains batch_size/min_per_speaker speakers and
    min_per_speaker utterances from each selected speaker. With DDP enabled,
    complete global batches are deterministically sharded across ranks so ranks
    do not train on duplicated batches.
    """
    def __init__(self, spk_ids, batch_size, min_per_speaker=2, drop_last=True, seed=0):
        self.spk_ids = list(spk_ids)
        self.batch_size = int(batch_size)
        self.min_per_speaker = int(min_per_speaker)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size % self.min_per_speaker != 0:
            raise ValueError(
                "batch_size must be divisible by min_per_speaker "
                f"(got {self.batch_size} and {self.min_per_speaker})."
            )
        self.num_speakers_per_batch = self.batch_size // self.min_per_speaker

        spk_to_indices = defaultdict(list)
        for idx, sid in enumerate(self.spk_ids):
            spk_to_indices[int(sid)].append(idx)
        self.spk_to_indices = dict(spk_to_indices)

        # Number of complete speaker-groups available per epoch.
        self.num_groups = sum(
            len(v) // self.min_per_speaker for v in self.spk_to_indices.values()
        )
        self.num_batches = self.num_groups // self.num_speakers_per_batch
        if not self.drop_last and self.num_groups % self.num_speakers_per_batch:
            self.num_batches += 1

    @staticmethod
    def _dist_info():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        groups = []
        for _, idxs0 in self.spk_to_indices.items():
            idxs = list(idxs0)
            rng.shuffle(idxs)
            n = len(idxs) // self.min_per_speaker
            for g in range(n):
                s = g * self.min_per_speaker
                groups.append(idxs[s:s + self.min_per_speaker])

        rng.shuffle(groups)
        batches = []
        for s in range(0, len(groups), self.num_speakers_per_batch):
            batch_groups = groups[s:s + self.num_speakers_per_batch]
            if len(batch_groups) < self.num_speakers_per_batch and self.drop_last:
                break
            batch = [idx for group in batch_groups for idx in group]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        rank, world_size = self._dist_info()
        if world_size > 1:
            # DDP requires every rank to execute the same number of optimizer steps.
            usable = (len(batches) // world_size) * world_size
            batches = batches[:usable]
            batches = batches[rank::world_size]

        yield from batches

    def __len__(self):
        _, world_size = self._dist_info()
        if world_size <= 1:
            return self.num_batches
        return self.num_batches // world_size
