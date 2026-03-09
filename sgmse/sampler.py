from torch.utils.data import Sampler
from collections import defaultdict
import random
from torch.utils.data.distributed import DistributedSampler
 
class SpeakerBatchSampler(Sampler):
    """
    按说话人分组采样，每个 batch 保证：
      - 包含若干个说话人；
      - 每个说话人至少 min_per_speaker 条样本。

    假设：
      - batch_size 是 min_per_speaker 的整数倍。
    """
    def __init__(self, spk_ids, batch_size, min_per_speaker=2, drop_last=True):
        """
        spk_ids: list[int]，长度为 N，对应 dataset 中每条样本的说话人 id
        """
        if isinstance(spk_ids, DistributedSampler):
            print("============")
            print("spk_ids.dataset type:",type(spk_ids.dataset))
            print("spk_ids.dataset.spk_ids type:",type(spk_ids.dataset))
            spk_ids = spk_ids.dataset.spk_ids
            
        self.spk_ids = list(spk_ids)

        print("self.spk_ids type:", type(self.spk_ids))
        # print("spk_ids type:", type(spk_ids))
        self.batch_size = batch_size
        self.min_per_speaker = min_per_speaker
        self.drop_last = drop_last

        assert batch_size % min_per_speaker == 0, \
            "batch_size 必须能被 min_per_speaker 整除，比如 batch=8, min_per_speaker=2"

        self.num_speakers_per_batch = batch_size // min_per_speaker
        # 构建 spk_id -> 数据索引的映射
        # print(self.spk_ids)
        spk_to_indices = defaultdict(list)
        for idx, sid in enumerate(self.spk_ids):
            spk_to_indices[sid].append(idx)

        # 3. 预先把每个说话人的 index 分成一组一组（每组 min_per_speaker 条）
        self.groups = []  # 每个元素是一个 list[int]，长度 = min_per_speaker
        for sid, idxs in spk_to_indices.items():
            random.shuffle(idxs)  # 每个 epoch 可以重建这个类，或在 __iter__ 里再 shuffle
            num_full_groups = len(idxs) // self.min_per_speaker #组数
            for g in range(num_full_groups):
                start = g * self.min_per_speaker
                group = idxs[start:start + self.min_per_speaker]
                self.groups.append(group)

        # 4. 由“总组数”决定“总 batch 数”，完全确定
        self.num_groups = len(self.groups)
        self.num_batches = self.num_groups // self.num_speakers_per_batch

        if not self.drop_last and self.num_groups % self.num_speakers_per_batch != 0:
            self.num_batches += 1  # 多出来的不满一个 batch 的组，后面在 __iter__ 处理


    # def __iter__(self):
    #     # 拷贝一份可变的索引池
    #     spk_to_indices = {sid: idxs.copy() for sid, idxs in self.spk_to_indices.items()}

    #     while True:
    #         # 能提供至少 min_per_speaker 条样本的说话人
    #         available_speakers = [
    #             sid for sid, idxs in spk_to_indices.items()
    #             if len(idxs) >= self.min_per_speaker
    #         ]

    #         # 不足以再凑一个 batch 了
    #         if len(available_speakers) < self.num_speakers_per_batch:
    #             break

    #         # 随机选出本 batch 用到的说话人
    #         chosen_speakers = random.sample(available_speakers, self.num_speakers_per_batch)

    #         batch_indices = []
    #         for sid in chosen_speakers:
    #             # 从该说话人剩余样本中取 min_per_speaker 条
    #             for _ in range(self.min_per_speaker):
    #                 batch_indices.append(spk_to_indices[sid].pop())

    #         # 一个 batch
    #         yield batch_indices

    def __iter__(self):
        # 每个 epoch 打乱一下“组”的顺序
        group_indices = list(range(self.num_groups))
        random.shuffle(group_indices)

        # drop_last=True 时，只用前 num_batches * num_speakers_per_batch 个组
        max_groups = (
            self.num_batches * self.num_speakers_per_batch
            if self.drop_last
            else self.num_groups
        )

        # 按组拼 batch
        for b_start in range(0, max_groups, self.num_speakers_per_batch):
            batch_groups = group_indices[b_start:b_start + self.num_speakers_per_batch]
            if self.drop_last and len(batch_groups) < self.num_speakers_per_batch:
                break

            batch_indices = []
            for gi in batch_groups:
                batch_indices.extend(self.groups[gi])

            yield batch_indices

    # def __len__(self):

    #     # 粗略估计一下总共能生成多少个 batch
        
    #     groups_per_spk = [
    #         len(idxs) // self.min_per_speaker
    #         for idxs in self.spk_to_indices.values()
    #     ]
    #     total_groups = sum(groups_per_spk)  # 总共能提供多少个「说话人组」

    #     if total_groups == 0:
    #         # 数据确实不够形成任何组
    #         return 0

    #     # 每个 batch 需要 num_speakers_per_batch 个说话人组
    #     total_batches = total_groups // self.num_speakers_per_batch

    #     if self.drop_last:
    #         return total_batches
    #     else:
    #         # 至少返回 1，避免出现 len=0 的情况
    #         return max(total_batches, 1)
    
    def __len__(self):
        return self.num_batches
