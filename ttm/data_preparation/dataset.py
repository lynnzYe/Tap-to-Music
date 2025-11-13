"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: 
"""
import os.path
import pickle
import random

import numpy as np
from torch.utils.data.dataset import Dataset

from ttm.config import MAX_PIANO_PITCH, RD_SEED, MIN_PIANO_PITCH, config
from ttm.data_preparation.data_augmentation import BaseDataAugmentation, UnconditionalDataAugmentation

random.seed(RD_SEED)


class BaseDataset(Dataset):
    def __init__(self, feature_folder, split, tap_type='none'):
        assert split in ['train', 'valid', 'test']
        assert os.path.exists(f'{feature_folder}/{tap_type}-{split}.pkl')
        self.data = pickle.load(open(f'{feature_folder}/{tap_type}-{split}.pkl', 'rb'))
        self.data_aug = BaseDataAugmentation()
        self.split = split
        self.prefix = tap_type
        self.max_length = config[tap_type]['max_length']

    def _get_data(self, idx):
        smpl = self.data[idx]
        smpl = self.data_aug(*smpl) if self.split == 'train' else smpl
        if np.max(smpl[0][:, 0]) > MAX_PIANO_PITCH or np.min(smpl[0][:, 0]) < MIN_PIANO_PITCH:
            raise Exception("note range exceeds 88 keys")

        # Randomly sample a segment that is at most max_length long
        if self.split == 'train':
            start_idx = random.randint(0, max(0, len(smpl[0]) - 1))
            end_idx = start_idx + self.max_length
            end_idx = min(len(smpl[0]), end_idx)
        elif self.split == 'valid':  # Always evaluate from the beginning
            start_idx, end_idx = 0, self.max_length
            end_idx = min(len(smpl[0]), end_idx)
        elif self.split == 'test':
            start_idx, end_idx = 0, len(smpl[0])
        else:
            raise ValueError("Unknown split type")

        note_sequence = smpl[0][start_idx:end_idx]
        annotation = smpl[1]
        return note_sequence, annotation


class UnconditionalDataset(BaseDataset):
    def __init__(self, feature_folder, split, tap_type='unconditional'):
        super().__init__(feature_folder, split, tap_type)
        self.data_aug = UnconditionalDataAugmentation()

    def __getitem__(self, item):
        noteseq, labels = self._get_data(item)

        # convert note seq to 88 pitch indices

        # Pad if train and len < max_length

        return noteseq, labels


def main():
    print("Hello, world!")


if __name__ == "__main__":
    main()
