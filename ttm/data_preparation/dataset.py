"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: 
"""
import os.path
import pickle
import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from ttm.config import MAX_PIANO_PITCH, RD_SEED, MIN_PIANO_PITCH, CHORD_N_ID, config
from ttm.data_preparation.data_augmentation import BaseDataAugmentation, UnconditionalDataAugmentation, ChordDataAugmentation

random.seed(RD_SEED)


class BaseDataset(Dataset):
    def __init__(self, feature_folder, split, feature_type='none'):
        assert split in ['train', 'validation', 'test']
        assert os.path.exists(f'{feature_folder}/{feature_type}-{split}.pkl')
        self.data = pickle.load(open(f'{feature_folder}/{feature_type}-{split}.pkl', 'rb'))
        self.data_aug = BaseDataAugmentation()
        self.split = split
        self.prefix = feature_type
        self.max_length = config[feature_type]['max_length']

    def _get_data(self, idx):
        smpl = self.data[idx]
        smpl = self.data_aug(*smpl) if self.split == 'train' else smpl
        if np.max(smpl[0][:, 0]) > MAX_PIANO_PITCH + 1 or np.min(smpl[0][:, 0]) < MIN_PIANO_PITCH:
            raise Exception("note range exceeds 88 keys")

        # Randomly sample a segment that is at most max_length long
        if self.split == 'train':
            start_idx = random.randint(0, max(0, len(smpl[0]) - 1 - self.max_length))
            end_idx = start_idx + self.max_length
            end_idx = min(len(smpl[0]), end_idx)
        elif self.split == 'validation':  # Always evaluate from the beginning
            start_idx, end_idx = 0, self.max_length
            end_idx = min(len(smpl[0]), end_idx)
        elif self.split == 'test':
            start_idx, end_idx = 0, len(smpl[0])
        else:
            raise ValueError("Unknown split type")

        note_sequence = smpl[0][start_idx:end_idx]
        annotation = smpl[1][start_idx: end_idx]

        n_feats = note_sequence.shape[1]
        pad_row = np.zeros((1, n_feats), dtype=note_sequence.dtype)
        pad_row[0, 0] = MAX_PIANO_PITCH + 1

        if n_feats > 4:
            # chord feature, give the pad the non chord id
            pad_row[0, 4] = float(CHORD_N_ID)

        # Pad sos
        note_sequence = np.concatenate([pad_row, note_sequence], axis=0)
        annotation = np.concatenate([np.array([smpl[0][start_idx][0]]), annotation], axis=0)
        
        if self.split != 'test':
            note_sequence = note_sequence[:self.max_length]
            annotation = annotation[:self.max_length]

        return note_sequence, annotation

    def __len__(self):
        return len(self.data)


class UnconditionalDataset(BaseDataset):
    def __init__(self, feature_folder, split, feature_type='unconditional'):
        super().__init__(feature_folder, split, feature_type)
        self.data_aug = UnconditionalDataAugmentation()

    def __getitem__(self, item):
        noteseq, labels = self._get_data(item)
        # convert note seq to 88 pitch indices
        noteseq[:, 0] -= MIN_PIANO_PITCH
        labels -= MIN_PIANO_PITCH

        # Pad if train/val and len < max_length
        if self.split != 'test' and len(noteseq) < self.max_length:
            pad_len = self.max_length - len(noteseq)
            pad_row = np.array([[88, 0, 0, 0]])  # pad token row
            pad_block = np.repeat(pad_row, pad_len, axis=0)
            noteseq = np.concatenate([noteseq, pad_block], axis=0)

            # Also pad labels if needed
            label_pad = np.full(pad_len, 88)  # pad label as well
            labels = np.concatenate([labels, label_pad])

        return torch.tensor(noteseq), torch.tensor(labels)

class ChordDataset(BaseDataset):
    def __init__(self, feature_folder, split, feature_type='chord'):

        super().__init__(feature_folder, split, feature_type)
        self.data_aug = ChordDataAugmentation()

    def __getitem__(self, item):
        noteseq, labels = self._get_data(item)  

        noteseq[:, 0] -= MIN_PIANO_PITCH
        labels = labels - MIN_PIANO_PITCH

        if self.split != 'test' and len(noteseq) < self.max_length:
            pad_len = self.max_length - len(noteseq)
            D = noteseq.shape[1]

            pad_row = np.zeros((1, D), dtype=noteseq.dtype)
            pad_row[0, 0] = 88

            if D > 4:
                pad_row[0, 4] = float(CHORD_N_ID)

            pad_block = np.repeat(pad_row, pad_len, axis=0)
            noteseq = np.concatenate([noteseq, pad_block], axis=0)

            label_pad = np.full(pad_len, 88)
            labels = np.concatenate([labels, label_pad])

        return torch.tensor(noteseq), torch.tensor(labels)


def main():
    data = ChordDataset(
        '/Users/000flms/10701-IML/Tap-to-Music/data/chord',
        'train'
    )
    for i in range(1):
        item = data.__getitem__(i)
        print(item, item[0].shape, item[1].shape)


if __name__ == "__main__":
    main()
