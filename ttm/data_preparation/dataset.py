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

from ttm.config import MAX_PIANO_PITCH, RD_SEED, MIN_PIANO_PITCH, config, dotenv_config
from ttm.data_preparation.data_augmentation import (
    BaseDataAugmentation,
    RangeDataAugmentation,
    UnconditionalDataAugmentation,
    ClusterAugmentation,
)
from ttm.data_preparation.data_augmentation import ChordDataAugmentation
from ttm.data_preparation.utils import ChordConstants

random.seed(RD_SEED)


class BaseDataset(Dataset):
    def __init__(self, feature_folder, split, feature_type='none'):
        assert split in ['train', 'validation', 'test']
        print(f'Loading {feature_folder}/{feature_type}-{split}.pkl')
        assert os.path.exists(f'{feature_folder}/{feature_type}-{split}.pkl')
        self.data = pickle.load(open(f'{feature_folder}/{feature_type}-{split}.pkl', 'rb'))
        self.data_aug = BaseDataAugmentation()
        self.split = split
        self.prefix = feature_type
        # Allow fallback to unconditional config so experimental features don't crash
        cfg = config.get(feature_type, config.get('unconditional', {}))
        self.max_length = cfg.get('max_length', 128)

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
            pad_row[0, 4] = float(ChordConstants.N_ID)

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
                pad_row[0, 4] = float(ChordConstants.N_ID)

            pad_block = np.repeat(pad_row, pad_len, axis=0)
            noteseq = np.concatenate([noteseq, pad_block], axis=0)

            label_pad = np.full(pad_len, 88)
            labels = np.concatenate([labels, label_pad])

        return torch.tensor(noteseq, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class RangeDataset(BaseDataset):
    """Dataset for range-conditioned features (shape: (T, 5) with final col = range id)."""

    def __init__(self, feature_folder, split, feature_type='unconditional'):
        super().__init__(feature_folder, split, feature_type)
        self.data_aug = RangeDataAugmentation()

    def __getitem__(self, item):
        smpl = self.data[item]
        # Expected tuple: (features, labels, range_label)
        if len(smpl) == 3:
            feats, labels, piece_range = smpl
        else:
            feats, labels = smpl
            piece_range = None

        # Optional augmentation on train split only
        if self.split == 'train':
            feats, labels, new_range = self.data_aug(feats, labels, piece_range)
            piece_range = new_range if new_range is not None else piece_range

        if np.max(feats[:, 0]) > MAX_PIANO_PITCH + 1 or np.min(feats[:, 0]) < MIN_PIANO_PITCH:
            raise Exception("note range exceeds 88 keys")

        # Segment sampling, mirroring BaseDataset logic
        if self.split == 'train':
            start_idx = random.randint(0, max(0, len(feats) - 1 - self.max_length))
            end_idx = min(len(feats), start_idx + self.max_length)
        elif self.split == 'validation':
            start_idx, end_idx = 0, min(len(feats), self.max_length)
        elif self.split == 'test':
            start_idx, end_idx = 0, len(feats)
        else:
            raise ValueError("Unknown split type")

        note_sequence = feats[start_idx:end_idx]
        annotation = labels[start_idx:end_idx]

        # Pad sos
        range_value = piece_range[1] if isinstance(piece_range, tuple) else (
            piece_range if piece_range is not None else 0)
        pad_row = np.array([[MAX_PIANO_PITCH + 1, 0, 0, 0, range_value]])
        note_sequence = np.concatenate([pad_row, note_sequence], axis=0)
        annotation = np.concatenate([np.array([labels[start_idx][0]]), annotation], axis=0)

        if self.split != 'test':
            note_sequence = note_sequence[:self.max_length]
            annotation = annotation[:self.max_length]

        # Normalize pitch to 0..88 and set pad token to 88
        note_sequence[:, 0] -= MIN_PIANO_PITCH
        annotation -= MIN_PIANO_PITCH
        if self.split != 'test' and len(note_sequence) < self.max_length:
            pad_len = self.max_length - len(note_sequence)
            pad_row = np.array([[88, 0, 0, 0, range_value]])
            pad_block = np.repeat(pad_row, pad_len, axis=0)
            note_sequence = np.concatenate([note_sequence, pad_block], axis=0)
            label_pad = np.full(pad_len, 88)
            annotation = np.concatenate([annotation, label_pad])

        # Return the per-piece range label alongside tensors
        return torch.tensor(note_sequence, dtype=torch.float32), torch.tensor(annotation, dtype=torch.long), piece_range


class ClusterDataset(BaseDataset):
    """
    Dataset that enforces column 4 to hold the per-sequence median (computed via ClusterAugmentation).
    Behaves like UnconditionalDataset for pitch handling, but preserves a 5th median column.
    """

    def __init__(self, feature_folder, split, feature_type='unconditional'):
        super().__init__(feature_folder, split, feature_type)
        self.data_aug = ClusterAugmentation()

    def __getitem__(self, item):
        smpl = self.data[item]
        if len(smpl) == 3:
            feats, labels, median_info = smpl
        else:
            feats, labels = smpl
            median_info = None

        # Optional augmentation on train split only
        if self.split == 'train':
            feats, labels, new_median = self.data_aug(feats, labels)
            median_info = new_median if new_median is not None else median_info

        if np.max(feats[:, 0]) > MAX_PIANO_PITCH + 1 or np.min(feats[:, 0]) < MIN_PIANO_PITCH:
            raise Exception("note range exceeds 88 keys")

        # Segment sampling
        if self.split == 'train':
            start_idx = random.randint(0, max(0, len(feats) - 1 - self.max_length))
            end_idx = min(len(feats), start_idx + self.max_length)
        elif self.split == 'validation':
            start_idx, end_idx = 0, min(len(feats), self.max_length)
        elif self.split == 'test':
            start_idx, end_idx = 0, len(feats)
        else:
            raise ValueError("Unknown split type")

        note_sequence = feats[start_idx:end_idx]
        annotation = labels[start_idx:end_idx]

        # Unpack medians
        if isinstance(median_info, (tuple, list)) and len(median_info) >= 3:
            median_scalar, left_med, right_med = median_info
        else:
            median_scalar = median_info if median_info is not None else 0
            left_med = right_med = median_scalar

        note_sequence = note_sequence.copy()
        # Ensure median column exists and is filled (col 4)
        if note_sequence.shape[1] < 5:
            note_sequence = np.concatenate([note_sequence, np.full((len(note_sequence), 1), median_scalar)], axis=1)
        else:
            note_sequence[:, 4] = median_scalar

        # Ensure hand flag and per-hand medians columns exist (cols 5,6,7) and are populated
        if note_sequence.shape[1] < 6:
            note_sequence = np.concatenate([note_sequence, np.zeros((len(note_sequence), 1))], axis=1)
        # left median (col 6)
        if note_sequence.shape[1] < 7:
            note_sequence = np.concatenate([note_sequence, np.full((len(note_sequence), 1), left_med)], axis=1)
        else:
            note_sequence[:, 6] = left_med
        # right median (col 7)
        if note_sequence.shape[1] < 8:
            note_sequence = np.concatenate([note_sequence, np.full((len(note_sequence), 1), right_med)], axis=1)
        else:
            note_sequence[:, 7] = right_med

        # Pad sos
        n_feat = note_sequence.shape[1]
        pad_row = np.zeros((1, n_feat))
        pad_row[0, 0] = MAX_PIANO_PITCH + 1
        pad_row[0, 4] = median_scalar
        if n_feat > 5:
            pad_row[0, 5] = 0  # hand flag pad
        if n_feat > 6:
            pad_row[0, 6] = left_med
        if n_feat > 7:
            pad_row[0, 7] = right_med
        note_sequence = np.concatenate([pad_row, note_sequence], axis=0)
        annotation = np.concatenate([np.array([labels[start_idx][0]]), annotation], axis=0)

        if self.split != 'test':
            note_sequence = note_sequence[:self.max_length]
            annotation = annotation[:self.max_length]

        # Normalize pitch to 0..88 and set pad token to 88
        note_sequence[:, 0] -= MIN_PIANO_PITCH
        annotation -= MIN_PIANO_PITCH
        if self.split != 'test' and len(note_sequence) < self.max_length:
            pad_len = self.max_length - len(note_sequence)
            pad_row = np.zeros((1, n_feat))
            pad_row[0, 0] = 88
            pad_row[0, 4] = median_scalar
            if n_feat > 5:
                pad_row[0, 5] = 0  # hand flag pad
            if n_feat > 6:
                pad_row[0, 6] = left_med
            if n_feat > 7:
                pad_row[0, 7] = right_med
            pad_block = np.repeat(pad_row, pad_len, axis=0)
            note_sequence = np.concatenate([note_sequence, pad_block], axis=0)
            label_pad = np.full(pad_len, 88)
            annotation = np.concatenate([annotation, label_pad])

        return torch.tensor(note_sequence, dtype=torch.float32), torch.tensor(annotation, dtype=torch.long), (
            median_scalar, left_med, right_med)


def main():
    data = UnconditionalDataset(
        feature_folder=dotenv_config['FEATURE_FOLDER'],
        split=dotenv_config['SPLIT'],
        feature_type=dotenv_config['FEATURE_TYPE']
    )
    for i in range(1):
        item = data.__getitem__(i)
        print(item, item[0].shape, item[1].shape)


if __name__ == "__main__":
    main()
