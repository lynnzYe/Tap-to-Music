"""
Author: YZQ
Created on: 2025/11/27
Brief: Dataset classes for POP909 data with extended features
       Extends BaseDataset from ttm module

       Two dataset classes available:
       - HandDataset: For 5-feature data (pitch, dt, dur, vel, hand)
       - WindowDataset: For 5-feature data (pitch, dt, dur, vel, window_avg)
"""
import os.path
import pickle
import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

# Import from ttm module
from ttm.config import MAX_PIANO_PITCH, MIN_PIANO_PITCH, RD_SEED, config
from ttm.data_preparation.dataset import BaseDataset

# Import augmentation classes from yzq module
from yzq.data_augmentation import HandDataAugmentation, ClusterDataAugmentation

random.seed(RD_SEED)


class HandDataset(BaseDataset):
    """
    Dataset for 5-feature hand data:
        - pitch (previous note pitch)
        - log1p delta time
        - log1p duration
        - velocity
        - hand (0=left, 1=right)
    
    Uses HandDataAugmentation for training augmentation.
    Loads from: pop909hand-{split}.pkl
    """

    def __init__(self, feature_folder, split, feature_type='pop909hand'):
        """
        Args:
            feature_folder: Path to folder containing pkl files
            split: One of 'train', 'validation', 'test'
            feature_type: Prefix for pkl files (default: 'pop909hand')
        """
        # Don't call super().__init__ directly since we need custom loading
        assert split in ['train', 'validation', 'test']
        
        pkl_path = f'{feature_folder}/{feature_type}-{split}.pkl'
        print(f'Loading {pkl_path}')
        assert os.path.exists(pkl_path), f"File not found: {pkl_path}"
        
        self.data = pickle.load(open(pkl_path, 'rb'))
        self.split = split
        self.prefix = feature_type
        
        # Get config, fallback to unconditional if feature_type not found
        cfg = config.get(feature_type, config.get('unconditional', {}))
        self.max_length = cfg.get('max_length', 128)
        
        # Use HandDataAugmentation for training
        self.data_aug = HandDataAugmentation()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        Returns:
            note_sequence: torch.Tensor of shape (max_length, 5)
            labels: torch.Tensor of shape (max_length,)
        """
        smpl = self.data[item]
        feats, labels = smpl[0], smpl[1]
        
        # Apply augmentation on train split only
        if self.split == 'train':
            feats, labels = self.data_aug(feats.copy(), labels.copy())
        else:
            feats, labels = feats.copy(), labels.copy()
        
        # Validate pitch range (skip pad row)
        if np.max(feats[1:, 0]) > MAX_PIANO_PITCH or np.min(feats[1:, 0]) < MIN_PIANO_PITCH:
            # Clamp instead of raising exception for robustness
            feats[1:, 0] = np.clip(feats[1:, 0], MIN_PIANO_PITCH, MAX_PIANO_PITCH)
        
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
        
        # Ensure 5 columns (pitch, dt, dur, vel, hand)
        if note_sequence.shape[1] < 5:
            # Pad with zeros for missing hand column
            pad_cols = np.zeros((len(note_sequence), 5 - note_sequence.shape[1]))
            note_sequence = np.concatenate([note_sequence, pad_cols], axis=1)
        
        # Get hand value for padding
        hand_value = note_sequence[0, 4] if len(note_sequence) > 0 else 0
        
        # Pad SOS (start of sequence)
        pad_row = np.array([[MAX_PIANO_PITCH + 1, 0, 0, 0, hand_value]])
        note_sequence = np.concatenate([pad_row, note_sequence], axis=0)
        
        # First annotation is the first pitch in the segment
        first_pitch = labels[start_idx] if start_idx < len(labels) else labels[0]
        annotation = np.concatenate([np.array([first_pitch]), annotation], axis=0)
        
        # Truncate to max_length
        if self.split != 'test':
            note_sequence = note_sequence[:self.max_length]
            annotation = annotation[:self.max_length]
        
        # Normalize pitch to 0..88 range (pad token becomes 88)
        note_sequence[:, 0] -= MIN_PIANO_PITCH
        annotation -= MIN_PIANO_PITCH
        
        # Pad if shorter than max_length (train/val only)
        if self.split != 'test' and len(note_sequence) < self.max_length:
            pad_len = self.max_length - len(note_sequence)
            # Pad row: pitch=88 (pad token), zeros for other features
            pad_row = np.array([[88, 0, 0, 0, hand_value]])
            pad_block = np.repeat(pad_row, pad_len, axis=0)
            note_sequence = np.concatenate([note_sequence, pad_block], axis=0)
            
            # Pad labels with 88 (pad token)
            label_pad = np.full(pad_len, 88)
            annotation = np.concatenate([annotation, label_pad])
        
        return (
            torch.tensor(note_sequence, dtype=torch.float32),
            torch.tensor(annotation, dtype=torch.long)
        )


class WindowDataset(BaseDataset):
    """
    Dataset for 5-feature window/cluster data:
        - pitch (previous note pitch)
        - log1p delta time
        - log1p duration
        - velocity
        - window_avg (average of previous N pitches)
    
    Uses ClusterDataAugmentation for training augmentation.
    Loads from: pop909window-{split}.pkl
    """

    def __init__(self, feature_folder, split, feature_type='pop909window', window_size=8):
        """
        Args:
            feature_folder: Path to folder containing pkl files
            split: One of 'train', 'validation', 'test'
            feature_type: Prefix for pkl files (default: 'pop909window')
            window_size: Window size for cluster augmentation
        """
        assert split in ['train', 'validation', 'test']
        
        pkl_path = f'{feature_folder}/{feature_type}-{split}.pkl'
        print(f'Loading {pkl_path}')
        assert os.path.exists(pkl_path), f"File not found: {pkl_path}"
        
        self.data = pickle.load(open(pkl_path, 'rb'))
        self.split = split
        self.prefix = feature_type
        self.window_size = window_size
        
        # Get config, fallback to unconditional if feature_type not found
        cfg = config.get(feature_type, config.get('unconditional', {}))
        self.max_length = cfg.get('max_length', 128)
        
        # Use ClusterDataAugmentation for training
        self.data_aug = ClusterDataAugmentation(window_size=window_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        Returns:
            note_sequence: torch.Tensor of shape (max_length, 5)
            labels: torch.Tensor of shape (max_length,)
        """
        smpl = self.data[item]
        feats, labels = smpl[0], smpl[1]
        
        # Apply augmentation on train split only
        if self.split == 'train':
            feats, labels = self.data_aug(feats.copy(), labels.copy())
        else:
            feats, labels = feats.copy(), labels.copy()
        
        # Validate pitch range (skip pad row)
        if np.max(feats[1:, 0]) > MAX_PIANO_PITCH or np.min(feats[1:, 0]) < MIN_PIANO_PITCH:
            # Clamp instead of raising exception for robustness
            feats[1:, 0] = np.clip(feats[1:, 0], MIN_PIANO_PITCH, MAX_PIANO_PITCH)
        
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
        
        # Ensure 5 columns (pitch, dt, dur, vel, window_avg)
        if note_sequence.shape[1] < 5:
            # Pad with zeros for missing window_avg column
            pad_cols = np.zeros((len(note_sequence), 5 - note_sequence.shape[1]))
            note_sequence = np.concatenate([note_sequence, pad_cols], axis=1)
        
        # Get window_avg value for padding (use first value or compute)
        window_avg_value = note_sequence[0, 4] if len(note_sequence) > 0 else 60  # default middle C
        
        # Pad SOS (start of sequence)
        pad_row = np.array([[MAX_PIANO_PITCH + 1, 0, 0, 0, window_avg_value]])
        note_sequence = np.concatenate([pad_row, note_sequence], axis=0)
        
        # First annotation is the first pitch in the segment
        first_pitch = labels[start_idx] if start_idx < len(labels) else labels[0]
        annotation = np.concatenate([np.array([first_pitch]), annotation], axis=0)
        
        # Truncate to max_length
        if self.split != 'test':
            note_sequence = note_sequence[:self.max_length]
            annotation = annotation[:self.max_length]
        
        # Normalize pitch to 0..88 range (pad token becomes 88)
        note_sequence[:, 0] -= MIN_PIANO_PITCH
        annotation -= MIN_PIANO_PITCH
        
        # Also normalize window_avg to 0..88 range
        note_sequence[:, 4] -= MIN_PIANO_PITCH
        
        # Pad if shorter than max_length (train/val only)
        if self.split != 'test' and len(note_sequence) < self.max_length:
            pad_len = self.max_length - len(note_sequence)
            # Pad row: pitch=88 (pad token), zeros for other features, normalized window_avg
            normalized_window_avg = window_avg_value - MIN_PIANO_PITCH
            pad_row = np.array([[88, 0, 0, 0, normalized_window_avg]])
            pad_block = np.repeat(pad_row, pad_len, axis=0)
            note_sequence = np.concatenate([note_sequence, pad_block], axis=0)
            
            # Pad labels with 88 (pad token)
            label_pad = np.full(pad_len, 88)
            annotation = np.concatenate([annotation, label_pad])
        
        return (
            torch.tensor(note_sequence, dtype=torch.float32),
            torch.tensor(annotation, dtype=torch.long)
        )


# Alias for backward compatibility
ClusterDataset = WindowDataset


def main():
    """Test the dataset classes."""
    from pathlib import Path
    
    # Get the yzq output directory
    yzq_dir = Path(__file__).resolve().parent
    features_dir = yzq_dir / "output"
    
    if not features_dir.exists():
        print(f"Features directory not found at {features_dir}")
        print("Run data_preparation.py first to generate pkl files.")
        return
    
    # Test HandDataset if files exist
    hand_pkl = features_dir / "pop909hand-train.pkl"
    if hand_pkl.exists():
        print("\n" + "=" * 60)
        print("Testing HandDataset")
        print("=" * 60)
        
        dataset = HandDataset(str(features_dir), 'train', feature_type='pop909hand')
        print(f"Dataset size: {len(dataset)}")
        
        # Fetch one sample
        note_seq, labels = dataset[0]
        print(f"Note sequence shape: {note_seq.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Feature columns: [pitch, dt, dur, vel, hand]")
        print(f"First 3 rows:\n{note_seq[:3]}")
    else:
        print(f"Hand dataset not found at {hand_pkl}")
    
    # Test WindowDataset if files exist
    window_pkl = features_dir / "pop909window-train.pkl"
    if window_pkl.exists():
        print("\n" + "=" * 60)
        print("Testing WindowDataset")
        print("=" * 60)
        
        dataset = WindowDataset(str(features_dir), 'train', feature_type='pop909window')
        print(f"Dataset size: {len(dataset)}")
        
        # Fetch one sample
        note_seq, labels = dataset[0]
        print(f"Note sequence shape: {note_seq.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Feature columns: [pitch, dt, dur, vel, window_avg]")
        print(f"First 3 rows:\n{note_seq[:3]}")
    else:
        print(f"Window dataset not found at {window_pkl}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

